#include <stdexcept>
#include <utility>

#include "../../include/perf/perf_npu_core.hpp"
#include "../../include/perf/perf_sim.hpp"
#include "../../include/util.hpp"

PerfNpuCore::PerfNpuCore(PerfSim* sim, const Config& cfg, Logger& logger)
    : BaseModule(sim, cfg, logger), reg_file(cfg) {

    logger.log("----------------------------------------------------");
    logger.log("  NPU Core Frequency: %2f GHz", cfg.npu_freq);

    irq = false;
    state = State::IDLE;

    // Initialize MicroArchitecture Components
    // Create Instruction Unit
    inst_cache = std::make_unique<InstCache>(cfg);
    inst_cache->clear();
    logger.log("  Inst Cache created! Size=%d", cfg.inst_cache_size);

    event_sync_unit = std::make_unique<EventSyncUnit>(cfg, logger);
    logger.log("  Event Sync created!");

    dma_1_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 1);
    dma_2_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 2);
    dma_3_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 3);
    sa_inst_queue    = std::make_unique<SAInstQueue>(this, logger, cfg, event_sync_unit.get());
    sfu_inst_queue   = std::make_unique<SFUInstQueue>(this, logger, cfg, event_sync_unit.get());
    dma_1_inst_queue->clear();
    dma_2_inst_queue->clear();
    dma_3_inst_queue->clear();
    sa_inst_queue->clear();
    sfu_inst_queue->clear();
    logger.log("  DMA Inst Queue 1 created!");
    logger.log("  DMA Inst Queue 2 created!");
    logger.log("  DMA Inst Queue 3 created!");
    logger.log("  SA  Inst Queue   created!");
    logger.log("  SFU Inst Queue   created!");

    decoder = std::make_unique<Decoder>(logger, inst_cache.get(), dma_1_inst_queue.get(), dma_2_inst_queue.get(), dma_3_inst_queue.get(), 
                                        sa_inst_queue.get(), sfu_inst_queue.get());
    logger.log("  Decoder created!");

    // Create Function Unit
    dma_1  = std::make_unique<DMA>(sim, this, logger, cfg, 1);
    logger.log("  DMA 1 created!");
    dma_2  = std::make_unique<DMA>(sim, this, logger, cfg, 2);
    logger.log("  DMA 2 created!");
    dma_3  = std::make_unique<DMA>(sim, this, logger, cfg, 3);
    logger.log("  DMA 3 created!");
    sa     = std::make_unique<SystolicArray>(sim, this, logger, cfg);
    logger.log("  SystolicArray created!");
    acc    = std::make_unique<Accumulator>(sim, this, logger, cfg);
    logger.log("  Accumulator created!");
    sfu    = std::make_unique<SFU>(sim, this, logger, cfg);
    logger.log("  SFU created!");
    gemv   = std::make_unique<GEMV>(sim, this, logger, cfg);
    logger.log("  GEMV created!");

    spm = std::make_unique<Scratchpad>(logger, cfg);
    logger.log("  Scratchpad created!");

    // Create Utilization
    util_tracker = std::make_unique<HardWareUtilizationTracker>(logger, cfg);
    util_tracker->reset();
    logger.log("  HardWare Utilization Tracker created!");

    // Create Power Analyzer
    power_analyzer = std::make_unique<PowerAnalyzer>(logger, cfg);
    logger.log("  Power Analyzer created!");

    logger.log("----------------------------------------------------");
}


void PerfNpuCore::loadProgram(Program program) {
    if (program_loaded_ && program_started_ && !_stats.isDone) {
        throw std::logic_error(
            "Previous Program has not completed; reset is required"
        );
    }

    if (hasPendingWork() || !sim->isDdrDrained()) {
        throw std::logic_error(
            "Cannot load Program while hardware has pending work"
        );
    }

    // Validate before changing the current Program or its result
    validateProgram(program);

    program_ = std::move(program);

    next_word_ = 0;
    fetched_inst_num = 0;
    simulated_inst_num = 0;

    _stats = Stats{};
    _stats.cycle = getCycle();

    program_loaded_ = true;
    program_started_ = false;
    program_exhausted_ = false;

    logger.log(
        "Program loaded: %zu words",
        program_.wordCount()
    );
}


bool PerfNpuCore::loadWordToCache() {
    if (!program_loaded_ ||
        program_exhausted_ ||
        inst_cache->full()) {
        return false;
    }

    if (next_word_ == program_.wordCount()) {
        program_exhausted_ = true;
        return false;
    }

    const Inst& inst = program_.words[next_word_];

    if (!inst_cache->write_inst(inst)) {
        throw std::runtime_error(
            "Failed to write instruction word to InstCache"
        );
    }

    ++next_word_;
    ++fetched_inst_num;

    logger.log(
        "[InstLoader] Word %zu loaded: 0x%016llx%016llx",
        next_word_,
        static_cast<unsigned long long>(inst.high),
        static_cast<unsigned long long>(inst.low)
    );

    return true;
}


void PerfNpuCore::handleInterrupts() {
    uint64_t mer_val = reg_file.read(reg_file.mer_reg_addr);
    uint64_t ier_val = reg_file.read(reg_file.ier_reg_addr);
    uint64_t isr_val = reg_file.read(reg_file.isr_reg_addr);
    uint64_t iar_val = reg_file.read(reg_file.iar_reg_addr);

    // Clear Interrupt Status Bits
    if(iar_val != 0) {
        uint64_t new_isr_val = isr_val & (~iar_val);
        reg_file.write(reg_file.isr_reg_addr, new_isr_val);
        reg_file.write(reg_file.iar_reg_addr, 0);

        logger.log("[Core] Clear IRQ: IAR=0x%lx, new ISR=0x%lx", iar_val, new_isr_val);
        isr_val = new_isr_val;
        iar_val = 0;
    }

    // Recompute Pending Interrupt
    bool global_irq_enable = (mer_val & 0x1ULL) != 0;
    uint64_t ipr_val = global_irq_enable ? (isr_val & ier_val) : 0;

    reg_file.write(reg_file.ipr_reg_addr, ipr_val);
    irq = (ipr_val != 0);

    if(irq) {
        reg_file.write(reg_file.iar_reg_addr, ipr_val);
        logger.log("[Core] Detect IRQ Pending: ISR=0x%lx, IER=0x%lx, IPR=0x%lx, Write IAR=0x%lx", isr_val, ier_val, ipr_val, ipr_val);
    }
}


void PerfNpuCore::issueInstQueues() {
    // Issue Priority: SA > SFU > DMA
    if(sa_inst_queue->isIdle() && sa->isIdle() && acc->isIdle()) {
        if(sa_inst_queue->inst_issue()) {
            logger.log("[Core] SA Queue Issued");
        }
    }

    if(sfu_inst_queue->isIdle() && sfu->isIdle()) {
        if(sfu_inst_queue->inst_issue()) {
            logger.log("[Core] SFU Queue Issued");
        }
    }

    if(dma_1_inst_queue->isIdle() && dma_1->isIdle()) {
        if(dma_1_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 1 Queue Issued");
        }
    }

    if(dma_2_inst_queue->isIdle() && dma_2->isIdle()) {
        if(dma_2_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 2 Queue Issued");
        }
    }

    if(dma_3_inst_queue->isIdle() && dma_3->isIdle()) {
        if(dma_3_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 3 Queue Issued");
        }
    }
}


void PerfNpuCore::retireInstQueues() {
    if(dma_1->isFinish()) {
        dma_1_inst_queue->inst_retire();
        dma_1->setIdle();
        logger.log("[Core] DMA 1 Instruction Retired");
    }

    if(dma_2->isFinish()) {
        dma_2_inst_queue->inst_retire();
        dma_2->setIdle();
        logger.log("[Core] DMA 2 Instruction Retired");
    }

    if(dma_3->isFinish()) {
        dma_3_inst_queue->inst_retire();
        dma_3->setIdle();
        logger.log("[Core] DMA 3 Instruction Retired");
    }

    if(sa->isFinish()) {
        sa->setIdle();
        logger.log("[Core] SA Finished");
    }

    if(acc->isFinish()) {
        sa_inst_queue->inst_retire();
        acc->setIdle();
        logger.log("[Core] SA Instruction Retired");
    }

    if(sfu->isFinish()) {
        sfu_inst_queue->inst_retire();
        sfu->setIdle();
        logger.log("[Core] SFU Instruction Retired");
    }

    if(gemv->isFinish()) {
        gemv->setIdle();
    }
}


bool PerfNpuCore::isBusy() {
    return (program_loaded_ && !program_exhausted_) || hasPendingWork();
}



bool PerfNpuCore::hasPendingWork() {
    return
        !inst_cache->empty() ||
        !decoder->isDrained() ||
        !event_sync_unit->isDrained() ||

        (dma_1_inst_queue->getInstCount() != 0) ||
        (dma_2_inst_queue->getInstCount() != 0) ||
        (dma_3_inst_queue->getInstCount() != 0) ||
        (sa_inst_queue->getInstCount() != 0) ||
        (sfu_inst_queue->getInstCount() != 0) ||

        !dma_1_inst_queue->empty() ||
        dma_1_inst_queue->isBusy() ||

        !dma_2_inst_queue->empty() ||
        dma_2_inst_queue->isBusy() ||

        !dma_3_inst_queue->empty() ||
        dma_3_inst_queue->isBusy() ||

        !sa_inst_queue->empty() ||
        sa_inst_queue->isBusy() ||

        !sfu_inst_queue->empty() ||
        sfu_inst_queue->isBusy() ||

        dma_1->isBusy() || dma_1->isFinish() ||
        dma_2->isBusy() || dma_2->isFinish() ||
        dma_3->isBusy() || dma_3->isFinish() ||

        sa->isBusy() || sa->isFinish() ||
        acc->isBusy() || acc->isFinish() ||
        sfu->isBusy() || sfu->isFinish() ||
        gemv->isBusy() || gemv->isFinish() ||

        !outbox_.empty() ||
        !inbox_.empty();
}


void PerfNpuCore::tick(){
    if(program_loaded_) {
        program_started_ = true;
    }

    logger.log("NPU Core is ticked at cycle %llu", static_cast<unsigned long long>(getCycle()));

    // 1. Handle Irq
    handleInterrupts();

    // 2. Set SPM && ACC SRAM Idle
    spm->clearPorts();
    AccSram* acc_sram = getAccSram();
    acc_sram->clearPorts();

    // 3. Tick Function Unit
    if(dma_1->isBusy())    dma_1->tick();
    if(dma_2->isBusy())    dma_2->tick();
    if(dma_3->isBusy())    dma_3->tick();
    if(sa->isBusy())       sa->tick();
    if(acc->isBusy())      acc->tick();
    if(sfu->isBusy())      sfu->tick();
    if(gemv->isBusy())     gemv->tick();

    // 4. Connect DMA To DDR
    while(!outbox_.empty()){
        Message& req = outbox_.front();
        auto DDR = sim->getDDR();
        bool accepted = DDR->recvMsg(req);
        if(!accepted){
            logger.log("[Core] DDR backpressure, keep req in outbox");
            break;
        }
        logger.log("[Core] send req to DDR, type = %s, src_addr = 0x%x, dst_addr = 0x%x, row_num = %d, col_num = %d, dram_stride = %d, sram_stride = %d, data_size = %d", 
            MsgTypeToStr(req.type).c_str(), req.src_addr, req.dst_addr, req.row_num + 1, req.col_num + 1, req.dram_stride, req.sram_stride, req.data_size);
        outbox_.pop();
    }

    // 5. Sample Util Tracker and Power Analyzer
    HardWareUtilizationTracker::ModuleState state{};
    state.dma1      = dma_1->isBusy();
    state.dma2      = dma_2->isBusy();
    state.dma3      = dma_3->isBusy();
    state.spm        = spm->isBusy();
    state.sa         = sa->isBusy();
    state.acc        = acc->isBusy();
    state.acc_bias   = (acc_sram != nullptr) ? acc_sram->isBiasBusy() : false;
    state.acc_data   = (acc_sram != nullptr) ? acc_sram->isDataBusy() : false;
    state.sfu        = sfu->isBusy();
    state.gemv       = gemv->isBusy();

    util_tracker->sample(state);

    if (power_analyzer && power_analyzer->enabled()) {
        auto dma_mode_of = [](const DMA* dma) -> DmaPowerMode {
            if (dma == nullptr || !dma->isBusy()) return DmaPowerMode::Idle;

            switch (dma->getCurrentTask()) {
                case DMATaskType::MVIN:  return DmaPowerMode::BusyMvin;
                case DMATaskType::MVOUT: return DmaPowerMode::BusyMvout;
                default:                 return DmaPowerMode::Idle;
            }
        };

        auto sfu_mode_of = [this]() -> SfuPowerMode {
            if (!sfu->isBusy()) return SfuPowerMode::Idle;

            switch (sfu->getOpType()) {
                case 0:  return SfuPowerMode::BusySoftmax;
                case 2:  return SfuPowerMode::BusyLayernorm;
                case 1:  return SfuPowerMode::BusyGelu;
                case 8:  return SfuPowerMode::BusyTranspose;
                case 6:  return SfuPowerMode::BusyPoolmax;
                default: return SfuPowerMode::BusyResample;
            }
        };

        auto sa_mode_of = [this]() -> SaPowerMode {
            if (!sa->isBusy()) return SaPowerMode::Idle;

            switch (sa->getCurrentOpType()) {
                case 0:  return SaPowerMode::BusyGemm;
                case 1:  return SaPowerMode::BusyConv;
                default: return SaPowerMode::Idle;
            }
        };

        auto acc_mode_of = [this, acc_sram]() -> AccPowerMode {
            if (acc->isBusy()) {
                switch (acc->getCurrentTask()) {
                    case SATaskType::MATADD: return AccPowerMode::BusyMatadd;
                    case SATaskType::SA:     return AccPowerMode::BusyAccadd;
                    default:                 break;
                }
            }

            auto dma_acc_mode = [](const DMA* dma, AccPowerMode& out_mode) -> bool {
                if (dma == nullptr || !dma->isBusy() || !dma->usesAcc()) return false;

                switch (dma->getCurrentTask()) {
                    case DMATaskType::MVIN:
                        out_mode = dma->usesBiasPath()
                                 ? AccPowerMode::BusyBiasLoad
                                 : AccPowerMode::BusyDmaWr;
                        return true;
                    case DMATaskType::MVOUT:
                        out_mode = AccPowerMode::BusyDmaRd;
                        return true;
                    default:
                        return false;
                }
            };

            AccPowerMode mode = AccPowerMode::Idle;
            if (dma_acc_mode(dma_1.get(), mode)) return mode;
            if (dma_acc_mode(dma_2.get(), mode)) return mode;
            if (dma_acc_mode(dma_3.get(), mode)) return mode;

            if (acc_sram != nullptr) {
                if (acc_sram->isBiasBusy()) return AccPowerMode::BusyBiasLoad;
                if (acc_sram->isDataBusy()) return AccPowerMode::BusyDmaWr;
            }

            return AccPowerMode::Idle;
        };

        power_analyzer->sample(
            dma_mode_of(dma_1.get()),
            dma_mode_of(dma_2.get()),
            dma_mode_of(dma_3.get()),
            acc_mode_of(),
            sfu_mode_of(),
            sa_mode_of()
        );
    }

    // 6. Issue Inst Queue
    issueInstQueues();

    // 7. Tick Decoder
    decoder->tick();

    // 8. Load Inst To Cache
    loadWordToCache();

    // 9. Inst Retire
    retireInstQueues();

    _cycle++;

    checkCompletion();
}


void PerfNpuCore::checkCompletion() {
    _stats.cycle = getCycle();
    _stats.inst_line_num = fetched_inst_num;
    _stats.inst_num = simulated_inst_num;

    _stats.isDone =
        program_loaded_ &&
        !isBusy() &&
        sim->isDdrDrained();
}


void PerfNpuCore::addInstNum() {
    ++simulated_inst_num;
}


PerfFunctionalUnit* PerfNpuCore::getUnit(UnitType name) {
    switch(name) {
        case UnitType::DMA1           : return dma_1.get();
        case UnitType::DMA2           : return dma_2.get();
        case UnitType::DMA3           : return dma_3.get();
        case UnitType::SystolicArray  : return sa.get();
        case UnitType::Accumulator    : return acc.get();
        case UnitType::SFU            : return sfu.get();
        case UnitType::GEMV           : return gemv.get();
        default                       : return nullptr;
    }
}

//Connect DMA To DDR

void PerfNpuCore::sendMsg(Message& msg){  //DMA Controll core->sendMsg
    outbox_.push(msg);
    logger.log("[Core] push req to outbox, type = %s, src_addr = 0x%x, dst_addr = 0x%x", MsgTypeToStr(msg.type).c_str(), msg.src_addr, msg.dst_addr);
}

bool PerfNpuCore::recvMsg(Message& msg){  //DDR Controll core->recvMsg
    inbox_.push(msg);
    logger.log("[Core] push resp into inbox, type = %s, src_addr = 0x%x, dst_addr = 0x%x", MsgTypeToStr(msg.type).c_str(), msg.src_addr, msg.dst_addr);
    return true;
}

bool PerfNpuCore::tryConsumeResp(MsgType expectedMsgType, Message& getMsg){
    if(inbox_.empty()) return false;
    bool found = false;
    std::queue<Message> tmp;
    while(!inbox_.empty()){
        Message frontMsg = inbox_.front();
        inbox_.pop();
        if(!found && frontMsg.type == expectedMsgType){
            getMsg = frontMsg;
            found = true;
        } else{
            tmp.push(frontMsg);
        }
    }
    inbox_ = std::move(tmp);
    return found;
}
