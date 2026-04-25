#include <sstream>
#include <cctype>
#include <stdexcept>
#include "../../include/perf/perf_npu_core.hpp"
#include "../../include/perf/perf_sim.hpp"
#include "../../include/util.hpp"

namespace {
    std::string trim(std::string s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
        return s;
    }

    Inst parse_inst(std::string str) {
        str = trim(str);

        size_t comment_pos = str.find('#');
        if(comment_pos != std::string::npos) {
            str = trim(str.substr(0, comment_pos));
        }

        if(str.empty()) {
            throw std::runtime_error("empty instruction line");
        }

        size_t h_pos = str.find("'h");
        if(h_pos != std::string::npos) {
            str = str.substr(h_pos + 2);
        } else if (str.find("0x") == 0 || str.find("0X") == 0) {
            str = str.substr(2);
        }

        str.erase(std::remove(str.begin(), str.end(), '_'), str.end());

        if(str.size() > 32) {
            throw std::runtime_error("128-bit instruction is wider than 32 hex chars");
        }

        if(str.size() < 32) {
            str = std::string(32 - str.size(), '0') + str;
        }

        Inst inst{};
        inst.high = std::stoull(str.substr(0, 16), nullptr, 16);
        inst.low = std::stoull(str.substr(16, 16), nullptr, 16);
        return inst;
    }

}  // namespace


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
    dma_4_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 4);
    dma_5_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 5);
    dma_6_inst_queue = std::make_unique<DMAInstQueue>(this, logger, cfg, event_sync_unit.get(), 6);
    sa_inst_queue    = std::make_unique<SAInstQueue>(this, logger, cfg, event_sync_unit.get());
    sfu_inst_queue   = std::make_unique<SFUInstQueue>(this, logger, cfg, event_sync_unit.get());
    dma_1_inst_queue->clear();
    dma_2_inst_queue->clear();
    dma_3_inst_queue->clear();
    dma_4_inst_queue->clear();
    dma_5_inst_queue->clear();
    dma_6_inst_queue->clear();
    sa_inst_queue->clear();
    sfu_inst_queue->clear();
    logger.log("  DMA Inst Queue 1 created!");
    logger.log("  DMA Inst Queue 2 created!");
    logger.log("  DMA Inst Queue 3 created!");
    logger.log("  SA  Inst Queue   created!");
    logger.log("  SFU Inst Queue   created!");

    decoder = std::make_unique<Decoder>(logger, cfg, inst_cache.get(), dma_1_inst_queue.get(), dma_2_inst_queue.get(), dma_3_inst_queue.get(), 
                                        dma_4_inst_queue.get(), dma_5_inst_queue.get(), dma_6_inst_queue.get(), sa_inst_queue.get(), sfu_inst_queue.get());
    logger.log("  Decoder created!");

    // Create Function Unit
    dma_1  = std::make_unique<DMA>(sim, this, logger, cfg, 1);
    logger.log("  DMA 1 created!");
    dma_2  = std::make_unique<DMA>(sim, this, logger, cfg, 2);
    logger.log("  DMA 2 created!");
    dma_3  = std::make_unique<DMA>(sim, this, logger, cfg, 3);
    logger.log("  DMA 3 created!");
    dma_4  = std::make_unique<DMA>(sim, this, logger, cfg, 4);
    logger.log("  DMA 4 created!");
    dma_5  = std::make_unique<DMA>(sim, this, logger, cfg, 5);
    logger.log("  DMA 5 created!");
    dma_6  = std::make_unique<DMA>(sim, this, logger, cfg, 6);
    logger.log("  DMA 6 created!");
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

    logger.log("Opening instructions from file: %s", cfg.inst_file.c_str());
    openInstFile(cfg.inst_file);
    logger.log("----------------------------------------------------");
}


void PerfNpuCore::openInstFile(const std::string& inst_file) {
    if(inst_ifs.is_open()) {
        inst_ifs.close();
    }

    inst_ifs.open(inst_file);
    if(!inst_ifs.is_open()) {
        throw std::runtime_error("Unable to open instruction file: " + inst_file);
    }

    inst_cache->clear();
    pc = 0;
    fetched_inst_num = 0;
    simulated_inst_num = 0;
    inst_file_line_num = 0;
    inst_file_done = false;

    logger.log("Instruction file opened successfully...");
}


bool PerfNpuCore::loadInstToCache() {
    if(inst_file_done) {
        return false;
    }

    if(!inst_ifs.is_open()) {
        throw std::runtime_error("Instruction file stream is not open");
    }

    if(inst_cache->full()) {
        logger.log("[InstLoader] Inst Cache is Full, Skip Loading This Cycle");
        return false;
    }

    std::string line;
    while(std::getline(inst_ifs, line)) {
        ++inst_file_line_num;

        std::string trimmed = trim(line);
        if(trimmed.empty()) {
            continue;
        }

        size_t comment_pos = trimmed.find('#');
        if(comment_pos != std::string::npos) {
            trimmed = trim(trimmed.substr(0, comment_pos));
        }

        if(trimmed.empty()) {
            continue;
        }

        Inst inst = parse_inst(trimmed);

        bool write_status = inst_cache->write_inst(inst);
        if(!write_status) {
            logger.log("[InstLoader][ERROR] Inst Cache Write Failed");
            return false;
        }

        ++fetched_inst_num;
        logger.log("[InstLoader] Line %d Loaded into Inst Cache, inst=0x%016llx%016llx, fetched inst num=%d, cache_size=%d", 
            inst_file_line_num, (unsigned long long)inst.high, (unsigned long long)inst.low, fetched_inst_num, inst_cache->size());
        return true;
    }

    inst_file_done = true;
    inst_ifs.close();
    logger.log("[InstLoader] Instruction File Read Finished, total fetched = %d", fetched_inst_num);
    return false;
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

    if(dma_4_inst_queue->isIdle() && dma_4->isIdle()) {
        if(dma_4_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 4 Queue Issued");
        }
    }

    if(dma_5_inst_queue->isIdle() && dma_5->isIdle()) {
        if(dma_5_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 5 Queue Issued");
        }
    }

    if(dma_6_inst_queue->isIdle() && dma_6->isIdle()) {
        if(dma_6_inst_queue->inst_issue()) {
            logger.log("[Core] DMA 6 Queue Issued");
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

    if(dma_4->isFinish()) {
        dma_4_inst_queue->inst_retire();
        dma_4->setIdle();
        logger.log("[Core] DMA 4 Instruction Retired");
    }

    if(dma_5->isFinish()) {
        dma_5_inst_queue->inst_retire();
        dma_5->setIdle();
        logger.log("[Core] DMA 5 Instruction Retired");
    }

    if(dma_6->isFinish()) {
        dma_6_inst_queue->inst_retire();
        dma_6->setIdle();
        logger.log("[Core] DMA 6 Instruction Retired");
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
    return
        !inst_file_done ||
        !inst_cache->empty() ||
        decoder->fetchBufferValid() ||
        decoder->decodeBufferValid() ||
        (decoder->currentInstType() != InstType::NONE) ||

        !dma_1_inst_queue->empty() || dma_1_inst_queue->isBusy() ||
        !dma_2_inst_queue->empty() || dma_2_inst_queue->isBusy() ||
        !dma_3_inst_queue->empty() || dma_3_inst_queue->isBusy() ||
        !dma_4_inst_queue->empty() || dma_4_inst_queue->isBusy() ||
        !dma_5_inst_queue->empty() || dma_5_inst_queue->isBusy() ||
        !dma_6_inst_queue->empty() || dma_6_inst_queue->isBusy() ||
        !sa_inst_queue->empty()    || sa_inst_queue->isBusy()    ||
        !sfu_inst_queue->empty()   || sfu_inst_queue->isBusy()   ||

        dma_1->isBusy()  || dma_1->isFinish() ||
        dma_2->isBusy()  || dma_2->isFinish() ||
        dma_3->isBusy()  || dma_3->isFinish() ||
        dma_4->isBusy()  || dma_4->isFinish() ||
        dma_5->isBusy()  || dma_5->isFinish() ||
        dma_6->isBusy()  || dma_6->isFinish() ||
        sa->isBusy()     || sa->isFinish()    ||
        acc->isBusy()    || acc->isFinish()   ||
        sfu->isBusy()    || sfu->isFinish()   ||
        gemv->isBusy()   || gemv->isFinish()  ||

        !outbox_.empty() || !inbox_.empty();
}


void PerfNpuCore::tick(){
    logger.log("NPU Core is ticked at cycle %d", getCycle());

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
    if(dma_4->isBusy())    dma_4->tick();
    if(dma_5->isBusy())    dma_5->tick();
    if(dma_6->isBusy())    dma_6->tick();
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
    state.dma4      = dma_4->isBusy();
    state.dma5      = dma_5->isBusy();
    state.dma6      = dma_6->isBusy();
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
            if (dma_acc_mode(dma_4.get(), mode)) return mode;
            if (dma_acc_mode(dma_5.get(), mode)) return mode;
            if (dma_acc_mode(dma_6.get(), mode)) return mode;

            if (acc_sram != nullptr) {
                if (acc_sram->isBiasBusy()) return AccPowerMode::BusyBiasLoad;
                if (acc_sram->isDataBusy()) return AccPowerMode::BusyDmaWr;
            }

            return AccPowerMode::Idle;
        };

        power_analyzer->sample(
            dma_mode_of(dma_1.get()), dma_mode_of(dma_2.get()), dma_mode_of(dma_3.get()),
            dma_mode_of(dma_4.get()), dma_mode_of(dma_5.get()), dma_mode_of(dma_6.get()),
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
    loadInstToCache();

    // 9. Inst Retire
    retireInstQueues();

    _cycle++;

    checkCompletion();
}


void PerfNpuCore::checkCompletion(){
    if(!isBusy()) {
        _stats.isDone = true;
        _stats.cycle = getCycle();
        _stats.inst_line_num = fetched_inst_num;
        _stats.inst_num = simulated_inst_num;
        logger.log("NPU Core: Simulation completed at cycle %d", getCycle());
    } else if(_cycle >= cfg.max_sim_cycle-1){
        _stats.inst_line_num = fetched_inst_num;
        _stats.inst_num = simulated_inst_num;
        logger.log("NPU Core: Simulation timeout at cycle %d", getCycle());
    }
}


void PerfNpuCore::addInstNum() {
    ++simulated_inst_num;
}


PerfFunctionalUnit* PerfNpuCore::getUnit(UnitType name) {
    switch(name) {
        case UnitType::DMA1           : return dma_1.get();
        case UnitType::DMA2           : return dma_2.get();
        case UnitType::DMA3           : return dma_3.get();
        case UnitType::DMA4           : return dma_4.get();
        case UnitType::DMA5           : return dma_5.get();
        case UnitType::DMA6           : return dma_6.get();
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
