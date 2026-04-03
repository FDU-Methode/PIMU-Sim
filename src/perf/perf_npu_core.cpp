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

    dma_1_inst_queue = std::make_unique<DMAInstQueue>(logger, reg_file, cfg, event_sync_unit.get());
    dma_2_inst_queue = std::make_unique<DMAInstQueue>(logger, reg_file, cfg, event_sync_unit.get());
    dma_3_inst_queue = std::make_unique<DMAInstQueue>(logger, reg_file, cfg, event_sync_unit.get());
    sa_inst_queue    = std::make_unique<SAInstQueue>(logger, reg_file, cfg, event_sync_unit.get());
    sfu_inst_queue   = std::make_unique<SFUInstQueue>(logger, reg_file, cfg, event_sync_unit.get());
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
    dma  = std::make_unique<DMA>(sim, this, logger, reg_file, cfg);
    sa   = std::make_unique<SystolicArray>(sim, this, logger, reg_file, cfg);
    acc  = std::make_unique<Accumulator>(sim, this, logger, reg_file, cfg);
    sfu  = std::make_unique<SFU>(sim, this, logger, reg_file, cfg);
    gemv = std::make_unique<GEMV>(sim, this, logger, reg_file, cfg);
    logger.log("  DMA created!");
    logger.log("  SystolicArray created!");
    logger.log("  Accumulator created!");
    logger.log("  SFU created!");
    logger.log("  GEMV created!");

    spm = std::make_unique<Scratchpad>(logger, cfg);
    logger.log("  Scratchpad created!");

    // Create Utilization
    util_tracker = std::make_unique<HardWareUtilizationTracker>(logger, cfg);
    util_tracker->reset();
    logger.log("  HardWare Utilization Tracker created!");

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


void PerfNpuCore::WriteRegStage(){
    if(state != IDLE) return;

    // 这里假设你当前把 WriteRegStage 当成 decoder/consumer 入口
    // 如果你有独立 decoder，就把下面 read_inst() 挪到 decoder 里
    Inst inst{};
    if(!inst_cache->read_inst(inst)) {
        logger.log("WriteRegStage: inst cache empty");
        return;
    }

    // TODO:
    // 在这里把 inst_word 送给 decoder
    // 例如：
    // decoder.decode(inst_word, reg_file, ...);


    // // Inst inst = instMemory[pc];
    // // reg_file.write(inst.reg_addr, inst.inst_data);
    // // logger.log("WriteRegStage: Write Reg addr: 0x%x, Data: 0x%lx", inst.reg_addr, inst.inst_data);

    // // // 如果是启动寄存器，则将状态置为 BUSY，等待完成中断
    // // if(inst.reg_addr == reg_file.start_reg_addr) {state = BUSY; opcode = inst.inst_data;}

    pc++;
}


void PerfNpuCore::ExecuteStage(){
    if(state != BUSY) return;

    if(pipeline_controller->isBusy()){
        logger.log("ExecuteStage: waiting for completion");
        pipeline_controller->tick();
    } else if(pipeline_controller->isIdle()){
        logger.log("ExecuteStage: 1");
        logger.log("ExecuteStage: start task execution");
        pipeline_controller->execute();
    } else if(pipeline_controller->isFinish()){
        pipeline_controller->setIdle();
    }
}


void PerfNpuCore::UpdataRegStage(){
    uint64_t read_mer_data = reg_file.read(reg_file.mer_reg_addr);
    uint64_t read_ier_data = reg_file.read(reg_file.ier_reg_addr);

    if(!(read_mer_data == 0x1 && read_ier_data == 0x3f)) {irq = false; return;}

    uint64_t read_isr_data = reg_file.read(reg_file.isr_reg_addr);;
    uint64_t ipr_val = 0;

    auto get_active_bits_str = [](uint64_t data) -> std::string {
        std::string str = "";
        for (int i = 0; i < 64; ++i) {
            if ((data >> i) & 0x1) {
                if (!str.empty()) {
                    str += ", ";
                }
                str += std::to_string(i);
            }
        }
        return str;
    };

    if(state == BUSY){
        if(read_isr_data != 0){ // 硬件执行完成，产生活跃中断信号
            std::string isr_str = get_active_bits_str(read_isr_data);
            logger.log("UpdataRegStage: Detected ISR Interrupt bit [%s] at cycle %d", isr_str.c_str(), getCycle());

            // ISR 与 IER 进行逻辑与，写入 IPR 寄存器
            ipr_val = read_isr_data & read_ier_data;
            reg_file.write(reg_file.ipr_reg_addr, ipr_val);
            logger.log("  Write IPR Reg addr: 0x%x, Data: 0x%lx", reg_file.ipr_reg_addr, ipr_val);
        } else{
            reg_file.write(reg_file.ipr_reg_addr, 0);
        }
    } else if(state == DONE){
        uint64_t read_iar_data = reg_file.read(reg_file.iar_reg_addr);
        if(read_iar_data != 0){
            std::string iar_str = get_active_bits_str(read_iar_data);
            logger.log("UpdataRegStage: IAR received for bits [%s]", iar_str.c_str());

            // 清除 ISR 对应标志位
            uint64_t new_isr_val = read_isr_data & (~read_iar_data);
            reg_file.write(reg_file.isr_reg_addr, new_isr_val);
            // 更新 IPR 对应标志位
            ipr_val = new_isr_val & read_ier_data;
            reg_file.write(reg_file.ipr_reg_addr, ipr_val);
            // 自清除 IAR
            reg_file.write(reg_file.iar_reg_addr, 0);

        }
    }

    // 中断信号 irq
    irq = ipr_val != 0;
}

void PerfNpuCore::tick(){
    logger.log("NPU Core is ticked at cycle %d", getCycle());

    loadInstToCache();

    // Connect DMA To DDR
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

    // Set SPM && ACC SRAM Idle
    spm->clearPorts();
    Scratchpad* acc_sram = getAccSram();
    acc_sram->clearPorts();

    // 执行指令
    ExecuteStage();
    UpdataRegStage();
    WriteRegStage();

    _cycle++;

    // 处理中断
    if(state == BUSY){
        if(!irq) return;
        else{
            logger.log("Simulator Detectes an irq signal at cycle %d", getCycle());
            // 读取 IPR 寄存器
            uint64_t read_ipr_data = reg_file.read(reg_file.ipr_reg_addr);
            // 比较 IPR 寄存器值 与 opcode
            if(read_ipr_data != opcode){
                logger.log("Error: IPR Reg does not match expected opcode %d", opcode);
                std::cout << "ipr val: " << read_ipr_data << " opcode: " << opcode << std::endl;
                throw std::runtime_error("IPR Reg does not match expected opcode");
            } else{
                logger.log("  IPR Reg matches expected opcode %d", opcode);
                reg_file.write(reg_file.iar_reg_addr, opcode);
                state = DONE;
            }
        }
    } else if(state == DONE){
        if(irq) return;
        else{
            logger.log("IRQ signal successfully cleared at cycle %d", getCycle());
            state = IDLE;
        }
    }

    checkCompletion();
}


void PerfNpuCore::checkCompletion(){
    if(inst_file_done && inst_cache->empty() && state == IDLE) {
        _stats.isDone = true;
        _cycle --;
        _stats.cycle = getCycle();
        _stats.inst_num = pc;
        logger.log("NPU Core: Simulation completed at cycle %d", getCycle());
    } else if(_cycle >= cfg.max_sim_cycle-1){
        _stats.inst_num = pc;
        logger.log("NPU Core: Simulation timeout at cycle %d", getCycle());
    }
}


PerfFunctionalUnit* PerfNpuCore::getUnit(UnitType name) {
    switch(name) {
        case UnitType::DMA           : return dma.get();
        case UnitType::SystolicArray : return sa.get();
        case UnitType::Accumulator   : return acc.get();
        case UnitType::SFU           : return sfu.get();
        case UnitType::GEMV          : return gemv.get();
        default                      : return nullptr;
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
