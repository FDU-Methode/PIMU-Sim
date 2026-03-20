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
    util_tracker = std::make_unique<HardWareUtilizationTracker>(logger, cfg);
    util_tracker->reset();
    logger.log("  HardWare Utilization Tracker created!");

    spm = std::make_unique<Scratchpad>(logger, cfg);
    logger.log("  Scratchpad created!");

    pipeline_controller = std::make_unique<PipelineController>(sim, logger, reg_file, cfg, util_tracker.get(), spm.get());
    logger.log("  PipelineController created!");

    logger.log("Loading instructions from file: %s", cfg.inst_file.c_str());
    loadInstFromFile(cfg.inst_file);
    logger.log("Loaded %zu instructions!", instMemory.size());
    logger.log("----------------------------------------------------");
}

static uint64_t parse_verilog_hex(std::string str) {
    // 1. 定位 'h' 的位置
    size_t h_pos = str.find("'h");
    if (h_pos != std::string::npos) {
        // 截取 'h' 之后的部分 (例如 "C0" 或 "0000_0001")
        str = str.substr(h_pos + 2);
    } else if (str.find("0x") == 0 || str.find("0X") == 0) {
        // 兼容普通的 0x 开头
        str = str.substr(2);
    }

    // 2. 移除所有的下划线 '_' (Verilog 中常用作分隔符)
    // std::remove 将不删除的字符移到前面，返回新逻辑结尾的迭代器，erase 真正删除
    str.erase(std::remove(str.begin(), str.end(), '_'), str.end());

    // 3. 转换为 64 位无符号整数，基数为 16
    try {
        return std::stoull(str, nullptr, 16);
    } catch (...) {
        return 0; // 解析失败返回0，或者可以在这里 throw 异常
    }
}

void PerfNpuCore::loadInstFromFile(const std::string& inst_file) {
    std::ifstream file(inst_file);
    if (!file.is_open()) {
        logger.log("Error: Unable to open file %s\n", inst_file.c_str());
        return;
    }

    instMemory.clear();
    pc = 0;
    inst_num = 0;

    std::string line;
    int line_idx = 0;

    while (std::getline(file, line)) {
        line_idx++;
        // 忽略注释 #
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::stringstream ss(line);
        std::string op_str, addr_str, data_str;

        // 格式: [OpName] [AddrStr] [DataStr]
        // 对应: MER      32'hC0    64'h01
        if (ss >> op_str >> addr_str >> data_str) {
            try {
                Inst new_inst;
                new_inst.op_name = op_str;
                new_inst.reg_addr = (uint32_t)parse_verilog_hex(addr_str);
                new_inst.inst_data = parse_verilog_hex(data_str);

                instMemory.push_back(new_inst);

            } catch (const std::exception& e) {
                logger.log("Error parsing line %d: %s (Content: %s)\n", line_idx, e.what(), line.c_str());
            }
        } else {
            // 如果一行只有1个或2个部分，视为格式错误，打印警告
             logger.log("Warning: Malformed instruction at line %d: %s\n", line_idx, line.c_str());
        }
    }

    file.close();
    inst_num = instMemory.size();
    logger.log("Successfully loaded %d instructions from %s\n", inst_num, inst_file.c_str());
}

void PerfNpuCore::WriteRegStage(){
    if(state != IDLE) return;

    Inst inst = instMemory[pc];
    reg_file.write(inst.reg_addr, inst.inst_data);
    logger.log("WriteRegStage: Write Reg addr: 0x%x, Data: 0x%lx", inst.reg_addr, inst.inst_data);

    // 如果是启动寄存器，则将状态置为 BUSY，等待完成中断
    if(inst.reg_addr == reg_file.start_reg_addr) {state = BUSY; opcode = inst.inst_data;}

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
        // // 更新 ISR 寄存器，产生活跃中断
        // uint64_t read_start_data = reg_file.read(reg_file.start_reg_addr);
        // uint64_t isr_val = read_start_data;

        // logger.log("ExecuteStage: task execution completed, ISR register updated");
        // reg_file.write(reg_file.isr_reg_addr, isr_val);
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
            MsgTypeToStr(req.type).c_str(), req.src_addr, req.dst_addr, req.row_num, req.col_num, req.dram_stride, req.sram_stride, req.data_size, req.data_size);
        outbox_.pop();
    }

    // Set SPM && ACC SRAM Idle
    spm->clearPorts();
    Scratchpad* acc_sram = pipeline_controller->getAccSram();
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
    if(pc >= instMemory.size() && state == IDLE){
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

HardWareUtilizationTracker* PerfNpuCore::getUtilTracker() {
    return util_tracker.get();
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
