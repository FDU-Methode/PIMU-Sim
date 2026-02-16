#include <cassert>
#include "../../include/perf/perf_func_unit.hpp"

/////////////////////////////
//    PerfFunctionalUnit   //
/////////////////////////////

PerfFunctionalUnit::PerfFunctionalUnit(PerfSim* sim, Logger& logger, RegFile& regfile) :
    simulator(sim), logger(logger), regfile(regfile) {}

void PerfFunctionalUnit::execute(){}  //need to complete
void PerfFunctionalUnit::tick(){}  //need to complete

void PerfFunctionalUnit::completeInstruction(){
    assert(state == State::BUSY);
    state = State::FINISH;
}

/////////////////////////////
//    PipelineController   //
/////////////////////////////

PipelineController::PipelineController(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {

    dma     = std::make_unique<DMA>(sim, logger, regfile);
    spm     = std::make_unique<SPM>(sim, logger, regfile);
    sa      = std::make_unique<SystolicArray>(sim, logger, regfile);
    acc     = std::make_unique<Accumulator>(sim, logger, regfile);
    sfu     = std::make_unique<SFU>(sim, logger, regfile);
    quant   = std::make_unique<Quant>(sim, logger, regfile);
    im2col  = std::make_unique<Im2col>(sim, logger, regfile);
    gemv    = std::make_unique<GEMV>(sim, logger, regfile);

}

void PipelineController::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

void PipelineController::tick(){
    if (task_queue.empty()) return;

    PerfFunctionalUnit* task = task_queue.front();

    task->tick();

    if (task->isFinish()){
        logger.log("  [PipelineController] %s module finished", task->getName().c_str());
        task->setIdle();
        task_queue.pop();

        if (!task_queue.empty()){
            PerfFunctionalUnit* next_task = task_queue.front();
            assert(next_task->isIdle());
            logger.log("  [PipelineController] start %s module", next_task->getName().c_str());
            next_task->execute();
        } else{
            logger.log("  [PipelineController] All tasks finished");
            completeInstruction();
        }
    }

}

void PipelineController::execute(){
    uint64_t read_start_data = regfile.read(regfile.start_reg_addr);

    // 清除任务队列
    while(!task_queue.empty()) task_queue.pop();

    // 根据启动寄存器值配置任务流
    if (read_start_data & (1ULL << OpBit::BIT_MVIN)){
        logger.log("[PipelineController] Configing MVIN instruction");
        bool is_quant = (regfile.read(regfile.config_mvin_rs1_reg_addr) >> 4) & 0x1;
        if(is_quant) {
            task_queue.push(dma.get());
            task_queue.push(quant.get());
        } else {
            task_queue.push(dma.get());
        }
    }
    if (read_start_data & (1ULL << OpBit::BIT_MVOUT)){
        logger.log("[PipelineController] Configing MVOUT instruction");
        bool is_quant = (regfile.read(regfile.config_mvout_rs1_reg_addr) >> 4) & 0x1;
        if(is_quant) {
            task_queue.push(dma.get());
            task_queue.push(quant.get());
        } else {
            task_queue.push(dma.get());
        }
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SA)){
        logger.log("[PipelineController] Configing COM_SA instruction");
        bool dataflow = regfile.read(regfile.config_compute_rs1_reg_addr) & 0x1;
        bool acc_dest = (regfile.read(regfile.config_compute_rs1_reg_addr) >> 27) & 0x1;
        if(!dataflow) {task_queue.push(im2col.get());}
        task_queue.push(sa.get());
        task_queue.push(acc.get());
        if(!acc_dest) {task_queue.push(quant.get());}
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATADD)){
        logger.log("[PipelineController] Configing COM_MATADD instruction");
        task_queue.push(acc.get());
        task_queue.push(quant.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATVEC)){
        logger.log("[PipelineController] Configing COM_MATVEC instruction");
        task_queue.push(gemv.get());
        task_queue.push(acc.get());
        task_queue.push(quant.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SFU)){
        logger.log("[PipelineController] Configing COM_SFU instruction");
        bool is_quant = (regfile.read(regfile.config_sfu_rs1_reg_addr) >> 8) & 0x1;
        if(is_quant) {
            task_queue.push(sfu.get());
            task_queue.push(quant.get());
        } else {
            task_queue.push(sfu.get());
        }
    }

    // 启动第一个任务
    if (!task_queue.empty()){
        assert(state == State::IDLE);
        state = State::BUSY;
        PerfFunctionalUnit* task = task_queue.front();
        assert(task->isIdle());
        task->execute();
    }
}

/////////////////////////////
//           DMA           //
/////////////////////////////

DMA::DMA(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "DMA";
}

void DMA::tick(){
    if(!isBusy()) return;

    if(remaining_cycles > 0){
        remaining_cycles--;
        if(remaining_cycles == 0) completeInstruction();
        return;
    }

    auto core = simulator->getCore();
    Message getMsg{};
    if(core->tryConsumeResp(expectedMsgType, getMsg)){
        logger.log("[DMA] get expected resp type = %s, DMA complete", MsgTypeToStr(getMsg.type).c_str());
        completeInstruction();
    }
}

void DMA::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint64_t start = regfile.read(regfile.start_reg_addr);
    bool do_mvin  = (start & (1ULL << 0)) != 0;
    bool do_mvout = (start & (1ULL << 1)) != 0;

    Message req{};
    uint8_t data_precision = 0;

    if(do_mvin && !do_mvout){
        req.type = MsgType::LD_REQ;
        req.row_num = (int)((regfile.read(regfile.mvin_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvin_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 48) & 0xFFFF);
        req.sram_stride = (int)((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 32) & 0xFFFF);

        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 1) & 0x1) << 0;
        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 2) & 0x3) << 1;
        data_precision |= ((regfile.read(regfile.config_mvin_rs1_reg_addr) >> 4) & 0x1) << 3;
        switch(data_precision){
            case 2:  req.data_size = 1; break;
            case 4:  req.data_size = 2; break;
            case 10: req.data_size = 4; break;
            case 3:  req.data_size = 1; break;
            default: req.data_size = 0; break;
        }

        req.quant = (regfile.read(regfile.config_mvin_rs1_reg_addr) >> 4) & 0x1;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvin_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvin_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::LD_RESP;
        logger.log("[DMA] MVIN: send LD_REQ, src_addr = 0x%x, dst_addr = 0x%x, row_num = %d, col_num = %d, dram_stride = %d, sram_stride = %d, data_size = %d", 
            req.src_addr, req.dst_addr, req.row_num, req.col_num, req.dram_stride, req.sram_stride, req.data_size);
    } else if(!do_mvin && do_mvout){
        req.type = MsgType::ST_REQ;
        req.row_num = (int)((regfile.read(regfile.mvout_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvout_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 48) & 0xFFFF);
        req.sram_stride = (int)((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 32) & 0xFFFF);

        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 1) & 0x1) << 0;
        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 2) & 0x3) << 1;
        data_precision |= ((regfile.read(regfile.config_mvout_rs1_reg_addr) >> 4) & 0x1) << 3;
        switch(data_precision){
            case 2:  req.data_size = 1; break;
            case 4:  req.data_size = 2; break;
            case 10: req.data_size = 4; break;
            case 3:  req.data_size = 1; break;
            default: req.data_size = 0; break;
        }

        req.quant = (regfile.read(regfile.config_mvout_rs1_reg_addr) >> 4) & 0x1;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvout_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvout_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::ST_RESP;
        logger.log("[DMA] MVOUT: send ST_REQ, src_addr = 0x%x, dst_addr = 0x%x, row_num = %d, col_num = %d, dram_stride = %d, sram_stride = %d, data_size = %d", 
            req.src_addr, req.dst_addr, req.row_num, req.col_num, req.dram_stride, req.sram_stride, req.data_size);
    } else{
        remaining_cycles = 10;
        logger.log("[DMA] No clear MVIN/MVOUT, fallback %d cycles", remaining_cycles);
        return;
    }

    auto core = simulator->getCore();
    core->sendMsg(req);
    logger.log("[DMA] Send request to DDR, DMA set to busy");
}

void DMA::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//           SPM           //
/////////////////////////////

/* SPM: not used, update later */

SPM::SPM(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "SPM";
}

void SPM::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void SPM::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;
    remaining_cycles = 5; // TODO: from tile bytes / spm bandwidth / ports
    logger.log("[SPM] Start, remaining = %d", remaining_cycles);
}

void SPM::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//      SystolicArray      //
/////////////////////////////

SystolicArray::SystolicArray(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "SystolicArray";
}

void SystolicArray::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void SystolicArray::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint8_t optype = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 25) & 0x3);
    
    if(optype == 0) {  // GEMM

        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;

        if(col_num_a != row_num_b) {
            logger.log("[SystolicArray][ERROR] GEMM Dimension Mismatch: input_a_col_num = %d, input_b_row_num = %d", col_num_a, row_num_b);
            assert(col_num_a == row_num_b && "SystolicArray Requires col_num_a = row_num_b");
        }

        remaining_cycles = row_num_a + col_num_a + col_num_b + 2;
        logger.log("[SystolicArray] GEMM Start, A:%d*%d, B:%d*%d, remaining = %d", row_num_a, col_num_a, row_num_b, col_num_b, remaining_cycles);

    } else if(optype == 1) {  // CONV
        // parameters
        uint8_t dataflow = (uint8_t)(regfile.read(regfile.config_compute_rs1_reg_addr) & 0x1);
        int pad_left = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 1) & 0x3);
        int pad_right = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 3) & 0x3);
        int pad_top = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 5) & 0x3);
        int pad_bottom = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 7) & 0x3);
        int pad_mode = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 9) & 0x3);
        int weight_shape = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 11) & 0xF) + 1;
        int weight_stride = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 15) & 0x3) + 1;
        int weight_dilation = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 17) & 0x1F) + 1;
        uint8_t is_group_conv = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 22) & 0x1);
        uint8_t int_type = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 23) & 0x3);

        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;

        // compute cycle
        int k_eff = weight_dilation * (weight_shape - 1) + 1;
        int ofm_w = ((col_num_a + pad_left + pad_right - k_eff) / weight_stride) + 1;
        int ofm_h = ((row_num_a + pad_top + pad_bottom - k_eff) / weight_stride) + 1;
        int cout = col_num_b;
        int cin = row_num_b;
        int A_row_num = (ofm_w * ofm_h) % 64;
        remaining_cycles = 6 + A_row_num + cout + weight_shape * weight_shape * 32;

        logger.log("[SystolicArray] CONV Start, pad_t/b/l/r/mode=%d/%d/%d/%d/%d, weight_shape/stride/dilation=%d/%d/%d, ifm_w=%d, ifm_h=%d, cin=%d, cout=%d, remaining = %d", 
            pad_top, pad_bottom, pad_left, pad_right, pad_mode, weight_shape, weight_stride, weight_dilation, col_num_a, row_num_a, row_num_b, col_num_b, remaining_cycles);

    } else {  // GEMV
        remaining_cycles = 10;
        logger.log("[SystolicArray] GEMV Start, remaining = %d", remaining_cycles);
    }
}

void SystolicArray::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//        Accumulator      //
/////////////////////////////

Accumulator::Accumulator(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "Accumulator";
}

void Accumulator::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void Accumulator::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint8_t matadd_start = (regfile.read(regfile.start_reg_addr) >> 3) & 0x1;
    if(matadd_start) {
        int row_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 40) & 0xFF);
        int col_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 32) & 0xFF);
        remaining_cycles = (row_num * col_num + 31) / 32 + 4;
        logger.log("[Accumulator] Matadd Start, row_num=%d, col_num=%d, remaining = %d", row_num, col_num, remaining_cycles);
    } else {
        remaining_cycles = 3; // TODO: from output tiles / accumaltion rounds
        logger.log("[Accumulator] Start, remaining = %d", remaining_cycles);
    }
}

void Accumulator::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//           SFU           //
/////////////////////////////

SFU::SFU(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "SFU";
}

void SFU::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void SFU::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint8_t sfu_optype = (uint8_t)(regfile.read(regfile.config_sfu_rs1_reg_addr) & 0x3F);
    int col_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 32) & 0xFFFF) + 1;
    int row_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 48) & 0xFFFF) + 1;

    switch(sfu_optype) {
        case 0: { // softmax
            int batch_num = (col_num + 31) / 32;
            int setup_delay = 1;
            int quant_delay = 1;
            int dequant_delay = 1;

            remaining_cycles = setup_delay + row_num * (2 * batch_num + 13 + quant_delay + dequant_delay);
            logger.log("[SFU] Softmax Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 1: { // gelu
            int gelu_num = 4;    // Num of gelu parallel processing units
            int batch_num = (col_num * row_num + gelu_num - 1) / gelu_num;
            int setup_delay = 3;
            int quant_delay = 1;
            int slow_gelu_delay = 60;
            int fast_gelu_delay = 6;
            float p_slow = 1;    // The proportion of slow gelu (0-1)

            remaining_cycles = setup_delay + batch_num * (p_slow * slow_gelu_delay + (1-p_slow) * fast_gelu_delay) + quant_delay;
            logger.log("[SFU] Gelu Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 2: { // layernorm
            remaining_cycles = 10;
            logger.log("[SFU] Layernorm Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 3: { // downsample_max
            remaining_cycles = 10;
            logger.log("[SFU] Downsample_max Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 4: { // downsample_avg
            remaining_cycles = 10;
            logger.log("[SFU] Downsample_avg Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 8: { // transpose
            remaining_cycles = 10;
            logger.log("[SFU] Transpose Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        default: {
            logger.log("[SFU][ERROR] Unsupported SFU Optype = %u", (unsigned)sfu_optype);
            throw std::runtime_error("Unsupported SFU Optype: " + std::to_string((unsigned)sfu_optype));
        }
    }
}

void SFU::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//          Quant          //
/////////////////////////////

Quant::Quant(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "Quant";
}

void Quant::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void Quant::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;
    remaining_cycles = 1; // TODO: from bytes / throughput
    logger.log("[Quant] Start, remaining = %d", remaining_cycles);
}

void Quant::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//         Im2col          //
/////////////////////////////

Im2col::Im2col(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "Im2col";
}

void Im2col::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void Im2col::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    remaining_cycles = 8; // TODO: from H/W/K/C stride etc.
    logger.log("[Im2col] Start, remaining = %d", remaining_cycles);
}

void Im2col::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//          GEMV           //
/////////////////////////////

GEMV::GEMV(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "GEMV";
}

void GEMV::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void GEMV::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;
    remaining_cycles = 15; // TODO: from vector length / tile size
    logger.log("[GEMV] Start, remaining = %d", remaining_cycles);
}

void GEMV::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}
