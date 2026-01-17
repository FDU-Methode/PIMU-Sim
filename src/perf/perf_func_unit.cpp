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
    dequant = std::make_unique<Dequant>(sim, logger, regfile);
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
        task_queue.push(dma.get());
        task_queue.push(spm.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_MVOUT)){
        logger.log("[PipelineController] Configing MVOUT instruction");
        task_queue.push(spm.get());
        task_queue.push(dma.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SA)){
        logger.log("[PipelineController] Configing COM_SA instruction");
        task_queue.push(spm.get());
        task_queue.push(sa.get());
        task_queue.push(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATADD)){
        logger.log("[PipelineController] Configing COM_MATADD instruction");
        task_queue.push(spm.get());
        task_queue.push(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATVEC)){
        logger.log("[PipelineController] Configing COM_MATVEC instruction");
        task_queue.push(spm.get());
        task_queue.push(gemv.get());
        task_queue.push(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SFU)){
        logger.log("[PipelineController] Configing COM_SFU instruction");
        task_queue.push(sfu.get());
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
    // regfile.read();
    uint64_t start = regfile.read(regfile.start_reg_addr);
    bool do_mvin  = (start & (1ULL << 0)) != 0;
    bool do_mvout = (start & (1ULL << 1)) != 0;

    Message req{};

    if(do_mvin && !do_mvout){
        req.type = MsgType::LD_REQ;
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvin_rs1_reg_addr) & 0xFFFFFFFF);
        req.request_data_size = (int)(regfile.read(regfile.mvin_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::LD_RESP;
        logger.log("[DMA] MVIN: send LD_REQ, addr = 0x%x, bytes = %d", req.dst_addr, req.request_data_size);
    } else if(!do_mvin && do_mvout){
        req.type = MsgType::ST_REQ;
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvout_rs1_reg_addr) & 0xFFFFFFFF);
        req.data_size = (int)(regfile.read(regfile.mvout_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::ST_RESP;
        logger.log("[DMA] MVOUT: send ST_REQ, addr = 0x%x, bytes = %d", req.dst_addr, req.request_data_size);
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
    // regfile.read();
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
    // regfile.read();
    remaining_cycles = 20; // TODO: from M/N/K, core size
    logger.log("[SystolicArray] Start, remaining = %d", remaining_cycles);
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
    // regfile.read();
    remaining_cycles = 8; // TODO: from output tiles / accumaltion rounds
    logger.log("[SystolicArray] Start, remaining = %d", remaining_cycles);
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
    // regfile.read();
    remaining_cycles = 6; // TODO: from vector length / op type
    logger.log("[SFU] Start, remaining = %d", remaining_cycles);
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
    // regfile.read();
    remaining_cycles = 4; // TODO: from bytes / throughput
    logger.log("[Quant] Start, remaining = %d", remaining_cycles);
}

void Quant::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}

/////////////////////////////
//         Dequant         //
/////////////////////////////

Dequant::Dequant(PerfSim* sim, Logger& logger, RegFile& regfile) :
    PerfFunctionalUnit(sim, logger, regfile) {
    UnitName = "Dequant";
}

void Dequant::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void Dequant::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;
    // regfile.read();
    remaining_cycles = 4; // TODO: from bytes / throughput
    logger.log("[Dequant] Start, remaining = %d", remaining_cycles);
}

void Dequant::completeInstruction(){
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
    // regfile.read();
    remaining_cycles = 12; // TODO: from H/W/K/C stride etc.
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
    // regfile.read();
    remaining_cycles = 15; // TODO: from vector length / tile size
    logger.log("[GEMV] Start, remaining = %d", remaining_cycles);
}

void GEMV::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}
