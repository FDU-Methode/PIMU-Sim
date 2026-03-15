#include <cassert>
#include "../../include/perf/pipeline_controller.hpp"

PipelineController::PipelineController(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg) :
    simulator(sim), cfg(cfg), logger(logger), regfile(regfile) {

    dma     = std::make_unique<DMA>(sim, this, logger, regfile, cfg);
    spm     = std::make_unique<SPM>(sim, this, logger, regfile, cfg);
    sa      = std::make_unique<SystolicArray>(sim, this, logger, regfile, cfg);
    acc     = std::make_unique<Accumulator>(sim, this, logger, regfile, cfg);
    sfu     = std::make_unique<SFU>(sim, this, logger, regfile, cfg);
    im2col  = std::make_unique<Im2col>(sim, this, logger, regfile, cfg);
    gemv    = std::make_unique<GEMV>(sim, this, logger, regfile, cfg);

}

void PipelineController::completeInstruction(){
    assert(state == State::BUSY);
    state = State::FINISH;
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
    }
    if (read_start_data & (1ULL << OpBit::BIT_MVOUT)){
        logger.log("[PipelineController] Configing MVOUT instruction");
        task_queue.push(dma.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SA)){
        logger.log("[PipelineController] Configing COM_SA instruction");
        bool dataflow = regfile.read(regfile.config_compute_rs1_reg_addr) & 0x1;
        if(!dataflow) {task_queue.push(im2col.get());}
        task_queue.push(sa.get());
        task_queue.push(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATADD)){
        logger.log("[PipelineController] Configing COM_MATADD instruction");
        task_queue.push(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATVEC)){
        logger.log("[PipelineController] Configing COM_MATVEC instruction");
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

PerfFunctionalUnit* PipelineController::getUnit(UnitType name) {
    switch(name) {
        case UnitType::DMA           : return dma.get();
        case UnitType::SPM           : return spm.get();
        case UnitType::SystolicArray : return sa.get();
        case UnitType::Accumulator   : return acc.get();
        case UnitType::SFU           : return sfu.get();
        case UnitType::Im2col        : return im2col.get();
        case UnitType::GEMV          : return gemv.get();
        default                      : return nullptr;
    }
}