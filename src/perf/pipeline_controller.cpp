#include <cassert>
#include "../../include/perf/pipeline_controller.hpp"

PipelineController::PipelineController(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg, HardWareUtilizationTracker* util_tracker, Scratchpad* spm) :
    simulator(sim), cfg(cfg), logger(logger), regfile(regfile), util_tracker(util_tracker), spm(spm) {

    dma     = std::make_unique<DMA>(sim, this, logger, regfile, cfg);
    sa      = std::make_unique<SystolicArray>(sim, this, logger, regfile, cfg);
    acc     = std::make_unique<Accumulator>(sim, this, logger, regfile, cfg);
    sfu     = std::make_unique<SFU>(sim, this, logger, regfile, cfg);
    gemv    = std::make_unique<GEMV>(sim, this, logger, regfile, cfg);
}

void PipelineController::completeInstruction(){
    assert(state == State::BUSY);
    state = State::FINISH;
}

void PipelineController::tick(){
    if (task_queue.empty()) return;

    for(int i = task_queue.size() - 1; i >= 0; --i) {
        if(task_queue[i]->isBusy()) {
            task_queue[i]->tick();
        }
    }

    util_tracker->sample({
        dma->isBusy(),
        spm ? spm->isBusy() : false,
        sa->isBusy(),
        acc->isBusy(),
        sfu->isBusy(),
        gemv->isBusy()
    });

    for(int i = 0; i < task_queue.size(); ) {
        if(task_queue[i]->isFinish()) {
            logger.log("  [PipelineController] %s module finished", task_queue[i]->getName().c_str());
            task_queue[i]->setIdle();
            task_queue.erase(task_queue.begin() + i);
        } else {
            ++i;
        }
    }

    if(task_queue.empty()) {
        logger.log("  [PipelineController] All tasks finished");
        completeInstruction();
    }

}

void PipelineController::execute(){
    uint64_t read_start_data = regfile.read(regfile.start_reg_addr);

    // 清除任务队列
    while(!task_queue.empty()) task_queue.clear();

    // 根据启动寄存器值配置任务流
    if (read_start_data & (1ULL << OpBit::BIT_MVIN)){
        logger.log("[PipelineController] Configing MVIN instruction");
        task_queue.push_back(dma.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_MVOUT)){
        logger.log("[PipelineController] Configing MVOUT instruction");
        task_queue.push_back(dma.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SA)){
        logger.log("[PipelineController] Configing COM_SA instruction");
        task_queue.push_back(sa.get());
        task_queue.push_back(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATADD)){
        logger.log("[PipelineController] Configing COM_MATADD instruction");
        task_queue.push_back(acc.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_MATVEC)){
        logger.log("[PipelineController] Configing COM_MATVEC instruction");
        task_queue.push_back(gemv.get());
    }
    if (read_start_data & (1ULL << OpBit::BIT_COM_SFU)){
        logger.log("[PipelineController] Configing COM_SFU instruction");
        task_queue.push_back(sfu.get());
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
        case UnitType::SystolicArray : return sa.get();
        case UnitType::Accumulator   : return acc.get();
        case UnitType::SFU           : return sfu.get();
        case UnitType::GEMV          : return gemv.get();
        default                      : return nullptr;
    }
}