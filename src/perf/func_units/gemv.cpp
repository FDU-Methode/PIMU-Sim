#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_npu_core.hpp"

GEMV::GEMV(PerfSim* sim, PerfNpuCore* core, Logger& logger, const Config& cfg) :
    PerfFunctionalUnit(sim, core, logger, cfg) {
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