#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

Im2col::Im2col(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
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

    // Serial queue model： model only the blocking startup delay from im2col request to first data
    // available for SA start, rather than full im2col completion.
    int startup_delay = 6;
    remaining_cycles = startup_delay;
    logger.log("[Im2col] Start (startup-only model), remaining = %d", remaining_cycles);
}

void Im2col::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}