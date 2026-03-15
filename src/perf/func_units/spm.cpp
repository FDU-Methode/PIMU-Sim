#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

/* SPM: not used, update later */

SPM::SPM(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
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