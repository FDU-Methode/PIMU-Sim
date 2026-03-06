#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

Quant::Quant(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, logger, regfile, cfg) {
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