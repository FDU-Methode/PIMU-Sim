#include <cassert>
#include "../../include/perf/perf_func_unit.hpp"

PerfFunctionalUnit::PerfFunctionalUnit(PerfSim* sim, PerfNpuCore* core, Logger& logger, const Config& cfg) :
    simulator(sim), core(core), cfg(cfg), logger(logger) {}

void PerfFunctionalUnit::execute(){}  //need to complete
void PerfFunctionalUnit::tick(){}  //need to complete

void PerfFunctionalUnit::completeInstruction(){
    assert(state == State::BUSY);
    state = State::FINISH;
}
