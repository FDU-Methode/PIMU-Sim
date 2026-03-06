#include <cassert>
#include "../../include/perf/perf_func_unit.hpp"

PerfFunctionalUnit::PerfFunctionalUnit(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg) :
    simulator(sim), cfg(cfg), logger(logger), regfile(regfile) {}

void PerfFunctionalUnit::execute(){}  //need to complete
void PerfFunctionalUnit::tick(){}  //need to complete

void PerfFunctionalUnit::completeInstruction(){
    assert(state == State::BUSY);
    state = State::FINISH;
}
