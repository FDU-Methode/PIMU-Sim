#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

GEMV::GEMV(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
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

    // Update ISR Reg
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    isr_val |= (1ULL << 4);
    regfile.write(regfile.isr_reg_addr, isr_val);
    logger.log("[GEMV] ISR Reg[4] Updated");
}