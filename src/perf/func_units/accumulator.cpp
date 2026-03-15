#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

Accumulator::Accumulator(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
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

    uint8_t matadd_start = (regfile.read(regfile.start_reg_addr) >> 3) & 0x1;
    bool acc_dest = (regfile.read(regfile.config_compute_rs1_reg_addr) >> 27) & 0x1;

    if(matadd_start) {
        int row_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 40) & 0xFF);
        int col_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 32) & 0xFF);
        remaining_cycles = (row_num * col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size + 5;
        logger.log("[Accumulator] Matadd Start, row_num=%d, col_num=%d, remaining = %d", row_num, col_num, remaining_cycles);
    } else {
        remaining_cycles = acc_dest ? 3 : 4;
        logger.log("[Accumulator] Start, remaining = %d", remaining_cycles);
    }
}

void Accumulator::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}