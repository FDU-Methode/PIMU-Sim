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
    uint8_t matvec_start = (regfile.read(regfile.start_reg_addr) >> 4) & 0x1;

    uint8_t optype = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 25) & 0x3);
    bool acc_dest = (regfile.read(regfile.config_compute_rs1_reg_addr) >> 27) & 0x1;

    const int tail_cycles = acc_dest ? 3 : 4;

    if(matadd_start) {  // MATADD
        int row_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 40) & 0xFF);
        int col_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 32) & 0xFF);
        remaining_cycles = (row_num * col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size + 5;
        logger.log("[Accumulator] Matadd Start, row_num=%d, col_num=%d, remaining = %d", row_num, col_num, remaining_cycles);
        return;
    }

    if(matvec_start) {  // MATVEC (Not Used)
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        remaining_cycles = row_num_a + tail_cycles;
        logger.log("[Accumulator] MATVEC Start, row_num=%d, remaining = %d", row_num_a, remaining_cycles);
        return;
    }

    if(optype == 0) {  // GEMM
        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;

        if(col_num_a != row_num_b) {
            logger.log("[Accumulator][ERROR] GEMM Dimension Mismatch: input_a_col_num = %d, input_b_row_num = %d", col_num_a, row_num_b);
            assert(col_num_a == row_num_b && "Accumulator GEMM Requires col_num_a = row_num_b");
        }

        remaining_cycles = row_num_a + tail_cycles;

        logger.log("[Accumulator] GEMM Start, M=%d, K=%d, N=%d, remaining = %d",
                   row_num_a, col_num_a, col_num_b, remaining_cycles);
        return;
    }

    if(optype == 1) {  // CONV
        // parameters

        int pad_left = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 1) & 0x3);
        int pad_right = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 3) & 0x3);
        int pad_top = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 5) & 0x3);
        int pad_bottom = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 7) & 0x3);
        int weight_shape = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 11) & 0xF) + 1;
        int weight_stride = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 15) & 0x3) + 1;
        int weight_dilation = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 17) & 0x1F) + 1;
        // uint8_t is_group_conv = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 22) & 0x1);
        // uint8_t int_type = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 23) & 0x3);

        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;  // ifm_w
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;  // ifm_h
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;  // cout
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;  // cin

        int k_eff = weight_dilation * (weight_shape - 1) + 1;
        int ofm_w = ((col_num_a + pad_left + pad_right - k_eff) / weight_stride) + 1;
        int ofm_h = ((row_num_a + pad_top + pad_bottom - k_eff) / weight_stride) + 1;

        assert((ofm_w > 0) && (ofm_h > 0) && "Accumulator CONV Invalid Output Shape");

        int sa_row_num = (ofm_w * ofm_h) & 0x3F;

        remaining_cycles = sa_row_num + tail_cycles;

        logger.log("[Accumulator] CONV Start, ifm=%dx%d, cin=%d, cout=%d, k=%d, s=%d, d=%d, "
                   "ofm=%dx%d, sa_row_num=%d, remaining = %d",
                   col_num_a, row_num_a, row_num_b, col_num_b, weight_shape, weight_stride, weight_dilation,
                   ofm_w, ofm_h, sa_row_num, remaining_cycles);
        return;
    }

    logger.log("[Accumulator][ERROR] Unsupported Execute: start_reg=0x%lx, optype=%u", regfile.read(regfile.start_reg_addr), optype);
    assert(false && "Accumulator Unsupported Excute");
}

void Accumulator::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}