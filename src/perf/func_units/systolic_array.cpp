#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

SystolicArray::SystolicArray(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "SystolicArray";
}

void SystolicArray::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void SystolicArray::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint8_t optype = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 25) & 0x3);

    if(optype == 0) {  // GEMM

        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;

        if(col_num_a != row_num_b) {
            logger.log("[SystolicArray][ERROR] GEMM Dimension Mismatch: input_a_col_num = %d, input_b_row_num = %d", col_num_a, row_num_b);
            assert(col_num_a == row_num_b && "SystolicArray Requires col_num_a = row_num_b");
        }

        remaining_cycles = row_num_a + col_num_a + col_num_b + 2;
        logger.log("[SystolicArray] GEMM Start, A:%d*%d, B:%d*%d, remaining = %d", row_num_a, col_num_a, row_num_b, col_num_b, remaining_cycles);

    } else if(optype == 1) {  // CONV
        // parameters

        // uint8_t dataflow = (uint8_t)(regfile.read(regfile.config_compute_rs1_reg_addr) & 0x1);
        int pad_left = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 1) & 0x3);
        int pad_right = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 3) & 0x3);
        int pad_top = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 5) & 0x3);
        int pad_bottom = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 7) & 0x3);
        int pad_mode = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 9) & 0x3);
        int weight_shape = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 11) & 0xF) + 1;
        int weight_stride = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 15) & 0x3) + 1;
        int weight_dilation = (int)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 17) & 0x1F) + 1;
        // uint8_t is_group_conv = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 22) & 0x1);
        // uint8_t int_type = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 23) & 0x3);

        int col_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 32) & 0x7FF) + 1;
        int row_num_a = (int)((regfile.read(regfile.compute_sa_rs1_reg_addr) >> 43) & 0x1F) + 1;
        int col_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
        int row_num_b = (int)((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 37) & 0x7FF) + 1;

        // compute cycle
        int k_eff = weight_dilation * (weight_shape - 1) + 1;
        int ofm_w = ((col_num_a + pad_left + pad_right - k_eff) / weight_stride) + 1;
        int ofm_h = ((row_num_a + pad_top + pad_bottom - k_eff) / weight_stride) + 1;
        int cout = col_num_b;
        int cin = row_num_b;
        // Sa_a_row_num is truncated to 6 bits: sa_a_row_num = sa_ofm_size[$clog2(ARRAY_HEIGHT):0]
        // Input_a_row_num used by SA timing is (ofm_w * ofm_h) mod 64
        int input_a_row_num = (ofm_w * ofm_h) & 0x3F;
        int input_a_col_num = weight_shape * weight_shape * cfg.systolic_array_size;
        int input_b_col_num = cout;

        remaining_cycles = input_a_row_num + input_b_col_num + input_a_col_num + 2;

        logger.log("[SystolicArray] CONV Start, pad_t/b/l/r/mode=%d/%d/%d/%d/%d, weight_shape/stride/dilation=%d/%d/%d, ifm_w=%d, ifm_h=%d, cin=%d, cout=%d, remaining = %d", 
            pad_top, pad_bottom, pad_left, pad_right, pad_mode, weight_shape, weight_stride, weight_dilation, col_num_a, row_num_a, row_num_b, col_num_b, remaining_cycles);

    } else {  // GEMV
        remaining_cycles = 10;
        logger.log("[SystolicArray] GEMV Start, remaining = %d", remaining_cycles);
    }
}

void SystolicArray::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
}