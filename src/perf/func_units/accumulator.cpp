#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/pipeline_controller.hpp"

Accumulator::Accumulator(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "Accumulator";

    acc_sram = std::make_unique<Scratchpad>(logger, cfg, cfg.acc_bytes, cfg.acc_ports, cfg.acc_data_width, cfg.systolic_array_size);
}

void Accumulator::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = controller->getSpm();
    assert(spm && "Accumulator requires a valid Scratchpad");
    assert(acc_sram && "Accumulator local SRAM is NULL");

    // Matadd
    if(matadd_start) {
        // Read From Local SRAM
        if(remaining_cycles >= 4) {
            int read_beat = total_cycles - remaining_cycles;
            uint32_t read_addr_a = matadd_a_base_addr + static_cast<uint32_t>(read_beat) * cfg.systolic_array_size * cfg.acc_data_width / 8;
            uint32_t read_addr_b = matadd_b_base_addr + static_cast<uint32_t>(read_beat) * cfg.systolic_array_size * cfg.acc_data_width / 8;
            bool read_status_a = acc_sram->accessPort(0, read_addr_a);
            assert(read_status_a && "[Accumulator] Local SRAM Read Request Failed");
            bool read_status_b = acc_sram->accessPort(1, read_addr_b);
            assert(read_status_b && "[Accumulator] Local SRAM Read Request Failed");
            // Compute Bandwidth
            if(read_status_a && read_status_b) {
                HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
                if(util_tracker) {
                    util_tracker->addAccReadBytes(2 * cfg.systolic_array_size * cfg.acc_data_width / 8);
                }
            }
        }
        // Write To Scratchpad
        if(total_cycles - remaining_cycles >= 4) {
            int write_beat = total_cycles - remaining_cycles - 4;
            uint32_t write_addr = matadd_output_base_addr + static_cast<uint32_t>(write_beat) * cfg.systolic_array_size * cfg.sram_data_width / 8;
            bool write_status = spm->WriteMemory(write_addr);
            assert(write_status && "[Accumulator] SPM Write Request Failed");
            // Compute Bandwidth
            if(write_status) {
                HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
                if(util_tracker) {
                    util_tracker->addSpmWriteBytes(cfg.systolic_array_size * cfg.sram_data_width / 8);
                }
            }
        }
    }

    // CONV / GEMM
    if(sa_start) {
        // Read From Local SRAM
        if(is_acc && remaining_cycles >= tail_cycles) {
            int read_beat = total_cycles - remaining_cycles;
            uint32_t read_addr = biaspsum_base_addr + static_cast<uint32_t>(read_beat) * cfg.systolic_array_size * cfg.acc_data_width / 8;
            bool read_status = acc_sram->ReadMemory(read_addr);
            assert(read_status && "[Accumulator] Local SRAM Read Request Failed");
            // Compute Bandwidth
            if(read_status) {
                HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
                if(util_tracker) {
                    util_tracker->addAccReadBytes(cfg.systolic_array_size * cfg.acc_data_width / 8);
                }
            }
        }
        // Write To Local SRAM / Scratchpad
        if(acc_dest) {
            // Write To Local SRAM
            if(total_cycles - remaining_cycles >= 3) {
                int write_beat = total_cycles - remaining_cycles - 3;
                uint32_t write_addr = (optype == 0) ? (sa_output_base_addr + static_cast<uint32_t>(write_beat) * sa_output_stride * cfg.acc_data_width / 8) :
                                      (optype == 1) ? (sa_output_base_addr + static_cast<uint32_t>(write_beat) * sa_output_stride * sa_input_b_col_num * cfg.acc_data_width / 8) :
                                      0;
                bool write_status = acc_sram->WriteMemory(write_addr);
                assert(write_status && "[Accumulator] Local SRAM Write Request Failed");
                // Compute Bandwidth
                if(write_status) {
                    HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
                    if(util_tracker) {
                        util_tracker->addAccWriteBytes(cfg.systolic_array_size * cfg.acc_data_width / 8);
                    }
                }
            }
        } else {
            // Write To Scratchpad
            if(total_cycles - remaining_cycles >= 4) {
                int write_beat = total_cycles - remaining_cycles - 4;
                uint32_t write_addr = (optype == 0) ? (sa_output_base_addr + static_cast<uint32_t>(write_beat) * sa_output_stride * cfg.sram_data_width / 8) :
                                      (optype == 1) ? (sa_output_base_addr + static_cast<uint32_t>(write_beat) * sa_output_stride * sa_input_b_col_num * cfg.sram_data_width / 8) :
                                      0;
                bool write_status = spm->WriteMemory(write_addr);
                assert(write_status && "[Accumulator] SPM Write Request Failed");
                // Compute Bandwidth
                if(write_status) {
                    HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
                    if(util_tracker) {
                        util_tracker->addSpmWriteBytes(cfg.systolic_array_size * cfg.sram_data_width / 8);
                    }
                }
            }
        }
    }



    remaining_cycles--;
    if(remaining_cycles == 0){
        completeInstruction();
    }
}

void Accumulator::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    matadd_start = (regfile.read(regfile.start_reg_addr) >> 3) & 0x1;
    sa_start     = (regfile.read(regfile.start_reg_addr) >> 2) & 0x1;

    optype = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 25) & 0x3);
    is_acc = (regfile.read(regfile.config_accumulate_rs2_reg_addr) >> 48) & 0x1;
    is_bias = (regfile.read(regfile.config_accumulate_rs2_reg_addr) >> 53) & 0x1;
    acc_dest = (regfile.read(regfile.config_compute_rs1_reg_addr) >> 27) & 0x1;

    biaspsum_base_addr = (regfile.read(regfile.config_accumulate_rs1_reg_addr)) & 0xFFFFFFFF;
    sa_output_base_addr = (regfile.read(regfile.config_accumulate_rs2_reg_addr)) & 0xFFFFFFFF;
    matadd_a_base_addr = (regfile.read(regfile.compute_matadd_rs1_reg_addr)) & 0xFFFFFFFF;
    matadd_b_base_addr = (regfile.read(regfile.compute_matadd_rs1_reg_addr) >> 32) & 0xFFFFFFFF;
    matadd_output_base_addr = (regfile.read(regfile.compute_matadd_rs2_reg_addr)) & 0xFFFFFFFF;
    sa_input_b_col_num = ((regfile.read(regfile.compute_sa_rs2_reg_addr) >> 32) & 0x1F) + 1;
    biaspsum_stride = (regfile.read(regfile.config_accumulate_rs1_reg_addr) >> 32) & 0xFFFF;
    sa_output_stride = (regfile.read(regfile.config_accumulate_rs2_reg_addr) >> 32) & 0xFFFF;

    tail_cycles = acc_dest ? 3 : 4;

    if(matadd_start) {  // MATADD
        int row_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 40) & 0xFF) + 1;
        int col_num = (int)((regfile.read(regfile.compute_matadd_rs2_reg_addr) >> 32) & 0xFF) + 1;
        remaining_cycles = (row_num * col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size + 5;
        total_cycles = remaining_cycles;
        logger.log("[Accumulator] Matadd Start, row_num=%d, col_num=%d, remaining = %d", row_num, col_num, remaining_cycles);
        return;
    }

    if(sa_start) {
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
            total_cycles = remaining_cycles;

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
            total_cycles = remaining_cycles;

            logger.log("[Accumulator] CONV Start, ifm=%dx%d, cin=%d, cout=%d, k=%d, s=%d, d=%d, "
                        "ofm=%dx%d, sa_row_num=%d, remaining = %d",
                        col_num_a, row_num_a, row_num_b, col_num_b, weight_shape, weight_stride, weight_dilation,
                        ofm_w, ofm_h, sa_row_num, remaining_cycles);
            return;
        }
    }

    logger.log("[Accumulator][ERROR] Unsupported Execute: start_reg=0x%lx, optype=%u", regfile.read(regfile.start_reg_addr), optype);
    assert(false && "Accumulator Unsupported Excute");
}

void Accumulator::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
    total_cycles = 0;

    // Update ISR Reg
    assert((matadd_start || sa_start) && "Accumulator No ISR Reg Write");
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    if(matadd_start) {
        isr_val |= (1ULL << 3);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[Accumulator] ISR Reg[3] Updated");
        matadd_start = false;
    }

    if(sa_start) {
        isr_val |= (1ULL << 2);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[Accumulator] ISR Reg[2] Updated");
        sa_start = false;
    }

    is_acc = false;
    is_bias = false;
}