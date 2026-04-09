#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_npu_core.hpp"

Accumulator::Accumulator(PerfSim* sim, PerfNpuCore* core, Logger& logger, const Config& cfg) :
    PerfFunctionalUnit(sim, core, logger, cfg) {
    UnitName = "Accumulator";

    acc_sram = std::make_unique<AccSram>(logger, cfg);
}

void Accumulator::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = core->getSpm();
    assert(spm && "Accumulator requires a valid Scratchpad");
    AccSram* acc_sram = core->getAccSram();
    assert(acc_sram && "Accumulator local SRAM is NULL");

    // Matadd
    if(matadd_start) {
        // Read From Local SRAM
        if(remaining_cycles >= 4) {
            int read_beat = total_cycles - remaining_cycles;
            uint32_t read_addr_a = matadd_a_base_addr + static_cast<uint32_t>(read_beat) * cfg.systolic_array_size * cfg.acc_data_width / 8;
            uint32_t read_addr_b = matadd_b_base_addr + static_cast<uint32_t>(read_beat) * cfg.systolic_array_size * cfg.acc_data_width / 8;
            bool read_status_a = acc_sram->accessDataPort(0, read_addr_a);
            assert(read_status_a && "[Accumulator] Local SRAM Read Request Failed");
            bool read_status_b = acc_sram->accessDataPort(1, read_addr_b);
            assert(read_status_b && "[Accumulator] Local SRAM Read Request Failed");
            // Compute Bandwidth
            if(read_status_a && read_status_b) {
                HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
                if(util_tracker) {
                    util_tracker->addAccDataReadBytes(2 * cfg.systolic_array_size * cfg.acc_data_width / 8);
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
                HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
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
            bool read_status = is_bias ? acc_sram->ReadBias(read_addr) : acc_sram->ReadData(read_addr);
            assert(read_status && "[Accumulator] ACC SRAM Read Request Failed");
            // Compute Bandwidth
            if(read_status) {
                HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
                if(util_tracker) {
                    if(is_bias) util_tracker->addAccBiasReadBytes(cfg.systolic_array_size * cfg.acc_data_width / 8);
                    else        util_tracker->addAccDataReadBytes(cfg.systolic_array_size * cfg.acc_data_width / 8);
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
                bool write_status = acc_sram->WriteData(write_addr);
                assert(write_status && "[Accumulator] Local SRAM Write Request Failed");
                // Compute Bandwidth
                if(write_status) {
                    HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
                    if(util_tracker) {
                        util_tracker->addAccDataWriteBytes(cfg.systolic_array_size * cfg.acc_data_width / 8);
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
                    HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
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

void Accumulator::execute() {
    logger.log("[Accumulator][ERROR] execute() without opcode is no longer supported");
    assert(false && "Accumulator::execute(uint8_t opcode) should be used");
}

void Accumulator::execute(uint8_t opcode){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    SAInstQueue* sa_queue = core->getSAInstQueue();
    assert(sa_queue && "[Accumulator] SA queue is NULL");

    const SAInstPacket& inst_packet = sa_queue->getInstBuffer();
    const Inst& inst0 = inst_packet.inst0;
    const Inst& inst1 = inst_packet.inst1;
    const Inst& inst2 = inst_packet.inst2;

    state = State::BUSY;

    if(opcode == 0x03) current_task = SATaskType::SA;
    else if(opcode == 0x04) current_task = SATaskType::MATADD;
    else throw std::runtime_error("[Accumulator] Unsupported Opcode");

    matadd_start = (current_task == SATaskType::MATADD);
    sa_start     = (current_task == SATaskType::SA);

    optype = static_cast<uint8_t>((inst1.low >> 1) & 0x3);
    acc_dest = (inst1.low >> 3) & 0x1;
    is_acc = (inst1.low >> 6) & 0x1;
    is_bias = (inst1.low >> 11) & 0x1;

    biaspsum_base_addr = static_cast<uint32_t>(inst1.high & 0xFFFFFFFF);
    sa_output_base_addr = static_cast<uint32_t>((inst1.high >> 32) & 0xFFFFFFFF);
    matadd_a_base_addr = static_cast<uint32_t>((inst0.low >> 32) & 0xFFFFFFFF);
    matadd_b_base_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
    matadd_output_base_addr = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);
    sa_input_b_col_num = static_cast<int>((inst2.high >> 32) & 0x1F) + 1;
    biaspsum_stride = static_cast<int>((inst1.low >> 32) & 0xFFFF);
    sa_output_stride = static_cast<int>((inst1.low >> 48) & 0xFFFF);

    tail_cycles = acc_dest ? 3 : 4;

    if(matadd_start) {  // MATADD
        int row_num = static_cast<int>((inst0.low >> 16) & 0xFF) + 1;
        int col_num = static_cast<int>((inst0.low >> 8) & 0xFF) + 1;
        remaining_cycles = (row_num * col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size + 5;
        total_cycles = remaining_cycles;
        logger.log("[Accumulator] Matadd Start, row_num=%d, col_num=%d, remaining = %d", row_num, col_num, remaining_cycles);
        return;
    }

    if(sa_start) {
        if(optype == 0) {  // GEMM
            int col_num_a = static_cast<int>((inst2.low >> 32) & 0x7FF) + 1;
            int row_num_a = static_cast<int>((inst2.low >> 43) & 0x1F) + 1;
            int col_num_b = static_cast<int>((inst2.high >> 32) & 0x1F) + 1;
            int row_num_b = static_cast<int>((inst2.high >> 37) & 0x7FF) + 1;

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

            int pad_left = (inst0.low >> 9) & 0x3;
            int pad_right = (inst0.low >> 12) & 0x3;
            int pad_top = (inst0.low >> 14) & 0x3;
            int pad_bottom = (inst0.low >> 16) & 0x3;
            int weight_shape = ((inst0.low >> 19) & 0xF) + 1;
            int weight_stride = ((inst0.low >> 23) & 0x3) + 1;
            int weight_dilation = ((inst0.low >> 25) & 0x1F) + 1;
            // uint8_t is_group_conv = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 22) & 0x1);
            // uint8_t int_type = (uint8_t)((regfile.read(regfile.config_compute_rs1_reg_addr) >> 23) & 0x3);

            int col_num_a = static_cast<int>((inst2.low >> 32) & 0x7FF) + 1;  // ifm_w
            int row_num_a = static_cast<int>((inst2.low >> 43) & 0x1F) + 1;  // ifm_h
            int col_num_b = static_cast<int>((inst2.high >> 32) & 0x1F) + 1;  // cout
            int row_num_b = static_cast<int>((inst2.high >> 37) & 0x7FF) + 1;  // cin

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

    logger.log("[Accumulator][ERROR] Unsupported Execute: opcode=0x%02x, optype=%u", opcode, optype);
    assert(false && "Accumulator Unsupported Execute");
}

void Accumulator::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();
    total_cycles = 0;

    matadd_start = false;
    sa_start = false;
    is_acc = false;
    is_bias = false;
    current_task = SATaskType::NONE;
}