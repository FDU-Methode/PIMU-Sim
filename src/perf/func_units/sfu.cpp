#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/pipeline_controller.hpp"

SFU::SFU(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "SFU";
}

void SFU::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = controller->getSpm();
    assert(spm && "SFU requires a valid Scratchpad");

    int current_cycle = sfu_cycle;
    bool issue_read = false;
    bool issue_write = false;
    uint32_t read_addr = 0;
    uint32_t write_addr = 0;

    switch(sfu_optype) {
        case 0:   // softmax
        case 2: { // layernorm
            int work_cycle = current_cycle - spm_setup_delay;
            if(work_cycle >= 0) {
                int row_span = 2 * spm_batch_num + spm_core_delay;
                int row_idx = work_cycle / row_span;
                int row_pos = work_cycle % row_span;

                if(row_idx < spm_row_num) {
                    if(row_pos < spm_batch_num) {
                        int batch_idx = row_pos;
                        issue_read = true;
                        read_addr = spm_input_base_addr
                                  + static_cast<uint32_t>(row_idx) * spm_row_stride_bytes
                                  + static_cast<uint32_t>(batch_idx) * spm_vector_stride_bytes;
                    }

                    if((row_pos >= spm_batch_num + spm_core_delay) &&
                       (row_pos <  spm_batch_num + spm_core_delay + spm_batch_num)) {
                        int batch_idx = row_pos - (spm_batch_num + spm_core_delay);
                        issue_write = true;
                        write_addr = spm_output_base_addr
                                   + static_cast<uint32_t>(row_idx) * spm_row_stride_bytes
                                   + static_cast<uint32_t>(batch_idx) * spm_vector_stride_bytes;
                    }
                }
            }
            break;
        }

        case 1: { // gelu
            int work_cycle = current_cycle - spm_setup_delay;
            if((work_cycle >= 0) && (spm_gelu_vector_period > 0)) {
                if(((work_cycle % spm_gelu_vector_period) == 0) &&
                   (spm_issued_reads < spm_expected_reads)) {
                    issue_read = true;
                    read_addr = spm_input_base_addr
                              + static_cast<uint32_t>(spm_issued_reads) * spm_gelu_group_stride_bytes;
                }

                if((work_cycle >= spm_gelu_vector_period) &&
                   (((work_cycle - spm_gelu_vector_period) % spm_gelu_vector_period) == 0) &&
                   (spm_issued_writes < spm_expected_writes)) {
                    issue_write = true;
                    write_addr = spm_output_base_addr
                               + static_cast<uint32_t>(spm_issued_writes) * spm_gelu_group_stride_bytes;
                }
            }
            break;
        }

        case 6: { // pool_max
            int work_cycle = current_cycle - spm_setup_delay;

            if(work_cycle >= 0 && work_cycle < spm_expected_reads) {
                int row_idx = work_cycle / spm_batch_num;
                int batch_idx = work_cycle % spm_batch_num;
                issue_read = true;
                read_addr = spm_input_base_addr
                          + static_cast<uint32_t>(row_idx) * spm_row_stride_bytes
                          + static_cast<uint32_t>(batch_idx) * spm_vector_stride_bytes;
            }

            if(work_cycle >= spm_done_delay &&
               (work_cycle - spm_done_delay) < spm_expected_writes) {
                int write_work_cycle = work_cycle - spm_done_delay;
                int row_idx = write_work_cycle / spm_batch_num;
                int batch_idx = write_work_cycle % spm_batch_num;
                issue_write = true;
                write_addr = spm_output_base_addr
                           + static_cast<uint32_t>(row_idx) * spm_row_stride_bytes
                           + static_cast<uint32_t>(batch_idx) * spm_vector_stride_bytes;
            }
            break;
        }

        case 8: { // transpose
            int work_cycle = current_cycle - spm_setup_delay;
            if(work_cycle >= 0 && spm_tile_stage_cycles > 0) {
                int tile_slot = work_cycle / spm_tile_stage_cycles;
                int phase = work_cycle % spm_tile_stage_cycles;

                // Read current tile row-by-row for first PE_WIDTH cycles
                if(tile_slot < spm_tile_num && phase < cfg.systolic_array_size) {
                    int tile_idx = tile_slot;
                    int tile_row = tile_idx / spm_transpose_tile_col_num;
                    int tile_col = tile_idx % spm_transpose_tile_col_num;
                    int line_idx = phase;

                    issue_read = true;
                    read_addr = spm_input_base_addr
                              + static_cast<uint32_t>(tile_row * cfg.systolic_array_size + line_idx) * spm_row_stride_bytes
                              + static_cast<uint32_t>(tile_col) * spm_vector_stride_bytes;
                }

                // Write previous tile after a 2-cycle pipeline delay
                if(tile_slot > 0 && tile_slot <= spm_tile_num &&
                   phase >= 2 && phase < cfg.systolic_array_size + 2) {
                    int tile_idx = tile_slot - 1;
                    int tile_row = tile_idx / spm_transpose_tile_col_num;
                    int tile_col = tile_idx % spm_transpose_tile_col_num;
                    int line_idx = phase - 2;

                    issue_write = true;
                    write_addr = spm_output_base_addr
                               + static_cast<uint32_t>(tile_col * cfg.systolic_array_size + line_idx) * spm_transpose_output_row_stride_bytes
                               + static_cast<uint32_t>(tile_row) * spm_vector_stride_bytes;
                }
            }
            break;
        }

        default:
            break;
    }

    if(issue_read) {
        bool read_status = spm->ReadMemory(read_addr);
        assert(read_status && "[SFU] SPM Read Request Failed");
        // Compute Bandwidth
        if(read_status) {
            HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
            if(util_tracker) {
                util_tracker->addSpmReadBytes(cfg.systolic_array_size * cfg.sram_data_width / 8);
            }
        }
        ++spm_issued_reads;
    }

    if(issue_write) {
        bool write_status = spm->WriteMemory(write_addr);
        assert(write_status && "[SFU] SPM Write Request Failed");
        // Compute Bandwidth
        if(write_status) {
            HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
            if(util_tracker) {
                util_tracker->addSpmWriteBytes(cfg.systolic_array_size * cfg.sram_data_width / 8);
            }
        }
        ++spm_issued_writes;
    }

    ++sfu_cycle;
    remaining_cycles--;

    if(remaining_cycles == 0){
        assert(spm_issued_reads == spm_expected_reads);
        assert(spm_issued_writes == spm_expected_writes);
        completeInstruction();
    }
}


void SFU::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    int col_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 32) & 0xFFFF) + 1;
    int row_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 48) & 0xFFFF) + 1;
    bool is_quant = (regfile.read(regfile.config_sfu_rs1_reg_addr) >> 8) & 0x1;

    sfu_cycle = 0;
    sfu_optype = (uint8_t)(regfile.read(regfile.config_sfu_rs1_reg_addr) & 0x3F);
    spm_input_base_addr = (uint32_t)(regfile.read(regfile.compute_sfu_rs1_reg_addr) & 0xFFFFFFFF);
    spm_output_base_addr = (uint32_t)(regfile.read(regfile.compute_sfu_rs2_reg_addr) & 0xFFFFFFFF);
    spm_expected_reads = 0;
    spm_expected_writes = 0;
    spm_issued_reads = 0;
    spm_issued_writes = 0;
    spm_setup_delay = 0;
    spm_batch_num = 0;
    spm_row_num = 0;
    spm_core_delay = 0;
    spm_done_delay = 0;
    spm_tile_num = 0;
    spm_tile_stage_cycles = 0;
    spm_gelu_vector_period = 0;
    spm_elem_bytes = cfg.sram_data_width / 8;
    spm_vector_stride_bytes = cfg.systolic_array_size * spm_elem_bytes;
    spm_row_stride_bytes = col_num * spm_elem_bytes;
    spm_gelu_group_stride_bytes = 0;
    spm_transpose_tile_col_num = 0;
    spm_transpose_tile_row_num = 0;
    spm_transpose_output_row_stride_bytes = row_num * spm_elem_bytes;

    switch(sfu_optype) {
        case 0: { // softmax
            int batch_num = (col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int setup_delay = 1;
            int quant_delay = 1;
            int dequant_delay = 1;

            remaining_cycles = setup_delay + row_num * (2 * batch_num + 13 + quant_delay + dequant_delay) + is_quant;

            spm_setup_delay = setup_delay;
            spm_batch_num = batch_num;
            spm_row_num = row_num;
            spm_core_delay = 13 + quant_delay + dequant_delay;
            spm_expected_reads = row_num * batch_num;
            spm_expected_writes = row_num * batch_num;

            logger.log("[SFU] Softmax Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 1: { // gelu
            int gelu_num = 4;    // Num of gelu parallel processing units
            int batch_num = (col_num * row_num + gelu_num - 1) / gelu_num;
            int setup_delay = 3;
            int quant_delay = 1;
            int slow_gelu_delay = 60;
            int fast_gelu_delay = 6;
            float p_slow = 1;    // The proportion of slow gelu (0-1)

            remaining_cycles = setup_delay + batch_num * (p_slow * slow_gelu_delay + (1-p_slow) * fast_gelu_delay) + quant_delay + is_quant;
            // The cycles of gelu is related to input data, take the maximum value

            spm_setup_delay = setup_delay;
            spm_expected_reads = batch_num;
            spm_expected_writes = batch_num;
            spm_gelu_vector_period = slow_gelu_delay;
            spm_gelu_group_stride_bytes = gelu_num * spm_elem_bytes;

            logger.log("[SFU] Gelu Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 2: { // layernorm
            int batch_num = (col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int setup_delay = 1;
            int quant_delay = 1;
            int dequant_delay = 1;
            int reduce_delay = 5;   // in buffer + 2-stage add trees + valid_sum/valid_sqrsum alignment
            int div_delay = 17;     // int_div_radix_16_v4 (WIDTH=32 typical): PRE+ITER+POST
            int post_delay = 4;     // log2 + pow2 + sign/output register path
            int layernorm_core_delay = reduce_delay + div_delay + post_delay;

            // Per row: read B groups, then write B groups, plus fixed core pipeline.
            remaining_cycles = setup_delay + row_num * (2 * batch_num + layernorm_core_delay + quant_delay + dequant_delay) + is_quant;
            // The cycles of layernorm is related to input data, take the maximum value

            spm_setup_delay = setup_delay;
            spm_batch_num = batch_num;
            spm_row_num = row_num;
            spm_core_delay = layernorm_core_delay + quant_delay + dequant_delay;
            spm_expected_reads = row_num * batch_num;
            spm_expected_writes = row_num * batch_num;

            logger.log("[SFU] Layernorm Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 6: { // pool_max
            // resample(pooling) path in RTL:
            // 1) sfu_req_en -> sfu_req_en_dly : 1 cycle
            // 2) read loop over all input vectors: row_num * ceil(col_num / PE_WIDTH)
            // 3) resample_done -> resample_done_out : 2-cycle delay pipeline
            int batch_num = (col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int setup_delay = 1;
            int done_delay = 2;

            remaining_cycles = setup_delay + row_num * batch_num + done_delay + is_quant;

            spm_setup_delay = setup_delay;
            spm_batch_num = batch_num;
            spm_row_num = row_num;
            spm_done_delay = done_delay;
            spm_expected_reads = row_num * batch_num;
            spm_expected_writes = row_num * batch_num;

            logger.log("[SFU] Pool_max Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        case 8: { // transpose
            // Transpose is processed in PE_WIDTH x PE_WIDTH tiles
            // HW timing fits: latency = (PE_WIDTH + 2) * (tile_num + 1), where +1 captures pipeline fill/drain and per-stage cost is PE+2 cycles
            // Applying padding has no effect on transpose done timing
            int tile_col_num = (col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int tile_row_num = (row_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int tile_num = tile_col_num * tile_row_num;
            int tile_stage_cycles = cfg.systolic_array_size + 2;

            remaining_cycles = tile_stage_cycles * (tile_num + 1) + is_quant;

            spm_setup_delay = 0;
            spm_tile_num = tile_num;
            spm_tile_stage_cycles = tile_stage_cycles;
            spm_transpose_tile_col_num = tile_col_num;
            spm_transpose_tile_row_num = tile_row_num;
            spm_expected_reads = tile_num * cfg.systolic_array_size;
            spm_expected_writes = tile_num * cfg.systolic_array_size;

            logger.log("[SFU] Transpose Start, col_num=%d, row_num=%d, remaining = %d", col_num, row_num, remaining_cycles);
            break;
        }

        default: {
            logger.log("[SFU][ERROR] Unsupported SFU Optype = %u", (unsigned)sfu_optype);
            throw std::runtime_error("Unsupported SFU Optype: " + std::to_string((unsigned)sfu_optype));
        }
    }
}

void SFU::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();

    // Update ISR Reg
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    isr_val |= (1ULL << 5);
    regfile.write(regfile.isr_reg_addr, isr_val);
    logger.log("[SFU] ISR Reg[5] Updated");
}
