#include <cassert>
#include "../../../include/perf/perf_func_unit.hpp"

SFU::SFU(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "SFU";
}

void SFU::tick(){
    if(isBusy()){
        remaining_cycles--;
        if(remaining_cycles == 0){
            completeInstruction();
        }
    }
}

void SFU::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    uint8_t sfu_optype = (uint8_t)(regfile.read(regfile.config_sfu_rs1_reg_addr) & 0x3F);
    int col_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 32) & 0xFFFF) + 1;
    int row_num = (int)((regfile.read(regfile.compute_sfu_rs1_reg_addr) >> 48) & 0xFFFF) + 1;
    bool is_quant = (regfile.read(regfile.config_sfu_rs1_reg_addr) >> 8) & 0x1;

    switch(sfu_optype) {
        case 0: { // softmax
            int batch_num = (col_num + cfg.systolic_array_size - 1) / cfg.systolic_array_size;
            int setup_delay = 1;
            int quant_delay = 1;
            int dequant_delay = 1;

            remaining_cycles = setup_delay + row_num * (2 * batch_num + 13 + quant_delay + dequant_delay) + is_quant;
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