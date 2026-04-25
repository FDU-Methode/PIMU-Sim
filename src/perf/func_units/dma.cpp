#include <algorithm>
#include <cassert>
#include <cstdint>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_sim.hpp"
#include "../../../include/perf/pipeline_controller.hpp"

namespace {

constexpr int kMaxAxiBurstBeats = 256;

inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

inline int ilog2_pow2(int v) {
    assert(v > 0 && (v & (v - 1)) == 0);
    int s = 0;
    while ((1 << s) < v) ++s;
    return s;
}

inline int scale_by_log2_shift(int bytes, int shift_log2) {
    if (shift_log2 >= 0) return bytes << shift_log2;
    return bytes >> (-shift_log2);
}

struct SizeDecode {
    int dram_bytes = 1;
    int local_bytes = 1;
};

SizeDecode decode_mvin_sizes(uint64_t cfg_word) {
    const int input_type_hi = static_cast<int>((cfg_word >> 1) & 0x1);
    const int precision = static_cast<int>((cfg_word >> 2) & 0x3);
    const int is_quant = static_cast<int>((cfg_word >> 4) & 0x1);

    const int key = (input_type_hi << 3) | (precision << 1) | is_quant;
    SizeDecode ret{1, 1};

    // RTL: case ({cfg_mvin_input_type[1], cfg_mvin_input_precision, cfg_mvin_is_quant})
    switch (key) {
        case 0b0010: ret = {1, 1}; break; // int8 -> int8
        case 0b0100: ret = {2, 2}; break; // fp16 -> fp16
        case 0b1100: ret = {2, 2}; break; // fp16(bias path) -> fp16
        case 0b1010: ret = {4, 4}; break; // int32 -> int32
        case 0b0011: ret = {1, 4}; break; // int8 -> int32 (quant/dequant)
        default:     ret = {1, 1}; break;
    }

    return ret;
}

SizeDecode decode_mvout_sizes(uint64_t cfg_word) {
    const int output_type_lsb = static_cast<int>((cfg_word >> 1) & 0x1);
    const int precision = static_cast<int>((cfg_word >> 2) & 0x3);

    const int key = (output_type_lsb << 2) | precision;
    SizeDecode ret{1, 1};

    // RTL: case ({cfg_mvout_output_type[0], cfg_mvout_output_precision})
    switch (key) {
        case 0b010: ret = {2, 2}; break; // fp16
        case 0b001: ret = {1, 1}; break; // int8
        case 0b101: ret = {4, 4}; break; // int32
        default:    ret = {1, 1}; break;
    }

    return ret;
}

int calc_beat_payload_bytes(int row_total_beats,
                            int beat_idx,
                            int head_offset,
                            int tail_bytes,
                            int row_payload_bytes,
                            int axi_beat_bytes) {
    if (row_total_beats <= 1) return row_payload_bytes;
    if (beat_idx == 0) return axi_beat_bytes - head_offset;
    if (beat_idx == row_total_beats - 1) return (tail_bytes == 0) ? axi_beat_bytes : tail_bytes;
    return axi_beat_bytes;
}

void build_dma_plans(uint32_t dram_base_addr,
                     int dram_row_stride_bytes,
                     int dram_row_payload_bytes,
                     int row_count,
                     int axi_beat_bytes,
                     std::vector<int>& row_beat_plan,
                     std::vector<int>& burst_beat_plan) {
    row_beat_plan.clear();
    burst_beat_plan.clear();

    for (int r = 0; r < row_count; ++r) {
        const uint32_t row_addr = dram_base_addr + static_cast<uint32_t>(r * dram_row_stride_bytes);
        const int head_offset = static_cast<int>(row_addr % static_cast<uint32_t>(axi_beat_bytes));
        const int row_beats = ceil_div_int(head_offset + dram_row_payload_bytes, axi_beat_bytes);

        row_beat_plan.push_back(row_beats);

        int remain = row_beats;
        uint32_t burst_addr = row_addr & ~(static_cast<uint32_t>(axi_beat_bytes) - 1U);

        while (remain > 0) {
            int beats_to_4k = (4096 - static_cast<int>(burst_addr & 0xFFFU)) / axi_beat_bytes;
            if (beats_to_4k <= 0) beats_to_4k = 4096 / axi_beat_bytes;

            const int burst_beats = std::min({remain, beats_to_4k, kMaxAxiBurstBeats});
            burst_beat_plan.push_back(burst_beats);

            remain -= burst_beats;
            burst_addr += static_cast<uint32_t>(burst_beats * axi_beat_bytes);
        }
    }
}

}  // namespace

DMA::DMA(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg) :
    PerfFunctionalUnit(sim, controller, logger, regfile, cfg) {
    UnitName = "DMA";
}

void DMA::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = controller->getSpm();
    assert(spm && "DMA requires a valid Scratchpad");
    Scratchpad* acc_sram = controller->getAccSram();
    assert(acc_sram && "DMA requires a valid ACC SRAM");

    // 1) local memory side transfer (models sram_rd/sram_wr side in DMA RTL)
    if (current_row_idx < row_beat_plan.size()) {
        const int row_total_beats = row_beat_plan[current_row_idx];
        const uint32_t dram_row_base = dram_base_addr + static_cast<uint32_t>(current_row_idx * dram_row_stride_bytes);
        const uint32_t local_row_base = spm_base_addr + static_cast<uint32_t>(current_row_idx * spm_row_stride_bytes);

        const int head_offset = static_cast<int>(dram_row_base % static_cast<uint32_t>(axi_beat_bytes));
        const int tail_bytes = (head_offset + dram_row_payload_bytes) % axi_beat_bytes;

        const int beat_payload_bytes = calc_beat_payload_bytes(
            row_total_beats,
            current_row_beat_idx,
            head_offset,
            tail_bytes,
            dram_row_payload_bytes,
            axi_beat_bytes);

        const int local_offset_bytes = scale_by_log2_shift(current_row_dram_offset_bytes, local_minus_dram_log2);
        const uint32_t local_addr = local_row_base + static_cast<uint32_t>(local_offset_bytes);

        bool access_ok = false;
        if (mvin_start) {
            access_ok = source ? acc_sram->WriteMemory(local_addr) : spm->WriteMemory(local_addr);
        } else if (mvout_start) {
            access_ok = source ? acc_sram->ReadMemory(local_addr) : spm->ReadMemory(local_addr);
        }
        
        assert(access_ok && "DMA Read or Write SPM Error");

        assert(access_ok && "DMA Read or Write SPM Error");

        // Match RTL behavior closer: if local memory is not ready, stall this beat and retry.
        if (access_ok) {
            HardWareUtilizationTracker* util_tracker = controller->getUtilTracker();
            if (util_tracker) {
                const int local_payload_bytes = scale_by_log2_shift(beat_payload_bytes, local_minus_dram_log2);

                if (mvin_start) {
                    util_tracker->addDdrReadBytes(axi_beat_bytes);
                    if (source) {
                        util_tracker->addAccWriteBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    } else {
                        util_tracker->addSpmWriteBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    }
                } else if (mvout_start) {
                    if (source) {
                        util_tracker->addAccReadBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    } else {
                        util_tracker->addSpmReadBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    }
                    util_tracker->addDdrWriteBytes(axi_beat_bytes);
                }
            }

            current_row_dram_offset_bytes += beat_payload_bytes;
            ++current_row_beat_idx;

            if (current_row_beat_idx >= row_total_beats) {
                ++current_row_idx;
                current_row_beat_idx = 0;
                current_row_dram_offset_bytes = 0;
            }

            if (current_burst_beats_remaining > 0) {
                --current_burst_beats_remaining;
                if (current_burst_beats_remaining == 0) {
                    ++current_burst_idx;
                    if (current_burst_idx < burst_beat_plan.size()) {
                        current_burst_beats_remaining = burst_beat_plan[current_burst_idx];
                    }
                }
            }
        }
    }

    // 2) DDR response side
    auto core = simulator->getCore();
    Message get_msg{};
    if (!resp_done && core->tryConsumeResp(expectedMsgType, get_msg)) {
        logger.log("[DMA] get expected resp type = %s", MsgTypeToStr(get_msg.type).c_str());
        resp_done = true;
    }

    const bool local_transfer_done = (current_row_idx >= row_beat_plan.size());

    if (resp_done && local_transfer_done) {
        logger.log("[DMA] DDR response and local transfer both completed, DMA complete");
        completeInstruction();
    }

}

void DMA::execute(){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    state = State::BUSY;

    resp_done = false;
    row_beat_plan.clear();
    burst_beat_plan.clear();
    current_row_idx = 0;
    current_row_beat_idx = 0;
    current_row_dram_offset_bytes = 0;
    current_burst_idx = 0;
    current_burst_beats_remaining = 0;

    uint64_t start_reg = regfile.read(regfile.start_reg_addr);
    mvin_start  = (start_reg & (1ULL << 0)) != 0;
    mvout_start = (start_reg & (1ULL << 1)) != 0;

    Message req{};
    uint64_t cfg_word = 0;
    SizeDecode sizes{};

    if(mvin_start && !mvout_start){
        current_task = DMATaskType::MVIN;
        cfg_word = regfile.read(regfile.config_mvin_rs1_reg_addr);

        req.type = MsgType::LD_REQ;
        source = ((cfg_word >> 5) & 0x1) != 0;  // cfg_mvin_dest: 0->SPM, 1->ACC
        is_bias = ((cfg_word >> 6) & 0x1) != 0;

        req.row_num = (int)((regfile.read(regfile.mvin_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvin_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((cfg_word >> 48) & 0xFFFF);
        req.sram_stride = (int)((cfg_word >> 32) & 0xFFFF);

        sizes = decode_mvin_sizes(cfg_word);
        req.data_size = sizes.dram_bytes;

        req.quant = ((cfg_word >> 4) & 0x1) != 0;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvin_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvin_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::LD_RESP;

        logger.log("[DMA] MVIN: send LD_REQ, src=0x%x dst=0x%x rows=%d cols=%d dram_stride=%d sram_stride=%d dram_bytes=%d local_bytes=%d dest=%s quant=%d",
                   req.src_addr, req.dst_addr, req.row_num + 1, req.col_num + 1,
                   req.dram_stride, req.sram_stride, sizes.dram_bytes, sizes.local_bytes,
                   source ? "ACC" : "SPM", req.quant ? 1 : 0);

    } else if(!mvin_start && mvout_start){
        current_task = DMATaskType::MVOUT;
        cfg_word = regfile.read(regfile.config_mvout_rs1_reg_addr);

        req.type = MsgType::ST_REQ;
        source = ((cfg_word >> 5) & 0x1) != 0;  // cfg_mvout_source: 0->SPM, 1->ACC
        is_bias = false;

        req.row_num = (int)((regfile.read(regfile.mvout_rs1_reg_addr) >> 32) & 0xFFFF);
        req.col_num = (int)((regfile.read(regfile.mvout_rs2_reg_addr) >> 32) & 0xFFFF);
        req.dram_stride = (int)((cfg_word >> 48) & 0xFFFF);
        req.sram_stride = (int)((cfg_word >> 32) & 0xFFFF);

        sizes = decode_mvout_sizes(cfg_word);
        req.data_size = sizes.dram_bytes;

        req.quant = ((cfg_word >> 4) & 0x1) != 0;
        req.src_addr = (uint32_t)(regfile.read(regfile.mvout_rs1_reg_addr) & 0xFFFFFFFF);
        req.dst_addr = (uint32_t)(regfile.read(regfile.mvout_rs2_reg_addr) & 0xFFFFFFFF);
        expectedMsgType = MsgType::ST_RESP;

        logger.log("[DMA] MVOUT: send ST_REQ, src=0x%x dst=0x%x rows=%d cols=%d dram_stride=%d sram_stride=%d dram_bytes=%d local_bytes=%d source=%s quant=%d",
                   req.src_addr, req.dst_addr, req.row_num + 1, req.col_num + 1,
                   req.dram_stride, req.sram_stride, sizes.dram_bytes, sizes.local_bytes,
                   source ? "ACC" : "SPM", req.quant ? 1 : 0);

    } else{
        logger.log("[DMA][ERROR] Invalid start_reg for DMA: 0x%lx", start_reg);
        assert(false && "DMA execute requires exactly one of MVIN/MVOUT");
    }

    axi_beat_bytes = cfg.axi_data_width / 8;

    const int dram_log2 = ilog2_pow2(sizes.dram_bytes);
    const int local_log2 = ilog2_pow2(sizes.local_bytes);
    local_minus_dram_log2 = local_log2 - dram_log2;

    dram_base_addr = (req.type == MsgType::LD_REQ) ? req.src_addr : req.dst_addr;
    dram_row_stride_bytes = req.dram_stride * req.data_size;
    dram_row_payload_bytes = (req.col_num + 1) * req.data_size;

    spm_base_addr = (req.type == MsgType::LD_REQ) ? req.dst_addr : req.src_addr;
    spm_row_stride_bytes = req.sram_stride * sizes.local_bytes;

    build_dma_plans(dram_base_addr,
                    dram_row_stride_bytes,
                    dram_row_payload_bytes,
                    req.row_num + 1,
                    axi_beat_bytes,
                    row_beat_plan,
                    burst_beat_plan);

    if(!burst_beat_plan.empty()) {
        current_burst_beats_remaining = burst_beat_plan[0];
    }

    auto core = simulator->getCore();
    core->sendMsg(req);
    logger.log("[DMA] Send request to DDR, DMA set to busy (rows=%zu, bursts=%zu)",
               row_beat_plan.size(), burst_beat_plan.size());
}

void DMA::completeInstruction(){
    PerfFunctionalUnit::completeInstruction();

    resp_done = false;
    row_beat_plan.clear();
    burst_beat_plan.clear();
    current_row_idx = 0;
    current_row_beat_idx = 0;
    current_row_dram_offset_bytes = 0;
    current_burst_idx = 0;
    current_burst_beats_remaining = 0;

    // Update ISR Reg
    assert((mvin_start || mvout_start) && "DMA No ISR Reg Write");
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);

    if(mvin_start) {
        isr_val |= (1ULL << 0);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMA] ISR Reg[0] Updated");
        mvin_start = false;
    }

    if(mvout_start) {
        isr_val |= (1ULL << 1);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMA] ISR Reg[1] Updated");
        mvout_start = false;
    }

    current_task = DMATaskType::NONE;
}
