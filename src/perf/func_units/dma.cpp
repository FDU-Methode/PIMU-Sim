#include <algorithm>
#include <cassert>
#include <cstdint>
#include "../../../include/perf/perf_func_unit.hpp"
#include "../../../include/perf/perf_sim.hpp"

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

SizeDecode decode_mvin_sizes(int input_type_hi, int precision, int is_quant) {
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

SizeDecode decode_mvout_sizes(int output_type_lsb, int precision) {
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

DMA::DMA(PerfSim* sim, PerfNpuCore* core, Logger& logger, const Config& cfg, int queue_id) :
    PerfFunctionalUnit(sim, core, logger, cfg), queue_id(queue_id) {
    UnitName = "DMA";
}

void DMA::tick(){
    if(!isBusy()) return;

    Scratchpad* spm = core->getSpm();
    assert(spm && "DMA requires a valid Scratchpad");
    AccSram* acc_sram = core->getAccSram();
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
            if(source) {
                access_ok = is_bias ? acc_sram->WriteBias(local_addr) : acc_sram->WriteData(local_addr);
            } else {
                access_ok = spm->WriteMemory(local_addr);
            }
        } else if (mvout_start) {
            if(source) {
                access_ok = acc_sram->ReadData(local_addr);
            } else {
                access_ok = spm->ReadMemory(local_addr);
            }
        }
        
        assert(access_ok && "DMA Read or Write SPM Error");

        assert(access_ok && "DMA Read or Write SPM Error");

        // Match RTL behavior closer: if local memory is not ready, stall this beat and retry.
        if (access_ok) {
            HardWareUtilizationTracker* util_tracker = core->getUtilTracker();
            if (util_tracker) {
                const int local_payload_bytes = scale_by_log2_shift(beat_payload_bytes, local_minus_dram_log2);

                if (mvin_start) {
                    util_tracker->addDdrReadBytes(axi_beat_bytes);
                    if (source) {
                        if(is_bias) util_tracker->addAccBiasWriteBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                        else        util_tracker->addAccDataWriteBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    } else {
                        util_tracker->addSpmWriteBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
                    }
                } else if (mvout_start) {
                    if (source) {
                        util_tracker->addAccDataReadBytes(static_cast<uint64_t>(std::max(local_payload_bytes, 0)));
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

void DMA::execute() {
    logger.log("[DMA][ERROR] execute() without opcode is no longer supported");
    assert(false && "DMA::execute(uint8_t opcode) should be used");
}

void DMA::execute(uint8_t opcode){
    assert(!isBusy() && !isFinish());
    assert(state == State::IDLE);

    DMAInstQueue* dma_queue = core->getDMAInstQueue(queue_id);
    assert(dma_queue && "[DMA] DMA queue is NULL");

    const DMAInstPacket& inst_packet = dma_queue->getInstBuffer();
    const Inst& inst0 = inst_packet.inst0;
    const Inst& inst1 = inst_packet.inst1;

    state = State::BUSY;

    // Store Task Type
    if(opcode == 0x01) current_task = DMATaskType::MVIN;
    else if(opcode == 0x02) current_task = DMATaskType::MVOUT;
    else throw std::runtime_error("[DMA] Unsupported Opcode");

    resp_done = false;
    row_beat_plan.clear();
    burst_beat_plan.clear();
    current_row_idx = 0;
    current_row_beat_idx = 0;
    current_row_dram_offset_bytes = 0;
    current_burst_idx = 0;
    current_burst_beats_remaining = 0;

    mvin_start  = (current_task == DMATaskType::MVIN);
    mvout_start = (current_task == DMATaskType::MVOUT);

    Message req{};
    SizeDecode sizes{};

    if(mvin_start && !mvout_start){
        req.type = MsgType::LD_REQ;
        source = ((inst0.low >> 13) & 0x1) != 0;  // cfg_mvin_dest: 0->SPM, 1->ACC
        is_bias = ((inst0.low >> 14) & 0x1) != 0;

        req.row_num = static_cast<int>((inst0.low >> 48) & 0xFFFF);
        req.col_num = static_cast<int>((inst0.low >> 32) & 0xFFFF);
        req.dram_stride = static_cast<int>((inst1.low >> 16) & 0xFFFF);
        req.sram_stride = static_cast<int>(inst1.low & 0xFFFF);

        uint8_t data_type = static_cast<uint8_t>((inst0.low >> 8) & 0x3);
        uint8_t precision = static_cast<uint8_t>((inst0.low >> 10) & 0x3);
        uint8_t is_quant = static_cast<uint8_t>((inst0.low >> 12) & 0x1);
        int input_type_hi = (data_type >> 1) & 0x1;

        sizes = decode_mvin_sizes(input_type_hi, precision, is_quant);

        req.data_size = sizes.dram_bytes;
        req.quant = is_quant != 0;
        req.src_addr = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);
        req.dst_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
        expectedMsgType = MsgType::LD_RESP;

        logger.log("[DMA] MVIN: send LD_REQ, src=0x%x dst=0x%x rows=%d cols=%d dram_stride=%d sram_stride=%d dram_bytes=%d local_bytes=%d dest=%s quant=%d",
                   req.src_addr, req.dst_addr, req.row_num + 1, req.col_num + 1,
                   req.dram_stride, req.sram_stride, sizes.dram_bytes, sizes.local_bytes,
                   source ? "ACC" : "SPM", req.quant ? 1 : 0);

    } else if(!mvin_start && mvout_start){
        req.type = MsgType::ST_REQ;
        source = ((inst0.low >> 13) & 0x1) != 0;  // cfg_mvout_source: 0->SPM, 1->ACC
        is_bias = false;

        req.row_num = static_cast<int>((inst0.low >> 48) & 0xFFFF);
        req.col_num = static_cast<int>((inst0.low >> 32) & 0xFFFF);
        req.dram_stride = static_cast<int>((inst1.low >> 16) & 0xFFFF);
        req.sram_stride = static_cast<int>(inst1.low & 0xFFFF);

        uint8_t data_type = static_cast<uint8_t>((inst0.low >> 8) & 0x3);
        uint8_t precision = static_cast<uint8_t>((inst0.low >> 10) & 0x3);
        int output_type_lsb = data_type & 0x1;

        sizes = decode_mvout_sizes(output_type_lsb, precision);

        req.data_size = sizes.dram_bytes;
        req.quant = ((inst0.low >> 12) & 0x1) != 0;
        req.src_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
        req.dst_addr = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);
        expectedMsgType = MsgType::ST_RESP;

        logger.log("[DMA] MVOUT: send ST_REQ, src=0x%x dst=0x%x rows=%d cols=%d dram_stride=%d sram_stride=%d dram_bytes=%d local_bytes=%d source=%s quant=%d",
                   req.src_addr, req.dst_addr, req.row_num + 1, req.col_num + 1,
                   req.dram_stride, req.sram_stride, sizes.dram_bytes, sizes.local_bytes,
                   source ? "ACC" : "SPM", req.quant ? 1 : 0);

    } else{
        logger.log("[DMA][ERROR] Invalid DMATaskType for DMA");
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

    mvin_start = false;
    mvout_start = false;
    current_task = DMATaskType::NONE;
}
