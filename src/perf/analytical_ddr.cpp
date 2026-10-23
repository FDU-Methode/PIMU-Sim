#include "../../include/perf/analytical_ddr.hpp"
#include "../../include/perf/perf_sim.hpp"
#include <algorithm>
#include <cstdint>

namespace {

inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace

AnalyticalDDR::AnalyticalDDR(PerfSim* sim, const Config& cfg, Logger& logger) :
    BaseModule(sim, cfg, logger) {
    base_latency = cfg.tRCD + cfg.tCAS + cfg.tRP;
    request_queue_depth = cfg.ddr_request_queue_depth;
    axi_beat_bytes = cfg.axi_data_width / 8;

    logger.log("----------------------------------------------------");
    logger.log("Initializing Analytical DDR Model");
    logger.log("  DDR Frequency: %f GHz", cfg.ddr_freq);
    logger.log("  Base Latency: %d (DDR cycles)", base_latency);
    logger.log("  AXI Beat Bytes: %d", axi_beat_bytes);
    logger.log("  DDR request queue depth: %d", static_cast<int>(request_queue_depth));
    logger.log("----------------------------------------------------");
}

void AnalyticalDDR::sendMsg(Message& msg) {
    auto core = sim->getCore();
    core->recvMsg(msg);
}

bool AnalyticalDDR::recvMsg(Message& msg) {
    if (request_queue.size() >= request_queue_depth) {
        logger.log("[DDR] request queue is full, backpressure asserted");
        return false;
    }

    request_queue.push(msg);
    logger.log("[DDR] received request from core, type=%s", MsgTypeToStr(msg.type).c_str());
    return true;
}

bool AnalyticalDDR::tryConsumeResp(MsgType, Message&) {
    return false;
}

void AnalyticalDDR::clearCurrentRequest() {
    current_request.active = false;
    current_request.msg = Message{};
    current_request.burst_beat_plan.clear();
    current_request.burst_idx = 0;
    current_request.burst_beats_remaining = 0;
    current_request.data_beats_remaining = 0;
    current_request.setup_cycles_remaining = 0;
    current_request.base_latency_remaining = 0;
    current_request.burst_gap_remaining = 0;
    current_request.pending_write_resp = 0;
}

void AnalyticalDDR::buildBurstPlan(const Message& msg, std::vector<int>& burst_beat_plan, int& total_beats) {
    burst_beat_plan.clear();
    total_beats = 0;

    const int row_count = msg.row_num + 1;
    const int col_count = msg.col_num + 1;

    const int row_payload_bytes = col_count * msg.data_size;
    const int dram_row_stride_bytes = msg.dram_stride * msg.data_size;
    const uint32_t dram_base_addr = (msg.type == MsgType::ST_REQ) ? msg.dst_addr : msg.src_addr;

    for (int r = 0; r < row_count; ++r) {
        const uint32_t row_addr = dram_base_addr + static_cast<uint32_t>(r * dram_row_stride_bytes);
        const int head_offset = static_cast<int>(row_addr % static_cast<uint32_t>(axi_beat_bytes));
        int remain_beats = ceil_div_int(head_offset + row_payload_bytes, axi_beat_bytes);

        total_beats += remain_beats;

        uint32_t burst_addr = row_addr & ~(static_cast<uint32_t>(axi_beat_bytes) - 1U);

        while (remain_beats > 0) {
            int beats_to_4k = (4096 - static_cast<int>(burst_addr & 0xFFFU)) / axi_beat_bytes;
            if (beats_to_4k <= 0) beats_to_4k = 4096 / axi_beat_bytes;

            const int burst_beats = std::min({remain_beats, beats_to_4k, kMaxAxiBurstBeats});
            burst_beat_plan.push_back(burst_beats);

            remain_beats -= burst_beats;
            burst_addr += static_cast<uint32_t>(burst_beats * axi_beat_bytes);
        }
    }
}

int AnalyticalDDR::getEstimatedLatency(const Message& msg) {
    std::vector<int> burst_plan;
    int total_beats = 0;
    buildBurstPlan(msg, burst_plan, total_beats);

    const int setup_cycles = (msg.type == MsgType::LD_REQ) ? 1 : 2;
    const int burst_gap_cycles = burst_plan.empty() ? 0 : static_cast<int>(burst_plan.size() - 1) * cfg.tBURST;
    const int write_resp_drain_cycles = (msg.type == MsgType::ST_REQ) ? static_cast<int>(burst_plan.size()) : 0;

    return base_latency + setup_cycles + total_beats + burst_gap_cycles + write_resp_drain_cycles;
}

void AnalyticalDDR::tick() {
    _cycle++;

    auto calc_remaining_cycles = [&]() -> int {
        if (!current_request.active) return 0;
        int remain = 0;
        remain += std::max(0, current_request.setup_cycles_remaining);
        remain += std::max(0, current_request.base_latency_remaining);
        remain += std::max(0, current_request.burst_gap_remaining);
        remain += std::max(0, current_request.data_beats_remaining);
        if (current_request.msg.type == MsgType::ST_REQ) {
            remain += std::max(0, current_request.pending_write_resp);
        }
        return remain;
    };

    if (current_request.active) {
        if (current_request.setup_cycles_remaining > 0) {
            --current_request.setup_cycles_remaining;
            logger.log("[DDR] processing request, remaining %d cycles... (setup=%d, base=%d, gap=%d, data=%d, bresp=%d)",
                       calc_remaining_cycles(),
                       current_request.setup_cycles_remaining,
                       current_request.base_latency_remaining,
                       current_request.burst_gap_remaining,
                       current_request.data_beats_remaining,
                       current_request.pending_write_resp);
            return;
        }

        if (current_request.base_latency_remaining > 0) {
            --current_request.base_latency_remaining;
            logger.log("[DDR] processing request, remaining %d cycles... (setup=%d, base=%d, gap=%d, data=%d, bresp=%d)",
                       calc_remaining_cycles(),
                       current_request.setup_cycles_remaining,
                       current_request.base_latency_remaining,
                       current_request.burst_gap_remaining,
                       current_request.data_beats_remaining,
                       current_request.pending_write_resp);
            return;
        }

        if (current_request.burst_gap_remaining > 0) {
            --current_request.burst_gap_remaining;
            logger.log("[DDR] processing request, remaining %d cycles... (setup=%d, base=%d, gap=%d, data=%d, bresp=%d)",
                       calc_remaining_cycles(),
                       current_request.setup_cycles_remaining,
                       current_request.base_latency_remaining,
                       current_request.burst_gap_remaining,
                       current_request.data_beats_remaining,
                       current_request.pending_write_resp);
            return;
        }

        if (current_request.data_beats_remaining > 0) {
            --current_request.data_beats_remaining;
            --current_request.burst_beats_remaining;

            if (current_request.burst_beats_remaining == 0) {
                if (current_request.msg.type == MsgType::ST_REQ) {
                    ++current_request.pending_write_resp; // one bresp token per finished burst
                }

                ++current_request.burst_idx;
                if (current_request.burst_idx < current_request.burst_beat_plan.size()) {
                    current_request.burst_beats_remaining = current_request.burst_beat_plan[current_request.burst_idx];
                    current_request.burst_gap_remaining = cfg.tBURST;
                }
            }
        }

        // Write-response drain (AW/W done but still waiting B responses), RTL-like approximation.
        if (current_request.msg.type == MsgType::ST_REQ && current_request.pending_write_resp > 0) {
            --current_request.pending_write_resp;
        }

        const bool data_done = (current_request.data_beats_remaining == 0);
        const bool write_rsp_done =
            (current_request.msg.type != MsgType::ST_REQ) || (current_request.pending_write_resp == 0);

        if (data_done && write_rsp_done) {
            Message resp_msg = current_request.msg;
            resp_msg.type = (current_request.msg.type == MsgType::LD_REQ) ? MsgType::LD_RESP : MsgType::ST_RESP;

            logger.log("[DDR] finish request, send %s", MsgTypeToStr(resp_msg.type).c_str());
            sendMsg(resp_msg);
            clearCurrentRequest();
        } else {
            logger.log("[DDR] processing request, remaining %d cycles... (setup=%d, base=%d, gap=%d, data=%d, bresp=%d)",
                       calc_remaining_cycles(),
                       current_request.setup_cycles_remaining,
                       current_request.base_latency_remaining,
                       current_request.burst_gap_remaining,
                       current_request.data_beats_remaining,
                       current_request.pending_write_resp);
        }

        return;
    }

    if (!request_queue.empty()) {
        current_request.msg = request_queue.front();
        request_queue.pop();

        int total_beats = 0;
        buildBurstPlan(current_request.msg, current_request.burst_beat_plan, total_beats);

        current_request.active = true;
        current_request.burst_idx = 0;
        current_request.data_beats_remaining = total_beats;
        current_request.setup_cycles_remaining = (current_request.msg.type == MsgType::LD_REQ) ? 1 : 2;
        current_request.base_latency_remaining = base_latency;
        current_request.burst_gap_remaining = 0;
        current_request.pending_write_resp = 0;

        if (!current_request.burst_beat_plan.empty()) {
            current_request.burst_beats_remaining = current_request.burst_beat_plan[0];
        } else {
            current_request.burst_beats_remaining = 0;
        }

        logger.log("[DDR] start request type=%s rows=%d cols=%d beats=%d bursts=%d estimated_latency=%d",
                   MsgTypeToStr(current_request.msg.type).c_str(),
                   current_request.msg.row_num + 1,
                   current_request.msg.col_num + 1,
                   total_beats,
                   static_cast<int>(current_request.burst_beat_plan.size()),
                   getEstimatedLatency(current_request.msg));
        
        logger.log("[DDR] start processing new request with %d estimated remaining DDR cycles", getEstimatedLatency(current_request.msg));
    }
}
