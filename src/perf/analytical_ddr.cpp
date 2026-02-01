#include "../../include/perf/analytical_ddr.hpp"
#include <algorithm>

AnalyticalDDR::AnalyticalDDR(PerfSim* sim, const Config& cfg, Logger& logger) :
    BaseModule(sim, cfg, logger) {
    //ddr_frequency_mhz = cfg.ddr_freq * 1000;
    //npu_frequency_mhz = cfg.npu_freq * 1000;
    //cycle_ratio = npu_frequency_mhz / ddr_frequency_mhz;
    base_latency = cfg.tRCD + cfg.tCAS + cfg.tRP;
    BL = cfg.BL;
    ddr_bus_width = cfg.ddr_bus_width;
    request_queue_depth = cfg.ddr_request_queue_depth;
    burst_size = BL * ddr_bus_width / 8; // 8 * (64/8) = 64 bytes，一个channel一次burst传输64字节

    // 每NPU周期的DDR传输带宽
    //ddr_bandwidth_types_per_cycle = (cfg.ddr_bandwidth * 1e9) / (npu_frequency_mhz * 1e6);
    logger.log("----------------------------------------------------");
        logger.log("Initializing Analytical DDR Model");
        logger.log("  DDR Frequency: %f GHz", cfg.ddr_freq);
        //logger.log("  Core/DDR Freq Ratio: %f", cycle_ratio);
        logger.log("  Base Latency: %d in ddr cycles", base_latency);
        logger.log("  Burst Size: %d bytes", burst_size);
        logger.log("  DDR request queue depth: %d", request_queue_depth);
        logger.log("----------------------------------------------------");
}

void AnalyticalDDR::sendMsg(Message& msg) {
    auto core = sim->getCore();
    core->recvMsg(msg);
}

bool AnalyticalDDR::recvMsg(Message& msg) {
    if (request_queue.size() >= request_queue_depth) {
        logger.log("[DDR] request queue is full, backpressure asserted");
        return false; // DDR不能接收新的请求
    } else {
        request_queue.push(msg);
        logger.log("[DDR] received request from core");
        return true;
    }
}

bool AnalyticalDDR::tryConsumeResp(MsgType expectedMsg, Message& getMsg) {};

int AnalyticalDDR::getEstimatedLatency(const Message& msg, const Config& cfg) {
    int row_num = msg.row_num + 1;
    int col_num = msg.col_num + 1;
    int max_byte = cfg.axi_data_width / 8;
    int clock_count = 0;
    for(int i = 0; i < row_num; i++) {
        int addr = msg.src_addr + i * msg.dram_stride * msg.data_size;
        int offset = addr % max_byte;
        int beat_count = (offset + col_num * msg.data_size) / max_byte;
        int remain_beat_count = beat_count;
        int burst_count = 0;
        uint32_t addr_aligned = addr & ~(max_byte - 1);
        while(remain_beat_count > 0) {
            int beats_to_4k = (4096 - addr_aligned) / max_byte;
            int burst_beat = std::min(std::min(remain_beat_count, beats_to_4k), 256);
            burst_count += 1;
            remain_beat_count -= burst_beat;
            addr_aligned += burst_beat * max_byte;
        }
        clock_count += beat_count + burst_count * cfg.tBURST;
    }
    // int data_size = msg.data_size;
    // int total_bursts = (data_size + burst_size - 1) / burst_size;
    // int bursts_per_channel = (total_bursts + cfg.ddr_channel_num - 1) / cfg.ddr_channel_num;
    // // 基础延时 + 所有通道并行传输延时（每个通道传输若干个burst，每个burst延时tBURST）
    return (base_latency + clock_count) * cfg.npu_freq / cfg.ddr_freq;
}

void AnalyticalDDR::tick() {
    //logger.log("DDR tick is ticked");
    _cycle++;

    if (current_request.active) {
        current_request.remaining_cycles--;
        logger.log("[DDR] processing request, remaining %d cycles...", current_request.remaining_cycles);
        
        if (current_request.remaining_cycles == 0) {
            // 访存请求完成，发送响应消息给核心
            logger.log("[DDR] finish processing current request");
            Message resp_msg = current_request.msg;
            if (current_request.msg.type == MsgType::LD_REQ) {
                resp_msg.type = MsgType::LD_RESP;
            } else if (current_request.msg.type == MsgType::ST_REQ) {
                resp_msg.type = MsgType::ST_RESP;
            }
            sendMsg(resp_msg);
            current_request.active = false;
        }
        
        return; // 当前请求未完成，继续等待
    }

    // 如果没有正在处理的请求，尝试从请求队列中获取新请求
    if (!request_queue.empty()) {
        Message msg = request_queue.front();
        request_queue.pop();
        current_request.msg = msg;
        current_request.remaining_cycles = getEstimatedLatency(msg, cfg);
        current_request.active = true;
        logger.log("[DDR] start processing new request with %d estimated remaining DDR cycles", current_request.remaining_cycles);
    }
}