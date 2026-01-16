#include "./perf/analytical_ddr.hpp"

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

int AnalyticalDDR::getEstimatedLatency(const Message& msg) {
    int data_size = msg.data_size;
    int total_bursts = (data_size + burst_size - 1) / burst_size;
    int bursts_per_channel = (total_bursts + cfg.ddr_channel_num - 1) / cfg.ddr_channel_num;
    // 基础延时 + 所有通道并行传输延时（每个通道传输若干个burst，每个burst延时tBURST）
    return base_latency + bursts_per_channel * cfg.tBURST;
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
                resp_msg.data_size = current_request.msg.request_data_size;
            } else if (current_request.msg.type == MsgType::ST_REQ) {
                resp_msg.type = MsgType::ST_RESP;
                resp_msg.data_size = 1; // 返回single-flit响应
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
        current_request.remaining_cycles = getEstimatedLatency(msg);
        current_request.active = true;
        logger.log("[DDR] start processing new request with %d estimated remaining DDR cycles", current_request.remaining_cycles);
    }
}