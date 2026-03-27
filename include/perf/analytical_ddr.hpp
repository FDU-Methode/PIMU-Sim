#ifndef _ANALYTICAL_DDR_HPP__
#define _ANALYTICAL_DDR_HPP__

#include <cstddef>
#include <queue>
#include "../util.hpp"
#include "base_module.hpp"
#include "perf_sim.hpp"
#include "../config.hpp"
#include "../logger.hpp"

class PerfSim;

class AnalyticalDDR : public BaseModule {
private:
    // DDR配置参数
    //double ddr_frequency_mhz; // DDR频率（MHz）
    //double npu_frequency_mhz; // NPU频率（MHz）
    //double cycle_ratio; // NPU周期与DDR周期比例
    //double ddr_bandwidth_types_per_cycle; // 每NPU周期的带宽（字节） 
    size_t request_queue_depth; // DDR请求队列深度

    // DDR时序参数 （单位：DDR cycle）
    int base_latency; // 基础延迟，包括CAS、RCD、RP延迟
    int axi_beat_bytes;

    static constexpr int kMaxAxiBurstBeats = 256;

    // 接收消息队列
    std::queue<Message> request_queue;

    struct CurrentRequest {
        bool active = false;
        Message msg{};

        std::vector<int> burst_beat_plan;
        size_t burst_idx = 0;

        int burst_beats_remaining = 0;
        int data_beats_remaining = 0;

        int setup_cycles_remaining = 0;     // RTL-like req->state start delay: mvin=1, mvout=2
        int base_latency_remaining = 0;     // tRCD+tCAS+tRP
        int burst_gap_remaining = 0;        // inter-burst gap approximation

        int pending_write_resp = 0;         // bresp drain count
    } current_request;

public:
    AnalyticalDDR(PerfSim* sim, const Config& cfg, Logger& logger);
    ~AnalyticalDDR() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    bool tryConsumeResp(MsgType expectedMsg, Message& getMsg) override;

private:
    void clearCurrentRequest();
    void buildBurstPlan(const Message& msg, std::vector<int>& burst_beat_plan, int& total_beats);
    int getEstimatedLatency(const Message& msg);
};

#endif