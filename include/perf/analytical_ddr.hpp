#ifndef _ANALYTICAL_DDR_HPP__
#define _ANALYTICAL_DDR_HPP__

#include <queue>
#include "../util.hpp"
#include "base_module.hpp"
#include "perf_sim.hpp"
#include "../config.hpp"
#include "../logger.hpp"

class PerfSim; // forward declaration

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
    int BL; // DDR BURST长度（一次burst取data words的数量，对于DDR通常为8)
    int ddr_bus_width; // DDR总线宽度（等于data word宽度，通常为64bit）
    int burst_size; // 每次BURST传输的字节数（根据BL和DDR总线宽度计算）

    // 接收消息队列
    std::queue<Message> request_queue;
    
    struct CurrentRequest {
        bool active = false;
        Message msg;
        int remaining_cycles = 0; // 剩余DDR访存周期数
        //int start_cycle = 0;      // 开始处理的DDR时刻
    } current_request;
    

public:
    AnalyticalDDR(PerfSim* sim, const Config& cfg, Logger& logger);
    ~AnalyticalDDR() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    bool tryConsumeResp(MsgType expectedMsg, Message& getMsg) override;

private:
    int getEstimatedLatency(const Message& msg, const Config& cfg);
};

#endif