#ifndef _DETAIL_DDR_HPP__
#define _DETAIL_DDR_HPP__

#include <queue>
#include "util.hpp"
#include "./perf/base_module.hpp"
#include "./perf/perf_sim.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "../../external/DRAMsim3/src/dramsim3.h"

using namespace dramsim3;

class PerfSim;

class DetailDDR : public BaseModule {
private:
    // 接收消息队列
    size_t request_queue_depth;
    std::queue<Message> request_queue;

    // dramsim3
    std::unique_ptr<MemorySystem> m_dramsim;
    int dram_unit = 64; // dram req unit size in bytes

    // dramsim3 callback
    void readComplete(uint64_t addr);
    void writeComplete(uint64_t addr);

    // 当前正在处理的请求
    struct CurrentRequest {
        bool active = false;
        Message msg;
        std::vector<bool> completed_trans;  // 跟踪每个事务是否完成
        int total_trans = 0;                // 总事务数
        int trans_index = 0;                // 当前DRAM事务索引
        int completed_count = 0;            // 已完成的事务数
    } current_request;

    void clearCurrentRequest();

public:
    DetailDDR(PerfSim* sim, const Config& cfg, Logger& logger);
    ~DetailDDR() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    bool tryConsumeResp(MsgType expectedMsg, Message& getMsg) override;
};

#endif // __DETAIL_DDR_HPP__