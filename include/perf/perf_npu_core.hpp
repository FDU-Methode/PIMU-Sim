#ifndef _PERF_NPU_CORE_HPP__
#define _PERF_NPU_CORE_HPP__

#include <vector>
#include <algorithm>
#include <fstream>
#include "base_module.hpp"
#include "perf_func_unit.hpp"
#include "pipeline_controller.hpp"
#include "scratchpad.hpp"
#include "hw_utilization.hpp"
#include "inst_cache.hpp"
#include "../regfile.hpp"

class PerfSim;
class PipelineController;

class PerfNpuCore : public BaseModule{
private:
    RegFile reg_file;
    std::unique_ptr<InstCache> inst_cache;
    std::unique_ptr<PipelineController> pipeline_controller;
    std::unique_ptr<Scratchpad> spm;
    std::unique_ptr<HardWareUtilizationTracker> util_tracker;

    bool irq;
    uint64_t opcode;
    enum State {IDLE, BUSY, DONE} state;

    size_t pc;
    int fetched_inst_num;
    int inst_file_line_num;
    bool inst_file_done = false;
    std::ifstream inst_ifs;

public:
    PerfNpuCore(PerfSim* sim, const Config& cfg, Logger& logger);
    ~PerfNpuCore() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;

    void openInstFile(const std::string& inst_file);  // Prepare to Read Inst File
    bool loadInstToCache();  // Read From Inst File
    void checkCompletion();

    // 执行流水线
    void WriteRegStage();
    void ExecuteStage();
    void UpdataRegStage();

    HardWareUtilizationTracker* getUtilTracker();

//Add DMA Connect To DDR
private:
    std::queue<Message> inbox_;
    std::queue<Message> outbox_;

public:
    bool tryConsumeResp(MsgType expectedMsgType, Message& getMsg) override;

};

#endif