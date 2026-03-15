#ifndef _PERF_NPU_CORE_HPP__
#define _PERF_NPU_CORE_HPP__

#include <vector>
#include <algorithm>
#include "base_module.hpp"
#include "perf_func_unit.hpp"
#include "pipeline_controller.hpp"
#include "../inst.hpp"
#include "../regfile.hpp"

class PerfSim;
class PipelineController;

class PerfNpuCore : public BaseModule{
private:
    RegFile reg_file;
    std::unique_ptr<PipelineController> pipeline_controller;
    bool irq;
    uint64_t opcode;
    enum State {IDLE, BUSY, DONE} state;

    size_t pc;
    std::vector<Inst> instMemory; // 外部指令内存，存放机器码
    int inst_num;

public:
    PerfNpuCore(PerfSim* sim, const Config& cfg, Logger& logger);
    ~PerfNpuCore() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;

    void loadInstFromFile(const std::string& inst_file);
    void checkCompletion();

    // 执行流水线
    void WriteRegStage();
    void ExecuteStage();
    void UpdataRegStage();

//Add DMA Connect To DDR
private:
    std::queue<Message> inbox_;
    std::queue<Message> outbox_;

public:
    bool tryConsumeResp(MsgType expectedMsgType, Message& getMsg) override;

};

#endif