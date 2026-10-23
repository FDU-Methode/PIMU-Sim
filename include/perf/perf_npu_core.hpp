#ifndef _PERF_NPU_CORE_HPP__
#define _PERF_NPU_CORE_HPP__

#include <vector>
#include <algorithm>
#include "base_module.hpp"
#include "perf_func_unit.hpp"
#include "scratchpad.hpp"
#include "hw_utilization.hpp"
#include "power_analyzer.hpp"
#include "inst_cache.hpp"
#include "inst_queue.hpp"
#include "decoder.hpp"
#include "event_sync_unit.hpp"
#include "../program.hpp"
#include "../regfile.hpp"

class PerfSim;
class PipelineController;

class PerfNpuCore : public BaseModule{
private:
    RegFile reg_file;

    // Instruction Unit
    std::unique_ptr<InstCache> inst_cache;
    std::unique_ptr<EventSyncUnit> event_sync_unit;

    std::unique_ptr<DMAInstQueue> dma_1_inst_queue;
    std::unique_ptr<DMAInstQueue> dma_2_inst_queue;
    std::unique_ptr<DMAInstQueue> dma_3_inst_queue;
    std::unique_ptr<SAInstQueue> sa_inst_queue;
    std::unique_ptr<SFUInstQueue> sfu_inst_queue;

    std::unique_ptr<Decoder> decoder;

    // Function Unit
    std::unique_ptr<DMA> dma_1;
    std::unique_ptr<DMA> dma_2;
    std::unique_ptr<DMA> dma_3;
    std::unique_ptr<SystolicArray> sa;
    std::unique_ptr<Accumulator> acc;
    std::unique_ptr<SFU> sfu;
    std::unique_ptr<GEMV> gemv;

    std::unique_ptr<Scratchpad> spm;

    // Utilization
    std::unique_ptr<HardWareUtilizationTracker> util_tracker;

    // Power
    std::unique_ptr<PowerAnalyzer> power_analyzer;

    bool irq;
    uint64_t opcode;
    enum State {IDLE, BUSY, DONE} state;

    Program program_;

    std::size_t next_word_ = 0;

    int fetched_inst_num = 0;
    int simulated_inst_num = 0;

    bool program_loaded_ = false;
    bool program_exhausted_ = true;

public:
    PerfNpuCore(PerfSim* sim, const Config& cfg, Logger& logger);
    ~PerfNpuCore() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;

    void loadProgram(Program program);
    bool hasProgram() const noexcept {
        return program_loaded_;
    }
    bool loadWordToCache();   // Read From Inst File
    void handleInterrupts();  // Handle Irq
    void issueInstQueues();   // Issue Inst From InstQueue
    void retireInstQueues();  // Retire Inst To InstQueue
    bool isBusy();            // Check If Core is Busy
    void checkCompletion();

    void addInstNum();

    RegFile& getRegFile() { return reg_file; }
    DMAInstQueue* getDMAInstQueue(int queue_id) {
        switch(queue_id) {
            case 1: return dma_1_inst_queue.get();
            case 2: return dma_2_inst_queue.get();
            case 3: return dma_3_inst_queue.get();
            default: return nullptr;
        }
    }
    SAInstQueue* getSAInstQueue() { return sa_inst_queue.get(); }
    SFUInstQueue* getSFUInstQueue() { return sfu_inst_queue.get(); }
    PerfFunctionalUnit* getUnit(UnitType name);
    HardWareUtilizationTracker* getUtilTracker() { return util_tracker.get(); };
    PowerAnalyzer* getPowerAnalyzer() { return power_analyzer.get(); }
    Scratchpad* getSpm() { return spm.get(); }
    AccSram* getAccSram() { return acc->getScratchpad(); }

//Add DMA Connect To DDR
private:
    std::queue<Message> inbox_;
    std::queue<Message> outbox_;

public:
    bool tryConsumeResp(MsgType expectedMsgType, Message& getMsg) override;

};

#endif