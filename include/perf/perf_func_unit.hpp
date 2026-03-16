#ifndef _FUNC_UNIT_HPP__
#define _FUNC_UNIT_HPP__

#include "../util.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"
// #include "perf_sim.hpp"

class PerfSim;
class PipelineController;

class PerfFunctionalUnit{
protected:
    PerfSim* simulator;
    PipelineController* controller;
    Config cfg;
    Logger& logger;
    RegFile& regfile;
    int remaining_cycles = 0;
    std::string UnitName;

    enum State{
        IDLE,
        BUSY,
        FINISH
    } state = IDLE;

    virtual void completeInstruction();

public:
    PerfFunctionalUnit(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    virtual ~PerfFunctionalUnit() = default;

    // 状态查询
    std::string getName() const { return UnitName; };
    bool isIdle() const { return state == IDLE; };
    bool isBusy() const { return state == BUSY; };
    bool isFinish() const { return state == FINISH; };
    void setIdle() { state = IDLE; };
    int getRemainingCycles() const { return remaining_cycles; };
    // 执行指令
    virtual void execute();
    // 周期推进
    virtual void tick();
};

//
// DMA
//

class DMA : public PerfFunctionalUnit{
public:
    DMA(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~DMA() = default;

    void execute() override;
    void tick() override;

protected:
    void completeInstruction() override;
    MsgType expectedMsgType;
};

//
// SPM
//

class SPM : public PerfFunctionalUnit{
public:
    SPM(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SPM() = default;

    void execute() override;
    void tick() override;

protected:
    void completeInstruction() override;
};

//
// Systolic Array
//

class SystolicArray : public PerfFunctionalUnit{
public:
    SystolicArray(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SystolicArray() = default;

    void execute() override;
    void tick() override;

protected:
    bool acc_started = false;
    int remaining_start_acc_cycles = -1;
    void completeInstruction() override;
};

//
// Accumulator
//

class Accumulator : public PerfFunctionalUnit{
public:
    Accumulator(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~Accumulator() = default;

    void execute() override;
    void tick() override;

protected:
    void completeInstruction() override;
};

//
// SFU
//

class SFU : public PerfFunctionalUnit{
public:
    SFU(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SFU() = default;

    void execute() override;
    void tick() override;

protected:
    void completeInstruction() override;
};

//
// GEMV
//

class GEMV : public PerfFunctionalUnit{
public:
    GEMV(PerfSim* sim, PipelineController* controller, Logger& logger, RegFile& regfile, const Config& cfg);
    ~GEMV() = default;

    void execute() override;
    void tick() override;

protected:
    void completeInstruction() override;
};

#endif