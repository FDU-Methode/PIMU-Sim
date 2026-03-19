#ifndef _PIPELINE_CONTROLLER_HPP__
#define _PIPELINE_CONTROLLER_HPP__

#include <memory>
#include <queue>
#include "hw_utilization.hpp"
#include "scratchpad.hpp"
#include "perf_func_unit.hpp"

class PerfSim;

class PipelineController{
private:
    enum OpBit{
        BIT_MVIN       = 0,
        BIT_MVOUT      = 1,
        BIT_COM_SA     = 2,
        BIT_COM_MATADD = 3,
        BIT_COM_MATVEC = 4,
        BIT_COM_SFU    = 5
    };

    // Function Units
    std::unique_ptr<DMA> dma;
    std::unique_ptr<SystolicArray> sa;
    std::unique_ptr<Accumulator> acc;
    std::unique_ptr<SFU> sfu;
    std::unique_ptr<GEMV> gemv;

    // Task Vector
    std::vector<PerfFunctionalUnit*> task_queue;

protected:
    PerfSim* simulator;
    Config cfg;
    Logger& logger;
    RegFile& regfile;

    // Util Tracker
    HardWareUtilizationTracker* util_tracker;

    // Scratchpad
    Scratchpad* spm;

    enum State{
        IDLE,
        BUSY,
        FINISH
    } state = IDLE;

    void completeInstruction();

public:
    PipelineController(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg, HardWareUtilizationTracker* util_tracker, Scratchpad* spm);
    virtual ~PipelineController() = default;

    bool isIdle() const { return state == IDLE; };
    bool isBusy() const { return state == BUSY; };
    bool isFinish() const { return state == FINISH; };
    void setIdle() { state = IDLE; };

    virtual void execute();

    virtual void tick();

    PerfFunctionalUnit* getUnit(UnitType name);
    HardWareUtilizationTracker* getUtilTracker() { return util_tracker; }
    Scratchpad* getSpm() { return spm; }
};

#endif