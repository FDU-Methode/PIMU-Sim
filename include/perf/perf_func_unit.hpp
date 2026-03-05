#ifndef _FUNC_UNIT_HPP__
#define _FUNC_UNIT_HPP__

#include "../logger.hpp"
#include "../regfile.hpp"
#include "perf_sim.hpp"

class PerfFunctionalUnit{
protected:
    PerfSim* simulator;
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
    PerfFunctionalUnit(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
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

class DMA : public PerfFunctionalUnit{
public:
    DMA(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~DMA() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
    MsgType expectedMsgType;
};

class SPM : public PerfFunctionalUnit{
public:
    SPM(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SPM() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class SystolicArray : public PerfFunctionalUnit{
public:
    SystolicArray(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SystolicArray() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class Accumulator : public PerfFunctionalUnit{
public:
    Accumulator(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~Accumulator() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class SFU : public PerfFunctionalUnit{
public:
    SFU(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~SFU() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class Quant : public PerfFunctionalUnit{
public:
    Quant(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~Quant() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class Dequant : public PerfFunctionalUnit{
public:
    Dequant(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~Dequant() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class Im2col : public PerfFunctionalUnit{
public:
    Im2col(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~Im2col() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

class GEMV : public PerfFunctionalUnit{
public:
    GEMV(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~GEMV() = default;
    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};



class PipelineController : public PerfFunctionalUnit{
private:
    enum OpBit{
        BIT_MVIN       = 0,
        BIT_MVOUT      = 1,
        BIT_COM_SA     = 2,
        BIT_COM_MATADD = 3,
        BIT_COM_MATVEC = 4,
        BIT_COM_SFU    = 5
    };

    // 功能单元组件
    std::unique_ptr<DMA> dma;
    std::unique_ptr<SPM> spm;
    std::unique_ptr<SystolicArray> sa;
    std::unique_ptr<Accumulator> acc;
    std::unique_ptr<SFU> sfu;
    std::unique_ptr<Quant> quant;
    std::unique_ptr<Dequant> dequant;
    std::unique_ptr<Im2col> im2col;
    std::unique_ptr<GEMV> gemv;

    // 任务队列
    std::queue<PerfFunctionalUnit*> task_queue;
public:
    PipelineController(PerfSim* sim, Logger& logger, RegFile& regfile, const Config& cfg);
    ~PipelineController() = default;

    void execute() override;
    void tick() override;
protected:
    void completeInstruction() override;
};

#endif