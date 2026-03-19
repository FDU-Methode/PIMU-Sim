#ifndef _FUNC_UNIT_HPP__
#define _FUNC_UNIT_HPP__

#include "../util.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"

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
    bool mvin_start = false;
    bool mvout_start = false;
    MsgType expectedMsgType;

    bool resp_done = false;

    uint32_t spm_base_addr = 0;
    int spm_row_stride_bytes = 0;
    int axi_beat_bytes = 0;

    std::vector<int> row_beat_plan;
    std::vector<int> burst_beat_plan;

    size_t current_row_idx = 0;
    int current_row_beat_idx = 0;
    size_t current_burst_idx = 0;
    int current_burst_beats_remaining = 0;

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

    uint32_t spm_rd1_addr = 0;   // weight port
    uint32_t spm_rd2_addr = 0;   // activation / ifm port
    int spm_rd1_stride_bytes = 0;
    int spm_rd2_stride_bytes = 0;
    int spm_rd1_remaining = 0;
    int spm_rd2_remaining = 0;

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
    bool matadd_start = false;
    bool sa_start = false;
    bool acc_dest = false;
    int total_cycles = 0;
    uint32_t sa_output_base_addr = 0;
    uint32_t matadd_output_base_addr = 0;
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
    int sfu_cycle = 0;
    uint8_t sfu_optype = 0;

    uint32_t spm_input_base_addr = 0;
    uint32_t spm_output_base_addr = 0;

    int spm_expected_reads = 0;
    int spm_expected_writes = 0;
    int spm_issued_reads = 0;
    int spm_issued_writes = 0;

    int spm_setup_delay = 0;
    int spm_batch_num = 0;
    int spm_row_num = 0;
    int spm_core_delay = 0;
    int spm_done_delay = 0;

    int spm_tile_num = 0;
    int spm_tile_stage_cycles = 0;
    int spm_gelu_vector_period = 0;

    int spm_elem_bytes = 1;
    int spm_vector_stride_bytes = 0;
    int spm_row_stride_bytes = 0;
    int spm_gelu_group_stride_bytes = 0;

    int spm_transpose_tile_col_num = 0;
    int spm_transpose_tile_row_num = 0;
    int spm_transpose_output_row_stride_bytes = 0;

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