#ifndef _FUNC_UNIT_HPP__
#define _FUNC_UNIT_HPP__

#include <memory>
#include "../util.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"
#include "scratchpad.hpp"

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

    bool source = false;  // 0/1 for SPM/ACC
    bool is_bias = false;

    bool resp_done = false;

    int axi_beat_bytes = 0;
    int local_minus_dram_log2 = 0;

    // DRAM side row geometry (byte domain)
    uint32_t dram_base_addr = 0;
    int dram_row_stride_bytes = 0;
    int dram_row_payload_bytes = 0;

    // Local memory side geometry (SPM/ACC byte domain)
    uint32_t spm_base_addr = 0;
    int spm_row_stride_bytes = 0;

    // Transfer plans
    std::vector<int> row_beat_plan;    // beats per row
    std::vector<int> burst_beat_plan;  // AXI burst beats (split by 4KB and <=256 beats)

    // Runtime cursors
    size_t current_row_idx = 0;
    int current_row_beat_idx = 0;
    int current_row_dram_offset_bytes = 0;

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

    Scratchpad* getScratchpad() { return acc_sram.get(); }

protected:
    // ACC SRAM
    std::unique_ptr<Scratchpad> acc_sram;

    bool matadd_start = false;
    bool sa_start = false;
    uint8_t optype = 0;
    bool is_bias = false;
    bool acc_dest = false;
    int tail_cycles = 0;
    int total_cycles = 0;

    uint32_t biaspsum_base_addr = 0;
    uint32_t sa_output_base_addr = 0;
    uint32_t matadd_a_base_addr = 0;
    uint32_t matadd_b_base_addr = 0;
    uint32_t matadd_output_base_addr = 0;
    int sa_input_b_col_num = 0;
    int biaspsum_stride = 0;
    int sa_output_stride = 0;

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