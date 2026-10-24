#ifndef __PERF_SIM_HPP__
#define __PERF_SIM_HPP__

#include "../sim_base.hpp"
#include "base_module.hpp"
#include "analytical_ddr.hpp"
#include "perf_npu_core.hpp"
#include "../run_result.hpp"

class PerfSim : public SimBase{
private:
    std::unique_ptr<PerfNpuCore> core;
    std::unique_ptr<AnalyticalDDR> ddr;

    bool run_started_ = false;
    RunResult result_;

    void rebuildDevice();

    int max_sim_cycle;

    uint64_t base_tick_ps;
    std::vector<uint64_t> ticks_per_period;
public:
    PerfSim(const Config& cfg);
    ~PerfSim() noexcept = default;

    void run() override;

    // Restore the device to its initial state using the same configuration
    void reset();

    void loadProgram(Program program);  // Outside Load Program Interface

    bool isDdrDrained() const noexcept {
        return ddr->isDrained();
    }

    RunResult getResult() const {
        return result_;
    }

    void print_stats();
    BaseModule* getCore() {return core.get();}
    BaseModule* getDDR() {return ddr.get();}
};

#endif