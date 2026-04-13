#ifndef __PERF_SIM_HPP__
#define __PERF_SIM_HPP__

#include "../sim_base.hpp"
#include "base_module.hpp"
#include "analytical_ddr.hpp"
#include "detail_ddr.hpp"
#include "perf_npu_core.hpp"

class PerfSim : public SimBase{
private:
    std::unique_ptr<BaseModule> core;
    std::unique_ptr<BaseModule> ddr;

    int max_sim_cycle;
    int cycle;
    uint64_t core_phase_counter = 0;
    uint64_t ddr_phase_counter = 0;
    uint64_t base_tick_ps;
    std::vector<uint64_t> ticks_per_period;
    std::chrono::high_resolution_clock::time_point sim_start_time, sim_end_time;
    double sim_time_us; // duration of the simulation (us)
public:
    PerfSim(const Config& cfg);
    ~PerfSim() noexcept = default;

    void run() override;
    void stepOneCoreCycle();
    // Compiler Interface
    void beginExternalExecution();
    void endExternalExecution();

    void print_stats();
    BaseModule* getCore() {return core.get();}
    BaseModule* getDDR() {return ddr.get();}
    PerfNpuCore* getPerfCore() {return dynamic_cast<PerfNpuCore*>(core.get());}
};

#endif