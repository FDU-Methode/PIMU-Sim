#ifndef __PERF_SIM_HPP__
#define __PERF_SIM_HPP__

#include "sim_base.hpp"
#include "./perf/base_module.hpp"
#include "./perf/analytical_ddr.hpp"
#include "./perf/detail_ddr.hpp"
#include "./perf/perf_npu_core.hpp"

class PerfSim : public SimBase{
private:
    std::unique_ptr<BaseModule> core;
    std::unique_ptr<BaseModule> ddr;

    int max_sim_cycle;
    int cycle;
    uint64_t base_tick_ps;
    std::vector<uint64_t> ticks_per_period;
    std::chrono::high_resolution_clock::time_point sim_start_time, sim_end_time;
    double sim_time_us; // duration of the simulation (us)
public:
    PerfSim(const Config& cfg);
    ~PerfSim() noexcept = default;

    void run() override;

    void print_stats();
    BaseModule* getCore() {return core.get();}
    BaseModule* getDDR() {return ddr.get();}
};

#endif