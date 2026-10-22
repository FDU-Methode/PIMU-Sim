#include <stdexcept>
#include "../../include/perf/perf_sim.hpp"

PerfSim::PerfSim(const Config& cfg) 
: SimBase(cfg), max_sim_cycle(cfg.max_sim_cycle), cycle(0){
    logger_.log("Initializing NPU Simulator...");

    if (cfg.ddr_mode != 1) {
        throw std::invalid_argument(
            "PIMU-Sim supports AnalyticalDDR only; set ddr_mode = 1"
        );
    }

    logger_.log("Creating Analytical DDR...");
    std::cout << "Creating Analytical DDR..." << std::endl;
    ddr = std::make_unique<AnalyticalDDR>(this, cfg, logger_);

    logger_.log("Creating NPU Core...");
    std::cout << "Creating NPU Core..." << std::endl;
    core = std::make_unique<PerfNpuCore>(this, cfg, logger_);

    std::vector<double> freqs = {cfg.npu_freq, cfg.ddr_freq};

    compute_base_tick_and_multipliers(freqs, base_tick_ps, ticks_per_period);
    logger_.log("Base tick = %llu ps; core_period = %llu base ticks; ddr_period = %llu base ticks",
            (unsigned long long)base_tick_ps,
            (unsigned long long)ticks_per_period[0],
            (unsigned long long)ticks_per_period[1]);
    logger_.log("NPU Simulator Initialized");
    std::cout << "NPU Simulator Initialized" << std::endl;
}

void PerfSim::run() {
    logger_.log("================================================");
    logger_.log("Starting Simulation...\n");
    std::cout << "Starting Simulation..." << std::endl;
    sim_start_time = std::chrono::high_resolution_clock::now();
    
    bool early_termination = false;

    // 使用gcd计算出来的base tick周期，可以保证所有组件的tick周期都是base tick的整数倍
    // 从而实现较为精确的core和ddr各组件的tick()函数调用
    uint64_t max_base_ticks = max_sim_cycle * ticks_per_period[0]; // if core is index 0
    uint64_t base_tick_counter = 0;
    uint64_t core_counter = 0;
    uint64_t ddr_counter = 0;

    while (base_tick_counter < max_base_ticks){
        // advance base tick
        // call tick on modules whose counters reach their period
        if (core_counter == 0) {
            logger_.log("\n[Core Cycle: %d]", core->getCycle());
            core->tick();
            core_counter = ticks_per_period[0];
            if (core->getStats().isDone) {
                logger_.log("Core finished execution at cycle %d", core->getCycle());
                early_termination = true;
                break;
            }
        }
        if (ddr_counter == 0) {
            //logger_.log("[DDR Cycle: %d]", ddr->getCycle());
            ddr->tick();
            ddr_counter = ticks_per_period[1];
        }
        // decrement counters
        core_counter--;
        ddr_counter--;
        base_tick_counter++;
    }
    sim_end_time = std::chrono::high_resolution_clock::now();
    sim_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sim_end_time - sim_start_time).count();

    if (!early_termination) {
        logger_.log("Simulation reached maximum cycle limit %d", max_sim_cycle);
    } else {
        logger_.log("Simulation finished early at cycle %d", core->getCycle());
    }
    
    print_stats();
}

void PerfSim::print_stats() {
    double sim_time_s = sim_time_us / 1000000.0;
    int sim_inst_line_num = core->getStats().inst_line_num;
    int sim_inst_num = core->getStats().inst_num;
    //double ips = (double)(sim_inst_num * 1000000) / sim_time_us;
    double ips = sim_inst_num / (sim_time_us / 1000000.0);
    int sim_cycles = core->getCycle();
    bool early_termination = (sim_cycles < max_sim_cycle);
    
    logger_.log("================================================");
    logger_.log("Simulation Finished!");
    logger_.log("Simulation Statistics:");
    logger_.log("  Simulation time: %3f s", sim_time_s);
    logger_.log("  Maximum cycle limit: %d", max_sim_cycle);
    logger_.log("  Early termination: %s", early_termination ? "Yes" : "No");
    logger_.log("  Final state: %s", core->getStats().isDone ? "DONE" : "RUNNING");
    logger_.log("  Total cycles: %d", sim_cycles);
    logger_.log("  Total simulated instruction lines: %d", sim_inst_line_num);
    logger_.log("  Total simulated instructions: %d", sim_inst_num);
    logger_.log("  Simulated instructions per second: %3f", ips);

    std::cout << "================================================" << std::endl;
    std::cout << "Simulation Statistics:" << std::endl;
    std::cout << "  Total cycles: " <<core->getCycle() << std::endl;
    std::cout << "  Maximum cycle limit: " << max_sim_cycle << std::endl;
    std::cout << "  Simulation time: " << sim_time_s << " s" << std::endl;
    std::cout << "  Total simulated instruction lines: " << sim_inst_line_num << std::endl;
    std::cout << "  Total simulated instructions: " << sim_inst_num << std::endl;
    std::cout << "  Simulated instructions per second: " << ips << std::endl;

    if(auto* perf_core = dynamic_cast<PerfNpuCore*>(core.get())) {
        if(HardWareUtilizationTracker* util_tracker = perf_core->getUtilTracker()) {
            util_tracker->report();
        }
        if (cfg_.enable_power) {
            if (PowerAnalyzer* power_analyzer = perf_core->getPowerAnalyzer()) {
                power_analyzer->report();
            }
        }
    }

    // TODO
    // DDR带宽利用率
}
