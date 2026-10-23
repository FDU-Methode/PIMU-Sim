#include <stdexcept>
#include <utility>
#include "../../include/perf/perf_sim.hpp"

PerfSim::PerfSim(const Config& cfg) 
: SimBase(cfg), max_sim_cycle(cfg.max_sim_cycle){
    if(cfg.simulator_type != SimMode::PERFORMANCE || cfg.max_sim_cycle <= 0) {
        throw std::invalid_argument("PERFORMANCE mode and positive max_sim_cycle are required");
    }

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
    for (const auto ticks : ticks_per_period) {
        if (ticks == 0) {
            throw std::invalid_argument(
                "Clock configuration produces a zero tick period"
            );
        }
    }
    logger_.log("Base tick = %llu ps; core_period = %llu base ticks; ddr_period = %llu base ticks",
            (unsigned long long)base_tick_ps,
            (unsigned long long)ticks_per_period[0],
            (unsigned long long)ticks_per_period[1]);
    logger_.log("NPU Simulator Initialized");
    std::cout << "NPU Simulator Initialized" << std::endl;
}

void PerfSim::loadProgram(Program program) {
    if (run_started_) {
        throw std::logic_error(
            "Create a new PerfSim for another Program"
        );
    }

    core->loadProgram(std::move(program));

    result_ = RunResult{};
    result_.status = RunStatus::READY;
}

void PerfSim::run() {
    if(run_started_) {
        throw std::logic_error("This PerfSim supports one run only");
    }

    if(!core->hasProgram()) {
        throw std::logic_error("Call loadProgram() before run()");
    }

    run_started_ = true;

    result_ = RunResult{};
    result_.status = RunStatus::RUNNING;

    const auto start = std::chrono::steady_clock::now();

    try{
        const auto max_base_ticks =
            static_cast<std::uint64_t>(max_sim_cycle) *
            ticks_per_period[0];

        std::uint64_t base_tick_counter = 0;
        std::uint64_t core_counter = 0;
        std::uint64_t ddr_counter = 0;

        while (base_tick_counter < max_base_ticks) {
            if (core_counter == 0) {
                core->tick();
                core_counter = ticks_per_period[0];

                if (core->getStats().isDone) {
                    result_.status = RunStatus::DONE;
                    break;
                }
            }

            if (ddr_counter == 0) {
                ddr->tick();
                ddr_counter = ticks_per_period[1];
            }

            --core_counter;
            --ddr_counter;
            ++base_tick_counter;
        }

        if (result_.status == RunStatus::RUNNING) {
            result_.status = RunStatus::TIMEOUT;
        }
    } catch (const std::exception& e) {
        result_.status = RunStatus::BACKEND_ERROR;
        result_.error = e.what();
    } catch (...) {
        result_.status = RunStatus::BACKEND_ERROR;
        result_.error = "Unknown exception during simulation";
    }

    const auto end = std::chrono::steady_clock::now();

    result_.host_elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();

    result_.core_cycles = static_cast<std::uint64_t>(core->getCycle());

    result_.ddr_cycles = static_cast<std::uint64_t>(ddr->getCycle());

    result_.fetched_words = core->fetchedWordCount();

    result_.assembled_packets = core->assembledPacketCount();

    print_stats();
}

void PerfSim::print_stats() {
    const auto& r = result_;

    logger_.log(
        "Final status: %s",
        toString(r.status)
    );

    std::cout
        << "Simulation result:\n"
        << "  Status: "
        << toString(r.status) << '\n'
        << "  Core cycles: "
        << r.core_cycles << '\n'
        << "  DDR cycles: "
        << r.ddr_cycles << '\n'
        << "  Fetched words: "
        << r.fetched_words << '\n'
        << "  Assembled packets: "
        << r.assembled_packets << '\n'
        << "  Host elapsed (us): "
        << r.host_elapsed_us << '\n';

    if (!r.error.empty()) {
        std::cout
            << "  Error: " << r.error << '\n';

        logger_.log(
            "Backend error: %s",
            r.error.c_str()
        );
    }

    if (auto* tracker = core->getUtilTracker()) {
        tracker->report();
    }

    if (cfg_.enable_power) {
        if (auto* power = core->getPowerAnalyzer()) {
            power->report();
        }
    }
}
