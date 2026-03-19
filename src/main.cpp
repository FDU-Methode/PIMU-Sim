#include <iostream>
#include <memory>

#include "../include/config.hpp"
#include "../include/func_sim.hpp"
#include "../include/perf/perf_sim.hpp"

int main(int argc, char* argv[]){
    // --- Process Configuration and Arguments ---
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << std::endl;
        return 1;
    }

    std::string cfg_file = argv[1];

    Config cfg = Config::from_file(cfg_file);

    int max_sim_cycle = cfg.max_sim_cycle;
    std::string inst_file = cfg.inst_file;
    std::string log_file = cfg.log_file;
    std::cout << "Starting to create simulator... Max sim cycle = " << max_sim_cycle << std::endl;

    // --- Initialize Simulator ---
    std::unique_ptr<SimBase> simulator;

    std::cout << "Initializing simulator in mode: " 
          << (cfg.simulator_type == SimMode::FUNCTIONAL ? "FUNCTIONAL" : "PERFORMANCE") 
          << std::endl;

    if (cfg.simulator_type == SimMode::FUNCTIONAL){
        simulator = std::make_unique<FuncSim>(cfg);
    } else{
        simulator = std::make_unique<PerfSim>(cfg);
    }

    // --- Run Simulation ---
    simulator->run();

    return 0;
}