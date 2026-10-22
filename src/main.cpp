#include <exception>
#include <iostream>
#include <memory>

#include "../include/config.hpp"
#include "../include/perf/perf_sim.hpp"

int main(int argc, char* argv[]){
    // --- Process Configuration and Arguments ---
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    try {
        const Config cfg = Config::from_file(argv[1]);

        if(cfg.simulator_type != SimMode::PERFORMANCE) {
            std::cerr << "Only PERFORMANCE mode is supported.\n";
            return 1;
        }

        PerfSim simulator(cfg);
        simulator.run();

        if(!simulator.getCore()->getStats().isDone) {
            std::cerr << "Simulation did not reach DONE.\n";
            return 2;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "npu_sim: " << e.what() << '\n';
        return 1;
    }
}