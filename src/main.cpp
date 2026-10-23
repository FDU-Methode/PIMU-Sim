#include <exception>
#include <iostream>

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

        PerfSim simulator(cfg);

        std::cout << "Simulator initialized; no Program executed.\n"
                  << "Submit Program through loadProgram(),"
                  << "then call run()." << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "npu_sim: " << e.what() << '\n';
        return 1;
    }
}