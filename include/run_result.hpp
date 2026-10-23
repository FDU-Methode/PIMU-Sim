#ifndef RUN_RESULT_HPP
#define RUN_RESULT_HPP

#include <cstdint>
#include <string>

enum class RunStatus {
    IDLE,
    READY,
    RUNNING,
    DONE,
    TIMEOUT,
    BACKEND_ERROR
};

inline const char* toString(RunStatus status) noexcept {
    switch (status) {
        case RunStatus::IDLE:
            return "IDLE";
        case RunStatus::READY:
            return "READY";
        case RunStatus::RUNNING:
            return "RUNNING";
        case RunStatus::DONE:
            return "DONE";
        case RunStatus::TIMEOUT:
            return "TIMEOUT";
        case RunStatus::BACKEND_ERROR:
            return "BACKEND_ERROR";
    }

    return "UNKNOWN";
}

struct RunResult {
    RunStatus status = RunStatus::IDLE;

    std::uint64_t core_cycles = 0;
    std::uint64_t ddr_cycles = 0;

    std::uint64_t fetched_words = 0;
    std::uint64_t assembled_packets = 0;

    double host_elapsed_us = 0.0;

    std::string error;
};

#endif