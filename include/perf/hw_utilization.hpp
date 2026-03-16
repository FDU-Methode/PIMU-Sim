#ifndef _HW_UTILIZATION_HPP_
#define _HW_UTILIZATION_HPP_

#include <array>
#include <sstream>
#include <iomanip>
#include "../logger.hpp"
#include "../util.hpp"

class HardWareUtilizationTracker {
private:
    Logger& logger;
    static constexpr int UnitNum = 6;
    int total_cycles = 0;
    std::array<int, UnitNum> busy_cycles{};

public:
    HardWareUtilizationTracker(Logger& logger) : logger(logger) {};
    ~HardWareUtilizationTracker() = default;

    struct ModuleState {  // true - busy, false - not busy
        bool dma  = false;
        bool spm  = false;
        bool sa   = false;
        bool acc  = false;
        bool sfu  = false;
        bool gemv = false;
    };

    void reset();
    void sample(const ModuleState& state);  // Sample The Status of Each Module

    int totalCycles() { return total_cycles; };
    int busyCycles(UnitType unit) { return busy_cycles[UnitTypeToIndex(unit)]; };
    double utilization(UnitType unit);
    void report();

};

#endif