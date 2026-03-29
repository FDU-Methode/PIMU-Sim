#ifndef _HW_UTILIZATION_HPP__
#define _HW_UTILIZATION_HPP__

#include <array>
#include <sstream>
#include <iomanip>
#include "../logger.hpp"
#include "../util.hpp"
#include "../config.hpp"

class HardWareUtilizationTracker {
private:
    Logger& logger;

    double core_freq_ghz = 1.0;
    static constexpr int UnitNum = 7;
    int total_cycles = 0;
    std::array<int, UnitNum> busy_cycles{};

    // Compute Bandwidth
    uint64_t ddr_read_bytes_total = 0;
    uint64_t ddr_write_bytes_total = 0;
    uint64_t spm_read_bytes_total = 0;
    uint64_t spm_write_bytes_total = 0;
    uint64_t acc_read_bytes_total = 0;
    uint64_t acc_write_bytes_total = 0;

    uint64_t ddr_read_bytes_current = 0;
    uint64_t ddr_write_bytes_current = 0;
    uint64_t spm_read_bytes_current = 0;
    uint64_t spm_write_bytes_current = 0;
    uint64_t acc_read_bytes_current = 0;
    uint64_t acc_write_bytes_current = 0;

    uint64_t ddr_peak_bytes_per_cycle = 0;
    uint64_t spm_peak_bytes_per_cycle = 0;
    uint64_t acc_peak_bytes_per_cycle = 0;

public:
    HardWareUtilizationTracker(Logger& logger, const Config& cfg) : logger(logger), core_freq_ghz(cfg.npu_freq) {};
    ~HardWareUtilizationTracker() = default;

    struct ModuleState {  // true - busy, false - not busy
        bool dma      = false;
        bool spm      = false;
        bool sa       = false;
        bool acc      = false;
        bool acc_sram = false;
        bool sfu      = false;
        bool gemv     = false;
    };

    void reset();
    void sample(const ModuleState& state);  // Sample The Status of Each Module

    int totalCycles() { return total_cycles; };
    int busyCycles(UnitType unit) { return busy_cycles[UnitTypeToIndex(unit)]; };
    double utilization(UnitType unit);
    void report();

    // Compute Bandwidth
    void addDdrReadBytes(uint64_t bytes);
    void addDdrWriteBytes(uint64_t bytes);
    void addSpmReadBytes(uint64_t bytes);
    void addSpmWriteBytes(uint64_t bytes);
    void addAccReadBytes(uint64_t bytes);
    void addAccWriteBytes(uint64_t bytes);

    uint64_t ddrReadBytes() { return ddr_read_bytes_total; };
    uint64_t ddrWriteBytes() { return ddr_write_bytes_total; };
    uint64_t spmReadBytes() { return spm_read_bytes_total; };
    uint64_t spmWriteBytes() { return spm_write_bytes_total; };
    uint64_t accReadBytes() { return acc_read_bytes_total; };
    uint64_t accWriteBytes() { return acc_write_bytes_total; };

    double avgBytesPerCycle(uint64_t bytes);
    double avgBandwidthGBps(uint64_t bytes);
};

#endif