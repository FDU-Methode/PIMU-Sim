#ifndef _HW_UTILIZATION_HPP__
#define _HW_UTILIZATION_HPP__

#include <array>
#include <sstream>
#include <iomanip>
#include "../logger.hpp"
#include "../util.hpp"
#include "../config.hpp"
#include "../perf_counters.hpp"

class HardWareUtilizationTracker {
private:
    Logger& logger;

    double core_freq_ghz = 1.0;
    static constexpr std::size_t UnitNum = kResourceUnitSlots;
    std::uint64_t total_cycles = 0;
    std::array<std::uint64_t, UnitNum> busy_cycles{};

    // Compute Bandwidth
    uint64_t ddr_read_bytes_total = 0;
    uint64_t ddr_write_bytes_total = 0;
    uint64_t spm_read_bytes_total = 0;
    uint64_t spm_write_bytes_total = 0;
    uint64_t acc_bias_read_bytes_total = 0;
    uint64_t acc_bias_write_bytes_total = 0;
    uint64_t acc_data_read_bytes_total = 0;
    uint64_t acc_data_write_bytes_total = 0;

    uint64_t ddr_read_bytes_current = 0;
    uint64_t ddr_write_bytes_current = 0;
    uint64_t spm_read_bytes_current = 0;
    uint64_t spm_write_bytes_current = 0;
    uint64_t acc_bias_read_bytes_current = 0;
    uint64_t acc_bias_write_bytes_current = 0;
    uint64_t acc_data_read_bytes_current = 0;
    uint64_t acc_data_write_bytes_current = 0;

    uint64_t ddr_peak_bytes_per_cycle = 0;
    uint64_t spm_peak_bytes_per_cycle = 0;
    uint64_t acc_bias_peak_bytes_per_cycle = 0;
    uint64_t acc_data_peak_bytes_per_cycle = 0;

public:
    HardWareUtilizationTracker(Logger& logger, const Config& cfg) : logger(logger), core_freq_ghz(cfg.npu_freq) {};
    ~HardWareUtilizationTracker() = default;

    struct ModuleState {  // true - busy, false - not busy
        bool dma1     = false;
        bool dma2     = false;
        bool dma3     = false;
        bool spm      = false;
        bool sa       = false;
        bool acc      = false;
        bool acc_bias = false;
        bool acc_data = false;
        bool sfu      = false;
        bool gemv     = false;
    };

    void reset();
    void sample(const ModuleState& state);  // Sample The Status of Each Module

    std::uint64_t totalCycles() const noexcept { return total_cycles; };
    std::uint64_t busyCycles(UnitType unit) const { return busy_cycles[UnitTypeToIndex(unit)]; };
    double utilization(UnitType unit);
    void report();

    // Compute Bandwidth
    void addDdrReadBytes(uint64_t bytes);
    void addDdrWriteBytes(uint64_t bytes);
    void addSpmReadBytes(uint64_t bytes);
    void addSpmWriteBytes(uint64_t bytes);
    void addAccBiasReadBytes(uint64_t bytes);
    void addAccBiasWriteBytes(uint64_t bytes);
    void addAccDataReadBytes(uint64_t bytes);
    void addAccDataWriteBytes(uint64_t bytes);

    uint64_t ddrReadBytes() { return ddr_read_bytes_total; };
    uint64_t ddrWriteBytes() { return ddr_write_bytes_total; };
    uint64_t spmReadBytes() { return spm_read_bytes_total; };
    uint64_t spmWriteBytes() { return spm_write_bytes_total; };
    uint64_t accBiasReadBytes() { return acc_bias_read_bytes_total; };
    uint64_t accBiasWriteBytes() { return acc_bias_write_bytes_total; };
    uint64_t accDataReadBytes() { return acc_data_read_bytes_total; };
    uint64_t accDataWriteBytes() { return acc_data_write_bytes_total; };

    double avgBytesPerCycle(uint64_t bytes);
    double avgBandwidthGBps(uint64_t bytes);

    ResourceCounters snapshot() const noexcept {
        ResourceCounters result;

        result.sampled_core_cycles = total_cycles;
        result.busy_cycles = busy_cycles;

        result.ddr = {
            ddr_read_bytes_total,
            ddr_write_bytes_total
        };

        result.spm = {
            spm_read_bytes_total,
            spm_write_bytes_total
        };

        result.acc_bias = {
            acc_bias_read_bytes_total,
            acc_bias_write_bytes_total
        };

        result.acc_data = {
            acc_data_read_bytes_total,
            acc_data_write_bytes_total
        };

        return result;
    }
};

#endif