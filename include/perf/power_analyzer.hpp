#ifndef _POWER_ANALYZER_HPP__
#define _POWER_ANALYZER_HPP__

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "../config.hpp"
#include "../logger.hpp"

enum class DmaPowerMode {
    Idle = 0,
    BusyMvin,
    BusyMvout
};

enum class AccPowerMode {
    Idle = 0,
    BusyAccadd,
    BusyBiasLoad,
    BusyDmaRd,
    BusyDmaWr,
    BusyMatadd
};

enum class SfuPowerMode {
    Idle = 0,
    BusySoftmax,
    BusyLayernorm,
    BusyGelu,
    BusyTranspose,
    BusyPoolmax,
    BusyResample
};

enum class SaPowerMode {
    Idle = 0,
    BusyGemm,
    BusyConv
};

class PowerAnalyzer {
private:
    Logger& logger;
    Config cfg;
    bool enabled_ = false;
    double cycle_ns_ = 0.0;
    uint64_t total_cycles_ = 0;

    double dma_energy_pj_ = 0.0;
    double acc_energy_pj_ = 0.0;
    double sfu_energy_pj_ = 0.0;
    double sa_energy_pj_ = 0.0;
    double total_energy_pj_ = 0.0;

    std::array<uint64_t, 3> dma_mode_cycles_{};
    std::array<uint64_t, 6> acc_mode_cycles_{};
    std::array<uint64_t, 7> sfu_mode_cycles_{};
    std::array<uint64_t, 3> sa_mode_cycles_{};

    double dma_power_mw(DmaPowerMode mode) const;
    double acc_power_mw(AccPowerMode mode) const;
    double sfu_power_mw(SfuPowerMode mode) const;
    double sa_power_mw(SaPowerMode mode) const;

    static const char* dma_mode_str(DmaPowerMode mode);
    static const char* acc_mode_str(AccPowerMode mode);
    static const char* sfu_mode_str(SfuPowerMode mode);
    static const char* sa_mode_str(SaPowerMode mode);

public:
    PowerAnalyzer(Logger& logger, const Config& cfg);

    bool enabled() const { return enabled_; }

    void sample(DmaPowerMode dma_mode,
                AccPowerMode acc_mode,
                SfuPowerMode sfu_mode,
                SaPowerMode sa_mode);

    void report();
};

#endif
