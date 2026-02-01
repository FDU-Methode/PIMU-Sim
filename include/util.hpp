#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>

// 统计模块
struct Stats {
    int cycle = 0;
    bool isDone = false;
    int inst_num = 0;
};

enum class MsgType {
    LD_REQ, LD_RESP, ST_REQ, ST_RESP
};

struct Message {
    MsgType type;
    int row_num = 0;
    int col_num = 0;
    int dram_stride = 0;
    int sram_stride = 0;
    int data_size = 0;
    bool quant = 0;
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
    // TODO: add more fields
};

/*
 Utilities for precise clock ratio calculations.
 Approach:
 - Represent periods in integer picoseconds (ps) to avoid floating error.
 - period_ps = round(1000.0 / freq_ghz)  (since period_ps = 1e12 / (freq_ghz * 1e9) = 1000 / freq_ghz)
 - base_tick_ps = gcd(period_ps_i over all i)
 - ticks_per_base[i] = period_ps_i / base_tick_ps
*/
// Convert frequency in GHz to period in picoseconds (integer).
// Throws std::invalid_argument if freq_ghz <= 0.
inline std::uint64_t period_ps_from_ghz(double freq_ghz) {
    if (!(freq_ghz > 0.0)) throw std::invalid_argument("frequency must be > 0");
    // period_ps = 1000 / freq_ghz. Use rounding to nearest integer.
    double period_ps_d = 1000.0 / freq_ghz;
    // Guard: if extremely small or large, ensure within uint64_t
    if (period_ps_d < 1.0) period_ps_d = 1.0;
    if (period_ps_d > static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("period too large");
    return static_cast<std::uint64_t>(period_ps_d + 0.5);
}

// Compute gcd for uint64_t (std::gcd works with unsigned long long via <numeric> in C++17)
inline std::uint64_t gcd_u64(std::uint64_t a, std::uint64_t b) {
    while (b != 0) {
        std::uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Compute gcd of a vector
inline std::uint64_t gcd_vector(const std::vector<std::uint64_t>& v) {
    if (v.empty()) throw std::invalid_argument("empty vector");
    std::uint64_t g = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        g = gcd_u64(g, v[i]);
        if (g == 1) return 1; // early exit
    }
    return g;
}

// Given list of frequencies (in GHz), compute:
// - base_tick_ps: base tick in picoseconds (GCD of periods)
// - ticks_per_period: for each frequency, how many base ticks compose one of its periods
// Throws on invalid input.
inline void compute_base_tick_and_multipliers(const std::vector<double>& freqs_ghz,
                                              std::uint64_t& base_tick_ps_out,
                                              std::vector<std::uint64_t>& ticks_per_period_out,
                                              std::uint64_t min_base_ps = 10) {
    if (freqs_ghz.empty()) throw std::invalid_argument("freq list empty");
    std::vector<std::uint64_t> periods;
    periods.reserve(freqs_ghz.size());
    for (double f : freqs_ghz) {
        periods.push_back(period_ps_from_ghz(f));
    }

    std::uint64_t base = gcd_vector(periods);
    // safety: base must be >= 1
    if (base < 1) base = 1;

    // --- 加入最小步长限制，以减小仿真循环次数
    if (base < min_base_ps) {
        // 对每个周期按新的 base 重新计算 tick 数
        base = min_base_ps;
    }

    ticks_per_period_out.clear();
    ticks_per_period_out.reserve(periods.size());
    for (auto p : periods) {
        // 使用四舍五入确保误差最小
        ticks_per_period_out.push_back((p + base / 2) / base);
    }
    base_tick_ps_out = base;
}

inline std::string MsgTypeToStr(MsgType msg){
    switch (msg) {
        case MsgType::LD_REQ:  return "LD_REQ";
        case MsgType::LD_RESP: return "LD_RESP";
        case MsgType::ST_REQ:  return "ST_REQ";
        case MsgType::ST_RESP: return "ST_RESP";
        default:               return "UNKNOWN";
    }
}

#endif // __UTIL_HPP__