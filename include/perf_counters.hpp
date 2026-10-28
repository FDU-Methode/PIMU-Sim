#ifndef PERF_COUNTERS_HPP
#define PERF_COUNTERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Indices follow UnitTypeToIndex().
// The aggregate DMA slot is not sampled.
inline constexpr std::size_t kResourceUnitSlots = 11;

struct MemoryTraffic {
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
};

struct ResourceCounters {
    std::uint64_t sampled_core_cycles = 0;

    std::array<
        std::uint64_t,
        kResourceUnitSlots
    > busy_cycles{};

    MemoryTraffic ddr;
    MemoryTraffic spm;
    MemoryTraffic acc_bias;
    MemoryTraffic acc_data;
};

inline std::uint64_t counterDelta(
    std::uint64_t end,
    std::uint64_t begin
) {
    if (end < begin) {
        throw std::logic_error(
            "Counters must be monotonic within a run"
        );
    }

    return end - begin;
}

inline MemoryTraffic trafficDelta(
    const MemoryTraffic& end,
    const MemoryTraffic& begin
) {
    return {
        counterDelta(end.read_bytes, begin.read_bytes),
        counterDelta(end.write_bytes, begin.write_bytes)
    };
}

inline ResourceCounters resourceDelta(
    const ResourceCounters& end,
    const ResourceCounters& begin
) {
    ResourceCounters result;

    result.sampled_core_cycles = counterDelta(
        end.sampled_core_cycles,
        begin.sampled_core_cycles
    );

    for (std::size_t i = 0;
         i < result.busy_cycles.size();
         ++i) {
        result.busy_cycles[i] = counterDelta(
            end.busy_cycles[i],
            begin.busy_cycles[i]
        );
    }

    result.ddr = trafficDelta(end.ddr, begin.ddr);
    result.spm = trafficDelta(end.spm, begin.spm);

    result.acc_bias = trafficDelta(
        end.acc_bias,
        begin.acc_bias
    );

    result.acc_data = trafficDelta(
        end.acc_data,
        begin.acc_data
    );

    return result;
}

#endif