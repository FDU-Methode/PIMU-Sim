#include "../include/run_result_json.hpp"
#include "../include/util.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

namespace {

void writeString(
    std::ostream& out,
    std::string_view value
) {
    static constexpr char hex[] = "0123456789abcdef";

    out.put('"');

    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            out.put('\\');
            out.put(static_cast<char>(c));
        } else if (c < 0x20) {
            out << "\\u00"
                << hex[c >> 4]
                << hex[c & 0x0f];
        } else {
            out.put(static_cast<char>(c));
        }
    }

    out.put('"');
}

void writeTraffic(
    std::ostream& out,
    const MemoryTraffic& traffic
) {
    out << "{\"read_bytes\":" << traffic.read_bytes
        << ",\"write_bytes\":" << traffic.write_bytes
        << '}';
}

void writeResources(
    std::ostream& out,
    const ResourceCounters& counters
) {
    struct UnitField {
        const char* name;
        UnitType unit;
    };

    static constexpr UnitField fields[] = {
        {"dma1", UnitType::DMA1},
        {"dma2", UnitType::DMA2},
        {"dma3", UnitType::DMA3},
        {"spm", UnitType::SPM},
        {"sa", UnitType::SystolicArray},
        {"accumulator", UnitType::Accumulator},
        {"acc_bias", UnitType::AccBiasSram},
        {"acc_data", UnitType::AccDataSram},
        {"sfu", UnitType::SFU},
        {"gemv", UnitType::GEMV}
    };

    out << "{\"sampled_core_cycles\":"
        << counters.sampled_core_cycles
        << ",\"busy_cycles\":{";

    bool first = true;

    for (const auto& field : fields) {
        if (!first) {
            out.put(',');
        }

        first = false;

        writeString(out, field.name);
        out.put(':');

        out << counters.busy_cycles.at(
            UnitTypeToIndex(field.unit)
        );
    }

    out << "},\"ddr\":";
    writeTraffic(out, counters.ddr);

    out << ",\"spm\":";
    writeTraffic(out, counters.spm);

    out << ",\"acc_bias\":";
    writeTraffic(out, counters.acc_bias);

    out << ",\"acc_data\":";
    writeTraffic(out, counters.acc_data);

    out.put('}');
}

} // namespace

std::string runResultToJson(const RunResult& result) {
    std::ostringstream out;

    out.imbue(std::locale::classic());

    out << std::setprecision(
        std::numeric_limits<double>::max_digits10
    );

    out << "{\"schema_version\":"
        << kRunResultSchemaVersion
        << ",\"status\":";

    writeString(out, toString(result.status));

    out << ",\"core_cycles\":" << result.core_cycles
        << ",\"ddr_cycles\":" << result.ddr_cycles
        << ",\"fetched_words\":" << result.fetched_words
        << ",\"assembled_packets\":"
        << result.assembled_packets
        << ",\"host_elapsed_us\":";

    if (std::isfinite(result.host_elapsed_us)) {
        out << result.host_elapsed_us;
    } else {
        out << "null";
    }

    out << ",\"error\":";
    writeString(out, result.error);

    out << ",\"resources\":";
    writeResources(out, result.resources);

    out << ",\"session_resources\":";
    writeResources(out, result.session_resources);

    out.put('}');

    return out.str();
}