#include <iostream>
#include "../../include/perf/hw_utilization.hpp"

void HardWareUtilizationTracker::reset() {
    total_cycles = 0;
    busy_cycles.fill(0);

    ddr_read_bytes_total = 0;
    ddr_write_bytes_total = 0;
    spm_read_bytes_total = 0;
    spm_write_bytes_total = 0;

    ddr_read_bytes_current = 0;
    ddr_write_bytes_current = 0;
    spm_read_bytes_current = 0;
    spm_write_bytes_current = 0;

    ddr_peak_bytes_per_cycle = 0;
    spm_peak_bytes_per_cycle = 0;
}

void HardWareUtilizationTracker::sample(const ModuleState& state) {
    ++total_cycles;
    if (state.dma)  ++busy_cycles[UnitTypeToIndex(UnitType::DMA)];
    if (state.spm)  ++busy_cycles[UnitTypeToIndex(UnitType::SPM)];
    if (state.sa)   ++busy_cycles[UnitTypeToIndex(UnitType::SystolicArray)];
    if (state.acc)  ++busy_cycles[UnitTypeToIndex(UnitType::Accumulator)];
    if (state.sfu)  ++busy_cycles[UnitTypeToIndex(UnitType::SFU)];
    if (state.gemv) ++busy_cycles[UnitTypeToIndex(UnitType::GEMV)];

    // Update Peak Bandwidth
    uint64_t ddr_total_current = ddr_read_bytes_current + ddr_write_bytes_current;
    uint64_t spm_total_current = spm_read_bytes_current + spm_write_bytes_current;

    if(ddr_total_current > ddr_peak_bytes_per_cycle) {
        ddr_peak_bytes_per_cycle = ddr_total_current;
    }
    if(spm_total_current > spm_peak_bytes_per_cycle) {
        spm_peak_bytes_per_cycle = spm_total_current;
    }

    ddr_read_bytes_current = 0;
    ddr_write_bytes_current = 0;
    spm_read_bytes_current = 0;
    spm_write_bytes_current = 0;
}

double HardWareUtilizationTracker::utilization(UnitType unit) {
    if(total_cycles == 0) return 0.0;
    return 100 * static_cast<double>(busyCycles(unit)) / static_cast<double>(total_cycles);
}

void HardWareUtilizationTracker::report() {
    const int w_unit  = 16;
    const int w_busy  = 12;
    const int w_total = 12;
    const int w_util  = 10;

    const std::string border1 =
        "+" + std::string(w_unit + 2, '-') +
        "+" + std::string(w_busy + 2, '-') +
        "+" + std::string(w_total + 2, '-') +
        "+" + std::string(w_util + 2, '-') + "+";

    auto emit = [&](const std::string& line) {
        std::cout << line << "\n";
        logger.log("%s", line.c_str());
    };

    auto make_title_row = [&](const std::string& title, size_t width) -> std::string {
        const size_t inner = width - 2;
        if (title.size() >= inner) return "|" + title.substr(0, inner) + "|";
        const size_t left  = (inner - title.size()) / 2;
        const size_t right = inner - title.size() - left;
        return "|" + std::string(left, ' ') + title + std::string(right, ' ') + "|";
    };

    emit(border1);
    emit(make_title_row("HW Utilization Report", border1.size()));
    emit(border1);

    {
        std::ostringstream os;
        os << "| " << std::left  << std::setw(w_unit)  << "Module"
           << " | " << std::right << std::setw(w_busy)  << "Busy Cycles"
           << " | " << std::right << std::setw(w_total) << "Total Cycles"
           << " | " << std::right << std::setw(w_util)  << "Util(%)"
           << " |";
        emit(os.str());
    }

    emit(border1);

    const UnitType units[] = {
        UnitType::DMA, UnitType::SPM, UnitType::SystolicArray,
        UnitType::Accumulator, UnitType::SFU, UnitType::GEMV
    };

    for (UnitType unit : units) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        os << "| " << std::left  << std::setw(w_unit)  << UnitTypeToStr(unit)
           << " | " << std::right << std::setw(w_busy)  << busyCycles(unit)
           << " | " << std::right << std::setw(w_total) << total_cycles
           << " | " << std::right << std::setw(w_util)  << utilization(unit)
           << " |";
        emit(os.str());
    }

    emit(border1);

    const int w_path  = 10;
    const int w_read  = 14;
    const int w_write = 14;
    const int w_bpc   = 12;
    const int w_gbps  = 12;
    const int w_peak  = 12;

    const std::string border2 =
        "+" + std::string(w_path + 2, '-') +
        "+" + std::string(w_read + 2, '-') +
        "+" + std::string(w_write + 2, '-') +
        "+" + std::string(w_bpc + 2, '-') +
        "+" + std::string(w_gbps + 2, '-') +
        "+" + std::string(w_peak + 2, '-') + "+";

    emit(border2);
    emit(make_title_row("Bandwidth Report", border2.size()));
    emit(border2);

    {
        std::ostringstream os;
        os << "| " << std::left  << std::setw(w_path)  << "Path"
           << " | " << std::right << std::setw(w_read)  << "Read Bytes"
           << " | " << std::right << std::setw(w_write) << "Write Bytes"
           << " | " << std::right << std::setw(w_bpc)   << "Avg B/Cycle"
           << " | " << std::right << std::setw(w_gbps)  << "Avg GB/s"
           << " | " << std::right << std::setw(w_peak)  << "Peak B/Cycle"
           << " |";
        emit(os.str());
    }

    emit(border2);

    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        os << "| " << std::left  << std::setw(w_path)  << "DDR"
           << " | " << std::right << std::setw(w_read)  << ddr_read_bytes_total
           << " | " << std::right << std::setw(w_write) << ddr_write_bytes_total
           << " | " << std::right << std::setw(w_bpc)   << avgBytesPerCycle(ddr_read_bytes_total + ddr_write_bytes_total)
           << " | " << std::right << std::setw(w_gbps)  << avgBandwidthGBps(ddr_read_bytes_total + ddr_write_bytes_total)
           << " | " << std::right << std::setw(w_peak)  << ddr_peak_bytes_per_cycle
           << " |";
        emit(os.str());
    }

    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        os << "| " << std::left  << std::setw(w_path)  << "SPM"
           << " | " << std::right << std::setw(w_read)  << spm_read_bytes_total
           << " | " << std::right << std::setw(w_write) << spm_write_bytes_total
           << " | " << std::right << std::setw(w_bpc)   << avgBytesPerCycle(spm_read_bytes_total + spm_write_bytes_total)
           << " | " << std::right << std::setw(w_gbps)  << avgBandwidthGBps(spm_read_bytes_total + spm_write_bytes_total)
           << " | " << std::right << std::setw(w_peak)  << spm_peak_bytes_per_cycle
           << " |";
        emit(os.str());
    }

    emit(border2);
}


// Compute Bandwidth
void HardWareUtilizationTracker::addDdrReadBytes(uint64_t bytes) {
    ddr_read_bytes_total += bytes;
    ddr_read_bytes_current += bytes;
}

void HardWareUtilizationTracker::addDdrWriteBytes(uint64_t bytes) {
    ddr_write_bytes_total += bytes;
    ddr_write_bytes_current += bytes;
}

void HardWareUtilizationTracker::addSpmReadBytes(uint64_t bytes) {
    spm_read_bytes_total += bytes;
    spm_read_bytes_current += bytes;
}

void HardWareUtilizationTracker::addSpmWriteBytes(uint64_t bytes) {
    spm_write_bytes_total += bytes;
    spm_write_bytes_current += bytes;
}

double HardWareUtilizationTracker::avgBytesPerCycle(uint64_t bytes) {
    if(total_cycles == 0) return 0.0;
    return static_cast<double>(bytes) / static_cast<double>(total_cycles);
}

double HardWareUtilizationTracker::avgBandwidthGBps(uint64_t bytes) {
    return avgBytesPerCycle(bytes) * core_freq_ghz;
}