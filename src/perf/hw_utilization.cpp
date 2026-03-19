#include <iostream>
#include "../../include/perf/hw_utilization.hpp"

void HardWareUtilizationTracker::reset() {
    total_cycles = 0;
    busy_cycles.fill(0);
}

void HardWareUtilizationTracker::sample(const ModuleState& state) {
    ++total_cycles;
    if (state.dma)  ++busy_cycles[UnitTypeToIndex(UnitType::DMA)];
    if (state.spm)  ++busy_cycles[UnitTypeToIndex(UnitType::SPM)];
    if (state.sa)   ++busy_cycles[UnitTypeToIndex(UnitType::SystolicArray)];
    if (state.acc)  ++busy_cycles[UnitTypeToIndex(UnitType::Accumulator)];
    if (state.sfu)  ++busy_cycles[UnitTypeToIndex(UnitType::SFU)];
    if (state.gemv) ++busy_cycles[UnitTypeToIndex(UnitType::GEMV)];
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

    const std::string border =
        "+" + std::string(w_unit + 2, '-') +
        "+" + std::string(w_busy + 2, '-') +
        "+" + std::string(w_total + 2, '-') +
        "+" + std::string(w_util + 2, '-') + "+";

    auto emit = [&](const std::string& line) {
        std::cout << line << "\n";
        logger.log("%s", line.c_str());
    };

    auto make_title_row = [&](const std::string& title) -> std::string {
        const size_t inner = border.size() - 2;  // remove two '|'
        if (title.size() >= inner) {
            return "|" + title.substr(0, inner) + "|";
        }
        const size_t left  = (inner - title.size()) / 2;
        const size_t right = inner - title.size() - left;
        return "|" + std::string(left, ' ') + title + std::string(right, ' ') + "|";
    };

    emit(border);
    emit(make_title_row("HW Utilization Report"));
    emit(border);

    {
        std::ostringstream os;
        os << "| " << std::left  << std::setw(w_unit)  << "Module"
           << " | " << std::right << std::setw(w_busy)  << "Busy Cycles"
           << " | " << std::right << std::setw(w_total) << "Total Cycles"
           << " | " << std::right << std::setw(w_util)  << "Util(%)"
           << " |";
        emit(os.str());
    }

    emit(border);

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

    emit(border);
}