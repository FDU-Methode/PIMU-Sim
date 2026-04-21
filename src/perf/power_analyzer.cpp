#include "../../include/perf/power_analyzer.hpp"

namespace {

double avg_power_mw(double energy_pj, uint64_t total_cycles, double cycle_ns) {
    const double sim_time_ns = static_cast<double>(total_cycles) * cycle_ns;
    if (sim_time_ns <= 0.0) return 0.0;
    return energy_pj / sim_time_ns;  // pJ/ns == mW
}

double energy_share_pct(double energy_pj, double total_energy_pj) {
    if (total_energy_pj <= 0.0) return 0.0;
    return 100.0 * energy_pj / total_energy_pj;
}

double cycle_share_pct(uint64_t cycles, uint64_t total_cycles) {
    if (total_cycles == 0) return 0.0;
    return 100.0 * static_cast<double>(cycles) / static_cast<double>(total_cycles);
}

}  // namespace

PowerAnalyzer::PowerAnalyzer(Logger& logger, const Config& cfg)
    : logger(logger), cfg(cfg), enabled_(cfg.enable_power) {
    if (cfg.npu_freq > 0.0) {
        cycle_ns_ = 1.0 / cfg.npu_freq;
    }
}

double PowerAnalyzer::dma_power_mw(DmaPowerMode mode) const {
    switch (mode) {
        case DmaPowerMode::Idle:      return cfg.dma_idle_power_mw;
        case DmaPowerMode::BusyMvin:  return cfg.dma_busy_mvin_power_mw;
        case DmaPowerMode::BusyMvout: return cfg.dma_busy_mvout_power_mw;
        default:                      return 0.0;
    }
}

double PowerAnalyzer::acc_power_mw(AccPowerMode mode) const {
    switch (mode) {
        case AccPowerMode::Idle:           return cfg.acc_idle_power_mw;
        case AccPowerMode::BusyAccadd:     return cfg.acc_busy_accadd_power_mw;
        case AccPowerMode::BusyBiasLoad:   return cfg.acc_busy_bias_load_power_mw;
        case AccPowerMode::BusyDmaRd:      return cfg.acc_busy_dma_rd_power_mw;
        case AccPowerMode::BusyDmaWr:      return cfg.acc_busy_dma_wr_power_mw;
        case AccPowerMode::BusyMatadd:     return cfg.acc_busy_matadd_power_mw;
        default:                           return 0.0;
    }
}

double PowerAnalyzer::sfu_power_mw(SfuPowerMode mode) const {
    switch (mode) {
        case SfuPowerMode::Idle:           return cfg.sfu_idle_power_mw;
        case SfuPowerMode::BusySoftmax:    return cfg.sfu_busy_softmax_power_mw;
        case SfuPowerMode::BusyLayernorm:  return cfg.sfu_busy_layernorm_power_mw;
        case SfuPowerMode::BusyGelu:       return cfg.sfu_busy_gelu_power_mw;
        case SfuPowerMode::BusyTranspose:  return cfg.sfu_busy_transpose_power_mw;
        case SfuPowerMode::BusyPoolmax:    return cfg.sfu_busy_poolmax_power_mw;
        case SfuPowerMode::BusyResample:   return cfg.sfu_busy_resample_power_mw;
        default:                           return 0.0;
    }
}

double PowerAnalyzer::sa_power_mw(SaPowerMode mode) const {
    switch (mode) {
        case SaPowerMode::Idle:      return cfg.sa_idle_power_mw;
        case SaPowerMode::BusyGemm:  return cfg.sa_busy_gemm_power_mw;
        case SaPowerMode::BusyConv:  return cfg.sa_busy_conv_power_mw;
        default:                     return 0.0;
    }
}

const char* PowerAnalyzer::dma_mode_str(DmaPowerMode mode) {
    switch (mode) {
        case DmaPowerMode::Idle:      return "idle";
        case DmaPowerMode::BusyMvin:  return "busy_mvin";
        case DmaPowerMode::BusyMvout: return "busy_mvout";
        default:                      return "unknown";
    }
}

const char* PowerAnalyzer::acc_mode_str(AccPowerMode mode) {
    switch (mode) {
        case AccPowerMode::Idle:          return "idle";
        case AccPowerMode::BusyAccadd:    return "busy_accadd";
        case AccPowerMode::BusyBiasLoad:  return "busy_bias_load";
        case AccPowerMode::BusyDmaRd:     return "busy_dma_rd";
        case AccPowerMode::BusyDmaWr:     return "busy_dma_wr";
        case AccPowerMode::BusyMatadd:    return "busy_matadd";
        default:                          return "unknown";
    }
}

const char* PowerAnalyzer::sfu_mode_str(SfuPowerMode mode) {
    switch (mode) {
        case SfuPowerMode::Idle:           return "idle";
        case SfuPowerMode::BusySoftmax:    return "busy_softmax";
        case SfuPowerMode::BusyLayernorm:  return "busy_layernorm";
        case SfuPowerMode::BusyGelu:       return "busy_gelu";
        case SfuPowerMode::BusyTranspose:  return "busy_transpose";
        case SfuPowerMode::BusyPoolmax:    return "busy_poolmax";
        case SfuPowerMode::BusyResample:   return "busy_resample";
        default:                           return "unknown";
    }
}

const char* PowerAnalyzer::sa_mode_str(SaPowerMode mode) {
    switch (mode) {
        case SaPowerMode::Idle:      return "idle";
        case SaPowerMode::BusyGemm:  return "busy_gemm";
        case SaPowerMode::BusyConv:  return "busy_conv";
        default:                     return "unknown";
    }
}

void PowerAnalyzer::sample(DmaPowerMode dma1_mode,
                           DmaPowerMode dma2_mode,
                           DmaPowerMode dma3_mode,
                           AccPowerMode acc_mode,
                           SfuPowerMode sfu_mode,
                           SaPowerMode sa_mode) {
    if (!enabled_) return;

    ++total_cycles_;

    const double dma1_pj = dma_power_mw(dma1_mode) * cycle_ns_;
    const double dma2_pj = dma_power_mw(dma2_mode) * cycle_ns_;
    const double dma3_pj = dma_power_mw(dma3_mode) * cycle_ns_;
    const double acc_pj  = acc_power_mw(acc_mode) * cycle_ns_;
    const double sfu_pj  = sfu_power_mw(sfu_mode) * cycle_ns_;
    const double sa_pj   = sa_power_mw(sa_mode) * cycle_ns_;

    dma1_energy_pj_ += dma1_pj;
    dma2_energy_pj_ += dma2_pj;
    dma3_energy_pj_ += dma3_pj;
    acc_energy_pj_  += acc_pj;
    sfu_energy_pj_  += sfu_pj;
    sa_energy_pj_   += sa_pj;
    total_energy_pj_ += dma1_pj + dma2_pj + dma3_pj + acc_pj + sfu_pj + sa_pj;

    dma1_mode_cycles_[static_cast<size_t>(dma1_mode)]++;
    dma2_mode_cycles_[static_cast<size_t>(dma2_mode)]++;
    dma3_mode_cycles_[static_cast<size_t>(dma3_mode)]++;
    acc_mode_cycles_[static_cast<size_t>(acc_mode)]++;
    sfu_mode_cycles_[static_cast<size_t>(sfu_mode)]++;
    sa_mode_cycles_[static_cast<size_t>(sa_mode)]++;
}

void PowerAnalyzer::report() {
    if (!enabled_) return;

    const double sim_time_ns = static_cast<double>(total_cycles_) * cycle_ns_;
    const double sim_time_us = sim_time_ns / 1000.0;
    const double total_energy_uj = total_energy_pj_ / 1.0e6;
    const double total_avg_power_mw = avg_power_mw(total_energy_pj_, total_cycles_, cycle_ns_);

    const double dma_total_energy_pj = dma1_energy_pj_ + dma2_energy_pj_ + dma3_energy_pj_;
    const double dma_total_energy_uj = dma_total_energy_pj / 1.0e6;

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

    // ---------------- Summary ----------------
    const std::string border0 =
        "+" + std::string(66, '-') + "+";

    emit(border0);
    emit(make_title_row("Power Report", border0.size()));
    emit(border0);

    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(4)
           << "Cycles=" << total_cycles_
           << ", Cycle(ns)=" << cycle_ns_
           << ", SimTime(us)=" << sim_time_us
           << ", TotalEnergy(uJ)=" << total_energy_uj
           << ", AvgPower(mW)=" << total_avg_power_mw;
        emit(os.str());
    }

    emit(border0);

    // ---------------- Block Breakdown ----------------
    const int w_block = 12;
    const int w_energy = 14;
    const int w_power = 14;
    const int w_share = 10;

    const std::string border1 =
        "+" + std::string(w_block + 2, '-') +
        "+" + std::string(w_energy + 2, '-') +
        "+" + std::string(w_power + 2, '-') +
        "+" + std::string(w_share + 2, '-') + "+";

    emit(border1);
    emit(make_title_row("Block Power Breakdown", border1.size()));
    emit(border1);

    {
        std::ostringstream os;
        os << "| " << std::left  << std::setw(w_block)  << "Block"
           << " | " << std::right << std::setw(w_energy) << "Energy(uJ)"
           << " | " << std::right << std::setw(w_power)  << "AvgPower(mW)"
           << " | " << std::right << std::setw(w_share)  << "Share(%)"
           << " |";
        emit(os.str());
    }

    emit(border1);

    auto emit_block_row = [&](const std::string& name, double energy_pj) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(4);
        os << "| " << std::left  << std::setw(w_block)  << name
           << " | " << std::right << std::setw(w_energy) << energy_pj / 1.0e6
           << " | " << std::right << std::setw(w_power)  << avg_power_mw(energy_pj, total_cycles_, cycle_ns_)
           << " | " << std::right << std::setw(w_share)  << energy_share_pct(energy_pj, total_energy_pj_)
           << " |";
        emit(os.str());
    };

    emit_block_row("DMA Total", dma_total_energy_pj);
    emit_block_row("DMA1", dma1_energy_pj_);
    emit_block_row("DMA2", dma2_energy_pj_);
    emit_block_row("DMA3", dma3_energy_pj_);
    emit_block_row("ACC",  acc_energy_pj_);
    emit_block_row("SFU",  sfu_energy_pj_);
    emit_block_row("SA",   sa_energy_pj_);

    emit(border1);

    // ---------------- Mode Breakdown ----------------
    const int w_unit = 8;
    const int w_mode = 18;
    const int w_cycles = 12;
    const int w_pct = 10;

    const std::string border2 =
        "+" + std::string(w_unit + 2, '-') +
        "+" + std::string(w_mode + 2, '-') +
        "+" + std::string(w_cycles + 2, '-') +
        "+" + std::string(w_pct + 2, '-') + "+";

    emit(border2);
    emit(make_title_row("Mode Cycle Breakdown", border2.size()));
    emit(border2);

    {
        std::ostringstream os;
        os << "| " << std::left  << std::setw(w_unit)  << "Unit"
           << " | " << std::left  << std::setw(w_mode)  << "Mode"
           << " | " << std::right << std::setw(w_cycles) << "Cycles"
           << " | " << std::right << std::setw(w_pct)    << "Share(%)"
           << " |";
        emit(os.str());
    }

    emit(border2);

    auto emit_mode_row = [&](const std::string& unit,
                             const std::string& mode,
                             uint64_t cycles) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        os << "| " << std::left  << std::setw(w_unit)  << unit
           << " | " << std::left  << std::setw(w_mode)  << mode
           << " | " << std::right << std::setw(w_cycles) << cycles
           << " | " << std::right << std::setw(w_pct)    << cycle_share_pct(cycles, total_cycles_)
           << " |";
        emit(os.str());
    };

    emit_mode_row("DMA1", dma_mode_str(DmaPowerMode::Idle),      dma1_mode_cycles_[0]);
    emit_mode_row("DMA1", dma_mode_str(DmaPowerMode::BusyMvin),  dma1_mode_cycles_[1]);
    emit_mode_row("DMA1", dma_mode_str(DmaPowerMode::BusyMvout), dma1_mode_cycles_[2]);

    emit_mode_row("DMA2", dma_mode_str(DmaPowerMode::Idle),      dma2_mode_cycles_[0]);
    emit_mode_row("DMA2", dma_mode_str(DmaPowerMode::BusyMvin),  dma2_mode_cycles_[1]);
    emit_mode_row("DMA2", dma_mode_str(DmaPowerMode::BusyMvout), dma2_mode_cycles_[2]);

    emit_mode_row("DMA3", dma_mode_str(DmaPowerMode::Idle),      dma3_mode_cycles_[0]);
    emit_mode_row("DMA3", dma_mode_str(DmaPowerMode::BusyMvin),  dma3_mode_cycles_[1]);
    emit_mode_row("DMA3", dma_mode_str(DmaPowerMode::BusyMvout), dma3_mode_cycles_[2]);

    emit_mode_row("ACC", acc_mode_str(AccPowerMode::Idle),         acc_mode_cycles_[0]);
    emit_mode_row("ACC", acc_mode_str(AccPowerMode::BusyAccadd),   acc_mode_cycles_[1]);
    emit_mode_row("ACC", acc_mode_str(AccPowerMode::BusyBiasLoad), acc_mode_cycles_[2]);
    emit_mode_row("ACC", acc_mode_str(AccPowerMode::BusyDmaRd),    acc_mode_cycles_[3]);
    emit_mode_row("ACC", acc_mode_str(AccPowerMode::BusyDmaWr),    acc_mode_cycles_[4]);
    emit_mode_row("ACC", acc_mode_str(AccPowerMode::BusyMatadd),   acc_mode_cycles_[5]);

    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::Idle),          sfu_mode_cycles_[0]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusySoftmax),   sfu_mode_cycles_[1]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusyLayernorm), sfu_mode_cycles_[2]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusyGelu),      sfu_mode_cycles_[3]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusyTranspose), sfu_mode_cycles_[4]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusyPoolmax),   sfu_mode_cycles_[5]);
    emit_mode_row("SFU", sfu_mode_str(SfuPowerMode::BusyResample),  sfu_mode_cycles_[6]);

    emit_mode_row("SA", sa_mode_str(SaPowerMode::Idle),     sa_mode_cycles_[0]);
    emit_mode_row("SA", sa_mode_str(SaPowerMode::BusyGemm), sa_mode_cycles_[1]);
    emit_mode_row("SA", sa_mode_str(SaPowerMode::BusyConv), sa_mode_cycles_[2]);

    emit(border2);
}
