#include "../include/config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>

static std::string trim(std::string s) {
    while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

static uint32_t parse_number(const std::string& val) {
    try {
        std::string clean_val = val;
        int base = 10;

        // 1. 处理 Verilog 风格 (e.g., 32'hB0)
        size_t h_pos = val.find("'h");
        if (h_pos != std::string::npos) {
            clean_val = val.substr(h_pos + 2); // 跳过 'h
            base = 16;
        }
        // 2. 处理 C 风格十六进制 (e.g., 0xB0)
        else if (val.find("0x") == 0 || val.find("0X") == 0) {
            clean_val = val.substr(2);
            base = 16;
        }

        // 转换为 unsigned long
        return std::stoul(clean_val, nullptr, base);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid number format in config: " + val);
    }
}

static bool parse_bool(const std::string& val) {
    std::string s = trim(val);
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    throw std::runtime_error("Invalid boolean format in config: " + val);
}

static std::optional<double> extract_last_double(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    std::optional<double> last_val;

    while (iss >> token) {
        try {
            size_t idx = 0;
            double v = std::stod(token, &idx);
            if (idx == token.size()) last_val = v;
        } catch (...) {
        }
    }
    return last_val;
}

static void assign_power_value(Config& c,
                               const std::string& unit,
                               const std::string& mode,
                               double total_power_mw) {
    if (unit == "dma") {
        if (mode == "idle") c.dma_idle_power_mw = total_power_mw;
        else if (mode == "busy_mvin") c.dma_busy_mvin_power_mw = total_power_mw;
        else if (mode == "busy_mvout") c.dma_busy_mvout_power_mw = total_power_mw;
    } else if (unit == "acc") {
        if (mode == "idle") c.acc_idle_power_mw = total_power_mw;
        else if (mode == "busy_accadd") c.acc_busy_accadd_power_mw = total_power_mw;
        else if (mode == "busy_bias_load") c.acc_busy_bias_load_power_mw = total_power_mw;
        else if (mode == "busy_dma_rd") c.acc_busy_dma_rd_power_mw = total_power_mw;
        else if (mode == "busy_dma_wr") c.acc_busy_dma_wr_power_mw = total_power_mw;
        else if (mode == "busy_matadd") c.acc_busy_matadd_power_mw = total_power_mw;
    } else if (unit == "sfu") {
        if (mode == "idle") c.sfu_idle_power_mw = total_power_mw;
        else if (mode == "busy_softmax") c.sfu_busy_softmax_power_mw = total_power_mw;
        else if (mode == "busy_layernorm") c.sfu_busy_layernorm_power_mw = total_power_mw;
        else if (mode == "busy_gelu") c.sfu_busy_gelu_power_mw = total_power_mw;
        else if (mode == "busy_transpose") c.sfu_busy_transpose_power_mw = total_power_mw;
        else if (mode == "busy_poolmax") c.sfu_busy_poolmax_power_mw = total_power_mw;
        else if (mode == "busy_resample") c.sfu_busy_resample_power_mw = total_power_mw;
    } else if (unit == "sa") {
        if (mode == "idle") c.sa_idle_power_mw = total_power_mw;
        else if (mode == "busy_gemm") c.sa_busy_gemm_power_mw = total_power_mw;
        else if (mode == "busy_conv") c.sa_busy_conv_power_mw = total_power_mw;
    }
}

static void load_power_report(Config& c, const std::filesystem::path& power_path) {
    std::ifstream ifs(power_path);
    if (!ifs) {
        throw std::runtime_error("Cannot open power file: " + power_path.string());
    }

    std::string current_unit;
    std::string current_mode;
    std::string line;

    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line[0] == '#') {
            std::string body = trim(line.substr(1));
            if (body == "dma" || body == "acc" || body == "sfu" || body == "sa") {
                current_unit = body;
            }
            continue;
        }

        if (!line.empty() && line.back() == ':') {
            current_mode = trim(line.substr(0, line.size() - 1));
            continue;
        }

        if (line.rfind("Total", 0) == 0 && !current_unit.empty() && !current_mode.empty()) {
            auto total_power = extract_last_double(line);
            if (total_power.has_value()) {
                assign_power_value(c, current_unit, current_mode, *total_power);
            }
        }
    }
}

Config Config::from_file(const std::string &path){
    Config c;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open config: " + path);

    std::string line;
    while (std::getline(ifs, line)) {
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        line = trim(line);
        if (line.empty()) continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (k == "simulator_type") {
            if (v == "FUNCTIONAL") c.simulator_type = SimMode::FUNCTIONAL;
            else if (v == "PERFORMANCE") c.simulator_type = SimMode::PERFORMANCE;
        }
        else if (k == "systolic_array_size") c.systolic_array_size = std::stoi(v);
        else if (k == "npu_freq") c.npu_freq = std::stod(v);
        else if (k == "ddr_freq") c.ddr_freq = std::stod(v);
        else if (k == "ddr_bandwidth") c.ddr_bandwidth = std::stod(v);
        else if (k == "tCAS") c.tCAS = std::stoi(v);
        else if (k == "tRCD") c.tRCD = std::stoi(v);
        else if (k == "tRP") c.tRP = std::stoi(v);
        else if (k == "tBURST") c.tBURST = std::stoi(v);
        else if (k == "BL") c.BL = std::stoi(v);
        else if (k == "ddr_bus_width") c.ddr_bus_width = std::stoi(v);
        else if (k == "ddr_channel_num") c.ddr_channel_num = std::stoi(v);
        else if (k == "ddr_request_queue_depth") c.ddr_request_queue_depth = std::stoi(v);
        else if (k == "max_sim_cycle") c.max_sim_cycle = std::stoi(v);
        else if (k == "inst_file") c.inst_file = v;
        else if (k == "log_file") c.log_file = v;
        else if (k == "enable_log") c.enable_log = std::stoi(v);
        else if (k == "ddr_mode") c.ddr_mode = std::stoi(v);
        else if (k == "dramsim_cfg_path") c.dramsim_cfg_path = v;
        else if (k == "axi_data_width") c.axi_data_width = std::stod(v);
        else if (k == "sram_bank_num") c.sram_bank_num = std::stoi(v);
        else if (k == "sram_bank_bytes") c.sram_bank_bytes = std::stoi(v);
        else if (k == "sram_data_width") c.sram_data_width = std::stoi(v);
        else if (k == "bias_bank_num") c.bias_bank_num = std::stoi(v);
        else if (k == "bias_bytes") c.bias_bytes = std::stoi(v);
        else if (k == "acc_bank_num") c.acc_bank_num = std::stoi(v);
        else if (k == "acc_bytes") c.acc_bytes = std::stoi(v);
        else if (k == "acc_data_width") c.acc_data_width = std::stoi(v);
        else if (k == "inst_cache_size") c.inst_cache_size = std::stoi(v);
        else if (k == "inst_queue_depth") c.inst_queue_depth = std::stoi(v);
        else if (k == "enable_power") c.enable_power = parse_bool(v);
        else if (k == "power_file") c.power_file = v;
        else if (k.length() > 9 && k.substr(k.length() - 9) == "_reg_addr") {
            uint32_t addr = parse_number(v);
            c.reg_addr_map[k] = addr;
            // Debug
            // std::cout << "Loaded Reg: " << k << " -> 0x" << std::hex << addr << std::dec << std::endl;
        }
        /*Add more configuration options here*/
    }

    if (c.enable_power) {
        std::filesystem::path power_path(c.power_file);
        if (!power_path.is_absolute() && !std::filesystem::exists(power_path)) {
            power_path = std::filesystem::path(path).parent_path() / power_path;
        }
        load_power_report(c, power_path);
    }

    return c;
}