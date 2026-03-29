#include "../include/config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <filesystem>
#include <iostream>

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
        else if (k == "sram_bytes") c.sram_bytes = std::stoi(v);
        else if (k == "sram_ports") c.sram_ports = std::stoi(v);
        else if (k == "sram_data_width") c.sram_data_width = std::stoi(v);
        else if (k == "acc_bytes") c.acc_bytes = std::stoi(v);
        else if (k == "acc_ports") c.acc_ports = std::stoi(v);
        else if (k == "acc_data_width") c.acc_data_width = std::stoi(v);
        else if (k == "inst_cache_size") c.inst_cache_size = std::stoi(v);
        else if (k.length() > 9 && k.substr(k.length() - 9) == "_reg_addr") {
            uint32_t addr = parse_number(v);
            c.reg_addr_map[k] = addr;
            // Debug
            // std::cout << "Loaded Reg: " << k << " -> 0x" << std::hex << addr << std::dec << std::endl;
        }
        /*Add more configuration options here*/
    }
    return c;
}