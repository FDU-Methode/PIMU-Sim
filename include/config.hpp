#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <string>
#include <vector>
#include <cstdint>
#include <map>

enum class SimMode {
    FUNCTIONAL,
    PERFORMANCE
};

struct Config{
    // NPU Core Config
    SimMode simulator_type = SimMode::FUNCTIONAL;
    int systolic_array_size = 32;

    // AXI Config
    int axi_data_width = 256;

    std::map<std::string, uint32_t> reg_addr_map;
    // Simulation Config
    std::string inst_file = "./benchmarks/tb_test/test.txt";
    std::string log_file = "./log/tb_test_log.txt";
    bool enable_log = false;

    // --------- Functional Model Config ---------
    /* Add functional model configuration options here */

    // --------- Performance Model Config ---------
    double npu_freq = 1; // NPU Core Running Frequency (GHz)
    int max_sim_cycle = 100000;

    // DDR Config
    int ddr_mode = 1; // 1: analytic 0: detail
    std::string dramsim_cfg_path = "";
    double ddr_freq = 2.13; // GHz
    double ddr_bandwidth = 20.0; // GB/s 直接使用带宽估算访存延时误差大，暂不考虑
    int tCAS = 15; // CAS Latency (DDR cycle)
    int tRCD = 15; // RAS to CAS Latency (DDR cycle)
    int tRP = 15; // Precharge Latency (DDR cycle)
    int tBURST = 4; // Burst Latency (ns)
    int BL = 8; // DDR Burst Length (通常为8bits)
    int ddr_bus_width = 64; // DDR总线宽度 (通常为64bits)
    int ddr_channel_num = 4; // DDR通道数 (移动端至高端NPU的DDR通常为2-4通道，HBM2/3通常为8通道)
    int ddr_request_queue_depth = 8; // DDR请求队列深度

    // SPM Config
    int sram_bank_num = 8;
    int sram_bank_bytes = 262144;
    int sram_ports = 2;
    int sram_data_width = 8;

    // ACC Config
    int acc_bank_num = 2;
    int bias_bank_num = 1;
    int acc_bytes = 262144;
    int acc_ports = 2;
    int acc_data_width = 32;

    // Inst Config
    int inst_cache_size = 256;
    int inst_queue_depth = 8;

    /*Add more configuration options here*/

    static Config from_file(const std::string &path);

};

#endif