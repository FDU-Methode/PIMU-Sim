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
    std::string compiled_model_lib = "";
    std::string onnx_model_file = "";
    std::string onnx_mlir_bin = "onnx-mlir";
    std::string onnx_mlir_flags = "";
    std::string compile_work_dir = "./build/compiler_run";
    std::string model_input_dir = "";
    std::string model_output_dir = "./sim_output";
    bool enable_log = false;

    // --------- Functional Model Config ---------
    /* Add functional model configuration options here */

    // --------- Performance Model Config ---------
    double npu_freq = 1;
    int max_sim_cycle = 100000;

    // DDR Config
    int ddr_mode = 1;
    std::string dramsim_cfg_path = "";
    double ddr_freq = 2.13;
    double ddr_bandwidth = 20.0;
    int tCAS = 15;
    int tRCD = 15;
    int tRP = 15;
    int tBURST = 4;
    int BL = 8;
    int ddr_bus_width = 64;
    int ddr_channel_num = 4;
    int ddr_request_queue_depth = 8;

    // SPM Config
    int sram_bank_num = 8;
    int sram_bank_bytes = 262144;
    int sram_data_width = 8;

    // ACC Config
    int bias_bank_num = 1;
    int bias_bytes = 262144;
    int acc_bank_num = 2;
    int acc_bytes = 262144;
    int acc_data_width = 32;

    // Inst Config
    int inst_cache_size = 256;
    int inst_queue_depth = 8;

    static Config from_file(const std::string &path);
};

#endif