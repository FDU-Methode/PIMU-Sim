#include "regfile.hpp"

RegFile::RegFile(const Config& cfg) {
    auto get_addr = [&](const std::string& name) -> uint32_t {
        auto it = cfg.reg_addr_map.find(name);
        if (it == cfg.reg_addr_map.end()) {
            throw std::runtime_error("Config missing register definition: " + name);
        }
        return it->second;
    };

    mvin_rs1_reg_addr              = get_addr("mvin_rs1_reg_addr");
    mvin_rs2_reg_addr              = get_addr("mvin_rs2_reg_addr");
    mvout_rs1_reg_addr             = get_addr("mvout_rs1_reg_addr");
    mvout_rs2_reg_addr             = get_addr("mvout_rs2_reg_addr");
    config_mvin_rs1_reg_addr       = get_addr("config_mvin_rs1_reg_addr");
    config_mvin_rs2_reg_addr       = get_addr("config_mvin_rs2_reg_addr");
    config_mvout_rs1_reg_addr      = get_addr("config_mvout_rs1_reg_addr");
    config_mvout_rs2_reg_addr      = get_addr("config_mvout_rs2_reg_addr");
    config_compute_rs1_reg_addr    = get_addr("config_compute_rs1_reg_addr");
    config_compute_rs2_reg_addr    = get_addr("config_compute_rs2_reg_addr");
    config_sfu_rs1_reg_addr        = get_addr("config_sfu_rs1_reg_addr");
    config_sfu_rs2_reg_addr        = get_addr("config_sfu_rs2_reg_addr");
    config_accumulate_rs1_reg_addr = get_addr("config_accumulate_rs1_reg_addr");
    config_accumulate_rs2_reg_addr = get_addr("config_accumulate_rs2_reg_addr");
    compute_sa_rs1_reg_addr        = get_addr("compute_sa_rs1_reg_addr");
    compute_sa_rs2_reg_addr        = get_addr("compute_sa_rs2_reg_addr");
    compute_matadd_rs1_reg_addr    = get_addr("compute_matadd_rs1_reg_addr");
    compute_matadd_rs2_reg_addr    = get_addr("compute_matadd_rs2_reg_addr");
    compute_matvec_rs1_reg_addr    = get_addr("compute_matvec_rs1_reg_addr");
    compute_matvec_rs2_reg_addr    = get_addr("compute_matvec_rs2_reg_addr");
    compute_sfu_rs1_reg_addr       = get_addr("compute_sfu_rs1_reg_addr");
    compute_sfu_rs2_reg_addr       = get_addr("compute_sfu_rs2_reg_addr");
    start_reg_addr                 = get_addr("start_reg_addr");
    iar_reg_addr                   = get_addr("iar_reg_addr");
    mer_reg_addr                   = get_addr("mer_reg_addr");
    ier_reg_addr                   = get_addr("ier_reg_addr");
    isr_reg_addr                   = get_addr("isr_reg_addr");
    ipr_reg_addr                   = get_addr("ipr_reg_addr");

    // 初始化寄存器值
    for (const auto& pair : cfg.reg_addr_map) {
        std::string name = pair.first;
        uint32_t addr = pair.second;

        regs_map[addr] = 0;
        addr_name_map[addr] = name;
    }

    std::cout << "[RegFile] Loaded " << regs_map.size() << " registers from config." << std::endl;
}

void RegFile::write(uint32_t addr, uint64_t data) {
    // 如果 addr 不在 map 中，报错
    auto it = regs_map.find(addr);
    if (it == regs_map.end()) {
        throw std::runtime_error("[RegFile Error] Write Access Violation! Addr: 0x" + std::to_string(addr));
        return;
    }
    it->second = data;
    // std::cout << "Write 0x" << std::hex << addr << " <= " << data << std::dec << std::endl;
}

uint64_t RegFile::read(uint32_t addr) const {
    // 如果 addr 不在 map 中，报错
    auto it = regs_map.find(addr);
    if (it == regs_map.end()) {
        throw std::runtime_error("[RegFile Error] Read Access Violation! Addr: 0x" + std::to_string(addr));
        return 0;
    }
    
    return it->second;
}

void RegFile::dump() const {
    std::cout << "=== RegFile Dump ===" << std::endl;
    for (auto const& [addr, val] : regs_map) {
        if (val != 0) {
            std::string name = "Unknown";
            auto name_it = addr_name_map.find(addr);
            if (name_it != addr_name_map.end()) {
                name = name_it->second;
            }

            // 格式化打印：地址 | 名字 | 值
            std::cout << "Addr: 0x" << std::setw(4) << std::setfill('0') << std::hex << addr 
                      << " | " << std::setw(30) << std::setfill(' ') << std::left << name // 左对齐名字
                      << " | Val: 0x" << std::hex << val << std::dec << std::endl;
        }
    }
    std::cout << "====================" << std::endl;
}