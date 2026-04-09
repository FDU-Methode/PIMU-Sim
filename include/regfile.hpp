#ifndef REGFILE_HPP
#define REGFILE_HPP

#include <map>
#include <iostream>
#include <vector>
#include <iomanip>
#include "config.hpp"

class RegFile {
private:
    std::map<uint32_t, uint64_t> regs_map;
    // 用于调试打印寄存器名
    std::map<uint32_t, std::string> addr_name_map;

public:
    RegFile() = delete;

    RegFile(const Config& cfg);

    // --- 写接口 ---
    void write(uint32_t addr, uint64_t data);

    // --- 读接口 ---
    uint64_t read(uint32_t addr) const;

    // --- 调试打印接口 ---
    void dump() const;

    // --- 寄存器地址 ---
    uint32_t iar_reg_addr;
    uint32_t mer_reg_addr;
    uint32_t ier_reg_addr;
    uint32_t isr_reg_addr;
    uint32_t ipr_reg_addr;
};

#endif