#ifndef _INST_HPP__
#define _INST_HPP__

#include <iostream>

struct Inst {
    std::string op_name; // mvin/config_compute/...
    uint32_t reg_addr;
    uint64_t inst_data;
};

#endif