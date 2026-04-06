#include "../../include/perf/scratchpad.hpp"

Scratchpad::Scratchpad(Logger& logger, const Config& cfg) :
    cfg(cfg), logger(logger) {

    sram_bank_num = cfg.sram_bank_num;
    sram_bank_bytes = cfg.sram_bank_bytes;
    sram_data_width = cfg.sram_data_width;
    sram_data_num = cfg.systolic_array_size;

    bank_state.assign(static_cast<size_t>(sram_bank_num), IDLE);
}

bool Scratchpad::isBusy() {
    for(State state : bank_state) {
        if(state == BUSY) {
            return true;
        }
    }
    return false;
}

void Scratchpad::clearPorts() {
    for(State& state : bank_state) {
        state = IDLE;
    }
}

bool Scratchpad::access(uint32_t addr) {
    uint32_t access_bytes = static_cast<uint32_t>(sram_data_num * sram_data_width / 8);
    uint32_t total_bytes = static_cast<uint32_t>(sram_bank_num * sram_bank_bytes);
    if(addr + access_bytes > total_bytes) {
        logger.log("[SPM][ERROR] Address Exceeds Boundary: addr=0x%x", addr);
        return false;
    }

    int start_bank = static_cast<int>(addr / sram_bank_bytes);
    int end_bank = static_cast<int>((addr + access_bytes - 1) / sram_bank_bytes);
    if(start_bank != end_bank) {
        logger.log("[SPM][ERROR] Access Crosses Bank Boundary: addr=0x%x", addr);
        return false;
    }

    if(bank_state[start_bank] == BUSY) {
        logger.log("[SPM][ERROR] Bank %d Conflict", start_bank);
        return false;
    }

    bank_state[start_bank] = BUSY;
    return true;
}

bool Scratchpad::ReadMemory(uint32_t addr) {
    return access(addr);
}

bool Scratchpad::WriteMemory(uint32_t addr) {
    return access(addr);
}