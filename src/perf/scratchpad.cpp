#include "../../include/perf/scratchpad.hpp"

Scratchpad::Scratchpad(Logger& logger, const Config& cfg) :
    cfg(cfg), logger(logger) {

    sram_bytes = cfg.sram_bytes;
    sram_ports = cfg.sram_ports;
}

bool Scratchpad::ReadMemory(uint32_t addr) {
    if(addr + (cfg.systolic_array_size * cfg.sram_data_width / 8) > static_cast<uint32_t>(sram_bytes)) {
        logger.log("[SPM][ERROR] Read Address Exceeds The Boundary");
        return false;
    }
    if(read_state == BUSY) {
        logger.log("[SPM][ERROR] Read Port Conflict");
        return false;
    }
    read_state = BUSY;
    return true;
}

bool Scratchpad::WriteMemory(uint32_t addr) {
    if(addr + (cfg.systolic_array_size * cfg.sram_data_width / 8) > static_cast<uint32_t>(sram_bytes)) {
        logger.log("[SPM][ERROR] Write Address Exceeds The Boundary");
        return false;
    }
    if(write_state == BUSY) {
        logger.log("[SPM][ERROR] Write Port Confilct");
        return false;
    }
    write_state = BUSY;
    return true;
}