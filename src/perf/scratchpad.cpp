#include "../../include/perf/scratchpad.hpp"

Scratchpad::Scratchpad(Logger& logger, const Config& cfg) :
    cfg(cfg), logger(logger) {

    sram_bytes = cfg.sram_bytes;
    sram_ports = cfg.sram_ports;
}

bool Scratchpad::accessPort(int port, uint32_t addr) {
    const uint32_t access_bytes = cfg.systolic_array_size * cfg.sram_data_width / 8;
    if((port < 0) || (port >= 2)) {
        logger.log("[SPM][ERROR] Invalid Port Index %d", port);
        return false;
    }
    if(addr + access_bytes > static_cast<uint32_t>(sram_bytes)) {
        logger.log("[SPM][ERROR] Address Exceeds Boundary: addr=0x%x bytes=%u", addr, access_bytes);
        return false;
    }
    if(port_state[port] == BUSY) {
        logger.log("[SPM][ERROR] Port %d Conflict", port);
        return false;
    }
    port_state[port] = BUSY;
    return true;
}

bool Scratchpad::ReadMemory(uint32_t addr) {  // Default Read Port 0
    return accessPort(0, addr);
}

bool Scratchpad::WriteMemory(uint32_t addr) {  // Default Write Port 1
    return accessPort(1, addr);
}