#include "../../include/perf/scratchpad.hpp"

Scratchpad::Scratchpad(Logger& logger, const Config& cfg) :
    cfg(cfg), logger(logger) {

    sram_bytes = cfg.sram_bytes;
    sram_ports = cfg.sram_ports;
    sram_data_width = cfg.sram_data_width;
    sram_data_num = cfg.systolic_array_size;
}

Scratchpad::Scratchpad(Logger& logger, const Config& cfg, int sram_bytes, int sram_ports, int sram_data_width, int sram_data_num) :
    cfg(cfg), logger(logger), sram_bytes(sram_bytes), sram_ports(sram_ports), sram_data_width(sram_data_width), sram_data_num(sram_data_num) {}

bool Scratchpad::accessPort(int port, uint32_t addr) {
    const uint32_t access_bytes = sram_data_num * sram_data_width / 8;
    if((port < 0) || (port >= sram_ports)) {
        logger.log("[SPM][ERROR] Invalid Port Index %d", port);
        return false;
    }
    // if(addr + access_bytes > static_cast<uint32_t>(sram_bytes)) {
    //     logger.log("[SPM][ERROR] Address Exceeds Boundary: addr=0x%x bytes=%u", addr, access_bytes);
    //     return false;
    // }
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