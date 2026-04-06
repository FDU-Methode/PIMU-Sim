#include "../../include/perf/acc_sram.hpp"

AccSram::AccSram(Logger& logger, const Config& cfg) :
    cfg(cfg), logger(logger) {

    bias_bytes = cfg.bias_bytes;
    data_bytes = cfg.acc_bytes;
    data_width = cfg.acc_data_width;
    data_num = cfg.systolic_array_size;
}

bool AccSram::isBusy() {
    return isBiasBusy() || isDataBusy();
}

bool AccSram::isBiasBusy() {
    return (bias_port_state[0] == BUSY) || (bias_port_state[1] == BUSY);
}

bool AccSram::isDataBusy() {
    return (data_port_state[0] == BUSY) || (data_port_state[1] == BUSY);
}

void AccSram::clearPorts() {
    bias_port_state[0] = IDLE;
    bias_port_state[1] = IDLE;
    data_port_state[0] = IDLE;
    data_port_state[1] = IDLE;
}

bool AccSram::accessBiasPort(int port, uint32_t addr) {
    uint32_t access_bytes = static_cast<uint32_t>(data_num * data_width / 8);

    if((port < 0) || (port >= 2)) {
        logger.log("[ACC SRAM][ERROR] Invalid Bias Port Index %d", port);
        return false;
    }

    if(addr + access_bytes > static_cast<uint32_t>(bias_bytes)) {
        logger.log("[ACC SRAM][ERROR] Bias Address Exceeds Boundary: addr=0x%x bytes=%u", addr, access_bytes);
        return false;
    }

    if(bias_port_state[port] == BUSY) {
        logger.log("[ACC SRAM][ERROR] Bias Port %d Conflict", port);
        return false;
    }

    bias_port_state[port] = BUSY;
    return true;
}

bool AccSram::accessDataPort(int port, uint32_t addr) {
    uint32_t access_bytes = static_cast<uint32_t>(data_num * data_width / 8);

    if((port < 0) || (port >= 2)) {
        logger.log("[ACC SRAM][ERROR] Invalid Data Port Index %d", port);
        return false;
    }

    if(addr + access_bytes > static_cast<uint32_t>(data_bytes)) {
        logger.log("[ACC SRAM][ERROR] Data Address Exceeds Boundary: addr=0x%x bytes=%u", addr, access_bytes);
        return false;
    }

    if(data_port_state[port] == BUSY) {
        logger.log("[ACC SRAM][ERROR] Data Port %d Conflict", port);
        return false;
    }

    data_port_state[port] = BUSY;
    return true;
}

bool AccSram::ReadBias(uint32_t addr) {
    return accessBiasPort(0, addr);
}

bool AccSram::WriteBias(uint32_t addr) {
    return accessBiasPort(1, addr);
}

bool AccSram::ReadData(uint32_t addr) {
    return accessDataPort(0, addr);
}

bool AccSram::WriteData(uint32_t addr) {
    return accessDataPort(1, addr);
}