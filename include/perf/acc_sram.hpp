#ifndef _ACC_SRAM_HPP__
#define _ACC_SRAM_HPP__

#include "../util.hpp"
#include "../logger.hpp"
#include "../config.hpp"

class AccSram{
protected:
    Config cfg;
    Logger& logger;

    int bias_bytes = 0;
    int data_bytes = 0;
    int data_width = 0;
    int data_num = 0;

    enum State{
        IDLE,
        BUSY
    };

    std::array<State, 2> bias_port_state{{IDLE, IDLE}};
    std::array<State, 2> data_port_state{{IDLE, IDLE}};

public:
    AccSram(Logger& logger, const Config& cfg);
    ~AccSram() = default;

    bool isBusy();
    bool isBiasBusy();
    bool isDataBusy();

    void clearPorts();

    bool accessBiasPort(int port, uint32_t addr);
    bool accessDataPort(int port, uint32_t addr);

    bool ReadBias(uint32_t addr);
    bool WriteBias(uint32_t addr);

    bool ReadData(uint32_t addr);
    bool WriteData(uint32_t addr);
};

#endif