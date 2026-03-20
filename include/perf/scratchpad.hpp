#ifndef _SCRATCHPAD_HPP__
#define _SCRATCHPAD_HPP__

#include "../util.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"

class Scratchpad{
protected:
    Config cfg;
    Logger& logger;

    int sram_bytes = 0;
    int sram_ports = 0;
    int sram_data_width = 0;
    int sram_data_num = 0;

    enum State{
        IDLE,
        BUSY
    };

    std::array<State, 2> port_state{{IDLE, IDLE}};

public:
    Scratchpad(Logger& logger, const Config& cfg);
    Scratchpad(Logger& logger, const Config& cfg, int sram_bytes, int sram_ports, int sram_data_width, int sram_data_num);
    virtual ~Scratchpad() = default;

    bool isBusy()  { return (port_state[0] == BUSY) || (port_state[1] == BUSY); }
    bool isReadBusy() { return isBusy(); }
    bool isWriteBusy() { return isBusy(); }

    void clearPorts() { port_state[0] = IDLE; port_state[1] = IDLE; }

    bool accessPort(int port, uint32_t addr);
    bool ReadMemory(uint32_t addr);
    bool WriteMemory(uint32_t addr);
};

#endif