#ifndef _SCRATCHPAD_HPP__
#define _SCRATCHPAD_HPP__

#include "../util.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"

class Scratchpad{
protected:
    Config cfg;
    Logger& logger;

    int sram_bank_num = 0;
    int sram_bank_bytes = 0;
    int sram_data_width = 0;
    int sram_data_num = 0;

    enum State{
        IDLE,
        BUSY
    };

    std::vector<State> bank_state;

public:
    Scratchpad(Logger& logger, const Config& cfg);
    ~Scratchpad() = default;

    bool isBusy();

    void clearPorts();

    bool access(uint32_t addr);
    bool ReadMemory(uint32_t addr);
    bool WriteMemory(uint32_t addr);
};

#endif