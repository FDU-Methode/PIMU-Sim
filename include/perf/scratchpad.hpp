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

    enum State{
        IDLE,
        BUSY
    };
    State read_state  = IDLE;
    State write_state = IDLE;

public:
    Scratchpad(Logger& logger, const Config& cfg);
    virtual ~Scratchpad() = default;

    bool isReadBusy() const { return read_state == BUSY; };
    bool isWriteBusy() const { return write_state == BUSY; };
    void setReadIdle() { read_state = IDLE; };
    void setWriteIdle() { write_state = IDLE; };

    bool ReadMemory(uint32_t addr);
    bool WriteMemory(uint32_t addr);
};

#endif