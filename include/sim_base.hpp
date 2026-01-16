#ifndef __SIM_BASE_HPP__
#define __SIM_BASE_HPP__

#include <iostream>
#include <memory>
#include <chrono>
#include "config.hpp"
#include "logger.hpp"
#include "util.hpp"

class SimBase{
public:
    SimBase(const Config& cfg): cfg_(cfg), 
                                logger_(cfg.log_file, cfg.enable_log) {};
    virtual ~SimBase() noexcept = default;

    virtual void run() = 0;

protected:
    Config cfg_;
    Logger logger_;
};

#endif