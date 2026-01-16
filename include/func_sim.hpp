#ifndef __FUNC_SIM_HPP__
#define __FUNC_SIM_HPP__

#include "sim_base.hpp"

class FuncSim : public SimBase{
private:
    std::unique_ptr<SimBase> core;
    std::unique_ptr<SimBase> ddr;
public:
    FuncSim(const Config& cfg);
    ~FuncSim() noexcept = default;

    void run() override;
};

#endif