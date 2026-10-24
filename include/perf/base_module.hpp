#ifndef __BASE_MODULE_HPP__
#define __BASE_MODULE_HPP__

#include <stdexcept>
#include "../util.hpp"
#include "../config.hpp"
#include "../logger.hpp"

class PerfSim;

class BaseModule {
protected:
    PerfSim *sim;
    Config cfg;
    Logger &logger;

    Stats _stats;
    std::uint64_t _cycle;

public:
    BaseModule(PerfSim *sim, const Config& cfg, Logger& logger) 
    : sim(sim), cfg(cfg), logger(logger), _cycle(0) {};
    virtual ~BaseModule() = default;

    // 基本周期推进接口
    virtual void tick() = 0;

    // 模块间通信接口
    virtual void sendMsg(Message& msg) = 0;
    virtual bool recvMsg(Message& msg) = 0;
    virtual bool tryConsumeResp(MsgType expectedMsg, Message& getMsg) = 0;

    // 统计接口
    Stats getStats() const { return _stats; }
    //virtual std::map<std::string, std::string> getStats() const = 0;
    // 获取当前周期
    std::uint64_t getCycle() const noexcept { return _cycle; }
};

#endif