#ifndef _EVENT_SYNC_UNIT_HPP__
#define _EVENT_SYNC_UNIT_HPP__

#include <cstdint>
#include <string>
#include <vector>
#include "../config.hpp"
#include "../logger.hpp"
#include "inst_queue.hpp"

class EventSyncUnit {
private:
    const Config& cfg;
    Logger& logger;

    int spm_bank_num = 0;
    int spm_bank_bytes = 0;
    int acc_bank_num = 0;
    int bias_bank_num = 0;
    int acc_bytes = 0;

    std::vector<bool> bank_valid;
    std::vector<bool> bank_busy;

public:
    EventSyncUnit(const Config& cfg, Logger& logger);
    ~EventSyncUnit() = default;

    int getBankId(uint32_t addr, bool is_acc);

    bool allowIssue(const DMAInstPacket& inst_packet);
    bool allowIssue(const SAInstPacket& inst_packet);
    bool allowIssue(const SFUInstPacket& inst_packet);

    bool updateValid(int bit, bool status);
    bool updateBusy(int bit, bool status);

};

#endif