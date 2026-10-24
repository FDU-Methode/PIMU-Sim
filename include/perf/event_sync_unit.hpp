#ifndef _EVENT_SYNC_UNIT_HPP__
#define _EVENT_SYNC_UNIT_HPP__

#include <cstdint>
#include <string>
#include <vector>
#include "../config.hpp"
#include "../logger.hpp"

struct DMAInstPacket;
struct SAInstPacket;
struct SFUInstPacket;

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

    // Check Event Sync Unit Drained
    bool isDrained() const noexcept {
        for(bool busy : bank_busy) {
            if(busy) return false;
        }

        return true;
    }

    int getBankId(uint32_t addr, bool is_acc);

    bool allowIssue(const DMAInstPacket& inst_packet);
    bool allowIssue(const SAInstPacket& inst_packet);
    bool allowIssue(const SFUInstPacket& inst_packet);

    void issueUpdate(const DMAInstPacket& inst_packet);
    void issueUpdate(const SAInstPacket& inst_packet);
    void issueUpdate(const SFUInstPacket& inst_packet);

    void retireUpdate(const DMAInstPacket& inst_packet);
    void retireUpdate(const SAInstPacket& inst_packet);
    void retireUpdate(const SFUInstPacket& inst_packet);

    // bool updateValid(int bit, bool status);
    // bool updateBusy(int bit, bool status);

};

#endif