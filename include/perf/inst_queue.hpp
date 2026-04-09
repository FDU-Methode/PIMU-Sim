#ifndef _INST_QUEUE_HPP__
#define _INST_QUEUE_HPP__

#include <cstdint>
#include <queue>
#include <string>
#include <vector>
#include "../util.hpp"
#include "../config.hpp"
#include "../logger.hpp"
#include "inst_cache.hpp"
#include "event_sync_unit.hpp"

class PerfNpuCore;

enum class InstQueueState {
    IDLE, BUSY
};

struct DMAInstPacket {
    uint8_t opcode = 0;
    Inst inst0{};
    Inst inst1{};
    uint64_t inst_id = 0;
};

struct SAInstPacket {
    uint8_t opcode = 0;
    Inst inst0{};
    Inst inst1{};
    Inst inst2{};
    uint64_t inst_id = 0;
};

struct SFUInstPacket {
    uint8_t opcode = 0;
    Inst inst0{};
    Inst inst1{};
    uint64_t inst_id = 0;
};


template <typename InstPacketT>
class InstQueue {
protected:
    PerfNpuCore* core;
    Logger& logger;
    EventSyncUnit* event_sync_unit;

    std::string queue_name;
    InstQueueType queue_type;
    InstQueueState queue_state = InstQueueState::IDLE;

    int inst_width = 0;
    int queue_depth = 0;

    int current_inst_num = 0;
    int inst_count = 0;
    int inst_count_upper = 0;

    uint64_t inst_id_counter = 0;

    std::queue<InstPacketT> inst_queue;

    InstPacketT inst_buffer;  // Store Current Issued Inst

public:
    InstQueue(PerfNpuCore* core, Logger& logger, EventSyncUnit* event_sync_unit, const std::string& queue_name, InstQueueType queue_type, int inst_width, int queue_depth, int inst_count_upper) : 
            core(core), logger(logger), event_sync_unit(event_sync_unit), queue_name(queue_name), queue_type(queue_type), inst_width(inst_width), queue_depth(queue_depth), inst_count_upper(inst_count_upper) {};

    virtual ~InstQueue() = default;
    
    virtual bool in_queue(const Inst& inst, uint8_t opcode) = 0;

    virtual bool out_queue(InstPacketT& inst_packet) {
        if(inst_queue.empty()) {
            logger.log("[%s] Out Queue Failed: Queue is Empty", queue_name.c_str());
            return false;
        }

        inst_packet = inst_queue.front();
        inst_buffer = inst_packet;
        inst_queue.pop();
        current_inst_num--;

        logger.log("[%s] Out Queue Success: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", 
                    queue_name.c_str(), (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
        return true;
    };

    virtual bool inst_issue() = 0;

    virtual void inst_retire() = 0;

    virtual void clear() {
        current_inst_num = 0;
        inst_count = 0;
        inst_id_counter = 0;
        queue_state = InstQueueState::IDLE;
        std::queue<InstPacketT> empty_queue;
        inst_queue.swap(empty_queue);
    };

    // Status Function
    bool empty() { return current_inst_num == 0; }
    bool full() { return current_inst_num >= queue_depth; }
    bool isIdle() { return queue_state == InstQueueState::IDLE; }
    bool isBusy() { return queue_state == InstQueueState::BUSY; }

    const InstPacketT& getInstBuffer() { return inst_buffer; }

    int getInstWidth() const { return inst_width; }
    int getQueueDepth() const { return queue_depth; }
    int getCurrentInstNum() const { return current_inst_num; }
    int getInstCount() const { return inst_count; }
    int getInstCountUpper() const { return inst_count_upper; }

protected:
    void inst_push(const InstPacketT& inst_packet) {
        inst_queue.push(inst_packet);
        current_inst_num++;
        inst_count = 0;
    }

};




class DMAInstQueue : public InstQueue<DMAInstPacket> {
private:
    DMAInstPacket inst_packet;
    int queue_id = 0;

public:
    DMAInstQueue(PerfNpuCore* core, Logger& logger, const Config& cfg, EventSyncUnit* event_sync_unit, int queue_id) : 
                InstQueue<DMAInstPacket>(core, logger, event_sync_unit, "DMAInstQueue", InstQueueType::DMA, 256, cfg.inst_queue_depth, 2), queue_id(queue_id) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void inst_retire() override;
};




class SAInstQueue : public InstQueue<SAInstPacket> {
private:
    SAInstPacket inst_packet{};

public:
    SAInstQueue(PerfNpuCore* core, Logger& logger, const Config& cfg, EventSyncUnit* event_sync_unit) : 
                InstQueue<SAInstPacket>(core, logger, event_sync_unit, "SAInstQueue", InstQueueType::SA, 384, cfg.inst_queue_depth, 3) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void inst_retire() override;
};




class SFUInstQueue : public InstQueue<SFUInstPacket> {
private:
    SFUInstPacket inst_packet{};

public:
    SFUInstQueue(PerfNpuCore* core, Logger& logger, const Config& cfg, EventSyncUnit* event_sync_unit) : 
                InstQueue<SFUInstPacket>(core, logger, event_sync_unit, "SFUInstQueue", InstQueueType::SFU, 256, cfg.inst_queue_depth, 2) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void inst_retire() override;
};


#endif