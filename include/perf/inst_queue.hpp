#ifndef _INST_QUEUE_HPP__
#define _INST_QUEUE_HPP__

#include <cstdint>
#include <queue>
#include <string>
#include <vector>
#include "../util.hpp"
#include "../config.hpp"
#include "../logger.hpp"
#include "../regfile.hpp"
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
    RegFile& regfile;
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
    InstQueue(const std::string& queue_name, InstQueueType queue_type, int inst_width, int queue_depth, int inst_count_upper, PerfNpuCore* core, Logger& logger, RegFile& regfile, EventSyncUnit* event_sync_unit) : 
            queue_name(queue_name), queue_type(queue_type), inst_width(inst_width), queue_depth(queue_depth), inst_count_upper(inst_count_upper), core(core), logger(logger), regfile(regfile), event_sync_unit(event_sync_unit) {};

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

    virtual void writeReg(InstPacketT& inst_packet) = 0;

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

public:
    DMAInstQueue(PerfNpuCore* core, Logger& logger, RegFile& regfile, const Config& cfg, EventSyncUnit* event_sync_unit) : 
                InstQueue<DMAInstPacket>("DMAInstQueue", InstQueueType::DMA, 256, cfg.inst_queue_depth, 2, core, logger, regfile, event_sync_unit) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void writeReg(DMAInstPacket& inst_packet) override;
    void inst_retire() override;
};




class SAInstQueue : public InstQueue<SAInstPacket> {
private:
    SAInstPacket inst_packet{};

public:
    SAInstQueue(PerfNpuCore* core, Logger& logger, RegFile& regfile, const Config& cfg, EventSyncUnit* event_sync_unit) : 
                InstQueue<SAInstPacket>("SAInstQueue", InstQueueType::SA, 384, cfg.inst_queue_depth, 3, core, logger, regfile, event_sync_unit) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void writeReg(SAInstPacket& inst_packet) override;
    void inst_retire() override;
};




class SFUInstQueue : public InstQueue<SFUInstPacket> {
private:
    SFUInstPacket inst_packet{};

public:
    SFUInstQueue(PerfNpuCore* core, Logger& logger, RegFile& regfile, const Config& cfg, EventSyncUnit* event_sync_unit) : 
                InstQueue<SFUInstPacket>("SFUInstQueue", InstQueueType::SFU, 256, cfg.inst_queue_depth, 2, core, logger, regfile, event_sync_unit) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override;
    bool inst_issue() override;
    void writeReg(SFUInstPacket& inst_packet) override;
    void inst_retire() override;
};


// class InstIssueArbiter {
// public:
//     // 优先级：SA > SFU > DMA
//     static bool issue_by_priority(SAInstQueue& sa_queue,
//                                   SFUInstQueue& sfu_queue,
//                                   DMAInstQueue& dma_queue,
//                                   SAInstPacket& sa_inst,
//                                   SFUInstPacket& sfu_inst,
//                                   DMAInstPacket& dma_inst,
//                                   InstType& issued_type) {
//         issued_type = InstType::NONE;

//         if (sa_queue.inst_issue(sa_inst)) {
//             issued_type = InstType::SA;
//             return true;
//         }
//         if (sfu_queue.inst_issue(sfu_inst)) {
//             issued_type = InstType::SFU;
//             return true;
//         }
//         if (dma_queue.inst_issue(dma_inst)) {
//             issued_type = InstType::DMA;
//             return true;
//         }
//         return false;
//     }
// };


#endif