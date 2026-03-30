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
    uint64_t inst_id = 0;
};


template <typename InstPacketT>
class InstQueue {
protected:
    Logger& logger;

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

public:
    InstQueue(const std::string& queue_name, InstQueueType queue_type, int inst_width, int queue_depth, int inst_count_upper, Logger& logger) : 
            queue_name(queue_name), queue_type(queue_type), inst_width(inst_width), queue_depth(queue_depth), inst_count_upper(inst_count_upper), logger(logger) {};

    virtual ~InstQueue() = default;
    
    virtual bool in_queue(const Inst& inst, uint8_t opcode) = 0;

    virtual bool out_queue(InstPacketT& inst_packet) {
        if(inst_queue.empty()) {
            logger.log("[%s] Out Queue Failed: Queue is Empty", queue_name.c_str());
            return false;
        }

        inst_packet = inst_queue.front();
        inst_queue.pop();
        current_inst_num--;

        logger.log("[%s] Out Queue Success: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", 
                    queue_name.c_str(), (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
        return true;
    };

    virtual bool inst_issue(InstPacketT& inst_packet) {
        if(queue_state == InstQueueState::BUSY) {
            logger.log("[%s] Inst Issue Skipped: queue is busy", queue_name.c_str());
            return false;
        }

        bool sync_allow_issue = true;  // TODO: 查询同步控制模块
        if(!sync_allow_issue) {
            logger.log("[%s] Inst Issue Blocked by Event Sync Unit", queue_name.c_str());
            return false;
        }

        if(!out_queue(inst_packet)) {
            return false;
        }

        queue_state = InstQueueState::BUSY;
        logger.log("[%s] Inst Issue Success: inst_id=%llu, opcode=0x%02x", queue_name.c_str(), (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

        // TODO: 写对应配置寄存器， 启动首个功能模块

        return true;
    };

    virtual void inst_retire() {
        queue_state = InstQueueState::IDLE;
        logger.log("[%s] Inst Retire: state -> IDLE", queue_name.c_str());

        // TODO: 写同步控制模块/写中断
    };

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
    DMAInstQueue(Logger& logger, const Config& cfg) : 
                InstQueue<DMAInstPacket>("DMAInstQueue", InstQueueType::DMA, 256, cfg.inst_queue_depth, 2, logger) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override {
        if(full()) {
            logger.log("[%s] In Queue Blocked: queue full", queue_name.c_str());
            return false;
        }

        if(inst_count == 0) {
            inst_packet = DMAInstPacket{};
            inst_packet.opcode = opcode;
            inst_packet.inst_id = inst_id_counter++;
            inst_packet.inst0 = inst;
        } else if(inst_count == 1) {
            inst_packet.inst1 = inst;
        } else {
            throw std::runtime_error(queue_name + ": invalid inst_count");
        }

        inst_count++;
        logger.log("[%s] In Queue: word=0x%016llx%016llx, inst_count=%d/%d", queue_name.c_str(), (unsigned long long)inst.high, 
                    (unsigned long long)inst.low, inst_count, inst_count_upper);

        if(inst_count == inst_count_upper) {
            inst_push(inst_packet);
            logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                        (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
            inst_packet = DMAInstPacket{};
        }

        return true;
    }
};


class SAInstQueue : public InstQueue<SAInstPacket> {
private:
    SAInstPacket inst_packet{};

public:
    SAInstQueue(Logger& logger, const Config& cfg) : 
                InstQueue<SAInstPacket>("SAInstQueue", InstQueueType::SA, 384, cfg.inst_queue_depth, 3, logger) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override {
        if(full()) {
            logger.log("[%s] In Queue Blocked: queue full", queue_name.c_str());
            return false;
        }

        if(inst_count == 0) {
            inst_packet = SAInstPacket{};
            inst_packet.opcode = opcode;
            inst_packet.inst_id = inst_id_counter++;
            inst_packet.inst0 = inst;
        } else if(inst_count == 1) {
            inst_packet.inst1 = inst;
        } else if(inst_count == 2) {
            inst_packet.inst2 = inst;
        } else {
            throw std::runtime_error(queue_name + ": invalid inst_count");
        }

        inst_count++;
        logger.log("[%s] In Queue : word=0x%016llx%016llx, inst_count=%d/%d", queue_name.c_str(), (unsigned long long)inst.high, 
                    (unsigned long long)inst.low, inst_count, inst_count_upper);

        if(inst_count == inst_count_upper) {
            inst_push(inst_packet);
            logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                        (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
            inst_packet = SAInstPacket{};
        }

        return true;
    }
};


class SFUInstQueue : public InstQueue<SFUInstPacket> {
private:
    SFUInstPacket inst_packet{};

public:
    SFUInstQueue(Logger& logger, const Config& cfg) : 
                InstQueue<SFUInstPacket>("SFUInstQueue", InstQueueType::SFU, 128, cfg.inst_queue_depth, 1, logger) {};

    bool in_queue(const Inst& inst, uint8_t opcode) override {
        if(full()) {
            logger.log("[%s] In Queue Blocked: queue full", queue_name.c_str());
            return false;
        }

        inst_packet = SFUInstPacket{};
        inst_packet.opcode = opcode;
        inst_packet.inst_id = inst_id_counter++;
        inst_packet.inst0 = inst;

        inst_count++;
        logger.log("[%s] In Queue : word=0x%016llx%016llx, inst_count=%d/%d", queue_name.c_str(), (unsigned long long)inst.high, 
                    (unsigned long long)inst.low, inst_count, inst_count_upper);

        if(inst_count == inst_count_upper) {
            inst_push(inst_packet);
            logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                        (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
            inst_packet = SFUInstPacket{};
        }

        return true;
    }
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