#include <cassert>
#include "../../include/perf/inst_queue.hpp"
#include "../../include/perf/perf_npu_core.hpp"

////////////////////////////
//     DMA Inst Queue     //
////////////////////////////

bool DMAInstQueue::in_queue(const Inst& inst, uint8_t opcode) {
    if(full()) {
        logger.log("[%s] In Queue Blocked: queue full", queue_name.c_str());
        return false;
    }

    if(inst_count == 0) {
        inst_packet = DMAInstPacket{};
        inst_packet.opcode = opcode;
        inst_packet.inst_id = inst_id_counter++;
        inst_packet.dma_id = queue_id;
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
        core->addInstNum();
        logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                    (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
        inst_packet = DMAInstPacket{};
    }

    return true;
}


bool DMAInstQueue::inst_issue() {
    if(queue_state == InstQueueState::BUSY) {
        logger.log("[DMAInstQueue] Inst Issue Skipped: queue is busy");
        return false;
    }

    if(inst_queue.empty()) {
        logger.log("[DMAInstQueue] Inst Issue Skipped: queue is empty");
        return false;
    }

    DMAInstPacket inst_packet = inst_queue.front();
    bool sync_allow_issue = event_sync_unit->allowIssue(inst_packet);
    if(!sync_allow_issue) {
        Inst inst0 = inst_packet.inst0;
        uint8_t opcode = inst_packet.opcode;
        bool is_acc = (inst0.low >> 13) & 0x1;
        bool is_bias = (inst0.low >> 14) & 0x1;
        uint32_t sram_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
        uint32_t dram_addr = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);
        logger.log("[DMAInstQueue] Inst Issue Blocked by Event Sync Unit: "
                   "queue_id=%d inst_id=%llu opcode=0x%02x is_acc=%d is_bias=%d sram_addr=0x%08x dram_addr=0x%08x",
                   queue_id, (unsigned long long)inst_packet.inst_id, opcode, is_acc, is_bias, sram_addr, dram_addr);
        return false;
    }

    if(!out_queue(inst_packet)) {
        return false;
    }

    queue_state = InstQueueState::BUSY;
    logger.log("[DMAInstQueue] Inst Issue Success: inst_id=%llu, opcode=0x%02x", (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

    // Update Event Sync Unit
    event_sync_unit->issueUpdate(inst_packet);
    // Start the First Functional Unit
    DMA* dma = nullptr;
    switch(queue_id) {
        case 1: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA1)); break;
        case 2: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA2)); break;
        case 3: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA3)); break;
        case 4: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA4)); break;
        case 5: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA5)); break;
        case 6: dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA6)); break;
        default: throw std::runtime_error("[DMAInstQueue] No Valid Queue ID");
    }
    dma->execute(inst_packet.opcode);

    return true;

};


void DMAInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[DMAInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
    RegFile& regfile = core->getRegFile();
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    if(inst_buffer.opcode == 0x01) {
        isr_val |= (1ULL << 0);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMAInstQueue] ISR Reg[0] Updated");
    } else if(inst_buffer.opcode == 0x02) {
        isr_val |= (1ULL << 1);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[DMAInstQueue] ISR Reg[1] Updated");
    } else {
        throw std::runtime_error("DMAInstQueue: unsupported opcode in inst_retire");
    }

    inst_buffer = DMAInstPacket{};  // Clear Inst Buffer
};

///////////////////////////
//     SA Inst Queue     //
///////////////////////////

bool SAInstQueue::in_queue(const Inst& inst, uint8_t opcode) {
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
        core->addInstNum();
        logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                    (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
        inst_packet = SAInstPacket{};
    }

    return true;
}


bool SAInstQueue::inst_issue() {
    if(queue_state == InstQueueState::BUSY) {
        logger.log("[SAInstQueue] Inst Issue Skipped: queue is busy");
        return false;
    }

    if(inst_queue.empty()) {
        logger.log("[SAInstQueue] Inst Issue Skipped: queue is empty");
        return false;
    }

    SAInstPacket inst_packet = inst_queue.front();
    bool sync_allow_issue = event_sync_unit->allowIssue(inst_packet);
    if(!sync_allow_issue) {
        Inst inst1 = inst_packet.inst1;
        Inst inst2 = inst_packet.inst2;

        bool is_acc = (inst1.low >> 6) & 0x1;
        bool is_bias = (inst1.low >> 11) & 0x1;
        bool acc_dest = (inst1.low >> 3) & 0x1;
        bool a_keep = (inst1.low >> 4) & 0x1;
        bool b_keep = (inst1.low >> 5) & 0x1;
        bool bias_keep = (inst1.low >> 12) & 0x1;
        uint32_t biaspsum_addr = static_cast<uint32_t>(inst1.high & 0xFFFFFFFF);
        uint32_t output_addr = static_cast<uint32_t>((inst1.high >> 32) & 0xFFFFFFFF);
        uint32_t a_addr = static_cast<uint32_t>(inst2.low & 0xFFFFFFFF);
        uint32_t b_addr = static_cast<uint32_t>(inst2.high & 0xFFFFFFFF);
        logger.log("[SAInstQueue] Inst Issue Blocked by Event Sync Unit: "
                   "inst_id=%llu opcode=0x%02x is_acc=%d is_bias=%d acc_dest=%d "
                   "a_keep=%d b_keep=%d bias_keep=%d a_addr=0x%08x b_addr=0x%08x "
                   "biaspsum_addr=0x%08x output_addr=0x%08x",
                   (unsigned long long)inst_packet.inst_id, inst_packet.opcode, is_acc, is_bias,
                   acc_dest, a_keep, b_keep, bias_keep, a_addr, b_addr, biaspsum_addr, output_addr);
        return false;
    }

    if(!out_queue(inst_packet)) {
        return false;
    }

    queue_state = InstQueueState::BUSY;
    logger.log("[SAInstQueue] Inst Issue Success: inst_id=%llu, opcode=0x%02x", (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

    // Update Event Sync Unit
    event_sync_unit->issueUpdate(inst_packet);
    // Start the First Functional Unit
    if(inst_packet.opcode == 0x03) {
        PerfFunctionalUnit* sa = core->getUnit(UnitType::SystolicArray);
        sa->execute();
    } else if(inst_packet.opcode == 0x04) {
        auto* acc = dynamic_cast<Accumulator*>(core->getUnit(UnitType::Accumulator));
        assert(acc && "[SAInstQueue] Accumulator is NULL");
        acc->execute(inst_packet.opcode);
    } else {
        throw std::runtime_error("[SAInstQueue] Not Correct Opcode");
    }

    return true;
};


void SAInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[SAInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
    RegFile& regfile = core->getRegFile();
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    if(inst_buffer.opcode == 0x03) {
        isr_val |= (1ULL << 2);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[SAInstQueue] ISR Reg[2] Updated");
    } else if(inst_buffer.opcode == 0x04) {
        isr_val |= (1ULL << 3);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[SAInstQueue] ISR Reg[3] Updated");
    } else {
        throw std::runtime_error("SAInstQueue: unsupported opcode in inst_retire");
    }

    inst_buffer = SAInstPacket{};  // Clear Inst Buffer
};

////////////////////////////
//     SFU Inst Queue     //
////////////////////////////

bool SFUInstQueue::in_queue(const Inst& inst, uint8_t opcode) {
    if(full()) {
        logger.log("[%s] In Queue Blocked: queue full", queue_name.c_str());
        return false;
    }

    if(inst_count == 0) {
        inst_packet = SFUInstPacket{};
        inst_packet.opcode = opcode;
        inst_packet.inst_id = inst_id_counter++;
        inst_packet.inst0 = inst;
    } else if(inst_count == 1) {
        inst_packet.inst1 = inst;
    } else {
        throw std::runtime_error(queue_name + ": invalid inst_count");
    }

    inst_count++;
    logger.log("[%s] In Queue : word=0x%016llx%016llx, inst_count=%d/%d", queue_name.c_str(), (unsigned long long)inst.high, 
                (unsigned long long)inst.low, inst_count, inst_count_upper);

    if(inst_count == inst_count_upper) {
        inst_push(inst_packet);
        core->addInstNum();
        logger.log("[%s] Complete Instruction Push: inst_id=%llu, opcode=0x%02x, current_inst_num=%d", queue_name.c_str(), 
                    (unsigned long long)inst_packet.inst_id, inst_packet.opcode, current_inst_num);
        inst_packet = SFUInstPacket{};
    }

    return true;
}


bool SFUInstQueue::inst_issue() {
    if(queue_state == InstQueueState::BUSY) {
        logger.log("[SFUInstQueue] Inst Issue Skipped: queue is busy");
        return false;
    }

    if(inst_queue.empty()) {
        logger.log("[SFUInstQueue] Inst Issue Skipped: queue is empty");
        return false;
    }

    SFUInstPacket inst_packet = inst_queue.front();
    bool sync_allow_issue = event_sync_unit->allowIssue(inst_packet);
    if(!sync_allow_issue) {
        logger.log("[SFUInstQueue] Inst Issue Blocked by Event Sync Unit");
        return false;
    }

    if(!out_queue(inst_packet)) {
        return false;
    }

    queue_state = InstQueueState::BUSY;
    logger.log("[SFUInstQueue] Inst Issue Success: inst_id=%llu, opcode=0x%02x", (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

    // Update Event Sync Unit
    event_sync_unit->issueUpdate(inst_packet);
    // Start the First Functional Unit
    PerfFunctionalUnit* sfu = core->getUnit(UnitType::SFU);
    sfu->execute();

    return true;
};


void SFUInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[SFUInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
    RegFile& regfile = core->getRegFile();
    uint64_t isr_val = regfile.read(regfile.isr_reg_addr);
    if(inst_buffer.opcode == 0x05) {
        isr_val |= (1ULL << 5);
        regfile.write(regfile.isr_reg_addr, isr_val);
        logger.log("[SFUInstQueue] ISR Reg[5] Updated");
    } else {
        throw std::runtime_error("SFUInstQueue: unsupported opcode in inst_retire");
    }

    inst_buffer = SFUInstPacket{};  // Clear Inst Buffer
};