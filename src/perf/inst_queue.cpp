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
        logger.log("[DMAInstQueue] Inst Issue Blocked by Event Sync Unit");
        return false;
    }

    if(!out_queue(inst_packet)) {
        return false;
    }

    queue_state = InstQueueState::BUSY;
    logger.log("[DMAInstQueue] Inst Issue Success: inst_id=%llu, opcode=0x%02x", (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

    // Update Event Sync Unit
    event_sync_unit->issueUpdate(inst_packet);
    // Write Config Register
    writeReg(inst_packet);
    // Start the First Functional Unit
    DMA* dma = nullptr;
    switch(queue_id) {
        case 1: {
            dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA1));
            break;
        }
        case 2: {
            dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA2));
            break;
        }
        case 3: {
            dma = dynamic_cast<DMA*>(core->getUnit(UnitType::DMA3));
            break;
        }
        default: throw std::runtime_error("[DMAInstQueue] No Valid Queue ID");
    }
    dma->execute(inst_packet.opcode);

    return true;

};


void DMAInstQueue::writeReg(DMAInstPacket& inst_packet) {
    uint8_t opcode = inst_packet.opcode;
    Inst inst0 = inst_packet.inst0;
    Inst inst1 = inst_packet.inst1;

    uint8_t  data_type        = static_cast<uint8_t>((inst0.low >> 8) & 0x3);
    uint8_t  data_precision   = static_cast<uint8_t>((inst0.low >> 10) & 0x3);
    uint8_t  is_quant         = static_cast<uint8_t>((inst0.low >> 12) & 0x1);
    uint8_t  source           = static_cast<uint8_t>((inst0.low >> 13) & 0x1);
    uint8_t  is_bias          = static_cast<uint8_t>((inst0.low >> 14) & 0x1);
    uint16_t col_num          = static_cast<uint16_t>((inst0.low >> 32) & 0xFFFF);
    uint16_t row_num          = static_cast<uint16_t>((inst0.low >> 48) & 0xFFFF);
    uint32_t sram_addr        = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
    uint32_t dram_addr        = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);

    uint16_t sram_stride      = static_cast<uint16_t>(inst1.low & 0xFFFF);
    uint16_t dram_stride      = static_cast<uint16_t>((inst1.low >> 16) & 0xFFFF);
    uint16_t quant_scale      = static_cast<uint16_t>((inst1.low >> 32) & 0xFFFF);
    uint16_t quant_scaleshift = static_cast<uint16_t>((inst1.low >> 48) & 0xFFFF);
    uint32_t quant_zeropoint  = static_cast<uint32_t>(inst1.high & 0xFFFFFFFF);

    uint64_t rs1      = static_cast<uint64_t>(dram_addr) | (static_cast<uint64_t>(row_num) << 32);
    uint64_t rs2      = static_cast<uint64_t>(sram_addr) | (static_cast<uint64_t>(col_num) << 32);
    uint64_t cfg_rs1  = static_cast<uint64_t>(data_type) |
                        (static_cast<uint64_t>(data_precision) << 2) |
                        (static_cast<uint64_t>(is_quant) << 4) |
                        (static_cast<uint64_t>(source) << 5) |
                        (static_cast<uint64_t>(is_bias) << 6) |
                        (static_cast<uint64_t>(sram_stride) << 32) |
                        (static_cast<uint64_t>(dram_stride) << 48);
    uint64_t cfg_rs2  = static_cast<uint64_t>(quant_zeropoint) |
                        (static_cast<uint64_t>(quant_scale) << 32) |
                        (static_cast<uint64_t>(quant_scaleshift) << 48);

    if(opcode == 0x01) {  // MVIN
        regfile.write(regfile.mvin_rs1_reg_addr, rs1);
        regfile.write(regfile.mvin_rs2_reg_addr, rs2);
        regfile.write(regfile.config_mvin_rs1_reg_addr, cfg_rs1);
        regfile.write(regfile.config_mvin_rs2_reg_addr, cfg_rs2);
    } else if(opcode == 0x02) {  // MVOUT
        regfile.write(regfile.mvout_rs1_reg_addr, rs1);
        regfile.write(regfile.mvout_rs2_reg_addr, rs2);
        regfile.write(regfile.config_mvout_rs1_reg_addr, cfg_rs1);
        regfile.write(regfile.config_mvout_rs2_reg_addr, cfg_rs2);
    } else {
        throw std::runtime_error("DMAInstQueue: unsupported opcode");
    }
}


void DMAInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[DMAInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
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
        logger.log("[SAInstQueue] Inst Issue Blocked by Event Sync Unit");
        return false;
    }

    if(!out_queue(inst_packet)) {
        return false;
    }

    queue_state = InstQueueState::BUSY;
    logger.log("[SAInstQueue] Inst Issue Success: inst_id=%llu, opcode=0x%02x", (unsigned long long)inst_packet.inst_id, inst_packet.opcode);

    // Update Event Sync Unit
    event_sync_unit->issueUpdate(inst_packet);
    // Write Config Register
    writeReg(inst_packet);
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


void SAInstQueue::writeReg(SAInstPacket& inst_packet) {
    uint8_t opcode = inst_packet.opcode;
    Inst inst0 = inst_packet.inst0;
    Inst inst1 = inst_packet.inst1;
    Inst inst2 = inst_packet.inst2;

    if(opcode == 0x03) {  // SA
        uint32_t config_1          = static_cast<uint32_t>((inst0.low >> 8) & 0xFFFFFF);
        uint16_t input_a_zeropoint = static_cast<uint16_t>((inst0.low >> 32) & 0xFFFF);
        uint16_t input_b_zeropoint = static_cast<uint16_t>((inst0.low >> 48) & 0xFFFF);
        uint16_t output_scale      = static_cast<uint16_t>(inst0.high & 0xFFFF);
        uint16_t output_scaleshift = static_cast<uint16_t>((inst0.high >> 16) & 0xFFFF);
        uint32_t output_zeropoint  = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);

        uint16_t config_2          = static_cast<uint16_t>(inst1.low & 0x1FFF);
        uint8_t  biaspsum_width    = static_cast<uint8_t>((inst1.low >> 16) & 0xFF);
        uint8_t  biaspsum_height   = static_cast<uint8_t>((inst1.low >> 24) & 0xFF);
        uint16_t biaspsum_stride   = static_cast<uint16_t>((inst1.low >> 32) & 0xFFFF);
        uint16_t output_stride     = static_cast<uint16_t>((inst1.low >> 48) & 0xFFFF);
        uint32_t biaspsum_addr     = static_cast<uint32_t>(inst1.high & 0xFFFFFFFF);
        uint32_t output_addr       = static_cast<uint32_t>((inst1.high >> 32) & 0xFFFFFFFF);

        uint32_t input_a_addr      = static_cast<uint32_t>(inst2.low & 0xFFFFFFFF);
        uint16_t input_a_col_num   = static_cast<uint16_t>((inst2.low >> 32) & 0x7FF);
        uint8_t  input_a_row_num   = static_cast<uint8_t>((inst2.low >> 43) & 0x1F);
        uint16_t input_a_stride    = static_cast<uint16_t>((inst2.low >> 48) & 0xFFFF);
        uint32_t input_b_addr      = static_cast<uint32_t>(inst2.high & 0xFFFFFFFF);
        uint8_t  input_b_col_num   = static_cast<uint8_t>((inst2.high >> 32) & 0x1F);
        uint16_t input_b_row_num   = static_cast<uint16_t>((inst2.high >> 37) & 0x7FF);
        uint16_t input_b_stride    = static_cast<uint16_t>((inst2.high >> 48) & 0xFFFF);

        uint64_t cfg_compute_rs1 = 0;
        cfg_compute_rs1    |= static_cast<uint64_t>(config_1);
        cfg_compute_rs1    |= (static_cast<uint64_t>(config_2 & 0x3F) << 24);
        cfg_compute_rs1    |= (static_cast<uint64_t>(input_a_zeropoint) << 32);
        cfg_compute_rs1    |= (static_cast<uint64_t>(input_b_zeropoint) << 48);

        uint64_t cfg_compute_rs2 = 0;
        cfg_compute_rs2    |= static_cast<uint64_t>(output_zeropoint);
        cfg_compute_rs2    |= (static_cast<uint64_t>(output_scale) << 32);
        cfg_compute_rs2    |= (static_cast<uint64_t>(output_scaleshift) << 48);

        uint64_t cfg_accumulate_rs1 = 0;
        cfg_accumulate_rs1 |= static_cast<uint64_t>(biaspsum_addr);
        cfg_accumulate_rs1 |= (static_cast<uint64_t>(biaspsum_stride) << 32);
        cfg_accumulate_rs1 |= (static_cast<uint64_t>(biaspsum_width) << 48);
        cfg_accumulate_rs1 |= (static_cast<uint64_t>(biaspsum_height) << 56);

        uint64_t cfg_accumulate_rs2 = 0;
        cfg_accumulate_rs2 |= static_cast<uint64_t>(output_addr);
        cfg_accumulate_rs2 |= (static_cast<uint64_t>(output_stride) << 32);
        cfg_accumulate_rs2 |= (static_cast<uint64_t>((config_2 >> 6) & 0x7F) << 48);

        uint64_t compute_sa_rs1 = 0;
        compute_sa_rs1     |= static_cast<uint64_t>(input_a_addr);
        compute_sa_rs1     |= (static_cast<uint64_t>(input_a_col_num) << 32);
        compute_sa_rs1     |= (static_cast<uint64_t>(input_a_row_num) << 43);
        compute_sa_rs1     |= (static_cast<uint64_t>(input_a_stride) << 48);

        uint64_t compute_sa_rs2 = 0;
        compute_sa_rs2     |= static_cast<uint64_t>(input_b_addr);
        compute_sa_rs2     |= (static_cast<uint64_t>(input_b_col_num) << 32);
        compute_sa_rs2     |= (static_cast<uint64_t>(input_b_row_num) << 37);
        compute_sa_rs2     |= (static_cast<uint64_t>(input_b_stride) << 48);

        regfile.write(regfile.config_compute_rs1_reg_addr, cfg_compute_rs1);
        regfile.write(regfile.config_compute_rs2_reg_addr, cfg_compute_rs2);
        regfile.write(regfile.config_accumulate_rs1_reg_addr, cfg_accumulate_rs1);
        regfile.write(regfile.config_accumulate_rs2_reg_addr, cfg_accumulate_rs2);
        regfile.write(regfile.compute_sa_rs1_reg_addr, compute_sa_rs1);
        regfile.write(regfile.compute_sa_rs2_reg_addr, compute_sa_rs2);

    } else if(opcode == 0x04) {  // MATADD

        uint8_t  col_num           = static_cast<uint8_t>((inst0.low >> 8) & 0xFF);
        uint8_t  row_num           = static_cast<uint8_t>((inst0.low >> 16) & 0xFF);
        uint32_t a_addr            = static_cast<uint32_t>((inst0.low >> 32) & 0xFFFFFFFF);
        uint32_t b_addr            = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
        uint32_t output_addr       = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);

        uint16_t output_scale      = static_cast<uint16_t>(inst1.low & 0xFFFF);
        uint16_t output_shift      = static_cast<uint16_t>((inst1.low >> 16) & 0xFFFF);
        uint32_t output_zeropoint  = static_cast<uint32_t>((inst1.low >> 32) & 0xFFFFFFFF);

        uint64_t compute_matadd_rs1 =   static_cast<uint64_t>(a_addr) |
                                        (static_cast<uint64_t>(b_addr) << 32);

        uint64_t compute_matadd_rs2 =   static_cast<uint64_t>(output_addr) |
                                        (static_cast<uint64_t>(col_num) << 32) |
                                        (static_cast<uint64_t>(row_num) << 40);

        uint64_t cfg_compute_rs2 = 0;
        cfg_compute_rs2 |= static_cast<uint64_t>(output_zeropoint);
        cfg_compute_rs2 |= (static_cast<uint64_t>(output_scale) << 32);
        cfg_compute_rs2 |= (static_cast<uint64_t>(output_shift) << 48);

        regfile.write(regfile.compute_matadd_rs1_reg_addr, compute_matadd_rs1);
        regfile.write(regfile.compute_matadd_rs2_reg_addr, compute_matadd_rs2);
        regfile.write(regfile.config_compute_rs2_reg_addr, cfg_compute_rs2);

    } else {
        throw std::runtime_error("SAInstQueue: unsupported opcode");
    }
}


void SAInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[SAInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
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
    // Write Config Register
    writeReg(inst_packet);
    // Start the First Functional Unit
    PerfFunctionalUnit* sfu = core->getUnit(UnitType::SFU);
    sfu->execute();

    return true;
};


void SFUInstQueue::writeReg(SFUInstPacket& inst_packet) {
    uint8_t opcode = inst_packet.opcode;
    Inst inst0 = inst_packet.inst0;
    Inst inst1 = inst_packet.inst1;

    if(opcode != 0x05) {
        throw std::runtime_error("SFUInstQueue: unsupported opcode");
    }

    uint8_t  config            = static_cast<uint8_t>((inst0.low >> 8) & 0x7FF);
    uint16_t col_num           = static_cast<uint16_t>((inst0.low >> 32) & 0xFFFF);
    uint16_t row_num           = static_cast<uint16_t>((inst0.low >> 48) & 0xFFFF);
    uint32_t input_addr        = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
    uint32_t output_addr       = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);

    uint32_t input_zeropoint   = static_cast<uint32_t>(inst1.low & 0xFFFFFFFF);
    uint16_t input_scale       = static_cast<uint16_t>((inst1.low >> 32) & 0xFFFF);
    uint16_t input_scaleshift  = static_cast<uint16_t>((inst1.low >> 48) & 0xFFFF);
    uint16_t output_zeropoint  = static_cast<uint16_t>(inst1.high & 0xFFFF);
    uint16_t output_scale      = static_cast<uint16_t>((inst1.high >> 16) & 0xFFFF);
    uint16_t output_scaleshift = static_cast<uint16_t>((inst1.high >> 32) & 0xFFFF);

    uint64_t config_sfu_rs1   = static_cast<uint64_t>(config) |
                                (static_cast<uint64_t>(output_zeropoint) << 16) |
                                (static_cast<uint64_t>(input_zeropoint) << 32);

    uint64_t config_sfu_rs2   = static_cast<uint64_t>(input_scale) |
                                (static_cast<uint64_t>(input_scaleshift) << 16) |
                                (static_cast<uint64_t>(output_scale) << 32) |
                                (static_cast<uint64_t>(output_scaleshift) << 48);

    uint64_t compute_sfu_rs1  = static_cast<uint64_t>(input_addr) |
                                (static_cast<uint64_t>(col_num) << 32) |
                                (static_cast<uint64_t>(row_num) << 48);

    uint64_t compute_sfu_rs2  = static_cast<uint64_t>(output_addr);

    regfile.write(regfile.config_sfu_rs1_reg_addr, config_sfu_rs1);
    regfile.write(regfile.config_sfu_rs2_reg_addr, config_sfu_rs2);
    regfile.write(regfile.compute_sfu_rs1_reg_addr, compute_sfu_rs1);
    regfile.write(regfile.compute_sfu_rs2_reg_addr, compute_sfu_rs2);
}


void SFUInstQueue::inst_retire() {
    queue_state = InstQueueState::IDLE;
    logger.log("[SFUInstQueue] Inst Retire: state -> IDLE");

    // Update Event Sync Unit
    event_sync_unit->retireUpdate(inst_buffer);

    // Write ISR
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