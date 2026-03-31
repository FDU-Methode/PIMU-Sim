#include <stdexcept>
#include "../../include/perf/event_sync_unit.hpp"

EventSyncUnit::EventSyncUnit(const Config& cfg, Logger& logger) : cfg(cfg), logger(logger) {
    if(cfg.sram_bank_num <= 0) {
        throw std::invalid_argument("Event Sync Unit: bank_num must be greater than 0");
    }
    spm_bank_num = cfg.sram_bank_num;
    spm_bank_bytes = cfg.sram_bytes;
    acc_bank_num = cfg.acc_bank_num;
    bias_bank_num = cfg.bias_bank_num;
    acc_bytes = cfg.acc_bytes;
    bank_valid.assign((spm_bank_num + acc_bank_num + bias_bank_num), 0);
    bank_busy.assign((spm_bank_num + acc_bank_num + bias_bank_num), 0);
}


int EventSyncUnit::getBankId(uint32_t addr, bool is_acc) {
    if(is_acc) {
        if(addr > acc_bytes) {
            logger.log("[EventSyncUnit][ERROR] ACC Addr Exceeds The Boundary");
            throw std::out_of_range("Event Sync Unit: acc addr out of range, addr=" + std::to_string(addr));
        }
        return ((addr * acc_bank_num) / acc_bytes + spm_bank_num);
    } else {
        if(addr > (spm_bank_num * spm_bank_bytes)) {
            logger.log("[EventSyncUnit][ERROR] SPM Addr Exceeds The Boundary");
            throw std::out_of_range("Event Sync Unit: spm addr out of range, addr=" + std::to_string(addr));
        }
        return (addr / spm_bank_bytes);
    }
}


bool EventSyncUnit::allowIssue(const DMAInstPacket& inst_packet) {
    uint8_t opcode = inst_packet.opcode;
    Inst inst0 = inst_packet.inst0;
    Inst inst1 = inst_packet.inst1;

    bool is_acc = (inst0.low >> 13) & 0x1;
    bool is_bias = (inst0.low >> 14) & 0x1;
    uint32_t sram_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);

    if(opcode == 0x01) {  // MVIN

        if(is_acc) {  // MVIN To ACC
            if(is_bias) {  // MVIN To ACC Bias
                if(bank_busy[spm_bank_num + acc_bank_num]) return false;
                else if(bank_valid[spm_bank_num + acc_bank_num]) return false;
                else return true;
            } else {  // MVIN To ACC SRAM
                int bank_id = getBankId(sram_addr, is_acc);
                if(bank_busy[bank_id]) return false;
                else if(bank_valid[bank_id]) return false;
                else return true;
            }
        } else {  // MVIN To SPM
            int bank_id = getBankId(sram_addr, is_acc);
            if(bank_busy[bank_id]) return false;
            else if(bank_valid[bank_id]) return false;
            else return true;
        }

    } else if(opcode == 0x02) {  // MVOUT

        if(is_acc) {  // MVOUT From ACC (SRAM)
            int bank_id = getBankId(sram_addr, is_acc);
            if(bank_busy[bank_id]) return false;
            else if(!bank_valid[bank_id]) return false;
            else return true;
        } else {  // MVOUT FROM SPM
            int bank_id = getBankId(sram_addr, is_acc);
            if(bank_busy[bank_id]) return false;
            else if(!bank_valid[bank_id]) return false;
            else return true;
        }

    } else {
        logger.log("[EventSyncUnit][ERROR] Unsupported Opcode: opcode=0x%02x", opcode);
        throw std::runtime_error("Event Sync Unit: Opcode does not match the function");
    }
};


bool EventSyncUnit::allowIssue(const SAInstPacket& inst_packet) {
    uint8_t opcode = inst_packet.opcode;
    Inst inst0 = inst_packet.inst0;
    Inst inst1 = inst_packet.inst1;
    Inst inst2 = inst_packet.inst2;

    bool is_acc = (inst1.low >> 10) & 0x1;
    bool is_bias = (inst1.low >> 15) & 0x1;
    bool acc_dest = (inst1.low >> 9) & 0x1;

    bool a_valid//
    bool b_valid//
    bool bias_valid//

    if(opcode == 0x03) {  // SA
        uint32_t sram_addr_a = static_cast<uint32_t>(inst2.low & 0xFFFFFFFF);
        uint32_t sram_addr_b = static_cast<uint32_t>(inst2.high & 0xFFFFFFFF);
        uint32_t output_addr = static_cast<uint32_t>((inst1.high >> 32) & 0xFFFFFFFF);
    } else if(opcode == 0x04) {  // MATADD
        //
    } else {
        logger.log("[EventSyncUnit][ERROR] Unsupported Opcode: opcode=0x%02x", opcode);
        throw std::runtime_error("Event Sync Unit: Opcode does not match the function");
    }
};


bool EventSyncUnit::allowIssue(const SFUInstPacket& inst_packet) {
    Inst inst0 = inst_packet.inst0;

    uint32_t input_addr = static_cast<uint32_t>(inst0.high & 0xFFFFFFFF);
    uint32_t output_addr = static_cast<uint32_t>((inst0.high >> 32) & 0xFFFFFFFF);

    int input_bank_id = getBankId(input_addr, 0);
    int output_bank_id = getBankId(output_addr, 0);

    if(bank_busy[input_bank_id] || bank_busy[output_bank_id]) return false;
    else if((!bank_valid[input_bank_id]) || bank_valid[output_bank_id]) return false;
    else return true;
};


bool EventSyncUnit::updateValid(int bit, bool status) {
    if(bit < 0 || bit >= spm_bank_num + acc_bank_num + bias_bank_num) {
        logger.log("[EventSyncUnit][ERROR] Update Valid Failed: bit=%d out of range", bit);
        return false;
    }
    bank_valid[bit] = status;
    return true;
}


bool EventSyncUnit::updateBusy(int bit, bool status) {
    if(bit < 0 || bit >= spm_bank_num + acc_bank_num + bias_bank_num) {
        logger.log("[EventSyncUnit][ERROR] Update Busy Failed: bit=%d out of range", bit);
        return false;
    }
    bank_busy[bit] = status;
    return true;
}