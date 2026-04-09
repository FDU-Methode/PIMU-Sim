#include "../../include/perf/decoder.hpp"
#include <stdexcept>

Decoder::Decoder(Logger& logger, const Config& cfg, InstCache* inst_cache, DMAInstQueue* dma_1_inst_queue, DMAInstQueue* dma_2_inst_queue, 
                DMAInstQueue* dma_3_inst_queue, DMAInstQueue* dma_4_inst_queue, DMAInstQueue* dma_5_inst_queue, DMAInstQueue* dma_6_inst_queue, 
                SAInstQueue* sa_inst_queue, SFUInstQueue* sfu_inst_queue)
                : logger(logger), cfg(cfg), inst_cache(inst_cache), dma_1_inst_queue(dma_1_inst_queue), dma_2_inst_queue(dma_2_inst_queue),
                dma_3_inst_queue(dma_3_inst_queue), dma_4_inst_queue(dma_4_inst_queue), dma_5_inst_queue(dma_5_inst_queue), dma_6_inst_queue(dma_6_inst_queue), 
                sa_inst_queue(sa_inst_queue), sfu_inst_queue(sfu_inst_queue) {};


void Decoder::tick() {
    // Stage 1 - Attempt to dispatch
    dispatch();
    // Stage 2
    if(!decode_valid) {
        decode();
    }
    // Stage 3
    if(!decode_valid && !fetch_valid) {
        fetch();
    }
}


bool Decoder::fetch() {
    if(fetch_valid || decode_valid) {
        return false;
    }

    Inst inst{};
    if(!inst_cache->read_inst(inst)) {
        return false;
    }

    // Write Fetch Buffer
    fetch_buffer = inst;
    fetch_valid = true;

    logger.log("[Decoder][Fetch] inst=0x%016llx%016llx", (unsigned long long)fetch_buffer.high, (unsigned long long)fetch_buffer.low);
    return true;
}


bool Decoder::decode() {
    if(decode_valid || !fetch_valid) {
        return false;
    }

    // Read Fetch Buffer
    Inst inst = fetch_buffer;
    fetch_valid = false;

    // Get the first inst line
    if(inst_type == InstType::NONE) {
        uint8_t opcode = getOpcode(inst);
        InstType type = classifyOpcode(opcode);
        int line_count = getInstLineCount(opcode);

        if(type == InstType::NONE || line_count <= 0) {
            logger.log("[Decoder][Decode][ERROR] Unsupported Opcode=0x%02x", opcode);
            throw std::runtime_error("Decoder get unsupported opcode");
        }

        // Write Decode Buffer
        decode_buffer.inst = inst;
        decode_buffer.type = type;
        decode_buffer.opcode = opcode;
        decode_valid = true;

        remaining_inst = line_count - 1;
        inst_type = (remaining_inst > 0) ? type : InstType::NONE;  // Do not save the inst type if consists of only one line

        logger.log("[Decoder][Decode] Head Inst Line: opcode=0x%02x, type=%s, remaining_inst=%d", 
                    opcode, InstTypeToStr(type).c_str(), remaining_inst);
        return true;
    }

    // Get the remaining inst line
    // Write Decode Buffer
    decode_buffer.inst = inst;
    decode_buffer.type = inst_type;
    decode_buffer.opcode = 0;
    decode_valid = true;

    remaining_inst--;

    logger.log("[Decoder][Decode] Remaining Inst LineL: type=%s, remaining_inst=%d", 
                InstTypeToStr(inst_type).c_str(), remaining_inst);

    if(remaining_inst == 0) {
        inst_type = InstType::NONE;
    }

    return true;
}


bool Decoder::dispatch() {
    if(!decode_valid) {
        return false;
    }

    // Dispatch Decode Buffer
    bool dispatch_status = false;

    switch(decode_buffer.type) {
        case InstType::MVIN:
        case InstType::MVOUT: {
            DMAInstQueue* target_queue = nullptr;
            // First Line of DMA Inst
            if(decode_buffer.opcode != 0) {
                DMATarget target = selectDMATarget(decode_buffer.opcode, decode_buffer.inst);
                if(target == DMATarget::NONE) {  // All DMA Queues Are Full
                    logger.log("[Decoder][Dispatch] DMA Dispatch Blocked: no available DMA queue");
                    return false;
                }
                // Store DMA Queue
                current_dma_target = target;
            }

            switch(current_dma_target) {
                case DMATarget::DMA1: target_queue = dma_1_inst_queue; break;
                case DMATarget::DMA2: target_queue = dma_2_inst_queue; break;
                case DMATarget::DMA3: target_queue = dma_3_inst_queue; break;
                case DMATarget::DMA4: target_queue = dma_4_inst_queue; break;
                case DMATarget::DMA5: target_queue = dma_5_inst_queue; break;
                case DMATarget::DMA6: target_queue = dma_6_inst_queue; break;
                default: throw std::runtime_error("Decoder DMA Dispatch Without Valid Target");
            }

            dispatch_status =  target_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);
            if(!dispatch_status) {
                return false;
            }

            // Clear DMA Target When DMA Inst Finish
            if(inst_type == InstType::NONE && remaining_inst == 0) {
                current_dma_target = DMATarget::NONE;
            }
            break;
        }

        case InstType::SA: {
            dispatch_status =  sa_inst_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);
            break;
        }
        case InstType::MATADD: {
            dispatch_status =  sa_inst_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);
            break;
        }
        case InstType::SFU: {
            dispatch_status =  sfu_inst_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);
            break;
        }
        default: throw std::runtime_error("Decoder dispatch get invalid InstType");
    }

    if(!dispatch_status) {
        logger.log("[Decoder][Dispatch] Dispatch Blocked: Target Inst Queue Full, type=%s, opcode=0x%02x", 
                    InstTypeToStr(decode_buffer.type).c_str(), decode_buffer.opcode);
        return false;
    }

    logger.log("[Decoder][Dispatch] Dispatch Success: type=%s, opcode=0x%02x", 
                InstTypeToStr(decode_buffer.type).c_str(), decode_buffer.opcode);

    // Clear Decode Buffer
    decode_buffer = DecodedLine{};
    decode_valid = false;
    return true;
}


DMATarget Decoder::selectDMATarget(uint8_t opcode, const Inst& inst) {
    bool is_acc = isAccDma(inst);
    uint32_t sram_addr = static_cast<uint32_t>(inst.high & 0xFFFFFFFF);

    if(is_acc) {
        if(opcode == 0x01) {
            return DMATarget::DMA3;
        }
        return DMATarget::NONE;
    }

    int bank_id = getSpmBankId(sram_addr);

    if(opcode == 0x01) {  // MVIN -> SPM
        if(isABank(bank_id)) {
            return DMATarget::DMA1;
        }
        if(isBBank(bank_id)) {
            return DMATarget::DMA2;
        }
        if(isOutputBank(bank_id)) {
            return DMATarget::DMA4;
        }
        return DMATarget::NONE;
    }

    if(opcode == 0x02) {  // MVOUT <- SPM
        if(!isOutputBank(bank_id)) {
            return DMATarget::NONE;
        }

        bool dma5_full = dma_5_inst_queue->full();
        bool dma6_full = dma_6_inst_queue->full();
        if(dma5_full && dma6_full) return DMATarget::NONE;
        if(!dma5_full && dma6_full) return DMATarget::DMA5;
        if(dma5_full && !dma6_full) return DMATarget::DMA6;
        // Round - Robin
        DMATarget target = next_spm_read_to_dma5 ? DMATarget::DMA5 : DMATarget::DMA6;
        next_spm_read_to_dma5 = !next_spm_read_to_dma5;
        return target;
    }

    return DMATarget::NONE;
}


uint8_t Decoder::getOpcode(const Inst& inst) {
    return static_cast<uint8_t>(inst.low & 0xFF);
}


InstType Decoder::classifyOpcode(uint8_t opcode) {
    switch(opcode) {
        case 0x01:        return InstType::MVIN;
        case 0x02:        return InstType::MVOUT;
        case 0x03:        return InstType::SA;
        case 0x04:        return InstType::MATADD;
        case 0x05:        return InstType::SFU;
        default:          return InstType::NONE;
    }
}


int Decoder::getInstLineCount(uint8_t opcode) {
    switch(opcode) {
        case 0x01:        return 2;
        case 0x02:        return 2;
        case 0x03:        return 3;
        case 0x04:        return 3;
        case 0x05:        return 2;
        default:          return 0;
    }
}


// Helper Function
bool Decoder::isAccDma(const Inst& inst) {  // Decide Whether to Transfer Between DMA and SPM
    return ((inst.low >> 13) & 0x1) != 0;
}


int Decoder::dmaQueueLoad(DMAInstQueue* inst_queue) {
    return inst_queue->getCurrentInstNum() + (inst_queue->isBusy() ? 1 : 0) + (inst_queue->getInstCount() ? 1 : 0);
}


int Decoder::getSpmBankId(uint32_t addr) {
    return static_cast<int>(addr / cfg.sram_bank_bytes);
}


bool Decoder::isABank(int bank_id) {
    return bank_id == 0 || bank_id == 1;
}


bool Decoder::isBBank(int bank_id) {
    return bank_id == 2 || bank_id == 3;
}


bool Decoder::isOutputBank(int bank_id) {
    return bank_id >= 4 && bank_id < cfg.sram_bank_num;
}