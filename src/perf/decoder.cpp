#include "../../include/perf/decoder.hpp"
#include <stdexcept>

Decoder::Decoder(Logger& logger, InstCache* inst_cache, DMAInstQueue* dma_1_inst_queue, DMAInstQueue* dma_2_inst_queue, 
                DMAInstQueue* dma_3_inst_queue, SAInstQueue* sa_inst_queue, SFUInstQueue* sfu_inst_queue)
                : logger(logger), inst_cache(inst_cache), dma_1_inst_queue(dma_1_inst_queue), dma_2_inst_queue(dma_2_inst_queue),
                dma_3_inst_queue(dma_3_inst_queue), sa_inst_queue(sa_inst_queue), sfu_inst_queue(sfu_inst_queue) {};


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
    if(opcode != 0x01 && opcode != 0x02) {
        return DMATarget::NONE;
    }

    const bool is_acc = isAccDma(inst);

    // ACC-related traffic must go through DMA1
    if(is_acc) {
        return dma_1_inst_queue->full() ? DMATarget::NONE : DMATarget::DMA1;
    }

    auto choose_rr_between_dma2_dma3 = [&](int& rr_idx) -> DMATarget {
        constexpr DMATarget kOrder[2] = {
            DMATarget::DMA2,
            DMATarget::DMA3
        };

        DMAInstQueue* queues[2] = {
            dma_2_inst_queue,
            dma_3_inst_queue
        };

        for(int step = 0; step < 2; ++step) {
            int idx = (rr_idx + step) % 2;
            if(!queues[idx]->full()) {
                rr_idx = (idx + 1) % 2;
                return kOrder[idx];
            }
        }

        return DMATarget::NONE;
    };

    if(opcode == 0x01) {
        // MVIN -> SPM:
        // keep this path stable, only use DMA2/3
        return choose_rr_between_dma2_dma3(next_spm_mvin_rr_idx);
    }

    // opcode == 0x02, MVOUT -> SPM:
    // allow DMA1 to help only when it is completely idle
    if(dmaQueueLoad(dma_1_inst_queue) == 0) {
        return DMATarget::DMA1;
    }

    return choose_rr_between_dma2_dma3(next_spm_mvout_rr_idx);


    // bool is_acc  = isAccDma(inst);
    // if(is_acc) {
    //     return DMATarget::DMA1;
    // }

    // bool dma1_full = dma_1_inst_queue->full();
    // bool dma2_full = dma_2_inst_queue->full();
    // bool dma3_full = dma_3_inst_queue->full();
    // // Compute Load of Each Inst Queue
    // int load1 = dmaQueueLoad(dma_1_inst_queue);
    // int load2 = dmaQueueLoad(dma_2_inst_queue);
    // int load3 = dmaQueueLoad(dma_3_inst_queue);

    // if(opcode == 0x01) {  // MVIN To SPM
    //     if(!dma2_full && !dma3_full) {
    //         if(load2 < load3) return DMATarget::DMA2;
    //         if(load3 < load2) return DMATarget::DMA3;
    //         // Round - Robin
    //         DMATarget target = next_spm_mvin_to_dma2 ? DMATarget::DMA2 : DMATarget::DMA3;
    //         next_spm_mvin_to_dma2 = !next_spm_mvin_to_dma2;
    //         return target;
    //     }
    //     if(!dma2_full) return DMATarget::DMA2;
    //     if(!dma3_full) return DMATarget::DMA3;
    //     if(!dma1_full) return DMATarget::DMA1;
    //     return DMATarget::NONE;
    // }

    // if(opcode == 0x02) {  // MVOUT From SPM
    //     if(!dma1_full && load1 == 0) {
    //         return DMATarget::DMA1;
    //     }
    //     if(!dma2_full && !dma3_full) {
    //         if(load2 < load3) return DMATarget::DMA2;
    //         if(load3 < load2) return DMATarget::DMA3;
    //         // Round - Robin
    //         DMATarget target = next_spm_mvout_to_dma2 ? DMATarget::DMA2 : DMATarget::DMA3;
    //         next_spm_mvout_to_dma2 = !next_spm_mvout_to_dma2;
    //         return target;
    //     }
    //     if(!dma2_full) return DMATarget::DMA2;
    //     if(!dma3_full) return DMATarget::DMA3;
    //     if(!dma1_full) return DMATarget::DMA1;
    //     return DMATarget::NONE;
    // }

    // return DMATarget::NONE;





    // bool dma1_full = dma_1_inst_queue->full();
    // if(dma1_full) {
    //     return DMATarget::NONE;
    // }
    // return DMATarget::DMA1;
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
    return static_cast<int>(
        instructionWordCount(opcode)
    );
}


// Helper Function
bool Decoder::isAccDma(const Inst& inst) {  // Decide Whether to Transfer Between DMA and SPM
    return ((inst.low >> 13) & 0x1) != 0;
}


int Decoder::dmaQueueLoad(DMAInstQueue* inst_queue) {
    return inst_queue->getCurrentInstNum() + (inst_queue->isBusy() ? 1 : 0) + (inst_queue->getInstCount() ? 1 : 0);
}