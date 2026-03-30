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
        case InstType::MVIN: {
            dispatch_status =  dma_1_inst_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);  // TODO: 如何设计分发机制？
            break;
        }
        case InstType::MVOUT: {
            dispatch_status =  dma_3_inst_queue->in_queue(decode_buffer.inst, decode_buffer.opcode);
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
        case 0x05:        return 1;
        default:          return 0;
    }
}