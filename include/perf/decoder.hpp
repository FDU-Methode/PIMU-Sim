#ifndef _DECODER_HPP__
#define _DECODER_HPP__

#include <cstdint>
#include "../util.hpp"
#include "../logger.hpp"
#include "inst_cache.hpp"
#include "inst_queue.hpp"

enum class DMATarget {
    NONE, DMA1, DMA2, DMA3
};

struct DecodedLine {
    Inst inst{};
    InstType type = InstType::NONE;
    uint8_t opcode = 0;  // Valid for first line
};

class Decoder {
private:
    Logger& logger;

    InstCache* inst_cache;

    DMAInstQueue* dma_1_inst_queue;
    DMAInstQueue* dma_2_inst_queue;
    DMAInstQueue* dma_3_inst_queue;
    SAInstQueue*  sa_inst_queue;
    SFUInstQueue* sfu_inst_queue;

    // Pipeline Buffers
    bool fetch_valid = false;   // True: Fetch Buffer Waiting for Decode, Disable Fetch
    Inst fetch_buffer{};
    bool decode_valid = false;  // True: Decode Buffer Waiting for Dispatch, Disable Decode
    DecodedLine decode_buffer{};

    // Multiple line inst status
    InstType inst_type = InstType::NONE;
    int remaining_inst = 0;

    // DMA Dispatch Status
    DMATarget current_dma_target = DMATarget::NONE;
    bool next_spm_mvin_to_dma2 = true;
    bool next_spm_mvout_to_dma2 = true;

public:
    Decoder(Logger& logger, InstCache* inst_cache, DMAInstQueue* dma_1_inst_queue, DMAInstQueue* dma_2_inst_queue, 
            DMAInstQueue* dma_3_inst_queue, SAInstQueue* sa_inst_queue, SFUInstQueue* sfu_inst_queue);
    ~Decoder() = default;

    void tick();

    // Excute Pipeline
    bool fetch();
    bool decode();
    bool dispatch();

    bool fetchBufferValid() { return fetch_valid; };
    bool decodeBufferValid() { return decode_valid; };
    InstType currentInstType() { return inst_type; };
    int remainingInst() { return remaining_inst; };

    // DMA Dispatch Option
    DMATarget selectDMATarget(uint8_t opcode, const Inst& inst);

private:
    uint8_t getOpcode(const Inst& inst);
    InstType classifyOpcode(uint8_t opcode);
    int getInstLineCount(uint8_t opcode);
 
};






#endif