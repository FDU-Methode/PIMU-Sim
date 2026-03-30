#ifndef _DECODER_HPP__
#define _DECODER_HPP__

#include <cstdint>
#include <queue>
#include <string>
#include <vector>
#include "../util.hpp"
#include "../config.hpp"
#include "../logger.hpp"
#include "inst_cache.hpp"

struct DecodedInst {
    InstType type = InstType::NONE;
    uint8_t opcode = 0;
    std::vector<Inst> words;
};

class Decoder {
private:
    Logger& logger;
    InstCache& inst_cache;

    // Pipeline Buffers
    bool fetch_valid = false;
    Inst fetch_buffer{};
    bool decode_valid = false;
    DecodedInst decode_buffer{};

    InstType inst_type = InstType::NONE;
    int remaining_inst = 0;
    std::vector<Inst> assembling_words;

    std::queue<>



public:





};






#endif