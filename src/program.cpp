#include "../include/program.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

void validateProgram(const Program& program) {
    if(program.empty()) {
        throw std::invalid_argument(
            "Program must not be empty"
        );
    }

    if(program.wordCount() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "Program exceeds current instruction counter range"
        );
    }

    std::size_t index = 0;

    while(index < program.wordCount()) {
        const auto opcode = static_cast<std::uint8_t>(
            program.words[index].low & 0xff
        );

        const auto count = instructionWordCount(opcode);

        if(count == 0) {
            std::ostringstream message;
            message << "word" << index +1 << ": unsupported opcode 0x"
            << std::hex << static_cast<unsigned>(opcode);

            throw std::invalid_argument(message.str());
        }

        if(count > program.wordCount() - index) {
            throw std::invalid_argument(
                "word" + std::to_string(index + 1) +
                ": incomplete instruction packet"
            );
        }

        index += count;
    }
}