#ifndef __PROGRAM_HPP__
#define __PROGRAM_HPP__

#include <cstddef>
#include <vector>

#include "instruction.hpp"

struct Program {
    std::vector<InstructionWord> words;

    std::size_t wordCount() const noexcept {
        return words.size();
    }

    bool empty() const noexcept {
        return words.empty();
    }
};

#endif