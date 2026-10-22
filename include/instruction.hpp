#ifndef __INSTRUCTION_HPP__
#define __INSTRUCTION_HPP__

#include <cstdint>
#include <type_traits>

// One 128-bit instruction word
struct InstructionWord {
    std::uint64_t high = 0; // bits [127:64]
    std::uint64_t low = 0;  // bits [63:0]
};

// Compatibility alias for existing simulator code
using Inst = InstructionWord;

static_assert(
    sizeof(InstructionWord) == 16,
    "InstructionWord must occupy 16 bytes"
);

static_assert(
    std::is_standard_layout_v<InstructionWord>
);

static_assert(
    std::is_trivially_copyable_v<InstructionWord>
);

#endif