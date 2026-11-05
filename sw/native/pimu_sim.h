#ifndef PIMU_SIM_C_API_H
#define PIMU_SIM_C_API_H

#include<stddef.h>
#include<stdint.h>

#define PIMU_SIM_ABI_VERSION 1u

#ifdef __cplusplus
#define PIMU_NOEXCEPT noexcept
extern "C" {
#else
#define PIMU_NOEXCEPT
#endif

typedef struct PimuSimHandle PimuSimHandle;

typedef struct PimuInstructionWord {
    uint64_t high;  // bits[127:64]
    uint64_t low;   // bits[63:0]
} PimuInstructionWord;

enum {
    PIMU_OK = 0,
    PIMU_INVALID_ARGUMENT = 1,
    PIMU_INVALID_STATE = 2,
    PIMU_BUFFER_TOO_SMALL = 3,
    PIMU_BACKEND_ERROR = 4,
    PIMU_TIMEOUT = 5
};

uint32_t pimu_sim_get_abi_version(void) PIMU_NOEXCEPT;

// On failure, *out_handel is NULL.
// Destroy each successfully created handle once.
int32_t pimu_sim_create(
    const char* config_path,
    PimuSimHandle** out_handle  // Need to change the handle ptr
) PIMU_NOEXCEPT;

// word_count counts 128-bit words, not complete instruction packets.
// Copies all words before returning.
// The caller retains ownership of words.
int32_t pimu_sim_load_program(
    PimuSimHandle* handle,
    const PimuInstructionWord* words,
    size_t word_count
) PIMU_NOEXCEPT;

// Synchronous run using max_sim_cycle from the session configuration.
// Returns PIMU_OK only for DONE.
// TIMEOUT and BACKEND_ERROR require reset before further execution.
int32_t pimu_sim_run(
    PimuSimHandle* handle
) PIMU_NOEXCEPT;

int32_t pimu_sim_reset(
    PimuSimHandle* handle
) PIMU_NOEXCEPT;

// required_size includes the trailing NUL.
// Query size with dst=NULL, capacity=0.
// The caller owns dst.
// BUFFER_TOO_SMALL leaves dst unchanged.
int32_t pimu_sim_get_result_json(
    const PimuSimHandle* handle,
    char* dst,
    size_t capacity,
    size_t* required_size
) PIMU_NOEXCEPT;

// NULL is allowed.
// Other values must be live handles returned by create().
void pimu_sim_destroy(
    PimuSimHandle* handle
) PIMU_NOEXCEPT;

// Borrowed thread-local diagnostic, up to 1023 bytes plus NUL.
// Copy before another API operation. Do not free it.
const char* pimu_sim_last_error(void) PIMU_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#undef PIMU_NOEXCEPT
#endif