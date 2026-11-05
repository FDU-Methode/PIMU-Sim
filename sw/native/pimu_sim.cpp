#include "pimu_sim.h"

#include "config.hpp"
#include "perf/perf_sim.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <limits>
#include <utility>

struct PimuSimHandle {
    PerfSim sim;

    explicit PimuSimHandle(const Config& cfg) : sim(cfg) {}
};

namespace {

thread_local char last_error[1024] = {};

int32_t fail(int32_t code, const char* message) noexcept {
    std::strncpy(
        last_error, message ? message : "Unknown error",
        sizeof(last_error) - 1
    );

    last_error[sizeof(last_error) - 1] = 0;

    return code;
};


// Convert C++ error to error code

template <typename F>
int32_t protect(F&& operation) noexcept {
    last_error[0] = 0;

    try {
        return operation();
    } catch (const std::invalid_argument& e) {
        return fail(PIMU_INVALID_ARGUMENT, e.what());
    } catch (const std::logic_error& e) {
        return fail(PIMU_INVALID_STATE, e.what());
    } catch (const std::exception& e) {
        return fail(PIMU_BACKEND_ERROR, e.what());
    } catch (...) {
        return fail(
            PIMU_BACKEND_ERROR,
            "Unknown C++ exception"
        );
    }
}

}  // namespace

extern "C" {

uint32_t pimu_sim_get_abi_version(void) noexcept {
    return PIMU_SIM_ABI_VERSION;
}

const char* pimu_sim_last_error(void) noexcept {
    return last_error;
}

int32_t pimu_sim_create(
    const char* config_path,
    PimuSimHandle** out_handle
) noexcept {
    return protect([&]() -> int32_t {
        if(!out_handle) {
            return fail(
                PIMU_INVALID_ARGUMENT,
                "out_handle is NULL"
            );
        }

        *out_handle = nullptr;

        if(!config_path || config_path[0] == 0) {
            return fail(
                PIMU_INVALID_ARGUMENT,
                "config_path is empty"
            );
        }

        auto handle = std::make_unique<PimuSimHandle>(
            Config::from_file(config_path)
        );

        *out_handle = handle.release();

        return PIMU_OK;
    });
}

int32_t pimu_sim_reset(
    PimuSimHandle* handle
) noexcept {
    return protect([&]() -> int32_t {
        if(!handle) {
            return fail(
                PIMU_INVALID_ARGUMENT,
                "handle is NULL"
            );
        }

        handle->sim.reset();

        return PIMU_OK;
    });
}

int32_t pimu_sim_get_result_json(
    const PimuSimHandle* handle,
    char* dst,
    size_t capacity,
    size_t* required_size
) noexcept {
    return protect([&]() -> int32_t {
        if(required_size) {
            *required_size = 0;
        }

        if(!handle || !required_size || (!dst && capacity != 0)) {
            return fail(
                PIMU_INVALID_ARGUMENT,
                "Invalid JSON buffer arguments"
            );
        }

        const auto json = handle->sim.getResultJson();

        *required_size = json.size() + 1;

        if(!dst) { return PIMU_OK; }  // Just check json length

        if(capacity < *required_size) {
            return fail(
                PIMU_BUFFER_TOO_SMALL,
                "JSON buffer is too small"
            );
        }

        std::memcpy(dst, json.c_str(), *required_size);

        return PIMU_OK;
    });
}

void pimu_sim_destroy(PimuSimHandle* handle) noexcept {
    last_error[0] = 0;
    delete handle;
}

}  // extern "C"