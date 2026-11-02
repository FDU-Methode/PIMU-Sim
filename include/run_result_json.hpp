#ifndef RUN_RESULT_JSON_HPP
#define RUN_RESULT_JSON_HPP

#include <cstdint>
#include <string>
#include "run_result.hpp"

inline constexpr std::uint32_t kRunResultSchemaVersion = 1;

std::string runResultToJson(const RunResult& result);

#endif