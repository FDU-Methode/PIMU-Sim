#ifndef _INST_CACHE_HPP__
#define _INST_CACHE_HPP__

#include <cstdint>
#include <cstddef>
#include <queue>
#include "../config.hpp"
#include "../instruction.hpp"

class InstCache {
private:
    int inst_cache_size;
    std::queue<Inst> inst_cache;

public:
    InstCache(const Config& cfg);
    ~InstCache() = default;

    bool write_inst(const Inst& inst);
    bool read_inst(Inst& inst);

    // Cache Status Function
    bool empty() { return inst_cache.empty(); }
    bool full()  { return static_cast<int>(inst_cache.size()) >= inst_cache_size; }

    int size() { return static_cast<int>(inst_cache.size()); }
    int inst_cache_size_() { return inst_cache_size; }

    void clear();

};

#endif