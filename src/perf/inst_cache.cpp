#include "../../include/perf/inst_cache.hpp"
#include <stdexcept>

InstCache::InstCache(Config& cfg) {
    inst_cache_size = cfg.inst_cache_size;
    if(inst_cache_size <= 0) {
        throw std::invalid_argument("InstCache capacity must be greater than 0");
    }
}

bool InstCache::write_inst(const Inst& inst) {
    if(full()) {
        return false;
    }
    inst_cache.push(inst);
    return true;
}

bool InstCache::read_inst(Inst& inst) {
    if(empty()) {
        return false;
    }
    inst = inst_cache.front();
    inst_cache.pop();
    return true;
}

void InstCache::clear() {
    std::queue<Inst> empty_queue;
    inst_cache.swap(empty_queue);
}