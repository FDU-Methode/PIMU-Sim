#include "../include/logger.hpp"
#include <cstdio>
#include <vector>
#include <filesystem> // C++17
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

Logger::Logger(const std::string &path, bool enable) 
    : enable(enable) {
    if (!enable) return; // 如果不需要记录日志，直接返回
    fs::path log_path(path);
    fs::path dir = log_path.parent_path();

    // 如果包含目录且不存在，就创建
    if (!dir.empty() && !fs::exists(dir)) {
        fs::create_directories(dir);
    }

    ofs.open(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Cannot open log file: " + path);
    }
}

Logger::~Logger() {
    if (ofs.is_open()) ofs.close();
}

void Logger::log(const char *fmt, ...) {
    if (!enable) return; // 日志关闭时不输出
    
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    ofs << buf << std::endl;
}