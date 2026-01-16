#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <fstream>
#include <string>
#include <cstdarg>

class Logger {
public:
    Logger(const std::string &log_file, bool enable);
    ~Logger();
    void log(const char *fmt, ...);
private:
    std::ofstream ofs;
    bool enable;
};

#endif
