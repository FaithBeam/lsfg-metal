/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "lsfg-metal-common/log.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include <unistd.h>

namespace {
    std::mutex& logMutex() {
        static std::mutex m;
        return m;
    }

    FILE* logFile() {
        static FILE* file = [] () -> FILE* {
            const char* path = std::getenv("LSFG_METAL_LOG");
            if (!path || !*path)
                return nullptr;
            return std::fopen(path, "a");
        }();
        return file;
    }

    const char* levelName(ls::log::Level level) {
        switch (level) {
            case ls::log::Level::Debug: return "debug";
            case ls::log::Level::Info: return "info";
            case ls::log::Level::Warn: return "warn";
            case ls::log::Level::Error: return "error";
        }
        return "?";
    }
}

bool ls::log::debugEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("LSFG_METAL_DEBUG");
        return v && *v && std::string(v) != "0";
    }();
    return enabled;
}

void ls::log::write(Level level, const std::string& msg) {
    if (level == Level::Debug && !debugEnabled())
        return;

    const std::lock_guard<std::mutex> lock(logMutex());
    std::fprintf(stderr, "lsfg-metal[%d] %s: %s\n", getpid(), levelName(level), msg.c_str());
    if (FILE* f = logFile()) {
        std::fprintf(f, "lsfg-metal[%d] %s: %s\n", getpid(), levelName(level), msg.c_str());
        std::fflush(f);
    }
}
