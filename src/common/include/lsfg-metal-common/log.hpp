/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>

namespace ls::log {
    /// log levels
    enum class Level { Debug, Info, Warn, Error };

    /// write a log line (prefixed with "lsfg-metal: ")
    ///
    /// Output goes to stderr and, if LSFG_METAL_LOG is set, is appended to that file.
    /// Debug lines are only emitted when LSFG_METAL_DEBUG=1.
    void write(Level level, const std::string& msg);

    inline void debug(const std::string& msg) { write(Level::Debug, msg); }
    inline void info(const std::string& msg) { write(Level::Info, msg); }
    inline void warn(const std::string& msg) { write(Level::Warn, msg); }
    inline void error(const std::string& msg) { write(Level::Error, msg); }

    /// whether debug logging is enabled
    bool debugEnabled();
}
