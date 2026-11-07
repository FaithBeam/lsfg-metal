/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <exception>
#include <stdexcept>
#include <string>

namespace ls {
    /// simple error type
    class error : public std::runtime_error {
    public:
        /// construct an error around an inner exception
        /// @param msg error message
        /// @param inner inner exception
        explicit error(const std::string& msg, const std::exception& inner)
            : std::runtime_error(msg + "\n- " + inner.what()) {}

        /// construct an error
        /// @param msg error message
        explicit error(const std::string& msg)
            : std::runtime_error(msg) {}
    };

    /// error raised by the Metal backend
    class metal_error : public error {
    public:
        explicit metal_error(const std::string& msg) : error(msg) {}
    };
}
