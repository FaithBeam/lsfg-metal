/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace ls {
    /// identification data for a process
    struct Identification {
        /// optional override name (LSFGVK_PROFILE / LSFG_METAL_PROFILE)
        std::optional<std::string> override;
        /// path to the Mach-O executable
        std::string executable;
        /// path to the Windows .exe when running under Wine (found via mapped memory regions)
        std::optional<std::string> wine_executable;
        /// traditional process name
        std::string process_name;
    };

    /// enum describing which identification method was used
    enum class IdentType : uint8_t {
        OVERRIDE,
        EXECUTABLE,
        WINE_EXECUTABLE,
        PROCESS_NAME
    };

    /// identify the current process
    ///
    /// Must be called late (e.g. on first present), since Wine maps the .exe after startup.
    Identification identify();

    /// find a profile for the current process
    std::optional<std::pair<IdentType, GameConf>> findProfile(
        const ConfigFile& config, const Identification& id);

    /// whether the current process looks like it runs under Wine
    bool runningUnderWine();

    /// locate Lossless.dll
    ///
    /// Search order: explicit path, $WINEPREFIX, Steam libraries of CrossOver / Whisky /
    /// Sikarugir / Kegworks / Heroic prefixes, macOS Steam, executable directory, cwd.
    /// @throws ls::error if not found
    std::filesystem::path findShaderDll(const std::optional<std::string>& configured);
}
