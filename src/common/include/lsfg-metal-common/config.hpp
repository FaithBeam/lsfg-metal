/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ls {
    /// global configuration
    struct GlobalConf {
        /// optional dll override
        std::optional<std::string> dll;
        /// should fp16 be allowed
        bool allow_fp16{true};
    };

    /// pacing methods
    enum class Pacing : uint8_t {
        /// do not perform any pacing, rely on vsync (same as lsfg-vk)
        None,
        /// space frames out using presentAfterMinimumDuration (macOS only)
        Duration
    };

    /// game profile configuration
    struct GameConf {
        /// name of the profile
        std::string name{"Profile"};
        /// optional activation string/array
        std::vector<std::string> active_in;
        /// gpu to use (in case of multiple), matched against MTLDevice.name
        std::optional<std::string> gpu;
        /// multiplier for frame generation
        size_t multiplier{2};
        /// non-inverted flow scale
        float flow_scale{1.00F};
        /// use performance mode
        bool performance_mode{false};
        /// pacing method
        Pacing pacing{Pacing::None};
        /// force vsync (displaySyncEnabled) on the presenting layer
        bool force_vsync{true};

        bool operator==(const GameConf&) const = default;
    };

    /// parsed configuration file
    class ConfigFile {
    public:
        /// create a default configuration file at the given path
        /// @throws ls::error on failure
        static void createDefaultConfigFile(const std::filesystem::path& path);

        /// load the default configuration
        ConfigFile();

        /// load configuration from file
        /// @throws ls::error on failure
        explicit ConfigFile(const std::filesystem::path& path);

        [[nodiscard]] auto& global() { return this->globalConf; }
        [[nodiscard]] auto& profiles() { return this->profileConfs; }
        [[nodiscard]] const auto& global() const { return this->globalConf; }
        [[nodiscard]] const auto& profiles() const { return this->profileConfs; }
    private:
        GlobalConf globalConf{};
        std::vector<GameConf> profileConfs;
    };

    /// configuration watcher with additional environment support
    class WatchedConfig {
    public:
        /// create a new configuration watcher
        /// @throws ls::error on failure
        WatchedConfig();

        /// reload the configuration from disk if it has changed
        /// @throws ls::error on failure
        /// @return true if the configuration was reloaded
        bool update();

        [[nodiscard]] const auto& get() const { return this->configFile; }
        [[nodiscard]] const auto& filePath() const { return this->path; }
    private:
        ConfigFile configFile;
        std::filesystem::path path;
        std::filesystem::file_time_type last_timestamp{};
    };

    /// find the configuration file in the most common locations
    std::filesystem::path findConfigurationFile();

    /// whether the environment-only configuration mode is enabled (LSFGVK_ENV / LSFG_METAL_ENV)
    bool environmentMode();
}
