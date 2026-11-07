/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Ported from lsfg-vk-common/src/configuration/config.cpp */

#include "lsfg-metal-common/config.hpp"
#include "lsfg-metal-common/errors.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <toml.hpp>

using namespace ls;

namespace {
    /// read an environment variable, checking the lsfg-metal name first, then the lsfg-vk name
    const char* env(const char* metalName, const char* vkName) {
        const char* v = std::getenv(metalName);
        if (v && *v) return v;
        v = std::getenv(vkName);
        if (v && *v) return v;
        return nullptr;
    }
}

bool ls::environmentMode() {
    return env("LSFG_METAL_ENV", "LSFGVK_ENV") != nullptr;
}

void ConfigFile::createDefaultConfigFile(const std::filesystem::path& path) {
    try {
        std::filesystem::create_directories(path.parent_path());
        if (!std::filesystem::exists(path.parent_path()))
            throw ls::error("unable to create configuration directory");

        std::ofstream ofs(path);
        if (!ofs.is_open())
            throw ls::error("unable to create default configuration file");

        ofs << R"(version = 2

[global]
# Path to Lossless.dll. If unset, lsfg-metal searches the current Wine prefix,
# CrossOver/Whisky/Sikarugir bottles and the macOS Steam library.
# dll = '/Users/me/Library/Application Support/CrossOver/Bottles/Steam/drive_c/Program Files (x86)/Steam/steamapps/common/Lossless Scaling/Lossless.dll'
allow_fp16 = true # Apple GPUs have fast fp16, leave this on

[[profile]]
name = "2x FG / 100%"
# Matched against the end of the Windows .exe path (Wine), the macOS executable path,
# or the process name. Set LSFGVK_PROFILE=<name> to force a profile.
active_in = [
    'Game.exe',
]
multiplier = 2
flow_scale = 1.0
performance_mode = false
pacing = 'none' # 'none' (vsync only, like lsfg-vk) or 'duration' (spread frames evenly)
force_vsync = true

[[profile]]
name = "3x FG / 75% [Performance]"
active_in = 'lsfg-metal-testapp'
multiplier = 3
flow_scale = 0.75
performance_mode = true
)";
        ofs.close();
    } catch (const std::filesystem::filesystem_error& e) {
        throw ls::error("unable to create default configuration file", e);
    }
}

ConfigFile::ConfigFile() {
    this->globalConf = { .allow_fp16 = true };
    this->profileConfs.emplace_back(GameConf {
        .name = "2x FG / 100%",
        .active_in = { "Game.exe" },
        .multiplier = 2
    });
}

namespace {
    /// parse an activity array from toml value
    std::vector<std::string> activityFromString(const toml::node_view<const toml::node>& val) {
        std::vector<std::string> active_in{};

        if (const auto& as_str = val.value<std::string>())
            active_in.push_back(*as_str);

        if (const auto& as_arr = val.as_array()) {
            for (const auto& item : *as_arr) {
                if (const auto& item_str = item.value<std::string>())
                    active_in.push_back(*item_str);
            }
        }

        return active_in;
    }
    /// parse a pacing method from string
    Pacing pacingFromString(const std::string& str) {
        if (str == "none")
            return Pacing::None;
        if (str == "duration")
            return Pacing::Duration;
        throw ls::error("unknown pacing method: " + str);
    }
    /// validate a profile
    void validate(const GameConf& conf) {
        if (conf.multiplier <= 1)
            throw ls::error("multiplier must be greater than 1");
        if (conf.multiplier > 16)
            throw ls::error("multiplier must be at most 16");
        if (conf.flow_scale < 0.25F || conf.flow_scale > 1.0F)
            throw ls::error("flow_scale must be between 0.25 and 1.0");
    }
    /// parse the global configuration
    GlobalConf parseGlobalConf(const toml::table& tbl) {
        const GlobalConf conf{
            .dll = tbl["dll"].value<std::string>(),
            .allow_fp16 = tbl["allow_fp16"].value_or(true)
        };

        if (conf.dll && !std::filesystem::exists(*conf.dll))
            throw ls::error("path to dll is invalid: " + *conf.dll);

        return conf;
    }
    /// parse a game profile configuration
    GameConf parseGameConf(const toml::table& tbl) {
        GameConf conf{
            .name = tbl["name"].value_or<std::string>("unnamed"),
            .active_in = activityFromString(tbl["active_in"]),
            .gpu = tbl["gpu"].value<std::string>(),
            .multiplier = tbl["multiplier"].value_or(2U),
            .flow_scale = tbl["flow_scale"].value_or(1.0F),
            .performance_mode = tbl["performance_mode"].value_or(false),
            .pacing = pacingFromString(tbl["pacing"].value_or<std::string>("none")),
            .force_vsync = tbl["force_vsync"].value_or(true)
        };
        validate(conf);
        return conf;
    }
    /// parse the global configuration from the environment
    GlobalConf parseGlobalConfFromEnv() {
        GlobalConf conf{ .dll = std::nullopt, .allow_fp16 = true };

        if (const char* dll = env("LSFG_METAL_DLL_PATH", "LSFGVK_DLL_PATH"))
            conf.dll = std::string(dll);
        if (const char* no_fp16 = env("LSFG_METAL_NO_FP16", "LSFGVK_NO_FP16"))
            conf.allow_fp16 = std::string(no_fp16) != "1";

        if (conf.dll && !std::filesystem::exists(*conf.dll))
            throw ls::error("path to dll is invalid: " + *conf.dll);

        return conf;
    }
    /// parse a game profile configuration from the environment
    GameConf parseGameConfFromEnv() {
        GameConf conf{ .name = "(environment)" };

        try {
            if (const char* gpu = env("LSFG_METAL_GPU", "LSFGVK_GPU"))
                conf.gpu = std::string(gpu);
            if (const char* m = env("LSFG_METAL_MULTIPLIER", "LSFGVK_MULTIPLIER"))
                conf.multiplier = static_cast<size_t>(std::stoul(m));
            if (const char* f = env("LSFG_METAL_FLOW_SCALE", "LSFGVK_FLOW_SCALE"))
                conf.flow_scale = std::stof(f);
            if (const char* p = env("LSFG_METAL_PERFORMANCE_MODE", "LSFGVK_PERFORMANCE_MODE"))
                conf.performance_mode = std::string(p) == "1";
            if (const char* p = env("LSFG_METAL_PACING", "LSFGVK_PACING"))
                conf.pacing = pacingFromString(std::string(p));
            if (const char* v = env("LSFG_METAL_FORCE_VSYNC", "LSFGVK_FORCE_VSYNC"))
                conf.force_vsync = std::string(v) != "0";
        } catch (const std::logic_error& e) { // stoul/stof failures
            throw ls::error("invalid number in environment configuration", e);
        }

        validate(conf);
        return conf;
    }
}

ConfigFile::ConfigFile(const std::filesystem::path& path) {
    toml::table table;
    try {
        table = toml::parse_file(path.string());
    } catch (const toml::parse_error& e) {
        throw ls::error("unable to parse configuration", e);
    }

    auto version = table["version"];
    if (!version || !version.is_integer() || *version.as_integer() != 2)
        throw ls::error("unsupported configuration version");

    auto global = table["global"];
    if (global && global.is_table())
        this->globalConf = parseGlobalConf(*global.as_table());

    auto profiles = table["profile"];
    if (profiles && profiles.is_array_of_tables())
        for (const auto& profile : *profiles.as_array())
            this->profileConfs.push_back(parseGameConf(*profile.as_table()));
}

WatchedConfig::WatchedConfig() : path(findConfigurationFile()) {
    if (environmentMode()) {
        auto& config = this->configFile;
        config.global() = parseGlobalConfFromEnv();
        config.profiles() = { parseGameConfFromEnv() };
        return;
    }

    if (!std::filesystem::exists(this->path))
        ConfigFile::createDefaultConfigFile(this->path);

    this->configFile = ConfigFile(this->path);
    std::error_code ec;
    this->last_timestamp = std::filesystem::last_write_time(this->path, ec);
}

bool WatchedConfig::update() {
    if (environmentMode())
        return false;

    std::error_code ec;
    const auto now = std::filesystem::last_write_time(this->path, ec);
    if (ec || now == this->last_timestamp)
        return false;
    this->last_timestamp = now;

    ConfigFile new_config{this->path};
    this->configFile = std::move(new_config);
    return true;
}

std::filesystem::path ls::findConfigurationFile() {
    // always honor the explicit override if set
    if (const char* envPath = env("LSFG_METAL_CONFIG", "LSFGVK_CONFIG"))
        return{envPath};

    // then check the XDG overriden location
    const char* xdgPath = std::getenv("XDG_CONFIG_HOME");
    if (xdgPath && *xdgPath != '\0')
        return std::filesystem::path(xdgPath) / "lsfg-metal" / "conf.toml";

    // fallback to typical user home
    const char* homePath = std::getenv("HOME");
    if (homePath && *homePath != '\0')
        return std::filesystem::path(homePath) / ".config" / "lsfg-metal" / "conf.toml";

    return "/usr/local/etc/lsfg-metal/conf.toml";
}
