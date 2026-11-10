/* SPDX-License-Identifier: GPL-3.0-or-later */
/* macOS port of lsfg-vk-common/src/configuration/detection.cpp and helpers/paths.cpp */

#include "lsfg-metal-common/detection.hpp"
#include "lsfg-metal-common/config.hpp"
#include "lsfg-metal-common/errors.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <libproc.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdlib.h>
#include <unistd.h>

using namespace ls;
namespace fs = std::filesystem;

namespace {
    std::optional<GameConf> matchByName(const std::vector<GameConf>& profiles, const std::string& id) {
        for (const auto& profile : profiles)
            if (profile.name == id)
                return profile;
        return std::nullopt;
    }
    std::optional<GameConf> matchById(const std::vector<GameConf>& profiles, const std::string& id) {
        for (const auto& profile : profiles)
            for (const auto& activation : profile.active_in)
                if (id == activation)
                    return profile;
        return std::nullopt;
    }
    std::optional<GameConf> matchEndsWithId(const std::vector<GameConf>& profiles, const std::string& id) {
        for (const auto& profile : profiles)
            for (const auto& activation : profile.active_in)
                if (!activation.empty() && id.ends_with(activation))
                    return profile;
        return std::nullopt;
    }

    std::string lower(std::string s) {
        std::ranges::transform(s, s.begin(), [](unsigned char c) { return std::tolower(c); });
        return s;
    }

    /// enumerate the backing files of all mapped memory regions (macOS replacement for /proc/self/maps)
    std::vector<std::string> mappedFiles() {
        std::vector<std::string> files;
        std::set<std::string> seen;

        const pid_t pid = getpid();
        mach_vm_address_t addr = 0;
        for (size_t guard = 0; guard < 1'000'000; guard++) {
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            const kern_return_t kr = mach_vm_region(mach_task_self(), &addr, &size,
                VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object);
            if (kr != KERN_SUCCESS)
                break;

            std::array<char, PROC_PIDPATHINFO_MAXSIZE> buf{};
            const int len = proc_regionfilename(pid, addr, buf.data(), static_cast<uint32_t>(buf.size()));
            if (len > 0) {
                std::string path(buf.data(), static_cast<size_t>(len));
                if (seen.insert(path).second)
                    files.push_back(std::move(path));
            }

            if (size == 0) break;
            addr += size;
        }
        return files;
    }

    std::string executablePath() {
        uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::string buf(size, '\0');
        if (_NSGetExecutablePath(buf.data(), &size) != 0)
            return {};
        buf.resize(std::char_traits<char>::length(buf.c_str()));

        std::array<char, PATH_MAX> resolved{};
        if (realpath(buf.c_str(), resolved.data()))
            return {resolved.data()};
        return buf;
    }
}

bool ls::runningUnderWine() {
    if (std::getenv("WINEPREFIX") || std::getenv("WINELOADER") || std::getenv("WINESERVER"))
        return true;
    const std::string exe = lower(executablePath());
    return exe.find("wine") != std::string::npos
        || exe.find("crossover") != std::string::npos;
}

Identification ls::identify() {
    Identification id{};

    for (const char* name : { "LSFG_METAL_PROFILE", "LSFGVK_PROFILE" }) {
        const char* override = std::getenv(name);
        if (override && *override != '\0') {
            id.override = std::string(override);
            break;
        }
    }

    id.executable = executablePath();

    if (runningUnderWine()) {
        // the game's .exe is mmap'd by Wine; system executables are less interesting
        std::optional<std::string> fallback;
        for (const auto& file : mappedFiles()) {
            const std::string l = lower(file);
            if (!l.ends_with(".exe"))
                continue;
            const bool system = l.find("/windows/system32/") != std::string::npos
                || l.find("/windows/syswow64/") != std::string::npos;
            if (!system) {
                id.wine_executable = file;
                break;
            }
            if (!fallback) fallback = file;
        }
        if (!id.wine_executable)
            id.wine_executable = fallback;
    }

    if (const char* prog = getprogname())
        id.process_name = prog;

    return id;
}

std::optional<std::pair<IdentType, GameConf>> ls::findProfile(
        const ConfigFile& config, const Identification& id) {
    const auto& profiles = config.profiles();
    if (profiles.empty())
        return std::nullopt;

    if (environmentMode())
        return std::make_pair(IdentType::OVERRIDE, profiles.front());

    if (id.override.has_value()) {
        const auto profile = matchByName(profiles, id.override.value());
        if (profile.has_value())
            return std::make_pair(IdentType::OVERRIDE, profile.value());
    }

    if (id.wine_executable.has_value()) {
        const auto wine_profile = matchEndsWithId(profiles, id.wine_executable.value());
        if (wine_profile.has_value())
            return std::make_pair(IdentType::WINE_EXECUTABLE, wine_profile.value());
    }

    const auto exe_profile = matchEndsWithId(profiles, id.executable);
    if (exe_profile.has_value())
        return std::make_pair(IdentType::EXECUTABLE, exe_profile.value());

    if (!id.process_name.empty()) {
        const auto proc_profile = matchById(profiles, id.process_name);
        if (proc_profile.has_value())
            return std::make_pair(IdentType::PROCESS_NAME, proc_profile.value());
    }

    return std::nullopt;
}

namespace {
    /// relative path of Lossless.dll inside a Steam library
    /// (a function, not a global: the hook library must not have static initializers, see hooks.mm)
    fs::path lsRel() {
        return fs::path("steamapps") / "common" / "Lossless Scaling" / "Lossless.dll";
    }

    /// list subdirectories of a directory (non-recursive, errors ignored)
    std::vector<fs::path> subdirs(const fs::path& dir) {
        std::vector<fs::path> out;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec))
            if (entry.is_directory(ec))
                out.push_back(entry.path());
        return out;
    }

    /// map a Windows path (e.g. "D:\\SteamLibrary") into a prefix
    std::optional<fs::path> windowsToUnix(const fs::path& prefix, std::string winPath) {
        if (winPath.size() < 2 || winPath[1] != ':')
            return std::nullopt;
        const std::string drive = lower(winPath.substr(0, 2));
        std::string rest = winPath.substr(2);
        std::ranges::replace(rest, '\\', '/');
        while (!rest.empty() && rest.front() == '/') rest.erase(rest.begin());

        fs::path root = prefix / "dosdevices" / drive;
        std::error_code ec;
        if (!fs::exists(root, ec)) {
            if (drive == "c:") root = prefix / "drive_c";
            else return std::nullopt;
        }
        return root / rest;
    }

    /// collect Steam library roots inside a Wine prefix
    std::vector<fs::path> steamLibrariesInPrefix(const fs::path& prefix) {
        std::vector<fs::path> roots;
        for (const char* pf : { "Program Files (x86)", "Program Files" }) {
            const fs::path steam = prefix / "drive_c" / pf / "Steam";
            std::error_code ec;
            if (!fs::exists(steam, ec))
                continue;
            roots.push_back(steam);

            // additional libraries from libraryfolders.vdf
            std::ifstream vdf(steam / "steamapps" / "libraryfolders.vdf");
            if (!vdf.is_open())
                continue;
            static const std::regex PATH_RE(R"re("path"\s+"([^"]+)")re");
            std::string line;
            while (std::getline(vdf, line)) {
                std::smatch m;
                if (!std::regex_search(line, m, PATH_RE))
                    continue;
                std::string p = m[1].str();
                // vdf escapes backslashes
                std::string unescaped;
                for (size_t i = 0; i < p.size(); i++) {
                    if (p[i] == '\\' && i + 1 < p.size() && p[i + 1] == '\\') i++;
                    unescaped += p[i];
                }
                if (auto mapped = windowsToUnix(prefix, unescaped))
                    roots.push_back(*mapped);
            }
        }
        return roots;
    }

    std::optional<fs::path> searchPrefix(const fs::path& prefix) {
        for (const auto& root : steamLibrariesInPrefix(prefix)) {
            const fs::path candidate = root / lsRel();
            std::error_code ec;
            if (fs::exists(candidate, ec))
                return candidate;
        }
        return std::nullopt;
    }
}

fs::path ls::findShaderDll(const std::optional<std::string>& configured) {
    if (configured) {
        if (fs::exists(*configured))
            return *configured;
        throw ls::error("configured Lossless.dll does not exist: " + *configured);
    }

    std::vector<fs::path> prefixes;
    if (const char* wp = std::getenv("WINEPREFIX"); wp && *wp)
        prefixes.emplace_back(wp);

    const char* homeEnv = std::getenv("HOME");
    const fs::path home = (homeEnv && *homeEnv) ? fs::path(homeEnv) : fs::path("/");

    // CrossOver
    for (const auto& b : subdirs(home / "Library/Application Support/CrossOver/Bottles"))
        prefixes.push_back(b);
    // Sikarugir / Kegworks / Wineskin style wrappers
    for (const char* dir : { "Applications/Sikarugir", "Applications/Kegworks", "Applications/Wineskin" })
        for (const auto& app : subdirs(home / dir))
            prefixes.push_back(app / "Contents/SharedSupport/prefix");
    // Heroic
    for (const auto& p : subdirs(home / "Games/Heroic/Prefixes/default"))
        prefixes.push_back(p);
    // plain wine
    prefixes.push_back(home / ".wine");

    for (const auto& prefix : prefixes)
        if (auto found = searchPrefix(prefix))
            return *found;

    // macOS Steam (Windows depot downloaded via steamcmd or similar)
    for (const fs::path& root : {
            home / "Library/Application Support/Steam",
            home / "Library/Application Support/Steam/SteamApps" }) {
        std::error_code ec;
        if (fs::exists(root / lsRel(), ec))
            return root / lsRel();
    }

    // next to the executable / in the working directory
    std::error_code ec;
    const auto local = fs::current_path(ec) / "Lossless.dll";
    if (!ec && fs::exists(local, ec))
        return local;

    throw ls::error("unable to locate Lossless.dll, set 'dll' in the configuration or LSFGVK_DLL_PATH");
}
