/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/extraction/shader_registry.cpp */

#include "shader_registry.hpp"
#include "shader.hpp"
#include "lsfg-metal-common/log.hpp"

#include <dispatch/dispatch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lsfgmetal::backend;

namespace {
    /// get the source code for a shader
    const std::vector<uint8_t>& getShaderSource(uint32_t id, bool fp16, bool perf,
            const std::unordered_map<uint32_t, std::vector<uint8_t>>& resources) {
        const size_t BASE_OFFSET = 49;
        const size_t OFFSET_PERF = 23;
        const size_t OFFSET_FP32 = 49;

        auto it = resources.find(static_cast<uint32_t>(BASE_OFFSET + id +
            (perf ? OFFSET_PERF : 0) +
            (fp16 ? 0 : OFFSET_FP32)));
        if (it == resources.end())
            throw std::runtime_error("unable to find shader with id: " + std::to_string(id));

        return it->second;
    }

    /// a pending compilation job
    struct Job {
        std::string name;
        std::vector<uint8_t> spirv;
        size_t p1, p2, p3, p4; // sampled, storage, buffers, samplers
        Shader* out;
    };

    bool envFlag(const char* name) {
        const char* v = std::getenv(name);
        return v && *v && std::string(v) != "0";
    }

    /// compile jobs in parallel
    void runJobs(id<MTLDevice> device, std::vector<Job>& jobs) {
        const ShaderCompileOptions options{
            .fastMath = envFlag("LSFG_METAL_FAST_MATH"),
            .dumpMSL = envFlag("LSFG_METAL_DUMP_MSL")
        };

        std::mutex errorMutex;
        std::string firstError;

        Job* data = jobs.data();
        std::mutex* errorMutexPtr = &errorMutex;
        std::string* firstErrorPtr = &firstError;
        const ShaderCompileOptions* optionsPtr = &options;
        dispatch_apply(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            Job& job = data[i];
            try {
                *job.out = compileShader(device, job.name, job.spirv,
                    job.p1, job.p2, job.p3, job.p4, *optionsPtr);
            } catch (const std::exception& e) {
                const std::lock_guard<std::mutex> lock(*errorMutexPtr);
                if (firstErrorPtr->empty())
                    *firstErrorPtr = e.what();
            }
        });

        if (!firstError.empty())
            throw std::runtime_error(firstError);
    }
}

void lsfgmetal::backend::patchGenerateShader(std::vector<uint8_t>& data, bool hdr) {
    const std::span<uint32_t> words(
        reinterpret_cast<uint32_t*>(data.data()),
        data.size() / sizeof(uint32_t)
    );

    const uint16_t SpvOpCapability = 17;
    const uint16_t SpvOpTypeImage = 25;
    const uint32_t SpvCapabilityStorageImageWriteWithoutFormat = 56;
    const uint32_t SpvCapabilityShader = 1;
    const uint32_t SpvImageFormatRgba16f = 2;
    const uint32_t SpvImageFormatRgba8 = 4;

    for (size_t i = 5; i < words.size();) {
        const uint32_t word = words[i];
        const uint16_t wc = (word >> 16);
        const uint16_t op = word & 0xFFFF;

        // remove write without format capability
        if (op == SpvOpCapability && wc >= 2 && i + 1 < words.size()) {
            uint32_t& cap = words[i + 1];
            if (cap == SpvCapabilityStorageImageWriteWithoutFormat)
                cap = SpvCapabilityShader;
        }

        // patch format in image instructions
        if (op == SpvOpTypeImage && wc >= 9 && i + 8 < words.size()) {
            const uint32_t sampled = words[i + 7];
            if (sampled == 2)
                words[i + 8] = hdr ? SpvImageFormatRgba16f : SpvImageFormatRgba8;
        }

        i += wc ? wc : 1;
    }
}

ShaderRegistry::ShaderRegistry(id<MTLDevice> device, bool fp16,
        std::unordered_map<uint32_t, std::vector<uint8_t>> resources)
    : device(device), fp16(fp16), resources(std::move(resources)) {
    // fail early if the dll does not contain the expected resources
    (void)getShaderSource(255, fp16, false, this->resources);
    (void)getShaderSource(256, fp16, false, this->resources);
}

bool ShaderRegistry::isPrepared(bool perf) const {
    const std::lock_guard<std::mutex> lock(this->mutex);
    return this->mipmapsShader.has_value() && (perf ? this->performance : this->quality) != nullptr;
}

void ShaderRegistry::prepare(bool perf) {
    const std::lock_guard<std::mutex> lock(this->mutex);

    std::vector<Job> jobs;
    const auto add = [&](Shader& out, const std::string& name, std::vector<uint8_t> spirv,
            size_t p1, size_t p2, size_t p3, size_t p4) {
        jobs.push_back(Job{ name, std::move(spirv), p1, p2, p3, p4, &out });
    };

    // common shaders
    std::optional<Shader> mipmaps, generate, generateHdr;
    if (!this->mipmapsShader) {
        mipmaps.emplace();
        generate.emplace();
        generateHdr.emplace();

        std::vector<uint8_t> generate_data = getShaderSource(256, fp16, false, resources);
        std::vector<uint8_t> generate_data_hdr = generate_data;
        patchGenerateShader(generate_data, false);
        patchGenerateShader(generate_data_hdr, true);

        add(*mipmaps, "mipmaps", getShaderSource(255, fp16, false, resources), 1, 7, 1, 1);
        add(*generate, "generate", std::move(generate_data), 5, 1, 1, 2);
        add(*generateHdr, "generate_hdr", std::move(generate_data_hdr), 5, 1, 1, 2);
    }

    // mode shaders
    std::unique_ptr<Shaders> shaders;
    if (!(perf ? this->performance : this->quality)) {
        shaders = std::make_unique<Shaders>();
        const std::string prefix = perf ? "perf_" : "quality_";
        const auto S = [&](Shader& out, uint32_t id, size_t p1, size_t p2, size_t p3, size_t p4) {
            add(out, prefix + std::to_string(id), getShaderSource(id, fp16, perf, resources),
                p1, p2, p3, p4);
        };

        Shaders& s = *shaders;
        if (!perf) {
            S(s.alpha[0], 267, 1, 2, 0, 1);
            S(s.alpha[1], 268, 2, 2, 0, 1);
            S(s.alpha[2], 269, 2, 4, 0, 1);
            S(s.alpha[3], 270, 4, 4, 0, 1);
            S(s.beta[0], 275, 12, 2, 0, 1);
            S(s.beta[1], 276, 2, 2, 0, 1);
            S(s.beta[2], 277, 2, 2, 0, 1);
            S(s.beta[3], 278, 2, 2, 0, 1);
            S(s.beta[4], 279, 2, 6, 1, 1);
            S(s.gamma[0], 257, 9, 3, 1, 2);
            S(s.gamma[1], 259, 3, 4, 0, 1);
            S(s.gamma[2], 260, 4, 4, 0, 1);
            S(s.gamma[3], 261, 4, 4, 0, 1);
            S(s.gamma[4], 262, 6, 1, 1, 2);
            S(s.delta[0], 257, 9, 3, 1, 2);
            S(s.delta[1], 263, 3, 4, 0, 1);
            S(s.delta[2], 264, 4, 4, 0, 1);
            S(s.delta[3], 265, 4, 4, 0, 1);
            S(s.delta[4], 266, 6, 1, 1, 2);
            S(s.delta[5], 258, 10, 2, 1, 2);
            S(s.delta[6], 271, 2, 2, 0, 1);
            S(s.delta[7], 272, 2, 2, 0, 1);
            S(s.delta[8], 273, 2, 2, 0, 1);
            S(s.delta[9], 274, 3, 1, 1, 2);
        } else {
            S(s.alpha[0], 267, 1, 1, 0, 1);
            S(s.alpha[1], 268, 1, 1, 0, 1);
            S(s.alpha[2], 269, 1, 2, 0, 1);
            S(s.alpha[3], 270, 2, 2, 0, 1);
            S(s.beta[0], 275, 6, 2, 0, 1);
            S(s.beta[1], 276, 2, 2, 0, 1);
            S(s.beta[2], 277, 2, 2, 0, 1);
            S(s.beta[3], 278, 2, 2, 0, 1);
            S(s.beta[4], 279, 2, 6, 1, 1);
            S(s.gamma[0], 257, 5, 3, 1, 2);
            S(s.gamma[1], 259, 3, 2, 0, 1);
            S(s.gamma[2], 260, 2, 2, 0, 1);
            S(s.gamma[3], 261, 2, 2, 0, 1);
            S(s.gamma[4], 262, 4, 1, 1, 2);
            S(s.delta[0], 257, 5, 3, 1, 2);
            S(s.delta[1], 263, 3, 2, 0, 1);
            S(s.delta[2], 264, 2, 2, 0, 1);
            S(s.delta[3], 265, 2, 2, 0, 1);
            S(s.delta[4], 266, 4, 1, 1, 2);
            S(s.delta[5], 258, 6, 1, 1, 2);
            S(s.delta[6], 271, 1, 1, 0, 1);
            S(s.delta[7], 272, 1, 1, 0, 1);
            S(s.delta[8], 273, 1, 1, 0, 1);
            S(s.delta[9], 274, 2, 1, 1, 2);
        }
    }

    if (jobs.empty())
        return;

    const auto start = std::chrono::steady_clock::now();
    runJobs(this->device, jobs);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    ls::log::info("compiled " + std::to_string(jobs.size()) + " shaders ("
        + (perf ? "performance" : "quality") + ", " + (fp16 ? "fp16" : "fp32")
        + ") in " + std::to_string(ms) + " ms");

    if (mipmaps) {
        this->mipmapsShader = std::move(mipmaps);
        this->generateShader = std::move(generate);
        this->generateHdrShader = std::move(generateHdr);
    }
    if (shaders)
        (perf ? this->performance : this->quality) = std::move(shaders);
}
