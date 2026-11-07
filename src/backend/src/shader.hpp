/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lsfgmetal::backend {

    ///
    /// A compiled compute shader (replacement for vk::Shader).
    ///
    /// The SPIR-V shaders from Lossless.dll use a single descriptor set laid out as:
    ///   binding  0+i  uniform buffer i  -> [[buffer(i)]]
    ///   binding 16+i  sampler i         -> [[sampler(i)]]
    ///   binding 32+i  sampled image i   -> [[texture(i)]]
    ///   binding 48+i  storage image i   -> [[texture(sampledImages + i)]]
    ///
    struct Shader {
        std::string name;
        id<MTLComputePipelineState> pipeline;
        MTLSize threadsPerThreadgroup;

        size_t sampledImages{};
        size_t storageImages{};
        size_t buffers{};
        size_t samplers{};
    };

    /// options for shader compilation
    struct ShaderCompileOptions {
        bool fastMath{false};
        bool dumpMSL{false}; //!< write generated MSL to $TMPDIR/lsfg-metal-msl/
    };

    ///
    /// Translate a SPIR-V compute shader to MSL and build a compute pipeline.
    ///
    /// @throws std::exception on failure (SPIRV-Cross or Metal errors)
    ///
    Shader compileShader(id<MTLDevice> device, const std::string& name,
        const std::vector<uint8_t>& spirv,
        size_t sampledImages, size_t storageImages,
        size_t buffers, size_t samplers,
        const ShaderCompileOptions& options);

    ///
    /// Translate SPIR-V to MSL only (no Metal compilation). Used by the CLI.
    ///
    /// @param threadgroupSize receives the shader's local workgroup size.
    ///
    std::string translateShader(const std::vector<uint8_t>& spirv,
        size_t sampledImages, size_t storageImages,
        size_t buffers, size_t samplers,
        MTLSize& threadgroupSize, std::string& entryPoint);

}
