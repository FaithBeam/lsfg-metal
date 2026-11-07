/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "shader.hpp"

#include "spirv_msl.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lsfgmetal::backend;
namespace sc = SPIRV_CROSS_NAMESPACE;

namespace {
    /// binding ranges used by lsfg-vk's descriptor set layouts
    constexpr uint32_t BUFFER_BASE = 0;
    constexpr uint32_t SAMPLER_BASE = 16;
    constexpr uint32_t SAMPLED_BASE = 32;
    constexpr uint32_t STORAGE_BASE = 48;

    void addBinding(sc::CompilerMSL& compiler, uint32_t binding,
            uint32_t buffer, uint32_t texture, uint32_t sampler) {
        sc::MSLResourceBinding rb;
        rb.stage = spv::ExecutionModelGLCompute;
        rb.desc_set = 0;
        rb.binding = binding;
        rb.count = 1;
        rb.msl_buffer = buffer;
        rb.msl_texture = texture;
        rb.msl_sampler = sampler;
        compiler.add_msl_resource_binding(rb);
    }

    /// verify that all declared resources fall into the expected binding ranges
    void verifyBindings(const sc::CompilerMSL& compiler,
            size_t sampledImages, size_t storageImages,
            size_t buffers, size_t samplers) {
        const sc::ShaderResources res = compiler.get_shader_resources();

        const auto check = [&](const sc::SmallVector<sc::Resource>& list,
                uint32_t base, size_t count, const char* kind) {
            for (const auto& r : list) {
                const uint32_t set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
                const uint32_t binding = compiler.get_decoration(r.id, spv::DecorationBinding);
                if (set != 0 || binding < base || binding >= base + count)
                    throw std::runtime_error(std::string("unexpected ") + kind
                        + " at set " + std::to_string(set) + " binding " + std::to_string(binding));
            }
        };
        check(res.uniform_buffers, BUFFER_BASE, buffers, "uniform buffer");
        check(res.separate_samplers, SAMPLER_BASE, samplers, "sampler");
        check(res.separate_images, SAMPLED_BASE, sampledImages, "sampled image");
        check(res.storage_images, STORAGE_BASE, storageImages, "storage image");

        if (!res.storage_buffers.empty() || !res.sampled_images.empty() || !res.push_constant_buffers.empty())
            throw std::runtime_error("shader uses unsupported resource types "
                "(storage buffers, combined image samplers or push constants)");
    }

    uint32_t workgroupDim(const sc::CompilerMSL& compiler, const sc::SpecializationConstant& spec,
            uint32_t index) {
        if (spec.id != sc::ID(0))
            return compiler.get_constant(spec.id).scalar();
        return compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, index);
    }
}

std::string lsfgmetal::backend::translateShader(const std::vector<uint8_t>& spirv,
        size_t sampledImages, size_t storageImages,
        size_t buffers, size_t samplers,
        MTLSize& threadgroupSize, std::string& entryPoint) {
    if (spirv.size() < 20 || spirv.size() % 4 != 0)
        throw std::runtime_error("invalid SPIR-V size");

    std::vector<uint32_t> words(spirv.size() / 4);
    std::memcpy(words.data(), spirv.data(), spirv.size());
    if (words.at(0) != 0x07230203)
        throw std::runtime_error("invalid SPIR-V magic number");

    sc::CompilerMSL compiler(std::move(words));

    // pick the compute entry point
    const auto entryPoints = compiler.get_entry_points_and_stages();
    const sc::EntryPoint* compute = nullptr;
    for (const auto& ep : entryPoints)
        if (ep.execution_model == spv::ExecutionModelGLCompute) {
            compute = &ep;
            break;
        }
    if (!compute)
        throw std::runtime_error("SPIR-V module has no compute entry point");
    compiler.set_entry_point(compute->name, compute->execution_model);

    sc::CompilerMSL::Options opts = compiler.get_msl_options();
    opts.platform = sc::CompilerMSL::Options::macOS;
    opts.set_msl_version(2, 4);
    opts.argument_buffers = false;
    opts.enable_decoration_binding = false;
    compiler.set_msl_options(opts);

    verifyBindings(compiler, sampledImages, storageImages, buffers, samplers);

    for (uint32_t i = 0; i < buffers; i++)
        addBinding(compiler, BUFFER_BASE + i, i, 0, 0);
    for (uint32_t i = 0; i < samplers; i++)
        addBinding(compiler, SAMPLER_BASE + i, 0, 0, i);
    for (uint32_t i = 0; i < sampledImages; i++)
        addBinding(compiler, SAMPLED_BASE + i, 0, i, 0);
    for (uint32_t i = 0; i < storageImages; i++)
        addBinding(compiler, STORAGE_BASE + i, 0, static_cast<uint32_t>(sampledImages) + i, 0);

    std::string source = compiler.compile();

    entryPoint = compiler.get_cleansed_entry_point_name(compute->name, compute->execution_model);

    sc::SpecializationConstant sx, sy, sz;
    compiler.get_work_group_size_specialization_constants(sx, sy, sz);
    threadgroupSize = MTLSizeMake(
        std::max<uint32_t>(workgroupDim(compiler, sx, 0), 1),
        std::max<uint32_t>(workgroupDim(compiler, sy, 1), 1),
        std::max<uint32_t>(workgroupDim(compiler, sz, 2), 1));

    return source;
}

Shader lsfgmetal::backend::compileShader(id<MTLDevice> device, const std::string& name,
        const std::vector<uint8_t>& spirv,
        size_t sampledImages, size_t storageImages,
        size_t buffers, size_t samplers,
        const ShaderCompileOptions& options) {
    MTLSize tg{};
    std::string entry;
    std::string source;
    try {
        source = translateShader(spirv, sampledImages, storageImages, buffers, samplers, tg, entry);
    } catch (const std::exception& e) {
        throw std::runtime_error("SPIR-V to MSL translation of " + name + " failed: " + e.what());
    }

    if (options.dumpMSL) {
        const char* tmp = std::getenv("TMPDIR");
        const std::filesystem::path dir = std::filesystem::path(tmp ? tmp : "/tmp") / "lsfg-metal-msl";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::ofstream(dir / (name + ".metal")) << source;
    }

    @autoreleasepool {
        MTLCompileOptions* compileOptions = [MTLCompileOptions new];
        compileOptions.languageVersion = MTLLanguageVersion2_4;
        if (@available(macOS 15.0, *)) {
            compileOptions.mathMode = options.fastMath ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            compileOptions.fastMathEnabled = options.fastMath;
#pragma clang diagnostic pop
        }

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str())
            options:compileOptions error:&error];
        if (!library)
            throw std::runtime_error("Metal compilation of " + name + " failed: "
                + std::string(error.localizedDescription.UTF8String ?: "unknown error"));

        id<MTLFunction> function = [library newFunctionWithName:@(entry.c_str())];
        if (!function)
            throw std::runtime_error("entry point " + entry + " not found in " + name);

        const NSUInteger threads = tg.width * tg.height * tg.depth;

        MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = function;
        desc.label = @(name.c_str());
        desc.maxTotalThreadsPerThreadgroup = threads;

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:desc
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline)
            throw std::runtime_error("pipeline creation for " + name + " failed: "
                + std::string(error.localizedDescription.UTF8String ?: "unknown error"));

        if (pipeline.maxTotalThreadsPerThreadgroup < threads)
            throw std::runtime_error("shader " + name + " needs " + std::to_string(threads)
                + " threads per threadgroup, device supports "
                + std::to_string(pipeline.maxTotalThreadsPerThreadgroup));

        return Shader {
            .name = name,
            .pipeline = pipeline,
            .threadsPerThreadgroup = tg,
            .sampledImages = sampledImages,
            .storageImages = storageImages,
            .buffers = buffers,
            .samplers = samplers
        };
    }
}
