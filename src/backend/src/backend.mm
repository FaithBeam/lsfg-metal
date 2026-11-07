/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/lsfgvk.cpp */

#include "lsfg-metal/backend.hpp"
#include "dll_reader.hpp"
#include "shader_registry.hpp"
#include "shaderchains.hpp"
#include "types.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace lsfgmetal;
using namespace lsfgmetal::backend;

namespace lsfgmetal::backend {

    class InstanceImpl {
    public:
        InstanceImpl(id<MTLDevice> device,
                std::unordered_map<uint32_t, std::vector<uint8_t>> resources,
                bool allowLowPrecision)
            : device(device),
              shaders(std::make_shared<ShaderRegistry>(device, allowLowPrecision,
                  std::move(resources))) {}

        id<MTLDevice> device;
        std::shared_ptr<ShaderRegistry> shaders;
    };

    class ContextImpl {
    public:
        ContextImpl(const InstanceImpl& instance, Extent extent, bool hdr, bool hdrSupport, float flow,
            bool perf, size_t count);

        void encode(id<MTLCommandBuffer> commandBuffer);

        std::shared_ptr<ShaderRegistry> registry; // keep shaders alive

        MTLPixelFormat format;
        Extent extent;
        std::pair<Image, Image> sourceImages;
        std::vector<Image> destImages;
        Image blackImage;
        bool blackCleared{false};

        uint64_t fidx{0}; // real frame index

        Ctx ctx;

        std::optional<Mipmaps> mipmaps;
        std::vector<Alpha0> alpha0;
        std::vector<Alpha1> alpha1;
        std::optional<Beta0> beta0;
        std::optional<Beta1> beta1;
        struct Pass {
            std::vector<Gamma0> gamma0;
            std::vector<Gamma1> gamma1;
            std::vector<Delta0> delta0;
            std::vector<Delta1> delta1;
            std::optional<Generate> generate;
        };
        std::vector<Pass> passes;
    };
}

// ---------------------------------------------------------------- Instance

namespace {
    std::unordered_map<uint32_t, std::vector<uint8_t>> loadResources(
            const std::filesystem::path& shaderDllPath) {
        try {
            return extractResourcesFromDLL(shaderDllPath);
        } catch (const std::exception& e) {
            throw backend::error("Unable to parse Lossless Scaling DLL", e);
        }
    }
}

Instance::Instance(id<MTLDevice> device,
        const std::filesystem::path& shaderDllPath,
        bool allowLowPrecision)
    : Instance(device, loadResources(shaderDllPath), allowLowPrecision) {}

Instance::Instance(id<MTLDevice> device,
        std::unordered_map<uint32_t, std::vector<uint8_t>> resources,
        bool allowLowPrecision) {
    if (!device)
        throw backend::error("No Metal device");
    try {
        this->m_impl = std::make_unique<InstanceImpl>(device, std::move(resources), allowLowPrecision);
    } catch (const std::exception& e) {
        throw backend::error("Unable to build shader registry", e);
    }
}

void Instance::prepare(bool perf) {
    try {
        this->m_impl->shaders->prepare(perf);
    } catch (const std::exception& e) {
        throw backend::error("Unable to compile shaders", e);
    }
}

bool Instance::isPrepared(bool perf) const {
    return this->m_impl->shaders->isPrepared(perf);
}

id<MTLDevice> Instance::device() const {
    return this->m_impl->device;
}

bool Instance::isFP16() const {
    return this->m_impl->shaders->isFP16();
}

std::unique_ptr<Context> Instance::openContext(uint32_t width, uint32_t height,
        bool hdr, float flow, bool perf, size_t count, bool hdrSupport) {
    if (count < 1)
        throw backend::error("frame count must be at least 1");
    if (width < 64 || height < 64)
        throw backend::error("frame size too small for frame generation");

    this->prepare(perf);
    try {
        return std::make_unique<Context>(std::make_unique<ContextImpl>(*this->m_impl,
            Extent{ width, height }, hdr, hdr && hdrSupport, flow, perf, count));
    } catch (const std::exception& e) {
        throw backend::error("Unable to create context", e);
    }
}

Instance::~Instance() = default;

// ---------------------------------------------------------------- Context

namespace {
    Ctx createCtx(const InstanceImpl& instance, Extent extent,
            bool hdr, bool hdrSupport, float flow, bool perf, size_t count) {
        id<MTLDevice> device = instance.device;

        std::vector<id<MTLBuffer>> constantBuffers;
        constantBuffers.reserve(count);
        for (size_t i = 0; i < count; ++i)
            constantBuffers.push_back(createBuffer(device, getDefaultConstantBuffer(i, count, flow, hdrSupport)));

        return Ctx {
            .device = device,
            .shaders = instance.shaders.get(),
            .constantBuffer = createBuffer(device, getDefaultConstantBuffer(0, 1, flow, hdrSupport)),
            .constantBuffers = std::move(constantBuffers),
            .bnbSampler = createSampler(device, MTLSamplerAddressModeClampToBorderColor, false),
            .bnwSampler = createSampler(device, MTLSamplerAddressModeClampToBorderColor, true),
            .eabSampler = createSampler(device, MTLSamplerAddressModeClampToEdge, false),
            .sourceExtent = extent,
            .flowExtent = Extent {
                .width = static_cast<uint32_t>(static_cast<float>(extent.width) / flow),
                .height = static_cast<uint32_t>(static_cast<float>(extent.height) / flow)
            },
            .hdr = hdr,
            .flow = flow,
            .perf = perf,
            .count = count
        };
    }
}

ContextImpl::ContextImpl(const InstanceImpl& instance, Extent extent, bool hdr, bool hdrSupport, float flow,
        bool perf, size_t count) :
        registry(instance.shaders),
        format(hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA8Unorm),
        extent(extent),
        sourceImages(
            Image(instance.device, extent, format, MTLTextureUsageRenderTarget),
            Image(instance.device, extent, format, MTLTextureUsageRenderTarget)),
        blackImage(instance.device, Extent{ 4, 4 }, MTLPixelFormatRGBA8Unorm, MTLTextureUsageRenderTarget),
        ctx(createCtx(instance, extent, hdr, hdrSupport, flow, perf, count)) {
    this->destImages.reserve(count);
    for (size_t i = 0; i < count; i++)
        this->destImages.emplace_back(instance.device, extent, format);

    this->mipmaps.emplace(ctx, sourceImages);

    this->alpha0.reserve(7);
    for (size_t i = 0; i < 7; i++)
        this->alpha0.emplace_back(ctx, mipmaps->getImages().at(i));

    this->alpha1.reserve(7);
    for (size_t i = 0; i < 7; i++)
        this->alpha1.emplace_back(ctx, i == 0 ? 3 : 2, alpha0.at(i).getImages());

    this->beta0.emplace(ctx, alpha1.at(0).getImages());
    this->beta1.emplace(ctx, beta0->getImages());

    // build main passes
    for (size_t i = 0; i < destImages.size(); ++i) {
        auto& pass = this->passes.emplace_back();

        pass.gamma0.reserve(7);
        pass.gamma1.reserve(7);
        pass.delta0.reserve(3);
        pass.delta1.reserve(3);
        for (size_t j = 0; j < 7; j++) {
            if (j == 0) { // first pass has no prior data
                pass.gamma0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage
                );
                pass.gamma1.emplace_back(ctx, i,
                    pass.gamma0.at(j).getImages(),
                    this->blackImage,
                    this->beta1->getImages().at(5)
                );
            } else { // other passes use prior data
                pass.gamma0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.gamma1.emplace_back(ctx, i,
                    pass.gamma0.at(j).getImages(),
                    pass.gamma1.at(j - 1).getImage(),
                    this->beta1->getImages().at(6 - j)
                );
            }

            if (j == 4) { // first special pass has no prior data
                pass.delta0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    this->blackImage,
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(ctx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    this->blackImage,
                    this->beta1->getImages().at(6 - j),
                    this->blackImage
                );
            } else if (j > 4) { // further passes do
                pass.delta0.emplace_back(ctx, i,
                    this->alpha1.at(6 - j).getImages(),
                    pass.delta1.at(j - 5).getImage0(),
                    pass.gamma1.at(j - 1).getImage()
                );
                pass.delta1.emplace_back(ctx, i,
                    pass.delta0.at(j - 4).getImages0(),
                    pass.delta0.at(j - 4).getImages1(),
                    pass.delta1.at(j - 5).getImage0(),
                    this->beta1->getImages().at(6 - j),
                    pass.delta1.at(j - 5).getImage1()
                );
            }
        }

        pass.generate.emplace(ctx, i,
            this->sourceImages,
            pass.gamma1.at(6).getImage(),
            pass.delta1.at(2).getImage0(),
            pass.delta1.at(2).getImage1(),
            this->destImages.at(i)
        );
    }
}

void ContextImpl::encode(id<MTLCommandBuffer> commandBuffer) {
    // the black image is read as "no prior data"; make sure it really is black
    if (!this->blackCleared) {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = this->blackImage.handle();
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        id<MTLRenderCommandEncoder> clear = [commandBuffer renderCommandEncoderWithDescriptor:rp];
        clear.label = @"lsfg-metal clear";
        [clear endEncoding];
        this->blackCleared = true;
    }

    id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
    if (!enc)
        throw backend::error("Unable to create compute command encoder");
    enc.label = @"lsfg-metal frame generation";

    // pre-pass
    [enc pushDebugGroup:@"pre-pass"];
    this->mipmaps->render(enc, this->fidx);
    for (size_t i = 0; i < 7; ++i) {
        this->alpha0.at(6 - i).render(enc);
        this->alpha1.at(6 - i).render(enc, this->fidx);
    }
    this->beta0->render(enc, this->fidx);
    this->beta1->render(enc);
    [enc popDebugGroup];

    // main passes
    for (size_t i = 0; i < this->destImages.size(); i++) {
        [enc pushDebugGroup:[NSString stringWithFormat:@"generate %zu", i]];
        const auto& pass = this->passes.at(i);
        for (size_t j = 0; j < 7; j++) {
            pass.gamma0.at(j).render(enc, this->fidx);
            pass.gamma1.at(j).render(enc);

            if (j < 4) continue;
            pass.delta0.at(j - 4).render(enc, this->fidx);
            pass.delta1.at(j - 4).render(enc);
        }
        pass.generate->render(enc, this->fidx);
        [enc popDebugGroup];
    }

    [enc endEncoding];
    this->fidx++;
}

Context::Context(std::unique_ptr<ContextImpl> impl) : m_impl(std::move(impl)) {}
Context::~Context() = default;

id<MTLTexture> Context::nextSourceTexture() const {
    return (this->m_impl->fidx % 2 == 0)
        ? this->m_impl->sourceImages.first.handle()
        : this->m_impl->sourceImages.second.handle();
}

id<MTLTexture> Context::destTexture(size_t i) const {
    return this->m_impl->destImages.at(i).handle();
}

size_t Context::count() const { return this->m_impl->destImages.size(); }
MTLPixelFormat Context::format() const { return this->m_impl->format; }
uint32_t Context::width() const { return this->m_impl->extent.width; }
uint32_t Context::height() const { return this->m_impl->extent.height; }
uint64_t Context::frameIndex() const { return this->m_impl->fidx; }

void Context::encode(id<MTLCommandBuffer> commandBuffer) {
    try {
        this->m_impl->encode(commandBuffer);
    } catch (const std::exception& e) {
        throw backend::error("Unable to schedule frames", e);
    }
}
