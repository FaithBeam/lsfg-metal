/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/shaderchains/ (sources)
 *
 * The structure intentionally mirrors the Vulkan implementation one-to-one so that
 * upstream changes can be carried over easily. Differences:
 *  - vk::Image -> Image (MTLTexture), vk::Sampler -> MTLSamplerState, vk::Buffer -> MTLBuffer
 *  - prepare() (initial layout transitions) is not needed on Metal
 *  - render() records into a compute command encoder instead of a VkCommandBuffer
 */

#include "shaderchains.hpp"

using namespace lsfgmetal::backend;

// ---------------------------------------------------------------- Mipmaps

Mipmaps::Mipmaps(const Ctx& ctx, const std::pair<Image, Image>& sourceImages) {
    // create output images for base and 6 mips
    this->images.reserve(7);
    for (uint32_t i = 0; i < 7; i++)
        this->images.emplace_back(ctx.device,
            shift_extent(ctx.flowExtent, i), MTLPixelFormatR8Unorm);

    // create descriptor sets for both input images
    this->sets.reserve(2);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampled(sourceImages.first)
        .storages(this->images)
        .sampler(ctx.bnbSampler)
        .buffer(ctx.constantBuffer)
        .build(ctx.shaders->mipmaps()));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampled(sourceImages.second)
        .storages(this->images)
        .sampler(ctx.bnbSampler)
        .buffer(ctx.constantBuffer)
        .build(ctx.shaders->mipmaps()));

    this->dispatchExtent = add_shift_extent(ctx.flowExtent, 63, 6);
}

void Mipmaps::render(Enc enc, size_t idx) const {
    this->sets.at(idx % 2).dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Alpha0

Alpha0::Alpha0(const Ctx& ctx, const Image& sourceImage) {
    const size_t m = ctx.perf ? 1 : 2; // multiplier
    const Extent halfExtent = add_shift_extent(sourceImage.getExtent(), 1, 1);
    const Extent quarterExtent = add_shift_extent(halfExtent, 1, 1);

    // create temporary & output images
    this->tempImages0.reserve(m);
    this->tempImages1.reserve(m);
    for (size_t i = 0; i < m; i++) {
        this->tempImages0.emplace_back(ctx.device, halfExtent);
        this->tempImages1.emplace_back(ctx.device, halfExtent);
    }

    this->images.reserve(2 * m);
    for (size_t i = 0; i < (2 * m); i++)
        this->images.emplace_back(ctx.device, quarterExtent);

    // create descriptor sets
    const auto& shaders = ctx.mode();
    this->sets.reserve(3);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampled(sourceImage)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.alpha.at(0)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .storages(this->tempImages1)
        .sampler(ctx.bnbSampler)
        .build(shaders.alpha.at(1)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages1)
        .storages(this->images)
        .sampler(ctx.bnbSampler)
        .build(shaders.alpha.at(2)));

    this->dispatchExtent0 = add_shift_extent(halfExtent, 7, 3);
    this->dispatchExtent1 = add_shift_extent(quarterExtent, 7, 3);
}

void Alpha0::render(Enc enc) const {
    this->sets.at(0).dispatch(enc, this->dispatchExtent0);
    this->sets.at(1).dispatch(enc, this->dispatchExtent0);
    this->sets.at(2).dispatch(enc, this->dispatchExtent1);
}

// ---------------------------------------------------------------- Alpha1

Alpha1::Alpha1(const Ctx& ctx, size_t temporal, const std::vector<Image>& sourceImages) {
    const size_t m = ctx.perf ? 1 : 2; // multiplier
    const Extent quarterExtent = sourceImages.at(0).getExtent();

    // create output images for mod3
    this->images.reserve(temporal);
    for (size_t i = 0; i < temporal; i++) {
        auto& vec = this->images.emplace_back();
        vec.reserve(2 * m);
        for (size_t j = 0; j < (2 * m); j++)
            vec.emplace_back(ctx.device, quarterExtent);
    }

    // create descriptor sets
    const auto& shaders = ctx.mode();
    this->sets.reserve(temporal);
    for (size_t i = 0; i < temporal; i++)
        this->sets.emplace_back(ManagedShaderBuilder()
            .sampleds(sourceImages)
            .storages(this->images.at(i))
            .sampler(ctx.bnbSampler)
            .build(shaders.alpha.at(3)));

    this->dispatchExtent = add_shift_extent(quarterExtent, 7, 3);
}

void Alpha1::render(Enc enc, size_t idx) const {
    this->sets.at(idx % this->sets.size()).dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Beta0

Beta0::Beta0(const Ctx& ctx, const std::vector<std::vector<Image>>& sourceImages) {
    const Extent extent = sourceImages.at(0).at(0).getExtent();

    // create output images
    this->images.reserve(2);
    for (size_t i = 0; i < 2; i++)
        this->images.emplace_back(ctx.device, extent);

    // create descriptor sets
    const auto& shader = ctx.mode().beta.at(0);
    this->sets.reserve(sourceImages.size());
    for (size_t i = 0; i < sourceImages.size(); i++)
        this->sets.emplace_back(ManagedShaderBuilder()
            .sampleds(sourceImages.at((i + (sourceImages.size() - 2)) % sourceImages.size()))
            .sampleds(sourceImages.at((i + (sourceImages.size() - 1)) % sourceImages.size()))
            .sampleds(sourceImages.at(i % sourceImages.size()))
            .storages(this->images)
            .sampler(ctx.bnwSampler)
            .build(shader));

    this->dispatchExtent = add_shift_extent(extent, 7, 3);
}

void Beta0::render(Enc enc, size_t idx) const {
    this->sets.at(idx % this->sets.size()).dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Beta1

Beta1::Beta1(const Ctx& ctx, const std::vector<Image>& sourceImages) {
    const Extent extent = sourceImages.at(0).getExtent();

    // create temporary & output images
    this->tempImages0.reserve(2);
    this->tempImages1.reserve(2);
    for (uint32_t i = 0; i < 2; i++) {
        this->tempImages0.emplace_back(ctx.device, extent);
        this->tempImages1.emplace_back(ctx.device, extent);
    }

    this->images.reserve(6);
    for (uint32_t i = 0; i < 6; i++)
        this->images.emplace_back(ctx.device, shift_extent(extent, i), MTLPixelFormatR8Unorm);

    // create descriptor sets
    const auto& shaders = ctx.mode().beta;
    this->sets.reserve(4);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(sourceImages)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(1)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .storages(this->tempImages1)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(2)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages1)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(3)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .storages(this->images)
        .sampler(ctx.bnbSampler)
        .buffer(ctx.constantBuffer)
        .build(shaders.at(4)));

    this->dispatchExtent0 = add_shift_extent(extent, 7, 3);
    this->dispatchExtent1 = add_shift_extent(extent, 31, 5);
}

void Beta1::render(Enc enc) const {
    this->sets.at(0).dispatch(enc, this->dispatchExtent0);
    this->sets.at(1).dispatch(enc, this->dispatchExtent0);
    this->sets.at(2).dispatch(enc, this->dispatchExtent0);
    this->sets.at(3).dispatch(enc, this->dispatchExtent1);
}

// ---------------------------------------------------------------- Gamma0

Gamma0::Gamma0(const Ctx& ctx, size_t idx,
        const std::vector<std::vector<Image>>& sourceImages,
        const Image& additionalInput) {
    const Extent extent = sourceImages.at(0).at(0).getExtent();

    // create output images
    this->images.reserve(3);
    for (size_t i = 0; i < 3; i++)
        this->images.emplace_back(ctx.device, extent);

    // create descriptor sets
    const auto& shader = ctx.mode().gamma.at(0);
    this->sets.reserve(sourceImages.size());
    for (size_t i = 0; i < sourceImages.size(); i++)
        this->sets.emplace_back(ManagedShaderBuilder()
            .sampleds(sourceImages.at((i + (sourceImages.size() - 1)) % sourceImages.size()))
            .sampleds(sourceImages.at(i % sourceImages.size()))
            .sampled(additionalInput)
            .storages(this->images)
            .sampler(ctx.bnwSampler)
            .sampler(ctx.eabSampler)
            .buffer(ctx.constantBuffers.at(idx))
            .build(shader));

    this->dispatchExtent = add_shift_extent(extent, 7, 3);
}

void Gamma0::render(Enc enc, size_t idx) const {
    this->sets.at(idx % this->sets.size()).dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Gamma1

Gamma1::Gamma1(const Ctx& ctx, size_t idx,
        const std::vector<Image>& sourceImages,
        const Image& additionalInput0,
        const Image& additionalInput1) {
    const size_t m = ctx.perf ? 1 : 2; // multiplier
    const Extent extent = sourceImages.at(0).getExtent();

    // create temporary & output images
    for (size_t i = 0; i < (2 * m); i++) {
        this->tempImages0.emplace_back(ctx.device, extent);
        this->tempImages1.emplace_back(ctx.device, extent);
    }
    this->image.emplace(ctx.device, extent, MTLPixelFormatRGBA16Float);

    // create descriptor sets
    const auto& shaders = ctx.mode().gamma;
    this->sets.reserve(4);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(sourceImages)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(1)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .storages(this->tempImages1)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(2)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages1)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(3)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .sampled(additionalInput0)
        .sampled(additionalInput1)
        .storage(*this->image)
        .sampler(ctx.bnbSampler)
        .sampler(ctx.eabSampler)
        .buffer(ctx.constantBuffers.at(idx))
        .build(shaders.at(4)));

    this->dispatchExtent = add_shift_extent(extent, 7, 3);
}

void Gamma1::render(Enc enc) const {
    for (const auto& set : this->sets)
        set.dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Delta0

Delta0::Delta0(const Ctx& ctx, size_t idx,
        const std::vector<std::vector<Image>>& sourceImages,
        const Image& additionalInput0,
        const Image& additionalInput1) {
    const size_t m = ctx.perf ? 1 : 2; // multiplier
    const Extent extent = sourceImages.at(0).at(0).getExtent();

    // create output images
    this->images0.reserve(3);
    for (size_t i = 0; i < 3; i++)
        this->images0.emplace_back(ctx.device, extent);
    this->images1.reserve(m);
    for (size_t i = 0; i < m; i++)
        this->images1.emplace_back(ctx.device, extent);

    // create descriptor sets
    const auto& shaders = ctx.mode().delta;
    this->sets0.reserve(sourceImages.size());
    for (size_t i = 0; i < sourceImages.size(); i++)
        this->sets0.emplace_back(ManagedShaderBuilder()
            .sampleds(sourceImages.at((i + (sourceImages.size() - 1)) % sourceImages.size()))
            .sampleds(sourceImages.at(i % sourceImages.size()))
            .sampled(additionalInput0)
            .storages(this->images0)
            .sampler(ctx.bnwSampler)
            .sampler(ctx.eabSampler)
            .buffer(ctx.constantBuffers.at(idx))
            .build(shaders.at(0)));
    this->sets1.reserve(sourceImages.size());
    for (size_t i = 0; i < sourceImages.size(); i++)
        this->sets1.emplace_back(ManagedShaderBuilder()
            .sampleds(sourceImages.at((i + (sourceImages.size() - 1)) % sourceImages.size()))
            .sampleds(sourceImages.at(i % sourceImages.size()))
            .sampled(additionalInput1)
            .sampled(additionalInput0)
            .storages(this->images1)
            .sampler(ctx.bnwSampler)
            .sampler(ctx.eabSampler)
            .buffer(ctx.constantBuffers.at(idx))
            .build(shaders.at(5)));

    this->dispatchExtent = add_shift_extent(extent, 7, 3);
}

void Delta0::render(Enc enc, size_t idx) const {
    this->sets0.at(idx % this->sets0.size()).dispatch(enc, this->dispatchExtent);
    this->sets1.at(idx % this->sets1.size()).dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Delta1

Delta1::Delta1(const Ctx& ctx, size_t idx,
        const std::vector<Image>& sourceImages0,
        const std::vector<Image>& sourceImages1,
        const Image& additionalInput0,
        const Image& additionalInput1,
        const Image& additionalInput2) {
    const size_t m = ctx.perf ? 1 : 2; // multiplier
    const Extent extent = sourceImages0.at(0).getExtent();

    // create temporary & output images
    for (size_t i = 0; i < (2 * m); i++) {
        this->tempImages0.emplace_back(ctx.device, extent);
        this->tempImages1.emplace_back(ctx.device, extent);
    }
    this->image0.emplace(ctx.device, extent, MTLPixelFormatRGBA16Float);
    this->image1.emplace(ctx.device, extent, MTLPixelFormatRGBA16Float);

    // create descriptor sets
    const auto& shaders = ctx.mode().delta;
    this->sets.reserve(4 + 4);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(sourceImages0)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(1)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .storages(this->tempImages1)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(2)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages1)
        .storages(this->tempImages0)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(3)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0)
        .sampled(additionalInput0)
        .sampled(additionalInput1)
        .storage(*this->image0)
        .sampler(ctx.bnbSampler)
        .sampler(ctx.eabSampler)
        .buffer(ctx.constantBuffers.at(idx))
        .build(shaders.at(4)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(sourceImages1)
        .storages(this->tempImages0, 0, m)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(6)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0, 0, m)
        .storages(this->tempImages1, 0, m)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(7)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages1, 0, m)
        .storages(this->tempImages0, 0, m)
        .sampler(ctx.bnbSampler)
        .build(shaders.at(8)));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampleds(this->tempImages0, 0, m)
        .sampled(additionalInput2)
        .storage(*this->image1)
        .sampler(ctx.bnbSampler)
        .sampler(ctx.eabSampler)
        .buffer(ctx.constantBuffers.at(idx))
        .build(shaders.at(9)));

    this->dispatchExtent = add_shift_extent(extent, 7, 3);
}

void Delta1::render(Enc enc) const {
    for (const auto& set : this->sets)
        set.dispatch(enc, this->dispatchExtent);
}

// ---------------------------------------------------------------- Generate

Generate::Generate(const Ctx& ctx, size_t idx,
        const std::pair<Image, Image>& sourceImages,
        const Image& inputImage1,
        const Image& inputImage2,
        const Image& inputImage3,
        const Image& outputImage) {
    const auto& shader = ctx.shaders->generate(ctx.hdr);
    this->sets.reserve(2);
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampled(sourceImages.second)
        .sampled(sourceImages.first)
        .sampled(inputImage1)
        .sampled(inputImage2)
        .sampled(inputImage3)
        .storage(outputImage)
        .sampler(ctx.bnbSampler)
        .sampler(ctx.eabSampler)
        .buffer(ctx.constantBuffers.at(idx))
        .build(shader));
    this->sets.emplace_back(ManagedShaderBuilder()
        .sampled(sourceImages.first)
        .sampled(sourceImages.second)
        .sampled(inputImage1)
        .sampled(inputImage2)
        .sampled(inputImage3)
        .storage(outputImage)
        .sampler(ctx.bnbSampler)
        .sampler(ctx.eabSampler)
        .buffer(ctx.constantBuffers.at(idx))
        .build(shader));

    this->dispatchExtent = add_shift_extent(ctx.sourceExtent, 15, 4);
}

void Generate::render(Enc enc, size_t idx) const {
    this->sets.at(idx % 2).dispatch(enc, this->dispatchExtent);
}
