/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal equivalents of lsfg-vk-common's vk::Image / vk::Sampler / vk::Buffer and
   lsfg-vk-backend/src/helpers/utils.hpp */

#pragma once

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace lsfgmetal::backend {

    /// 2D extent (replacement for VkExtent2D)
    struct Extent {
        uint32_t width{};
        uint32_t height{};
    };

    /// round down an extent
    inline Extent shift_extent(Extent extent, uint32_t i) {
        return { extent.width >> i, extent.height >> i };
    }

    /// round up an extent
    inline Extent add_shift_extent(Extent extent, uint32_t a, uint32_t i) {
        return { (extent.width + a) >> i, (extent.height + a) >> i };
    }

    /// constant buffer used in shaders (std140 layout, 48 bytes)
    struct ConstantBuffer {
        std::array<uint32_t, 2> inputOffset;
        uint32_t firstIter;
        uint32_t firstIterS;
        uint32_t advancedColorKind;
        uint32_t hdrSupport;
        float resolutionInvScale;
        float timestamp;
        float uiThreshold;
        std::array<uint32_t, 3> pad;
    };
    static_assert(sizeof(ConstantBuffer) == 48);

    /// get a prefilled constant buffer
    inline ConstantBuffer getDefaultConstantBuffer(size_t index, size_t total, float invFlow,
            bool hdrSupport = false) {
        return ConstantBuffer {
            .inputOffset = {0, 0},
            .firstIter = 0,
            .firstIterS = 0,
            .advancedColorKind = 0,
            .hdrSupport = hdrSupport ? 1U : 0U,
            .resolutionInvScale = invFlow,
            .timestamp = static_cast<float>(index + 1) / static_cast<float>(total + 1),
            .uiThreshold = 0.5F,
            .pad = {0, 0, 0}
        };
    }

    /// a 2D GPU image (replacement for vk::Image)
    class Image {
    public:
        Image(id<MTLDevice> device, Extent extent,
                MTLPixelFormat format = MTLPixelFormatRGBA8Unorm,
                MTLTextureUsage extraUsage = 0) : extent(extent) {
            MTLTextureDescriptor* desc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:format
                width:std::max<uint32_t>(extent.width, 1)
                height:std::max<uint32_t>(extent.height, 1)
                mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | extraUsage;
            desc.storageMode = MTLStorageModePrivate;
            this->texture = [device newTextureWithDescriptor:desc];
            if (!this->texture)
                throw std::runtime_error("newTextureWithDescriptor failed");
        }

        [[nodiscard]] id<MTLTexture> handle() const { return this->texture; }
        [[nodiscard]] Extent getExtent() const { return this->extent; }
    private:
        id<MTLTexture> texture;
        Extent extent;
    };

    /// create a sampler matching lsfg-vk's vk::Sampler
    inline id<MTLSamplerState> createSampler(id<MTLDevice> device,
            MTLSamplerAddressMode mode, bool white) {
        MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
        desc.minFilter = MTLSamplerMinMagFilterLinear;
        desc.magFilter = MTLSamplerMinMagFilterLinear;
        desc.mipFilter = MTLSamplerMipFilterLinear;
        desc.sAddressMode = mode;
        desc.tAddressMode = mode;
        desc.rAddressMode = mode;
        desc.borderColor = white ? MTLSamplerBorderColorOpaqueWhite
                                 : MTLSamplerBorderColorTransparentBlack;
        desc.normalizedCoordinates = YES;
        id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:desc];
        if (!sampler)
            throw std::runtime_error("newSamplerStateWithDescriptor failed");
        return sampler;
    }

    /// create a constant buffer
    inline id<MTLBuffer> createBuffer(id<MTLDevice> device, const ConstantBuffer& data) {
        id<MTLBuffer> buffer = [device newBufferWithBytes:&data length:sizeof(data)
            options:MTLResourceStorageModeShared];
        if (!buffer)
            throw std::runtime_error("newBufferWithBytes failed");
        return buffer;
    }

}
