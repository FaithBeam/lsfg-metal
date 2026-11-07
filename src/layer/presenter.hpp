/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lsfg-metal/backend.hpp"
#include "lsfg-metal-common/config.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace lsfgmetal {

    /// whether a drawable pixel format is supported for frame generation
    bool formatSupported(MTLPixelFormat format);
    /// whether a drawable pixel format needs the RGBA16Float (hdr) path
    bool formatIsHDR(MTLPixelFormat format);
    /// whether a format is an 8-bit sRGB format (values need re-encoding to stay gamma-encoded)
    bool formatIsSRGB8(MTLPixelFormat format);
    /// printable name of a pixel format
    const char* formatName(MTLPixelFormat format);

    /// how the values in a drawable are encoded (from the layer's color space)
    enum class ColorEncoding {
        SDR,    ///< gamma-encoded (sRGB-like), also the fallback - what lsfg-vk assumes
        Linear, ///< linear extended range (EDR, e.g. extendedLinearSRGB), 1.0 = SDR white
        PQ,     ///< HDR10: SMPTE ST 2084 (PQ) encoded, e.g. D3DMetal with an HDR10 swapchain
    };
    /// classify a layer's color space for a drawable format
    ColorEncoding colorEncodingFor(MTLPixelFormat format, CGColorSpaceRef colorspace);
    /// printable name of a color encoding
    const char* colorEncodingName(ColorEncoding encoding);

    ///
    /// Fullscreen copy pipelines (drawable <-> backend textures), one instance per device.
    ///
    /// Copies use a render pass instead of a blit because drawable and backend formats
    /// differ (BGRA vs RGBA, sRGB vs UNORM, 10-bit vs float); the equivalent of
    /// vkCmdBlitImage's format conversion in lsfg-vk.
    ///
    class CopyPipelines {
    public:
        explicit CopyPipelines(id<MTLDevice> device);

        /// encode a full-surface copy from src to dst
        /// @param encodeSRGB re-encode linear values read from src to sRGB
        /// @param decodeSRGB decode sRGB values to linear before writing to dst
        /// @param decodePQ convert PQ values read from src to linear scRGB (1.0 = 80 nits)
        /// @param encodePQ convert linear scRGB values read from src to PQ
        void encodeCopy(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst,
            bool encodeSRGB, bool decodeSRGB, NSString* label,
            bool decodePQ = false, bool encodePQ = false);
    private:
        id<MTLRenderPipelineState> pipeline(MTLPixelFormat format, bool encodeSRGB, bool decodeSRGB,
            bool decodePQ, bool encodePQ);

        id<MTLDevice> device;
        id<MTLLibrary> library;
        std::mutex mutex;
        std::map<std::tuple<MTLPixelFormat, bool, bool, bool, bool>, id<MTLRenderPipelineState>> cache;
    };

    ///
    /// Per-surface frame generation state: a backend context plus the copy logic.
    ///
    class FrameGenerator {
    public:
        /// @throws std::exception on failure
        FrameGenerator(backend::Instance& instance, CopyPipelines& copies,
            uint32_t width, uint32_t height, MTLPixelFormat drawableFormat,
            const ls::GameConf& profile, ColorEncoding encoding = ColorEncoding::SDR);

        /// whether this generator can be reused for the given parameters
        [[nodiscard]] bool matches(uint32_t width, uint32_t height, MTLPixelFormat drawableFormat,
            const ls::GameConf& profile, ColorEncoding encoding = ColorEncoding::SDR) const;
        [[nodiscard]] ColorEncoding colorEncoding() const { return this->encoding; }

        /// copy a real frame into the backend
        void encodeCopyIn(id<MTLCommandBuffer> cb, id<MTLTexture> drawableTexture);
        /// record the generation of all intermediate frames for the last copied frame
        void encodeGenerate(id<MTLCommandBuffer> cb);
        /// copy generated frame i into a drawable texture
        void encodeCopyOut(id<MTLCommandBuffer> cb, size_t i, id<MTLTexture> drawableTexture);

        /// number of generated frames per real frame
        [[nodiscard]] size_t count() const { return this->context->count(); }
        /// number of real frames processed
        [[nodiscard]] uint64_t frameIndex() const { return this->context->frameIndex(); }
        [[nodiscard]] backend::Context& backendContext() { return *this->context; }
    private:
        CopyPipelines& copies;
        std::unique_ptr<backend::Context> context;
        uint32_t width, height;
        MTLPixelFormat drawableFormat;
        ls::GameConf profile;
        bool srgb;
        ColorEncoding encoding;
    };

}
