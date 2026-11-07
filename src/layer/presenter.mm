/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "presenter.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

using namespace lsfgmetal;

namespace {
    const char* COPY_SHADER = R"msl(
#include <metal_stdlib>
using namespace metal;

constant bool kEncodeSRGB [[function_constant(0)]];
constant bool kDecodeSRGB [[function_constant(1)]];
constant bool kDecodePQ [[function_constant(2)]];
constant bool kEncodePQ [[function_constant(3)]];

// SMPTE ST 2084 (PQ); linear values are scRGB-scaled (1.0 = 80 nits), which is what Lossless
// Scaling's HDR mode (cb.HdrSupport) expects
constant float PQ_M1 = 2610.0 / 16384.0;
constant float PQ_M2 = 2523.0 / 4096.0 * 128.0;
constant float PQ_C1 = 3424.0 / 4096.0;
constant float PQ_C2 = 2413.0 / 4096.0 * 32.0;
constant float PQ_C3 = 2392.0 / 4096.0 * 32.0;

static float3 pq_to_scrgb(float3 e) {
    const float3 p = pow(clamp(e, 0.0, 1.0), 1.0 / PQ_M2);
    const float3 l = pow(max(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1);
    return l * (10000.0 / 80.0);
}

static float3 scrgb_to_pq(float3 c) {
    const float3 l = pow(clamp(c * (80.0 / 10000.0), 0.0, 1.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * l) / (1.0 + PQ_C3 * l), PQ_M2);
}

struct VOut { float4 pos [[position]]; };

vertex VOut lsfg_fullscreen_vertex(uint vid [[vertex_id]]) {
    const float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

static float3 linear_to_srgb(float3 c) {
    c = clamp(c, 0.0, 1.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

static float3 srgb_to_linear(float3 c) {
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

fragment float4 lsfg_copy_fragment(VOut in [[stage_in]],
        texture2d<float, access::read> src [[texture(0)]]) {
    uint2 p = uint2(in.pos.xy);
    p = min(p, uint2(src.get_width() - 1, src.get_height() - 1));
    float4 c = src.read(p);
    if (kEncodeSRGB) c.rgb = linear_to_srgb(c.rgb);
    if (kDecodeSRGB) c.rgb = srgb_to_linear(c.rgb);
    if (kDecodePQ) c.rgb = pq_to_scrgb(c.rgb);
    if (kEncodePQ) c.rgb = scrgb_to_pq(c.rgb);
    return c;
}
)msl";
}

bool lsfgmetal::formatSupported(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatBGRA10_XR:
        case MTLPixelFormatBGRA10_XR_sRGB:
        case MTLPixelFormatBGR10_XR:
        case MTLPixelFormatBGR10_XR_sRGB:
            return true;
        default:
            return false;
    }
}

bool lsfgmetal::formatIsHDR(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatBGRA10_XR:
        case MTLPixelFormatBGRA10_XR_sRGB:
        case MTLPixelFormatBGR10_XR:
        case MTLPixelFormatBGR10_XR_sRGB:
            return true;
        default:
            return false;
    }
}

bool lsfgmetal::formatIsSRGB8(MTLPixelFormat format) {
    return format == MTLPixelFormatBGRA8Unorm_sRGB || format == MTLPixelFormatRGBA8Unorm_sRGB;
}

const char* lsfgmetal::formatName(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatBGRA8Unorm: return "BGRA8Unorm";
        case MTLPixelFormatBGRA8Unorm_sRGB: return "BGRA8Unorm_sRGB";
        case MTLPixelFormatRGBA8Unorm: return "RGBA8Unorm";
        case MTLPixelFormatRGBA8Unorm_sRGB: return "RGBA8Unorm_sRGB";
        case MTLPixelFormatRGB10A2Unorm: return "RGB10A2Unorm";
        case MTLPixelFormatBGR10A2Unorm: return "BGR10A2Unorm";
        case MTLPixelFormatRGBA16Float: return "RGBA16Float";
        case MTLPixelFormatBGRA10_XR: return "BGRA10_XR";
        case MTLPixelFormatBGRA10_XR_sRGB: return "BGRA10_XR_sRGB";
        case MTLPixelFormatBGR10_XR: return "BGR10_XR";
        case MTLPixelFormatBGR10_XR_sRGB: return "BGR10_XR_sRGB";
        default: return "unsupported";
    }
}

ColorEncoding lsfgmetal::colorEncodingFor(MTLPixelFormat format, CGColorSpaceRef colorspace) {
    if (!colorspace || !formatIsHDR(format))
        return ColorEncoding::SDR; // 8-bit storage can't hold linear/scRGB values anyway
    if (CGColorSpaceIsPQBased(colorspace))
        return ColorEncoding::PQ;
    if (format == MTLPixelFormatRGBA16Float) {
        const CFStringRef name = CGColorSpaceGetName(colorspace);
        if (name) {
            for (const CFStringRef linear : { kCGColorSpaceExtendedLinearSRGB, kCGColorSpaceLinearSRGB,
                    kCGColorSpaceExtendedLinearDisplayP3, kCGColorSpaceLinearDisplayP3,
                    kCGColorSpaceExtendedLinearITUR_2020, kCGColorSpaceLinearITUR_2020,
                    kCGColorSpaceGenericRGBLinear })
                if (CFStringCompare(name, linear, 0) == kCFCompareEqualTo)
                    return ColorEncoding::Linear;
        }
    }
    return ColorEncoding::SDR; // gamma-encoded, HLG, or unknown: use the values as they are
}

const char* lsfgmetal::colorEncodingName(ColorEncoding encoding) {
    switch (encoding) {
        case ColorEncoding::PQ: return "HDR10 (PQ)";
        case ColorEncoding::Linear: return "linear EDR";
        default: return "SDR";
    }
}

// ---------------------------------------------------------------- CopyPipelines

CopyPipelines::CopyPipelines(id<MTLDevice> device) : device(device) {
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion2_4;
    this->library = [device newLibraryWithSource:@(COPY_SHADER) options:options error:&error];
    if (!this->library)
        throw std::runtime_error("failed to compile copy shaders: "
            + std::string(error.localizedDescription.UTF8String ?: "unknown"));
}

id<MTLRenderPipelineState> CopyPipelines::pipeline(MTLPixelFormat format,
        bool encodeSRGB, bool decodeSRGB, bool decodePQ, bool encodePQ) {
    const std::lock_guard<std::mutex> lock(this->mutex);

    const auto key = std::make_tuple(format, encodeSRGB, decodeSRGB, decodePQ, encodePQ);
    if (auto it = this->cache.find(key); it != this->cache.end())
        return it->second;

    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&encodeSRGB type:MTLDataTypeBool atIndex:0];
    [constants setConstantValue:&decodeSRGB type:MTLDataTypeBool atIndex:1];
    [constants setConstantValue:&decodePQ type:MTLDataTypeBool atIndex:2];
    [constants setConstantValue:&encodePQ type:MTLDataTypeBool atIndex:3];

    NSError* error = nil;
    id<MTLFunction> fragment = [this->library newFunctionWithName:@"lsfg_copy_fragment"
        constantValues:constants error:&error];
    if (!fragment)
        throw std::runtime_error("failed to specialize copy shader: "
            + std::string(error.localizedDescription.UTF8String ?: "unknown"));

    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.label = @"lsfg-metal copy";
    desc.vertexFunction = [this->library newFunctionWithName:@"lsfg_fullscreen_vertex"];
    desc.fragmentFunction = fragment;
    desc.colorAttachments[0].pixelFormat = format;

    id<MTLRenderPipelineState> state = [this->device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!state)
        throw std::runtime_error("failed to create copy pipeline: "
            + std::string(error.localizedDescription.UTF8String ?: "unknown"));

    this->cache.emplace(key, state);
    return state;
}

void CopyPipelines::encodeCopy(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst,
        bool encodeSRGB, bool decodeSRGB, NSString* label, bool decodePQ, bool encodePQ) {
    id<MTLRenderPipelineState> state = this->pipeline(dst.pixelFormat, encodeSRGB, decodeSRGB,
        decodePQ, encodePQ);

    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    enc.label = label;
    [enc setRenderPipelineState:state];
    [enc setFragmentTexture:src atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}

// ---------------------------------------------------------------- FrameGenerator

FrameGenerator::FrameGenerator(backend::Instance& instance, CopyPipelines& copies,
        uint32_t width, uint32_t height, MTLPixelFormat drawableFormat,
        const ls::GameConf& profile, ColorEncoding encoding)
        : copies(copies), width(width), height(height),
          drawableFormat(drawableFormat), profile(profile),
          srgb(formatIsSRGB8(drawableFormat)),
          encoding(formatIsHDR(drawableFormat) ? encoding : ColorEncoding::SDR) {
    if (!formatSupported(drawableFormat))
        throw std::runtime_error(std::string("unsupported drawable format ")
            + std::to_string(static_cast<unsigned long>(drawableFormat)));

    this->context = instance.openContext(width, height,
        formatIsHDR(drawableFormat),
        1.0F / profile.flow_scale,
        profile.performance_mode,
        profile.multiplier - 1,
        this->encoding != ColorEncoding::SDR); // linear scRGB-style values: Lossless' HDR mode
}

bool FrameGenerator::matches(uint32_t width, uint32_t height, MTLPixelFormat drawableFormat,
        const ls::GameConf& profile, ColorEncoding encoding) const {
    const ColorEncoding effective = formatIsHDR(drawableFormat) ? encoding : ColorEncoding::SDR;
    return this->width == width && this->height == height
        && this->encoding == effective
        && this->drawableFormat == drawableFormat
        && this->profile.multiplier == profile.multiplier
        && this->profile.flow_scale == profile.flow_scale
        && this->profile.performance_mode == profile.performance_mode;
}

void FrameGenerator::encodeCopyIn(id<MTLCommandBuffer> cb, id<MTLTexture> drawableTexture) {
    this->copies.encodeCopy(cb, drawableTexture, this->context->nextSourceTexture(),
        this->srgb, false, @"lsfg-metal copy in", this->encoding == ColorEncoding::PQ, false);
}

void FrameGenerator::encodeGenerate(id<MTLCommandBuffer> cb) {
    this->context->encode(cb);
}

void FrameGenerator::encodeCopyOut(id<MTLCommandBuffer> cb, size_t i, id<MTLTexture> drawableTexture) {
    this->copies.encodeCopy(cb, this->context->destTexture(i), drawableTexture,
        false, this->srgb, @"lsfg-metal copy out", false, this->encoding == ColorEncoding::PQ);
}
