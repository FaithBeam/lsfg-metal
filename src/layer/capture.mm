/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "capture.hpp"
#include "presenter.hpp"
#include "lsfg-metal-common/log.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fstream>
#include <memory>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace lsfgmetal;

namespace {
    constexpr int kRealFrames = 20; // real frames per capture (plus their generated frames)
    constexpr double kDefaultDelay = 5.0; // seconds between the trigger and the capture

    std::string home() {
        const char* h = std::getenv("HOME");
        return (h && *h) ? h : "/tmp";
    }

    size_t bytesPerPixel(MTLPixelFormat f) {
        return f == MTLPixelFormatRGBA16Float ? 8 : 4;
    }

    bool readable(MTLPixelFormat f) {
        switch (f) {
            case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
            case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
            case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatBGR10A2Unorm:
            case MTLPixelFormatRGBA16Float:
                return true;
            default:
                return false;
        }
    }

    /// convert raw texels to RGBA8 (values written as stored: PQ stays PQ, linear stays linear)
    std::vector<uint8_t> toRGBA8(const uint8_t* src, size_t w, size_t h, MTLPixelFormat f) {
        std::vector<uint8_t> out(w * h * 4);
        const auto q = [](float v) { return static_cast<uint8_t>(std::lround(std::min(std::max(v, 0.0F), 1.0F) * 255.0F)); };
        for (size_t i = 0; i < w * h; i++) {
            uint8_t* o = &out[i * 4];
            switch (f) {
                case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
                    o[0] = src[i * 4 + 2]; o[1] = src[i * 4 + 1]; o[2] = src[i * 4]; o[3] = 255;
                    break;
                case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
                    o[0] = src[i * 4]; o[1] = src[i * 4 + 1]; o[2] = src[i * 4 + 2]; o[3] = 255;
                    break;
                case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatBGR10A2Unorm: {
                    uint32_t v;
                    std::memcpy(&v, &src[i * 4], 4);
                    const float a = static_cast<float>(v & 0x3ffU) / 1023.0F;
                    const float b = static_cast<float>((v >> 10) & 0x3ffU) / 1023.0F;
                    const float c = static_cast<float>((v >> 20) & 0x3ffU) / 1023.0F;
                    const bool rgb = f == MTLPixelFormatRGB10A2Unorm;
                    o[0] = q(rgb ? a : c); o[1] = q(b); o[2] = q(rgb ? c : a); o[3] = 255;
                    break;
                }
                default: { // RGBA16Float
                    const auto* px = reinterpret_cast<const __fp16*>(&src[i * 8]);
                    o[0] = q(static_cast<float>(px[0])); o[1] = q(static_cast<float>(px[1]));
                    o[2] = q(static_cast<float>(px[2])); o[3] = 255;
                    break;
                }
            }
        }
        return out;
    }

    void writePNG(const std::string& path, std::vector<uint8_t>& px, size_t w, size_t h) {
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(px.data(), w, h, 8, w * 4, cs,
            static_cast<uint32_t>(kCGImageAlphaNoneSkipLast) | kCGBitmapByteOrder32Big);
        CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : nullptr;
        if (img) {
            NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
            CGImageDestinationRef dst = CGImageDestinationCreateWithURL(
                (__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
            if (dst) {
                CGImageDestinationAddImage(dst, img, nullptr);
                CGImageDestinationFinalize(dst);
                CFRelease(dst);
            }
            CGImageRelease(img);
        }
        if (ctx) CGContextRelease(ctx);
        CGColorSpaceRelease(cs);
    }

    std::string colorspaceName(CGColorSpaceRef cs) {
        if (!cs) return "(none)";
        CFStringRef name = CGColorSpaceGetName(cs);
        if (!name) return "(unnamed)";
        return [(__bridge NSString*)name UTF8String] ?: "?";
    }
}

bool FrameCapture::poll(CAMetalLayer* layer, id<MTLTexture> real, const std::string& details) {
    if (this->remaining > 0)
        return true;
    const double now = CACurrentMediaTime();
    if (now - this->lastPoll < 0.25)
        return false;
    this->lastPoll = now;

    if (this->startAt == 0) {
        const std::string trigger = home() + "/lsfg-capture";
        struct stat sb{};
        if (stat(trigger.c_str(), &sb) != 0)
            return false;
        double delay = kDefaultDelay; // optional: number of seconds written into the file
        if (std::ifstream f(trigger); f.is_open()) {
            double v = -1;
            if (f >> v && v >= 0 && v <= 600)
                delay = v;
        }
        unlink(trigger.c_str());
        this->startAt = now + delay;
        ls::log::info("capture: starting in " + std::to_string(static_cast<int>(std::lround(delay))) + " seconds");
    }
    if (now < this->startAt)
        return false;
    this->startAt = 0;

    char stamp[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&t));
    this->dir = home() + "/lsfg-capture-" + stamp;
    if (mkdir(this->dir.c_str(), 0755) != 0) {
        ls::log::warn("capture: unable to create " + this->dir);
        return false;
    }
    if (!readable(real.pixelFormat)) {
        ls::log::warn(std::string("capture: unsupported format ") + formatName(real.pixelFormat));
        return false;
    }

    std::ofstream info(this->dir + "/info.txt");
    info << "frames are numbered in display order; for each real frame its generated frames are\n"
            "shown first, then the real frame. Pixel values are written as stored (HDR10 stays PQ).\n\n"
         << details << "\n"
         << "drawable: " << real.width << "x" << real.height << " " << formatName(real.pixelFormat) << "\n"
         << "layer.colorspace: " << colorspaceName(layer.colorspace) << "\n"
         << "layer.wantsExtendedDynamicRangeContent: " << (layer.wantsExtendedDynamicRangeContent ? "yes" : "no") << "\n"
         << "layer.edrMetadata: " << (layer.EDRMetadata ? "set" : "nil") << "\n"
         << "layer.presentsWithTransaction: " << (layer.presentsWithTransaction ? "yes" : "no") << "\n"
         << "layer.displaySyncEnabled: " << (layer.displaySyncEnabled ? "yes" : "no") << "\n"
         << "layer.maximumDrawableCount: " << layer.maximumDrawableCount << "\n"
         << "layer.drawableSize: " << layer.drawableSize.width << "x" << layer.drawableSize.height << "\n"
         << "layer.contentsScale: " << layer.contentsScale << "\n";
    ls::log::info("capture: writing " + std::to_string(kRealFrames) + " real frames to " + this->dir);
    this->remaining = kRealFrames;
    this->base = 0;
    return true;
}

void FrameCapture::finishFrame(size_t framesPerRealFrame) {
    if (this->remaining <= 0)
        return;
    this->base += static_cast<uint32_t>(framesPerRealFrame);
    if (--this->remaining == 0)
        ls::log::info("capture: done (" + this->dir + ")");
}

void FrameCapture::capture(id<MTLCommandBuffer> cb, id<MTLTexture> texture, uint32_t seq,
        const char* kind, double presentTime) {
    if (!texture || !readable(texture.pixelFormat))
        return;
    const size_t w = texture.width, h = texture.height, bpr = w * bytesPerPixel(texture.pixelFormat);
    id<MTLBuffer> buffer = [cb.device newBufferWithLength:bpr * h options:MTLResourceStorageModeShared];
    if (!buffer)
        return;
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    blit.label = @"lsfg-metal capture";
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:MTLSizeMake(w, h, 1) toBuffer:buffer destinationOffset:0
        destinationBytesPerRow:bpr destinationBytesPerImage:bpr * h];
    [blit endEncoding];

    char name[64];
    std::snprintf(name, sizeof name, "/%03u-%s.png", seq, kind);
    const std::string path = this->dir + name;
    const MTLPixelFormat format = texture.pixelFormat;
    const std::string timing = this->dir + "/timing.txt";
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            auto px = toRGBA8(static_cast<const uint8_t*>(buffer.contents), w, h, format);
            writePNG(path, px, w, h);
        });
    }];
    char line[128];
    std::snprintf(line, sizeof line, "%s submitted at %.4f s\n", name + 1, presentTime);
    std::ofstream(timing, std::ios::app) << line;
}
