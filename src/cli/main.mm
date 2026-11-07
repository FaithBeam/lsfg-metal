/* SPDX-License-Identifier: GPL-3.0-or-later */
/* lsfg-metal command line tool (counterpart of lsfg-vk-cli) */

#include "lsfg-metal/backend.hpp"
#include "lsfg-metal-common/config.hpp"
#include "lsfg-metal-common/detection.hpp"
#include "presenter.hpp"

// internal backend headers for MSL dumping
#include "dll_reader.hpp"
#include "shader.hpp"
#include "shader_registry.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace lsfgmetal;
namespace fs = std::filesystem;

namespace {
    struct Args {
        std::string command;
        std::optional<std::string> dll;
        std::optional<std::string> out;
        uint32_t width{1920};
        uint32_t height{1080};
        size_t multiplier{2};
        float flow{1.0F};
        bool perf{false};
        bool hdr{false};
        bool fp32{false};
        std::string format{"bgra8"};
        int pq{0}; // motiontest: 1 = HDR10 scene, handled as PQ; 2 = HDR10 scene handled as SDR (lsfg-vk)
        size_t frames{300};
    };

    void usage() {
        std::cout <<
R"(lsfg-metal-cli - Lossless Scaling frame generation on Metal

usage: lsfg-metal-cli <command> [options]

commands:
  find-dll                 print the Lossless.dll that would be used
  validate                 translate and compile all shaders (quality + performance)
  dump-msl --out DIR       write the translated Metal shaders to DIR
  benchmark                run frame generation offscreen and report GPU time
  selftest                 numeric end-to-end test (use with tests/build_fake_dll.py)
  motiontest [--out DIR]   check that generated frames interpolate motion (real Lossless.dll)

options:
  --dll PATH               path to Lossless.dll (default: search like the hook does)
  --fp32                   use the fp32 shader variants
  -W, --width N            frame width (default 1920)
  -H, --height N           frame height (default 1080)
  -m, --multiplier N       frame multiplier (default 2)
  -f, --flow F             flow scale 0.25..1.0 (default 1.0)
  -p, --perf               performance mode
  --hdr                    use the RGBA16Float path
  -n, --frames N           benchmark frames (default 300)
  --format F               motiontest drawable format: bgra8, rgb10a2, rgba16f (default bgra8)
  --pq / --pq-as-sdr       motiontest: HDR10 (PQ) scene in rgb10a2, with / without PQ handling
)";
    }

    Args parse(int argc, char** argv) {
        Args a;
        if (argc < 2) return a;
        a.command = argv[1];
        for (int i = 2; i < argc; i++) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
                return argv[++i];
            };
            if (s == "--dll") a.dll = next();
            else if (s == "--out") a.out = next();
            else if (s == "-W" || s == "--width") a.width = static_cast<uint32_t>(std::stoul(next()));
            else if (s == "-H" || s == "--height") a.height = static_cast<uint32_t>(std::stoul(next()));
            else if (s == "-m" || s == "--multiplier") a.multiplier = std::stoul(next());
            else if (s == "-f" || s == "--flow") a.flow = std::stof(next());
            else if (s == "-p" || s == "--perf") a.perf = true;
            else if (s == "--hdr") a.hdr = true;
            else if (s == "--fp32") a.fp32 = true;
            else if (s == "--format") a.format = next();
            else if (s == "--pq") a.pq = 1;
            else if (s == "--pq-as-sdr") a.pq = 2;
            else if (s == "-n" || s == "--frames") a.frames = std::stoul(next());
            else throw std::runtime_error("unknown option: " + s);
        }
        return a;
    }

    fs::path resolveDll(const Args& a) {
        if (a.dll) return *a.dll;
        std::optional<std::string> configured;
        try {
            const ls::ConfigFile config(ls::findConfigurationFile());
            configured = config.global().dll;
        } catch (...) {} // missing configuration is fine here
        return ls::findShaderDll(configured);
    }

    id<MTLDevice> device() {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw std::runtime_error("no Metal device");
        return dev;
    }

    int cmdFindDll(const Args& a) {
        std::cout << resolveDll(a).string() << "\n";
        return 0;
    }

    int cmdValidate(const Args& a) {
        const auto path = resolveDll(a);
        id<MTLDevice> dev = device();
        std::cout << "device: " << dev.name.UTF8String << "\n";
        std::cout << "dll:    " << path.string() << "\n";

        backend::Instance instance(dev, path, !a.fp32);
        for (bool perf : { false, true }) {
            const auto start = std::chrono::steady_clock::now();
            instance.prepare(perf);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << (perf ? "performance" : "quality    ") << " shaders OK ("
                << ms << " ms, " << (instance.isFP16() ? "fp16" : "fp32") << ")\n";
        }

        // make sure contexts can be built and run once for both modes
        for (bool perf : { false, true }) {
            auto ctx = instance.openContext(a.width, a.height, a.hdr, 1.0F / a.flow, perf, a.multiplier - 1);
            id<MTLCommandQueue> queue = [dev newCommandQueue];
            for (int i = 0; i < 2; i++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                ctx->encode(cb);
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("command buffer failed: "
                        + std::string(cb.error.localizedDescription.UTF8String ?: "?"));
            }
            std::cout << (perf ? "performance" : "quality    ") << " context OK\n";
        }
        std::cout << "all good\n";
        return 0;
    }

    int cmdDumpMSL(const Args& a) {
        if (!a.out) throw std::runtime_error("--out DIR is required");
        const auto resources = backend::extractResourcesFromDLL(resolveDll(a));
        fs::create_directories(*a.out);

        // (id, sampled, storage, buffers, samplers) for quality, then performance
        struct S { uint32_t id; size_t p1, p2, p3, p4; };
        const std::vector<S> quality = {
            {255,1,7,1,1},{256,5,1,1,2},{267,1,2,0,1},{268,2,2,0,1},{269,2,4,0,1},{270,4,4,0,1},
            {275,12,2,0,1},{276,2,2,0,1},{277,2,2,0,1},{278,2,2,0,1},{279,2,6,1,1},
            {257,9,3,1,2},{259,3,4,0,1},{260,4,4,0,1},{261,4,4,0,1},{262,6,1,1,2},
            {263,3,4,0,1},{264,4,4,0,1},{265,4,4,0,1},{266,6,1,1,2},{258,10,2,1,2},
            {271,2,2,0,1},{272,2,2,0,1},{273,2,2,0,1},{274,3,1,1,2}};
        const std::vector<S> perf = {
            {267,1,1,0,1},{268,1,1,0,1},{269,1,2,0,1},{270,2,2,0,1},
            {275,6,2,0,1},{276,2,2,0,1},{277,2,2,0,1},{278,2,2,0,1},{279,2,6,1,1},
            {257,5,3,1,2},{259,3,2,0,1},{260,2,2,0,1},{261,2,2,0,1},{262,4,1,1,2},
            {263,3,2,0,1},{264,2,2,0,1},{265,2,2,0,1},{266,4,1,1,2},{258,6,1,1,2},
            {271,1,1,0,1},{272,1,1,0,1},{273,1,1,0,1},{274,2,1,1,2}};

        size_t written = 0;
        for (bool isPerf : { false, true }) {
            for (const auto& s : isPerf ? perf : quality) {
                const uint32_t rid = 49 + s.id + (isPerf ? 23 : 0) + (a.fp32 ? 49 : 0);
                auto it = resources.find(rid);
                if (it == resources.end())
                    throw std::runtime_error("resource " + std::to_string(rid) + " missing");
                std::vector<uint8_t> spirv = it->second;
                if (s.id == 256) backend::patchGenerateShader(spirv, false);
                MTLSize tg{};
                std::string entry;
                const std::string msl = backend::translateShader(spirv, s.p1, s.p2, s.p3, s.p4, tg, entry);
                const std::string name = std::string(isPerf ? "perf_" : "quality_") + std::to_string(s.id);
                std::ofstream(fs::path(*a.out) / (name + ".metal")) << msl;
                std::ofstream(fs::path(*a.out) / (name + ".spv"), std::ios::binary)
                    .write(reinterpret_cast<const char*>(spirv.data()), static_cast<std::streamsize>(spirv.size()));
                std::cout << name << ": threadgroup " << tg.width << "x" << tg.height << "x" << tg.depth << "\n";
                written++;
            }
        }
        std::cout << "wrote " << written << " shaders to " << *a.out << "\n";
        return 0;
    }

    int cmdBenchmark(const Args& a) {
        const auto path = resolveDll(a);
        id<MTLDevice> dev = device();
        backend::Instance instance(dev, path, !a.fp32);
        auto ctx = instance.openContext(a.width, a.height, a.hdr, 1.0F / a.flow, a.perf, a.multiplier - 1);
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        std::cout << "device:     " << dev.name.UTF8String << "\n"
                  << "resolution: " << a.width << "x" << a.height << (a.hdr ? " (hdr)" : "") << "\n"
                  << "multiplier: " << a.multiplier << ", flow " << a.flow
                  << (a.perf ? ", performance mode" : ", quality mode")
                  << (instance.isFP16() ? ", fp16" : ", fp32") << "\n";

        // warmup
        for (int i = 0; i < 10; i++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            ctx->encode(cb);
            [cb commit];
            [cb waitUntilCompleted];
        }

        std::vector<double> gpu;
        gpu.reserve(a.frames);
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < a.frames; i++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            ctx->encode(cb);
            [cb commit];
            [cb waitUntilCompleted];
            gpu.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::sort(gpu.begin(), gpu.end());
        double sum = 0;
        for (double v : gpu) sum += v;
        const double avg = sum / static_cast<double>(gpu.size());
        std::printf("gpu time per real frame: avg %.3f ms, p50 %.3f ms, p99 %.3f ms\n",
            avg, gpu[gpu.size() / 2], gpu[std::min(gpu.size() - 1, gpu.size() * 99 / 100)]);
        std::printf("=> max sustainable output: %.0f fps (%.0f real fps x %zu)\n",
            1000.0 / avg * static_cast<double>(a.multiplier), 1000.0 / avg, a.multiplier);
        std::printf("wall clock: %.2f s for %zu frames\n", wall, a.frames);
        return 0;
    }

    // ------------------------------------------------------------------------------ selftest

    id<MTLTexture> makeTexture(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLPixelFormat fmt) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        d.storageMode = dev.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
        return [dev newTextureWithDescriptor:d];
    }

    size_t bytesPerPixel(MTLPixelFormat fmt) {
        return fmt == MTLPixelFormatRGBA16Float ? 8 : 4;
    }

    /// fill a texture with a solid color given in the texture's raw channel space (0..1)
    void fill(id<MTLTexture> t, const float rgba[4]) {
        const size_t w = t.width, h = t.height, bpp = bytesPerPixel(t.pixelFormat);
        std::vector<uint8_t> data(w * h * bpp);
        for (size_t i = 0; i < w * h; i++) {
            if (t.pixelFormat == MTLPixelFormatRGBA16Float) {
                auto* px = reinterpret_cast<__fp16*>(&data[i * bpp]);
                for (int c = 0; c < 4; c++) px[c] = static_cast<__fp16>(rgba[c]);
            } else {
                const bool bgra = t.pixelFormat == MTLPixelFormatBGRA8Unorm
                    || t.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
                const int order[4] = { bgra ? 2 : 0, 1, bgra ? 0 : 2, 3 };
                for (int c = 0; c < 4; c++)
                    data[i * bpp + static_cast<size_t>(c)] =
                        static_cast<uint8_t>(std::lround(rgba[order[c]] * 255.0F));
            }
        }
        [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data.data() bytesPerRow:w * bpp];
    }

    /// read a texture back into RGBA floats (raw channel space)
    std::vector<float> readback(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLTexture> t) {
        if (t.storageMode == MTLStorageModeManaged) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit synchronizeResource:t];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        (void)dev;
        const size_t w = t.width, h = t.height, bpp = bytesPerPixel(t.pixelFormat);
        std::vector<uint8_t> data(w * h * bpp);
        [t getBytes:data.data() bytesPerRow:w * bpp fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
        std::vector<float> out(w * h * 4);
        for (size_t i = 0; i < w * h; i++) {
            if (t.pixelFormat == MTLPixelFormatRGB10A2Unorm) {
                uint32_t v;
                std::memcpy(&v, &data[i * bpp], 4);
                out[i * 4 + 0] = static_cast<float>(v & 0x3ffU) / 1023.0F;
                out[i * 4 + 1] = static_cast<float>((v >> 10) & 0x3ffU) / 1023.0F;
                out[i * 4 + 2] = static_cast<float>((v >> 20) & 0x3ffU) / 1023.0F;
                out[i * 4 + 3] = static_cast<float>(v >> 30) / 3.0F;
            } else if (t.pixelFormat == MTLPixelFormatRGBA16Float) {
                const auto* px = reinterpret_cast<const __fp16*>(&data[i * bpp]);
                for (int c = 0; c < 4; c++) out[i * 4 + static_cast<size_t>(c)] = static_cast<float>(px[c]);
            } else {
                const bool bgra = t.pixelFormat == MTLPixelFormatBGRA8Unorm
                    || t.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
                const int order[4] = { bgra ? 2 : 0, 1, bgra ? 0 : 2, 3 };
                for (int c = 0; c < 4; c++)
                    out[i * 4 + static_cast<size_t>(c)] =
                        static_cast<float>(data[i * bpp + static_cast<size_t>(order[c])]) / 255.0F;
            }
        }
        return out;
    }

    int cmdSelftest(const Args& a) {
        const auto path = resolveDll(a);
        id<MTLDevice> dev = device();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        backend::Instance instance(dev, path, !a.fp32);
        CopyPipelines copies(dev);

        // odd sizes on purpose: exercises partial workgroups and rounding in every chain
        const uint32_t W = a.width == 1920 ? 1001 : a.width;
        const uint32_t H = a.height == 1080 ? 567 : a.height;

        const float A[4] = { 0.2F, 0.4F, 0.9F, 1.0F };
        const float B[4] = { 0.8F, 0.1F, 0.3F, 1.0F };

        int failures = 0;
        for (MTLPixelFormat fmt : { MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB,
                                    MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float }) {
            for (bool perf : { false, true }) {
                for (size_t mult : { size_t{2}, size_t{4} }) {
                    ls::GameConf profile;
                    profile.multiplier = mult;
                    profile.performance_mode = perf;
                    profile.flow_scale = 0.75F;

                    FrameGenerator gen(instance, copies, W, H, fmt, profile);
                    id<MTLTexture> src = makeTexture(dev, W, H, fmt);
                    std::vector<id<MTLTexture>> outs;
                    for (size_t i = 0; i < gen.count(); i++)
                        outs.push_back(makeTexture(dev, W, H, fmt));

                    // frame 0 (A) primes the history, frame 1 (B) generates A->B
                    for (int frame = 0; frame < 2; frame++) {
                        fill(src, frame == 0 ? A : B);
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        gen.encodeCopyIn(cb, src);
                        gen.encodeGenerate(cb);
                        if (frame == 1)
                            for (size_t i = 0; i < gen.count(); i++)
                                gen.encodeCopyOut(cb, i, outs[i]);
                        [cb commit];
                        [cb waitUntilCompleted];
                        if (cb.status != MTLCommandBufferStatusCompleted)
                            throw std::runtime_error("command buffer failed: "
                                + std::string(cb.error.localizedDescription.UTF8String ?: "?"));
                    }

                    // expected: the synthetic generate shader outputs mix(prev, curr, t)
                    const float tol = fmt == MTLPixelFormatRGBA16Float ? 0.004F : 2.5F / 255.0F;
                    for (size_t i = 0; i < gen.count(); i++) {
                        const float t = static_cast<float>(i + 1) / static_cast<float>(mult);
                        const auto px = readback(dev, queue, outs[i]);
                        float maxErr = 0;
                        size_t worst = 0;
                        for (size_t p = 0; p < static_cast<size_t>(W) * H; p++)
                            for (int c = 0; c < 4; c++) {
                                const float expect = A[c] + (B[c] - A[c]) * t;
                                const float err = std::fabs(px[p * 4 + static_cast<size_t>(c)] - expect);
                                if (!(err <= maxErr)) { maxErr = err; worst = p; } // NaN-aware
                            }
                        const bool ok = maxErr <= tol;
                        failures += ok ? 0 : 1;
                        std::printf("%-16s %-11s x%zu frame %zu (t=%.2f): max error %.4f %s",
                            formatName(fmt), perf ? "performance" : "quality", mult, i, t, maxErr,
                            ok ? "OK\n" : "FAIL");
                        if (!ok)
                            std::printf(" (worst pixel %zu,%zu = %.3f %.3f %.3f %.3f)\n",
                                worst % W, worst / W, px[worst * 4], px[worst * 4 + 1],
                                px[worst * 4 + 2], px[worst * 4 + 3]);
                    }
                }
            }
        }

        std::printf(failures ? "\n%d checks FAILED\n" : "\nall checks passed\n", failures);
        return failures ? 1 : 0;
    }

    // ---------------------------------------------------------------------------- motiontest

    uint32_t hash2(uint32_t x, uint32_t y, uint32_t seed) {
        uint32_t h = x * 0x8da6b343U ^ y * 0xd8163841U ^ seed * 0xcb1ab31fU;
        h ^= h >> 13; h *= 0x5bd1e995U; h ^= h >> 15;
        return h;
    }
    float hashf(int x, int y, uint32_t seed) {
        return static_cast<float>(hash2(static_cast<uint32_t>(x), static_cast<uint32_t>(y), seed) & 0xffffU) / 65535.0F;
    }
    /// smooth value noise with the given cell size
    float valueNoise(int x, int y, int cell, uint32_t seed) {
        const int cx = x / cell, cy = y / cell;
        const float fx = static_cast<float>(x % cell) / static_cast<float>(cell);
        const float fy = static_cast<float>(y % cell) / static_cast<float>(cell);
        const float sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
        const float a = hashf(cx, cy, seed), b = hashf(cx + 1, cy, seed);
        const float c = hashf(cx, cy + 1, seed), d = hashf(cx + 1, cy + 1, seed);
        return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
    }

    /// the test scene: static textured background + a textured square at (ox, oy)
    void renderScene(std::vector<float>& rgba, uint32_t W, uint32_t H, int ox, int oy, int S) {
        rgba.resize(static_cast<size_t>(W) * H * 4);
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++) {
                float* px = &rgba[(static_cast<size_t>(y) * W + x) * 4];
                const int ix = static_cast<int>(x), iy = static_cast<int>(y);
                if (ix >= ox && ix < ox + S && iy >= oy && iy < oy + S) {
                    const int u = ix - ox, v = iy - oy;
                    const float n = valueNoise(u + 64, v + 64, 12, 7);
                    const float m = valueNoise(u + 64, v + 64, 5, 9);
                    px[0] = 0.65F + 0.3F * n;
                    px[1] = 0.05F + 0.25F * m;
                    px[2] = 0.08F + 0.1F * n;
                } else {
                    const float v = 0.25F + 0.25F * valueNoise(ix, iy, 32, 1) + 0.15F * valueNoise(ix, iy, 6, 2);
                    px[0] = v; px[1] = v; px[2] = v * 1.15F;
                }
                px[3] = 1.0F;
            }
    }

    void upload(id<MTLTexture> t, const std::vector<float>& rgba) {
        const size_t w = t.width, h = t.height, bpp = bytesPerPixel(t.pixelFormat);
        std::vector<uint8_t> data(w * h * bpp);
        const auto q = [](float v, float m) { return static_cast<uint32_t>(std::lround(std::clamp(v, 0.0F, 1.0F) * m)); };
        for (size_t i = 0; i < w * h; i++) {
            const float* c = &rgba[i * 4];
            if (t.pixelFormat == MTLPixelFormatRGB10A2Unorm) {
                const uint32_t v = q(c[0], 1023) | (q(c[1], 1023) << 10) | (q(c[2], 1023) << 20) | (q(c[3], 3) << 30);
                std::memcpy(&data[i * bpp], &v, 4);
            } else if (t.pixelFormat == MTLPixelFormatRGBA16Float) {
                auto* px = reinterpret_cast<__fp16*>(&data[i * bpp]);
                for (int k = 0; k < 4; k++) px[k] = static_cast<__fp16>(c[k]);
            } else { // BGRA8
                const int order[4] = { 2, 1, 0, 3 };
                for (int k = 0; k < 4; k++) data[i * bpp + static_cast<size_t>(k)] = static_cast<uint8_t>(q(c[order[k]], 255));
            }
        }
        [t replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:data.data() bytesPerRow:w * bpp];
    }

    void writePNG(const fs::path& path, const std::vector<float>& rgba, uint32_t W, uint32_t H) {
        std::vector<uint8_t> px(static_cast<size_t>(W) * H * 4);
        for (size_t i = 0; i < px.size(); i++)
            px[i] = static_cast<uint8_t>(std::lround(std::clamp(rgba[i], 0.0F, 1.0F) * 255.0F));
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(px.data(), W, H, 8, W * 4, cs,
            static_cast<uint32_t>(kCGImageAlphaPremultipliedLast) | kCGBitmapByteOrder32Big);
        CGImageRef img = CGBitmapContextCreateImage(ctx);
        NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
        CGImageDestinationRef dst = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
        if (dst) {
            CGImageDestinationAddImage(dst, img, nullptr);
            CGImageDestinationFinalize(dst);
            CFRelease(dst);
        }
        CGImageRelease(img);
        CGContextRelease(ctx);
        CGColorSpaceRelease(cs);
    }

    // HDR10 test scene: SDR scene values (sRGB) mapped to 203 nits reference white, PQ encoded
    float pqEncode(float srgb) {
        const float lin = srgb <= 0.04045F ? srgb / 12.92F : std::pow((srgb + 0.055F) / 1.055F, 2.4F);
        const float l = std::pow(std::clamp(lin * 203.0F / 10000.0F, 0.0F, 1.0F), 2610.0F / 16384.0F);
        return std::pow((0.8359375F + 18.8515625F * l) / (1.0F + 18.6875F * l), 78.84375F);
    }
    float pqDecode(float e) { // back to the sRGB-encoded scene value
        const float p = std::pow(std::clamp(e, 0.0F, 1.0F), 1.0F / 78.84375F);
        const float l = std::pow(std::max(p - 0.8359375F, 0.0F) / (18.8515625F - 18.6875F * p), 16384.0F / 2610.0F);
        const float lin = std::clamp(l * 10000.0F / 203.0F, 0.0F, 1.0F);
        return lin <= 0.0031308F ? lin * 12.92F : 1.055F * std::pow(lin, 1.0F / 2.4F) - 0.055F;
    }
    void toPQ(std::vector<float>& rgba) {
        for (size_t i = 0; i < rgba.size(); i++)
            if (i % 4 != 3) rgba[i] = pqEncode(rgba[i]);
    }

    int cmdMotiontest(const Args& a) {
        const auto path = resolveDll(a);
        id<MTLDevice> dev = device();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        backend::Instance instance(dev, path, !a.fp32);
        CopyPipelines copies(dev);

        const uint32_t W = a.width == 1920 ? 1280 : a.width;
        const uint32_t H = a.height == 1080 ? 720 : a.height;
        const int S = 160;          // square size
        const int dx = 24, dy = 0;  // motion per real frame (divisible by 2, 3, 4, 6, 8)
        const int frames = 10;
        const int oy = static_cast<int>(H) / 2 - S / 2;
        const auto ox = [&](double k) { return 100.0 + k * dx; };

        if (a.out) fs::create_directories(*a.out);
        const MTLPixelFormat fmt = (a.format == "rgb10a2" || a.pq) ? MTLPixelFormatRGB10A2Unorm
            : a.format == "rgba16f" ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
        const ColorEncoding encoding = a.pq == 1 ? ColorEncoding::PQ : ColorEncoding::SDR;
        std::printf("format %s (%s path), %s scene, processed as %s\n", formatName(fmt),
            formatIsHDR(fmt) ? "hdr" : "8-bit", a.pq ? "HDR10" : "SDR", colorEncodingName(encoding));

        int failures = 0;
        for (bool perf : { false, true }) {
            ls::GameConf profile;
            profile.multiplier = a.multiplier;
            profile.performance_mode = perf;
            profile.flow_scale = a.flow;
            FrameGenerator gen(instance, copies, W, H, fmt, profile, encoding);

            id<MTLTexture> src = makeTexture(dev, W, H, fmt);
            std::vector<id<MTLTexture>> outs;
            for (size_t i = 0; i < gen.count(); i++)
                outs.push_back(makeTexture(dev, W, H, fmt));

            double sumGen = 0, sumBlend = 0, sumRepeat = 0;
            double sumCentroidErr = 0;
            size_t samples = 0;
            std::vector<float> prev, curr, ideal;
            for (int k = 0; k < frames; k++) {
                prev = curr;
                renderScene(curr, W, H, static_cast<int>(ox(k)), oy + k * dy, S);
                if (a.pq) toPQ(curr);
                upload(src, curr);
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                gen.encodeCopyIn(cb, src);
                gen.encodeGenerate(cb);
                for (size_t i = 0; i < gen.count(); i++)
                    gen.encodeCopyOut(cb, i, outs[i]);
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("command buffer failed");
                if (k < 3) continue; // let the temporal history fill up

                // region of interest: around the object's path between the two frames
                const int x0 = std::max(0, static_cast<int>(ox(k - 1)) - 32);
                const int x1 = std::min(static_cast<int>(W), static_cast<int>(ox(k)) + S + 32);
                const int y0 = std::max(0, oy - 32), y1 = std::min(static_cast<int>(H), oy + S + 32);

                for (size_t i = 0; i < gen.count(); i++) {
                    const double t = static_cast<double>(i + 1) / static_cast<double>(a.multiplier);
                    const int ix = static_cast<int>(std::lround(ox(k - 1 + t)));
                    renderScene(ideal, W, H, ix, oy, S);
                    if (a.pq) toPQ(ideal);
                    const auto got = readback(dev, queue, outs[i]);

                    double eGen = 0, eBlend = 0, eRepeat = 0, cx = 0, cn = 0;
                    for (int y = y0; y < y1; y++)
                        for (int x = x0; x < x1; x++) {
                            const size_t p = (static_cast<size_t>(y) * W + static_cast<size_t>(x)) * 4;
                            for (int c = 0; c < 3; c++) {
                                const double ref = ideal[p + c];
                                const double blend = prev[p + c] + (curr[p + c] - prev[p + c]) * t;
                                eGen += std::fabs(got[p + c] - ref);
                                eBlend += std::fabs(blend - ref);
                                eRepeat += std::fabs(prev[p + c] - ref);
                            }
                            // "redness" mask for the object's centroid
                            const double red = a.pq ? pqDecode(got[p]) - pqDecode(got[p + 1]) : got[p] - got[p + 1];
                            if (red > 0.35) { cx += x; cn += 1; }
                        }
                    const double n = static_cast<double>((x1 - x0) * (y1 - y0) * 3) / 255.0;
                    eGen /= n; eBlend /= n; eRepeat /= n;
                    const double centroidErr = cn > 0 ? std::fabs(cx / cn - (ix + (S - 1) / 2.0)) : 1e9;
                    sumGen += eGen; sumBlend += eBlend; sumRepeat += eRepeat;
                    sumCentroidErr += centroidErr;
                    samples++;

                    if (a.out && k == frames - 1) {
                        const std::string base = std::string(perf ? "perf" : "quality") + "_t"
                            + std::to_string(i + 1) + "of" + std::to_string(a.multiplier);
                        writePNG(fs::path(*a.out) / (base + "_generated.png"), got, W, H);
                        writePNG(fs::path(*a.out) / (base + "_ideal.png"), ideal, W, H);
                        if (i == 0) {
                            writePNG(fs::path(*a.out) / (std::string(perf ? "perf" : "quality") + "_prev.png"), prev, W, H);
                            writePNG(fs::path(*a.out) / (std::string(perf ? "perf" : "quality") + "_curr.png"), curr, W, H);
                        }
                    }
                }
            }
            const double s = static_cast<double>(samples);
            const double g = sumGen / s, b = sumBlend / s, r = sumRepeat / s, ce = sumCentroidErr / s;
            // pass if clearly better than both a cross-fade and frame repetition, and the object is
            // within a few pixels of where it should be
            const bool ok = g < 0.6 * b && g < 0.6 * r && ce < 4.0;
            failures += ok ? 0 : 1;
            std::printf("%-11s x%zu flow %.2f: mean abs error (0-255) generated %.2f | cross-fade %.2f | "
                "repeat %.2f | object position error %.1f px  %s\n",
                perf ? "performance" : "quality", a.multiplier, a.flow, g, b, r, ce, ok ? "OK" : "FAIL");
        }
        if (a.out) std::printf("images written to %s\n", a.out->c_str());
        return failures ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const Args a = parse(argc, argv);
            if (a.command == "find-dll") return cmdFindDll(a);
            if (a.command == "validate") return cmdValidate(a);
            if (a.command == "dump-msl") return cmdDumpMSL(a);
            if (a.command == "benchmark") return cmdBenchmark(a);
            if (a.command == "selftest") return cmdSelftest(a);
            if (a.command == "motiontest") return cmdMotiontest(a);
            usage();
            return a.command.empty() || a.command == "help" || a.command == "--help" ? 0 : 1;
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
}
