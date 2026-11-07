/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/include/lsfg-vk-backend/lsfgvk.hpp */

#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lsfgmetal::backend {

    class InstanceImpl;
    class ContextImpl;

    ///
    /// Primitive exception class that delivers a detailed error message
    ///
    class error : public std::runtime_error {
    public:
        explicit error(const std::string& msg, const std::exception& inner)
            : std::runtime_error(msg + "\n- " + inner.what()) {}
        explicit error(const std::string& msg)
            : std::runtime_error(msg) {}
    };

    ///
    /// A frame generation context bound to a fixed resolution and image count.
    ///
    /// Unlike lsfg-vk, which shares images with the layer through file descriptors and
    /// synchronizes through a timeline semaphore, everything here lives in one process on
    /// one MTLDevice: the caller writes into sourceTexture(), calls encode() on a command
    /// buffer and reads destTexture(i) in later work on the same command queue (Metal's
    /// hazard tracking orders the accesses).
    ///
    class Context {
    public:
        ///
        /// Texture the next real frame must be written to before encode() is called.
        /// Alternates between two textures (frame index % 2).
        ///
        [[nodiscard]] id<MTLTexture> nextSourceTexture() const;

        ///
        /// Output texture for the i-th generated frame (0-based, count = multiplier - 1).
        /// Valid after the work recorded by encode() has run.
        ///
        [[nodiscard]] id<MTLTexture> destTexture(size_t i) const;

        /// number of generated frames per real frame
        [[nodiscard]] size_t count() const;

        /// pixel format of source and destination textures
        [[nodiscard]] MTLPixelFormat format() const;

        /// width and height of the frames
        [[nodiscard]] uint32_t width() const;
        [[nodiscard]] uint32_t height() const;

        /// number of real frames encoded so far
        [[nodiscard]] uint64_t frameIndex() const;

        ///
        /// Record the full frame generation chain (pre-pass and all generated frames).
        ///
        /// The first call only primes the history; outputs are meaningful from the second
        /// call on (same as lsfg-vk).
        ///
        /// @throws backend::error on failure
        ///
        void encode(id<MTLCommandBuffer> commandBuffer);

        explicit Context(std::unique_ptr<ContextImpl> impl);
        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;
        ~Context();
    private:
        std::unique_ptr<ContextImpl> m_impl;
    };

    ///
    /// Main entry point of the library
    ///
    class Instance {
    public:
        ///
        /// Create an lsfg-metal instance.
        ///
        /// Parses Lossless.dll; shaders are translated and compiled lazily per mode
        /// (see prepare()).
        ///
        /// @param device Metal device to run on (must be the device of the presenting layer).
        /// @param shaderDllPath Path to the Lossless.dll file to load shaders from.
        /// @param allowLowPrecision Whether to load low-precision (FP16) shaders.
        ///
        /// @throws backend::error on failure
        ///
        Instance(id<MTLDevice> device,
            const std::filesystem::path& shaderDllPath,
            bool allowLowPrecision);

        ///
        /// Create an instance from already extracted DLL resources (used for testing).
        ///
        Instance(id<MTLDevice> device,
            std::unordered_map<uint32_t, std::vector<uint8_t>> resources,
            bool allowLowPrecision);

        ///
        /// Translate and compile the shaders for a mode. Blocking and thread-safe;
        /// openContext() calls it implicitly. Call it from a background thread to avoid
        /// stalling the presenting thread.
        ///
        /// @throws backend::error on failure
        ///
        void prepare(bool perf);

        /// whether prepare(perf) has completed successfully
        [[nodiscard]] bool isPrepared(bool perf) const;

        ///
        /// Open a frame generation context.
        ///
        /// @param width Width of the frames.
        /// @param height Height of the frames.
        /// @param hdr Use RGBA16Float instead of RGBA8Unorm images.
        /// @param flow Inverted flow scale (1 / flow_scale).
        /// @param perf Whether to enable performance mode.
        /// @param count Number of frames to generate per real frame (multiplier - 1).
        /// @param hdrSupport Frames hold linear scRGB values (1.0 = 80 nits), like Lossless
        ///     Scaling's HDR mode on Windows; the shaders then derive their luma accordingly.
        ///     Not set by lsfg-vk. Requires hdr.
        ///
        /// @throws backend::error on failure
        ///
        std::unique_ptr<Context> openContext(uint32_t width, uint32_t height,
            bool hdr, float flow, bool perf, size_t count, bool hdrSupport = false);

        [[nodiscard]] id<MTLDevice> device() const;

        /// whether fp16 shaders are in use
        [[nodiscard]] bool isFP16() const;

        Instance(const Instance&) = delete;
        Instance& operator=(const Instance&) = delete;
        ~Instance();
    private:
        std::unique_ptr<InstanceImpl> m_impl;
    };

}
