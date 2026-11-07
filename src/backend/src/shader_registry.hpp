/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/extraction/shader_registry.hpp */

#pragma once

#include "shader.hpp"

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lsfgmetal::backend {

    /// shader collection struct
    struct Shaders {
        std::array<Shader, 4> alpha;
        std::array<Shader, 5> beta;
        std::array<Shader, 5> gamma;
        std::array<Shader, 10> delta;
    };

    ///
    /// Shader registry. Unlike lsfg-vk, shaders are compiled lazily per mode
    /// (quality / performance), since runtime MSL compilation is comparatively slow.
    ///
    class ShaderRegistry {
    public:
        ShaderRegistry(id<MTLDevice> device, bool fp16,
            std::unordered_map<uint32_t, std::vector<uint8_t>> resources);

        /// compile common shaders and the shaders for one mode (blocking, thread-safe)
        /// @throws std::exception on failure
        void prepare(bool perf);

        /// whether a mode is ready
        [[nodiscard]] bool isPrepared(bool perf) const;

        /// accessors, only valid after prepare()
        [[nodiscard]] const Shader& mipmaps() const { return *this->mipmapsShader; }
        [[nodiscard]] const Shader& generate(bool hdr) const {
            return hdr ? *this->generateHdrShader : *this->generateShader;
        }
        [[nodiscard]] const Shaders& mode(bool perf) const {
            return perf ? *this->performance : *this->quality;
        }

        [[nodiscard]] bool isFP16() const { return this->fp16; }
    private:
        id<MTLDevice> device;
        bool fp16;
        std::unordered_map<uint32_t, std::vector<uint8_t>> resources;

        mutable std::mutex mutex;
        std::optional<Shader> mipmapsShader;
        std::optional<Shader> generateShader;
        std::optional<Shader> generateHdrShader;
        std::unique_ptr<Shaders> quality;
        std::unique_ptr<Shaders> performance;
    };

    /// patch the generate shader's storage image format (exposed for tests)
    void patchGenerateShader(std::vector<uint8_t>& data, bool hdr);

}
