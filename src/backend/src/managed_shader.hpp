/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/helpers/managed_shader.hpp */

#pragma once

#include "shader.hpp"
#include "types.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace lsfgmetal::backend {

    ///
    /// A shader together with its bound resources.
    ///
    /// Vulkan needs explicit barriers between dispatches; here all dispatches are recorded
    /// into one serial compute encoder with tracked resources, so Metal orders them.
    ///
    class ManagedShader {
        friend class ManagedShaderBuilder;
    public:
        /// dispatch the managed shader
        /// @param encoder compute encoder to record into
        /// @param extent number of threadgroups (same as vkCmdDispatch)
        void dispatch(id<MTLComputeCommandEncoder> encoder, Extent extent) const {
            if (extent.width == 0 || extent.height == 0)
                return;

            [encoder setComputePipelineState:this->shader->pipeline];
            for (size_t i = 0; i < this->textures.size(); i++)
                [encoder setTexture:this->textures[i] atIndex:i];
            for (size_t i = 0; i < this->samplers.size(); i++)
                [encoder setSamplerState:this->samplers[i] atIndex:i];
            for (size_t i = 0; i < this->buffers.size(); i++)
                [encoder setBuffer:this->buffers[i] offset:0 atIndex:i];
            [encoder dispatchThreadgroups:MTLSizeMake(extent.width, extent.height, 1)
                threadsPerThreadgroup:this->shader->threadsPerThreadgroup];
        }
    private:
        const Shader* shader{};
        std::vector<id<MTLTexture>> textures; // sampled images first, then storage images
        std::vector<id<MTLSamplerState>> samplers;
        std::vector<id<MTLBuffer>> buffers;
    };

    /// class for building managed shaders
    class ManagedShaderBuilder {
    public:
        [[nodiscard]] ManagedShaderBuilder& sampled(const Image& image) {
            this->sampledImages.push_back(image.handle());
            return *this;
        }
        [[nodiscard]] ManagedShaderBuilder& sampleds(const std::vector<Image>& images,
                size_t offset = 0, size_t count = 0) {
            if (count == 0 || offset + count > images.size())
                count = images.size() - offset;
            for (size_t i = 0; i < count; ++i)
                this->sampledImages.push_back(images.at(offset + i).handle());
            return *this;
        }
        [[nodiscard]] ManagedShaderBuilder& storage(const Image& image) {
            this->storageImages.push_back(image.handle());
            return *this;
        }
        [[nodiscard]] ManagedShaderBuilder& storages(const std::vector<Image>& images,
                size_t offset = 0, size_t count = 0) {
            if (count == 0 || offset + count > images.size())
                count = images.size() - offset;
            for (size_t i = 0; i < count; ++i)
                this->storageImages.push_back(images.at(offset + i).handle());
            return *this;
        }
        [[nodiscard]] ManagedShaderBuilder& sampler(id<MTLSamplerState> sampler) {
            this->imageSamplers.push_back(sampler);
            return *this;
        }
        [[nodiscard]] ManagedShaderBuilder& buffer(id<MTLBuffer> buffer) {
            this->constantBuffers.push_back(buffer);
            return *this;
        }

        /// build the managed shader
        /// @throws std::runtime_error if the resources do not match the shader layout
        [[nodiscard]] ManagedShader build(const Shader& shader) const {
            if (this->sampledImages.size() != shader.sampledImages
                    || this->storageImages.size() != shader.storageImages
                    || this->imageSamplers.size() != shader.samplers
                    || this->constantBuffers.size() != shader.buffers)
                throw std::runtime_error("resource layout mismatch for shader " + shader.name);

            ManagedShader ms;
            ms.shader = &shader;
            ms.textures = this->sampledImages;
            ms.textures.insert(ms.textures.end(),
                this->storageImages.begin(), this->storageImages.end());
            ms.samplers = this->imageSamplers;
            ms.buffers = this->constantBuffers;
            return ms;
        }
    private:
        std::vector<id<MTLTexture>> sampledImages;
        std::vector<id<MTLTexture>> storageImages;
        std::vector<id<MTLSamplerState>> imageSamplers;
        std::vector<id<MTLBuffer>> constantBuffers;
    };

}
