/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Metal port of lsfg-vk-backend/src/shaderchains/ (headers) and helpers/utils.hpp (Ctx) */

#pragma once

#include "managed_shader.hpp"
#include "shader_registry.hpp"
#include "types.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace lsfgmetal::backend {

    /// exposed context data
    struct Ctx {
        id<MTLDevice> device;
        const ShaderRegistry* shaders;

        id<MTLBuffer> constantBuffer;
        std::vector<id<MTLBuffer>> constantBuffers;
        id<MTLSamplerState> bnbSampler; //!< border, no compare, black
        id<MTLSamplerState> bnwSampler; //!< border, no compare, white
        id<MTLSamplerState> eabSampler; //!< edge, (always compare - unused), black

        Extent sourceExtent;
        Extent flowExtent;

        bool hdr;
        float flow;
        bool perf;
        size_t count;

        [[nodiscard]] const Shaders& mode() const { return this->shaders->mode(this->perf); }
    };

    using Enc = id<MTLComputeCommandEncoder>;

    class Mipmaps {
    public:
        Mipmaps(const Ctx& ctx, const std::pair<Image, Image>& sourceImages);
        void render(Enc enc, size_t idx) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<Image> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Alpha0 {
    public:
        Alpha0(const Ctx& ctx, const Image& sourceImage);
        void render(Enc enc) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<Image> tempImages0;
        std::vector<Image> tempImages1;
        std::vector<Image> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent0{};
        Extent dispatchExtent1{};
    };

    class Alpha1 {
    public:
        Alpha1(const Ctx& ctx, size_t temporal, const std::vector<Image>& sourceImages);
        void render(Enc enc, size_t idx) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<std::vector<Image>> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Beta0 {
    public:
        Beta0(const Ctx& ctx, const std::vector<std::vector<Image>>& sourceImages);
        void render(Enc enc, size_t idx) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<Image> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Beta1 {
    public:
        Beta1(const Ctx& ctx, const std::vector<Image>& sourceImages);
        void render(Enc enc) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<Image> tempImages0;
        std::vector<Image> tempImages1;
        std::vector<Image> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent0{};
        Extent dispatchExtent1{};
    };

    class Gamma0 {
    public:
        Gamma0(const Ctx& ctx, size_t idx,
            const std::vector<std::vector<Image>>& sourceImages,
            const Image& additionalInput);
        void render(Enc enc, size_t idx) const;
        [[nodiscard]] const auto& getImages() const { return this->images; }
    private:
        std::vector<Image> images;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Gamma1 {
    public:
        Gamma1(const Ctx& ctx, size_t idx,
            const std::vector<Image>& sourceImages,
            const Image& additionalInput0,
            const Image& additionalInput1);
        void render(Enc enc) const;
        [[nodiscard]] const auto& getImage() const { return *this->image; }
    private:
        std::vector<Image> tempImages0;
        std::vector<Image> tempImages1;
        std::optional<Image> image;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Delta0 {
    public:
        Delta0(const Ctx& ctx, size_t idx,
            const std::vector<std::vector<Image>>& sourceImages,
            const Image& additionalInput0,
            const Image& additionalInput1);
        void render(Enc enc, size_t idx) const;
        [[nodiscard]] const auto& getImages0() const { return this->images0; }
        [[nodiscard]] const auto& getImages1() const { return this->images1; }
    private:
        std::vector<Image> images0;
        std::vector<Image> images1;
        std::vector<ManagedShader> sets0;
        std::vector<ManagedShader> sets1;
        Extent dispatchExtent{};
    };

    class Delta1 {
    public:
        Delta1(const Ctx& ctx, size_t idx,
            const std::vector<Image>& sourceImages0,
            const std::vector<Image>& sourceImages1,
            const Image& additionalInput0,
            const Image& additionalInput1,
            const Image& additionalInput2);
        void render(Enc enc) const;
        [[nodiscard]] const auto& getImage0() const { return *this->image0; }
        [[nodiscard]] const auto& getImage1() const { return *this->image1; }
    private:
        std::vector<Image> tempImages0;
        std::vector<Image> tempImages1;
        std::optional<Image> image0;
        std::optional<Image> image1;
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

    class Generate {
    public:
        Generate(const Ctx& ctx, size_t idx,
            const std::pair<Image, Image>& sourceImages,
            const Image& inputImage1,
            const Image& inputImage2,
            const Image& inputImage3,
            const Image& outputImage);
        void render(Enc enc, size_t idx) const;
    private:
        std::vector<ManagedShader> sets;
        Extent dispatchExtent{};
    };

}
