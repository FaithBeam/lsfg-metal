/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace lsfgmetal {

    ///
    /// Frame capture for diagnosing visual problems: `touch ~/lsfg-capture` while the game runs
    /// and, 5 seconds later (or after the number of seconds written into the file), the next
    /// real and generated frames are written as PNGs, numbered in the order they are shown, to
    /// ~/lsfg-capture-<time>/ (plus info.txt with the layer configuration).
    ///
    class FrameCapture {
    public:
        /// check for the trigger file (cheap, rate limited); starts a capture if present
        /// @return whether frames of this real frame should be captured
        bool poll(CAMetalLayer* layer, id<MTLTexture> real, const std::string& details);

        /// display-order sequence number for frame i of the current real frame
        /// (generated frames first, then the real frame at i == count)
        [[nodiscard]] uint32_t sequence(size_t i) const { return this->base + static_cast<uint32_t>(i); }
        /// call after all frames of the current real frame were encoded
        void finishFrame(size_t framesPerRealFrame);

        /// file name used for a captured frame
        [[nodiscard]] std::string fileName(uint32_t seq, const char* kind) const {
            char name[64];
            std::snprintf(name, sizeof name, "%03u-%s.png", seq, kind);
            return name;
        }
        [[nodiscard]] const std::string& directory() const { return this->dir; }

        /// encode a readback of texture into cb and write it as <seq>-<kind>.png when done
        void capture(id<MTLCommandBuffer> cb, id<MTLTexture> texture, uint32_t seq, const char* kind,
            double presentTime);
    private:
        double lastPoll{0};
        double startAt{0}; // pending capture: starts at this time (CACurrentMediaTime)
        int remaining{0};
        uint32_t base{0};
        std::string dir;
    };

}
