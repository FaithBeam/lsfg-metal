/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * lsfg-metal hook library.
 *
 * Injected with DYLD_INSERT_LIBRARIES. Takes the place of lsfg-vk's Vulkan layer by
 * intercepting Metal/CoreAnimation presentation instead of vkQueuePresentKHR:
 *
 *  Classic path (MTLCommandBuffer, used by MoltenVK and D3DMetal):
 *    -[MTLCommandBuffer presentDrawable:*]  -> recorded on the command buffer, not executed
 *    -[MTLCommandBuffer commit]             -> signal an MTLEvent at the end of the game's
 *                                              work, commit, then generate + present frames
 *  Metal 4 path (D3DMetal on macOS 26+):
 *    -[MTL4CommandQueue signalDrawable:]    -> additionally signal an MTLEvent
 *    -[CAMetalDrawable present*]            -> generate + present frames
 *
 * Generated frames are produced on a private command queue that waits on the event, copied
 * into freshly acquired drawables of the same layer and presented in order, followed by the
 * real frame. Presents are issued with -[CAMetalDrawable present] after the producing command
 * buffer has been scheduled, which is exactly what -presentDrawable: does internally, but
 * keeps the presentation order deterministic.
 */

#include "presenter.hpp"
#include "capture.hpp"
#include "lsfg-metal/backend.hpp"
#include "lsfg-metal-common/config.hpp"
#include "lsfg-metal-common/detection.hpp"
#include "lsfg-metal-common/log.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace lsfgmetal;

// ============================================================================================
// associated object payloads
// ============================================================================================

/// a present recorded on a command buffer (classic path)
@interface LSFGPendingPresent : NSObject {
@public
    id<CAMetalDrawable> drawable;
}
@end
@implementation LSFGPendingPresent
@end

/// an event signal recorded for a drawable (Metal 4 path)
@interface LSFGDrawableSignal : NSObject {
@public
    id<MTLEvent> event;
    uint64_t value;
}
@end
@implementation LSFGDrawableSignal
@end

namespace {
    struct LayerState;
}

/// per-layer state, attached to the CAMetalLayer
@interface LSFGLayerState : NSObject {
@public
    std::unique_ptr<LayerState> state;
}
@end

namespace {
    char kPendingKey;
    char kSignalKey;
    char kCommitKey;
    char kLayerStateKey;

    /// thread-specific "lsfg-metal itself is calling" flag.
    ///
    /// Implemented with pthread TSD rather than thread_local: in Wine processes this image is
    /// never initialized by dyld (see bootstrap at the end of the file), so nothing here may
    /// depend on static initializers or TLV setup.
    pthread_key_t internalKey() {
        static pthread_key_t key;
        static pthread_once_t once = PTHREAD_ONCE_INIT;
        pthread_once(&once, [] { pthread_key_create(&key, nullptr); });
        return key;
    }

    bool isInternal() {
        return pthread_getspecific(internalKey()) != nullptr;
    }

    struct InternalScope {
        void* previous;
        InternalScope() : previous(pthread_getspecific(internalKey())) {
            pthread_setspecific(internalKey(), reinterpret_cast<void*>(1));
        }
        ~InternalScope() { pthread_setspecific(internalKey(), previous); }
        InternalScope(const InternalScope&) = delete;
        InternalScope& operator=(const InternalScope&) = delete;
    };

    /// bare-present path: the LSFGDrawableSignal of the command buffer whose scheduled handler
    /// is running on this thread (unretained; the command buffer keeps it alive)
    pthread_key_t scheduledKey() {
        static pthread_key_t key;
        static pthread_once_t once = PTHREAD_ONCE_INIT;
        pthread_once(&once, [] { pthread_key_create(&key, nullptr); });
        return key;
    }

    /// bare-present path: the LSFGDrawableSignal of the last command buffer committed on this
    /// thread (retained)
    pthread_key_t committedKey() {
        static pthread_key_t key;
        static pthread_once_t once = PTHREAD_ONCE_INIT;
        pthread_once(&once, [] { pthread_key_create(&key, [](void* p) { if (p) CFRelease(p); }); });
        return key;
    }

    void setCommitted(LSFGDrawableSignal* signal) {
        void* previous = pthread_getspecific(committedKey());
        pthread_setspecific(committedKey(), signal ? (void*)CFBridgingRetain(signal) : nullptr);
        if (previous)
            CFRelease(previous);
    }

    /// an MTLEvent per game command queue, signalled at the end of each of its command buffers
    /// (one event per queue keeps the signalled values increasing)
    struct QueueSync {
        id<MTLEvent> event;
        std::atomic<uint64_t> value{0};
    };

    /// log the first invocation of a hook (debug builds of bug reports)
#define LSFG_TRACE_ONCE(what) do { \
        static std::atomic<bool> traced{false}; \
        if (!traced.exchange(true)) ls::log::debug(std::string("first call: ") + (what)); \
    } while (0)

    // ========================================================================================
    // process-wide configuration
    // ========================================================================================

    struct Global {
        std::once_flag initOnce;
        std::atomic<bool> enabled{false};

        std::mutex configMutex;
        std::optional<ls::WatchedConfig> config;
        ls::GameConf profile;
        uint64_t profileGeneration{0};
        std::optional<std::string> dllOverride;
        bool allowFP16{true};
        std::chrono::steady_clock::time_point lastConfigCheck{};
    };

    Global& global() {
        static Global* g = new Global(); // intentionally leaked (used during teardown)
        return *g;
    }

    void initializeProcess() {
        auto& g = global();
        std::call_once(g.initOnce, [&g] {
            try {
                g.config.emplace();
                const auto id = ls::identify();
                const auto match = ls::findProfile(g.config->get(), id);

                ls::log::debug("executable: " + id.executable);
                if (id.wine_executable)
                    ls::log::debug("wine executable: " + *id.wine_executable);
                ls::log::debug("process name: " + id.process_name);

                if (!match) {
                    ls::log::debug("no profile matches this process, staying inactive "
                        "(config: " + g.config->filePath().string() + ")");
                    return;
                }

                g.profile = match->second;
                g.dllOverride = g.config->get().global().dll;
                g.allowFP16 = g.config->get().global().allow_fp16;
                g.lastConfigCheck = std::chrono::steady_clock::now();
                ls::log::info("using profile \"" + g.profile.name + "\": multiplier "
                    + std::to_string(g.profile.multiplier) + ", flow scale "
                    + std::to_string(g.profile.flow_scale)
                    + (g.profile.performance_mode ? ", performance mode" : ""));
                g.enabled = true;
            } catch (const std::exception& e) {
                ls::log::error(std::string("failed to load configuration: ") + e.what());
            }
        });
    }

    /// current profile (with hot reload of the configuration file)
    std::pair<ls::GameConf, uint64_t> currentProfile() {
        auto& g = global();
        const std::lock_guard<std::mutex> lock(g.configMutex);

        const auto now = std::chrono::steady_clock::now();
        if (g.config && now - g.lastConfigCheck > std::chrono::seconds(1)) {
            g.lastConfigCheck = now;
            try {
                if (g.config->update()) {
                    const auto match = ls::findProfile(g.config->get(), ls::identify());
                    if (match && !(match->second == g.profile)) {
                        g.profile = match->second;
                        g.profileGeneration++;
                        ls::log::info("configuration reloaded, profile \"" + g.profile.name + "\"");
                    } else if (!match) {
                        ls::log::warn("configuration reloaded, but no profile matches anymore; "
                            "keeping the previous profile");
                    }
                }
            } catch (const std::exception& e) {
                ls::log::error(std::string("failed to reload configuration: ") + e.what());
            }
        }
        return { g.profile, g.profileGeneration };
    }

    // ========================================================================================
    // per-device state
    // ========================================================================================

    struct DeviceState {
        id<MTLDevice> device;
        id<MTLCommandQueue> queue;
        id<MTLEvent> event;
        std::atomic<uint64_t> eventValue{0};
        std::unique_ptr<CopyPipelines> copies;

        std::mutex mutex;
        std::unique_ptr<backend::Instance> instance;
        bool loading{false};
        bool failed{false};
        std::set<bool> preparing; // modes currently compiling

        std::mutex queueMutex;
        std::unordered_map<void*, QueueSync*> queues; // game queue -> sync (never freed)
    };

    std::mutex& devicesMutex() {
        static auto* m = new std::mutex();
        return *m;
    }
    std::unordered_map<void*, DeviceState*>& devices() {
        static auto* d = new std::unordered_map<void*, DeviceState*>();
        return *d;
    }

    void installDeviceHooks(id<MTLDevice> device);

    DeviceState* deviceState(id<MTLDevice> device) {
        const std::lock_guard<std::mutex> lock(devicesMutex());
        auto& map = devices();
        void* key = (__bridge void*)device;
        if (auto it = map.find(key); it != map.end())
            return it->second;

        auto* ds = new DeviceState(); // never freed: devices live for the whole process
        ds->device = device;
        map.emplace(key, ds);
        return ds;
    }

    /// like deviceState(), but nullptr for devices lsfg-metal hasn't seen a layer of
    DeviceState* findDeviceState(id<MTLDevice> device) {
        const std::lock_guard<std::mutex> lock(devicesMutex());
        auto& map = devices();
        auto it = map.find((__bridge void*)device);
        return it != map.end() ? it->second : nullptr;
    }

    /// kick off (or check) asynchronous backend initialization; returns true when usable
    bool deviceReady(DeviceState* ds, bool perf) {
        const std::lock_guard<std::mutex> lock(ds->mutex);
        if (ds->failed)
            return false;
        if (ds->instance && ds->instance->isPrepared(perf))
            return true;
        if (ds->loading || ds->preparing.contains(perf))
            return false;

        if (!ds->instance)
            ds->loading = true;
        ds->preparing.insert(perf);

        id<MTLDevice> device = ds->device;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            try {
                std::unique_ptr<backend::Instance> created;
                bool needsInstance = false;
                {
                    const std::lock_guard<std::mutex> l(ds->mutex);
                    needsInstance = !ds->instance;
                }
                if (needsInstance) {
                    std::optional<std::string> dll;
                    bool fp16 = true;
                    {
                        auto& g = global();
                        const std::lock_guard<std::mutex> l(g.configMutex);
                        dll = g.dllOverride;
                        fp16 = g.allowFP16;
                    }
                    const auto path = ls::findShaderDll(dll);
                    ls::log::info("loading shaders from " + path.string()
                        + " on " + std::string(device.name.UTF8String ?: "?"));
                    created = std::make_unique<backend::Instance>(device, path, fp16);

                    auto copies = std::make_unique<CopyPipelines>(device);
                    id<MTLCommandQueue> queue = [device newCommandQueue];
                    queue.label = @"lsfg-metal";
                    id<MTLEvent> event = [device newEvent];

                    created->prepare(perf);

                    const std::lock_guard<std::mutex> l(ds->mutex);
                    ds->copies = std::move(copies);
                    ds->queue = queue;
                    ds->event = event;
                    ds->instance = std::move(created);
                    ds->loading = false;
                } else {
                    backend::Instance* inst = nullptr;
                    {
                        const std::lock_guard<std::mutex> l(ds->mutex);
                        inst = ds->instance.get();
                    }
                    inst->prepare(perf);
                }
                ls::log::info("frame generation ready");
            } catch (const std::exception& e) {
                ls::log::error(std::string("failed to initialize frame generation: ") + e.what());
                const std::lock_guard<std::mutex> l(ds->mutex);
                ds->failed = true;
                ds->loading = false;
            }
            const std::lock_guard<std::mutex> l(ds->mutex);
            ds->preparing.erase(perf);
        });
        return false;
    }

    // ========================================================================================
    // per-layer state
    // ========================================================================================

    struct LayerState {
        std::mutex mutex;
        // frames are processed on a per-layer worker so the game's present call returns at once;
        // at most one frame is in flight, the next present waits for it (bounded latency)
        dispatch_queue_t worker = dispatch_queue_create("lsfg-metal.present",
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        dispatch_semaphore_t inflight = dispatch_semaphore_create(1);
        std::unique_ptr<FrameGenerator> generator;
        FrameCapture capture;
        uint64_t profileGeneration{~0ULL};
        bool failed{false};
        bool announced{false};

        // pacing
        CFTimeInterval lastPresent{0};
        double frameInterval{0}; // smoothed real frame interval in seconds

        // statistics
        CFTimeInterval statsStart{0};
        uint64_t statsReal{0};
        uint64_t statsGenerated{0};
        uint64_t statsMissed{0};
        double statsIntervalSum{0}, statsIntervalMax{0};
        uint64_t statsIntervals{0}, statsHitches{0};
        std::atomic<uint64_t> waitSumUs{0}, waitMaxUs{0}, waitCount{0};
        void recordWait(double seconds) {
            const auto us = static_cast<uint64_t>(std::max(seconds, 0.0) * 1e6);
            waitSumUs += us;
            waitCount++;
            uint64_t prev = waitMaxUs.load();
            while (us > prev && !waitMaxUs.compare_exchange_weak(prev, us)) {}
        }
        struct Gpu { // written from Metal completion handlers
            std::atomic<uint64_t> sumUs{0}, maxUs{0}, count{0};
        };
        std::shared_ptr<Gpu> gpu = std::make_shared<Gpu>();
        struct Display { // written from presented handlers: what actually reached the screen
            std::mutex m;
            double last{0}, ema{0}, sumGap{0}, maxGap{0};
            uint64_t gaps{0}, late{0}, dropped{0};
            uint64_t submitted{0};    // presents in submission order (worker thread)
            uint64_t lastShownIndex{0};
            double lastShownTime{0};
            uint64_t reordered{0};    // shown before a frame that was presented earlier
            std::string captureLog;   // when set: append "<index> <name> presented <time>" lines
        };
        std::shared_ptr<Display> display = std::make_shared<Display>();
    };

    LayerState& layerState(CAMetalLayer* layer) {
        LSFGLayerState* holder = objc_getAssociatedObject(layer, &kLayerStateKey);
        if (!holder) {
            @synchronized (layer) {
                holder = objc_getAssociatedObject(layer, &kLayerStateKey);
                if (!holder) {
                    holder = [LSFGLayerState new];
                    holder->state = std::make_unique<LayerState>();
                    objc_setAssociatedObject(layer, &kLayerStateKey, holder,
                        OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                }
            }
        }
        return *holder->state;
    }

    LayerState* existingLayerState(CAMetalLayer* layer) {
        LSFGLayerState* holder = objc_getAssociatedObject(layer, &kLayerStateKey);
        return holder ? holder->state.get() : nullptr;
    }

    /// apply the layer properties frame generation depends on
    void configureLayer(CAMetalLayer* layer, const ls::GameConf& profile) {
        const InternalScope internal;
        if (layer.framebufferOnly)
            layer.framebufferOnly = NO; // drawables must be readable
        if (layer.maximumDrawableCount < 3)
            layer.maximumDrawableCount = 3; // the real frame is held while generated ones are acquired
        if (profile.force_vsync && !layer.displaySyncEnabled)
            layer.displaySyncEnabled = YES; // generated frames must not be dropped
    }

    // ========================================================================================
    // frame generation
    // ========================================================================================

    /// record when a drawable actually appears on screen (or that it was dropped)
    void trackPresented(id<CAMetalDrawable> drawable, const std::shared_ptr<LayerState::Display>& display,
            const std::string& captureName = {}) {
        std::shared_ptr<LayerState::Display> d = display;
        const std::string name = captureName; // copied into the block (not the reference)
        uint64_t index;
        std::string logPath;
        {
            const std::lock_guard<std::mutex> lock(d->m);
            index = ++d->submitted;
            if (!captureName.empty())
                logPath = d->captureLog;
        }
        [drawable addPresentedHandler:^(id<MTLDrawable> shown) {
            const double t = shown.presentedTime;
            const std::lock_guard<std::mutex> lock(d->m);
            if (!logPath.empty()) {
                char line[160];
                std::snprintf(line, sizeof line, "%s presented at %.4f s (present #%llu)\n",
                    name.c_str(), t, static_cast<unsigned long long>(index));
                std::ofstream(logPath, std::ios::app) << line;
            }
            if (t <= 0) {
                d->dropped++;
                return;
            }
            // order check: a later present must not appear before an earlier one
            if (d->lastShownIndex > index || (d->lastShownIndex < index && t < d->lastShownTime - 1e-4))
                d->reordered++;
            if (index > d->lastShownIndex) {
                d->lastShownIndex = index;
                d->lastShownTime = t;
            }
            if (d->last > 0) {
                const double gap = t - d->last;
                if (gap > 0 && gap < 0.25) {
                    d->sumGap += gap;
                    d->maxGap = std::max(d->maxGap, gap);
                    d->gaps++;
                    if (d->ema > 0 && gap > 1.5 * d->ema)
                        d->late++;
                    d->ema = d->ema > 0 ? 0.95 * d->ema + 0.05 * gap : gap;
                }
            }
            d->last = t;
        }];
    }

    /// present a drawable directly (bypassing our own hooks)
    void presentDirect(id<CAMetalDrawable> drawable, CFTimeInterval minDuration) {
        const InternalScope internal;
        if (minDuration > 0.0)
            [drawable presentAfterMinimumDuration:minDuration];
        else
            [drawable present];
    }

    /// decide whether a drawable should go through frame generation
    bool wantsDrawable(id<MTLDrawable> drawable, DeviceState** outDevice) {
        if (!global().enabled.load(std::memory_order_relaxed))
            return false;
        if (![(id)drawable conformsToProtocol:@protocol(CAMetalDrawable)])
            return false;

        id<CAMetalDrawable> md = (id<CAMetalDrawable>)drawable;
        CAMetalLayer* layer = md.layer;
        id<MTLTexture> texture = md.texture;
        if (!layer || !texture)
            return false;

        auto& lst = layerState(layer);
        if (lst.failed)
            return false;

        const auto [profile, gen] = currentProfile();
        if (texture.framebufferOnly || !formatSupported(texture.pixelFormat)) {
            configureLayer(layer, profile); // takes effect for the next drawables
            if (!formatSupported(texture.pixelFormat) && !lst.announced) {
                lst.announced = true;
                ls::log::warn(std::string("unsupported drawable format ")
                    + std::to_string(static_cast<unsigned long>(texture.pixelFormat))
                    + ", frame generation disabled for this layer");
                lst.failed = true;
            }
            return false;
        }
        if (texture.width < 64 || texture.height < 64)
            return false;

        DeviceState* ds = deviceState(texture.device);
        if (!deviceReady(ds, profile.performance_mode))
            return false;

        *outDevice = ds;
        return true;
    }

    ///
    /// Generate and present frames for a real frame.
    ///
    /// @param drawable the real frame
    /// @param event / value GPU signal that marks the end of the game's rendering into it
    /// @param gameCommandBuffer committed command buffer of the game (classic path) or nil
    ///
    void processFrame(DeviceState* ds, id<CAMetalDrawable> drawable,
            id<MTLEvent> event, uint64_t value, id<MTLCommandBuffer> gameCommandBuffer,
            CFTimeInterval now, double gameWaited) {
        CAMetalLayer* layer = drawable.layer;
        id<MTLTexture> texture = drawable.texture;
        auto& st = layerState(layer);
        const std::lock_guard<std::mutex> lock(st.mutex);

        const auto presentReal = [&](CFTimeInterval minDuration) {
            if (gameCommandBuffer)
                [gameCommandBuffer waitUntilScheduled];
            presentDirect(drawable, minDuration);
        };

        const auto [profile, generation] = currentProfile();
        configureLayer(layer, profile);

        // (re)create the generator when the surface or profile changes
        const uint32_t width = static_cast<uint32_t>(texture.width);
        const uint32_t height = static_cast<uint32_t>(texture.height);
        const MTLPixelFormat format = texture.pixelFormat;
        static const bool pqAsSdr = [] { // LSFG_METAL_PQ_AS_SDR=1: lsfg-vk behaviour for HDR10
            const char* v = std::getenv("LSFG_METAL_PQ_AS_SDR");
            return v && *v == '1';
        }();
        ColorEncoding encoding = colorEncodingFor(format, layer.colorspace);
        if (pqAsSdr && encoding == ColorEncoding::PQ)
            encoding = ColorEncoding::SDR;
        if (!st.generator || st.profileGeneration != generation
                || !st.generator->matches(width, height, format, profile, encoding)) {
            st.generator.reset();
            try {
                backend::Instance* instance = nullptr;
                {
                    const std::lock_guard<std::mutex> l(ds->mutex);
                    instance = ds->instance.get();
                }
                st.generator = std::make_unique<FrameGenerator>(*instance, *ds->copies,
                    width, height, format, profile, encoding);
                st.profileGeneration = generation;
                ls::log::info("frame generation active: " + std::to_string(width) + "x"
                    + std::to_string(height) + " " + formatName(format) + " "
                    + colorEncodingName(st.generator->colorEncoding()) + ", "
                    + std::to_string(profile.multiplier) + "x"
                    + (instance->isFP16() ? ", fp16" : ", fp32"));
            } catch (const std::exception& e) {
                ls::log::error(std::string("failed to create frame generation context: ") + e.what());
                st.failed = true;
                presentReal(0);
                return;
            }
        }
        FrameGenerator& gen = *st.generator;

        // timing (now = when the game presented)
        if (st.lastPresent > 0) {
            const double raw = now - st.lastPresent;
            if (raw < 0.25) { // ignore pauses (loading screens, alt-tab) in the jitter statistics
                st.statsIntervalSum += raw;
                st.statsIntervalMax = std::max(st.statsIntervalMax, raw);
                st.statsIntervals++;
                if (st.frameInterval > 0 && raw > 1.5 * st.frameInterval)
                    st.statsHitches++;
            }
            const double dt = std::clamp(raw, 1.0 / 1000.0, 0.25);
            st.frameInterval = st.frameInterval > 0 ? (0.9 * st.frameInterval + 0.1 * dt) : dt;
        }
        st.lastPresent = now;
        const CFTimeInterval slice = (profile.pacing == ls::Pacing::Duration && st.frameInterval > 0)
            ? (st.frameInterval / static_cast<double>(profile.multiplier)) * 0.95 : 0.0;

        // copy the real frame and generate the intermediate frames
        // wait for the game's rendering in its own command buffer, so the GPU time of the
        // generate command buffer below measures only frame generation
        id<MTLCommandBuffer> wait = [ds->queue commandBuffer];
        wait.label = @"lsfg-metal wait for game";
        [wait encodeWaitForEvent:event value:value];
        [wait commit];
        id<MTLCommandBuffer> cb = [ds->queue commandBuffer];
        cb.label = @"lsfg-metal generate";
        const bool capturing = st.capture.poll(layer, texture,
            "profile: " + profile.name + ", multiplier " + std::to_string(profile.multiplier)
            + ", flow scale " + std::to_string(profile.flow_scale)
            + (profile.performance_mode ? ", performance mode" : ", quality mode")
            + "\ncolor encoding: " + colorEncodingName(gen.colorEncoding())
            + "\npresent path: " + (gameCommandBuffer ? "MTLCommandBuffer presentDrawable" : "Metal 4 / drawable present"));
        if (capturing) {
            {
                const std::lock_guard<std::mutex> dl(st.display->m);
                st.display->captureLog = st.capture.directory() + "/timing.txt";
            }
            st.capture.capture(cb, texture, st.capture.sequence(gen.count()), "real", now);
        }
        try {
            gen.encodeCopyIn(cb, texture);
            gen.encodeGenerate(cb);
        } catch (const std::exception& e) {
            ls::log::error(std::string("failed to encode frame generation: ") + e.what());
            [cb commit];
            st.failed = true;
            st.generator.reset();
            presentReal(0);
            return;
        }
        {
            std::shared_ptr<LayerState::Gpu> gpuStats = st.gpu;
            [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                const double t = done.GPUEndTime - done.GPUStartTime;
                if (t <= 0 || t > 1.0) return;
                const auto us = static_cast<uint64_t>(t * 1e6);
                gpuStats->sumUs += us;
                gpuStats->count++;
                uint64_t prev = gpuStats->maxUs.load();
                while (us > prev && !gpuStats->maxUs.compare_exchange_weak(prev, us)) {}
            }];
        }
        [cb commit];

        // Each frame is presented from the completion handler of the command buffer that renders
        // it: a drawable is shown once its GPU work is done, not in present order, so presenting
        // a generated frame that is still being rendered right before the long finished real frame
        // lets the real frame overtake it (seen as judder, e.g. with HDR on an external display).
        // Completion handlers of one queue run in commit order, so frames stay in order, and the
        // worker never blocks on the GPU. Each generated frame is presented as soon as it is done,
        // so the display advances and frees a drawable for the next one (3x and up).
        std::shared_ptr<LayerState::Display> display = st.display;
        const CFTimeInterval duration = slice;
        size_t presented = 0;
        if (gen.frameIndex() > 1) { // the very first frame only primes the history
            for (size_t i = 0; i < gen.count(); i++) {
                id<CAMetalDrawable> target = nil;
                {
                    const InternalScope internal;
                    target = [layer nextDrawable]; // may block until a drawable is free (FIFO pacing)
                }
                if (!target) {
                    st.statsMissed++;
                    break;
                }
                if (target.texture.width != texture.width || target.texture.height != texture.height
                        || target.texture.pixelFormat != format) {
                    break; // layer was resized in the meantime
                }

                id<MTLCommandBuffer> copy = [ds->queue commandBuffer];
                copy.label = @"lsfg-metal present generated";
                gen.encodeCopyOut(copy, i, target.texture);
                std::string name;
                if (capturing) {
                    st.capture.capture(copy, target.texture, st.capture.sequence(i), "generated",
                        CACurrentMediaTime());
                    name = st.capture.fileName(st.capture.sequence(i), "generated");
                }
                [copy addCompletedHandler:^(id<MTLCommandBuffer>) {
                    @autoreleasepool {
                        trackPresented(target, display, name);
                        presentDirect(target, duration);
                    }
                }];
                [copy commit];
                presented++;
            }
        }

        // the real frame last, through the queue as well so it can't overtake the generated ones
        {
            id<MTLCommandBuffer> last = [ds->queue commandBuffer];
            last.label = @"lsfg-metal present real";
            const std::string name = capturing
                ? st.capture.fileName(st.capture.sequence(gen.count()), "real") : std::string();
            id<CAMetalDrawable> real = drawable;
            [last addCompletedHandler:^(id<MTLCommandBuffer>) {
                @autoreleasepool {
                    trackPresented(real, display, name);
                    presentDirect(real, duration); // the game's rendering finished before ours started
                }
            }];
            [last commit];
        }
        if (capturing)
            st.capture.finishFrame(gen.count() + 1);

        // statistics
        st.recordWait(gameWaited);
        st.statsReal++;
        st.statsGenerated += presented;
        if (st.statsStart == 0)
            st.statsStart = now;
        if (now - st.statsStart >= 5.0) {
            const double t = now - st.statsStart;
            const auto ms = [](double v) {
                char b[32];
                std::snprintf(b, sizeof b, "%.1f", v * 1000.0);
                return std::string(b);
            };
            std::string line = "real " + std::to_string(static_cast<int>(std::lround(st.statsReal / t)))
                + " fps, generated " + std::to_string(static_cast<int>(std::lround(st.statsGenerated / t)))
                + " fps, total " + std::to_string(static_cast<int>(
                    std::lround((st.statsReal + st.statsGenerated) / t))) + " fps";
            if (st.statsIntervals)
                line += " | frame time avg " + ms(st.statsIntervalSum / static_cast<double>(st.statsIntervals))
                    + " ms, max " + ms(st.statsIntervalMax) + " ms, "
                    + std::to_string(st.statsHitches) + " hitches";
            const uint64_t gpuCount = st.gpu->count.exchange(0);
            const uint64_t gpuSum = st.gpu->sumUs.exchange(0);
            const uint64_t gpuMax = st.gpu->maxUs.exchange(0);
            if (gpuCount)
                line += " | lsfg gpu avg " + ms(static_cast<double>(gpuSum) / static_cast<double>(gpuCount) / 1e6)
                    + " ms, max " + ms(static_cast<double>(gpuMax) / 1e6) + " ms";
            // per real frame: time the game spent waiting for lsfg-metal (present + next drawable)
            const uint64_t waitSum = st.waitSumUs.exchange(0);
            const uint64_t waitMax = st.waitMaxUs.exchange(0);
            st.waitCount.exchange(0);
            if (st.statsReal)
                line += " | game waited avg " + ms(static_cast<double>(waitSum) / 1e6 / static_cast<double>(st.statsReal))
                    + " ms/frame, max " + ms(static_cast<double>(waitMax) / 1e6) + " ms";
            {
                // on screen: gaps between consecutive frames as displayed; "late" = a frame stayed
                // up noticeably longer than usual (visible stutter), "dropped" = never shown
                auto& d = *st.display;
                const std::lock_guard<std::mutex> dl(d.m);
                if (d.gaps)
                    line += " | on screen avg " + ms(d.sumGap / static_cast<double>(d.gaps))
                        + " ms, max " + ms(d.maxGap) + " ms, " + std::to_string(d.late) + " late, "
                        + std::to_string(d.dropped) + " dropped, "
                        + std::to_string(d.reordered) + " out of order";
                else if (d.dropped)
                    line += " | on screen: nothing shown (window hidden?), "
                        + std::to_string(d.dropped) + " dropped";
                d.sumGap = d.maxGap = 0;
                d.gaps = d.late = d.dropped = d.reordered = 0;
            }
            if (st.statsMissed)
                line += " | " + std::to_string(st.statsMissed) + " drawable timeouts";
            ls::log::info(line);
            st.statsStart = now;
            st.statsReal = st.statsGenerated = st.statsMissed = 0;
            st.statsIntervalSum = st.statsIntervalMax = 0;
            st.statsIntervals = st.statsHitches = 0;
        }
    }

    /// hand a real frame to the layer's worker (called on the game's thread)
    void submitFrame(DeviceState* ds, id<CAMetalDrawable> drawable,
            id<MTLEvent> event, uint64_t value, id<MTLCommandBuffer> gameCommandBuffer) {
        auto& st = layerState(drawable.layer);
        const CFTimeInterval now = CACurrentMediaTime();
        static const bool sync = [] { // LSFG_METAL_SYNC_PRESENT=1: old behaviour, on the game thread
            const char* v = std::getenv("LSFG_METAL_SYNC_PRESENT");
            return v && *v == '1';
        }();
        if (sync) {
            @autoreleasepool {
                processFrame(ds, drawable, event, value, gameCommandBuffer, now, 0.0);
            }
            st.recordWait(CACurrentMediaTime() - now);
            return;
        }
        dispatch_semaphore_wait(st.inflight, DISPATCH_TIME_FOREVER);
        const double waited = CACurrentMediaTime() - now;
        dispatch_semaphore_t inflight = st.inflight;
        dispatch_async(st.worker, ^{
            @autoreleasepool {
                processFrame(ds, drawable, event, value, gameCommandBuffer, now, waited);
            }
            dispatch_semaphore_signal(inflight);
        });
    }

    // ========================================================================================
    // hook installation
    // ========================================================================================

    std::mutex& hookMutex() {
        static auto* m = new std::mutex();
        return *m;
    }
    std::set<IMP>& ourImps() {
        static auto* s = new std::set<IMP>();
        return *s;
    }
    std::set<Method>& hookedMethods() {
        static auto* s = new std::set<Method>();
        return *s;
    }

    /// replace the implementation of cls's method for sel with a block built from the original IMP
    template<typename Fn, typename MakeBlock>
    bool hook(Class cls, SEL sel, MakeBlock makeBlock) {
        if (!cls) return false;
        Method m = class_getInstanceMethod(cls, sel);
        if (!m) return false;

        const std::lock_guard<std::mutex> lock(hookMutex());
        if (hookedMethods().contains(m) || ourImps().contains(method_getImplementation(m)))
            return true;

        auto original = reinterpret_cast<Fn>(method_getImplementation(m));
        IMP replacement = imp_implementationWithBlock(makeBlock(original));
        ourImps().insert(replacement);
        hookedMethods().insert(m);
        method_setImplementation(m, replacement);
        ls::log::debug(std::string("hooked -[") + class_getName(cls) + " " + sel_getName(sel) + "]");
        return true;
    }

    // ---------------------------------------------------------------- MTLCommandBuffer

    /// shared logic of the three presentDrawable variants
    bool interceptCommandBufferPresent(id<MTLCommandBuffer> cb, id<MTLDrawable> drawable) {
        LSFG_TRACE_ONCE(std::string("-[") + object_getClassName(cb) + " presentDrawable:*]");
        if (isInternal())
            return false;
        DeviceState* ds = nullptr;
        if (!wantsDrawable(drawable, &ds) || cb.device != ds->device)
            return false;

        LSFGPendingPresent* pending = [LSFGPendingPresent new];
        pending->drawable = (id<CAMetalDrawable>)drawable;
        objc_setAssociatedObject(cb, &kPendingKey, pending, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        return true;
    }

    /// bare-present path: signal the queue's event at the end of a game command buffer, so a
    /// later -[CAMetalDrawable present] for a drawable it renders can wait for it
    void recordCommit(id<MTLCommandBuffer> cb) {
        DeviceState* ds = findDeviceState(cb.device);
        id<MTLCommandQueue> queue = cb.commandQueue;
        if (!ds || !queue)
            return;
        LSFG_TRACE_ONCE(std::string("-[") + object_getClassName(cb) + " commit]");
        QueueSync* qs;
        {
            const std::lock_guard<std::mutex> lock(ds->queueMutex);
            QueueSync*& entry = ds->queues[(__bridge void*)queue];
            if (!entry) {
                entry = new QueueSync();
                entry->event = [ds->device newEvent];
            }
            qs = entry;
        }
        const uint64_t value = qs->value.fetch_add(1) + 1;
        [cb encodeSignalEvent:qs->event value:value];

        LSFGDrawableSignal* signal = [LSFGDrawableSignal new];
        signal->event = qs->event;
        signal->value = value;
        objc_setAssociatedObject(cb, &kCommitKey, signal, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        setCommitted(signal);
    }

    void hookCommandBufferClass(Class cls) {
        using PresentFn = void (*)(id, SEL, id);
        hook<PresentFn>(cls, @selector(presentDrawable:), [](PresentFn orig) {
            return ^(id self, id drawable) {
                if (!interceptCommandBufferPresent(self, drawable))
                    orig(self, @selector(presentDrawable:), drawable);
            };
        });

        using PresentTimeFn = void (*)(id, SEL, id, CFTimeInterval);
        hook<PresentTimeFn>(cls, @selector(presentDrawable:atTime:), [](PresentTimeFn orig) {
            return ^(id self, id drawable, CFTimeInterval t) {
                if (!interceptCommandBufferPresent(self, drawable))
                    orig(self, @selector(presentDrawable:atTime:), drawable, t);
            };
        });
        hook<PresentTimeFn>(cls, @selector(presentDrawable:afterMinimumDuration:), [](PresentTimeFn orig) {
            return ^(id self, id drawable, CFTimeInterval d) {
                if (!interceptCommandBufferPresent(self, drawable))
                    orig(self, @selector(presentDrawable:afterMinimumDuration:), drawable, d);
            };
        });

        using CommitFn = void (*)(id, SEL);
        hook<CommitFn>(cls, @selector(commit), [](CommitFn orig) {
            return ^(id self) {
                LSFGPendingPresent* pending = objc_getAssociatedObject(self, &kPendingKey);
                if (!pending) {
                    if (!isInternal() && global().enabled.load(std::memory_order_relaxed))
                        recordCommit(self);
                    orig(self, @selector(commit));
                    return;
                }
                objc_setAssociatedObject(self, &kPendingKey, nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

                id<MTLCommandBuffer> cb = self;
                DeviceState* ds = deviceState(cb.device);
                const uint64_t value = ds->eventValue.fetch_add(1) + 1;
                [cb encodeSignalEvent:ds->event value:value];
                orig(self, @selector(commit));

                submitFrame(ds, pending->drawable, ds->event, value, cb);
            };
        });

        // MoltenVK (e.g. CrossOver's 1.2.10) presents with -[CAMetalDrawable present] from a
        // scheduled handler of the command buffer that renders the frame: tell the present
        // hook which command buffer that is
        using AddHandlerFn = void (*)(id, SEL, MTLCommandBufferHandler);
        hook<AddHandlerFn>(cls, @selector(addScheduledHandler:), [](AddHandlerFn orig) {
            return ^(id self, MTLCommandBufferHandler block) {
                if (!block || isInternal() || !global().enabled.load(std::memory_order_relaxed)) {
                    orig(self, @selector(addScheduledHandler:), block);
                    return;
                }
                MTLCommandBufferHandler wrapped = ^(id<MTLCommandBuffer> cb) {
                    void* previous = pthread_getspecific(scheduledKey());
                    pthread_setspecific(scheduledKey(),
                        (__bridge void*)objc_getAssociatedObject(cb, &kCommitKey));
                    block(cb);
                    pthread_setspecific(scheduledKey(), previous);
                };
                orig(self, @selector(addScheduledHandler:), wrapped);
            };
        });
    }

    // ---------------------------------------------------------------- MTL4CommandQueue

    void hookMTL4QueueClass(Class cls) {
        using SignalFn = void (*)(id, SEL, id);
        hook<SignalFn>(cls, @selector(signalDrawable:), [](SignalFn orig) {
            return ^(id self, id drawable) {
                LSFG_TRACE_ONCE(std::string("-[") + object_getClassName(self) + " signalDrawable:]");
                DeviceState* ds = nullptr;
                if (!isInternal() && wantsDrawable(drawable, &ds)) {
                    const uint64_t value = ds->eventValue.fetch_add(1) + 1;
                    using SignalEventFn = void (*)(id, SEL, id, uint64_t);
                    reinterpret_cast<SignalEventFn>(objc_msgSend)(
                        self, @selector(signalEvent:value:), ds->event, value);

                    LSFGDrawableSignal* signal = [LSFGDrawableSignal new];
                    signal->event = ds->event;
                    signal->value = value;
                    objc_setAssociatedObject(drawable, &kSignalKey, signal,
                        OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                }
                orig(self, @selector(signalDrawable:), drawable);
            };
        });
    }

    // ---------------------------------------------------------------- CAMetalDrawable

    /// returns true if the present was handled by frame generation
    bool interceptDrawablePresent(id drawable) {
        LSFG_TRACE_ONCE(std::string("-[") + object_getClassName(drawable) + " present*]");
        if (isInternal())
            return false;
        LSFGDrawableSignal* signal = objc_getAssociatedObject(drawable, &kSignalKey);
        if (signal) { // Metal 4: signalDrawable: recorded the event
            objc_setAssociatedObject(drawable, &kSignalKey, nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            id<CAMetalDrawable> md = drawable;
            DeviceState* ds = deviceState(md.texture.device);
            submitFrame(ds, md, signal->event, signal->value, nil);
            return true;
        }

        // bare present: wait for the command buffer whose scheduled handler is presenting, or
        // else for the last one committed on this thread (present after waitUntilScheduled)
        signal = (__bridge LSFGDrawableSignal*)pthread_getspecific(scheduledKey());
        if (!signal)
            signal = (__bridge LSFGDrawableSignal*)pthread_getspecific(committedKey());
        if (!signal)
            return false;
        setCommitted(nil); // used up: never pair a later present with this commit
        DeviceState* ds = nullptr;
        if (!wantsDrawable(drawable, &ds) || signal->event.device != ds->device)
            return false;
        LSFG_TRACE_ONCE("bare present path");
        submitFrame(ds, (id<CAMetalDrawable>)drawable, signal->event, signal->value, nil);
        return true;
    }

    void hookDrawableClass(Class cls) {
        using PresentFn = void (*)(id, SEL);
        hook<PresentFn>(cls, @selector(present), [](PresentFn orig) {
            return ^(id self) {
                if (!interceptDrawablePresent(self))
                    orig(self, @selector(present));
            };
        });
        using PresentTimeFn = void (*)(id, SEL, CFTimeInterval);
        hook<PresentTimeFn>(cls, @selector(presentAtTime:), [](PresentTimeFn orig) {
            return ^(id self, CFTimeInterval t) {
                if (!interceptDrawablePresent(self))
                    orig(self, @selector(presentAtTime:), t);
            };
        });
        hook<PresentTimeFn>(cls, @selector(presentAfterMinimumDuration:), [](PresentTimeFn orig) {
            return ^(id self, CFTimeInterval d) {
                if (!interceptDrawablePresent(self))
                    orig(self, @selector(presentAfterMinimumDuration:), d);
            };
        });
    }

    // ---------------------------------------------------------------- device probing

    std::set<void*>& probedDevices() {
        static auto* s = new std::set<void*>();
        return *s;
    }

    /// discover the concrete command buffer / queue classes of a device and hook them
    void installDeviceHooks(id<MTLDevice> device) {
        {
            const std::lock_guard<std::mutex> lock(devicesMutex());
            if (!probedDevices().insert((__bridge void*)device).second)
                return;
        }
        deviceState(device); // recordCommit only signals for known devices

        @autoreleasepool {
            const InternalScope internal;
            id<MTLCommandQueue> queue = [device newCommandQueue];
            std::set<Class> classes;
            if (id<MTLCommandBuffer> cb = [queue commandBuffer])
                classes.insert(object_getClass(cb));
            if (id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences])
                classes.insert(object_getClass(cb));
            MTLCommandBufferDescriptor* desc = [MTLCommandBufferDescriptor new];
            desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            if (id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:desc])
                classes.insert(object_getClass(cb));
            for (Class cls : classes)
                hookCommandBufferClass(cls);

            if (@available(macOS 26.0, *)) {
                if ([device respondsToSelector:@selector(newMTL4CommandQueue)]) {
                    id queue4 = [device newMTL4CommandQueue];
                    if (queue4)
                        hookMTL4QueueClass(object_getClass(queue4));
                }
            }
        }
    }

    // ---------------------------------------------------------------- CAMetalLayer

    void hookLayerClass() {
        Class cls = objc_getClass("CAMetalLayer");
        if (!cls) {
            ls::log::error("CAMetalLayer not found");
            return;
        }

        using NextFn = id (*)(id, SEL);
        hook<NextFn>(cls, @selector(nextDrawable), [](NextFn orig) {
            return ^id(id self) {
                LSFG_TRACE_ONCE(std::string("-[") + object_getClassName(self) + " nextDrawable]");
                if (!isInternal()) {
                    initializeProcess();
                    if (global().enabled) {
                        CAMetalLayer* layer = self;
                        configureLayer(layer, currentProfile().first);
                        if (layer.device)
                            installDeviceHooks(layer.device);
                        // let the worker finish the previous frame first: it needs a drawable for
                        // every generated frame, and a game that grabbed the last free drawable
                        // would starve it (the layer has at most 3)
                        if (LayerState* lst = existingLayerState(layer)) {
                            const CFTimeInterval t0 = CACurrentMediaTime();
                            dispatch_sync(lst->worker, ^{});
                            lst->recordWait(CACurrentMediaTime() - t0);
                        }
                    }
                }
                id drawable = orig(self, @selector(nextDrawable));
                if (drawable && global().enabled)
                    hookDrawableClass(object_getClass(drawable));
                return drawable;
            };
        });

        using SetBoolFn = void (*)(id, SEL, BOOL);
        hook<SetBoolFn>(cls, @selector(setFramebufferOnly:), [](SetBoolFn orig) {
            return ^(id self, BOOL value) {
                if (!isInternal() && global().enabled)
                    value = NO;
                orig(self, @selector(setFramebufferOnly:), value);
            };
        });
        hook<SetBoolFn>(cls, @selector(setDisplaySyncEnabled:), [](SetBoolFn orig) {
            return ^(id self, BOOL value) {
                if (!isInternal() && global().enabled && currentProfile().first.force_vsync)
                    value = YES;
                orig(self, @selector(setDisplaySyncEnabled:), value);
            };
        });

        using SetCountFn = void (*)(id, SEL, NSUInteger);
        hook<SetCountFn>(cls, @selector(setMaximumDrawableCount:), [](SetCountFn orig) {
            return ^(id self, NSUInteger value) {
                if (!isInternal() && global().enabled && value < 3)
                    value = 3;
                orig(self, @selector(setMaximumDrawableCount:), value);
            };
        });
    }
}

@implementation LSFGLayerState
@end

// ============================================================================================
// bootstrap
// ============================================================================================
//
// Normally dyld runs lsfgMetalBootstrap() as an initializer of the inserted library. Wine's
// macOS preloader, however, starts every Windows process as an LC_UNIXTHREAD executable and
// dlopen()s Wine afterwards; in those processes dyld maps DYLD_INSERT_LIBRARIES images but
// never runs their initializers. Interposing (applied at bind time, no initializer needed)
// still works, so dlopen() is interposed and bootstraps the library on the first call after
// the Objective-C runtime and QuartzCore are usable. Wine calls dlopen() for ntdll.so,
// winemac.so, every builtin DLL's unix side and D3DMetal long before anything is presented.
//
// Consequence: this image must not rely on static initializers (checked by CMake).
//
// The same goes for the initializers of its dependencies, libc++ among them. Since macOS 27,
// libc++'s typed operator new (used inside libc++.dylib, e.g. by std::string::append) aborts
// when libc++'s initializer has not run yet. So before allocating anything, the bootstrap
// dlopen()s this image once: dyld then runs the pending initializers of the image and of
// everything it links against (the constructor below re-enters the bootstrap and returns).

namespace {
    std::atomic<int> g_bootstrapState{0}; // 0 = not done, 1 = in progress, 2 = done (constant-initialized)

    void lsfgMetalBootstrap() {
        if (g_bootstrapState.load(std::memory_order_acquire) == 2)
            return;
        int expected = 0;
        if (!g_bootstrapState.compare_exchange_strong(expected, 1))
            return; // done or in progress on another thread

        const char* disable = std::getenv("LSFG_METAL_DISABLE");
        if (disable && *disable && std::strcmp(disable, "0") != 0) {
            g_bootstrapState = 2;
            return;
        }

        // too early (runtime or QuartzCore not ready yet): try again on the next dlopen
        if (!objc_getClass("NSObject") || !objc_getClass("CAMetalLayer")) {
            g_bootstrapState = 0;
            return;
        }

        // run the initializers dyld skipped (see above); a no-op where they already ran.
        // (dlopen from this image is not interposed; the handle is kept on purpose.)
        Dl_info self{};
        if (dladdr(reinterpret_cast<const void*>(&lsfgMetalBootstrap), &self) && self.dli_fname)
            dlopen(self.dli_fname, RTLD_LAZY | RTLD_GLOBAL);

        @autoreleasepool {
            hookLayerClass();
        }
#if defined(__x86_64__)
        const char* arch = "x86_64";
#else
        const char* arch = "arm64";
#endif
        ls::log::debug(std::string("loaded into ") + (getprogname() ?: "?") + " (" + arch + ")");
        g_bootstrapState.store(2, std::memory_order_release);
    }
}

__attribute__((constructor))
static void lsfgMetalInit() {
    lsfgMetalBootstrap();
}

extern "C" void* lsfg_metal_dlopen(const char* path, int mode) {
    if (g_bootstrapState.load(std::memory_order_relaxed) != 2)
        lsfgMetalBootstrap();
    // dyld resolves @rpath / @loader_path relative to the *caller* of dlopen (its return
    // address), so this must be a real tail call: the caller's return address stays intact.
    // (references to dlopen from the interposing image itself are not interposed)
    [[clang::musttail]] return dlopen(path, mode);
}

namespace {
    struct Interpose {
        const void* replacement;
        const void* original;
    };
    __attribute__((used, section("__DATA,__interpose")))
    const Interpose g_interposeDlopen = {
        reinterpret_cast<const void*>(&lsfg_metal_dlopen),
        reinterpret_cast<const void*>(&dlopen)
    };
}
