/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * lsfg-metal-testapp: a minimal Metal app presenting to a CAMetalLayer at a fixed rate,
 * used to exercise the hook library. Supports both presentation styles D3DMetal uses:
 *
 *   --mode classic   [commandBuffer presentDrawable:] + commit           (also MoltenVK)
 *   --mode mtl4      MTL4CommandQueue waitForDrawable/signalDrawable + [drawable present]
 *   --mode scheduled [drawable present] from a scheduled handler      (MoltenVK 1.2.10, CrossOver)
 *   --mode waitscheduled  commit + waitUntilScheduled + [drawable present]
 *
 * Renders a vertical bar sweeping across the window (classic) or a cycling clear color
 * (mtl4), prints how many of its own drawables were actually presented.
 */

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {
    struct Options {
        std::string mode = "classic";
        std::string present = "drawable"; // drawable | duration | time
        MTLPixelFormat format = MTLPixelFormatBGRA8Unorm;
        double fps = 30;
        double seconds = 6;
        bool framebufferOnly = true;
        NSUInteger drawables = 2;
        bool edr = false; // HDR10 like D3DMetal: EDR + PQ colorspace
        double cpuMs = 0; // simulated game CPU work per frame, before acquiring the drawable
    };

    Options g_options;
    std::atomic<uint64_t> g_rendered{0};
    std::atomic<uint64_t> g_presented{0};

    const char* SHADER = R"msl(
#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; };
vertex V bar_vertex(uint vid [[vertex_id]], constant float& x [[buffer(0)]]) {
    const float2 corners[6] = { {0,-1}, {1,-1}, {0,1}, {1,-1}, {1,1}, {0,1} };
    float2 c = corners[vid];
    V o; o.pos = float4(x + c.x * 0.1, c.y, 0, 1); return o;
}
fragment float4 bar_fragment(V in [[stage_in]]) { return float4(1.0, 0.85, 0.2, 1.0); }
)msl";
}

@interface Renderer : NSObject
- (instancetype)initWithLayer:(CAMetalLayer*)layer;
- (void)run;
@end

@implementation Renderer {
    CAMetalLayer* _layer;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLRenderPipelineState> _pipeline;
    id _queue4;      // id<MTL4CommandQueue>
    id _allocator4;  // id<MTL4CommandAllocator>
    id<MTLSharedEvent> _event4;
    uint64_t _event4Value;
}

- (instancetype)initWithLayer:(CAMetalLayer*)layer {
    if ((self = [super init])) {
        _layer = layer;
        _device = layer.device;
        _queue = [_device newCommandQueue];

        NSError* error = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:@(SHADER) options:nil error:&error];
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"bar_vertex"];
        d.fragmentFunction = [lib newFunctionWithName:@"bar_fragment"];
        d.colorAttachments[0].pixelFormat = layer.pixelFormat;
        _pipeline = [_device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!_pipeline) { NSLog(@"pipeline: %@", error); std::exit(1); }

        if (g_options.mode == "mtl4") {
            if (@available(macOS 26.0, *)) {
                _queue4 = [_device newMTL4CommandQueue];
                _allocator4 = [_device newCommandAllocator];
                _event4 = [_device newSharedEvent];
            }
            if (!_queue4) { std::fprintf(stderr, "Metal 4 is not available\n"); std::exit(2); }
        }
    }
    return self;
}

- (MTLClearColor)colorForFrame:(uint64_t)frame {
    const double h = std::fmod(static_cast<double>(frame) * 0.02, 1.0) * 6.0;
    const double x = 1.0 - std::fabs(std::fmod(h, 2.0) - 1.0);
    double r = 0, g = 0, b = 0;
    switch (static_cast<int>(h)) {
        case 0: r = 1; g = x; break;
        case 1: r = x; g = 1; break;
        case 2: g = 1; b = x; break;
        case 3: g = x; b = 1; break;
        case 4: r = x; b = 1; break;
        default: r = 1; b = x; break;
    }
    return MTLClearColorMake(r * 0.4, g * 0.4, b * 0.4, 1.0);
}

- (void)presentClassic:(id<CAMetalDrawable>)drawable frame:(uint64_t)frame {
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = [self colorForFrame:frame];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    float x = static_cast<float>(std::fmod(static_cast<double>(frame) * 0.02, 2.0) - 1.0);
    [enc setRenderPipelineState:_pipeline];
    [enc setVertexBytes:&x length:sizeof(x) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];

    if (g_options.mode == "scheduled") { // MoltenVK 1.2.10 (CrossOver)
        [cb addScheduledHandler:^(id<MTLCommandBuffer>) { [drawable present]; }];
        [cb commit];
        return;
    }
    if (g_options.mode == "waitscheduled") {
        [cb commit];
        [cb waitUntilScheduled];
        [drawable present];
        return;
    }
    if (g_options.present == "duration")
        [cb presentDrawable:drawable afterMinimumDuration:1.0 / g_options.fps];
    else if (g_options.present == "time")
        [cb presentDrawable:drawable atTime:CACurrentMediaTime()];
    else
        [cb presentDrawable:drawable];
    [cb commit];
}

- (void)presentMTL4:(id<CAMetalDrawable>)drawable frame:(uint64_t)frame API_AVAILABLE(macos(26.0)) {
    // wait for the previous frame so the allocator can be reset
    [_event4 waitUntilSignaledValue:_event4Value timeoutMS:1000];
    [(id<MTL4CommandAllocator>)_allocator4 reset];

    id<MTL4CommandBuffer> cb = [_device newCommandBuffer];
    [cb beginCommandBufferWithAllocator:_allocator4];
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = drawable.texture;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = [self colorForFrame:frame];
    id<MTL4RenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc endEncoding];
    [cb endCommandBuffer];

    id<MTL4CommandQueue> q = _queue4;
    [q waitForDrawable:drawable];
    const id<MTL4CommandBuffer> cbs[1] = { cb };
    [q commit:cbs count:1];
    [q signalDrawable:drawable];
    [q signalEvent:_event4 value:++_event4Value];
    [drawable present];
}

- (void)run {
    const double interval = 1.0 / g_options.fps;
    const double start = CACurrentMediaTime();
    double next = start;
    uint64_t frame = 0;
    while (CACurrentMediaTime() - start < g_options.seconds) {
        @autoreleasepool {
            if (g_options.cpuMs > 0) { // busy work like a game's simulation/encoding
                const double until = CACurrentMediaTime() + g_options.cpuMs / 1000.0;
                while (CACurrentMediaTime() < until) {}
            }
            id<CAMetalDrawable> drawable = [_layer nextDrawable];
            if (!drawable) continue;
            [drawable addPresentedHandler:^(id<MTLDrawable> d) {
                if (d.presentedTime > 0) g_presented++;
            }];
            if (g_options.mode == "mtl4") {
                if (@available(macOS 26.0, *))
                    [self presentMTL4:drawable frame:frame];
            } else {
                [self presentClassic:drawable frame:frame];
            }
            g_rendered++;
            frame++;
        }
        next += interval;
        const double sleep = next - CACurrentMediaTime();
        if (sleep > 0)
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
        else
            next = CACurrentMediaTime();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const double elapsed = CACurrentMediaTime() - start;
    std::printf("testapp: mode=%s rendered=%llu presented=%llu (%.1f fps) over %.1fs\n",
        g_options.mode.c_str(),
        static_cast<unsigned long long>(g_rendered.load()),
        static_cast<unsigned long long>(g_presented.load()),
        static_cast<double>(g_rendered.load()) / elapsed, elapsed);
    std::fflush(stdout);
    dispatch_async(dispatch_get_main_queue(), ^{ [NSApp terminate:nil]; });
}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate {
    NSWindow* _window;
    Renderer* _renderer;
}
- (void)applicationDidFinishLaunching:(NSNotification*)note {
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(200, 200, 960, 540)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        backing:NSBackingStoreBuffered defer:NO];
    _window.title = @"lsfg-metal-testapp";

    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = MTLCreateSystemDefaultDevice();
    layer.pixelFormat = g_options.format;
    layer.framebufferOnly = g_options.framebufferOnly;
    layer.maximumDrawableCount = g_options.drawables;
    layer.contentsScale = _window.backingScaleFactor;
    if (g_options.format == MTLPixelFormatRGBA16Float)
        layer.wantsExtendedDynamicRangeContent = YES;
    if (g_options.edr) {
        layer.wantsExtendedDynamicRangeContent = YES;
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
        layer.colorspace = cs;
        CGColorSpaceRelease(cs);
    }

    NSView* view = _window.contentView;
    view.wantsLayer = YES;
    view.layer = layer;
    layer.drawableSize = CGSizeMake(view.bounds.size.width * layer.contentsScale,
                                    view.bounds.size.height * layer.contentsScale);
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];

    _renderer = [[Renderer alloc] initWithLayer:layer];
    Renderer* renderer = _renderer;
    std::thread([renderer] { [renderer run]; }).detach();
}
@end

int main(int argc, const char** argv) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--mode") g_options.mode = next();
        else if (a == "--present") g_options.present = next();
        else if (a == "--fps") g_options.fps = std::stod(next());
        else if (a == "--seconds") g_options.seconds = std::stod(next());
        else if (a == "--drawables") g_options.drawables = std::stoul(next());
        else if (a == "--readable") g_options.framebufferOnly = false;
        else if (a == "--cpu-ms") g_options.cpuMs = std::stod(next());
        else if (a == "--edr") g_options.edr = true;
        else if (a == "--format") {
            const std::string f = next();
            if (f == "bgra8") g_options.format = MTLPixelFormatBGRA8Unorm;
            else if (f == "bgra8srgb") g_options.format = MTLPixelFormatBGRA8Unorm_sRGB;
            else if (f == "rgba16f") g_options.format = MTLPixelFormatRGBA16Float;
            else if (f == "rgb10a2") g_options.format = MTLPixelFormatRGB10A2Unorm;
            else { std::fprintf(stderr, "unknown format %s\n", f.c_str()); return 1; }
        } else {
            std::fprintf(stderr, "usage: %s [--mode classic|mtl4|scheduled|waitscheduled][--present drawable|duration|time] "
                "[--format bgra8|bgra8srgb|rgba16f|rgb10a2] [--fps N] [--seconds N] [--drawables 2|3] [--cpu-ms N] [--edr]\n", argv[0]);
            return 1;
        }
    }

    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        app.activationPolicy = NSApplicationActivationPolicyRegular;
        AppDelegate* delegate = [AppDelegate new];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
