# lsfg-metal

A macOS port of [lsfg-vk](https://github.com/PancakeTAS/lsfg-vk). It runs Lossless Scaling frame generation on Metal. It works with native Metal apps and with Windows games under Wine, including **D3DMetal** (Apple Game Porting Toolkit, CrossOver, Sikarugir, Whisky). It also works with DXVK/MoltenVK and DXMT.

Like lsfg-vk, it ships no shaders. You need your own copy of **Lossless Scaling** (Steam) for `Lossless.dll`.

## How it works

| lsfg-vk (Linux) | lsfg-metal (macOS) |
|---|---|
| Vulkan implicit layer | `liblsfg-metal.dylib`, injected with `DYLD_INSERT_LIBRARIES` |
| hooks `vkQueuePresentKHR` / swapchain | swizzles `CAMetalLayer`, `MTLCommandBuffer`, `MTL4CommandQueue` and `CAMetalDrawable` (Objective-C) |
| SPIR-V shaders from `Lossless.dll` | same shaders, translated to MSL by SPIRV-Cross at load time |
| Vulkan compute backend | Metal compute backend: a 1:1 port of all shader chains |

Every Metal swapchain on macOS goes through `CAMetalLayer`. Hooking it therefore covers D3DMetal, MoltenVK, DXMT and native games with one mechanism.

**Present interception.** It handles both present styles:
- Classic: `-[MTLCommandBuffer presentDrawable:*]` followed by `commit`.
- Metal 4: `-[MTL4CommandQueue signalDrawable:]` followed by `-[CAMetalDrawable present]`.

D3DMetal uses `presentDrawable:afterMinimumDuration:`.

**Frame generation.** Each real frame is handed to a per-layer worker thread, so the game's present call returns at once. The game's next `nextDrawable` call waits only if the worker hasn't finished the previous frame; this stops the game from taking the drawable the worker needs. For each real frame, the worker does the following:
1. Copy the drawable into a private texture. This happens on the GPU, ordered after the game's work through an `MTLEvent`.
2. Run the LSFG pipeline.
3. Present each generated frame into its own drawable from the layer.
4. Present the real frame.

Each present is issued from the completion handler of the command buffer that renders that frame. macOS shows a drawable once its GPU work is done, not in present order. Without this, the real frame (long finished) would overtake a generated frame that is still rendering, and you'd see judder.

The layer is set to `framebufferOnly = NO`, at least 3 drawables, and display sync enabled. Generated frames are paced by drawable availability (vsync), as lsfg-vk does. `pacing = 'duration'` additionally spreads them evenly.

**Wine.** Wine's macOS preloader maps `DYLD_INSERT_LIBRARIES` images but never runs their initializers. lsfg-metal therefore also bootstraps itself from an interposed `dlopen` (`__DATA,__interpose`). For this to work the library must contain exactly one static initializer, which the build checks (`cmake/check_initializers.sh`).

**Formats.** Supported drawable formats:
- BGRA8 / RGBA8, with or without sRGB.
- RGB10A2 / BGR10A2.
- BGR10_XR.
- RGBA16Float. 10-bit, XR and fp16 drawables use the HDR (fp16) path.

**Shader loading.** Shaders compile in the background on first use. Frames pass through untouched until they are ready (about 0.3 s on M-series GPUs).

## Building

You need the Xcode SDK, CMake ≥ 3.20, and git.

```sh
git submodule update --init                  # SPIRV-Cross (thirdparty/SPIRV-Cross)
cmake -S . -B build \
  -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
cmake --build build -j
cmake --install build --prefix ~/.local     # optional: lib/liblsfg-metal.dylib, bin/lsfg-metal-run, bin/lsfg-metal-crossover, bin/lsfg-metal-cli
```

The library is universal (arm64 + x86_64). Wine/CrossOver/D3DMetal run as x86_64 under Rosetta 2. Native games are usually arm64.

## Usage

### Native Metal games

```sh
lsfg-metal-run -m 2 /Applications/SomeGame.app/Contents/MacOS/SomeGame
```

### Wine / D3DMetal

Every process in the Wine tree must see `DYLD_INSERT_LIBRARIES`. lsfg-metal only does work in processes that create a Metal drawable. With a config file it only does so for games matching a profile.

**SIP caveat.** macOS removes `DYLD_*` variables when it launches a SIP-protected binary such as `/bin/sh`, `/bin/bash`, `/usr/bin/env`, `/usr/bin/arch` or `perl`. Any variable set *before* such a binary runs is lost.
- The variable has to be exported by the last script in the chain, right before it starts a real Mach-O executable such as `wine`, `wine64`, `wine-preloader`, or a wrapper's `Contents/MacOS/launcher`.
- For Rosetta, use `arch -x86_64 -e DYLD_INSERT_LIBRARIES=... cmd`, not `DYLD_INSERT_LIBRARIES=... arch -x86_64 cmd`.

**Plain wine / GPTK from a terminal**

```sh
export DYLD_INSERT_LIBRARIES=~/.local/lib/liblsfg-metal.dylib
wine game.exe
# or with a fixed multiplier, ignoring conf.toml:
lsfg-metal-run -m 3 wine game.exe
```

**Sikarugir / Kegworks / Wineskin wrappers.** Use `lsfg-metal-sikarugir`:

```sh
lsfg-metal-sikarugir ~/Applications/Sikarugir/MyGame.app            # the wrapper's configured program
lsfg-metal-sikarugir -m 2 ~/Applications/Sikarugir/MyGame.app 'C:\path\game.exe'
```

The wrapper's own launcher passes the environment on to Wine, but it drops `DYLD_*` variables. The script therefore starts the wrapper's Wine directly, with the environment Sikarugir sets up: prefix, renderer (D3DMetal/DXMT/DXVK from the wrapper's settings), msync, and the AVX advertisement for Rosetta. It adds lsfg-metal on top and does not modify the wrapper. The wrapper's Wine is ad-hoc signed without the hardened runtime, so no re-signing is needed.

**CrossOver.** Use `lsfg-metal-crossover`:

```sh
lsfg-metal-crossover enable Steam                                      # once per bottle
lsfg-metal-crossover run Steam "C:\\Program Files (x86)\\Steam\\steam.exe"   # games launched from this Steam get lsfg-metal
lsfg-metal-crossover run -m 3 Steam "C:\\Games\\Game\\game.exe"          # or a game directly, fixed 3x
lsfg-metal-crossover disable Steam
```

Why this is needed: CrossOver's Wine loader is notarized with the hardened runtime and lacks the `allow-dyld-environment-variables` entitlement. macOS therefore ignores `DYLD_INSERT_LIBRARIES` for it, wherever the variable is set. CrossOver also resets the D3DMetal library path itself, so a shim library can't be slipped in that way either.

The script does **not** modify CrossOver.app:
- `setup` (run automatically) makes an APFS clone of `CrossOver.app/Contents/SharedSupport/CrossOver` in `~/Library/Application Support/lsfg-metal/crossover`. A clone uses no extra disk space.
- In that clone it re-signs only the Wine loader, ad-hoc, with its original entitlements plus `allow-dyld-environment-variables`.
- `enable` adds `DYLD_INSERT_LIBRARIES` to the bottle's `[EnvironmentVariables]`. The patched clone honours it; the original CrossOver ignores it.

So games started from the CrossOver window run without lsfg-metal, and games started through `lsfg-metal-crossover run` get it. Starting Steam (or another launcher) through `run` is enough: its child processes get it too. Your `conf.toml` profiles decide which games actually use frame generation. The clone is refreshed automatically when CrossOver updates.

**Whisky** (untested). Whisky launches Wine itself, so run its bundled wine binary from a terminal, as in "Plain wine" above (`~/Library/Application Support/com.isaacmarovitz.Whisky/Libraries/Wine/bin/wine64`, `WINEPREFIX=<bottle path>`).

**Checking that it works.** Set `LSFG_METAL_LOG=/tmp/lsfg.log`; add `LSFG_METAL_DEBUG=1` to also see `loaded into <process>` for every process it reaches. A working setup logs these lines:

```
lsfg-metal[1234] info: using profile "2x FG / 100%": multiplier 2, flow scale 1.000000
lsfg-metal[1234] info: frame generation active: 2560x1440 BGRA8Unorm, 2x, fp16
lsfg-metal[1234] info: real 60 fps, generated 60 fps, total 120 fps | frame time avg 16.7 ms, max 18.0 ms, 0 hitches | lsfg gpu avg 2.1 ms, max 3.0 ms | game waited avg 0.1 ms/frame, max 0.9 ms
```

The stats line appears every 5 seconds:
- **frame time** is the interval between the game's real frames. **hitches** counts frames that took more than 1.5× the average. Loading screens are excluded.
- **lsfg gpu** is the GPU time of frame generation itself.
- **game waited** is how long the game was held up by lsfg-metal.
- **on screen** shows what macOS actually displayed: the average gap between frames, frames that stayed up an extra refresh (late), frames that were never shown (dropped), and frames shown in the wrong order (out of order; should always be 0).

If real fps is below half the refresh rate (2x) or hitches are frequent, try `performance_mode = true` or a lower `flow_scale`.

`LSFG_METAL_DISABLE=1` turns the library off without removing it from the environment.

**Capturing frames.** To show what is actually displayed, run `touch ~/lsfg-capture` while the game runs. Capturing starts 5 seconds later (`echo 10 > ~/lsfg-capture` waits 10 seconds instead). The next 20 real frames and their generated frames are written as PNGs, numbered in display order, to `~/lsfg-capture-<time>/`, together with `info.txt` (the layer configuration). HDR10 frames are written as stored, so they look washed out.

**HDR.** The colour encoding is taken from the layer's colour space:
- **HDR10 (PQ):** used e.g. by D3DMetal with an HDR10 swapchain. Frames are converted to linear scRGB for frame generation, with the shaders' `HdrSupport` flag set, as Lossless Scaling does with Windows HDR. They are converted back to PQ on output.
- **Linear EDR:** handled the same way, without the conversion.
- **Everything else:** treated as SDR.

## Configuration

The config lives in `~/.config/lsfg-metal/conf.toml`, or at `$LSFG_METAL_CONFIG`. It is created on first run, uses the lsfg-vk v2 format, and reloads while the game runs.

```toml
version = 2

[global]
# dll = '/path/to/Lossless.dll'   # default: auto-detected (see below)
allow_fp16 = true

[[profile]]
name = "2x FG / 100%"
active_in = ['Cyberpunk2077.exe', 'SomeNativeGame']
multiplier = 2          # 2..16
flow_scale = 1.0        # 0.25..1.0, lower = faster
performance_mode = false
pacing = 'none'         # 'none' or 'duration'
force_vsync = true
```

**Profile matching.** `active_in` is compared, in this order, against:
1. the end of the Windows `.exe` path (Wine);
2. the end of the macOS executable path;
3. the process name.

`LSFG_METAL_PROFILE=<name>` forces a profile.

**Finding `Lossless.dll`.** The automatic search checks, in order:
1. `$WINEPREFIX`;
2. CrossOver bottles, Whisky bottles, Sikarugir/Kegworks/Wineskin wrappers in `~/Applications`, Heroic prefixes and `~/.wine`, including extra Steam libraries listed in `libraryfolders.vdf`;
3. the macOS Steam folder;
4. the current directory.

`lsfg-metal-cli find-dll` shows what it finds.

### Environment variables

Every option also accepts the lsfg-vk spelling (`LSFGVK_*`).

| Variable | Meaning |
|---|---|
| `LSFG_METAL_ENV=1` | ignore conf.toml and configure entirely from the environment |
| `LSFG_METAL_MULTIPLIER`, `_FLOW_SCALE`, `_PERFORMANCE_MODE`, `_PACING`, `_FORCE_VSYNC` | profile settings (environment mode) |
| `LSFG_METAL_DLL_PATH` | path to Lossless.dll |
| `LSFG_METAL_NO_FP16=1` | use the fp32 shaders |
| `LSFG_METAL_CONFIG`, `LSFG_METAL_PROFILE` | config path, forced profile name |
| `LSFG_METAL_LOG=FILE`, `LSFG_METAL_DEBUG=1` | logging |
| `LSFG_METAL_DISABLE=1` | turn the library off |
| `LSFG_METAL_SYNC_PRESENT=1` | process frames on the game's thread (old behaviour, for comparison) |
| `LSFG_METAL_PQ_AS_SDR=1` | treat HDR10 (PQ) output like SDR, as lsfg-vk does (for comparison) |
| `LSFG_METAL_DUMP_MSL=1`, `LSFG_METAL_FAST_MATH=1` | dump translated shaders to `$TMPDIR/lsfg-metal-msl`, compile with fast math |

## Tools and tests

```sh
build/lsfg-metal-cli validate                 # translate + compile all shaders from Lossless.dll
build/lsfg-metal-cli benchmark -W 2560 -H 1440 -m 3
build/lsfg-metal-cli selftest --dll fake.dll  # end-to-end check of the pipeline (fake DLL only)
build/lsfg-metal-cli motiontest -m 3 --out /tmp/motion   # real DLL: checks generated frames follow motion, writes PNGs
build/lsfg-metal-cli dump-msl --out msl/
```

The real shaders are proprietary, so the tests use a synthetic DLL. It has the same resource IDs, bindings and dispatch sizes, but simple, predictable math:

```sh
brew install glslang
python3 tests/build_fake_dll.py fake.dll
build/lsfg-metal-cli selftest --dll fake.dll
# native presenter test (classic and Metal 4 present paths)
LSFG_METAL_ENV=1 LSFG_METAL_MULTIPLIER=2 LSFG_METAL_DLL_PATH=fake.dll \
  DYLD_INSERT_LIBRARIES=build/liblsfg-metal.dylib build/lsfg-metal-testapp --mode mtl4 --fps 30
```

`tests/d3d11test/d3d11test.c` is a minimal D3D11 flip-model app for testing under Wine + D3DMetal:

```sh
x86_64-w64-mingw32-gcc -O2 -o d3d11test.exe tests/d3d11test/d3d11test.c -ld3d11 -ldxgi -luser32 -lgdi32 -luuid
```

## Status / limitations

- **Tested** on an M5 Max with macOS 27:
  - the native test app on both present paths, at 8-bit, sRGB, 10-bit and fp16;
  - arm64, and x86_64 under Rosetta;
  - Wine 11 + D3DMetal with a D3D11 flip-discard swapchain, both in a Sikarugir wrapper (through `lsfg-metal-sikarugir`) and in CrossOver Preview 27 (through `lsfg-metal-crossover`): 30 → 60 / 90 / 120 fps at 2x / 3x / 4x.
  - the real `Lossless.dll`: all 51 shaders (fp16 and fp32) translate and compile, and they run cleanly under Metal API and GPU validation.
- **Motion check with the real shaders:** `lsfg-metal-cli motiontest` moves a textured square 24 px per frame over a textured background and compares each generated frame to a reference rendered at the true in-between position. Results:
  - the square lands within 1–3 px of where it should be;
  - mean error is about 2.5× lower than cross-fading or repeating frames (≈3.5 vs ≈9 on a 0–255 scale);
  - some edge warping remains where background is revealed or covered, which is normal for optical-flow interpolation.
- **Tested in a real game:** S.T.A.L.K.E.R. Anomaly / GAMMA (D3D11 → D3DMetal) in a Sikarugir wrapper via `lsfg-metal-sikarugir`: 1080p, 2x, performance mode, 60 → 120 fps on a 120 Hz display.
- **Not tested:** the Whisky setup.
- Generated frames add latency, as on Linux. The real frame is shown after the generated ones.
- Frame generation needs more drawables, and the GPU time grows with resolution. Use `performance_mode` and a lower `flow_scale` at high resolutions.
- Games that render with OpenGL (wined3d) are not supported. lsfg-metal needs a `CAMetalLayer`.

## License

GPL-3.0-or-later, the same as lsfg-vk. See `LICENSE.md`.
