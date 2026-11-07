#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Measure frame-to-frame camera motion in an lsfg-metal frame capture (~/lsfg-capture-<time>/).

For each consecutive pair of displayed frames, prints the global image shift (phase
correlation). With correct frame generation, the two halves of every real-frame step
(real -> generated -> real) are equal; uneven real-frame steps mean the game itself
moves the camera unevenly (frame pacing / input), which frame generation cannot fix.

usage: analyze_capture.py CAPTURE_DIR      (needs numpy; uses macOS sips to read PNGs)
"""
import glob, os, struct, subprocess, sys, tempfile
import numpy as np
def load(p):
    b = open(p,'rb').read()
    off = struct.unpack_from('<I', b, 10)[0]
    w, h = struct.unpack_from('<ii', b, 18)
    bpp = struct.unpack_from('<H', b, 28)[0]
    row = ((w * bpp // 8) + 3) & ~3
    a = np.frombuffer(b, np.uint8, count=row*abs(h), offset=off).reshape(abs(h), row)[:, :w*bpp//8]
    a = a.reshape(abs(h), w, bpp//8)[..., :3].astype(np.float32)
    if h > 0: a = a[::-1]
    return a.mean(axis=2)
def shift(a, b):
    # phase correlation on a centred window, sub-pixel peak
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    A = np.fft.fft2((a - a.mean()) * win); B = np.fft.fft2((b - b.mean()) * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    y, x = np.unravel_index(np.argmax(r), r.shape)
    def sub(c, n, get):
        l, m, rr = get((c-1) % n), get(c), get((c+1) % n)
        d = (l - rr) / (2*(l - 2*m + rr) + 1e-12)
        return c + d
    fx = sub(x, w, lambda i: r[y, i]); fy = sub(y, h, lambda i: r[i, x])
    if fx > w/2: fx -= w
    if fy > h/2: fy -= h
    return fx, fy, r.max()
if len(sys.argv) != 2:
    print(__doc__); sys.exit(1)
tmp = tempfile.mkdtemp(prefix="lsfg-capture-")
for png in sorted(glob.glob(os.path.join(sys.argv[1], '*.png'))):
    subprocess.run(["sips", "-s", "format", "bmp", png, "--out",
                    os.path.join(tmp, os.path.basename(png)[:-4] + ".bmp")], check=True, capture_output=True)
files = sorted(glob.glob(os.path.join(tmp, '*.bmp')))
imgs = [load(f) for f in files]
print("frame pair                      dx(px)   dy(px)")
for i in range(1, len(imgs)):
    a, b = imgs[i-1], imgs[i]
    # middle region (skip HUD at edges)
    h, w = a.shape
    sl = (slice(h//6, h*5//6), slice(w//8, w*7//8))
    dx, dy, peak = shift(b[sl], a[sl])
    print(f"{os.path.basename(files[i-1])[:-4]:>14} -> {os.path.basename(files[i])[:-4]:<14} {dx:7.2f}  {dy:7.2f}  (conf {peak:.2f})")

# summary: real-frame steps and how evenly the generated frame splits them
reals = [i for i, f in enumerate(files) if f.endswith("real.bmp")]
steps, splits = [], []
for a, b in zip(reals, reals[1:]):
    h, w = imgs[a].shape
    sl = (slice(h//6, h*5//6), slice(w//8, w*7//8))
    total = shift(imgs[b][sl], imgs[a][sl])[0]
    first = shift(imgs[b-1][sl], imgs[a][sl])[0]
    steps.append(total)
    if abs(total) > 0.5: splits.append(first / total)
if steps:
    s = np.array(steps)
    print(f"\nreal frame steps: mean {s.mean():.2f} px, min {s.min():.2f}, max {s.max():.2f}, "
          f"variation {100*s.std()/max(abs(s.mean()),1e-9):.0f}%")
if splits:
    p = np.array(splits)
    print(f"generated frame position within its step: mean {p.mean():.3f} (ideal 0.5 at 2x), "
          f"min {p.min():.3f}, max {p.max():.3f}")
