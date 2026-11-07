#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Build a fake Lossless.dll for testing lsfg-metal without the real Lossless Scaling.

Every shader resource the backend loads is replaced by a synthetic GLSL compute shader with
the exact same resource signature (descriptor bindings, counts, dispatch tile size):

  * intermediate shaders write 0.5 into every texel of every storage image (bounds-checked)
    and touch every bound input, so missing bindings / bad dispatch extents show up
  * the generate shader writes mix(prev, curr, timestamp) + 10 * |flow_input - 0.5|, so the
    output is predictable and any broken stage of the chain becomes visible as an error

The shaders are compiled with glslangValidator and packed into a PE32+ DLL with RT_RCDATA
resources at the IDs lsfg-vk expects (49 + id [+23 performance] [+49 fp32]).

usage: build_fake_dll.py OUTPUT.dll [--keep DIR]
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile

# (id, sampled, storage, buffers, samplers) - from lsfg-vk's shader_registry.cpp
COMMON = [(255, 1, 7, 1, 1), (256, 5, 1, 1, 2)]
QUALITY = [
    (267, 1, 2, 0, 1), (268, 2, 2, 0, 1), (269, 2, 4, 0, 1), (270, 4, 4, 0, 1),
    (275, 12, 2, 0, 1), (276, 2, 2, 0, 1), (277, 2, 2, 0, 1), (278, 2, 2, 0, 1), (279, 2, 6, 1, 1),
    (257, 9, 3, 1, 2), (259, 3, 4, 0, 1), (260, 4, 4, 0, 1), (261, 4, 4, 0, 1), (262, 6, 1, 1, 2),
    (263, 3, 4, 0, 1), (264, 4, 4, 0, 1), (265, 4, 4, 0, 1), (266, 6, 1, 1, 2),
    (258, 10, 2, 1, 2), (271, 2, 2, 0, 1), (272, 2, 2, 0, 1), (273, 2, 2, 0, 1), (274, 3, 1, 1, 2),
]
PERF = [
    (267, 1, 1, 0, 1), (268, 1, 1, 0, 1), (269, 1, 2, 0, 1), (270, 2, 2, 0, 1),
    (275, 6, 2, 0, 1), (276, 2, 2, 0, 1), (277, 2, 2, 0, 1), (278, 2, 2, 0, 1), (279, 2, 6, 1, 1),
    (257, 5, 3, 1, 2), (259, 3, 2, 0, 1), (260, 2, 2, 0, 1), (261, 2, 2, 0, 1), (262, 4, 1, 1, 2),
    (263, 3, 2, 0, 1), (264, 2, 2, 0, 1), (265, 2, 2, 0, 1), (266, 4, 1, 1, 2),
    (258, 6, 1, 1, 2), (271, 1, 1, 0, 1), (272, 1, 1, 0, 1), (273, 1, 1, 0, 1), (274, 2, 1, 1, 2),
]

UBO = """
layout(std140, binding = 0) uniform CB {
    uvec2 inputOffset;
    uint firstIter;
    uint firstIterS;
    uint advancedColorKind;
    uint hdrSupport;
    float resolutionInvScale;
    float timestamp;
    float uiThreshold;
} cb;
"""


def tile_info(sid):
    """dispatch tile size (pixels per workgroup) and whether storage image j is mip level j"""
    if sid == 255:
        return 64, True   # mipmaps: add_shift_extent(flow, 63, 6), 7 mip levels
    if sid == 279:
        return 32, True   # beta1 last pass: add_shift_extent(extent, 31, 5), 6 levels
    return 8, False       # everything else: add_shift_extent(extent, 7, 3)


def header(fp16, sampled, storage, buffers, samplers):
    src = "#version 450\n"
    if fp16:
        src += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
    src += "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
    if buffers:
        src += UBO
    for i in range(samplers):
        src += f"layout(binding = {16 + i}) uniform sampler samp{i};\n"
    for i in range(sampled):
        src += f"layout(binding = {32 + i}) uniform texture2D tex{i};\n"
    for i in range(storage):
        # no format qualifier: like the real shaders (StorageImageWriteWithoutFormat)
        src += f"layout(binding = {48 + i}) uniform writeonly image2D img{i};\n"
    return src


def generic_shader(sid, fp16, sampled, storage, buffers, samplers):
    tile, mips = tile_info(sid)
    src = header(fp16, sampled, storage, buffers, samplers)
    ftype = "float16_t" if fp16 else "float"
    src += "void main() {\n"
    src += f"    {ftype} dep = {ftype}(0.0);\n"
    src += "    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + 0.5) / 512.0;\n"
    for i in range(sampled):
        s = i % samplers
        src += f"    dep += {ftype}(textureLod(sampler2D(tex{i}, samp{s}), uv, 0.0).x > 6e4);\n"
    if buffers:
        src += f"    dep += {ftype}(cb.timestamp > 6e4);\n"
    for j in range(storage):
        level = j if mips else 0
        t = max(tile >> level, 1)
        src += f"""    {{
        ivec2 size = imageSize(img{j});
        ivec2 origin = ivec2(gl_WorkGroupID.xy) * {t};
        for (int y = int(gl_LocalInvocationID.y); y < {t}; y += 8)
            for (int x = int(gl_LocalInvocationID.x); x < {t}; x += 8) {{
                ivec2 p = origin + ivec2(x, y);
                if (all(lessThan(p, size)))
                    imageStore(img{j}, p, vec4({ftype}(0.5) + dep));
            }}
    }}
"""
    src += "}\n"
    return src


def generate_shader(fp16):
    # bindings: 32 prev, 33 curr, 34..36 flow inputs, 48 output, 16/17 samplers, 0 ubo
    src = header(fp16, 5, 1, 1, 2)
    ftype = "float16_t" if fp16 else "float"
    src += f"""void main() {{
    ivec2 size = imageSize(img0);
    ivec2 origin = ivec2(gl_WorkGroupID.xy) * 16;
    for (int y = int(gl_LocalInvocationID.y); y < 16; y += 8)
        for (int x = int(gl_LocalInvocationID.x); x < 16; x += 8) {{
            ivec2 p = origin + ivec2(x, y);
            if (!all(lessThan(p, size)))
                continue;
            vec4 prev = texelFetch(sampler2D(tex0, samp0), p, 0);
            vec4 curr = texelFetch(sampler2D(tex1, samp0), p, 0);
            {ftype} err = {ftype}(0.0);
"""
    for k in (2, 3, 4):
        src += f"""            {{
                ivec2 s = textureSize(sampler2D(tex{k}, samp1), 0);
                ivec2 q = min(p * s / size, s - 1);
                err += abs({ftype}(texelFetch(sampler2D(tex{k}, samp1), q, 0).r) - {ftype}(0.5));
            }}
"""
    src += """            imageStore(img0, p, mix(prev, curr, cb.timestamp) + vec4(float(err) * 10.0));
        }
}
"""
    return src


def compile_glsl(src, path):
    with open(path + ".comp", "w") as f:
        f.write(src)
    subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.1", "-S", "comp",
                    "-o", path + ".spv", path + ".comp"], check=True, stdout=subprocess.DEVNULL)
    with open(path + ".spv", "rb") as f:
        return f.read()


def build_pe(resources):
    """build a minimal PE32+ DLL whose .rsrc section holds RT_RCDATA resources"""
    SECTION_RVA = 0x1000
    FILE_ALIGN = 0x200

    ids = sorted(resources)
    dir_size = lambda n: 16 + 8 * n

    # layout: root dir (3 types) | RCDATA dir | dummy dir | language dirs | data entries | data
    root_off = 0
    rcdata_off = root_off + dir_size(3)
    dummy_off = rcdata_off + dir_size(len(ids))
    lang_off = dummy_off + dir_size(0)
    entry_off = lang_off + dir_size(1) * len(ids)
    data_off = entry_off + 16 * len(ids)

    body = bytearray()
    blobs = []
    cur = data_off
    for rid in ids:
        cur = (cur + 7) & ~7
        blobs.append((cur, resources[rid]))
        cur += len(resources[rid])
    total = (cur + 7) & ~7
    body = bytearray(total)

    def directory(off, entries):
        struct.pack_into("<IIHHHH", body, off, 0, 0, 0, 0, 0, len(entries))
        for i, (eid, target) in enumerate(entries):
            struct.pack_into("<II", body, off + 16 + 8 * i, eid, target)

    # the reader requires at least 3 type entries; only RT_RCDATA (10) is used
    directory(root_off, [(3, dummy_off | 0x80000000), (10, rcdata_off | 0x80000000),
                         (16, dummy_off | 0x80000000)])
    directory(rcdata_off, [(rid, (lang_off + dir_size(1) * i) | 0x80000000) for i, rid in enumerate(ids)])
    directory(dummy_off, [])
    for i, (rid, (blob_off, blob)) in enumerate(zip(ids, blobs)):
        directory(lang_off + dir_size(1) * i, [(0x409, entry_off + 16 * i)])
        struct.pack_into("<IIII", body, entry_off + 16 * i, SECTION_RVA + blob_off, len(blob), 0, 0)
        body[blob_off:blob_off + len(blob)] = blob

    raw_size = (len(body) + FILE_ALIGN - 1) & ~(FILE_ALIGN - 1)
    body += bytes(raw_size - len(body))

    # headers
    dos = bytearray(0x40)
    struct.pack_into("<H", dos, 0, 0x5A4D)
    struct.pack_into("<i", dos, 0x3C, 0x40)

    coff = struct.pack("<4sHHIIIHH", b"PE\0\0", 0x8664, 1, 0, 0, 0, 240, 0x2022)

    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)                       # PE32+
    struct.pack_into("<I", opt, 56, SECTION_RVA + raw_size)     # SizeOfImage
    struct.pack_into("<I", opt, 60, FILE_ALIGN)                 # SizeOfHeaders
    struct.pack_into("<I", opt, 108, 16)                        # NumberOfRvaAndSizes
    struct.pack_into("<II", opt, 112 + 2 * 8, SECTION_RVA, len(body))  # resource directory

    section = struct.pack("<8sIIIIIIHHI", b".rsrc\0\0\0", len(body), SECTION_RVA, raw_size,
                          FILE_ALIGN, 0, 0, 0, 0, 0x40000040)

    headers = dos + coff + opt + section
    headers += bytes(FILE_ALIGN - len(headers))
    return bytes(headers) + bytes(body)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = sys.argv[1]
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    if not shutil.which("glslangValidator"):
        print("glslangValidator not found (brew install glslang)")
        return 1

    work = keep or tempfile.mkdtemp(prefix="lsfg-fake-dll-")
    os.makedirs(work, exist_ok=True)
    resources = {}
    try:
        for fp16 in (True, False):
            for perf, shaders in ((False, COMMON + QUALITY), (True, PERF)):
                for (sid, p1, p2, p3, p4) in shaders:
                    rid = 49 + sid + (23 if perf else 0) + (0 if fp16 else 49)
                    if rid in resources:
                        continue
                    src = generate_shader(fp16) if sid == 256 else \
                        generic_shader(sid, fp16, p1, p2, p3, p4)
                    name = f"{'perf' if perf else 'quality'}_{sid}_{'fp16' if fp16 else 'fp32'}"
                    resources[rid] = compile_glsl(src, os.path.join(work, name))
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)

    with open(out, "wb") as f:
        f.write(build_pe(resources))
    print(f"wrote {out}: {len(resources)} shader resources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
