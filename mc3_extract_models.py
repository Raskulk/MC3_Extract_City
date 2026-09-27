#!/usr/bin/env python3
"""
mc3_extract_models.py -- geometry, texture and lighting extraction for
Midnight Club 3: DUB Edition (PS2) from .pck files. Output for Blender:
import_atlanta_city_fast.py places everything. Tested on all four cities
(Atlanta, Detroit, San Diego, Tokyo). Full format specification: MC3_FORMATS.md.

    python3 mc3_extract_models.py <file.pck> [output_folder] [options]

The mode is detected automatically from the file (resource type at +4).

CITY (<city>_<time>_<weather>.pck, type 0x43)
    python3 mc3_extract_models.py atlanta_midnight_clear.pck atlanta_city --props --fog
  Next to it: the .ppf of the same weather (texture pages) or --ppf path.
  Output:
    obj/<type>.obj, obj/<type>__vN.obj -- instances in LOCAL coordinates
        (__vN variants: different sector textures or a different baked
        lighting set -- each placement has its own lighting)
    obj/block_<name>.obj -- blocks, WORLD coordinates
    obj/block_sky.obj -- sky dome (scaled over the city)
    obj/city.mtl + textures/ -- only the textures that are used
    instance_placements.csv -- name, obj, sector, position, 3 axes
        (world = local @ [a;b;c] + position; axes may be non-uniformly scaled)
    scene.json -- sky clear colour (mcSkyHatClass m_clearColor)
    props/   -- with --props: props in the same format
    fog/     -- with --fog: low particle fog (<city>_<time>_<weather>_fog.pck)
    physics/ -- with --physics: collision mesh (<city>_bnd.pck)

  Vertex colours ("v x y z r g b") are the game's baked lighting: palette
  colour (mcCity+44, 256 float RGB) by per-vertex index -- from the model's
  CPV set (the set number of each placement is at instance +0x1c), or from
  the w component of V4-8 / STROW.w of V3-8 blocks (hdr, reflect, props).
  1.0 = texture unchanged (verified in the VU1 microcode, no x2).

  Materials follow the shader templates (shaderlib/city/*.shadert):
    _mask + _win  window shader: wall (alpha = glass) + self-lit interior;
    _alpha        foliage/fences cutout;   _hdr  night glow surfaces;
    _roadN        roads with the 4x detail texture (detail_map in the MTL);
    _add          additive flares (light cards, smoke);  _decal  overlays.
  Layers: main, ground (roads/terrain), reflect (mirrored copy below the
  ground for wet-road reflections -- skipped, --with-reflect keeps it),
  hdr, alpha.

PROPS (<city>_<time>_<weather>_props.pck, type 0x40) -- ATMs, benches, lamps,
  trees, signs; same output format. Alone or with --props. Light sprites
  (category 0) become crossed quads with a soft glow texture. Props take
  the lighting palette from the city .pck of the same time of day.

FOG (<city>_<time>_<weather>_fog.pck, type 0x04) -- mcParticleFogMgr clouds;
  alone or with --fog.

PHYSICS (<city>_bnd.pck, type 0x09) -- collision geometry (walls, ground),
  WORLD coordinates; alone or with --physics.

OTHER (cars, pedestrians, traffic) -- embedded models and textures.

VALIDATION against the game's own bounding spheres: instances 99.3-99.9%,
blocks 99.4-100%, textures on 100% of chunks in all four cities.
Some areas (tall Tokyo buildings, fenced-off lots) have no top surfaces in
the data: the game camera stays at street level.
"""
import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

BASE, HDR = 0x6800000, 0x80
ALLOC_HDR = 144
SECTOR = 2048

# --- GS addressing tables for material textures (from the ELF SLUS_123.45) ---
BLOCK32 = [0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
           8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31]
COLUMNWORD32 = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]



# Emission tint of opaque hdr-layer surfaces. The colour now comes from
# the data (palette index, V4-8 block), so no tint by default.
# (It used to be (1, 0.82, 0.70), matched to a game screenshot.)
HDR_SURFACE_TINT = (1.0, 1.0, 1.0)

# Scale of palette vertex colours (baked lighting). Verified in the VU1
# microcode of indexed colour (rmcSetCpvMode -> 0x5A97B0, entry 6):
# colour = min(lighting, 1) * palette * C42, where C42 = city colour * 255 *
# 128/255 <= 128 (rmcState, mode 0), and 128 on the GS means 'unchanged'.
# So a palette value of 1.0 = texture as is, no doubling. (This used to be
# an assumed x2 -- the city came out twice as bright as in the game.)
CPV_SCALE = 1.0

# V4-8 block of vertices without their own CPV sets (hdr, reflections, some
# foliage, props): xyz -- normal, w -- INDEX into the city lighting palette
# (verified in the microcode: MTIR VI11, VF21.w -> palette read). Vertex colour
# = palette[w] (x CPV_SCALE).

def unswizzle8(buf, w, h):
    sw = np.frombuffer(buf[:w * h], dtype=np.uint8)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    bl = (y & ~0xf) * w + (x & ~0xf) * 2
    ss = (((y + 2) >> 2) & 1) * 4
    py = (((y & ~3) >> 1) + (y & 1)) & 7
    cl = py * w * 2 + ((x + ss) & 7) * 4
    bn = ((y >> 1) & 1) + ((x >> 2) & 2)
    return sw[np.clip(bl + cl + bn, 0, len(sw) - 1)].reshape(h, w)


def csm1(pal):
    if len(pal) != 256:
        return pal
    out = pal.copy()
    for i in range(256):
        out[(i & ~0x18) | ((i >> 1) & 8) | ((i << 1) & 0x10)] = pal[i]
    return out


def fo(p):
    return p - BASE + HDR


def ok(p, filelen):
    # a pointer is an address inside the loaded file (there used to be a
    # hard limit of 0x7000000, i.e. 8 MB of data: everything beyond it
    # in Tokyo (10.2 MB) was silently dropped)
    return BASE <= p and 0 <= fo(p) < filelen


# ---------------------------------------------------------------------------
# Geometry: finding all VIF packets with vertices/UVs by scanning the file

def collect_packets(data, max_qw=2048):
    """Descriptors [pointer][u16 qw]. max_qw=2048 finds individual
    VIF sub-packets (vertices/UVs); finding LARGE geometry containers of
    named models (up to ~65000 quadwords) needs max_qw=65536 --
    they have a different typical size; not a bug but different levels
    of the structure (a container versus its inner sub-packets)."""
    n = len(data)
    seen = set()
    packets = []
    for off in range(0, n - 8, 4):
        p = struct.unpack_from('<I', data, off)[0]
        if not ok(p, n):
            continue
        t = fo(p)
        if t in seen:
            continue
        qw = struct.unpack_from('<H', data, off + 4)[0]
        if not (1 <= qw <= max_qw and t + qw * 16 <= n):
            continue
        seen.add(t)
        packets.append((t, qw))
    return packets


def parse_packet_geometry(data, t, qw):
    """Returns (vertices Nx3 int16 or None, UV Mx2 int16 or None).

    Vertices -- UNPACK V3-16 blocks. Texture coordinates -- a V2-16
    OR V4-16 block (the first two components are taken) following the
    vertex block with the same element count. About 10% of City packets
    (1020 of ~10300) store UVs in V4-16 (components 3 and 4 are the
    second UV set of the window shader); without them the strip restart
    flag (low bit of u) was lost as well -- strips were joined with
    seams (rays on fragments like a_mt_blk_fox_01x#geom5). The flag in
    V4-16 works the same way: on vertices 0 and 1 in all 1020 packets, runs of 2.
    Commands with data are skipped together with their data: STMASK (0x20) --
    one word, STROW/STCOL (0x30/0x31) -- four words; their data used to be
    read as commands, and packet parsing stopped.
    ALL vertex blocks of a packet are collected, not only the last one."""
    n = len(data)
    end = t + qw * 16
    pos = t
    vparts, uparts = [], []
    pending = None          # vertex count of the last V3-16 block without UVs
    while pos < end - 4:
        w = struct.unpack_from('<I', data, pos)[0]
        c = (w >> 24) & 0x7F
        num = (w >> 16) & 0xFF
        if c & 0x60 == 0x60:
            vn = (c >> 2) & 3
            vl = c & 3
            bits = {0: 32, 1: 16, 2: 8, 3: 5}[vl]
            comps = vn + 1
            size = ((comps * bits * num + 31) // 32) * 4 if vl != 3 else ((16 * num + 31) // 32) * 4
            if vn == 2 and vl == 1 and num >= 3 and pos + 4 + num * 6 <= n:
                vparts.append(np.frombuffer(data[pos + 4:pos + 4 + num * 6], dtype='<i2').reshape(-1, 3))
                pending = num
            elif vl == 1 and vn in (1, 3) and num == pending and pos + 4 + num * 2 * comps <= n:
                arr = np.frombuffer(data[pos + 4:pos + 4 + num * 2 * comps], dtype='<i2').reshape(-1, comps)
                uparts.append(arr[:, :2])
                pending = None
            pos += 4 + size
        elif c == 0x20:                       # STMASK + 1 word
            pos += 8
        elif c in (0x30, 0x31):               # STROW / STCOL + 4 words
            pos += 20
        elif c in (0, 1, 2, 3, 4, 5, 0x10, 0x11, 0x14, 0x15, 0x17, 0x32, 0x33):
            pos += 4
        else:
            break
    verts = np.concatenate(vparts) if vparts else None
    uv = np.concatenate(uparts) if uparts else None
    return verts, uv


def packet_header_scale(data, t):
    """Vertex dequantisation multiplier of a City packet -- stored in the data.

    Every VIF packet of City geometry starts with UNPACK V1-32 num=2:
      word 0 -- float, the multiplier the VU1 microcode applies to
                int16 coordinates (always a power of two: 2^-8, 2^-9, ...);
      word 1 -- vertex count of the packet.
    Verified on Atlanta: the header is present in 10348 of 10348 packets,
    the vertex count matches the V3-16 command in 99.9%; the multiplier
    matches the scale derived from the world bounding spheres of
    placements (the game's reference) for 228 of 231 types, and in one of the
    three mismatches (a_inst_awning_02x) a check by sphere centre showed
    that the reference fit was wrong (centre error 0.000 with
    the header multiplier versus 0.95 with the fitted one). All earlier
    scale fitting (by type radius, voting, quantisation
    rules) tried to guess this number. The scale can DIFFER
    between packets of one type, so it is applied per packet.
    Returns None if there is no header."""
    if t + 12 > len(data):
        return None
    w = struct.unpack_from('<I', data, t)[0]
    if (w >> 24) & 0x7F != 0x60 or ((w >> 16) & 0xFF) != 2:
        return None
    f = struct.unpack_from('<f', data, t + 4)[0]
    return f if 0 < f < 1 else None


def parse_packet_batches(data, t, qw):
    """Parses a City/props VIF packet into BATCHES: [(float vertices, int UV|None, palette index|None), ...].

    A batch starts with an UNPACK V1-32 num=2 header [multiplier, vertex
    count]; a region may hold several in a row, each with its own
    multiplier (e.g. the glowing hdr panels on the top of the
    a_mt_blk_bellsouth_01x#geom2 tower). Vertices -- V3-16, multiplied
    by the batch multiplier. UV -- the first UNPACK after the vertices with the same
    element count:
      V2-16 or V4-16 (first two components) -- as is;
      V2-8 in STMOD=1 mode -- start value from STROW + byte;
      V2-8 in STMOD=2 mode -- deltas: each value = previous +
          byte, starting from STROW (example: STROW 2357, 2652 -- odd u,
          i.e. the restart flag on the first vertex, as expected).
    The +2 slot: V4-8 (normal + palette index in w) or masked V3-8
    (normal + one palette index from STROW.w). Commands with data are
    skipped together with their data: STMASK -- 1 word, STROW/STCOL -- 4 words."""
    n = len(data)
    end = t + qw * 16
    pos = t
    batches = []
    scale = None
    row = [0, 0, 0, 0]
    stmask = 0
    mode = 0
    cur = None               # current batch: [vertices, uv, palette index]
    while pos < end - 4:
        w = struct.unpack_from('<I', data, pos)[0]
        c = (w >> 24) & 0x7F
        num = (w >> 16) & 0xFF
        if c & 0x60 == 0x60:
            vn = (c >> 2) & 3
            vl = c & 3
            masked = (c >> 4) & 1
            usn = (w >> 14) & 1
            bits = {0: 32, 1: 16, 2: 8, 3: 5}[vl]
            comps = vn + 1
            size = ((comps * bits * num + 31) // 32) * 4 if vl != 3 else ((16 * num + 31) // 32) * 4
            body = pos + 4
            if pos + 4 + size > n:
                break
            if vn == 0 and vl == 0 and num == 2 and not masked:
                f = struct.unpack_from('<f', data, body)[0]
                scale = f if 0 < f < 1 else None
                cur = None
            elif masked:
                if vn == 2 and vl == 2 and stmask == 0x40404040 and cur is not None \
                        and num == len(cur[0]) and cur[2] is None:
                    # V3-8 with mask 0x40404040: x,y,z -- normal from the data, w --
                    # from STROW: ONE palette index for all
                    # vertices of the block (signed; -1 = index 255)
                    cur[2] = np.full(num, float(row[3] & 0xFF))
            elif vn == 3 and vl == 2 and cur is not None and num == len(cur[0]) and cur[2] is None:
                # V4-8: x,y,z -- normal (int8, length ~127), w -- palette INDEX
                # (microcode: MTIR VI11, VF21.w -> palette)
                cur[2] = np.frombuffer(data[body:body + num * 4], dtype=np.uint8).reshape(-1, 4)[:, 3].astype(np.float64)
            elif vn == 2 and vl == 1 and num >= 3:
                v = np.frombuffer(data[body:body + num * 6], dtype='<i2').reshape(-1, 3).astype(np.float64)
                if mode == 1:
                    v = v + np.array(row[:3], dtype=np.float64)
                elif mode == 2:
                    v = np.cumsum(v, axis=0) + np.array(row[:3], dtype=np.float64)
                if scale is not None:
                    v = v * scale
                cur = [v, None, None]       # vertices, UV, palette index (V4-8)
                batches.append(cur)
            elif cur is not None and cur[1] is None and num == len(cur[0]):
                uv = None
                if vl == 1 and vn in (1, 3):
                    raw16 = np.frombuffer(data[body:body + num * 2 * comps],
                                          dtype='<i2').reshape(-1, comps).astype(np.int64)
                    # V4-16: the second pair -- UVs of the second pass (windows: the
                    # interior behind the glass, shader type 16 mcShaderCityWindow)
                    uv = raw16[:, :4] if comps == 4 else raw16[:, :2]
                elif vl == 2 and vn == 1:
                    raw = np.frombuffer(data[body:body + num * 2], dtype='<u1' if usn else '<i1').reshape(-1, 2).astype(np.int64)
                    if mode == 1:
                        uv = raw + np.array(row[:2])
                    elif mode == 2:
                        uv = np.cumsum(raw, axis=0) + np.array(row[:2])
                    else:
                        uv = raw
                if uv is not None:
                    if uv.shape[1] == 2 and comps == 4 and vl == 1:
                        pass
                    cur[1] = uv
            pos += 4 + size
        elif c == 0x20:
            if pos + 8 <= n:
                stmask = struct.unpack_from('<I', data, pos + 4)[0]
            pos += 8
        elif c in (0x30, 0x31):
            if c == 0x30 and pos + 20 <= n:
                row = list(struct.unpack_from('<4i', data, pos + 4))
            pos += 20
        elif c == 0x05:
            mode = w & 3
            pos += 4
        elif c in (0, 1, 2, 3, 4, 0x10, 0x11, 0x14, 0x15, 0x17, 0x32, 0x33):
            pos += 4
        else:
            break
    return batches


def extract_region_mesh(data, region_start, region_end, packets_in_region,
                        header_scale=False):
    """Collects vertices+UVs+faces (naive triangle strips, WITHOUT seam
    filtering) from a region. Filtering is a separate step, see
    filter_strip_artifacts: the threshold must be shared by the WHOLE object, not
    per packet (checked: the same road object has
    packets with different natural edge scales -- a local
    per-packet threshold either misses real seams where the packet is
    large, or cuts legitimate geometry where the packet is small)."""
    verts_all, uv_all, faces = [], [], []
    voff = 0
    region_scale = packet_header_scale(data, region_start) if header_scale else None
    if header_scale:
        # City and props: parse by batches (each has its own multiplier, UVs and flags)
        for t, qw in packets_in_region:
            for v, uv, lt in parse_packet_batches(data, t, qw):
                nv = len(v)
                if nv < 3:
                    continue
                verts_all.append(v)
                adc = None
                if uv is not None and len(uv) == nv:
                    uvi = uv.astype(np.int64).copy()
                    adc = (uvi[:, 0] & 1).astype(bool)
                    uvi[:, 0] &= ~1
                    uvf = uvi.astype(np.float64) / 4096.0
                    if uvf.shape[1] < 4:
                        uvf = np.hstack([uvf, np.zeros((nv, 4 - uvf.shape[1]))])
                else:
                    uvf = np.zeros((nv, 4))
                # 5th column -- palette index from V4-8 (-1 if absent)
                ltc = lt.reshape(-1, 1) if lt is not None and len(lt) == nv else np.full((nv, 1), -1.0)
                uv_all.append(np.hstack([uvf[:, :4], ltc]))
                for j in range(2, nv):
                    if adc is not None and adc[j]:
                        continue
                    # winding parity -- from the start of the batch (for single packets
                    # this equals the start of the packet: see the comment below)
                    if j % 2 == 0:
                        a, b, c = j - 2, j - 1, j
                    else:
                        a, b, c = j - 1, j - 2, j
                    faces.append((voff + a, voff + b, voff + c))
                voff += nv
        if not verts_all:
            return None, None, None
        return np.concatenate(verts_all), np.concatenate(uv_all), faces
    for t, qw in packets_in_region:
        v, uv = parse_packet_geometry(data, t, t + qw * 16 - t if False else qw)
        if v is None or len(v) < 3:
            continue
        nv = len(v)
        vf = v.astype(np.float64)
        if header_scale:
            # the sub-packet's own header, otherwise -- the header of the region (packet
            # from the rmcGeometry list) it belongs to: some sub-packets
            # found by searching inside a region do not start with a header,
            # and without this their vertices stayed raw int16 (checked:
            # 8 Atlanta types came out hundreds to thousands of times larger than
            # the reference placement sphere)
            hs = packet_header_scale(data, t)
            if hs is None:
                hs = region_scale
            if hs is not None:
                vf = vf * hs
        verts_all.append(vf)
        have_uv = uv is not None and len(uv) == nv
        adc = None
        if have_uv:
            uvi = uv.astype(np.int32)
            if header_scale:
                # ADC flag (strip restart) -- low bit of u, see below
                adc = (uvi[:, 0] & 1).astype(bool)
                uvi = uvi.copy()
                uvi[:, 0] &= ~1
            uv_all.append(np.hstack([uvi.astype(np.float64) / 4096.0, np.zeros((nv, 2))]))
        else:
            uv_all.append(np.zeros((nv, 4)))
        if adc is not None:
            # STRIP TOPOLOGY FROM THE DATA. The low bit of u of each City vertex is
            # the ADC flag: a triangle ending on a flagged vertex is NOT
            # drawn. Verified on all Atlanta packets: the flag on vertices 0 and
            # 1 of a packet -- 99.9%, on vertex 2 -- 0.0%; flags come strictly
            # in pairs (all 50842 runs have length exactly 2) -- the start of each
            # new strip; the two low bits of v and bit 1 of u are always 0,
            # i.e. bit 0 of u is not a coordinate. Among flag-forbidden
            # triangles 38% are long seams, among allowed ones 6% (roofs,
            # panels). Holes in the top view of the city: 3.7% -> 1.3%.
            # WINDING ALTERNATES FROM THE START OF THE PACKET (j % 2), NOT from the
            # start of the strip: checked on shared edges at joints of neighbouring strips
            # (27219 joints) -- parity from the packet start is consistent in
            # 97.0%, from the strip start only in 77.2%. The ADC flag cuts the seam
            # but does not reset parity: the microcode alternates it through the packet.
            for j in range(2, nv):
                if adc[j]:
                    continue
                if j % 2 == 0:
                    a, b, c = j - 2, j - 1, j
                else:
                    a, b, c = j - 1, j - 2, j
                faces.append((voff + a, voff + b, voff + c))
        else:
            for i in range(nv - 2):
                a, b, c = (i, i + 1, i + 2) if i % 2 == 0 else (i + 1, i, i + 2)
                faces.append((voff + a, voff + b, voff + c))
        voff += nv
    if not verts_all:
        return None, None, None
    return np.concatenate(verts_all), np.concatenate(uv_all), faces


def filter_strip_artifacts(verts, faces, thr=None, uvs=None):
    """Drops seam faces at joints of triangle strips.

    Two criteria:
      1) a degenerate, almost collinear triangle (area less than
         0.006 * longest_side^2) -- this is how the PS2 encodes
         a strip restart;
      2) an abnormally long face (longest side > thr) AND AT THE SAME TIME
         a drop in texture density: log2(UV_area / 3D_area) below the
         chunk median by more than 3 (8+ times).

    Why not just by length (as in earlier versions): checked on
    all of Atlanta -- the length filter threw away 3917 full faces,
    1619 of them horizontal (roofs, road panels), a median of 9.7%
    of the object area, up to 63% for every tenth; the top view of the city
    showed 13.6% 'holes' inside the outline versus 5.3% without the filter.
    A real roof is two large triangles with NORMAL
    texture density (the texture lies as on neighbouring
    faces), while a seam joins vertices of two different strips: the 3D area
    is huge, the UV area normal -- density drops. Among the faces
    the old filter cut, 32% have a drop below -3 (seams), and 29%
    -- normal density (roofs/panels, now kept); among
    ordinary faces a drop below -3 occurs in only 6%.
    Each triangle is checked independently (the rule 'the one after
    a seam is bad too' was tested and rejected earlier -- only holes)."""
    if not faces:
        return faces
    F = np.asarray(faces)
    P, Q, R = verts[F[:, 0]], verts[F[:, 1]], verts[F[:, 2]]
    longest = np.maximum(np.maximum(np.linalg.norm(P - Q, axis=1),
                                    np.linalg.norm(Q - R, axis=1)),
                         np.linalg.norm(P - R, axis=1))
    area = 0.5 * np.linalg.norm(np.cross(Q - P, R - P), axis=1)
    if thr is None:
        finite = longest[np.isfinite(longest) & (longest > 0)]
        thr = max(6 * np.median(finite), 1.0) if len(finite) else 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        degenerate = np.where(longest > 0, area < 0.006 * longest ** 2, True)
    seam = np.zeros(len(F), dtype=bool)
    long_ = (longest > thr) & ~degenerate
    if long_.any():
        if uvs is not None and len(uvs) == len(verts):
            U = np.asarray(uvs, dtype=np.float64)
            d1 = U[F[:, 1]] - U[F[:, 0]]
            d2 = U[F[:, 2]] - U[F[:, 0]]
            auv = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
            valid = (area > 0) & (auv > 0) & ~degenerate
            if valid.sum() >= 3:
                dens = np.full(len(F), np.nan)
                dens[valid] = np.log2(auv[valid] / area[valid])
                med = np.nanmedian(dens[valid])
                # seam: a long face with a density drop (or no UV area at all)
                low = np.where(valid, dens - med < -3, auv <= 0)
                seam = long_ & low
            else:
                seam = long_
        else:
            seam = long_   # no UVs -- the old behaviour
    bad = degenerate | seam
    return [tuple(f) for f, b in zip(faces, bad) if not b]


def edge_len_threshold(verts, faces):
    """Global (6x median) edge length for filter_strip_artifacts --
    computed over all given faces at once, see its docstring."""
    if not faces:
        return 1.0
    P = verts[np.array([f[0] for f in faces])]
    Q = verts[np.array([f[1] for f in faces])]
    R = verts[np.array([f[2] for f in faces])]
    e = np.concatenate([np.linalg.norm(P - Q, axis=1), np.linalg.norm(Q - R, axis=1),
                        np.linalg.norm(P - R, axis=1)])
    e = e[np.isfinite(e) & (e > 0)]
    return max(6 * np.median(e), 1.0) if len(e) else 1.0


def index_packets_by_range(packets):
    """Builds a start->(qw) dict to quickly check 'is there a packet at address t'."""
    return {t: qw for t, qw in packets}


# ---------------------------------------------------------------------------
# Mode 1: embedded models (cars/pedestrians/traffic) -- with materials

def log2sh(v):
    n = 0
    v >>= 1
    while v:
        v >>= 1
        n += 1
    return n


def find_embedded_textures(data):
    """TEX0 entries addressed by ordinary pointers (not page references)."""
    n = len(data)
    tex = []
    for off in range(8, n - 8, 4):
        lo = struct.unpack_from('<I', data, off)[0]
        psm = (lo >> 20) & 0x3F
        if psm not in (19, 20):
            continue
        tbw, tw = (lo >> 14) & 0x3F, (lo >> 26) & 0xF
        if tbw == 0 or tbw > 16 or tw < 3 or tw > 10:
            continue
        hi = struct.unpack_from('<I', data, off + 4)[0]
        th = ((lo >> 30) & 0x3) | ((hi & 0x3) << 2)
        if th < 3 or th > 10:
            continue
        w, h = 1 << tw, 1 << th
        if tbw != max(1, w // 64):
            continue
        ptr = None
        for dd in range(8, 72, 4):
            v = struct.unpack_from('<I', data, off + dd)[0]
            if ok(v, n):
                ptr = v
                break
        if ptr is None:
            continue
        tex.append({'off': off, 'psm': psm, 'w': w, 'h': h, 'data_off': fo(ptr) + ALLOC_HDR})
    return tex


def decode_embedded_texture(data, tinfo, outdir, name):
    n = len(data)
    w, h, psm = tinfo['w'], tinfo['h'], tinfo['psm']
    a = tinfo['data_off']
    size = w * h if psm == 19 else w * h // 2
    if a + size > n or psm != 19:
        return None
    idx = unswizzle8(data[a:a + size], w, h)[::-1]
    palsz = 1024
    for shift in (size, size + ALLOC_HDR):
        pa = a + shift
        if pa + palsz <= n and max(data[pa + 3:pa + palsz:4]) <= 0x80:
            pal = csm1(np.frombuffer(data[pa:pa + palsz], dtype=np.uint8).reshape(256, 4).copy())
            pal[:, 3] = np.clip(pal[:, 3].astype(np.int16) * 2, 0, 255)
            c = pal[idx]
            # RGBA, not BGRA -- see mc3_extract_textures.py.
            # Embedded textures (addressed by pointers, not .ppf pages)
            # use RGBA order; checked on the STOP sign and pedestrian faces.
            rgba = np.dstack([c[..., 0], c[..., 1], c[..., 2], c[..., 3]]).astype(np.uint8)
            if Image:
                fn = f"{name}.png"
                Image.fromarray(rgba, 'RGBA').save(os.path.join(outdir, fn))
                return fn
    return None


def normalize_asset_name(s):
    """Normalises a resource name for matching a model and its texture.

    Checked on three real files -- the naming schemes DIFFER:

    1) props (atlanta_midnight_clear_props.pck): the same item
       is written two ways -- texture 'a_prop_atm01', object
       'a_prop_atm_01x'. Normalising '_01x' -> '01' links them.
       Result: 32 one-to-one matches.

    2) traffic (atlanta_traffic.pck): textures have an extension --
       'va_citybus_a.tex', the object 'va_citybus_a'. Dropping
       '.tex' links them. Result: 10 of 11 textures.

    3) pedestrians (atlanta_peds.pck): a ONE-TO-MANY link -- the model
       'ATLfped01', and several texture variants ('ATLfped01a1',
       'ATLfped01a2', 'ATLfped01b1'... -- apparently clothing/appearance
       variants). Exact name matching does not work here,
       prefix matching is needed (see match_by_prefix)."""
    s = s.lower()
    s = re.sub(r'\.tex$', '', s)
    s = re.sub(r'_(\d+)([a-z]*)$', r'\1', s)
    s = re.sub(r'x$', '', s)
    return s


def collect_asset_names(data):
    """All resource names that occur in the file as ASCII strings.

    A wide search on purpose: names are stored differently (some
    zero-terminated, some not), and a strict filter "only between
    zero bytes" loses most real names (checked:
    props 32 -> 22 matches, pedestrians 32 -> 0)."""
    out = set()
    for m in re.finditer(rb'[A-Za-z][A-Za-z0-9_]{4,45}(?:\.tex)?', data):
        out.add(m.group().decode('ascii', 'replace'))
    return out


def match_by_prefix(obj_names, tex_names, min_len=6):
    """ONE-TO-MANY link: the model 'ATLfped01' and its texture variants
    'ATLfped01a1', 'ATLfped01a2', 'ATLfped01b1'...  (clothing and
    appearance variants of pedestrians). Exact matching fails here,
    so a texture is attached to the LONGEST model whose name
    is its prefix.

    Truncated names are dropped: the file contains cut strings
    ('va_civic_sh' instead of 'va_civic_share'), and without this filter
    a texture would go to the fragment instead of the full name."""
    truncated = {o for o in obj_names
                 if any(p != o and p.startswith(o) for p in obj_names)}
    objs = sorted((o for o in obj_names
                   if len(o) >= min_len and o not in truncated),
                  key=len, reverse=True)
    out = {}
    for t in tex_names:
        base = re.sub(r'\.tex$', '', t, flags=re.I)
        for o in objs:
            if len(base) > len(o) and base.lower().startswith(o.lower()):
                out.setdefault(o, []).append(t)
                break
    return out


def base_object_name(name):
    """'a_prop_mailbox_01x_breakpart01' -> 'a_prop_mailbox_01x':
    damage/particle variants belong to the same object as the
    whole model and must not count as the boundary of a new object."""
    return re.sub(r'_(breakpart\d*|particle_\w+)$', '', name, flags=re.I)


def split_geometry_by_name_boundaries(data, packets):
    """Splits a shared geometry pool into separate objects
    by the boundaries between neighbouring names in the file.

    Found in atlanta_midnight_clear_props.pck: a simple search for
    "the nearest large geometry block next to a name" finds NOT the
    geometry of one object but a shared pool with SEVEN
    different objects in a row (mailbox, ballhoop, bench, fence, gaspump,
    newsstand + their breakpart/particle variants) -- a visual
    check with textures showed an unreadable jumble of panels.
    Limiting the region to [this_model .. next_DIFFERENT_model)
    instead of the whole block gives clean, recognisable geometry (checked:
    the mailbox silhouette became clear after this fix,
    vertices 438 -> 84 per object)."""
    packets_sorted = sorted(packets)
    pstarts = [t for t, _ in packets_sorted]

    raw_names = []
    i = 0
    while i < len(data) - 6:
        b = data[i]
        if 97 <= b <= 122 or 65 <= b <= 90:
            j = i
            while j < len(data) and 32 <= data[j] < 127:
                j += 1
            s = data[i:j]
            if len(s) >= 6 and s.startswith((b'a_', b'd_', b'fx_', b't_', b'v_')):
                raw_names.append((i, s.decode('ascii', 'replace')))
            i = j
        else:
            i += 1
    raw_names.sort()

    # group consecutive occurrences of one base name into one region
    regions = []
    cur_base = None
    cur_start = None
    for pos, nm in raw_names:
        base = base_object_name(nm)
        if base != cur_base:
            if cur_base is not None:
                regions.append((cur_start, pos, cur_base))
            cur_base = base
            cur_start = pos
    if cur_base is not None:
        regions.append((cur_start, len(data), cur_base))

    result = {}
    for start, end, base in regions:
        lo = bisect.bisect_left(pstarts, start)
        hi = bisect.bisect_left(pstarts, end)
        sub = packets_sorted[lo:hi]
        if not sub:
            continue
        verts, uvs, faces = extract_region_mesh(data, start, end, sub)
        if verts is None or len(verts) < 4:
            continue
        if base in result:
            continue  # the first occurrence (the full model, not a breakpart shard)
        result[base] = (verts, uvs, faces)
    return result


def extract_embedded_models(data, outdir):
    """Extracts geometry and textures; textures are saved UNDER THE NAMES
    of objects, and the model<->texture mapping is written to matches.txt
    (see normalize_asset_name -- the link goes through the naming
    convention, not through a pointer in the data)."""
    n = len(data)
    os.makedirs(outdir, exist_ok=True)
    packets = collect_packets(data)
    textures = find_embedded_textures(data)
    print(f"  VIF packets found: {len(packets)}, embedded textures: {len(textures)}")

    packets_sorted = sorted(packets)
    verts, uv, faces = extract_region_mesh(
        data, packets_sorted[0][0],
        packets_sorted[-1][0] + packets_sorted[-1][1] * 16, packets_sorted)
    if verts is not None:
        with open(os.path.join(outdir, 'geometry.obj'), 'w') as f:
            f.write("# all geometry of the file as ONE mesh (for reference/viewing).\n"
                     "# For geometry PER OBJECT see obj/*.obj -- it is more reliable.\n")
            for v in verts:
                f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
            for u in uv:
                f.write(f"vt {u[0]:.4f} {1 - u[1]:.4f}\n")
            for a, b, c in faces:
                f.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")
        print(f"  vertices extracted: {len(verts)} -> geometry.obj (shared; per-object below)")

    # --- geometry PER OBJECT, split by name boundaries (more reliable
    # than the shared mesh -- see split_geometry_by_name_boundaries) ---
    per_obj = split_geometry_by_name_boundaries(data, packets)
    obj_dir = os.path.join(outdir, 'obj')
    os.makedirs(obj_dir, exist_ok=True)
    for base, (v, u, f_) in per_obj.items():
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', base)
        with open(os.path.join(obj_dir, f"{safe}.obj"), 'w') as fh:
            fh.write(f"# {base} -- geometry of ONE object (boundary by neighbouring names)\n")
            for vv in v:
                fh.write(f"v {vv[0]:.2f} {vv[1]:.2f} {vv[2]:.2f}\n")
            for uu in u:
                fh.write(f"vt {uu[0]:.4f} {1 - uu[1]:.4f}\n")
            for a, b, c in f_:
                fh.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")
    print(f"  geometry per object: {len(per_obj)} -> obj/*.obj")

    # --- textures: saved under the object name, if known ---
    try:
        import mc3_extract_textures as MT
        chains = MT.collect_descriptors(data)
        tex_name_by_off = {}
        for ch in chains:
            nm = MT.find_object_name(data, ch)
            if nm:
                tex_name_by_off[ch[0]['off']] = nm
    except Exception:
        tex_name_by_off = {}

    tex_dir = os.path.join(outdir, 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    n_tex = 0
    saved_names = {}
    for i, tinfo in enumerate(textures):
        nm = tex_name_by_off.get(tinfo['off'])
        base = re.sub(r'\.tex$', '', nm, flags=re.I) if nm else None
        base = re.sub(r'[^A-Za-z0-9_.-]', '_', base) if base else f"tex_{i:03d}"
        fn = decode_embedded_texture(data, tinfo, tex_dir, base)
        if fn:
            n_tex += 1
            if nm:
                saved_names[nm] = fn
    print(f"  textures saved: {n_tex} -> textures/ ({len(saved_names)} with object names)")

    # --- model <-> texture matching by normalised name ---
    all_names = collect_asset_names(data)
    obj_names = all_names - set(tex_name_by_off.values())
    tex_by_norm = {}
    for nm in tex_name_by_off.values():
        tex_by_norm.setdefault(normalize_asset_name(nm), []).append(nm)

    matches = []
    for o in sorted(obj_names):
        k = normalize_asset_name(o)
        if k in tex_by_norm:
            for t in tex_by_norm[k]:
                matches.append((o, t, saved_names.get(t, '')))

    # In addition -- a one-to-many link by prefix (texture
    # variants of one model, e.g. pedestrians). Only what
    # the exact match above did not catch is added.
    already = {t for _, t, _ in matches}
    prefix_hits = match_by_prefix(obj_names, set(tex_name_by_off.values()))
    n_prefix = 0
    for o, tlist in sorted(prefix_hits.items()):
        for t in tlist:
            if t not in already:
                matches.append((o, t, saved_names.get(t, '')))
                n_prefix += 1

    with open(os.path.join(outdir, 'matches.txt'), 'w') as f:
        f.write("# MODEL <-> TEXTURE <-> GEOMETRY mapping\n")
        f.write("# format: object_name  texture_name  texture_file  geometry_file\n")
        for o, t, fn in matches:
            geom_fn = ''
            ob = base_object_name(o)
            if ob in per_obj:
                geom_fn = f"obj/{re.sub(r'[^A-Za-z0-9_.-]', '_', ob)}.obj"
            f.write(f"{o}\t{t}\t{fn}\t{geom_fn}\n")
    print(f"  model<->texture matches: {len(matches)} "
          f"(exact {len(matches)-n_prefix}, by prefix {n_prefix}) -> matches.txt")

    with open(os.path.join(outdir, 'NOTES.txt'), 'w') as f:
        f.write(
            "Textures are saved under OBJECT NAMES (not tex_NNN).\n"
            "matches.txt holds the model<->texture<->geometry mapping.\n"
            "obj/*.obj -- geometry of SEPARATE OBJECTS (not one shared mesh).\n\n"
            "The texture<->name link is found NOT through a pointer in the data (there\n"
            "is none -- textures and geometry live in separate sections\n"
            "of the file) but through the naming convention: the texture\n"
            "'a_prop_atm01' and the object 'a_prop_atm_01x' are one item\n"
            "written in two different naming schemes.\n\n"
            "Geometry PER OBJECT (obj/*.obj) is split by the boundaries between\n"
            "neighbouring names in the file -- CHECKED visually with a real\n"
            "textured render: a naive search for 'the nearest large\n"
            "geometry block' found not one object but a shared pool of SEVERAL\n"
            "different items in a row (example: the 'mailbox block'\n"
            "also held a basketball hoop, a bench, a fence, a gas\n"
            "pump and a newsstand) -- a render of such a block looked\n"
            "like an unreadable jumble of panels. Limiting the region by name\n"
            "boundaries gives clean, recognisable geometry of one object.\n\n"
            "REMAINING LIMITATION (checked on real renders):\n"
            "the name boundary is a REAL COMPROMISE, not an unconditional win.\n"
            "For the ATM and the mailbox the fix gave a clean,\n"
            "recognisable result. But for the bench and especially the hydrant\n"
            "(down to 12 vertices) the boundary turned out TOO NARROW -- instead\n"
            "of mixing with neighbours the silhouette is cut.\n\n"
            "ON THE HYDRANT (the real cause found, not just\n"
            "'the boundary is narrow'): its region has 14 sub-packets, but 9 of them\n"
            "start with a NOP byte (0x00) followed by real\n"
            "float data that are not VIF commands (probably bones or\n"
            "material parameters that ended up in the shared\n"
            "region). A simple VIF parser that reaches the first unrecognised\n"
            "byte STOPS COMPLETELY instead of skipping the fragment and\n"
            "continuing -- so these 9 large (768-1026 quadword)\n"
            "packets give zero vertices, and in total only what\n"
            "was found in two small packets of 181 quadwords is extracted. This is a\n"
            "separate, deeper problem of parser fragility that cannot be fixed\n"
            "by simply retuning the name boundary.\n\n"
            "Before using a particular obj/*.obj it is worth\n"
            "checking it visually rather than relying on the automation blindly.\n")
    return {'n_vertices': len(verts) if verts is not None else 0,
            'n_textures': n_tex, 'n_matches': len(matches), 'n_objects': len(per_obj)}


# Mode 2: city objects (mcHood/mcInstCityModel) and props

SLOT_NAMES = ('main', 'ground', 'reflect', 'hdr', 'alpha')


def cpv_indices(data, m, set_idx):
    """Vertex colour indices of CPV set set_idx of model m: a list per chunk (a byte
    array per vertex, the chunk's packets in a row) or None.

    A model has as many sets as its type has placements (exactly for 154
    Atlanta instance types); the placement's set number is a u16 at +0x1c
    of the instance record: for all 151 types with several sets it runs
    exactly 0..N-1. I.e. the lighting is baked for EACH placement
    separately (the same building under a street lamp and in a dark alley is
    lit differently); normal rendering passes the set number to
    rmcModelGeom::DrawCpv, which takes (+20)[number]."""
    n = len(data)
    nch, ncpv = struct.unpack_from('<HH', data, m + 8)
    if set_idx >= ncpv:
        return None
    cp = struct.unpack_from('<I', data, m + 20)[0]
    gp = struct.unpack_from('<I', data, m + 16)[0]
    if not (ok(cp, n) and ok(gp, n)):
        return None
    o = struct.unpack_from('<I', data, fo(cp) + 4 * set_idx)[0]
    if not ok(o, n):
        return None
    o = fo(o)
    out = []
    for i in range(nch):
        pp, cnt = struct.unpack_from('<IH', data, fo(gp) + 8 * i)
        _, arrp = struct.unpack_from('<II', data, o + 8 * i)
        parts = []
        if ok(pp, n) and ok(arrp, n):
            for k in range(cnt):
                p2, qw = struct.unpack_from('<IH', data, fo(pp) + 8 * k)
                cv = struct.unpack_from('<I', data, fo(arrp) + 4 * k)[0]
                if not (ok(p2, n) and ok(cv, n)):
                    parts = None
                    break
                nv = sum(len(b[0]) for b in parse_packet_batches(data, fo(p2), qw))
                parts.append(np.frombuffer(data[fo(cv):fo(cv) + nv], dtype=np.uint8))
        out.append(np.concatenate(parts).astype(np.float64) if parts else None)
    return out


def read_model_geom(data, m, small, pstarts):
    """rmcModelGeom -> [(shader_index, int16 vertices, uv, faces), ...] per chunk.

    Layout confirmed by the fixup constructor rmcModelGeom::rmcModelGeom_2:
      +8   u16  chunk count
      +10  u16  CPV set count
      +12  ptr  u16[chunks] -- shader indices (rmcModel::Draw: shader =
                *(group+4)[index], checked against *(group+8))
      +16  ptr  rmcGeometry[chunks], 8 bytes each: [ptr to packet list][u16 count]
                each packet -- [ptr][u16 quadwords] of the chunk's VIF data
      +20  ptr  rmcModelCpv[set count]
    (+12..+28 used to be misread as 'material geometry blocks' --
    overlapping windows of ~700 KB and models of ~100
    thousand vertices resulted; the real geometry of a low building is ~270 vertices.)"""
    n = len(data)
    nch = struct.unpack_from('<H', data, m + 8)[0]
    ip = struct.unpack_from('<I', data, m + 12)[0]
    gp = struct.unpack_from('<I', data, m + 16)[0]
    if not (0 < nch < 256) or not ok(ip, n) or not ok(gp, n):
        return None
    idx = struct.unpack_from(f'<{nch}H', data, fo(ip))
    # Vertex colours (baked lighting): CPV set 0 -- drawn by the normal
    # rendering of blocks (mcCityModel::Render passes set number 0).
    # +10 set count, +20 -> sets; a set: 8 bytes per chunk
    # [-> counts][-> array of per-packet pointers], per packet -- one
    # byte per vertex: an index into the 256-colour float RGBA palette
    # (rmcCpvPalette, mcCity+44; rmcCpvIndexed flag).
    ncpv = struct.unpack_from('<H', data, m + 10)[0]
    cpv0 = None
    cp = struct.unpack_from('<I', data, m + 20)[0]
    if ncpv and ok(cp, n):
        o0 = struct.unpack_from('<I', data, fo(cp))[0]
        if ok(o0, n):
            cpv0 = fo(o0)
    chunks = []
    for i in range(nch):
        p, cnt = struct.unpack_from('<IH', data, fo(gp) + 8 * i)
        V, U, F, voff = [], [], [], 0
        if ok(p, n) and 0 < cnt < 256:
            for k in range(cnt):
                p2, qw = struct.unpack_from('<IH', data, fo(p) + 8 * k)
                if not ok(p2, n) or not (0 < qw < 4096):
                    continue
                s = fo(p2)
                e = s + qw * 16
                lo = bisect.bisect_left(pstarts, s)
                hi = bisect.bisect_left(pstarts, e)
                # a region from the rmcGeometry list is exactly ONE packet that starts with
                # its own header (multiplier, vertex count). Searching for 'sub-packets'
                # inside the region (left from early stages of the analysis) found false
                # packets at unaligned addresses: random bytes gave
                # hundreds of 'vertices' that were joined into huge 'stars'.
                v, u, f = extract_region_mesh(data, s, e, [(s, qw)],
                                              header_scale=True)
                if v is None:
                    continue
                if u is None or len(u) != len(v):
                    u = np.zeros((len(v), 4))
                if u.shape[1] < 4:
                    u = np.hstack([u, np.zeros((len(u), 4 - u.shape[1]))])
                light = u[:, 4:5].copy() if u.shape[1] >= 5 else np.full((len(v), 1), -1.0)
                # columns 4..6 -- vertex colour index (or -1 if there is no colour)
                col = np.full((len(v), 1), -1.0)
                if cpv0 is not None:
                    cntp, arrp = struct.unpack_from('<II', data, cpv0 + 8 * i)
                    if ok(arrp, n):
                        cv = struct.unpack_from('<I', data, fo(arrp) + 4 * k)[0]
                        if ok(cv, n) and fo(cv) + len(v) <= n:
                            col = np.frombuffer(data[fo(cv):fo(cv) + len(v)],
                                                dtype=np.uint8).astype(np.float64).reshape(-1, 1)
                # columns: 0-3 UV, 4 -- palette index from the CPV set (-1), 5 -- palette index from V4-8 (-1)
                u = np.hstack([u[:, :4], col, light])
                V.append(np.asarray(v, dtype=np.float64))
                U.append(np.asarray(u, dtype=np.float64))
                F.extend((a + voff, b + voff, c + voff) for a, b, c in f)
                voff += len(v)
        if V:
            chunks.append((idx[i], np.concatenate(V), np.concatenate(U), F))
    return chunks


def face_key(mat, w, a, b, c):
    """Face key for dropping duplicates: material + vertices regardless of order.
    Both copies with the same winding (redundant) and pairs with the opposite
    one match (double-sided faces: in Blender one face is visible from both
    sides anyway, and two in one plane flicker). Faces with a DIFFERENT material
    in the same plane are a second pass with another texture and are kept.
    Verified on Atlanta: copies with the same winding and material 667,
    double-sided pairs 1304, second passes 158 (of 266 thousand faces)."""
    return (mat,) + tuple(sorted(tuple(np.round(w[i], 4)) for i in (a, b, c)))


def shader_texture_ptr(data, S, resolve_tex):
    """Texture object of a shader (file offset of TEX0) or None.

    rmcShaderBasic (type 0): texture at +8 (rmcShaderBasic::Bind).
    Shaders with a complex layout -- rmcShaderComplex (type 1, e.g.
    water.shadert) and mcShaderFlareTexScroll (type 19, city_flarebg.shadert:
    light cards, smoke columns) -- keep the texture as the first element of
    the array at +0x10 (at +8 they have a pass object). The fallback
    is used only if +8 gave no texture, and the result must
    point to a real texture. Verified: Atlanta City via the normal
    path -- 8408 chunks, via the fallback -- 23 more (water), none without a texture;
    for props the fallback adds 8 chunks (flares and smoke).
    """
    n = len(data)
    tv = struct.unpack_from('<I', data, S + 8)[0]
    if ok(tv, n):
        t0 = resolve_tex(fo(tv))
        if t0 is not None:
            return t0
    if True:
        a = struct.unpack_from('<I', data, S + 0x10)[0]
        if ok(a, n):
            tv = struct.unpack_from('<I', data, fo(a))[0]
            if ok(tv, n):
                return resolve_tex(fo(tv))
    return None


def extract_city_models(data, outdir, pck_path=None, ppf_path=None, channel_order='rgba',
                        include_reflect=False, per_instance_lighting=True):
    """City geometry with the REAL textures through the engine chain:

      instance +0x0c -> type; byte +10 -> sector g (shader group)
      type +60..+76  -> slot models main/ground/reflect/hdr/alpha
      model          -> chunks: shader index + own geometry
      mcCity+12 [g]  -> +4 shader array; shader[index] +8 -> texture:
                       type 0 rmcTexturePS2 (TEX0 at +16) |
                       type 2 reference, empty name -> +12 -> rmcTexturePS2
    Groups are spatial sectors (~4-5 x 4 areas); one index may map to
    different textures in different sectors, so the material is
    determined by the instance's (type, sector), not by the type.
    Scale: the packet header multiplier (see parse_packet_batches).
    Lighting: palette colour per vertex (see read_model_geom, CPV_SCALE).
    Blocks (mcCityModel) are centred at zero, world = vertex*scale + centre."""
    n = len(data)
    os.makedirs(outdir, exist_ok=True)
    city = 0x80
    if city + 28 > n or not ok(struct.unpack_from('<I', data, city + 24)[0], n):
        return None
    hood_count = struct.unpack_from('<I', data, city + 20)[0]
    hood_arr = fo(struct.unpack_from('<I', data, city + 24)[0])
    n_groups = struct.unpack_from('<I', data, city + 8)[0]
    garr = fo(struct.unpack_from('<I', data, city + 12)[0])
    if not (0 < hood_count < 100):
        return None

    small = collect_packets(data, max_qw=2048)
    small.sort()
    pstarts = [t for t, _ in small]
    # vertex colour palette (rmcCpvPalette): mcCity+44 -> 256 float RGBA entries
    cpv_pal = None
    pp_ = struct.unpack_from('<I', data, city + 44)[0]
    if ok(pp_, n) and fo(pp_) + 4096 <= n:
        cpv_pal = np.frombuffer(data[fo(pp_):fo(pp_) + 4096], dtype='<f4').reshape(256, 4)[:, :3].astype(np.float64)
    print(f"  districts (mcHood): {hood_count}, shader sectors: {n_groups}, "
          f"VIF packets: {len(small)}")

    # --- textures: extract the .pck/.ppf pair directly into outdir/textures ---
    tex_dir = os.path.join(outdir, 'textures')
    obj_dir = os.path.join(outdir, 'obj')
    os.makedirs(obj_dir, exist_ok=True)
    tex0_file = {}
    tex0_set = set()
    try:
        import mc3_extract_textures as MT
        tex0_set = {ch[0]['off'] for ch in MT.collect_descriptors(data)}
        if pck_path and ppf_path and os.path.exists(ppf_path):
            os.makedirs(tex_dir, exist_ok=True)
            tman = MT.extract(pck_path, ppf_path, tex_dir, base_only=True,
                              channel_order=channel_order)
            # the same picture is stored in copies on pages of different sectors
            # (pages are self-contained for streaming) -- identical files
            # are merged into one, so that OBJ variants appear only
            # where the textures really differ
            canon = {}
            tex0_file = {}
            for m in tman:
                if 'tex0_off' not in m or m.get('mip_level', 0) != 0:
                    continue
                h = hashlib.md5(open(os.path.join(tex_dir, m['file']), 'rb').read()).hexdigest()
                tex0_file[m['tex0_off']] = canon.setdefault(h, m['file'])
            print(f"  textures extracted: {len(tex0_file)} (from {os.path.basename(ppf_path)})")
        else:
            print("  .ppf not found -- geometry will have NO textures (use --ppf)")
    except Exception as ex:
        print(f"  textures not extracted: {ex}")

    glow_cache = {}

    def glow_texture(tex):
        """A copy of a window texture with alpha from luminance: the interior atlas
        is painted with light on a black background, and without transparency that
        background would cover the wall. Saved as <name>_glow.png next to the other textures."""
        if tex in glow_cache:
            return glow_cache[tex]
        out = None
        try:
            from PIL import Image
            src = os.path.join(tex_dir, tex)
            dst_name = os.path.splitext(tex)[0] + '_glow.png'
            dst = os.path.join(tex_dir, dst_name)
            if not os.path.isfile(dst):
                im = np.asarray(Image.open(src).convert('RGBA')).copy()
                lum = (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
                im[..., 3] = np.clip(lum * 1.6, 0, 255).astype(np.uint8)
                Image.fromarray(im, 'RGBA').save(dst)
            out = dst_name
        except Exception as ex:
            print(f"    could not make the window texture from {tex}: {ex}")
        glow_cache[tex] = out
        return out

    decal_cache = {}

    def is_decal(tex):
        """Does the texture of a lifted chunk have transparent areas (a decal:
        markings, manholes, stains on a transparent background)."""
        if tex not in decal_cache:
            fr = 0.0
            try:
                from PIL import Image
                a = np.asarray(Image.open(os.path.join(tex_dir, tex)).convert('RGBA'))[..., 3]
                fr = float((a < 128).mean())
            except Exception:
                pass
            decal_cache[tex] = fr > 0.01
        return decal_cache[tex]

    detail_kinds = {}       # (detail|su|sv) -> material suffix _roadN
    detail_of = {}          # _roadN -> (detail file, su, sv)
    wall_cache = {}

    def wall_texture(tex):
        """Wall of the window shader (type 16): the first texture is the facade (stone,
        granite, frames), and its alpha is 1 ONLY in the window glass (there the
        picture has a black opening). The game shows the wall where alpha is 0, and in the glass --
        the interior from the second texture. For Blender: a copy <name>_wall.png with
        inverted alpha (the glass is cut out). If alpha is 1 everywhere (the extractor
        replaces all-zero alpha that way), there are no windows -- a solid wall."""
        if tex in wall_cache:
            return wall_cache[tex]
        out = tex
        try:
            from PIL import Image
            im = np.asarray(Image.open(os.path.join(tex_dir, tex)).convert('RGBA')).copy()
            a = im[..., 3]
            if a.min() < 255:
                im[..., 3] = 255 - a
                out = os.path.splitext(tex)[0] + '_wall.png'
                Image.fromarray(im, 'RGBA').save(os.path.join(tex_dir, out))
            else:
                out = tex
        except Exception as ex:
            print(f"    could not make the wall from {tex}: {ex}")
        wall_cache[tex] = out
        return out

    hdr_cache = {}

    def hdr_alpha_texture(tex):
        """The file giving transparency to an hdr-slot material: the texture itself if
        it has transparent areas (spotlight beams have a transparent background
        around the light strip); a copy with luminance alpha (<name>_lum.png) for
        pictures on a black background; None -- an opaque glowing surface."""
        if tex in hdr_cache:
            return hdr_cache[tex]
        out = tex
        try:
            from PIL import Image
            im = np.asarray(Image.open(os.path.join(tex_dir, tex)).convert('RGBA')).copy()
            if float((im[..., 3] < 128).mean()) < 0.01:
                lum = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
                if float((lum < 25).mean()) > 0.5:
                    # a glowing picture on a black background (4 of 99 hdr textures
                    # without own transparency in Atlanta): alpha from luminance
                    im[..., 3] = np.clip(lum * 1.6, 0, 255).astype(np.uint8)
                    out = os.path.splitext(tex)[0] + '_lum.png'
                    Image.fromarray(im, 'RGBA').save(os.path.join(tex_dir, out))
                    used_tex.add(out)
                else:
                    # an ordinary surface that glows at night (the Westin crown
                    # etc., 86 of 99): opaque. Luminance alpha
                    # punched holes in it along the dark parts of the texture
                    out = None
        except Exception:
            pass
        hdr_cache[tex] = out
        return out

    def resolve_tex(tx):
        for _ in range(4):
            if tx is None or tx + 16 > n:
                return None
            typ = data[tx + 4]
            if typ == 0 and (tx + 16) in tex0_set:
                return tx + 16
            if typ == 2:
                p = struct.unpack_from('<I', data, tx + 12)[0]
                tx = fo(p) if ok(p, n) else None
                continue
            return None
        return None

    def window_pass(g, shader_idx):
        """Second window pass: (texture file, whether it is needed).

        The type 16 shader (mcShaderCityWindow) stores TWO textures: at +8 --
        the wall with the glass marked by alpha, at +0x0c -- the interior,
        which the game adds as a second pass through the glass, using the SECOND
        UV pair (V4-16 block). Verified: of 1232 chunks with this shader 1226
        use V4-16, and all other shader classes always have one
        UV pair. Without this pass buildings lack their windows."""
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n) or (data[fo(shv) + 4] & 0x7F) != 16:
            return None
        tv = struct.unpack_from('<I', data, fo(shv) + 0x0c)[0]
        if not ok(tv, n):
            return None
        t0 = resolve_tex(fo(tv))
        return tex0_file.get(t0) if t0 is not None else None

    def shader_type(g, shader_idx):
        """Shader type byte (+4 & 0x7F) of shader shader_idx in sector g, or None."""
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n):
            return None
        return data[fo(shv) + 4] & 0x7F

    def shader_basecolor(g, shader_idx):
        """basecolor of hdr_object / double_sided_hdr_object shaders (type 2,
        template 5 or 7): four floats 0..255 at +12..+24; the template modulates
        the texture by it (white font -> red/green/blue sign lettering). None for
        other shaders or white."""
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n):
            return None
        S = fo(shv)
        if (data[S + 4] & 0x7F) != 2 or ((struct.unpack_from('<I', data, S + 4)[0] >> 15) & 0x7F) not in (5, 7):
            return None
        rgba = struct.unpack_from('<4f', data, S + 12)
        if not all(0.0 <= x <= 255.5 for x in rgba):
            return None
        rgb = tuple(min(max(x / 255.0, 0.0), 1.0) for x in rgba[:3])
        if all(x >= 0.995 for x in rgb):
            return None
        return rgb

    def window_tint(g, shader_idx):
        """WindowTint of the window shader (type 16): RGBA floats 0..255 at
        +16..+28 (city_window.shadert: 'basecolor %2 %3 %4 255' of the second
        pass -- the interior texture is modulated by it). None for white."""
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n) or (data[fo(shv) + 4] & 0x7F) != 16:
            return None
        rgba = struct.unpack_from('<4f', data, fo(shv) + 16)
        if not all(0.0 <= x <= 255.5 for x in rgba):
            return None
        rgb = tuple(min(max(x / 255.0, 0.0), 1.0) for x in rgba[:3])
        return None if all(x >= 0.995 for x in rgb) else rgb

    def road_detail(g, shader_idx):
        """Road detail texture: (file, U scale, V scale) or None.

        The city_road template (shader type 17) draws a road in 2 passes: the base
        texture (+8), then a second one (+0x0c), tiled with the scale at +16/+20
        (4.0 / 4.0), with normal blending. The second texture is black with alpha
        (the city's own textures mcCity+472/+476): it darkens the road
        with stains and grain. Result: road * (1 - detail alpha)."""
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n) or (data[fo(shv) + 4] & 0x7F) != 17:
            return None
        tv = struct.unpack_from('<I', data, fo(shv) + 0x0c)[0]
        if not ok(tv, n):
            return None
        t0 = resolve_tex(fo(tv))
        f_ = tex0_file.get(t0) if t0 is not None else None
        if not f_:
            return None
        su, sv = struct.unpack_from('<2f', data, fo(shv) + 16)
        if not (0.01 < su < 64 and 0.01 < sv < 64):
            su = sv = 4.0
        return f_, round(su, 3), round(sv, 3)

    def chunk_texture(g, shader_idx):
        if g >= n_groups:
            return None
        grp = garr + 16 * g
        if shader_idx > struct.unpack_from('<H', data, grp + 8)[0]:
            return None
        sp = struct.unpack_from('<I', data, grp + 4)[0]
        if not ok(sp, n):
            return None
        shv = struct.unpack_from('<I', data, fo(sp) + 4 * shader_idx)[0]
        if not ok(shv, n):
            return None
        t0 = shader_texture_ptr(data, fo(shv), resolve_tex)
        return tex0_file.get(t0) if t0 is not None else None

    type_cache = {}

    def compute_fallback_scales():
        """Fallback scale -- the most common (typical) one AMONG RELIABLE
        objects of the same category (inst/block); used only when
        the value computed from the radius is far from an integer power of two.

        IMPORTANT: do NOT replace with one constant for all types -- checked
        in practice: the median radius/half over all types gives 2^-10,
        but that is just ONE OF SEVERAL clusters of roughly equal
        frequency (for blocks, e.g.: 0.0039~32%, 0.00195~23%,
        0.000977~21%...), not a dominant universal value. Forcing
        this single constant for ALL types
        broke the size of most objects at once (visually -- total
        chaos instead of a city). A per-type scale from the radius (with a fallback
        only for the clearly unreliable ~4-7% of cases, mostly
        block fragments #geomN, where the radius means something else)
        remains the right approach. (Superseded by the packet header multiplier.)"""
        from collections import Counter
        counters = {'inst': Counter(), 'block': Counter()}
        for kind, cnt_off, arr_off, stride in (('inst', 12, 16, 96), ('block', 4, 8, 28)):
            for hi in range(hood_count):
                h = hood_arr + 20 * hi
                nr = struct.unpack_from('<I', data, h + cnt_off)[0]
                av = struct.unpack_from('<I', data, h + arr_off)[0]
                if not ok(av, n) or not (0 < nr < 20000):
                    continue
                for i in range(nr):
                    e = fo(av) + stride * i
                    if e + stride > n:
                        continue
                    tp = struct.unpack_from('<I', data, e + 0x0c)[0]
                    if not ok(tp, n):
                        continue
                    t = fo(tp)
                    if t + 80 > n:
                        continue
                    radius = struct.unpack_from('<f', data, t + 56)[0]
                    if not (0 < radius < 1e5):
                        continue
                    vs = []
                    for k in range(5):
                        mp = struct.unpack_from('<I', data, t + 60 + 4 * k)[0]
                        if ok(mp, n):
                            ch = read_model_geom(data, fo(mp), small, pstarts)
                            if ch:
                                vs.extend(c[1] for c in ch)
                    if not vs:
                        continue
                    allv = np.concatenate(vs)
                    half = np.linalg.norm(allv.max(0) - allv.min(0)) / 2
                    if half <= 0:
                        continue
                    log2r = np.log2(radius / half)
                    if abs(log2r - round(log2r)) < 0.15:
                        counters[kind][2.0 ** round(log2r)] += 1
        out = {}
        for kind, c in counters.items():
            out[kind] = c.most_common(1)[0][0] if c else 1.0
        print(f"  fallback scale (typical, among reliable ones): "
              f"inst={out['inst']:.6f} block={out['block']:.6f}")
        return out

    fallback_scale = {}   # unused: the scale comes from the packet header

    def quant_rule_scale(V, radius):
        """int16 quantisation rule -- ONLY for blocks. The engine
        chooses the model's binary exponent so that the largest raw
        coordinate fills almost the whole int16 range: for 98.9% of Atlanta
        blocks max|coordinate| lies in [16384, 32767]. Hence the scale s
        -- the power of two for which max|coord|*s lies in [R/sqrt3, R]
        (R -- the radius of the sphere centred at the zero of raw coordinates);
        the interval width sqrt3 < 2, so at most one
        power fits. Verified: where the type radius is reliable, the rule
        matches it for 590 of 591 blocks. For INSTANCES the rule does NOT
        work (right for only 35 of 231 -- their coordinates are not centred
        at zero), they have their own reference, see vote_instance_scales.
        Returns None if no power fell into the interval."""
        m = np.abs(V).max()
        if m <= 0 or radius <= 0:
            return None
        lo, hi = radius / (np.sqrt(3) * m), radius / m
        s = 2.0 ** np.floor(np.log2(hi))
        return s if s >= lo * 0.999 else None

    def vote_instance_scales():
        """Instance type scale from the game's own REFERENCE. An instance record
        (96 bytes) stores the world bounding sphere of the placement:
        +0x18 radius, +0x50..+0x58 centre. For each placement
        log2(R_world / (raw_half_diagonal * matrix_scale)), where
        matrix_scale = |det|^(1/3), is rounded to an integer; the type
        gets the most common answer. Verified: with this scale the centre
        of the transformed geometry matches the reference centre with
        a median error of 0.000 (99.1% within a third of the radius);
        the old fit by the TYPE radius (+56) was 2-8 times off for 4 of
        231 types (a_inst_fwy_offramp_02x -- 8 times)."""
        from collections import Counter, defaultdict
        votes = defaultdict(Counter)
        half_cache = {}
        for hi in range(hood_count):
            h = hood_arr + 20 * hi
            nr = struct.unpack_from('<I', data, h + 12)[0]
            av = struct.unpack_from('<I', data, h + 16)[0]
            if not ok(av, n) or not (0 < nr < 20000):
                continue
            for i in range(nr):
                e = fo(av) + 96 * i
                if e + 96 > n:
                    continue
                tp = struct.unpack_from('<I', data, e + 0x0c)[0]
                if not ok(tp, n):
                    continue
                t = fo(tp)
                if t not in half_cache:
                    vs = []
                    for k in range(5):
                        mp = struct.unpack_from('<I', data, t + 60 + 4 * k)[0]
                        if ok(mp, n):
                            ch = read_model_geom(data, fo(mp), small, pstarts)
                            if ch:
                                vs.extend(c[1] for c in ch)
                    if vs:
                        V = np.concatenate(vs)
                        half_cache[t] = np.linalg.norm(V.max(0) - V.min(0)) / 2
                    else:
                        half_cache[t] = 0
                half = half_cache[t]
                R = struct.unpack_from('<f', data, e + 0x18)[0]
                M = np.array([struct.unpack_from('<3f', data, e + o)
                              for o in (0x20, 0x2c, 0x38)])
                sm = abs(np.linalg.det(M)) ** (1 / 3)
                if half > 0 and R > 0 and sm > 0:
                    votes[t][int(round(np.log2(R / (half * sm))))] += 1
        out = {t: 2.0 ** c.most_common(1)[0][0] for t, c in votes.items() if c}
        print(f"  instance scale from the game reference: {len(out)} types")
        return out

    inst_scale = {}       # unused: the scale comes from the packet header

    def type_geom(t, kind='inst'):
        if t in type_cache:
            return type_cache[t]
        res = None
        if t + 80 <= n:
            radius = struct.unpack_from('<f', data, t + 56)[0]
            name_ptr = struct.unpack_from('<I', data, t + 0x08)[0]
            name = None
            if ok(name_ptr, n):
                s = data[fo(name_ptr):fo(name_ptr) + 48].split(b'\x00')[0]
                if 3 <= len(s) <= 47 and all(32 <= b < 127 for b in s):
                    name = s.decode()
            slots = []
            for k in range(5):
                # slot 2 (reflect) -- simplified geometry mirrored about the
                # ground, for reflections on wet asphalt: for 69% of types with this
                # slot the lower bound of the reflection = -upper bound of main (the extents
                # are mirrored, the vertices are a separate simplified model). It is not
                # visible geometry: in the scene it gave upside-down buildings under
                # the streets. Not exported by default (--with-reflect -- export).
                if k == 2 and not include_reflect:
                    continue
                mp = struct.unpack_from('<I', data, t + 60 + 4 * k)[0]
                if ok(mp, n):
                    ch = read_model_geom(data, fo(mp), small, pstarts)
                    if ch:
                        slots.append((k, ch))
            if name and slots and 0 < radius < 1e5:
                allv = np.concatenate([c[1] for _, chs in slots for c in chs])
                half = np.linalg.norm(allv.max(0) - allv.min(0)) / 2
                # coordinates are already multiplied by the header multiplier of each
                # packet (packet_header_scale) -- no extra scale is needed
                scale = 1.0
                center = np.array(struct.unpack_from('<3f', data, t + 44))
                # Seams between strips are cut exactly, by the ADC flag from the data
                # (see extract_region_mesh). The heuristic filter_strip_artifacts
                # is no longer applied here: it also cut legitimate faces
                # (roofs, panels, thin kerbs), and seams are no longer created
                # at all -- exactly the triangles the game draws are drawn.
                res = (name, slots, scale, center)
        type_cache[t] = res
        return res

    variants = {}          # (t, texture key) -> obj file name
    name_count = {}
    used_tex = set()
    alpha_tex = set()
    mat_kinds = set()     # (texture file, material suffix[, basecolor tint suffix])
    tint_of = {}          # '_tRRGGBB' -> (r, g, b) 0..1
    placements, manifest = [], []
    stats = {'tinted': 0, 'chunks': 0, 'textured': 0, 'dupes_removed': 0, 'window_passes': 0,
             'colored_verts': 0, 'own_lighting_verts': 0,
             'lifted_chunks': 0, 'decals': 0, 'v48_verts': 0,
             'road_detail': 0}

    cpv_cache = {}

    def overlay_levels(slots, xform):
        """Overlay levels of chunks within an object: {(k, ci): level}.

        Within one object chunks often lie on top of each other in the same
        plane: asphalt with markings over it, paths on a lawn, stains -- and
        also VERTICAL overlays on walls (San Diego s_inst_dt_blk01_01x: a row
        of arched windows, main_c4, lies exactly on the window wall main_c2).
        The main pass has no blending and no alpha test (rmcState: ALPHA
        0x1000A = Cs, TEST ATST=ALWAYS), so the game simply draws the later
        chunk over the earlier one at the same depth. In Blender coinciding
        faces flicker and the overlay is mostly hidden. Chunks are walked in
        draw order (main, then hdr, alpha). A pair counts as overlapping if
        face centres of one chunk lie (+-2 cm) on a PARALLEL face of the other
        -- in any orientation, in BOTH directions. The later chunk gets a
        level one higher than the earlier."""
        rng = np.random.default_rng(0)

        def samples(T):
            # centres and unit normals of ALL faces (up to 300): thin markings
            # consist of tiny faces, a random point sample missed them
            n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            ln = np.linalg.norm(n, axis=1)
            good = np.flatnonzero(ln > 2e-6)
            if not len(good):
                return None
            if len(good) > 300:
                good = rng.choice(good, size=300, replace=False)
            return T[good].mean(1), n[good] / ln[good, None]

        def tri_frame(T2):
            a = T2[:, 0]; e1 = T2[:, 1] - a; e2 = T2[:, 2] - a
            n = np.cross(e1, e2); ln = np.linalg.norm(n, axis=1)
            ok_ = ln > 2e-6
            nh = n / np.where(ok_, ln, 1)[:, None]
            d00 = (e1 * e1).sum(1); d01 = (e1 * e2).sum(1); d11 = (e2 * e2).sum(1)
            den = d00 * d11 - d01 * d01
            ok_ &= np.abs(den) > 1e-14
            lo = T2.min(1) - 0.03; hi = T2.max(1) + 0.03
            return a, e1, e2, nh, d00, d01, d11, np.where(ok_, den, 1), ok_, lo, hi

        def any_on(S, F2):
            if S is None:
                return False
            P, Np = S
            a, e1, e2, nh, d00, d01, d11, den, ok_, lo, hi = F2
            for pnt, npn in zip(P, Np):
                near = ok_ & np.all(lo <= pnt, 1) & np.all(hi >= pnt, 1)
                if not near.any():
                    continue
                idx = np.flatnonzero(near)
                w = pnt - a[idx]
                dist = np.abs((w * nh[idx]).sum(1))
                par = np.abs((nh[idx] * npn).sum(1)) > 0.95
                m = (dist < 0.02) & par
                if not m.any():
                    continue
                idx = idx[m]; w = w[m]
                d20 = (w * e1[idx]).sum(1); d21 = (w * e2[idx]).sum(1)
                vv = (d11[idx] * d20 - d01[idx] * d21) / den[idx]
                ww = (d00[idx] * d21 - d01[idx] * d20) / den[idx]
                if np.any((vv >= -1e-4) & (ww >= -1e-4) & (vv + ww <= 1 + 1e-4)):
                    return True
            return False

        levels, done = {}, []
        for k, chs in slots:
            if k not in (0, 3, 4):
                continue
            for ci, (si, v, u, f) in enumerate(chs):
                lvl = 0
                if len(f):
                    T = xform(v)[np.array(f)]
                    S = samples(T)
                    F = tri_frame(T)
                    for S2, F2, l2 in done:
                        if l2 + 1 > lvl and (any_on(S, F2) or any_on(S2, F)):
                            lvl = l2 + 1
                    done.append((S, F, lvl))
                levels[(k, ci)] = lvl
        return levels

    def write_obj(path, header, slots, g, xform, t_off=None, cpv_set=0):
        seen_faces = set()
        levels = overlay_levels(slots, xform)
        with open(path, 'w') as fh:
            fh.write(header)
            fh.write("mtllib city.mtl\n")
            voff = 0
            for k, chs in slots:
                for ci, (si, v, u, f) in enumerate(chs):
                    tex = chunk_texture(g, si)
                    lvl = levels.get((k, ci), 0)
                    stats['chunks'] += 1
                    fh.write(f"g {SLOT_NAMES[k]}_c{ci}\n")
                    if tex:
                        stats['textured'] += 1
                        used_tex.add(tex)
                        # separate materials for the hdr (emission) and alpha
                        # (cutout) slots: the same textures are also used in the
                        # main geometry (52 -- in main and alpha, 92 -- in main and
                        # hdr), emission/cutout must not end up on ordinary walls
                        kind = '_hdr' if k == 3 else '_alpha' if k == 4 else \
                            '_ground' if k == 1 else ''
                        rd = road_detail(g, si) if tex else None
                        if rd:
                            # one material per (road, detail, scale) combination
                            key_ = f"{os.path.splitext(rd[0])[0]}|{rd[1]}|{rd[2]}"
                            if key_ not in detail_kinds:
                                detail_kinds[key_] = f"_road{len(detail_kinds)}"
                                detail_of[detail_kinds[key_]] = rd
                            kind = detail_kinds[key_]
                            used_tex.add(rd[0])
                            stats['road_detail'] += 1
                        # window shader (type 16): the first texture is the WALL, its alpha
                        # marks the glass; the second (+0x0c, second UV pair) is
                        # the interior seen through the glass
                        if k != 3 and u.shape[1] >= 4 and np.ptp(u[:, 2:4]) > 1e-9 \
                                and window_pass(g, si):
                            kind = '_mask'
                        # a lifted overlay gets transparency ONLY with a template shader
                        # (type 2: doublesided, city_window_cutout... -- their .shadert
                        # sets an alpha test). Basic shaders (type 0) are drawn in the
                        # main/ground pass with no blending and ATST=ALWAYS
                        # (mcInstCityModelClass::SetRenderStates -> rmcState: ALPHA
                        # 0x1000A = Cs, TEST ATST 1), i.e. the texture alpha is ignored.
                        if lvl > 0 and kind == '' and shader_type(g, si) == 2 and is_decal(tex):
                            kind = '_decal'
                            stats['decals'] += 1
                        if kind == '_mask':
                            wtex = wall_texture(tex)
                            if wtex != tex:
                                used_tex.add(wtex)
                                tex = wtex
                            else:
                                kind = ''          # no windows -- a solid wall
                        # basecolor of hdr_object templates: a separate material per
                        # (texture, colour); the tint goes BEFORE the kind suffix so the
                        # importer's suffix checks (_hdr, _win...) keep working
                        bc = shader_basecolor(g, si)
                        tsfx = ''
                        if bc:
                            tsfx = '_t%02x%02x%02x' % tuple(int(round(x * 255)) for x in bc)
                            tint_of[tsfx] = bc
                        mat_kinds.add((tex, kind, tsfx))
                        fh.write(f"usemtl {os.path.splitext(tex)[0]}{tsfx}{kind}\n")
                    else:
                        tsfx = ''
                        fh.write("usemtl no_texture\n")
                    mat = (os.path.splitext(tex)[0] + tsfx + kind) if tex else 'no_texture'
                    w = xform(v)
                    if lvl > 0:
                        # the chunk lies on top of an earlier one -- move it 5 cm per
                        # level OUTWARDS along its own face normals (up for road
                        # markings, out of the wall for decals on facades)
                        w = w.copy()
                        vn = np.zeros_like(w)
                        if len(f):
                            fa = np.array(f)
                            fn = np.cross(w[fa[:, 1]] - w[fa[:, 0]], w[fa[:, 2]] - w[fa[:, 0]])
                            for j in range(3):
                                np.add.at(vn, fa[:, j], fn)
                        ln_ = np.linalg.norm(vn, axis=1)
                        vn = np.where(ln_[:, None] > 1e-9, vn / np.where(ln_ > 1e-9, ln_, 1)[:, None], 0)
                        w += vn * (0.05 * lvl)
                        stats['lifted_chunks'] += 1
                    # vertex colour = palette colour (1.0 -- texture unchanged);
                    # without colour -- just 'v x y z'. The scale is verified in the microcode
                    cols = None
                    if cpv_pal is not None and u.shape[1] >= 5:
                        ci_ = u[:, 4]
                        if t_off is not None and cpv_set:
                            mp_ = struct.unpack_from('<I', data, t_off + 60 + 4 * k)[0]
                            if ok(mp_, n):
                                kk = (fo(mp_), cpv_set)
                                if kk not in cpv_cache:
                                    cpv_cache[kk] = cpv_indices(data, fo(mp_), cpv_set)
                                arrs = cpv_cache[kk]
                                if arrs and ci < len(arrs) and arrs[ci] is not None \
                                        and len(arrs[ci]) == len(v):
                                    ci_ = arrs[ci]
                                    stats['own_lighting_verts'] += len(v)
                        if np.all(ci_ >= 0):
                            cols = CPV_SCALE * cpv_pal[ci_.astype(int)]
                            stats['colored_verts'] += len(ci_)
                    if cols is None and cpv_pal is not None and u.shape[1] >= 6 and np.all(u[:, 5] >= 0):
                        # a chunk without its own CPV sets but with a V4-8 block (hdr
                        # layer, reflections, some foliage): the 4th byte is a palette
                        # INDEX into the same lighting palette. Microcode (0x5A97B0): MTIR
                        # VI11, VF21.w -> LQ palette(VI11), where VF21 is the +2 slot
                        # of the vertex (xyz -- normal for lighting).
                        cols = CPV_SCALE * cpv_pal[u[:, 5].astype(int).clip(0, 255)]
                        stats['v48_verts'] += len(cols)
                    if cols is not None:
                        for p, c3 in zip(w, cols):
                            fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                                     f"{c3[0]:.3f} {c3[1]:.3f} {c3[2]:.3f}\n")
                    else:
                        for p in w:
                            fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
                    # V is written AS IS for every layer (no 1 - v). The extracted
                    # PNGs are already upright, and the game's UVs address them the
                    # same way everywhere. Verified by texture content, not by
                    # statistics: shop fronts (awning up, doors down) and garage
                    # doors of a_inst_gasstation_ax, the arched windows of San Diego
                    # s_inst_dt_blk01_01x (dome up), the sky strips (clouds up,
                    # horizon haze down), tree atlases (canopy up, bark strip down),
                    # the 'Water Park' and 'Rx' signs of the hdr layer and the
                    # 'nitro cola' prop sign. With 1 - v all of these came out
                    # upside down; an earlier 'GO-GAS' check that suggested
                    # flipping the main layer was wrong.
                    for q in u:
                        fh.write(f"vt {q[0]:.4f} {q[1]:.4f}\n")
                    for a, b, c in f:
                        key = face_key(mat, w, a, b, c)
                        if key in seen_faces:
                            stats['dupes_removed'] += 1
                            continue
                        seen_faces.add(key)
                        fh.write(f"f {a+voff+1}/{a+voff+1} {b+voff+1}/{b+voff+1} "
                                 f"{c+voff+1}/{c+voff+1}\n")
                    voff += len(v)

                    # interior of the window shader (type 16, mcShaderCityWindow): the second
                    # texture (+0x0c) on the second UV pair -- the view through the glass
                    # (lit rooms, curtains); the wall from the first texture lies
                    # on top with the glass cut out. The interior is moved 5 cm INWARDS,
                    # behind the wall (the game has a fixed pass order; 1 cm flickered at a distance),
                    # and is self-lit. Verified on the textures: the first one has alpha = 1
                    # exactly in the black window openings, and stone and granite under
                    # zero alpha (the Capitol, the Georgia-Pacific tower).
                    wt = window_pass(g, si) if u.shape[1] >= 4 else None
                    if wt and np.ptp(u[:, 2:4]) > 1e-9:
                        nrm = np.zeros_like(w)
                        for a, b, c in f:
                            fn_ = np.cross(w[b] - w[a], w[c] - w[a])
                            nrm[a] += fn_; nrm[b] += fn_; nrm[c] += fn_
                        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
                        nrm = np.divide(nrm, np.where(ln > 0, ln, 1))
                        w2 = w - nrm * 0.05      # 5 cm behind the frames (1 cm flickered at a distance)
                        stats['window_passes'] += 1
                        used_tex.add(wt)
                        wtint = window_tint(g, si)
                        wsfx = ''
                        if wtint:
                            wsfx = '_t%02x%02x%02x' % tuple(int(round(x * 255)) for x in wtint)
                            tint_of[wsfx] = wtint
                        mat_kinds.add((wt, '_win', wsfx))
                        fh.write(f"g {SLOT_NAMES[k]}_c{ci}_win\nusemtl {os.path.splitext(wt)[0]}{wsfx}_win\n")
                        if cols is not None:
                            for p, c3 in zip(w2, cols):
                                fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                                         f"{c3[0]:.3f} {c3[1]:.3f} {c3[2]:.3f}\n")
                        else:
                            for p in w2:
                                fh.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
                        for q in u:
                            fh.write(f"vt {q[2]:.4f} {q[3]:.4f}\n")   # second UV pair: as is, like the first
                        for a, b, c in f:
                            fh.write(f"f {a+voff+1}/{a+voff+1} {b+voff+1}/{b+voff+1} "
                                     f"{c+voff+1}/{c+voff+1}\n")
                        voff += len(v)

    def tex_key(slots, g):
        return tuple(chunk_texture(g, si) for _, chs in slots for (si, *_r) in chs)

    n_inst = n_block = 0
    for kind, cnt_off, arr_off, stride in (('inst', 12, 16, 96), ('block', 4, 8, 28)):
        for hi in range(hood_count):
            h = hood_arr + 20 * hi
            nr = struct.unpack_from('<I', data, h + cnt_off)[0]
            av = struct.unpack_from('<I', data, h + arr_off)[0]
            if not ok(av, n) or not (0 < nr < 20000):
                continue
            for i in range(nr):
                e = fo(av) + stride * i
                if e + stride > n:
                    continue
                tp = struct.unpack_from('<I', data, e + 0x0c)[0]
                if not ok(tp, n):
                    continue
                t = fo(tp)
                tg = type_geom(t, kind)
                if tg is None:
                    continue
                name, slots, scale, center = tg
                g = data[e + 10]
                safe = re.sub(r'[^A-Za-z0-9_.#-]', '_', name)
                if kind == 'inst':
                    Xr = np.array(struct.unpack_from('<3f', data, e + 0x20))
                    Yr = np.array(struct.unpack_from('<3f', data, e + 0x2c))
                    Zr = np.array(struct.unpack_from('<3f', data, e + 0x38))
                    P = np.array(struct.unpack_from('<3f', data, e + 0x44))
                    # lighting set number of this placement (+0x1c); it goes into the
                    # variant key only if the model has more than one set
                    # (mcInstCityModel::Render passes *(instance+28) as the set number
                    # to mcInstCityModelType::Render; for 154 types with set count =
                    # copy count the +28 values over the copies are exactly 0..k-1)
                    cpv_set = struct.unpack_from('<H', data, e + 0x1c)[0]
                    nsets = 0
                    for k_ in range(5):                 # trees have no main slot
                        mp0 = struct.unpack_from('<I', data, t + 60 + 4 * k_)[0]
                        if ok(mp0, n):
                            nsets = max(nsets, struct.unpack_from('<H', data, fo(mp0) + 10)[0])
                    if nsets <= 1 or cpv_set >= nsets or not per_instance_lighting:
                        cpv_set = 0
                    key = (t, tex_key(slots, g), cpv_set)
                    if key not in variants:
                        j = name_count.get(name, 0)
                        name_count[name] = j + 1
                        fn = f"{safe}.obj" if j == 0 else f"{safe}__v{j}.obj"
                        variants[key] = fn
                        write_obj(os.path.join(obj_dir, fn),
                                  f"# {name} (sector {g}) -- LOCAL coordinates, scale {scale:g};\n"
                                  f"# placement: instance_placements.csv "
                                  f"(world = local @ [a;b;c] + position)\n",
                                  slots, g, lambda v: v * scale, t_off=t, cpv_set=cpv_set)
                    placements.append([name, variants[key], g, *P.round(3).tolist(),
                                       *Xr.round(5).tolist(), *Yr.round(5).tolist(),
                                       *Zr.round(5).tolist()])
                    n_inst += 1
                else:
                    fn = f"block_{safe}.obj"
                    write_obj(os.path.join(obj_dir, fn),
                              f"# {name} -- block (mcCityModel), WORLD coordinates, "
                              f"sector {g}\n",
                              slots, g, lambda v: v * scale + center)
                    placements.append([name, fn, g, *center.round(3).tolist(),
                                       1, 0, 0, 0, 1, 0, 0, 0, 1])
                    n_block += 1

    # --- sky: a dome of two layers (sky, clouds) ---
    # The sky shader group is mcCity+16 (10 shaders). The dome models are
    # rmcModelGeom not referenced by any city type, whose shader indices
    # are all smaller than the size of this group (Atlanta: two models of 5
    # chunks, indices 0-4 and 5-9, radius ~98, height 0-33). Sky textures
    # are embedded directly in the .pck (they have no page references). In the game the dome
    # moves with the camera; for a static scene it is placed at the centre
    # of the city with a scale that covers the whole map.
    try:
        sgp = struct.unpack_from('<I', data, city + 16)[0]
        sky_arr, sky_cnt = None, 0
        if ok(sgp, n):
            sa = struct.unpack_from('<I', data, fo(sgp) + 4)[0]
            if ok(sa, n):
                sky_arr, sky_cnt = fo(sa), struct.unpack_from('<H', data, fo(sgp) + 8)[0]
        used_models = set()
        for tt in type_cache:
            for k in range(5):
                mp = struct.unpack_from('<I', data, tt + 60 + 4 * k)[0]
                if ok(mp, n):
                    used_models.add(fo(mp))
        model_vt = None
        for tt in type_cache:
            mp = struct.unpack_from('<I', data, tt + 60)[0]
            if ok(mp, n):
                model_vt = struct.unpack_from('<I', data, fo(mp))[0]
                break
        sky_models = []
        if sky_arr is not None and model_vt is not None:
            pat = struct.pack('<I', model_vt)
            i0 = data.find(pat)
            while i0 != -1:
                if i0 % 4 == 0 and i0 not in used_models:
                    nch_ = struct.unpack_from('<H', data, i0 + 8)[0]
                    ip_ = struct.unpack_from('<I', data, i0 + 12)[0]
                    if 0 < nch_ < 64 and ok(ip_, n):
                        idx_ = struct.unpack_from(f'<{nch_}H', data, fo(ip_))
                        if max(idx_) < sky_cnt:
                            ch_ = read_model_geom(data, i0, small, pstarts)
                            if ch_:
                                sky_models.append(ch_)
                i0 = data.find(pat, i0 + 1)
        if sky_models:
            allc = np.array([r[3:6] for r in placements], dtype=float)
            cen = (allc.max(0) + allc.min(0)) / 2 if len(allc) else np.zeros(3)
            half = (np.ptp(allc[:, [0, 2]], axis=0).max() / 2) if len(allc) else 1000.0
            skyv = np.concatenate([c[1] for ch_ in sky_models for c in ch_])
            rad = max(np.abs(skyv[:, [0, 2]]).max(), 1e-6)
            sc = 1.3 * half / rad
            cen[1] = 0.0

            def sky_tex(si):
                shv = struct.unpack_from('<I', data, sky_arr + 4 * si)[0]
                if not ok(shv, n):
                    return None
                t0 = shader_texture_ptr(data, fo(shv), resolve_tex)
                return tex0_file.get(t0) if t0 is not None else None

            with open(os.path.join(obj_dir, 'block_sky.obj'), 'w') as fh:
                fh.write(f"# sky: {len(sky_models)} dome layers, WORLD coordinates (city centre, "
                         f"scale x{sc:.1f} of the original radius {rad:.0f})\nmtllib city.mtl\n")
                voff = 0
                for li, ch_ in enumerate(sky_models):
                    for ci, (si, v, u, f) in enumerate(ch_):
                        tex = sky_tex(si)
                        fh.write(f"g sky{li}_c{ci}\n")
                        if tex:
                            used_tex.add(tex)
                            mat_kinds.add((tex, '_sky'))
                            fh.write(f"usemtl {os.path.splitext(tex)[0]}_sky\n")
                        else:
                            fh.write("usemtl no_texture\n")
                        w = v * sc + cen
                        for p in w:
                            fh.write(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
                        for q in u:
                            fh.write(f"vt {q[0]:.4f} {q[1]:.4f}\n")   # sky: as is (clouds up, horizon down)
                        for a, b, c in f:
                            fh.write(f"f {a+voff+1}/{a+voff+1} {b+voff+1}/{b+voff+1} "
                                     f"{c+voff+1}/{c+voff+1}\n")
                        voff += len(v)
            placements.append(['sky', 'block_sky.obj', 0, 0.0, 0.0, 0.0,
                               1, 0, 0, 0, 1, 0, 0, 0, 1])
            print(f"  sky: {len(sky_models)} dome layers, scale x{sc:.1f} -> obj/block_sky.obj")
    except Exception as ex:
        print(f"  sky not exported: {ex}")

    # --- scene parameters: sky clear colour ---
    # The sky object mcSkyHatClass (pointer at mcCity+400 in the file; in the game's
    # memory -- +384). Its parameters (mcSkyHatClass::FileIO) lie 8 bytes earlier
    # in the file than in memory: m_clearColor (+184 in memory) is at +176..+184
    # of the object. Changes with weather and time of day (Atlanta:
    # midnight-clear (0.12, 0.12, 0.10), cloudy (0.19, 0.19, 0.18), dawn
    # (0.15, 0.16, 0.17)); the neighbouring fields (lightning, m_glowColor, offset)
    # read correctly with the same shift, which confirms it.
    try:
        import json
        scene = {}
        sp = struct.unpack_from('<I', data, city + 400)[0]
        if ok(sp, n):
            rgb = struct.unpack_from('<3f', data, fo(sp) + 176)
            if all(0 <= c <= 2 for c in rgb):
                scene['sky_clear_color'] = [round(c, 4) for c in rgb]
            i0 = data.find(b'skyhat_')
            if i0 >= 0:
                scene['sky_name'] = data[i0:i0 + 64].split(b'\x00')[0].decode(errors='replace')
        if scene:
            with open(os.path.join(outdir, 'scene.json'), 'w') as f:
                json.dump(scene, f, ensure_ascii=False, indent=1)
            print(f"  sky clear colour: {scene.get('sky_clear_color')} ({scene.get('sky_name')}) -> scene.json")
    except Exception as ex:
        print(f"  scene parameters not written: {ex}")

    # --- materials and removal of unused textures ---
    with open(os.path.join(obj_dir, 'city.mtl'), 'w') as f:
        f.write("# City materials -- through the engine chain (sector -> shader -> texture)\n")
        f.write("newmtl no_texture\nKd 1 1 1\n")   # no texture (incl. a reference to the texture 'none'): colour from the vertex colours

        for entry in sorted(mat_kinds):
            tex, kind = entry[0], entry[1]
            tsfx = entry[2] if len(entry) > 2 else ''
            kd = tint_of.get(tsfx, (1.0, 1.0, 1.0))
            f.write(f"\nnewmtl {os.path.splitext(tex)[0]}{tsfx}{kind}\n"
                    f"Kd {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\n"
                    f"map_Kd ../textures/{tex}\n")
            if kind == '_alpha':
                # alpha slot (foliage, fences): cutout by the texture alpha
                f.write(f"map_d ../textures/{tex}\n")
            elif kind == '_sky':
                # sky: emission only, no scene lighting
                f.write(f"Ke 1 1 1\nmap_Ke ../textures/{tex}\n")
            elif kind.startswith('_road'):
                # road + detail texture (city_road template): the importer
                # multiplies the colour by (1 - detail alpha) on UV x scale
                dt_, su_, sv_ = detail_of[kind]
                f.write(f"detail_map ../textures/{dt_}\ndetail_scale {su_} {sv_}\n")
            elif kind == '_decal':
                # a decal over another chunk (markings, manholes): transparency
                f.write(f"map_d ../textures/{tex}\n")
            elif kind == '_ground':
                # the ground layer -- the roads and terrain themselves (there is no other
                # geometry under them: 0% coincide with main faces). The alpha of its textures is low
                # (5-15%, twice as high in rain) -- in the game the share of the wet-road
                # reflection (reflect layer); used as transparency it made the roads almost
                # invisible, leaving dark streaks. Opaque.
                pass
            elif kind == '_mask':
                # window shader wall: cutout by the inverted alpha (the glass is transparent)
                f.write(f"map_d ../textures/{tex}\n")
            elif kind == '_win':
                # interior behind the glass: self-lit, modulated by WindowTint
                f.write(f"Ke {kd[0]:.3f} {kd[1]:.3f} {kd[2]:.3f}\nmap_Ke ../textures/{tex}\n")
            elif kind == '_hdr':
                # hdr slot: night glow panels and spotlight beams -- emission
                # and transparency (the texture's own alpha, or luminance if it has none)
                dt = hdr_alpha_texture(tex)
                if dt:
                    f.write(f"Ke 1 1 1\nmap_Ke ../textures/{tex}\nmap_d ../textures/{dt}\n")
                else:
                    # an opaque surface lit at night (the Westin crown and
                    # the like). Its colour comes from the vertex colours (palette
                    # index from V4-8); HDR_SURFACE_TINT is an optional extra
                    # tint (neutral by default).
                    t = HDR_SURFACE_TINT
                    f.write(f"Ke {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}\nmap_Ke ../textures/{tex}\n")
    if os.path.isdir(tex_dir):
        for fn in os.listdir(tex_dir):
            if fn not in used_tex:
                os.remove(os.path.join(tex_dir, fn))

    with open(os.path.join(outdir, 'instance_placements.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'obj', 'sector', 'x', 'y', 'z', 'ax', 'ay', 'az',
                    'bx', 'by', 'bz', 'cx', 'cy', 'cz'])
        w.writerows(placements)
    pct = 100 * stats['textured'] / max(stats['chunks'], 1)
    print(f"  instances: {n_inst}, blocks: {n_block}, obj files: "
          f"{len(variants) + n_block} (variants by sector and lighting: {len(variants)} for "
          f"{len(name_count)} types)")
    print(f"  chunks written: {stats['chunks']}, textured: {stats['textured']} ({pct:.1f}%), "
          f"textures used: {len(used_tex)} -> textures/, obj/city.mtl; "
          f"duplicate faces removed: {stats['dupes_removed']}; "
          f"window passes: {stats['window_passes']}; "
          f"vertices with baked lighting: {stats['colored_verts']}, "
          f"with their placement's own set: {stats['own_lighting_verts']}; "
          f"chunks lifted over others: {stats['lifted_chunks']} (decals with transparency: {stats['decals']}); "
          f"vertices coloured by V4-8 palette index: {stats['v48_verts']}; "
          f"road chunks with a detail texture: {stats['road_detail']}")
    return placements


def euler_xyz_rows(x, y, z):
    """Matrix34::FromEulersXYZ (game code): rows are the local axes,
    world = local @ [a; b; c] + position. The cos/sin order was checked on
    the data: lamp posts/trees stand upright for 90% (with the reverse
    order -- 35%)."""
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.array([[cy * cz, cy * sz, -sy],
                     [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy],
                     [cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy]])


def looks_like_props(data):
    """A props file (<city>_<time>_<weather>_props.pck): the resource starts with
    mcPropManagerData -- 29 record counters at +16 and 29 pointers
    to record arrays at +132 (mcPropManagerData constructor). City has
    a pointer at +16, so they cannot be confused."""
    n = len(data)
    base = 0x80
    if base + 248 > n:
        return False
    if ok(struct.unpack_from('<I', data, base + 16)[0], n):
        return False
    counts = struct.unpack_from('<29I', data, base + 16)
    arrays = struct.unpack_from('<29I', data, base + 132)
    nz = [(c, a) for c, a in zip(counts, arrays) if c]
    return (len(nz) >= 2 and all(c < 100000 for c in counts)
            and all(ok(a, n) for c, a in nz))


def extract_prop_models(data, outdir, pck_path=None, channel_order='rgba', glow_texture=None,
                        cpv_pal=None):
    """Props (ATMs, benches, lamps, trees, signs) with real
    geometry, textures and placement -- through the engine chain:

      mcPropManagerData (0x80): +16 -- 29 counters, +132 -- 29 arrays
          of record pointers (categories = prop classes);
          +12 -> shader group (one, like City: +4 array, +8 size)
      record +0x0c -> prop type: +8 name, +160 radius, +164 centre,
          +180 model (rmcModelGeom, like City: chunks, shader indices,
          packets with a multiplier header and the ADC flag)
      placement -- two record layouts:
          explicit matrix: axes +0x30/+0x3c/+0x48, position +0x54;
          mcProp: position +0x14, Euler angles XYZ +0x20 (mcProp::GetMatrix
          -> Matrix34::FromEulersXYZ)
    Verified on Atlanta: a model at +180 for 95 of 111 types (the rest
    -- glow_* light sprites and invisible props), farthest vertex / type sphere radius --
    median 0.966; records with an explicit matrix are 100% upright, with Euler
    angles -- 90% (the rest are apparently wall lamps); a texture
    is found for 97% of chunks. Lighting: V4-8 palette index (see CPV_SCALE).
    File names are the same as for City (instance_placements.csv, obj/city.mtl),
    so the same Blender importer works unchanged; the sector column
    holds the category number."""
    n = len(data)
    base = 0x80
    os.makedirs(outdir, exist_ok=True)
    obj_dir = os.path.join(outdir, 'obj')
    tex_dir = os.path.join(outdir, 'textures')
    os.makedirs(obj_dir, exist_ok=True)
    counts = struct.unpack_from('<29I', data, base + 16)
    arrays = struct.unpack_from('<29I', data, base + 132)

    small = collect_packets(data, max_qw=2048)
    small.sort()
    pstarts = [t for t, _ in small]

    # --- textures: embedded in the props .pck ---
    tex0_file, tex0_set = {}, set()
    try:
        import mc3_extract_textures as MT
        tex0_set = {ch[0]['off'] for ch in MT.collect_descriptors(data)}
        if pck_path:
            os.makedirs(tex_dir, exist_ok=True)
            tman = MT.extract(pck_path, None, tex_dir, base_only=True,
                              channel_order=channel_order)
            canon = {}
            for m in tman:
                if 'tex0_off' not in m or m.get('mip_level', 0) != 0:
                    continue
                h = hashlib.md5(open(os.path.join(tex_dir, m['file']), 'rb').read()).hexdigest()
                tex0_file[m['tex0_off']] = canon.setdefault(h, m['file'])
            print(f"  textures extracted: {len(tex0_file)}")
    except Exception as ex:
        print(f"  textures not extracted: {ex}")

    def resolve_tex(tx):
        for _ in range(4):
            if tx is None or tx + 16 > n:
                return None
            if data[tx + 4] == 0 and (tx + 16) in tex0_set:
                return tx + 16
            if data[tx + 4] == 2:
                p = struct.unpack_from('<I', data, tx + 12)[0]
                tx = fo(p) if ok(p, n) else None
                continue
            return None
        return None

    gp = struct.unpack_from('<I', data, base + 12)[0]
    sh_arr, sh_cnt = None, 0
    if ok(gp, n):
        a = struct.unpack_from('<I', data, fo(gp) + 4)[0]
        if ok(a, n):
            sh_arr, sh_cnt = fo(a), struct.unpack_from('<H', data, fo(gp) + 8)[0]

    def is_additive(si):
        """Flare shader: type 19 (mcShaderFlareTexScroll; templates city_flarebg,
        city_flarelightmap). Additive blending --
        black adds nothing; the texture's own alpha is solid, so
        transparency is taken from luminance. Light cards
        (a_prop_light_card_*) and smoke columns are drawn this way."""
        if sh_arr is None or si > sh_cnt:
            return False
        sh = struct.unpack_from('<I', data, sh_arr + 4 * si)[0]
        if not ok(sh, n):
            return False
        S = fo(sh)
        # by type only: searching the shader name in a byte window caught neighbouring
        # objects (a hydrant got the flare material)
        return (data[S + 4] & 0x7F) == 19

    add_cache = {}

    def additive_texture(tex):
        """A copy of the texture with alpha from luminance: <name>_add.png."""
        if tex in add_cache:
            return add_cache[tex]
        out = None
        try:
            from PIL import Image
            im = np.asarray(Image.open(os.path.join(tex_dir, tex)).convert('RGBA')).copy()
            lum = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]
            im[..., 3] = np.clip(lum * 1.6, 0, 255).astype(np.uint8)
            out = os.path.splitext(tex)[0] + '_add.png'
            Image.fromarray(im, 'RGBA').save(os.path.join(tex_dir, out))
        except Exception as ex:
            print(f"    could not make the flare texture from {tex}: {ex}")
        add_cache[tex] = out
        return out

    def chunk_texture(si):
        if sh_arr is None or si > sh_cnt:
            return None
        sh = struct.unpack_from('<I', data, sh_arr + 4 * si)[0]
        if not ok(sh, n):
            return None
        t0 = shader_texture_ptr(data, fo(sh), resolve_tex)
        return tex0_file.get(t0) if t0 is not None else None

    type_cache = {}

    def prop_type(t):
        if t in type_cache:
            return type_cache[t]
        res = None
        if t + 184 <= n:
            p = struct.unpack_from('<I', data, t + 8)[0]
            name = None
            if ok(p, n):
                s = data[fo(p):fo(p) + 48].split(b'\x00')[0]
                if 3 <= len(s) <= 47 and all(32 <= b < 127 for b in s):
                    name = s.decode()
            mp = struct.unpack_from('<I', data, t + 180)[0]
            if name and ok(mp, n):
                ch = read_model_geom(data, fo(mp), small, pstarts)
                if ch:
                    res = (name, ch)
        type_cache[t] = res
        return res

    def placement(r):
        """(3x3 axes as rows, position) or None."""
        if r + 0x60 > n:
            return None
        M = np.array([struct.unpack_from('<3f', data, r + o) for o in (0x30, 0x3c, 0x48)])
        if np.all(np.isfinite(M)) and np.allclose(M @ M.T, np.eye(3), atol=0.05) \
                and abs(np.linalg.det(M) - 1) < 0.05:
            return M, np.array(struct.unpack_from('<3f', data, r + 0x54))
        ang = np.array(struct.unpack_from('<3f', data, r + 0x20))
        P = np.array(struct.unpack_from('<3f', data, r + 0x14))
        if np.all(np.isfinite(ang)) and np.all(np.abs(ang) <= 3.2) \
                and np.all(np.isfinite(P)) and np.all(np.abs(P) < 1e5):
            return euler_xyz_rows(*ang), P
        return None

    written, used_tex, rows = {}, set(), []
    mat_kinds = set()
    glow_quads = []          # (position, size, colour) for light sprites
    alpha_frac = {}          # share of transparent pixels in a texture

    def is_cutout(tex):
        """Does the texture have an alpha cutout. Props have no slots like City,
        so it is decided by the picture itself: 1% to 95% transparent pixels
        -- a cutout (foliage, grates, signs with a transparent background)."""
        if tex not in alpha_frac:
            frac = 0.0
            try:
                from PIL import Image
                import numpy as _np
                a8 = _np.asarray(Image.open(os.path.join(tex_dir, tex)).convert('RGBA'))[..., 3]
                frac = float((a8 < 128).mean())
            except Exception:
                frac = 0.0
            alpha_frac[tex] = frac
        f = alpha_frac[tex]
        return 0.01 < f < 0.95

    GLOW_COLORS = {'ylw': (1.0, 0.85, 0.35), 'yel': (1.0, 0.85, 0.35),
                   'org': (1.0, 0.55, 0.15), 'red': (1.0, 0.25, 0.2),
                   'blu': (0.35, 0.6, 1.0), 'blue': (0.35, 0.6, 1.0),
                   'grn': (0.4, 1.0, 0.45), 'ppl': (0.75, 0.45, 1.0),
                   'purp': (0.75, 0.45, 1.0), 'halide': (0.85, 0.95, 1.0),
                   'wht': (1.0, 1.0, 1.0)}

    def glow_color(name):
        for k, c in GLOW_COLORS.items():
            if '_' + k in name:
                return k, c
        return 'wht', GLOW_COLORS['wht']

    def glow_placement(r):
        """Position of a light sprite (mcPropLight): the record itself (32 bytes)
        does not have it; the pointer at +0x14 leads to an object whose first three
        floats are the world coordinates."""
        q = struct.unpack_from('<I', data, r + 0x14)[0]
        if not ok(q, n):
            return None
        P = np.array(struct.unpack_from('<3f', data, fo(q)))
        if not np.all(np.isfinite(P)) or np.all(P == 0) or np.max(np.abs(P)) > 1e5:
            return None
        return P
    stat = {'chunks': 0, 'textured': 0, 'no_model': 0, 'no_placement': 0,
            'dupes_removed': 0, 'glow_sprites': 0, 'flares': 0, 'v48_verts': 0}
    for cat in range(29):
        if not counts[cat]:
            continue
        arr = fo(arrays[cat])
        for k in range(counts[cat]):
            rp = struct.unpack_from('<I', data, arr + 4 * k)[0]
            if not ok(rp, n):
                continue
            r = fo(rp)
            tp = struct.unpack_from('<I', data, r + 0x0c)[0]
            pt = prop_type(fo(tp)) if ok(tp, n) else None
            if pt is None:
                # light sprites (glow_*): no geometry, but a position.
                # Category 0 only (mcPropLight): props of other categories without
                # a model are invisible (breakable windows over glass from the
                # city geometry, a particle fountain, a planter trigger); their
                # pointer at +0x14 may be valid too, but it is not a light.
                P = glow_placement(r) if cat == 0 else None
                if P is not None and ok(tp, n):
                    t = fo(tp)
                    nm_ptr = struct.unpack_from('<I', data, t + 8)[0]
                    gname = ''
                    if ok(nm_ptr, n):
                        sname = data[fo(nm_ptr):fo(nm_ptr) + 48].split(b'\x00')[0]
                        if sname:
                            gname = sname.decode(errors='replace')
                    size = struct.unpack_from('<f', data, t + 160)[0]
                    if not (0.1 < size < 100):
                        size = 1.0
                    glow_quads.append((P, size, glow_color(gname)))
                    stat['glow_sprites'] += 1
                else:
                    stat['no_model'] += 1
                continue
            pl = placement(r)
            if pl is None:
                stat['no_placement'] += 1
                continue
            name, ch = pt
            safe = re.sub(r'[^A-Za-z0-9_.#-]', '_', name)
            fn = f"{safe}.obj"
            if fn not in written:
                seen_faces = set()
                with open(os.path.join(obj_dir, fn), 'w') as fh:
                    fh.write(f"# {name} -- prop, LOCAL coordinates; placement: "
                             f"instance_placements.csv (world = local @ [a;b;c] + position)\n")
                    fh.write("mtllib city.mtl\n")
                    voff = 0
                    for ci, (si, v, u, f) in enumerate(ch):
                        tex = chunk_texture(si)
                        kind = ''
                        stat['chunks'] += 1
                        fh.write(f"g c{ci}\n")
                        if tex:
                            stat['textured'] += 1
                            used_tex.add(tex)
                            # alpha cutout -- for props with '_alpha_' in the name
                            # (the game's convention: a_prop_alpha_tree_01x etc.)
                            kind = '_alpha' if ('_alpha_' in name or is_cutout(tex)) else ''
                            if is_additive(si):
                                at = additive_texture(tex)
                                if at:
                                    used_tex.add(at)
                                    tex, kind = at, '_add'
                                    stat['flares'] += 1
                            mat_kinds.add((tex, kind))
                            fh.write(f"usemtl {os.path.splitext(tex)[0]}{kind}\n")
                        else:
                            fh.write("usemtl no_texture\n")
                        # baked lighting from V4-8 (84% of prop vertices;
                        # props have no CPV sets of their own)
                        if cpv_pal is not None and u.shape[1] >= 6 and np.all(u[:, 5] >= 0):
                            # V4-8: the 4th byte is an index into the city
                            # lighting palette (props have none of their own; the game uses
                            # the current one, rmcCpvPalette::sm_Current)
                            cc_ = CPV_SCALE * cpv_pal[u[:, 5].astype(int).clip(0, 255)]
                            for p, c3 in zip(v, cc_):
                                fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c3[0]:.3f} {c3[1]:.3f} {c3[2]:.3f}\n")
                            stat['v48_verts'] += len(v)
                        else:
                            for p in v:
                                fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
                        for q in u:
                            # for props V is NOT flipped (unlike the city):
                            # with the flip the top of the picture matched the top in the world
                            # on only 32% of vertical faces; the direct check -- the
                            # nitro cola sign stood upside down
                            fh.write(f"vt {q[0]:.4f} {q[1]:.4f}\n")
                        mat = (os.path.splitext(tex)[0] + kind) if tex else 'no_texture'
                        for a, b, c in f:
                            key = face_key(mat, v, a, b, c)
                            if key in seen_faces:
                                stat['dupes_removed'] += 1
                                continue
                            seen_faces.add(key)
                            fh.write(f"f {a+voff+1}/{a+voff+1} {b+voff+1}/{b+voff+1} "
                                     f"{c+voff+1}/{c+voff+1}\n")
                        voff += len(v)
                written[fn] = True
            M, P = pl
            rows.append([name, fn, cat, *np.round(P, 3).tolist(),
                         *np.round(M[0], 5).tolist(), *np.round(M[1], 5).tolist(),
                         *np.round(M[2], 5).tolist()])

    # --- light sprites: crossed quads, one object, world coordinates ---
    glow_mats = set()
    if glow_quads:
        # Glow texture. The game's picture is shared (not in the city resources),
        # but its look is known from a PCSX2 VRAM dump: a soft round spot,
        # brightness ~ exp(-(r/0.35)^2) from the centre (r = 1 at the edge), alpha
        # full everywhere -- in the game the glow is added to the picture. Here -- the same
        # shape, alpha from brightness (for Blender), the tint from the type name.
        try:
            from PIL import Image
            os.makedirs(tex_dir, exist_ok=True)
            if glow_texture and os.path.isfile(glow_texture):
                # a real picture (e.g. from a PCSX2 VRAM dump):
                # brightness is normalised, alpha from brightness
                src = np.asarray(Image.open(glow_texture).convert('RGB')).astype(np.float64).mean(2)
                g = np.clip(255 * src / max(src.max(), 1e-6), 0, 255).astype(np.uint8)
                print(f"  glow texture: {os.path.basename(glow_texture)}")
            else:
                yy, xx = np.mgrid[0:128, 0:128]
                rr = np.hypot(xx - 63.5, yy - 63.5) / 64.0
                g = np.clip(255 * np.exp(-(rr / 0.35) ** 2), 0, 255).astype(np.uint8)
            Image.fromarray(np.dstack([g, g, g, g]), 'RGBA').save(os.path.join(tex_dir, 'glow.png'))
            used_tex.add('glow.png')
        except Exception as ex:
            print(f"  glow texture not created: {ex}")
        with open(os.path.join(obj_dir, 'block_glows.obj'), 'w') as fh:
            fh.write("# light sprites (glow_*): no geometry, only\n"
                     "# a position and a colour from the name. Here -- crossed quads in WORLD\n"
                     "# coordinates, as one object (there are over a thousand).\n")
            fh.write("mtllib city.mtl\n")
            voff = 0
            for (P, size, (cname, col)) in glow_quads:
                mat = 'glow_' + cname
                glow_mats.add((cname, col))
                fh.write(f"g {mat}\nusemtl {mat}\n")
                h = size
                for dx, dz in ((1, 0), (0, 1)):      # two crossed quads
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        fh.write(f"v {P[0]+sx*h*dx:.4f} {P[1]+sy*h:.4f} {P[2]+sx*h*dz:.4f}\n")
                        fh.write(f"vt {(sx+1)/2:.1f} {(sy+1)/2:.1f}\n")
                for q in range(2):
                    b = voff + 4 * q
                    fh.write(f"f {b+1}/{b+1} {b+2}/{b+2} {b+3}/{b+3}\n"
                             f"f {b+1}/{b+1} {b+3}/{b+3} {b+4}/{b+4}\n")
                voff += 8
        rows.append(['glows', 'block_glows.obj', 0, 0.0, 0.0, 0.0,
                     1, 0, 0, 0, 1, 0, 0, 0, 1])

    with open(os.path.join(obj_dir, 'city.mtl'), 'w') as f:
        f.write("# prop materials -- through the engine chain (shader group -> texture)\n")
        f.write("newmtl no_texture\nKd 1 1 1\n")   # no texture (incl. a reference to the texture 'none'): colour from the vertex colours

        for tex, kind in sorted(mat_kinds):
            f.write(f"\nnewmtl {os.path.splitext(tex)[0]}{kind}\nKd 1 1 1\nmap_Kd ../textures/{tex}\n")
            if kind == '_alpha':
                f.write(f"map_d ../textures/{tex}\n")
            elif kind == '_add':
                # flare: emissive and transparent where the picture is dark
                f.write(f"map_d ../textures/{tex}\nKe 1 1 1\nmap_Ke ../textures/{tex}\n")
        for cname, col in sorted(glow_mats):
            # emission without a texture: the glow pictures are in the game's shared files,
            # the colour comes from the type name (ylw/org/blu/grn/red/ppl/halide)
            f.write(f"\nnewmtl glow_{cname}\nKd {col[0]:.3f} {col[1]:.3f} {col[2]:.3f}\n"
                    f"Ke {col[0]:.3f} {col[1]:.3f} {col[2]:.3f}\n"
                    f"map_Kd ../textures/glow.png\nmap_Ke ../textures/glow.png\n"
                    f"map_d ../textures/glow.png\n")
    if os.path.isdir(tex_dir):
        for fn in os.listdir(tex_dir):
            if fn not in used_tex:
                os.remove(os.path.join(tex_dir, fn))
    with open(os.path.join(outdir, 'instance_placements.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'obj', 'sector', 'x', 'y', 'z', 'ax', 'ay', 'az',
                    'bx', 'by', 'bz', 'cx', 'cy', 'cz'])
        w.writerows(rows)
    pct = 100 * stat['textured'] / max(stat['chunks'], 1)
    print(f"  placements: {len(rows)}, types with geometry: {len(written)}; "
          f"light sprites: {stat['glow_sprites']}, invisible (no model): {stat['no_model']}, "
          f"without placement: {stat['no_placement']}")
    print(f"  chunks: {stat['chunks']}, textured: {stat['textured']} ({pct:.1f}%), "
          f"textures: {len(used_tex)} -> textures/, obj/city.mtl; "
          f"duplicate faces removed: {stat['dupes_removed']}; "
          f"flare chunks (alpha from luminance): {stat['flares']}; "
          f"vertices coloured by V4-8 palette index: {stat['v48_verts']}"
          + ("" if cpv_pal is not None else " (no city palette -- props without lighting)"))
    return rows


PH_BOUND_TYPES = {0: 'sphere', 1: 'capsule', 2: 'box', 3: 'mesh',
                  9: 'octree', 10: 'quadtree', 11: 'octree grid',
                  13: 'road', 16: 'composite'}


def extract_physics(data, outdir):
    """City collision geometry (<city>_bnd.pck, resource 0x09).

    From the game code (mcLayerCity::InitPhys -> mcPhysics -> mcLevelBounds):
      root mcPhysics (0x80): +12 -> mcLevelBounds
      mcLevelBounds: +0, +4 -> physics instances phInst; +8/+12 an array
        of 64-byte instances and their count; +16 -> CurveMgr
      phInst: +4 -> archetype, the archetype's +12 -> bound phBound;
        matrix at +16..+63 (identity in Atlanta -- vertices are already world)
      phBound: byte +4 -- type (phBound::VirtualConstructFromPtr);
        for polyhedra (mesh, octree, quadtree):
        +8..+28 extents, +76 vertices, +80 polygons,
        +104 -> vertices (float3, 12 bytes), +108 -> polygons (32 bytes):
          +0 normal float3, +12 area, +16 four u16 vertex indices
          (the fourth = 0 -- a triangle), +24 four u16 neighbours
    Atlanta: a quadtree (ground, 98 polygons) and an octree (buildings and
    the rest, 21,617 polygons). Surface names are not stored in the file
    (+112 -- a table of numbers for the global material manager)."""
    n = len(data)
    base = 0x80
    obj_dir = os.path.join(outdir, 'obj')
    os.makedirs(obj_dir, exist_ok=True)
    lbp = struct.unpack_from('<I', data, base + 12)[0]
    if not ok(lbp, n):
        print("  mcLevelBounds not found")
        return None
    lb = fo(lbp)
    insts = []
    for k in (0, 4):
        p = struct.unpack_from('<I', data, lb + k)[0]
        if ok(p, n):
            insts.append(fo(p))
    ap, ac = struct.unpack_from('<II', data, lb + 8)
    if ok(ap, n) and 0 < ac < 100000:
        insts += [fo(ap) + 64 * i for i in range(ac)]
    rows, total = [], 0
    for ii, inst in enumerate(insts):
        arch = struct.unpack_from('<I', data, inst + 4)[0]
        if not ok(arch, n):
            continue
        bp = struct.unpack_from('<I', data, fo(arch) + 12)[0]
        if not ok(bp, n):
            continue
        b = fo(bp)
        typ = data[b + 4]
        if typ not in (3, 9, 10, 11):
            print(f"  bound {ii}: type {typ} ({PH_BOUND_TYPES.get(typ, '?')}) -- not exported")
            continue
        nv, npoly = struct.unpack_from('<II', data, b + 76)
        vp, pp = struct.unpack_from('<II', data, b + 104)
        if not (ok(vp, n) and ok(pp, n)) or not (0 < nv < 10 ** 6) or not (0 < npoly < 10 ** 6):
            print(f"  bound {ii}: data unreadable")
            continue
        V = np.frombuffer(data[fo(vp):fo(vp) + nv * 12], dtype='<f4').reshape(-1, 3).astype(np.float64)
        M = np.array(struct.unpack_from('<12f', data, inst + 16)).reshape(4, 3)
        if np.all(np.isfinite(M)) and abs(np.linalg.det(M[:3])) > 1e-6:
            V = V @ M[:3] + M[3]
        name = f"collision_{ii}_{PH_BOUND_TYPES.get(typ, typ)}"
        fname = f"block_{name}.obj"
        nt = 0
        with open(os.path.join(obj_dir, fname), 'w') as fh:
            fh.write(f"# collision: {PH_BOUND_TYPES.get(typ)}, {nv} vertices, {npoly} polygons, "
                     f"WORLD coordinates\nmtllib city.mtl\nusemtl collision\n")
            for x, y, z in V:
                fh.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
            for i in range(npoly):
                a, bb, c, dd = struct.unpack_from('<4H', data, fo(pp) + 32 * i + 16)
                if max(a, bb, c) >= nv:
                    continue
                fh.write(f"f {a+1} {bb+1} {c+1}\n"); nt += 1
                # the fourth index 0 is empty: a triangular polygon (Atlanta:
                # 9721 of 21,617); otherwise a quad
                if dd != 0 and dd != a and dd < nv and dd not in (bb, c):
                    fh.write(f"f {a+1} {c+1} {dd+1}\n"); nt += 1
        total += nt
        rows.append([name, fname, 0, 0.0, 0.0, 0.0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        print(f"  bound {ii}: {PH_BOUND_TYPES.get(typ)} -- {nv} vertices, {npoly} polygons -> {nt} triangles")
    with open(os.path.join(obj_dir, 'city.mtl'), 'w') as f:
        f.write("# collision geometry\nnewmtl collision\nKd 0.9 0.35 0.1\nd 0.5\n")
    with open(os.path.join(outdir, 'instance_placements.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'obj', 'sector', 'x', 'y', 'z', 'ax', 'ay', 'az',
                    'bx', 'by', 'bz', 'cx', 'cy', 'cz'])
        w.writerows(rows)
    print(f"  collision: {len(rows)} bounds, {total} triangles -> {obj_dir}")
    return rows


def find_physics_for_city(city_pck):
    """<city>_bnd.pck next to the city file: one physics file per city
    ($/resources/city/%s_bnd in mcLayerCity::InitPhys)."""
    folder = os.path.dirname(os.path.abspath(city_pck))
    city = os.path.basename(city_pck).split('_')[0]
    p = os.path.join(folder, city + '_bnd.pck')
    return p if os.path.isfile(p) else None


def extract_fog(data, outdir):
    """Particle fog (<city>_<time>_<weather>_fog.pck, resource type 0x04,
    mcParticleFogMgr): 49 clouds in Atlanta, ~60 particles each.

    Manager: +0 -> textures (+4 u16 count; one, by the name
    fx_particlefog_particle -- not in the file), +8 -> cloud array, +16 count.
    Cloud (mcParticleFog, parameters mcParticleFogTune): +12 m_alpha,
    +20 m_particleSize, +32 m_additive, +34 m_litFog, +36 m_fogTint (RGB),
    +48 m_emissive, +52/+56 m_fogNear/m_fogFar, +172 particle count,
    +180 -> 24-byte particles (xyz position in world coordinates + runtime
    fields). Clouds are named after places (atlanta/midnight_cloudy/dt_road_04a);
    the height depends on the weather. Export: two crossed cards per particle, a soft spot with the cloud opacity in alpha."""
    n = len(data)
    fo_ = lambda va: va - BASE + HDR
    ok_ = lambda va: BASE <= va and 0 <= fo_(va) < n
    M = 0x80
    arrp = struct.unpack_from('<I', data, M + 8)[0]
    cnt = struct.unpack_from('<I', data, M + 16)[0]
    if not ok_(arrp) or not (0 < cnt < 10000):
        print("  fog: structure not recognised")
        return
    obj_dir = os.path.join(outdir, 'obj')
    tex_dir = os.path.join(outdir, 'textures')
    os.makedirs(obj_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    clouds = []
    for i in range(cnt):
        cp = struct.unpack_from('<I', data, fo_(arrp) + 4 * i)[0]
        if not ok_(cp):
            continue
        F = fo_(cp)
        alpha = struct.unpack_from('<f', data, F + 12)[0]
        size = struct.unpack_from('<f', data, F + 20)[0]
        tint = struct.unpack_from('<3f', data, F + 36)
        emis = struct.unpack_from('<f', data, F + 48)[0]
        npart = struct.unpack_from('<I', data, F + 172)[0]
        pp = struct.unpack_from('<I', data, F + 180)[0]
        if not ok_(pp) or not (0 < npart < 10000):
            continue
        P = [struct.unpack_from('<3f', data, fo_(pp) + 24 * k) for k in range(npart)]
        clouds.append((alpha, size, tint, emis, P))
    # materials by unique parameters (usually one)
    mats = {}
    for a_, s_, t_, e_, P in clouds:
        key = (round(a_, 3), tuple(round(x, 3) for x in t_), round(e_, 3))
        mats.setdefault(key, f"fog_{len(mats)}")
    try:
        from PIL import Image
        for (a_, t_, e_), nm in mats.items():
            S = 64
            yy, xx = np.mgrid[0:S, 0:S]
            r = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) / (S / 2)
            puff = np.clip(1 - r, 0, 1) ** 1.5          # a soft spot (the real texture is not available)
            im = np.zeros((S, S, 4), dtype=np.uint8)
            im[..., :3] = 255
            im[..., 3] = np.clip(puff * a_ * 255, 0, 255).astype(np.uint8)
            Image.fromarray(im, 'RGBA').save(os.path.join(tex_dir, nm + '.png'))
    except Exception as ex:
        print(f"  fog texture not created: {ex}")
    with open(os.path.join(obj_dir, 'city.mtl'), 'w') as f:
        for (a_, t_, e_), nm in mats.items():
            f.write(f"newmtl {nm}\nKd {t_[0]:.3f} {t_[1]:.3f} {t_[2]:.3f}\n"
                    f"Ke {t_[0]*e_:.3f} {t_[1]*e_:.3f} {t_[2]*e_:.3f}\n"
                    f"map_Kd ../textures/{nm}.png\nmap_Ke ../textures/{nm}.png\n"
                    f"map_d ../textures/{nm}.png\n\n")
    npt = 0
    with open(os.path.join(obj_dir, 'block_fog.obj'), 'w') as fh:
        fh.write("# particle fog (mcParticleFogMgr), game world coordinates\nmtllib city.mtl\n")
        voff = 0
        for a_, s_, t_, e_, P in clouds:
            nm = mats[(round(a_, 3), tuple(round(x, 3) for x in t_), round(e_, 3))]
            fh.write(f"usemtl {nm}\n")
            h = s_ / 2
            for (x, y, z) in P:
                for dx, dz in ((1, 0), (0, 1)):
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        fh.write(f"v {x+sx*h*dx:.3f} {y+sy*h:.3f} {z+sx*h*dz:.3f}\n")
                        fh.write(f"vt {(sx+1)/2:.1f} {(sy+1)/2:.1f}\n")
                for q in range(2):
                    b = voff + 4 * q
                    fh.write(f"f {b+1}/{b+1} {b+2}/{b+2} {b+3}/{b+3}\nf {b+1}/{b+1} {b+3}/{b+3} {b+4}/{b+4}\n")
                voff += 8
                npt += 1
    with open(os.path.join(outdir, 'instance_placements.csv'), 'w') as f:
        f.write("name,obj,sector,x,y,z,ax,ay,az,bx,by,bz,cx,cy,cz\n")
        f.write("fog,block_fog.obj,0,0,0,0,1,0,0,0,1,0,0,0,1\n")
    print(f"  fog: clouds {len(clouds)}, particles {npt}, materials {len(mats)} -> {obj_dir}")


def find_fog_for_city(city_pck):
    """<city>_<time>_<weather>_fog.pck next to it (mcParticleFogMgr loads
    $/resources/city/%s_%s_%s_fog)."""
    c = os.path.splitext(os.path.abspath(city_pck))[0] + '_fog.pck'
    return c if os.path.isfile(c) else None


def load_city_palette(path):
    """Lighting palette (256 x RGB float) from the city .pck (mcCity+44)."""
    try:
        d = open(path, 'rb').read()
        pp = struct.unpack_from('<I', d, 0x80 + 44)[0]
        a = pp - BASE + HDR
        if 0 <= a and a + 4096 <= len(d):
            return np.frombuffer(d[a:a + 4096], dtype='<f4').reshape(256, 4)[:, :3].astype(np.float64)
    except Exception:
        pass
    return None


def city_pck_for_props(props_path):
    """City .pck of the same time of day: <city>_<time>_<weather>_props.pck ->
    <city>_<time>_<weather>.pck next to it."""
    b = os.path.basename(props_path)
    if b.endswith('_props.pck'):
        c = os.path.join(os.path.dirname(props_path), b[:-len('_props.pck')] + '.pck')
        if os.path.isfile(c):
            return c
    return None


def find_props_for_city(city_pck):
    """The props file for a city file. The game loads props from
    $/resources/prop/<city>_<time>_<weather>_props (mcPropManager::Load),
    so the exact name <city_name>_props.pck next to the city .pck is looked
    up first; if absent -- any <city>_*_props.pck in the same folder,
    preferring the same time of day. Returns a path or None."""
    import glob
    stem = os.path.splitext(city_pck)[0]
    exact = stem + '_props.pck'
    if os.path.isfile(exact):
        return exact
    folder = os.path.dirname(os.path.abspath(city_pck))
    parts = os.path.basename(stem).split('_')
    cands = sorted(glob.glob(os.path.join(folder, parts[0] + '_*_props.pck')))
    if not cands:
        return None
    if len(parts) > 1:
        same_time = [c for c in cands if os.path.basename(c).split('_')[1:2] == parts[1:2]]
        if same_time:
            return same_time[0]
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='.pck file')
    ap.add_argument('outdir', nargs='?', default=None)
    ap.add_argument('--ppf', default=None,
                     help='City: the paired .ppf with texture pages (by default '
                          'the .ppf with the same name next to the .pck). Without it the geometry '
                          'is written without textures')
    ap.add_argument('--with-reflect', action='store_true',
                     help='City: also export the reflect slot (upside-down simplified '
                          'geometry below the ground for wet-road reflections)')
    ap.add_argument('--props', nargs='?', const='auto', default=None, metavar='PATH',
                     help='City: also extract props (ATMs, benches, lamps, '
                          'trees) into props/ -- the Blender script places them '
                          'together with the city. Without a path the file is looked up next to the city .pck '
                          '(<city>_<time>_<weather>_props.pck)')
    ap.add_argument('--glow-texture', default=None, metavar='PNG',
                     help='props: a picture for the light sprites instead of the generated one '
                          '(it is not in the city files; e.g. a PCSX2 VRAM dump -- '
                          'a soft radial spot)')
    ap.add_argument('--shared-lighting', action='store_true',
                     help='City: one baked lighting per object type (set 0) '
                          'instead of a separate one for every copy -- fewer files, but all '
                          'copies repeat the lighting of the first')
    ap.add_argument('--fog', nargs='?', const='auto', default=None, metavar='PATH',
                     help='City: low particle fog (<city>_<time>_<weather>_fog.pck; '
                          'without a path -- looked up next to the .pck) -> fog/ subfolder')
    ap.add_argument('--physics', nargs='?', const='auto', default=None, metavar='PATH',
                     help='City: also extract the collision geometry into '
                          'physics/ (without a path <city>_bnd.pck next to it is used)')
    ap.add_argument('-c', '--channels', choices=['rgba', 'bgra', 'auto'], default='rgba',
                     help='texture palette channel order. rgba (default) -- for all '
                          'files (verified with PCSX2 dumps for Atlanta and Tokyo); bgra -- '
                          'only if red and blue are swapped; auto -- same as rgba')
    args = ap.parse_args()

    outdir = args.outdir or os.path.splitext(args.src)[0] + '_models'
    os.makedirs(outdir, exist_ok=True)
    data = open(args.src, 'rb').read()

    is_city = False
    if len(data) > 0x80 + 28:
        p = struct.unpack_from('<I', data, 0x80 + 24)[0]
        cnt = struct.unpack_from('<I', data, 0x80 + 20)[0]
        is_city = ok(p, len(data)) and 0 < cnt < 100

    if len(data) >= 8 and struct.unpack_from('<I', data, 4)[0] == 4:
        print(f"{os.path.basename(args.src)}: particle fog (type 0x04)")
        print("mode 5: fog")
        extract_fog(data, outdir)
    elif len(data) >= 8 and struct.unpack_from('<I', data, 4)[0] == 9:
        print(f"{os.path.basename(args.src)}: physics resource (type 0x09)")
        print("mode 4: collision geometry")
        extract_physics(data, outdir)
    elif looks_like_props(data):
        print(f"{os.path.basename(args.src)}: looks like props (mcPropManagerData found)")
        print("mode 3: props with geometry, textures and placement through the engine chain")
        cp_ = city_pck_for_props(args.src)
        if cp_:
            print(f"  lighting palette -- from {os.path.basename(cp_)}")
        else:
            print("  no city .pck of the same time of day next to it -- props without lighting")
        extract_prop_models(data, outdir, pck_path=args.src,
                            channel_order='bgra' if args.channels == 'bgra' else 'rgba',
                            glow_texture=args.glow_texture,
                            cpv_pal=load_city_palette(cp_) if cp_ else None)
    elif is_city:
        print(f"{os.path.basename(args.src)}: looks like City geometry (mcCity/mcHood found)")
        print("mode 2: city objects with textures through the engine chain")
        ppf = args.ppf or os.path.splitext(args.src)[0] + '.ppf'
        extract_city_models(data, outdir, pck_path=args.src, ppf_path=ppf,
                             channel_order='bgra' if args.channels == 'bgra' else 'rgba',
                             include_reflect=args.with_reflect,
                             per_instance_lighting=not args.shared_lighting)
        if args.fog is not None:
            fg = find_fog_for_city(args.src) if args.fog == 'auto' else args.fog
            if fg and os.path.isfile(fg):
                print(f"\nfog: {os.path.basename(fg)} -> {os.path.join(outdir, 'fog')}")
                extract_fog(open(fg, 'rb').read(), os.path.join(outdir, 'fog'))
            else:
                print(f"\nfog: file not found ({fg or '<city>_<time>_<weather>_fog.pck next to it'}) -- give the path: --fog file")
        if args.physics is not None:
            ph = find_physics_for_city(args.src) if args.physics == 'auto' else args.physics
            if ph and os.path.isfile(ph):
                print(f"\nphysics: {os.path.basename(ph)} -> {os.path.join(outdir, 'physics')}")
                extract_physics(open(ph, 'rb').read(), os.path.join(outdir, 'physics'))
            else:
                print(f"\nphysics: file not found ({ph or '<city>_bnd.pck next to it'}) -- give the path: --physics file")
        if args.props is not None:
            props_path = find_props_for_city(args.src) if args.props == 'auto' else args.props
            if not props_path or not os.path.isfile(props_path):
                print(f"\nprops: file not found ({props_path or 'no <city>_*_props.pck next to it'})"
                      f" -- give the path: --props file_props.pck")
            else:
                pdata = open(props_path, 'rb').read()
                if not looks_like_props(pdata):
                    print(f"\nprops: {os.path.basename(props_path)} does not look like a props file -- skipped")
                else:
                    print(f"\nprops: {os.path.basename(props_path)} -> {os.path.join(outdir, 'props')}")
                    extract_prop_models(pdata, os.path.join(outdir, 'props'), pck_path=props_path,
                                        channel_order='bgra' if args.channels == 'bgra' else 'rgba',
                                        glow_texture=args.glow_texture,
                                        cpv_pal=load_city_palette(args.src))
    else:
        print(f"{os.path.basename(args.src)}: embedded models")
        print("mode 1: embedded models (textures under object names + matches.txt)")
        extract_embedded_models(data, outdir)

    print(f"\ndone -> {outdir}")


if __name__ == '__main__':
    main()
