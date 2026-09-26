#!/usr/bin/env python3
"""
mc3_extract_textures.py -- texture extraction for Midnight Club 3: DUB Edition
(PS2) from .pck and .ppf files. Used by mc3_extract_models.py; also works on
its own.

    python3 mc3_extract_textures.py <file.pck | file.ppf> [output_folder]
        [-b] [--dedupe] [-c rgba|bgra]

The script detects the file type and mode itself:

  * .pck with embedded textures (cars, pedestrians, traffic, props, UI)
        textures live inside the .pck itself and are addressed by pointers.
        One file is enough.

  * city .pck + .ppf pages (e.g. atlanta_midnight_cloudy)
        texture descriptors are in the .pck, pixels in the .ppf. The pair file
        is looked up next to it by the same name; either one may be given.
        Each weather has its own .ppf (they differ in ~90 road textures).

  * .ppf without .pck
        not supported: a .ppf holds only pixels; sizes, format and addresses
        of textures are stored in the .pck. The script reports this.

FORMAT (derived from the game code SLUS_123.45, verified with PCSX2 dumps;
full specification: MC3_FORMATS.md)

.pck header: 128 bytes, dword[0] = pointer base (0x06800000), dword[1] =
resource type, dword[2] = 1, dword[3] = data size. File offset of a pointer =
ptr - base + 128. A pointer is valid if it lands inside the file (no 8 MB
limit: the Tokyo file is 10.2 MB).

.ppf header: "pf05", count, page size (69632 = 34 sectors); then count
4-byte entries: start sector = val & 0xFFFFF, sectors = val >> 20. A page is
a contiguous chain of memory blocks from offset 0; block size =
(dword +12 >> 5) * 32 bytes, data starts 144 bytes (0x90) after the header.

Texture object rmcTexturePS2: +16 the first 16-byte level descriptor
(upload rectangle log2 W/H in bits 0-4/5-9 and upload PSM in bits 10-15,
block count, TEX0), +72 location entries per level (12 bytes: pointer in the
.pck | 0, offset in page, (dictionary << 23) | page), +134 level count.

Swizzling: data is writeTexPSMCT8/4 then read as a smaller PSMCT32 (8bpp:
w/2 x h/2; 4bpp: w==h -> h/2 x h/4, else w/4 x h/2). Small levels
(stride*height < 256) are stored linearly in their own format.

Palette: PSMCT32 RGBA (verified pixel-exact with PCSX2 VRAM dumps for both
Atlanta and Tokyo), alpha 0-128 (multiplied by 2). 256-colour CLUTs use the
CSM1 permutation (bits 3 and 4 of the index swapped); 16-colour CLUTs are
linear. The palette follows the data of the last level of the chain.

Textures are stored bottom-up and flipped on output.
"""
import json
import re
import os
import struct
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

PTR_BASE = 0x6800000
PCK_HDR = 0x80
ALLOC_HDR = 144
SECTOR = 2048

# --- GS addressing tables (from the ELF SLUS_123.45, segment vaddr 0x1a0000 -> 0x1000) ---
BLOCK32 = [0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
           8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31]
COLUMNWORD32 = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
BLOCK4 = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15,
          16, 18, 24, 26, 17, 19, 25, 27, 20, 22, 28, 30, 21, 23, 29, 31]
COLUMNWORD4 = [0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15, 2, 3, 6, 7, 10, 11, 14, 15]
COLUMNBYTE4 = ([0] * 8 + [2] * 8 + [4] * 8 + [6] * 8) * 2 + ([1] * 8 + [3] * 8 + [5] * 8 + [7] * 8) * 2


def _log2sh(v):
    n = 0
    v >>= 1
    while v:
        v >>= 1
        n += 1
    return n


def unswizzle8(buf, w, h):
    sw = np.frombuffer(buf[:w * h], dtype=np.uint8)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    bl = (y & ~0xf) * w + (x & ~0xf) * 2
    ss = (((y + 2) >> 2) & 1) * 4
    py = (((y & ~3) >> 1) + (y & 1)) & 7
    cl = py * w * 2 + ((x + ss) & 7) * 4
    bn = ((y >> 1) & 1) + ((x >> 2) & 2)
    return sw[np.clip(bl + cl + bn, 0, len(sw) - 1)].reshape(h, w)


_map4_cache = {}


def _gs_word32(x, y, dbw):
    v17 = y >> 5
    v18 = y - 32 * v17
    v21 = x >> 6
    v23 = x - (v21 << 6)
    v22 = 8 * (v18 >> 3)
    return (16 * ((v18 - v22) >> 1) + (BLOCK32[(v23 >> 3) + v22] << 6)
            + ((v21 + v17 * dbw) << 11)
            + COLUMNWORD32[8 * ((v18 - v22) & 1) + v23 - 8 * (v23 >> 3)])


def _build_map4(w, h):
    key = (w, h)
    if key in _map4_cache:
        return _map4_cache[key]
    dbw4 = max(1, (w + 63) >> 6)
    ow, oh = ((h >> 1, h >> 2) if w == h else (w >> 2, h >> 1))
    ow, oh = max(1, ow), max(1, oh)
    dbw32 = max(1, (ow + 63) >> 6)
    rev = {}
    for y32 in range(oh):
        for x32 in range(ow):
            rev[_gs_word32(x32, y32, dbw32)] = y32 * ow + x32
    bpos = np.full(w * h, -1, dtype=np.int64)
    nib = np.zeros(w * h, dtype=np.uint8)
    for y in range(h):
        v16 = y >> 7
        v17 = v16 * (dbw4 >> 1)
        v18 = y - (v16 << 7)
        v21 = v18 >> 4
        v23 = v18 - 16 * v21
        v27 = v23 >> 2
        for x in range(w):
            v22 = x >> 7
            v24 = x - (v22 << 7)
            v28 = v24 >> 5
            v29 = v24 - 32 * v28 + 32 * (v23 - 4 * v27)
            word = (16 * v27 + (BLOCK4[4 * v21 + v28] << 6)
                    + ((v22 + v17) << 11) + COLUMNWORD4[128 * (v27 & 1) + v29])
            v32 = COLUMNBYTE4[v29]
            gsb = 4 * word + (v32 >> 1)
            n = rev.get(gsb >> 2)
            if n is not None:
                bpos[y * w + x] = 4 * n + (gsb & 3)
                nib[y * w + x] = v32 & 1
    _map4_cache[key] = (bpos, nib)
    return bpos, nib


def unswizzle4(buf, w, h):
    bpos, nib = _build_map4(w, h)
    data = np.frombuffer(buf[:w * h // 2], dtype=np.uint8)
    safe = np.where((bpos >= 0) & (bpos < len(data)), bpos, 0)
    b = data[safe]
    out = np.where(nib == 1, b >> 4, b & 0xF).astype(np.uint8)
    out[bpos < 0] = 0
    return out.reshape(h, w)


def csm1(pal):
    """CLUT permutation for 256 colours: bits 3 and 4 of the index swapped."""
    if len(pal) != 256:
        return pal
    out = pal.copy()
    for i in range(256):
        out[(i & ~0x18) | ((i >> 1) & 8) | ((i << 1) & 0x10)] = pal[i]
    return out


def decode_indices(chunk, w, h, psm, swizzled):
    if not swizzled:
        raw = np.frombuffer(chunk, dtype=np.uint8)
        if psm == 19:
            return raw[:w * h].reshape(h, w)
        px = np.zeros(w * h, dtype=np.uint8)
        half = raw[:w * h // 2]
        px[0::2] = half & 0xF
        px[1::2] = half >> 4
        return px.reshape(h, w)
    return unswizzle8(chunk, w, h) if psm == 19 else unswizzle4(chunk, w, h)


# ----------------------------------------------------------------------------

def read_pck_header(data):
    if len(data) < PCK_HDR:
        return None
    base, rtype, magic, size = struct.unpack_from('<IIII', data, 0)
    if magic != 1 or not (0x06000000 <= base <= 0x08000000):
        return None
    return {'base': base, 'type': rtype, 'size': size}


def is_ppf(data):
    return data[:4] == b'pf05'


def load_pages(ppf):
    count = struct.unpack_from('<I', ppf, 4)[0]
    pages = []
    for i in range(count):
        v = struct.unpack_from('<I', ppf, 12 + i * 4)[0]
        pages.append(((v & 0x7FFFF) * SECTOR, (v >> 20) * SECTOR))
    return pages


def collect_descriptors(pck):
    """Finds texture descriptors and groups them into mip chains."""
    recs = []
    for off in range(8, len(pck) - 8, 4):
        lo = struct.unpack_from('<I', pck, off)[0]
        psm = (lo >> 20) & 0x3F
        if psm not in (19, 20):
            continue
        tbw, tw = (lo >> 14) & 0x3F, (lo >> 26) & 0xF
        if tbw == 0 or tbw > 16 or tw < 3 or tw > 10:
            continue
        hi = struct.unpack_from('<I', pck, off + 4)[0]
        th = ((lo >> 30) & 0x3) | ((hi & 0x3) << 2)
        if th < 3 or th > 10:
            continue
        w, h = 1 << tw, 1 << th
        if tbw != max(1, w // 64):
            continue
        q0 = struct.unpack_from('<I', pck, off - 8)[0]
        aw, ah = q0 & 0x1F, (q0 >> 5) & 0x1F
        if psm == 19:
            ow, oh = w >> 1, h >> 1
        else:
            ow, oh = ((h >> 1, h >> 2) if w == h else (w >> 2, h >> 1))
        conv = (aw == _log2sh(ow) and ah == _log2sh(oh))
        lin = (aw == _log2sh(w) and ah == _log2sh(h))
        if not (conv or lin):
            continue          # detector false positive
        recs.append({'off': off, 'psm': psm, 'w': w, 'h': h, 'swizzled': conv,
                     'size': w * h if psm == 19 else w * h // 2})
    chains, cur = [], []
    for r in recs:
        if cur and r['off'] - cur[-1]['off'] != 16:
            chains.append(cur)
            cur = []
        cur.append(r)
    if cur:
        chains.append(cur)
    return chains


def find_object_name(pck, chain):
    """Looks for a readable texture name stored next to the descriptor.

    For embedded models (pedestrians/traffic/props etc., addressed by
    ordinary pointers) the engine keeps the original .tex file name in
    the texture owner object when rmcSaveTextureNames is enabled
    (SLUS_123_45.c, around :216725): the name pointer is at
    offset +120 from the start of the object, and the object starts at
    the address of the chain's first TEX0 entry minus 16 bytes.

    Confirmed on real files: 84 of 101 chains in
    atlanta_midnight_clear_props.pck give clean, meaningful names
    (a_prop_atm01, a_org_tree_02, a_prop_garagedoor etc.), similar
    for atlanta_peds.pck / atlanta_traffic.pck.

    CITY textures (addressed by .ppf pages through dictionary 123)
    physically have NO name: the +120 field of the object is always
    zero -- checked on 3060 of 3149 chains in
    atlanta_midnight_cloudy.pck. Not an extractor limitation but
    how the game baked city geometry (names were not
    kept for streamed textures, apparently to save
    memory on a static, index-addressed resource).
    """
    a1 = chain[0]['off'] - 16
    cand = a1 + 120
    if cand < 0 or cand + 4 > len(pck):
        return None
    v = struct.unpack_from('<I', pck, cand)[0]
    if not (PTR_BASE <= v < PTR_BASE + 0x2000000):
        return None
    fo_v = v - PTR_BASE + PCK_HDR
    if fo_v < 0 or fo_v + 64 > len(pck):
        return None
    s = pck[fo_v:fo_v + 64].split(b'\x00')[0]
    try:
        s2 = s.decode('ascii')
    except UnicodeDecodeError:
        return None
    if 2 <= len(s2) <= 48 and all(32 <= ord(c) < 127 for c in s2):
        return s2
    return None


def _runs(offsets, stride):
    out, cur = [], []
    for x in offsets:
        if cur and x - cur[-1] != stride:
            out.append(cur)
            cur = []
        cur.append(x)
    if cur:
        out.append(cur)
    return out


def build_ref_index(pck, pages):
    """Collects datChunkRef entries: either page references or pointers.

    A page reference is recognised by its high bits (dictionary number)
    being the same for most entries and the index fitting the page count.
    Entries come in runs with a 12-byte step, one per mip level.
    """
    from collections import Counter
    mode = None
    runs = []
    if pages:
        cnt = Counter()
        for off in range(4, len(pck) - 12, 4):
            v = struct.unpack_from('<I', pck, off)[0]
            d = v >> 23
            if 0 < d < 512 and (v & 0x7FFFFF) < len(pages):
                cnt[d] += 1
        best = (0, None, None)
        for d, n in cnt.most_common(12):
            if n < 32:
                continue
            pos = [off for off in range(4, len(pck) - 12, 4)
                   if (struct.unpack_from('<I', pck, off)[0] >> 23) == d
                   and (struct.unpack_from('<I', pck, off)[0] & 0x7FFFFF) < len(pages)]
            rr = _runs(pos, 12)
            # real references come in runs of the mip chain length
            score = sum(len(r) for r in rr if len(r) >= 2)
            if score > best[0]:
                best = (score, d, rr)
        if best[0] >= 64:
            runs = best[2]
            mode = 'page'
    if mode is None:
        pos = [off for off in range(0, len(pck) - 12, 4)
               if PTR_BASE <= struct.unpack_from('<I', pck, off)[0] < PTR_BASE + 0x2000000]
        runs = _runs(pos, 12)
        mode = 'ptr'
    starts = sorted(r[0] for r in runs)
    by_start = {r[0]: r for r in runs}
    return mode, starts, by_start


def locate_exact(pck, chain, pages):
    """Exact data location -- read from the rmcTexturePS2 object itself.

    The first level's TEX0 descriptor is at +16 of the object, so the object
    starts 16 bytes before it. At +72 the object has mip level entries
    of 12 bytes: [pointer to data in the .pck | 0][offset in page]
    [page reference (dictionary << 23) | page, or 0]; level count --
    byte +134. In both cases data starts 144 bytes (0x90)
    after the address: the memory block header (a .ppf page is a heap
    snapshot; the header is almost entirely debug 0xCD).
    Verified: for all 8735 page references of Atlanta and 11419 of Tokyo the data
    matches what the old statistical search found; in addition
    this finds textures embedded directly in the city .pck (10 in Atlanta:
    the sky strips 256x128 etc.) -- the old search missed them.
    Returns a list of (source, data offset, page, offset in
    page) per level, or None."""
    obj = chain[0]['off'] - 16
    if obj < 0 or obj + 72 + 12 * len(chain) > len(pck) or pck[obj + 4] != 0:
        return None
    out = []
    for k in range(len(chain)):
        ptr, off, ref = struct.unpack_from('<3I', pck, obj + 72 + 12 * k)
        if ref >> 23:
            pidx = ref & 0x7FFFFF
            if not pages or pidx >= len(pages) or off > 0x11000:
                return None
            out.append(('ppf', pages[pidx][0] + off + ALLOC_HDR, pidx, off))
        elif PTR_BASE <= ptr < PTR_BASE + 0x2000000:
            out.append(('pck', ptr - PTR_BASE + PCK_HDR + ALLOC_HDR, 0, 0))
        else:
            return None
    return out


def locate_data(pck, chain, pages, refidx):
    """Returns data addresses for each level of the chain."""
    import bisect
    mode, starts, by_start = refidx
    i = bisect.bisect_right(starts, chain[-1]['off'])
    if i >= len(starts):
        return None
    run = by_start[starts[i]]
    out = []
    for k, lvl in enumerate(chain):
        if k >= len(run):
            return None
        P = run[k]
        v = struct.unpack_from('<I', pck, P)[0]
        if mode == 'ptr':
            out.append(('pck', v - PTR_BASE + PCK_HDR + ALLOC_HDR, 0, 0))
        else:
            pidx = v & 0x7FFFFF
            poff = struct.unpack_from('<I', pck, P - 4)[0]
            if pidx >= len(pages) or poff > 0x11000:
                return None
            out.append(('ppf', pages[pidx][0] + poff + ALLOC_HDR, pidx, poff))
    return out


def alpha_ok(buf, a, sz):
    return 0 <= a and a + sz <= len(buf) and max(buf[a + 3:a + sz:4]) <= 0x80


def fix_unused_alpha(pal):
    """A palette with COMPLETELY zero alpha (after scaling *2) --
    not a real cutout but an unused byte: the game apparently does not
    read the alpha channel for ordinary opaque surfaces, and it
    stays 0 "by default", not because the texture should be
    transparent. Checked statistically on 1996 Atlanta palettes: 85.3%
    are already opaque (max>=250), 2.9% -- entirely zero
    (this case), 11.8% -- partial (possibly real foliage
    with cutouts -- NOT touched, to keep real transparency).
    Only the unambiguous case (max==0 over the whole palette) is fixed."""
    if pal[:, 3].max() == 0:
        pal[:, 3] = 255
    return pal


def extract(pck_path, ppf_path, outdir, base_only=False, channel_order='rgba'):
    pck = open(pck_path, 'rb').read()
    hdr = read_pck_header(pck)
    if hdr is None:
        raise SystemExit(f"{pck_path}: does not look like a .pck (no header with magic 1)")
    ppf = open(ppf_path, 'rb').read() if ppf_path else None
    pages = load_pages(ppf) if ppf else []
    os.makedirs(outdir, exist_ok=True)

    chains = collect_descriptors(pck)
    refidx = build_ref_index(pck, pages)
    manifest, saved = [], 0
    used_pages = set()
    for ch in chains:
        locs = locate_exact(pck, ch, pages) or locate_data(pck, ch, pages, refidx)
        if locs is None:
            continue
        for loc in locs:
            if loc[0] == 'ppf':
                used_pages.add(loc[2])
        obj_name = find_object_name(pck, ch)
        # palette -- after the data of the last level
        last, lloc = ch[-1], locs[-1]
        buf = pck if lloc[0] == 'pck' else ppf
        palsz = 1024 if last['psm'] == 19 else 64
        pa = lloc[1] + last['size']
        pal = None
        if alpha_ok(buf, pa, palsz):
            pal = np.frombuffer(buf[pa:pa + palsz], dtype=np.uint8).reshape(-1, 4).copy()
        elif alpha_ok(buf, pa + ALLOC_HDR, palsz):
            pa += ALLOC_HDR
            pal = np.frombuffer(buf[pa:pa + palsz], dtype=np.uint8).reshape(-1, 4).copy()
        if pal is not None:
            pal = csm1(pal)
            pal[:, 3] = np.clip(pal[:, 3].astype(np.int16) * 2, 0, 255)
            pal = fix_unused_alpha(pal)

        for k, (lvl, loc) in enumerate(zip(ch, locs)):
            if base_only and k > 0:
                # the palette above is read through the LAST level
                # of the chain anyway (it is shared by the chain and physically lies there),
                # so a break here does not spoil the base level palette.
                break
            src = pck if loc[0] == 'pck' else ppf
            a = loc[1]
            if a + lvl['size'] > len(src):
                continue
            idx = decode_indices(src[a:a + lvl['size']], lvl['w'], lvl['h'],
                                 lvl['psm'], lvl['swizzled'])
            idx = idx[::-1]                      # stored bottom-up
            tag = '8bpp' if lvl['psm'] == 19 else '4bpp'
            where = f"p{loc[2]:04d}_o{loc[3]:05x}" if loc[0] == 'ppf' else f"a{a:07x}"
            name_part = (re.sub(r'[^A-Za-z0-9_.-]', '_', obj_name) + "_") if obj_name else ""
            stem = f"{name_part}{where}_{lvl['w']}x{lvl['h']}_{tag}_l{k}"
            if pal is None or idx.max() >= len(pal):
                name = 'idx_' + stem + '.png'
                Image.fromarray(idx, 'L').save(os.path.join(outdir, name))
                entry = {'file': name, 'color': False}
            else:
                c = pal[idx]
                # Palette channel order: RGBA for all files -- verified
                # pixel-exact against PCSX2 VRAM dumps of the running game
                # (8 Tokyo textures as is, 5 Atlanta textures); an old rule
                # 'ppf pages -> BGRA' was wrong and swapped red and blue.
                # --channels bgra is kept only as a fallback.
                if channel_order in ('bgra', 'rgba'):
                    order = channel_order
                else:
                    order = 'rgba'
                if order == 'bgra':
                    rgba = np.dstack([c[..., 2], c[..., 1], c[..., 0], c[..., 3]]).astype(np.uint8)
                else:
                    rgba = np.dstack([c[..., 0], c[..., 1], c[..., 2], c[..., 3]]).astype(np.uint8)
                name = 'col_' + stem + '.png'
                Image.fromarray(rgba, 'RGBA').save(os.path.join(outdir, name))
                entry = {'file': name, 'color': True, 'palette_abs': int(pa)}
            entry.update({'w': lvl['w'], 'h': lvl['h'], 'psm': lvl['psm'],
                          'mip_level': k, 'swizzled': lvl['swizzled'],
                          'source': loc[0], 'object_name': obj_name,
                          # offset of the TEX0 entry in the .pck -- the model
                          # script uses it to find the texture file of an
                          # rmcTexturePS2 object (TEX0 lies at its +16)
                          'tex0_off': int(ch[0]['off'])})
            manifest.append(entry)
            saved += 1

    # Sky pages: NOT addressed through the usual reference chain (in the game
    # code mcCityTextureFactory::Create loads them by name in a separate branch,
    # strstr(name, "_sky_"), not through object geometry references).
    # So the main loop above does not find them: no TEX0
    # chain references them. Here all .ppf pages
    # not touched by the main loop are checked separately: if a page
    # decodes as 256x256 8bpp with a real palette next to it,
    # it is saved separately with the sky_ prefix.
    if ppf and pages:
        for pidx, (base, size) in enumerate(pages):
            if pidx in used_pages:
                continue
            w = h = 256
            a = base + ALLOC_HDR
            if a + w * h > len(ppf):
                continue
            idx = unswizzle8(ppf[a:a + w * h], w, h)[::-1]
            # A real 256-colour palette for these pages is often missing
            # within the page itself (little space is left -- sometimes
            # less than 2 KB). The first alpha_ok match often
            # turns out degenerate (almost all entries zero, which
            # passes the "alpha<=0x80" check trivially). So a real
            # variety of colours is required here as well --
            # otherwise a random, certainly wrong colour would be
            # used instead of an honest indexed image.
            pal = None
            for d0 in range(0, max(size - ALLOC_HDR - w * h - 1024, 0) + 1, 16):
                pa = a + w * h + d0
                if not alpha_ok(ppf, pa, 1024):
                    continue
                cand = np.frombuffer(ppf[pa:pa + 1024], dtype=np.uint8).reshape(256, 4)
                if len(np.unique(cand[:, :3], axis=0)) < 30:
                    continue
                pal = cand.copy()
                break
            if pal is None:
                # No real palette found -- save as is,
                # in greyscale, instead of painting it with
                # a random wrong colour.
                name = f"sky_p{pidx:04d}_{w}x{h}_8bpp_NOPALETTE.png"
                Image.fromarray(idx, 'L').save(os.path.join(outdir, name))
                manifest.append({'file': name, 'color': False, 'palette_abs': None,
                                 'w': w, 'h': h, 'psm': 19, 'mip_level': 0,
                                 'swizzled': True, 'source': 'ppf_sky',
                                 'object_name': None})
                saved += 1
                continue
            pal = csm1(pal)
            pal[:, 3] = np.clip(pal[:, 3].astype(np.int16) * 2, 0, 255)
            pal = fix_unused_alpha(pal)
            if idx.max() >= len(pal):
                continue
            c = pal[idx]
            # channel order -- as for other textures (RGBA by default;
            # BGRA used to be hard-coded here)
            if channel_order == 'bgra':
                rgba = np.dstack([c[..., 2], c[..., 1], c[..., 0], c[..., 3]]).astype(np.uint8)
            else:
                rgba = np.dstack([c[..., 0], c[..., 1], c[..., 2], c[..., 3]]).astype(np.uint8)
            name = f"sky_p{pidx:04d}_{w}x{h}_8bpp.png"
            Image.fromarray(rgba, 'RGBA').save(os.path.join(outdir, name))
            manifest.append({'file': name, 'color': True, 'palette_abs': int(pa),
                             'w': w, 'h': h, 'psm': 19, 'mip_level': 0,
                             'swizzled': True, 'source': 'ppf_sky',
                             'object_name': None})
            saved += 1
    return manifest


def find_companion(path):
    """Finds the pair file: .ppf with the same name for a .pck and vice versa."""
    root, ext = os.path.splitext(path)
    other = root + ('.ppf' if ext.lower() == '.pck' else '.pck')
    if os.path.exists(other):
        return other
    d, base = os.path.split(root)
    for cand in os.listdir(d or '.'):
        if os.path.splitext(cand)[0] == base and cand != os.path.basename(path):
            return os.path.join(d, cand)
    return None


def dedupe_output(outdir, manifest):
    """Removes files byte-identical in content to others
    (the same texture is often baked into many
    .ppf pages to keep each page self-contained for streaming
    -- not an extraction error but a property of the format).

    Keeps ONE file per unique image (the first by
    name, for reproducibility) and deletes the rest from disk. Nothing
    is lost in the manifest: every entry of a group has its 'file'
    pointing to the kept file, and the original name is stored in
    'original_file' -- showing which pages/offsets shared the same
    image."""
    import hashlib
    by_hash = defaultdict(list)
    for m in manifest:
        path = os.path.join(outdir, m['file'])
        if not os.path.exists(path):
            continue
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        by_hash[h].append(m)

    removed = 0
    for h, entries in by_hash.items():
        entries.sort(key=lambda m: m['file'])
        keep = entries[0]['file']
        for m in entries:
            m['original_file'] = m['file']
            m['file'] = keep
            if m['original_file'] != keep:
                p = os.path.join(outdir, m['original_file'])
                if os.path.exists(p):
                    os.remove(p)
                    removed += 1
    return removed


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Texture extraction for Midnight Club 3 (PS2) from .pck/.ppf",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='.pck or .ppf file')
    ap.add_argument('outdir', nargs='?', default=None,
                     help='output folder (default <file name>_textures)')
    ap.add_argument('-b', '--base-only', action='store_true',
                     help='save only the base level (l0) of each texture, '
                          'without smaller mip copies (l1, l2 -- the same images '
                          'at half and quarter size, used by the game for LOD)')
    ap.add_argument('-d', '--dedupe', action='store_true',
                     help='delete files byte-identical to others '
                          '(the city often bakes the same texture into many '
                          '.ppf pages). Keeps one file per '
                          'unique image; the original name of each entry '
                          'is kept in the manifest as original_file')
    ap.add_argument('-c', '--channels', choices=['rgba', 'bgra', 'auto'], default='rgba',
                     help='palette channel order. rgba (default) -- for all files: '
                          'verified with PCSX2 VRAM dumps for Atlanta and Tokyo. '
                          'bgra -- only if red and blue are swapped in the images. '
                          'auto -- same as rgba (kept for old commands)')
    args = ap.parse_args()

    src = args.src
    outdir = args.outdir or os.path.splitext(src)[0] + '_textures'
    data = open(src, 'rb').read()

    if is_ppf(data):
        pck_path = find_companion(src)
        if not pck_path:
            raise SystemExit(
                f"{src}: these are .ppf pages, but the texture descriptors are in the pair .pck.\n"
                f"Put a file with the same name and .pck extension next to it or give it as the first argument.")
        print(f"pages: {os.path.basename(src)}\ndescriptors: {os.path.basename(pck_path)}")
        co = 'bgra' if args.channels == 'bgra' else 'rgba'
        man = extract(pck_path, src, outdir, base_only=args.base_only, channel_order=co)
    else:
        if read_pck_header(data) is None:
            raise SystemExit(f"{src}: recognised neither as .pck nor as .ppf")
        ppf_path = find_companion(src)
        if ppf_path:
            print(f"descriptors: {os.path.basename(src)}\npages: {os.path.basename(ppf_path)}")
        else:
            print(f"descriptors and data: {os.path.basename(src)} (embedded textures)")
        co = 'bgra' if args.channels == 'bgra' else 'rgba'
        man = extract(src, ppf_path, outdir, base_only=args.base_only, channel_order=co)

    with open(outdir.rstrip('/') + '_manifest.json', 'w') as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    col = sum(1 for m in man if m['color'])
    tag = " (base level only, no mip copies)" if args.base_only else ""
    print(f"extracted {len(man)} textures{tag}: {col} colour, {len(man) - col} indexed -> {outdir}")

    if args.dedupe:
        removed = dedupe_output(outdir, man)
        with open(outdir.rstrip('/') + '_manifest.json', 'w') as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
        uniq = len(man) - removed
        print(f"dedupe: removed {removed} duplicate files, {uniq} unique left")


if __name__ == '__main__':
    main()
