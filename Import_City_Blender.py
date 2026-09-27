"""
import_city_blender.py -- places a Midnight Club 3 (PS2) city and its
props in Blender from the output of mc3_extract_models.py.

Works for every city of the game (Atlanta, Detroit, San Diego, Tokyo); the
name is historical.

USAGE
  1. python mc3_extract_models.py atlanta_midnight_clear.pck atlanta_city --props [--fog]
     (next to the city .pck: its .ppf of the same weather and *_props.pck)
  2. In Blender, Scripting tab: open this file, set OUTPUT_DIR to the output
     folder (e.g. r"C:/Users/.../atlanta_city") and press Run Script.
     Or: blender --background --python import_city_blender.py -- /path
  Best run in a new empty .blend file.

RESULT
  Collection <folder> -- the city: blocks and instances (one object per
  placement, the Z-up rotation baked into each object's matrix -- no parent
  object, so no relationship lines). Sub-collections:
    <folder>_props   -- props from props/ (if present), merged per type;
    <folder>_fog     -- low particle fog from fog/ (hidden);
    <folder>_physics -- collision mesh from physics/ (off by default).
  The sky dome is loaded hidden.
  Materials follow the game's shader templates (shaderlib/city/*.shadert):
    baked lighting -- vertex colour attribute 'cpv' multiplies the texture;
    *_mask / *_win -- window shader: wall with glass cut out + self-lit
                      interior behind it;
    *_alpha, *_decal -- alpha cutout (foliage, fences) / soft alpha;
    *_hdr          -- night glow surfaces (emission x vertex colour);
    *_add, glow_*, fog_* -- additive flares, light sprites, fog cards;
    *_roadN        -- roads with the 4x tiled detail texture (second pass).
  Re-running in the same file first removes this import's previous
  collections, meshes, materials and images.

The import does not use bpy.ops: its own light .obj parser, one material per
MTL entry, UVs / colours set with foreach_set -- a whole city with props
usually takes a minute or two.
"""

import csv
import os
import re
import sys
import time

import math

import bpy
import mathutils


# ---------------------------------------------------------------------
OUTPUT_DIR = r"C:\path\to\atlanta_output"
# Blender collection name. None -- the output folder name: a city and props
# imported one after another from different folders go to different collections.
COLLECTION_NAME = None
PROGRESS_EVERY = 500
# Merge props (there are thousands) into one object per type: the Blender
# viewport slows down with the number of OBJECTS, not triangles. False -- every
# prop is a separate object (easier to select one by one, but noticeably slower).
MERGE_PROPS = True
# The same for the city. Off by default: the city has a few hundred unique
# meshes for thousands of placements, and copies share meshes (less memory).
# Turn on if the viewport is still slow: far fewer objects,
# but the geometry is duplicated in memory.
MERGE_CITY = False
# Baked lighting (vertex colours from the game data): effect strength. 1.0 --
# as in the game: the .obj holds the palette colour, 1.0 = texture unchanged
# (scale verified in the VU1 microcode). 0.0 -- off.
CPV_STRENGTH = 1.0
# Brightness gain of the baked lighting, to approximate the game's final image.
# The .obj holds the exact per-pass vertex colour (1.0 = texture unchanged,
# verified in the VU1 microcode), but the game then brightens the whole frame:
# mcFbGlow downsamples the ENTIRE frame with no threshold (blend set 14, Cs*1),
# blurs it and ADDS it back (blend set 7, Cs + Cd) -- roughly doubling
# brightness (its strength is in the external *_hdrparamN files). Blender has no
# such pass, so with 1.0 walls look much darker than in the game (e.g. the
# stone around the arches of San Diego s_inst_dt_blk01_01x: ~25/255 instead of
# ~65/255). 2.0 -- close to the game; 1.0 -- the raw per-pass colour.
CPV_GAIN = 2.0
# Collision mesh (physics/ subfolder, extracted with --physics). Not visible
# geometry: its ~33k triangles follow the ground and roads, and when shown
# (even as wireframe) it looks like hatching over the whole map.
# Not loaded by default; True -- load into a collection EXCLUDED from the
# view layer (enable with the checkbox in the Outliner).
IMPORT_PHYSICS = False
# Low particle fog (fog/ subfolder, extracted with --fog): thousands of cards
# over parks, a cemetery, streets. Loaded hidden (enable with the eye icon in
# the Outliner). False -- do not load.
IMPORT_FOG = True
# Texture filtering. False (default) -- 'Linear': Blender uses mip levels and
# smooths textures at a distance. True -- 'Closest': sharp PS2 pixels up close,
# but at a distance fine repeating patterns produce moire -- dark streaks and
# rings around the view centre over the whole ground.
PIXELATED_TEXTURES = False
# The ground layer -- block roads and terrain. False -- do not load these faces
# (for testing: the roads disappear without it).
IMPORT_GROUND_LAYER = True
# 3D viewport clip range the import sets (m). With the default near clip
# of 0.01 m close layers flicker in stripes at a distance.
CLIP_START = 1.0
CLIP_END = 20000.0
# Hide the floor grid and X/Y axes in the viewport (turn back on in
# Overlays -> Floor, X, Y). They lie at height 0, where the city ground is.
HIDE_FLOOR_GRID = True
# ---------------------------------------------------------------------


def get_output_dir():
    argv = sys.argv
    if "--" in argv:
        after = argv[argv.index("--") + 1:]
        if after:
            return after[0]
    return OUTPUT_DIR


# ---------------------------------------------------------------------
# materials: built ONCE from city.mtl, then only reused
# ---------------------------------------------------------------------

def _input(node, *names):
    """Node input by one of several names (they differ between Blender 3.x and 4.x)."""
    for nm in names:
        if nm in node.inputs:
            return node.inputs[nm]
    return None


def build_materials(obj_dir, tex_dir):
    """Parses city.mtl and creates one material per 'newmtl'
    -- once per run, not per file.

      map_Kd  -- base texture (all);
      map_d   -- alpha: cutout for alpha/mask materials, soft alpha for
                 glow/flare/fog/hdr/decal materials (see below);
                 other materials stay opaque on purpose (textures with
                 meaningless alpha would punch holes in walls);
                 detail_map / detail_scale -- road detail texture;
      map_Ke  -- emission: hdr surfaces, window interiors, flares,
                 light sprites and fog.
    """
    mtl_path = os.path.join(obj_dir, 'city.mtl')
    mats = {}
    if not os.path.isfile(mtl_path):
        print(f"  city.mtl not found in {obj_dir} -- objects will have no textures")
        return mats

    cur = None
    n_created = n_alpha = n_emit = 0
    t0 = time.time()

    def flush():
        nonlocal n_created, n_alpha, n_emit
        if cur is None:
            return
        mat = bpy.data.materials.new(cur['name'])
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get('Principled BSDF')
        tex = cur.get('map_Kd')
        if tex and os.path.isfile(tex):
            try:
                img = load_image(tex)
                tn = nt.nodes.new('ShaderNodeTexImage')
                tn.image = img
                tn.interpolation = 'Closest' if PIXELATED_TEXTURES else 'Linear'
                if bsdf:
                    base_out = tn.outputs['Color']
                    # texture tint Kd (basecolor of hdr_object templates: white
                    # font -> red/green/blue sign lettering); white -- no node
                    kd = cur.get('Kd_rgb')
                    if kd and tuple(kd) != (1.0, 1.0, 1.0):
                        try:
                            tmul = nt.nodes.new('ShaderNodeVectorMath')
                            tmul.operation = 'MULTIPLY'
                            tmul.inputs[1].default_value = tuple(kd)
                            nt.links.new(base_out, tmul.inputs[0])
                            base_out = tmul.outputs['Vector']
                        except Exception as ex:
                            print(f"  tint not applied ({cur['name']}): {ex}")
                    # road detail texture (city_road template, 2nd pass):
                    # black with alpha, tiled by the scale (4 x 4); in the game
                    # normal blending -> colour * (1 - detail alpha)
                    if cur.get('detail_map') and os.path.isfile(cur['detail_map']):
                        try:
                            su, sv = cur.get('detail_scale', (4.0, 4.0))
                            tc = nt.nodes.new('ShaderNodeTexCoord')
                            mp = nt.nodes.new('ShaderNodeMapping')
                            mp.inputs['Scale'].default_value = (su, sv, 1.0)
                            nt.links.new(tc.outputs['UV'], mp.inputs['Vector'])
                            dn = nt.nodes.new('ShaderNodeTexImage')
                            dn.image = load_image(cur['detail_map'])
                            dn.interpolation = 'Closest' if PIXELATED_TEXTURES else 'Linear'
                            nt.links.new(mp.outputs['Vector'], dn.inputs['Vector'])
                            inv = nt.nodes.new('ShaderNodeMath')
                            inv.operation = 'SUBTRACT'
                            inv.inputs[0].default_value = 1.0
                            nt.links.new(dn.outputs['Alpha'], inv.inputs[1])
                            sc = nt.nodes.new('ShaderNodeVectorMath')
                            sc.operation = 'SCALE'
                            nt.links.new(base_out, sc.inputs[0])
                            nt.links.new(inv.outputs['Value'], sc.inputs['Scale'])
                            base_out = sc.outputs['Vector']
                        except Exception as ex:
                            print(f"  road detail not connected ({cur['name']}): {ex}")
                    # baked lighting -- for everything except the sky, light sprites,
                    # fog and window interiors (_win are self-lit)
                    if not (cur['name'].endswith(('_sky', '_win')) or cur['name'].startswith(('glow_', 'fog_'))):
                        # baked lighting: texture x vertex colour 'cpv'
                        try:
                            at = nt.nodes.new('ShaderNodeAttribute')
                            at.attribute_name = 'cpv'
                            mul = nt.nodes.new('ShaderNodeVectorMath')
                            mul.operation = 'MULTIPLY'
                            nt.links.new(base_out, mul.inputs[0])
                            nt.links.new(at.outputs['Color'], mul.inputs[1])
                            base_out = mul.outputs['Vector']
                            if CPV_GAIN != 1.0:
                                gn = nt.nodes.new('ShaderNodeVectorMath')
                                gn.operation = 'SCALE'
                                gn.inputs['Scale'].default_value = CPV_GAIN
                                nt.links.new(base_out, gn.inputs[0])
                                base_out = gn.outputs['Vector']
                        except Exception:
                            pass
                    nt.links.new(base_out, bsdf.inputs['Base Color'])
                    if cur.get('map_d'):
                        inp = _input(bsdf, 'Alpha')
                        if inp is not None:
                            nt.links.new(tn.outputs['Alpha'], inp)
                        # soft alpha, not cutout: light sprites (glow_), fog (fog_),
                        # flares (_add: light cards, smoke -- additive in the
                        # game), hdr surfaces (_hdr: night panels, spotlight
                        # beams), ground (_ground) and decals (_decal). Window
                        # walls (_mask) and foliage (_alpha)
                        # -- alpha cutout.
                        soft = cur['name'].startswith(('glow_', 'fog_')) or cur['name'].endswith(('_add', '_hdr', '_ground', '_decal'))
                        try:
                            mat.blend_method = 'BLEND' if soft else 'CLIP'   # Blender 3.x / 4.0-4.1
                        except Exception:
                            pass
                        if hasattr(mat, 'surface_render_method'):
                            mat.surface_render_method = 'BLENDED' if soft else 'DITHERED'   # 4.2+
                        if soft:
                            # translucent overlays cast no shadow (otherwise an almost
                            # transparent layer produces stripes on the ground in EEVEE)
                            for attr, val in (('shadow_method', 'NONE'), ('use_transparent_shadow', True)):
                                try:
                                    setattr(mat, attr, val)
                                except Exception:
                                    pass
                        n_alpha += 1
                    if cur.get('map_Ke'):
                        inp = _input(bsdf, 'Emission Color', 'Emission')
                        em_out = tn.outputs['Color']
                        ke = cur.get('Ke_rgb')
                        if cur['name'].endswith('_hdr'):
                            # hdr: emission of the same colour as the surface --
                            # texture x vertex colour (palette index from the
                            # V4-8 block: the Westin crown -- peach and grey)
                            em_out = base_out
                            ke = None
                        if ke and tuple(ke) != (1.0, 1.0, 1.0):
                            # emission tint (glow_* sprites: colour from the type name)
                            try:
                                tint = nt.nodes.new('ShaderNodeVectorMath')
                                tint.operation = 'MULTIPLY'
                                nt.links.new(tn.outputs['Color'], tint.inputs[0])
                                tint.inputs[1].default_value = tuple(ke)
                                em_out = tint.outputs['Vector']
                                inp_b = _input(bsdf, 'Base Color')
                                if inp_b is not None:
                                    nt.links.new(em_out, inp_b)
                            except Exception:
                                pass
                        if inp is not None:
                            nt.links.new(em_out, inp)
                        st = _input(bsdf, 'Emission Strength')
                        if st is not None:
                            st.default_value = 1.0
                        n_emit += 1
            except Exception as ex:
                print(f"    texture failed to load ({tex}): {ex}")
        elif bsdf is not None and not cur.get('Ke_rgb'):
            # material without texture and without emission (no_texture: chunks
            # that reference the texture 'none' in the game, e.g. Tokyo neon
            # signs): colour = vertex colour 'cpv' x Kd
            kd = cur.get('Kd_rgb', (1.0, 1.0, 1.0))
            try:
                at = nt.nodes.new('ShaderNodeAttribute')
                at.attribute_name = 'cpv'
                mul = nt.nodes.new('ShaderNodeVectorMath')
                mul.operation = 'MULTIPLY'
                nt.links.new(at.outputs['Color'], mul.inputs[0])
                mul.inputs[1].default_value = (kd[0] * CPV_GAIN, kd[1] * CPV_GAIN, kd[2] * CPV_GAIN)
                nt.links.new(mul.outputs['Vector'], bsdf.inputs['Base Color'])
            except Exception:
                inp = _input(bsdf, 'Base Color')
                if inp is not None:
                    inp.default_value = (kd[0], kd[1], kd[2], 1.0)
        elif bsdf is not None:
            # material without texture: colour and emission given as numbers
            # (Kd/Ke) -- used for prop light sprites (glow_*), which have no
            # picture of their own in the city files
            kd = cur.get('Kd_rgb')
            ke = cur.get('Ke_rgb')
            if kd:
                inp = _input(bsdf, 'Base Color')
                if inp is not None:
                    inp.default_value = (kd[0], kd[1], kd[2], 1.0)
            if ke:
                inp = _input(bsdf, 'Emission Color', 'Emission')
                if inp is not None:
                    inp.default_value = (ke[0], ke[1], ke[2], 1.0)
                st = _input(bsdf, 'Emission Strength')
                if st is not None:
                    st.default_value = 2.0
                n_emit += 1
            if cur.get('d', 1.0) < 1.0:
                inp = _input(bsdf, 'Alpha')
                if inp is not None:
                    inp.default_value = cur['d']
                try:
                    mat.blend_method = 'BLEND'
                except Exception:
                    pass
                if hasattr(mat, 'surface_render_method'):
                    mat.surface_render_method = 'BLENDED'
        mats[cur['name']] = mat
        _CREATED_MATS.add(mat.name)
        n_created += 1

    for line in open(mtl_path, encoding='utf-8', errors='replace'):
        p = line.split()
        if not p:
            continue
        if p[0] == 'newmtl':
            flush()
            cur = {'name': p[1]}
        elif p[0] in ('map_Kd', 'map_d', 'map_Ke') and cur is not None:
            rel = line.split(None, 1)[1].strip()
            cur[p[0]] = os.path.normpath(os.path.join(obj_dir, rel))
        elif p[0] == 'detail_map' and cur is not None:
            rel = line.split(None, 1)[1].strip()
            cur['detail_map'] = os.path.normpath(os.path.join(obj_dir, rel))
        elif p[0] == 'detail_scale' and cur is not None and len(p) >= 3:
            cur['detail_scale'] = (float(p[1]), float(p[2]))
        elif p[0] in ('Kd', 'Ke') and cur is not None and len(p) >= 4:
            cur[p[0] + '_rgb'] = tuple(float(x) for x in p[1:4])
        elif p[0] == 'd' and cur is not None and len(p) >= 2:
            cur['d'] = float(p[1])
    flush()

    print(f"  materials created: {n_created} (alpha cutout: {n_alpha}, "
          f"emissive: {n_emit}) in {time.time()-t0:.1f}s")
    return mats

    cur_name = None
    cur_tex = None
    n_created = 0
    t0 = time.time()

    def flush():
        nonlocal n_created
        if cur_name is None:
            return
        mat = bpy.data.materials.new(cur_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if cur_tex and os.path.isfile(cur_tex):
            try:
                img = load_image(cur_tex)
                tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
                tex_node.image = img
                tex_node.interpolation = 'Closest' if PIXELATED_TEXTURES else 'Linear'
                if bsdf:
                    mat.node_tree.links.new(
                        tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                # The alpha channel of City textures is NOT connected to transparency:
                # some extracted PNGs have alpha=0 almost or entirely
                # (checked on real output -- some files have
                # 100% zero-alpha pixels), and that does NOT always mean
                # "cutout" -- for ordinary opaque surfaces (brick,
                # asphalt) it makes the material COMPLETELY invisible with
                # blend_method='CLIP', hence "holes" in the scene. Materials
                # stay fully opaque by default; the price --
                # vegetation/fences (alpha slot) would be solid
                # cards instead of cut-out leaves/grates, but that is
                # safer than disappearing buildings. Cutout can be
                # enabled manually for alpha-slot materials only.
            except Exception as ex:
                print(f"    texture failed to load ({cur_tex}): {ex}")
        mats[cur_name] = mat
        n_created += 1

    for line in open(mtl_path, encoding='utf-8', errors='replace'):
        p = line.split()
        if not p:
            continue
        if p[0] == 'newmtl':
            flush()
            cur_name = p[1]
            cur_tex = None
        elif p[0] == 'map_Kd' and cur_name is not None:
            rel = line.split(None, 1)[1].strip()
            cur_tex = os.path.normpath(os.path.join(obj_dir, rel))
    flush()

    print(f"  materials created: {n_created} in {time.time()-t0:.1f}s")
    return mats


# ---------------------------------------------------------------------
# own light .obj parser -- no bpy.ops, no re-reading of the .mtl
# ---------------------------------------------------------------------

def add_cpv(mesh, vcols):
    """Vertex colour attribute 'cpv' (present on every mesh: materials multiply
    the texture by it, and a missing attribute would give black)."""
    k = CPV_STRENGTH
    if not vcols:
        flat = [1.0] * (4 * len(mesh.vertices))      # neutral colour
    elif k == 1.0:
        flat = [x for r, g, b in vcols for x in (r, g, b, 1.0)]
    else:
        flat = [x for r, g, b in vcols
                for x in (1 + (r - 1) * k, 1 + (g - 1) * k, 1 + (b - 1) * k, 1.0)]
    try:
        attr = mesh.color_attributes.new('cpv', 'FLOAT_COLOR', 'POINT')   # Blender 3.2+
        attr.data.foreach_set('color', flat)
    except Exception:
        try:
            attr = mesh.attributes.new('cpv', 'FLOAT_COLOR', 'POINT')
            attr.data.foreach_set('color', flat)
        except Exception as ex:
            print(f"    vertex colours not written ({mesh.name}): {ex}")


def read_obj_arrays(filepath):
    """Raw arrays from an .obj: vertices, UVs, faces (vertex indices), UV indices,
    material name per face. Used to merge many placements into one mesh."""
    verts, uvs, faces, face_uv, face_mat = [], [], [], [], []
    cur = None
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif p[0] == 'vt':
                uvs.append((float(p[1]), float(p[2])))
            elif p[0] == 'usemtl':
                cur = p[1]
            elif p[0] == 'f':
                idx = [q.split('/') for q in p[1:4]]
                faces.append(tuple(int(q[0]) - 1 for q in idx))
                face_uv.append(tuple(int(q[1]) - 1 if len(q) > 1 and q[1] else -1 for q in idx))
                face_mat.append(cur)
    return verts, uvs, faces, face_uv, face_mat


def build_merged_mesh(name, parts, materials):
    """One mesh from many placements of one type: the vertices of each placement
    are transformed by its matrix and appended to a common list.

    Why: there are thousands of props (5721 placements of 95 types in Atlanta).
    A separate Blender object per placement slows the viewport a lot -- the
    overhead is per object, not per triangle. Merging gives 95
    objects instead of 5721 with the same geometry.
    parts: [(vertices, uv, faces, uv_indices, face_materials, matrix|None)]
    """
    V, UV, F, FUV, FM = [], [], [], [], []
    for verts, uvs, faces, face_uv, face_mat, M in parts:
        vo, uo = len(V), len(UV)
        if M is None:
            V.extend(verts)
        else:
            for x, y, z in verts:
                V.append((M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
                          M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3],
                          M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3]))
        UV.extend(uvs)
        for (a, b, c), (ua, ub, uc), m in zip(faces, face_uv, face_mat):
            F.append((a + vo, b + vo, c + vo))
            FUV.append((ua + uo if ua >= 0 else -1, ub + uo if ub >= 0 else -1,
                        uc + uo if uc >= 0 else -1))
            FM.append(m)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(V, [], F)
    mesh.update(calc_edges=False)
    add_cpv(mesh, None)
    slot_of, slots = {}, []
    for m in FM:
        if m not in slot_of:
            slot_of[m] = len(slots)
            slots.append(m)
    for m in slots:
        mesh.materials.append(materials.get(m))
    if F:
        mesh.polygons.foreach_set('material_index', [slot_of[m] for m in FM])
    if UV:
        nuv = len(UV)
        flat = []
        for tri in FUV:
            for ui in tri:
                flat.extend(UV[ui] if 0 <= ui < nuv else (0.0, 0.0))
        uv_layer = mesh.uv_layers.new(name='UVMap')
        uv_layer.data.foreach_set('uv', flat)
    return mesh


def load_obj_mesh(filepath, name, materials):
    """Builds a bpy.types.Mesh directly from an .obj (v/vt/f/usemtl/g).

    Speed: the file is read at once; UVs, vertex colours and face materials
    are passed to Blender with one foreach_set call each. UVs used to be
    assigned in a per-corner loop with a vertex search -- that was
    about half of the whole import time."""
    verts = []
    vcols = []          # vertex colour (baked lighting), 1.0 -- neutral
    has_col = False
    uvs = []
    faces = []          # (i0,i1,i2)
    face_uv = []        # UV indices per corner, flat: 3 per face
    face_mat = []       # material slot index per face
    slot_of = {}        # material name -> mesh slot index
    slot_names = []
    cur_slot = 0
    skip_group = False

    with open(filepath, encoding='utf-8', errors='replace') as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line:
            continue
        c0 = line[0]
        if c0 == 'v':
            c1 = line[1:2]
            if c1 == ' ':
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
                if len(p) >= 7:
                    vcols.append((float(p[4]), float(p[5]), float(p[6])))
                    has_col = True
                else:
                    vcols.append((1.0, 1.0, 1.0))
            elif c1 == 't':
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
        elif c0 == 'f':
            if skip_group:
                continue
            p = line.split()
            a, b, c = p[1].split('/'), p[2].split('/'), p[3].split('/')
            faces.append((int(a[0]) - 1, int(b[0]) - 1, int(c[0]) - 1))
            if len(a) > 1 and a[1]:
                face_uv.append(int(a[1]) - 1)
                face_uv.append(int(b[1]) - 1)
                face_uv.append(int(c[1]) - 1)
            else:
                face_uv.extend((-1, -1, -1))
            face_mat.append(cur_slot)
        elif c0 == 'u' and line.startswith('usemtl'):
            mname = line.split()[1]
            if mname not in slot_of:
                slot_of[mname] = len(slot_names)
                slot_names.append(mname)
            cur_slot = slot_of[mname]
        elif c0 == 'g':
            skip_group = (not IMPORT_GROUND_LAYER) and line[2:].startswith('ground')

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)     # face and vertex order is preserved
    mesh.update(calc_edges=False)
    add_cpv(mesh, vcols if has_col else None)

    for mname in slot_names:
        mesh.materials.append(materials.get(mname))   # None allowed -- empty slot

    if faces:
        mesh.polygons.foreach_set('material_index', face_mat)
        if uvs and any(i >= 0 for i in face_uv):
            nuv = len(uvs)
            flat = []
            for i in face_uv:
                if 0 <= i < nuv:
                    flat.extend(uvs[i])
                else:
                    flat.extend((0.0, 0.0))
            uv_layer = mesh.uv_layers.new(name='UVMap')
            uv_layer.data.foreach_set('uv', flat)
    return mesh


# ---------------------------------------------------------------------

_IMAGES = {}          # path -> image loaded in THIS run
_CREATED_MATS = set() # names of materials created in this run


def load_image(path):
    """An image is always read from disk anew (check_existing=True used to take
    the old one from the .blend even if the texture on disk had changed); within
    one run -- one image per path."""
    img = _IMAGES.get(path)
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        _IMAGES[path] = img
    return img


def remove_previous_import(names):
    """Removes what is left from previous imports of the same folder: collections
    (city, props, fog, physics), their objects and meshes, then materials and images
    left without users. Without this an old physics collection (wireframe --
    hatching over the whole map) stayed in the scene, and merging duplicates
    replaced new materials with old ones."""
    n_obj = n_coll = 0
    for name in names:
        coll = bpy.data.collections.get(name)
        if coll is None:
            continue
        stack = [coll]
        while stack:
            c = stack.pop()
            stack.extend(list(c.children))
            for o in list(c.objects):
                data = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                n_obj += 1
                if data is not None and getattr(data, 'users', 1) == 0:
                    try:
                        bpy.data.meshes.remove(data)
                    except Exception:
                        pass
        for c in [coll] + list(coll.children_recursive if hasattr(coll, 'children_recursive') else []):
            try:
                bpy.data.collections.remove(c)
                n_coll += 1
            except Exception:
                pass
    n_mat = n_img = 0
    for _ in range(2):
        for m in list(bpy.data.meshes):
            if m.users == 0:
                bpy.data.meshes.remove(m)
        for m in list(bpy.data.materials):
            if m.users == 0:
                bpy.data.materials.remove(m); n_mat += 1
        for im in list(bpy.data.images):
            if im.users == 0:
                bpy.data.images.remove(im); n_img += 1
    if n_obj or n_coll or n_mat or n_img:
        print(f"  removed from the previous import: collections {n_coll}, objects {n_obj}, "
              f"materials {n_mat}, images {n_img}")


def dedupe_materials():
    print("merging duplicate materials (if any)...")
    groups = {}
    pat = re.compile(r'^(.*)\.\d{3}$')
    for mat in list(bpy.data.materials):
        m = pat.match(mat.name)
        base = m.group(1) if m else mat.name
        groups.setdefault(base, []).append(mat)
    removed = 0
    for base, mats in groups.items():
        if len(mats) <= 1:
            continue
        fresh = [m for m in mats if m.name in _CREATED_MATS]
        keep = fresh[0] if fresh else mats[0]     # a new material wins over an old one
        for dup in [m for m in mats if m is not keep]:
            for mesh in bpy.data.meshes:
                for i, slot_mat in enumerate(mesh.materials):
                    if slot_mat == dup:
                        mesh.materials[i] = keep
            bpy.data.materials.remove(dup)
            removed += 1
    if removed:
        print(f"  duplicates removed: {removed}")


def make_matrix(row):
    x, y, z = row['x'], row['y'], row['z']
    ax, ay, az = row['ax'], row['ay'], row['az']
    bx, by, bz = row['bx'], row['by'], row['bz']
    cx, cy, cz = row['cx'], row['cy'], row['cz']
    return mathutils.Matrix((
        (ax, bx, cx, x),
        (ay, by, cy, y),
        (az, bz, cz, z),
        (0.0, 0.0, 0.0, 1.0),
    ))


def read_placements(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for k in ('x', 'y', 'z', 'ax', 'ay', 'az', 'bx', 'by', 'bz',
                      'cx', 'cy', 'cz'):
                row[k] = float(row[k])
            row['sector'] = int(row['sector'])
            rows.append(row)
    return rows


def get_collection(coll_name, parent):
    """A collection with this name (cleared of its previous contents) or a new one.
    Clearing matters when importing into the same file again: otherwise the scene has
    two copies in the same places, and all surfaces flicker
    (z-fighting -- rings over the whole map)."""
    if coll_name in bpy.data.collections:
        coll = bpy.data.collections[coll_name]
        old = list(coll.objects)
        if old:
            for o in old:
                bpy.data.objects.remove(o, do_unlink=True)
            print(f"  collection '{coll_name}' already existed: removed old objects: {len(old)}")
        if coll.name not in parent.children:
            try:
                parent.children.link(coll)
            except RuntimeError:
                pass
    else:
        coll = bpy.data.collections.new(coll_name)
        parent.children.link(coll)
    return coll


def import_dir(out_dir, coll_name, parent, merge=False):
    """Import of one mc3_extract_models.py output folder (city OR props OR fog
    obj/*.obj, obj/city.mtl, textures/, instance_placements.csv."""
    obj_dir = os.path.join(out_dir, 'obj')
    tex_dir = os.path.join(out_dir, 'textures')
    csv_path = os.path.join(out_dir, 'instance_placements.csv')
    if not os.path.isdir(obj_dir) or not os.path.isfile(csv_path):
        print(f"ERROR: obj/ or instance_placements.csv not found in {out_dir}")
        return None

    t0 = time.time()
    print(f"\n=== {coll_name} ({out_dir})")
    rows = read_placements(csv_path)
    print(f"rows in instance_placements.csv: {len(rows)}")
    materials = build_materials(obj_dir, tex_dir)
    coll = get_collection(coll_name, parent)

    # Geometry and placement matrices are in the game coordinate system (Y up).
    # A +90 degree rotation about X turns the scene into Blender Z-up with a PURE
    # rotation, no mirroring -- face orientation is preserved.
    # This rotation used to be set by a root empty with every object as its child.
    # Blender draws a dashed relationship line from each child to its parent
    # (Relationship Lines overlay): thousands of dashes converging at the origin
    # gave black rays and concentric rings over the whole map -- visible even
    # while models were not drawn. Now there is no parent: the rotation is baked
    # into each object's matrix (ROT @ placement matrix).
    #
    ROT = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X')

    block_rows = [r for r in rows if r['obj'].startswith('block_')]
    inst_rows = [r for r in rows if not r['obj'].startswith('block_')]
    print(f"blocks: {len(block_rows)}, placements: {len(inst_rows)}")

    n = 0
    for row in block_rows:
        fp = os.path.join(obj_dir, row['obj'])
        if not os.path.isfile(fp):
            continue
        mesh = load_obj_mesh(fp, row['name'], materials)
        obj = bpy.data.objects.new(row['name'], mesh)
        obj.matrix_world = ROT
        coll.objects.link(obj)
        if row['name'] == 'sky':
            # the sky dome covers the whole city and from above would hide the
            # map -- loaded hidden in the viewport (enable: eye icon in the Outliner)
            try:
                obj.hide_set(True)
            except Exception:
                pass
            print("  sky loaded hidden in the viewport (object 'sky', enable with the eye icon in the Outliner)")
        n += 1
        if n % PROGRESS_EVERY == 0:
            print(f"  blocks: {n}/{len(block_rows)}  ({time.time()-t0:.0f}s)")
    if block_rows:
        print(f"blocks placed: {n}  ({time.time()-t0:.0f}s)")

    if merge:
        # merge per type (props): one object per type instead of thousands
        by_obj = {}
        for row in inst_rows:
            by_obj.setdefault(row['obj'], []).append(row)
        n_placed = 0
        for fn, rws in by_obj.items():
            fp = os.path.join(obj_dir, fn)
            if not os.path.isfile(fp):
                continue
            base = read_obj_arrays(fp)
            parts = []
            for row in rws:
                M = make_matrix(row)
                parts.append((*base, [[M[i][j] for j in range(4)] for i in range(4)]))
            mesh = build_merged_mesh(fn[:-4], parts, materials)
            obj = bpy.data.objects.new(fn[:-4], mesh)
            obj.matrix_world = ROT
            coll.objects.link(obj)
            n_placed += len(rws)
            if n_placed % PROGRESS_EVERY == 0 or len(by_obj) < 20:
                print(f"  merged: {n_placed}/{len(inst_rows)} placements "
                      f"into {len(coll.objects)-1} objects ({time.time()-t0:.0f}s)")
        print(f"placed: {n_placed} in {len(by_obj)} objects (merged per type) "
              f"in {time.time()-t0:.0f}s")
        return coll

    template_mesh = {}
    n_imported = 0
    n_placed = 0
    for row in inst_rows:
        fn = row['obj']
        mesh = template_mesh.get(fn)
        if mesh is None:
            fp = os.path.join(obj_dir, fn)
            if not os.path.isfile(fp):
                continue
            mesh = load_obj_mesh(fp, fn[:-4], materials)
            template_mesh[fn] = mesh
            n_imported += 1
        placed = bpy.data.objects.new(row['name'], mesh)
        placed.matrix_world = ROT @ make_matrix(row)   # Blender Z-up, no parent
        coll.objects.link(placed)
        n_placed += 1
        if n_placed % PROGRESS_EVERY == 0:
            print(f"  placed: {n_placed}/{len(inst_rows)}  "
                  f"(unique meshes: {n_imported})  "
                  f"({time.time()-t0:.0f}s)")
    print(f"placed: {n_placed} (unique meshes: {n_imported}) "
          f"in {time.time()-t0:.0f}s")
    return coll


def main():
    out_dir = get_output_dir()
    coll_name = COLLECTION_NAME or os.path.basename(os.path.normpath(out_dir))
    props_dir = os.path.join(out_dir, 'props')
    props_name = coll_name + '_props'

    # Warning about other imports in the scene (e.g. from old versions of the
    # script with the 'Atlanta_City' collection): if it is the same city, delete
    # it manually, otherwise surfaces coincide and flicker.
    mine = {coll_name, props_name, coll_name + '_physics'}
    others = [o.name for o in bpy.context.scene.objects
              if o.type == 'EMPTY' and o.name.endswith('_root')
              and o.name[:-5] not in mine]
    old_coll = [c.name for c in bpy.data.collections
                if c.name not in mine and len(c.objects) > 100]
    if others or old_coll:
        print("  WARNING: the scene already has other imports: "
              + ", ".join(others + old_coll)
              + ". If it is the same city -- delete them (Outliner -> Delete "
                "Hierarchy), otherwise coinciding surfaces will flicker in rings.")

    t0 = time.time()
    remove_previous_import([coll_name, props_name, coll_name + '_physics', coll_name + '_fog'])
    city = import_dir(out_dir, coll_name, bpy.context.scene.collection, merge=MERGE_CITY)
    if city is None:
        return
    # Props (ATMs, benches, lamps, trees): the props/ subfolder created
    # by mc3_extract_models.py with --props. Placed in a child
    # collection of the city.
    if os.path.isfile(os.path.join(props_dir, 'instance_placements.csv')):
        import_dir(props_dir, props_name, city, merge=MERGE_PROPS)
    else:
        print(f"\n(no props/ subfolder -- props are not imported; "
              f"extract the city with --props)")
    # Particle fog (--fog) and collision geometry (--physics): fog/ and physics/
    # subfolders in child collections; not visual geometry of the city itself --
    # the fog is loaded hidden, physics only with IMPORT_PHYSICS = True.
    fog_dir = os.path.join(out_dir, 'fog')
    if IMPORT_FOG and os.path.isfile(os.path.join(fog_dir, 'instance_placements.csv')):
        fg = import_dir(fog_dir, coll_name + '_fog', city)
        if fg is not None:
            for o in fg.objects:
                try:
                    o.hide_set(True)
                except Exception:
                    pass
            print("  fog: loaded hidden (enable with the collection's eye icon in the Outliner)")
    phys_dir = os.path.join(out_dir, 'physics')
    if os.path.isfile(os.path.join(phys_dir, 'instance_placements.csv')):
        if not IMPORT_PHYSICS:
            print("\n(physics/ has a collision mesh, but it is not loaded: IMPORT_PHYSICS = False)")
        else:
            ph = import_dir(phys_dir, coll_name + '_physics', city)
            if ph is not None:
                for o in ph.objects:
                    try:
                        o.hide_render = True
                    except Exception:
                        pass
                excluded = False
                try:
                    def find_lc(lc, name):
                        if lc.collection.name == name:
                            return lc
                        for ch in lc.children:
                            r = find_lc(ch, name)
                            if r:
                                return r
                        return None
                    lc = find_lc(bpy.context.view_layer.layer_collection, ph.name)
                    if lc:
                        lc.exclude = True        # not drawn at all until enabled
                        excluded = True
                except Exception as ex:
                    print(f"  could not exclude the collision collection: {ex}")
                print("  collision: collection excluded from the view layer (enable with the checkbox in the Outliner)"
                      if excluded else "  collision loaded (VISIBLE -- exclude the collection manually)")

    dedupe_materials()
    # Viewport clip range. The default near clip of 0.01 m gives depth precision
    # of tens of centimetres or worse at hundreds of metres: close layers
    # (window interiors behind walls etc.) flicker in stripes and circles over
    # the whole map. 1 m improves precision about 100 times.
    n_views = 0
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    for sp in area.spaces:
                        if sp.type == 'VIEW_3D':
                            sp.clip_start = CLIP_START
                            sp.clip_end = CLIP_END
                            if HIDE_FLOOR_GRID:
                                # The floor grid and X/Y axes lie at height 0 -- where
                                # the city ground is (-3..7 m): they show through
                                # roads and lawns as dark stripes, and at a distance (with
                                # a far clip of kilometres) merge into rings
                                # around the view centre
                                ov = sp.overlay
                                for attr in ('show_floor', 'show_axis_x', 'show_axis_y'):
                                    try:
                                        setattr(ov, attr, False)
                                    except Exception:
                                        pass
                            n_views += 1
        for ob in bpy.data.objects:
            if ob.type == 'CAMERA':
                ob.data.clip_start = CLIP_START
                ob.data.clip_end = CLIP_END
    except Exception as ex:
        print(f"  clip range not set: {ex}")
    if n_views:
        print(f"  3D viewport clip range: {CLIP_START} m .. {CLIP_END} m (viewports: {n_views})"
              + ("; floor grid and X/Y axes hidden (Overlays -> Floor, X, Y)" if HIDE_FLOOR_GRID else ""))
    # sky clear colour from the game data (scene.json) -- the scene world colour
    try:
        import json
        sc = json.load(open(os.path.join(out_dir, 'scene.json')))
        col = sc.get('sky_clear_color')
        if col:
            world = bpy.context.scene.world or bpy.data.worlds.new('World')
            bpy.context.scene.world = world
            world.use_nodes = True
            bg = world.node_tree.nodes.get('Background')
            if bg is not None:
                bg.inputs['Color'].default_value = (col[0], col[1], col[2], 1.0)
            print(f"  sky clear colour: {col} ({sc.get('sky_name', '')})")
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"  world colour not set: {ex}")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
