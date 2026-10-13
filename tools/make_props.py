"""Build static/props.glb, the things lying on the table around the board, from Poly Haven
models (all CC0, https://polyhaven.com/models):

    tea_set_01, stationery_supplies, round_spectacles, digital_wrist_watch, potted_plant_04,
    croissant, food_apple_01, magnifying_glass_01, vintage_pocket_watch

Download the glTF version of each (1k) and unzip them side by side in one folder, so that
<folder>/tea_set_01/tea_set_01_1k.gltf and so on exist, then:

    python tools/make_props.py <folder> props-full.glb
    npx gltfpack@0.22.0 -i props-full.glb -o static/props.glb -si 0.3 -kn -km

The second step simplifies the meshes (to about 30% of their triangles, less where that would
change their shape) and quantizes them, from 2.8 MB to under 0.9 MB.

Each prop becomes one node named as in PROPS below, with the parts it is made of merged into
one mesh, in metres, standing on y = 0 and centred on x = z = 0. Textures are shrunk (256 or
512px) and glass becomes a plain see-through material.
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

# prop name: (model, [nodes], texture size, rotation about x in degrees to lay it down)
PROPS = {
    'cup': ('tea_set_01', ['tea_set_01_cup_small_01'], 512, 0),
    'saucer': ('tea_set_01', ['tea_set_01_saucer_circular_03'], 512, 0),
    'plate': ('tea_set_01', ['tea_set_01_plate_large_circular_03'], 512, 0),
    'teapot': ('tea_set_01', ['tea_set_01_teapot_01', 'tea_set_01_teapot_01_lid'], 512, 0),
    'pen_blue': ('stationery_supplies', ['stationery_supplies_pen_blue'], 512, 0),
    'pen_red': ('stationery_supplies', ['stationery_supplies_pen_red'], 512, 0),
    'pen_fancy': ('stationery_supplies', ['stationery_supplies_pen_fancy'], 512, 0),
    'pencil': ('stationery_supplies', ['stationery_supplies_pencil_new_a'], 512, 0),
    'pencil_used': ('stationery_supplies', ['stationery_supplies_pencil_used'], 512, 0),
    'eraser': ('stationery_supplies', ['stationery_supplies_eraser'], 512, 0),
    'pencil_cup': ('stationery_supplies', ['stationery_supplies_pencilcup'], 512, 0),
    'spectacles': ('round_spectacles', ['round_spectacles'], 256, 0),
    'wrist_watch': ('digital_wrist_watch', ['digital_wrist_watch', 'digital_wrist_watch_clasp',
                                            'digital_wrist_watch_strap_a', 'digital_wrist_watch_strap_b'], 256, 0),
    'plant': ('potted_plant_04', ['potted_plant_04_ground', 'potted_plant_04_dirt', 'potted_plant_04_plant',
                                  'potted_plant_04_pot'], 512, 0),
    'croissant': ('croissant', ['croissant'], 256, 0),
    'apple': ('food_apple_01', ['food_apple_01'], 256, 0),
    'magnifier': ('magnifying_glass_01', ['magnifying_glass_01'], 256, -90),
    'pocket_watch': ('vintage_pocket_watch', ['vintage_pocket_watch', 'vintage_pocket_watch_hour_hand',
                                              'vintage_pocket_watch_minute_hand',
                                              'vintage_pocket_watch_second_hand'], 256, -90),
}
GLASS = {'pbrMetallicRoughness': {'baseColorFactor': [0.86, 0.9, 0.9, 0.16], 'metallicFactor': 0,
                                  'roughnessFactor': 0.05},
         'alphaMode': 'BLEND', 'doubleSided': True}
COMPONENTS = {5126: (np.float32, 4), 5123: (np.uint16, 2), 5125: (np.uint32, 4), 5121: (np.uint8, 1)}
WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Source:
    def __init__(self, folder, model):
        path = os.path.join(folder, model, model + '_1k.gltf')
        self.dir = os.path.dirname(path)
        self.g = json.load(open(path))
        self.data = open(os.path.join(self.dir, self.g['buffers'][0]['uri']), 'rb').read()
        self.nodes = {n['name']: n for n in self.g['nodes']}

    def read(self, i):
        a = self.g['accessors'][i]
        bv = self.g['bufferViews'][a['bufferView']]
        dtype, size = COMPONENTS[a['componentType']]
        n = WIDTH[a['type']]
        assert bv.get('byteStride', size * n) == size * n
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        arr = np.frombuffer(self.data, dtype, a['count'] * n, start)
        return arr.reshape(-1, n) if n > 1 else arr


def main(folder, dst):
    out = bytearray()
    views, accessors, images, textures, materials = [], [], [], [], []
    sources, image_ids, material_ids = {}, {}, {}

    def add_view(blob, target=None):
        while len(out) % 4:
            out.append(0)
        views.append({'buffer': 0, 'byteOffset': len(out), 'byteLength': len(blob)})
        if target:
            views[-1]['target'] = target
        out.extend(blob)
        return len(views) - 1

    def add_accessor(arr, ctype, kind, target, bounds=False):
        acc = {'bufferView': add_view(arr.tobytes(), target), 'componentType': ctype,
               'count': len(arr), 'type': kind}
        if bounds:
            acc['min'] = arr.min(0).tolist()
            acc['max'] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    def texture(src, model, tex_index, size):
        img = src.g['textures'][tex_index]['source']
        key = (model, img, size)
        if key not in image_ids:
            im = Image.open(os.path.join(src.dir, src.g['images'][img]['uri'])).convert('RGB')
            im = im.resize((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=85, optimize=True)
            images.append({'bufferView': add_view(buf.getvalue()), 'mimeType': 'image/jpeg'})
            textures.append({'sampler': 0, 'source': len(images) - 1})
            image_ids[key] = len(textures) - 1
        return {'index': image_ids[key]}

    def material(src, model, i, size):
        key = (model, i, size)
        if key not in material_ids:
            m = src.g['materials'][i]
            if m.get('alphaMode') == 'BLEND':
                new = dict(GLASS, name=m['name'])
            else:
                pbr = m.get('pbrMetallicRoughness', {})
                new = {'name': m['name'], 'doubleSided': m.get('doubleSided', False), 'pbrMetallicRoughness': {}}
                out_pbr = new['pbrMetallicRoughness']
                for k in ('baseColorFactor', 'metallicFactor', 'roughnessFactor'):
                    if k in pbr:
                        out_pbr[k] = pbr[k]
                if 'baseColorTexture' in pbr:
                    out_pbr['baseColorTexture'] = texture(src, model, pbr['baseColorTexture']['index'], size)
                # the colour map at full size, the others at half: they matter less on things this small
                if 'metallicRoughnessTexture' in pbr:
                    t = texture(src, model, pbr['metallicRoughnessTexture']['index'], size // 2)
                    out_pbr['metallicRoughnessTexture'] = t
                    img = src.g['textures'][pbr['metallicRoughnessTexture']['index']]['source']
                    if '_arm_' in src.g['images'][img]['uri']:  # some have a roughness map, not ARM
                        new['occlusionTexture'] = dict(t)
                if 'normalTexture' in m:
                    new['normalTexture'] = texture(src, model, m['normalTexture']['index'], size // 2)
            materials.append(new)
            material_ids[key] = len(materials) - 1
        return material_ids[key]

    meshes, nodes = [], []
    for name, (model, node_names, size, tilt) in PROPS.items():
        if model not in sources:
            sources[model] = Source(folder, model)
        src = sources[model]
        parts = {}  # material -> [positions, normals, uvs, indices]
        for nn in node_names:
            node = src.nodes[nn]
            rot = quat_matrix(node['rotation']) if 'rotation' in node else np.eye(3)
            scale = np.array(node.get('scale', [1, 1, 1]))
            move = np.array(node.get('translation', [0, 0, 0]))
            for p in src.g['meshes'][node['mesh']]['primitives']:
                pos = (src.read(p['attributes']['POSITION']) * scale) @ rot.T + move
                nor = (src.read(p['attributes']['NORMAL']) / scale) @ rot.T
                uv = src.read(p['attributes']['TEXCOORD_0'])
                idx = src.read(p['indices']).astype(np.uint32)
                mat = material(src, model, p['material'], size)
                bucket = parts.setdefault(mat, [[], [], [], [], 0])
                bucket[0].append(pos); bucket[1].append(nor); bucket[2].append(uv)
                bucket[3].append(idx + bucket[4]); bucket[4] += len(pos)
        a = math.radians(tilt)
        tilt_m = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        merged = {m: [np.concatenate(b[k]) for k in range(4)] for m, b in parts.items()}
        for b in merged.values():
            b[0] = b[0] @ tilt_m.T
            b[1] = b[1] @ tilt_m.T
        allpos = np.concatenate([b[0] for b in merged.values()])
        lo, hi = allpos.min(0), allpos.max(0)
        shift = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
        prims = []
        for m, (pos, nor, uv, idx) in merged.items():
            pos = (pos - shift).astype(np.float32)
            nor = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-8)
            small = len(pos) < 65536
            prims.append({
                'attributes': {
                    'POSITION': add_accessor(pos, 5126, 'VEC3', 34962, bounds=True),
                    'NORMAL': add_accessor(nor.astype(np.float32), 5126, 'VEC3', 34962),
                    'TEXCOORD_0': add_accessor(uv.astype(np.float32), 5126, 'VEC2', 34962),
                },
                'indices': add_accessor(idx.astype(np.uint16 if small else np.uint32), 5123 if small else 5125,
                                        'SCALAR', 34963),
                'material': m,
            })
        meshes.append({'name': name, 'primitives': prims})
        nodes.append({'name': name, 'mesh': len(meshes) - 1})
        size_cm = ', '.join(f'{v * 100:.1f}' for v in hi - lo)
        print(f'{name}: {size_cm} cm')

    gltf = {
        'asset': {'version': '2.0', 'generator': 'tools/make_props.py',
                  'copyright': 'Models from Poly Haven (CC0): ' + ', '.join(sorted(sources))},
        'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes, 'meshes': meshes,
        'materials': materials, 'textures': textures, 'images': images,
        'samplers': [{'magFilter': 9729, 'minFilter': 9987}],
        'accessors': accessors, 'bufferViews': views, 'buffers': [{'byteLength': len(out)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    out.extend(b'\0' * (-len(out) % 4))
    glb = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(out))
    glb += struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(out), 0x004E4942) + bytes(out)
    open(dst, 'wb').write(glb)
    print(f'{dst}: {len(glb) / 1024:.0f} KB')


if __name__ == '__main__':
    main(*sys.argv[1:3])
