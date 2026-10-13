"""Build static/pieces.glb from Poly Haven's "Chess Set" by Riley Queen (CC0):
https://polyhaven.com/a/chess_set

Download the glTF version (1k is plenty) and unzip it, then:

    python tools/make_pieces.py path/to/chess_set_1k.gltf static/pieces.glb

The source has 32 pieces; the black ones are the white ones turned 180 degrees with the
same UVs, and both colours share their normal and roughness maps, so this keeps one mesh per
piece type, two materials and four textures (shrunk to 512px). Nodes are named by FEN letter
(P N B R Q K for white, p n b r q k for black), every piece standing on y = 0 in metres and
facing the way the source's white pieces do.
"""
import io
import json
import os
import struct
import sys

from PIL import Image

SRC_NODES = {'p': 'piece_pawn_white_01', 'n': 'piece_knight_white_01', 'b': 'piece_bishop_white_01',
             'r': 'piece_rook_white_01', 'q': 'piece_queen_white', 'k': 'piece_king_white'}
TEX_SIZE = 512


def main(src, dst):
    g = json.load(open(src))
    folder = os.path.dirname(src)
    data = open(os.path.join(folder, g['buffers'][0]['uri']), 'rb').read()
    out = bytearray()
    views, accessors = [], []

    def add_view(blob, target=None):
        while len(out) % 4:
            out.append(0)
        views.append({'buffer': 0, 'byteOffset': len(out), 'byteLength': len(blob)})
        if target:
            views[-1]['target'] = target
        out.extend(blob)
        return len(views) - 1

    def copy_accessor(i, target):
        a = g['accessors'][i]
        bv = g['bufferViews'][a['bufferView']]
        n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[a['type']]
        size = {5126: 4, 5123: 2, 5125: 4}[a['componentType']] * n
        assert bv.get('byteStride', size) == size
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        acc = {k: a[k] for k in ('componentType', 'count', 'type', 'min', 'max') if k in a}
        acc['bufferView'] = add_view(data[start:start + size * a['count']], target)
        accessors.append(acc)
        return len(accessors) - 1

    by_name = {n['name']: n for n in g['nodes']}
    prims = {}
    for ch, name in SRC_NODES.items():
        prims[ch] = []
        for p in g['meshes'][by_name[name]['mesh']]['primitives']:
            attrs = {k: copy_accessor(p['attributes'][k], 34962) for k in ('POSITION', 'NORMAL', 'TEXCOORD_0')}
            prims[ch].append({'attributes': attrs, 'indices': copy_accessor(p['indices'], 34963)})

    def add_image(path):
        im = Image.open(os.path.join(folder, path)).convert('RGB').resize((TEX_SIZE, TEX_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=88, optimize=True)
        return {'bufferView': add_view(buf.getvalue()), 'mimeType': 'image/jpeg'}

    tex = 'textures/chess_set_pieces_%s_1k.jpg'
    images = [add_image(tex % 'white_diff'), add_image(tex % 'black_diff'),
              add_image(tex % 'white_nor_gl'), add_image(tex % 'white_arm')]

    def material(name, diffuse):
        return {'name': name, 'normalTexture': {'index': 2}, 'occlusionTexture': {'index': 3},
                'pbrMetallicRoughness': {'baseColorTexture': {'index': diffuse}, 'metallicFactor': 0,
                                         'metallicRoughnessTexture': {'index': 3}}}

    meshes, nodes = [], []
    for colour, mat in (('w', 0), ('b', 1)):
        for ch in 'pnbrqk':
            name = ch.upper() if colour == 'w' else ch
            meshes.append({'name': name, 'primitives': [dict(p, material=mat) for p in prims[ch]]})
            nodes.append({'name': name, 'mesh': len(meshes) - 1})

    gltf = {
        'asset': {'version': '2.0', 'generator': 'tools/make_pieces.py',
                  'copyright': 'Chess Set by Riley Queen, Poly Haven (CC0)'},
        'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes, 'meshes': meshes,
        'materials': [material('white', 0), material('black', 1)],
        'textures': [{'sampler': 0, 'source': i} for i in range(len(images))],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987}],
        'images': images, 'accessors': accessors, 'bufferViews': views,
        'buffers': [{'byteLength': len(out)}],
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
