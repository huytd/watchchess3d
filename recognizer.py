"""Square classifier: the same small CNN the browser version uses, run with OpenCV's dnn
module. The weights in model.json are handed to OpenCV as an in-memory Darknet model
(a plain-text layer list plus raw float32 weights), which runs the convolutions with
OpenCV's optimised kernels: far faster and lighter than doing them in numpy."""
import base64, json, os
import numpy as np
import cv2

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
TILE = 32
CHUNK = 64  # tiles per forward pass; OpenCV's buffers grow with the batch, so keep it to one board


def load_weights(path):
    m = json.load(open(path))
    flat = np.frombuffer(base64.b64decode(m['data']), dtype='<f2').astype(np.float32)
    w, off = [], 0
    for shape in m['shapes']:
        n = int(np.prod(shape))
        w.append(flat[off:off + n].reshape(shape))
        off += n
    return w


def darknet_model(w):
    """3x3 convs (+ReLU), 2x2 max pools, two dense layers and a softmax, as Darknet cfg + weights."""
    cfg = ['[net]', f'width={TILE}', f'height={TILE}', 'channels=3', 'batch=1', '']
    blobs = [np.array([0, 2, 0], np.int32).tobytes(), np.zeros(1, np.int64).tobytes()]  # version, images seen

    def conv(k, b):  # k: 3,3,C,F (HWIO) -> Darknet wants F,C,3,3
        cfg.extend(['[convolutional]', f'filters={k.shape[3]}', 'size=3', 'stride=1', 'pad=1', 'activation=relu', ''])
        blobs.extend([b.astype('<f4').tobytes(), k.transpose(3, 2, 0, 1).astype('<f4').tobytes()])

    def pool():
        cfg.extend(['[maxpool]', 'size=2', 'stride=2', ''])

    def dense(k, b, act):  # k: in,out -> Darknet wants out,in
        cfg.extend(['[connected]', f'output={k.shape[1]}', f'activation={act}', ''])
        blobs.extend([b.astype('<f4').tobytes(), k.T.astype('<f4').tobytes()])

    conv(w[0], w[1]); conv(w[2], w[3]); pool()
    conv(w[4], w[5]); conv(w[6], w[7]); pool()
    conv(w[8], w[9]); pool()
    # the model flattens 4x4x64 channels-last; Darknet flattens channels-first
    side = TILE // 8
    c = w[8].shape[3]
    dense(w[10].reshape(side, side, c, -1).transpose(2, 0, 1, 3).reshape(side * side * c, -1), w[11], 'relu')
    dense(w[12], w[13], 'linear')
    cfg.extend(['[softmax]', ''])
    return '\n'.join(cfg).encode(), b''.join(blobs)


class Recognizer:
    def __init__(self, path=os.path.join(os.path.dirname(__file__), 'model.json')):
        cfg, weights = darknet_model(load_weights(path))
        self.net = cv2.dnn.readNetFromDarknet(np.frombuffer(cfg, np.uint8), np.frombuffer(weights, np.uint8))

    def predict(self, tiles):
        """tiles: (N,32,32,3) uint8 RGB -> (N,13) probabilities."""
        out = np.empty((len(tiles), len(CLASSES)), np.float32)
        for i in range(0, len(tiles), CHUNK):
            part = tiles[i:i + CHUNK]
            self.net.setInput(cv2.dnn.blobFromImages(part, 1 / 255.0))
            out[i:i + len(part)] = self.net.forward().reshape(len(part), -1)
        return out


def board_tiles(img256):
    """(256,256,3) -> (64,32,32,3), top-left cell first."""
    return img256.reshape(8, TILE, 8, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(64, TILE, TILE, 3)
