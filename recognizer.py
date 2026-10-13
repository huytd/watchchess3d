"""Square classifier: the same small CNN the browser version uses, run with plain numpy."""
import base64, json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
TILE = 32


class Recognizer:
    def __init__(self, path=os.path.join(os.path.dirname(__file__), 'model.json')):
        m = json.load(open(path))
        flat = np.frombuffer(base64.b64decode(m['data']), dtype='<f2').astype(np.float32)
        self.w, off = [], 0
        for shape in m['shapes']:
            n = int(np.prod(shape))
            self.w.append(flat[off:off + n].reshape(shape))
            off += n

    @staticmethod
    def _conv(x, k, b):
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        win = sliding_window_view(xp, (3, 3), axis=(1, 2))           # N,H,W,C,3,3
        y = np.tensordot(win, k, axes=([3, 4, 5], [2, 0, 1])) + b     # k: 3,3,C,F
        return np.maximum(y, 0, out=y)

    @staticmethod
    def _pool(x):
        n, h, w, c = x.shape
        return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))

    def predict(self, tiles):
        """tiles: (N,32,32,3) uint8 RGB -> (N,13) probabilities."""
        w = self.w
        x = tiles.astype(np.float32) / 255.0
        x = self._conv(x, w[0], w[1]); x = self._conv(x, w[2], w[3]); x = self._pool(x)
        x = self._conv(x, w[4], w[5]); x = self._conv(x, w[6], w[7]); x = self._pool(x)
        x = self._conv(x, w[8], w[9]); x = self._pool(x)
        x = x.reshape(len(x), -1)
        x = np.maximum(x @ w[10] + w[11], 0)
        z = x @ w[12] + w[13]
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)


def board_tiles(img256):
    """(256,256,3) -> (64,32,32,3), top-left cell first."""
    return img256.reshape(8, TILE, 8, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(64, TILE, TILE, 3)
