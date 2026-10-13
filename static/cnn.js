/* The square classifier: a small CNN (3x3 convs + ReLU, 2x2 max pools, two dense layers
   and a softmax) run with TensorFlow.js. The weights in model.json are already in
   TensorFlow's layout (HWIO kernels, channels-last flatten). */
import { decodeWeights, TILE } from './engine.js';

export function makePredictor(tf, model, { buckets = false } = {}){
  const w = decodeWeights(model).map(({ shape, data }) => tf.tensor(data, shape));
  const conv = (x, i) => tf.relu(tf.add(tf.conv2d(x, w[i], 1, 'same'), w[i + 1]));
  const pool = x => tf.maxPool(x, 2, 2, 'valid');
  const run = async (tiles, n) => {
    // on the GPU every new batch size compiles new shaders: round up to a power of two
    const m = buckets ? 2 ** Math.ceil(Math.log2(n)) : n;
    const input = new Float32Array(m * TILE * TILE * 3);
    for (let i = 0; i < n * TILE * TILE * 3; i++) input[i] = tiles[i] / 255;
    const out = tf.tidy(() => {
      let x = tf.tensor4d(input, [m, TILE, TILE, 3]);
      x = pool(conv(conv(x, 0), 2));
      x = pool(conv(conv(x, 4), 6));
      x = pool(conv(x, 8));
      x = tf.relu(tf.add(tf.matMul(tf.reshape(x, [m, -1]), w[10]), w[11]));
      return tf.softmax(tf.add(tf.matMul(x, w[12]), w[13]));
    });
    const data = await out.data();
    out.dispose();
    return data.subarray(0, n * 13);
  };
  /* tiles: n x 32x32 RGB bytes -> Float32Array of n x 13 probabilities */
  const predict = (tiles, n) => {
    const parts = [];
    // keep each batch to a few boards, so a long list doesn't need one huge buffer
    for (let i = 0; i < n; i += 256) parts.push(run(tiles.subarray(i * TILE * TILE * 3), Math.min(256, n - i)));
    return Promise.all(parts).then(ps => {
      if (ps.length === 1) return ps[0];
      const out = new Float32Array(n * 13);
      ps.forEach((p, i) => out.set(p, i * 256 * 13));
      return out;
    });
  };
  predict.dispose = () => w.forEach(t => t.dispose());
  return predict;
}
