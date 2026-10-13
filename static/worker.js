/* Runs the reading off the page's thread. Given an MP4 it does everything here, decoding the
   video with WebCodecs; otherwise the page steps a <video> through the video and posts each
   frame's pixels here. Either way this finds the board, classifies squares and follows the
   game. */
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.22.0/+esm';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@4.22.0/+esm';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.22.0/+esm';
import { setWasmPaths } from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/+esm';
import * as E from './engine.js';
import { makePredictor } from './cnn.js';
import { Mp4Source } from './mp4.js';
import { readSource } from './pipeline.js';

let predict = null, reader = null, loading = null;
setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');

/* Use whichever of the GPU (WebGL) and WebAssembly runs the CNN faster here: WebGL wins on
   most devices, but not where the GPU is slow or emulated. Timed on a full board and on the
   few squares a typical frame changes. */
async function pick(model){
  const tiles = new Uint8Array(64 * E.TILE * E.TILE * 3).map((_, i) => (i * 37) & 255);
  let best = null;
  for (const name of ['webgl', 'wasm', 'cpu']){
    if (best && name === 'cpu') break;
    try {
      if (!await tf.setBackend(name)) continue;
      await tf.ready();
      const p = makePredictor(tf, model, { buckets: name === 'webgl' });
      await p(tiles, 64); await p(tiles, 4);  // the first runs compile shaders
      let t = performance.now();
      await p(tiles, 64);
      for (let i = 0; i < 5; i++) await p(tiles, 4);
      t = performance.now() - t;
      p.dispose();
      if (!best || t < best.t) best = { name, t };
      if (name === 'webgl' && t < 100) break;  // a real GPU: nothing will beat it
    } catch (e) { console.warn('TensorFlow.js backend ' + name + ' unusable:', e); }
  }
  if (!best) throw new Error('worker_failed');
  await tf.setBackend(best.name);
  return makePredictor(tf, model, { buckets: best.name === 'webgl' });
}

function init(){
  loading = loading || (async () => {
    const model = await (await fetch(new URL('model.json', import.meta.url))).json();
    predict = await pick(model);
    return tf.getBackend();
  })();
  loading.catch(() => { loading = null; });
  return loading;
}

/* RGBA pixels of w x h -> RGB of W x H, each output pixel the average of the input area it
   covers (what ffmpeg's and OpenCV's "area" scaling do) */
function areaResize(src, w, h, W, H){
  const out = new Uint8Array(W * H * 3), acc = new Float64Array(W * 3);
  const sx = w / W, sy = h / H;
  for (let Y = 0; Y < H; Y++){
    acc.fill(0);
    const y0 = Y * sy, y1 = y0 + sy;
    for (let y = Math.floor(y0); y < Math.min(h, Math.ceil(y1)); y++){
      const wy = Math.min(y1, y + 1) - Math.max(y0, y);
      for (let X = 0; X < W; X++){
        const x0 = X * sx, x1 = x0 + sx;
        for (let x = Math.floor(x0); x < Math.min(w, Math.ceil(x1)); x++){
          const wgt = wy * (Math.min(x1, x + 1) - Math.max(x0, x)), o = (y * w + x) * 4;
          acc[X * 3] += wgt * src[o]; acc[X * 3 + 1] += wgt * src[o + 1]; acc[X * 3 + 2] += wgt * src[o + 2];
        }
      }
    }
    const area = sx * sy;
    for (let i = 0; i < W * 3; i++) out[Y * W * 3 + i] = Math.round(acc[i] / area);
  }
  return out;
}
const toBoard = img => areaResize(img.data, img.w, img.h, 256, 256);

/* the steps of a reading, done here */
const local = {
  /* a whole frame -> {score, box} of its strongest checkerboard, box in frame pixels;
     bigger frames are searched shrunk */
  search(img){
    let px = img.data, w = img.w, h = img.h, ch = 4;
    if (Math.min(w, h) > E.SEARCH_LINES){
      const k = E.SEARCH_LINES / Math.min(w, h);
      w = Math.round(img.w * k); h = Math.round(img.h * k);
      px = areaResize(img.data, img.w, img.h, w, h); ch = 3;
    }
    const [score, x, y, c] = E.checkerSearch(px, w, h, ch), kx = w / img.w, ky = h / img.h;
    return { score, box: [x / kx, y / ky, 8 * c / kx, 8 * c / ky] };
  },
  refine(img, box){ return E.refine(img.data, img.w, img.h, 4, box); },
  /* board crops spread through the video -> true if White is at the bottom, null if unclear */
  async orient(crops){ await init(); return E.detectOrientation(predict, crops.map(toBoard)); },
  async start(whiteBottom, decided){ await init(); reader = new E.Reader(predict, { whiteBottom, decided }); },
  frame(t, img){ return reader.frame(t, toBoard(img)); },
};

const cancelled = new Set();
const ops = {
  init,
  search: ({ img }) => local.search(img),
  refine: ({ img, box }) => local.refine(img, box),
  orient: ({ crops }) => local.orient(crops),
  start: ({ whiteBottom, decided }) => local.start(whiteBottom, decided),
  frame: ({ t, img }) => local.frame(t, img),
  /* read a whole MP4 here; {fallback: true} if it can't be, and the page uses a <video> */
  async read({ src, region, orientation }, id){
    const live = () => { if (cancelled.has(id)) throw new Error('cancelled'); };
    try {
      let source;
      try { source = await Mp4Source.open(src); } catch (e) { return { fallback: true, why: String(e && e.message || e) }; }
      live();
      await init(); live();
      // frames come from the decoder as fast as it goes; read them one at a time, in order,
      // with this reading's own reader (one called off may still have frames in hand)
      let chain = Promise.resolve(), mine = null;
      const steps = { ...local,
        async start(whiteBottom, decided){ mine = new E.Reader(predict, { whiteBottom, decided }); },
        frame: (t, img) => (chain = chain.then(() => mine.frame(t, toBoard(img)))) };
      await readSource(source, steps, { region, orientation, live, report: u => self.postMessage({ id, progress: u }) });
      return { done: true };
    } finally { cancelled.delete(id); }
  },
};

// one request at a time, in order: frames must reach the reader in sequence
let chain = Promise.resolve();
self.onmessage = e => {
  const { id, op, ...args } = e.data;
  if (op === 'cancel'){ cancelled.add(args.run); return; }
  chain = chain.then(() => ops[op](args, id)).then(
    result => self.postMessage({ id, result }),
    err => self.postMessage({ id, error: String(err && err.message || err) }));
};
