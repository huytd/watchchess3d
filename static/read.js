/* Reads a video in the browser. An MP4 (every video the server downloads, most files) is
   decoded and read entirely in the worker with WebCodecs. Anything else, or a browser
   without WebCodecs, falls back to stepping a hidden <video> through the video ~3 times a
   second here and posting each board crop to the worker. */
import { readSource } from './pipeline.js';

let worker = null, seq = 0;
const waiting = new Map();
function call(op, args = {}, transfer = [], onProgress = null){
  if (!worker){
    worker = new Worker(new URL('worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = e => {
      const w = waiting.get(e.data.id);
      if (!w) return;
      if (e.data.progress){ if (w.onProgress) w.onProgress(e.data.progress); return; }
      waiting.delete(e.data.id);
      if (e.data.error) w.reject(new Error(e.data.error)); else w.resolve(e.data.result);
    };
    // e.g. TensorFlow.js couldn't be fetched: start over with a fresh worker next time
    worker.onerror = () => {
      for (const w of waiting.values()) w.reject(new Error('worker_failed'));
      waiting.clear();
      worker.terminate(); worker = null;
    };
  }
  const id = ++seq;
  const p = new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, op, ...args }, transfer);
  });
  p.id = id;
  return p;
}
/* start loading TensorFlow.js and the model early, while the video downloads */
export const warmUp = () => call('init').catch(() => null);

const once = (el, ok, bad) => new Promise((resolve, reject) => {
  const done = f => e => { el.removeEventListener(ok, a); if (bad) el.removeEventListener(bad, b); f(e); };
  const a = done(resolve), b = done(() => reject(new Error('unplayable')));
  el.addEventListener(ok, a);
  if (bad) el.addEventListener(bad, b);
});

/* a muted, hidden <video> for grabbing frames (kept in the page: some browsers won't load
   or seek one that isn't) */
export async function openVideo(src){
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  v.setAttribute('aria-hidden', 'true');
  Object.assign(v.style, { position: 'fixed', left: '0', top: '0', width: '2px', height: '2px', opacity: '0', pointerEvents: 'none' });
  document.body.appendChild(v);
  const ready = once(v, 'loadeddata', 'error');
  v.src = src;
  try {
    await ready;
    // recordings often say nothing about their length until the end has been looked for
    if (!isFinite(v.duration)){
      const known = once(v, 'durationchange');
      v.currentTime = 1e9;
      await known;
    }
  } catch (e) { closeVideo(v); throw e; }
  return v;
}
export function closeVideo(v){ v.removeAttribute('src'); v.load(); v.remove(); }

export async function seek(v, t){
  t = Math.min(Math.max(0, t), v.duration);
  if (v.currentTime === t && !v.seeking) return;
  const done = once(v, 'seeked', 'error');
  v.currentTime = t;
  await done;
}

let canvas = null, ctx = null;
/* the current frame's pixels (RGBA), all of it or the area [x, y, w, h] in whole pixels */
export function grab(v, area){
  const [x, y, w, h] = area || [0, 0, v.videoWidth, v.videoHeight];
  if (!canvas){ canvas = document.createElement('canvas'); ctx = canvas.getContext('2d', { willReadFrequently: true }); }
  if (canvas.width !== w || canvas.height !== h){ canvas.width = w; canvas.height = h; }
  ctx.drawImage(v, x, y, w, h, 0, 0, w, h);
  return { w, h, data: ctx.getImageData(0, 0, w, h).data };
}

/* frames from a <video>, by seeking (pipeline.js's source) */
class VideoSource {
  constructor(v){ Object.assign(this, { v, width: v.videoWidth, height: v.videoHeight, duration: v.duration }); }
  async frameAt(t, area){ await seek(this.v, t); return grab(this.v, area); }
  async scan(area, fps, onFrame, live){
    // keep a few frames queued so the worker never waits on the seeking
    const queue = [];
    for (let k = 0; k / fps < this.duration; k++){
      const t = k / fps;
      // a hair past the sample time, so a frame that starts exactly there is the one shown
      await seek(this.v, t + 1e-3); live();
      queue.push(onFrame(t, grab(this.v, area)));
      if (queue.length >= 4) await queue.shift();
    }
    await Promise.all(queue);
  }
}
/* the worker's steps, run from here */
const send = (op, img, extra = {}) => call(op, { img, ...extra }, [img.data.buffer]);
const remote = {
  search: img => send('search', img),
  refine: (img, box) => send('refine', img, { box }),
  orient: crops => call('orient', { crops }, crops.map(c => c.data.buffer)),
  start: (whiteBottom, decided) => call('start', { whiteBottom, decided }),
  frame: (t, img) => send('frame', img, { t }),
};

class Cancelled extends Error {}

/* Reads a video: src is a URL, file the File it came from (if any). region: [x, y, w, h] in
   video pixels or null to find the board; orientation: 'auto' | 'w' | 'b'. on(update) gets
   {status} changes and {region, width, height, duration}, {events, marks, whiteBottom,
   progress} as reading goes. Returns a function that stops it. */
export function readVideo(src, { file = null, region = null, orientation = 'auto', on }){
  let stopped = false, run = null;
  const live = () => { if (stopped) throw new Cancelled(); };
  const report = u => { if (!stopped) on(u); };
  (async () => {
    report({ status: 'loading' });
    // an MP4 is read entirely in the worker
    run = call('read', { src: file || new URL(src, location.href).href, region, orientation }, [], report);
    const r = await run;
    run = null;
    live();
    if (!r.fallback) return;
    const v = await openVideo(src);
    try {
      live();
      await readSource(new VideoSource(v), remote, { region, orientation, report, live });
    } finally { closeVideo(v); }
  })().catch(e => { if (!stopped && !(e instanceof Cancelled)) report({ status: 'error', error: e.message }); });
  return () => {
    stopped = true;
    if (run && worker) worker.postMessage({ op: 'cancel', run: run.id });
  };
}
