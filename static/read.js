/* Reads a video in the browser. An MP4 (every video the server downloads, most files) is
   decoded and read entirely in the worker with WebCodecs. Anything else, or a browser
   without WebCodecs, falls back to stepping a hidden <video> through the video ~3 times a
   second here and posting each board crop to the worker. A shared tab is read as it plays
   (readScreen). */
import { readSource } from './pipeline.js';
import { FPS } from './engine.js';

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
  start: (whiteBottom, decided, fixed) => call('start', { whiteBottom, decided, fixed }),
  frame: (t, img) => send('frame', img, { t }),
  live: (t, img, minGap) => send('live', img, { t, minGap }),
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

/* ---------------------------------------------------------------- a shared tab, as it plays */
const LIVE_LINES = 360;  // a shared tab is searched for the board shrunk to this many lines
const sleep = ms => new Promise(res => setTimeout(res, ms));
let small = null, sctx = null;
/* the current frame shrunk so its shorter side is about `lines` pixels; k: the scale */
function grabSmall(v, lines){
  const k = Math.min(1, lines / Math.min(v.videoWidth, v.videoHeight));
  const w = Math.round(v.videoWidth * k), h = Math.round(v.videoHeight * k);
  if (!small){ small = document.createElement('canvas'); sctx = small.getContext('2d', { willReadFrequently: true }); }
  if (small.width !== w || small.height !== h){ small.width = w; small.height = h; }
  sctx.imageSmoothingQuality = 'high';
  sctx.drawImage(v, 0, 0, w, h);
  return { w, h, k, data: sctx.getImageData(0, 0, w, h).data };
}
const close = (a, b) => [0, 1, 2, 3].every(i => Math.abs(a[i] - b[i]) < 0.04 * a[2]);

/* Reads a tab (or window, or screen) being shared, as it plays: v is a playing <video> showing
   it. Finds the board, then reads it FPS times a second, with t in seconds since the reading
   began. Each board crop is checked against the board it was found on first: while it doesn't
   look like it (an ad, a change of video, the page scrolled), frames are skipped, and after a
   second of that the board is looked for again; so it is when the shared tab changes size.
   region, orientation and on(update) as for readVideo, with status 'live' while reading and
   {lost: true} when the board is being looked for again. A region drawn by hand is kept until
   the shared tab changes size. Returns a function that stops it. */
export function readScreen(v, { region = null, orientation = 'auto', on }){
  let stopped = false;
  const live = () => { if (stopped) throw new Cancelled(); };
  const report = u => { if (!stopped) on(u); };
  const t0 = performance.now();
  (async () => {
    report({ status: 'loading' });
    await remote.start(orientation !== 'b', orientation !== 'auto', orientation !== 'auto'); live();
    let manual = !!region, size = '', crop = null, minGap = 0, misses = 0, prev = null, status = 'loading';
    const say = s => { if (s !== status) report({ status: (status = s) }); };
    const lose = () => { region = crop = prev = null; manual = false; report({ lost: true }); };
    for (;;){
      const tick = performance.now(), W = v.videoWidth, H = v.videoHeight;
      if (!W || !H || v.readyState < 2){ await sleep(250); live(); continue; }
      if (W + 'x' + H !== size){
        if (size && region) lose();
        size = W + 'x' + H;
      }
      if (!region){
        say('finding_board');
        const img = grabSmall(v, LIVE_LINES), k = img.k;
        const r = await remote.search(img); live();
        const box = r.box.map(x => x / k), found = r.score > 12 && box[2] > 0;
        // the same board twice running, so a passing frame can't set it
        if (!(found && prev && close(prev, box))){
          prev = found ? box : null;
          await sleep(400); live();
          continue;
        }
        region = await remote.refine(grab(v), box); live();
      }
      if (!crop){
        const x = Math.max(0, Math.round(region[0])), y = Math.max(0, Math.round(region[1]));
        crop = [x, y, Math.max(8, Math.min(W - x, Math.round(region[2]))), Math.max(8, Math.min(H - y, Math.round(region[3])))];
        minGap = 0; misses = 0;
        report({ region, width: W, height: H, duration: 0 });
        say('live');
      }
      const r = await remote.live((performance.now() - t0) / 1000, grab(v, crop), minGap); live();
      if (!minGap) minGap = Math.max(4, 0.45 * r.gap);
      if (r.events){
        misses = 0;
        report({ events: r.events, marks: r.marks, whiteBottom: r.decided ? r.whiteBottom : null });
      } else if (++misses >= FPS && !manual){ lose(); continue; }
      await sleep(Math.max(0, 1000 / FPS - (performance.now() - tick))); live();
    }
  })().catch(e => { if (!stopped && !(e instanceof Cancelled)) report({ status: 'error', error: e.message }); });
  return () => { stopped = true; };
}
