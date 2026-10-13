/* Frames from an MP4 file without a <video> element: the sample table is read from the file's
   index (its 'moov' box) and the video is decoded with WebCodecs. Works in a worker, decodes
   each frame once when reading straight through (seeking a <video> decodes from the last
   keyframe every time), and doesn't stop in a background tab. Only plain (non-fragmented)
   MP4/MOV with H.264, VP9 or AV1 video; anything else throws 'unsupported' and the caller
   falls back to a <video>. */

/* ---------------------------------------------------------------- reading the index */
function boxes(buf, start = 0, end = buf.byteLength){
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), out = [];
  let p = start;
  while (p + 8 <= end){
    let size = dv.getUint32(p), head = 8;
    const type = String.fromCharCode(buf[p + 4], buf[p + 5], buf[p + 6], buf[p + 7]);
    if (size === 1){ size = Number(dv.getBigUint64(p + 8)); head = 16; }
    else if (size === 0) size = end - p;
    if (size < head || p + size > end) break;
    out.push({ type, start: p + head, end: p + size });
    p += size;
  }
  return out;
}
const child = (buf, box, type) => box && boxes(buf, box.start, box.end).find(b => b.type === type);
const path = (buf, box, ...types) => types.reduce((b, t) => child(buf, b, t), box);
const hex2 = n => n.toString(16).padStart(2, '0');
const pad2 = n => String(n).padStart(2, '0');

/* the codec string and decoder description from a visual sample entry */
function codecOf(buf, entry){
  const kids = boxes(buf, entry.start + 78, entry.end);  // after the fixed VisualSampleEntry fields
  const get = t => kids.find(b => b.type === t);
  const body = b => buf.slice(b.start, b.end);
  if (['avc1', 'avc3'].includes(entry.type) && get('avcC')){
    const c = get('avcC');
    return { codec: 'avc1.' + hex2(buf[c.start + 1]) + hex2(buf[c.start + 2]) + hex2(buf[c.start + 3]), description: body(c) };
  }
  if (entry.type === 'vp09' && get('vpcC')){
    const c = get('vpcC').start + 4;  // after version and flags
    return { codec: `vp09.${pad2(buf[c])}.${pad2(buf[c + 1])}.${pad2(buf[c + 2] >> 4)}` };
  }
  if (entry.type === 'av01' && get('av1C')){
    const c = get('av1C'), b1 = buf[c.start + 1], b2 = buf[c.start + 2];
    const depth = b2 & 0x40 ? (b2 & 0x20 ? 12 : 10) : 8;
    return { codec: `av01.${b1 >> 5}.${pad2(b1 & 31)}${b2 & 0x80 ? 'H' : 'M'}.${pad2(depth)}`, description: body(c) };
  }
  return null;
}

/* moov box bytes -> the first video track's codec and samples (in decode order, with
   presentation times in seconds, edit list applied) */
export function parseMoov(buf){
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const moov = { start: 0, end: buf.byteLength };
  const mvhd = child(buf, moov, 'mvhd');
  const movieScale = mvhd ? dv.getUint32(mvhd.start + (buf[mvhd.start] === 1 ? 20 : 12)) : 1000;
  for (const trak of boxes(buf).filter(b => b.type === 'trak')){
    const mdia = child(buf, trak, 'mdia'), hdlr = child(buf, mdia, 'hdlr');
    if (!hdlr || String.fromCharCode(...buf.subarray(hdlr.start + 8, hdlr.start + 12)) !== 'vide') continue;
    const mdhd = child(buf, mdia, 'mdhd'), v1 = buf[mdhd.start] === 1;
    const scale = dv.getUint32(mdhd.start + (v1 ? 20 : 12));
    const stbl = path(buf, mdia, 'minf', 'stbl');
    const stsd = child(buf, stbl, 'stsd');
    const entry = boxes(buf, stsd.start + 8, stsd.end)[0];
    const codec = entry && codecOf(buf, entry);
    if (!codec) throw new Error('unsupported');
    const width = dv.getUint16(entry.start + 24), height = dv.getUint16(entry.start + 26);
    const full = b => b && { v: buf[b.start], n: dv.getUint32(b.start + 4), at: b.start + 8 };
    // decode times and composition offsets
    const stsz = child(buf, stbl, 'stsz');
    const fixed = dv.getUint32(stsz.start + 4), count = dv.getUint32(stsz.start + 8);
    if (!count) throw new Error('unsupported');  // fragmented: the samples live elsewhere
    const size = i => fixed || dv.getUint32(stsz.start + 12 + 4 * i);
    const dts = new Float64Array(count), cts = new Float64Array(count);
    let i = 0, t = 0;
    const stts = full(child(buf, stbl, 'stts'));
    for (let e = 0; e < stts.n; e++){
      const n = dv.getUint32(stts.at + 8 * e), d = dv.getUint32(stts.at + 8 * e + 4);
      for (let k = 0; k < n && i < count; k++, i++){ dts[i] = t; t += d; }
    }
    for (; i < count; i++) dts[i] = t;
    cts.set(dts);
    const ctts = full(child(buf, stbl, 'ctts'));
    if (ctts){
      i = 0;
      for (let e = 0; e < ctts.n; e++){
        const n = dv.getUint32(ctts.at + 8 * e), o = ctts.v ? dv.getInt32(ctts.at + 8 * e + 4) : dv.getUint32(ctts.at + 8 * e + 4);
        for (let k = 0; k < n && i < count; k++, i++) cts[i] += o;
      }
    }
    const stss = full(child(buf, stbl, 'stss'));
    const sync = new Uint8Array(count).fill(stss ? 0 : 1);
    if (stss) for (let e = 0; e < stss.n; e++) sync[dv.getUint32(stss.at + 4 * e) - 1] = 1;
    // where each sample is: chunks of consecutive samples
    const stco = child(buf, stbl, 'stco'), co64 = child(buf, stbl, 'co64');
    const chunks = full(stco || co64);
    const chunkAt = c => co64 ? Number(dv.getBigUint64(chunks.at + 8 * c)) : dv.getUint32(chunks.at + 4 * c);
    const stsc = full(child(buf, stbl, 'stsc'));
    const offset = new Float64Array(count), sizes = new Uint32Array(count);
    i = 0;
    for (let e = 0; e < stsc.n; e++){
      const first = dv.getUint32(stsc.at + 12 * e) - 1, per = dv.getUint32(stsc.at + 12 * e + 4);
      const last = e + 1 < stsc.n ? dv.getUint32(stsc.at + 12 * (e + 1)) - 1 : chunks.n;
      for (let c = first; c < last && i < count; c++){
        let o = chunkAt(c);
        for (let k = 0; k < per && i < count; k++, i++){ offset[i] = o; sizes[i] = size(i); o += sizes[i]; }
      }
    }
    // the edit list says which media time is shown at time 0 (B-frame delay, trimmed starts)
    let shift = 0;
    const elst = full(path(buf, trak, 'edts', 'elst'));
    if (elst){
      let empty = 0;
      for (let e = 0, p = elst.at; e < elst.n; e++){
        const dur = elst.v ? Number(dv.getBigUint64(p)) : dv.getUint32(p);
        const media = elst.v ? Number(dv.getBigInt64(p + 8)) : dv.getInt32(p + 4);
        p += elst.v ? 20 : 12;
        if (media === -1){ empty += dur / movieScale; continue; }
        shift = media / scale - empty;
        break;
      }
    }
    const samples = Array.from({ length: count }, (_, k) => ({
      offset: offset[k], size: sizes[k], dts: dts[k] / scale - shift, pts: cts[k] / scale - shift, sync: !!sync[k],
      duration: (k + 1 < count ? dts[k + 1] - dts[k] : 0) / scale,
    }));
    const end = Math.max(...samples.map(s => s.pts + s.duration));
    return { ...codec, width, height, samples, duration: end > 0 ? end : 0 };
  }
  throw new Error('unsupported');
}

/* read(start, end) -> Promise<Uint8Array>, size: the file's length. Finds and parses moov. */
async function readIndex(read, size){
  let p = 0;
  while (p + 8 <= size){
    const h = await read(p, Math.min(size, p + 16));
    const box = boxes(h, 0, h.length)[0] || (() => {
      // a box too big for the 16 bytes read: only its header matters here
      const dv = new DataView(h.buffer, h.byteOffset, h.byteLength);
      let s = dv.getUint32(0), head = 8;
      if (s === 1){ s = Number(dv.getBigUint64(8)); head = 16; } else if (s === 0) s = size - p;
      return { type: String.fromCharCode(...h.subarray(4, 8)), start: head, end: s };
    })();
    if (box.end <= 0) break;
    if (box.type === 'moov') return parseMoov(await read(p + box.start, p + box.end));
    p += box.end;
  }
  throw new Error('unsupported');
}

/* a File/Blob, or a URL the server answers range requests for */
function reader(src){
  if (typeof src !== 'string'){
    return { size: src.size, read: async (a, b) => new Uint8Array(await src.slice(a, b).arrayBuffer()) };
  }
  return {
    size: null,
    async read(a, b){
      const r = await fetch(src, { headers: { Range: `bytes=${a}-${b - 1}` } });
      if (r.status !== 206 && !(r.status === 200 && a === 0)) throw new Error('unsupported');
      const buf = new Uint8Array(await r.arrayBuffer());
      if (!this.size){
        const m = /\/(\d+)$/.exec(r.headers.get('Content-Range') || '');
        this.size = m ? +m[1] : buf.length;
      }
      return buf.subarray(0, b - a);
    },
  };
}

/* ---------------------------------------------------------------- decoding */
export class Mp4Source {
  static async open(src){
    if (typeof VideoDecoder === 'undefined' || typeof OffscreenCanvas === 'undefined') throw new Error('unsupported');
    const io = reader(src);
    if (io.size === null) await io.read(0, 8);  // learns the size
    const index = await readIndex((a, b) => io.read(a, b), io.size);
    const config = { codec: index.codec, codedWidth: index.width, codedHeight: index.height,
      ...(index.description ? { description: index.description } : {}), optimizeForLatency: false };
    const ok = await VideoDecoder.isConfigSupported(config).catch(() => null);
    if (!ok || !ok.supported) throw new Error('unsupported');
    const s = new Mp4Source();
    Object.assign(s, { io, config, samples: index.samples, duration: index.duration, width: 0, height: 0 });
    // the decode order index of each sample, sorted by presentation time
    s.byPts = s.samples.map((_, i) => i).sort((a, b) => s.samples[a].pts - s.samples[b].pts);
    // the picture size as a <video> would show it (pixel aspect applied): from a frame
    const f = await s.decodeAround(s.byPts[0]);
    s.width = f.displayWidth; s.height = f.displayHeight;
    f.close();
    return s;
  }

  canvas(w, h){
    if (!this.cv){ this.cv = new OffscreenCanvas(w, h); this.cx = this.cv.getContext('2d', { willReadFrequently: true }); }
    if (this.cv.width !== w || this.cv.height !== h){ this.cv.width = w; this.cv.height = h; }
    return this.cx;
  }

  /* a VideoFrame's pixels (RGBA), all of it or the area [x, y, w, h] in whole pixels */
  pixels(frame, area){
    const [x, y, w, h] = area || [0, 0, this.width, this.height];
    const cx = this.canvas(w, h);
    cx.drawImage(frame, -x, -y, this.width, this.height);
    return { w, h, data: cx.getImageData(0, 0, w, h).data };
  }

  /* fetch samples [i, j) (decode order) in a few big reads, as EncodedVideoChunks */
  async chunks(i, j){
    const out = [];
    for (let a = i; a < j;){
      // samples next to each other in the file go in one read of up to ~4 MB
      let b = a + 1;
      while (b < j && this.samples[b].offset === this.samples[b - 1].offset + this.samples[b - 1].size &&
             this.samples[b].offset + this.samples[b].size - this.samples[a].offset < 4e6) b++;
      const s0 = this.samples[a], last = this.samples[b - 1];
      const buf = await this.io.read(s0.offset, last.offset + last.size);
      for (let k = a; k < b; k++){
        const s = this.samples[k], o = s.offset - s0.offset;
        out.push(new EncodedVideoChunk({ type: s.sync ? 'key' : 'delta', timestamp: Math.round(s.pts * 1e6),
          duration: Math.round(s.duration * 1e6), data: buf.subarray(o, o + s.size) }));
      }
      a = b;
    }
    return out;
  }

  /* the decoded frame of sample k (decode from the keyframe before it) */
  async decodeAround(k){
    let key = k;
    while (key > 0 && !this.samples[key].sync) key--;
    const want = Math.round(this.samples[k].pts * 1e6);
    let got = null, error = null;
    const dec = new VideoDecoder({
      output: f => { if (!got && f.timestamp === want) got = f; else f.close(); },
      error: e => { error = e; },
    });
    try {
      dec.configure(this.config);
      for (const c of await this.chunks(key, k + 1)) dec.decode(c);
      await dec.flush();
    } finally { if (dec.state !== 'closed') dec.close(); }
    if (!got) throw error || new Error('unplayable');
    return got;
  }

  /* the frame shown at time t, as RGBA pixels */
  async frameAt(t, area){
    // the last frame starting at or before t
    let lo = 0, hi = this.byPts.length - 1, k = this.byPts[0];
    while (lo <= hi){
      const mid = (lo + hi) >> 1;
      if (this.samples[this.byPts[mid]].pts <= t + 1e-6){ k = this.byPts[mid]; lo = mid + 1; } else hi = mid - 1;
    }
    const f = await this.decodeAround(k);
    try { return this.pixels(f, area); } finally { f.close(); }
  }

  /* Decode straight through; for each time k/fps hand onFrame(t, pixels) the frame nearest
   it. onFrame's promises are awaited a few at a time, holding the decoding back. live() is
   called often and throws to stop. */
  async scan(area, fps, onFrame, live){
    const pending = [];
    let k = 0, prev = null, error = null, wake = null;
    const nudge = () => { if (wake){ const w = wake; wake = null; w(); } };
    const emit = (f, t) => { pending.push(onFrame(t, this.pixels(f, area))); };
    const dec = new VideoDecoder({
      output: f => {
        const ts = f.timestamp / 1e6;
        while (k / fps < this.duration && k / fps <= ts){
          const t = k / fps;
          emit(prev && t - prev.timestamp / 1e6 <= ts - t ? prev : f, t);
          k++;
        }
        if (prev) prev.close();
        prev = f;
        nudge();
      },
      error: e => { error = e; nudge(); },
    });
    dec.addEventListener?.('dequeue', nudge);
    const settle = async () => {
      // hold back while the decoder or the reading is behind
      while (!error && (dec.decodeQueueSize > 8 || pending.length > 6)){
        if (pending.length > 6) await pending.shift();
        else await new Promise(r => { wake = r; setTimeout(nudge, 20); });
        live();
      }
      if (error) throw error;
    };
    try {
      dec.configure(this.config);
      const n = this.samples.length;
      for (let i = 0; i < n;){
        // a few hundred samples at a time, so the reads stay a sensible size
        let j = Math.min(n, i + 120);
        const batch = await this.chunks(i, j);
        live();
        for (const c of batch){ await settle(); dec.decode(c); }
        i = j;
      }
      await dec.flush();
      if (error) throw error;
      for (; prev && k / fps < this.duration; k++) emit(prev, k / fps);
      await Promise.all(pending);
    } finally {
      if (prev) prev.close();
      if (dec.state !== 'closed') dec.close();
    }
  }
}
