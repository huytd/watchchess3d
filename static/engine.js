/* Video frames -> timeline of chess positions, in the browser. A port of the server's old
   pipeline.py and marks.py; the browser only hands it pixels and a square classifier.

   1. find the board: search every size and position for the strongest 8x8 checkerboard
      pattern, then fine-tune the edges so each cell's rim sits inside one square
   2. read the board ~3 times a second with the CNN (only the squares that changed)
   3. turn noisy per-frame readings into a clean sequence of positions, preferring legal
      moves so a shaky square can't derail the game
   4. alongside, pick out the arrows and circles streamers draw on the board

   Images are flat arrays of pixels, row by row, with `ch` channels (4 for canvas RGBA, 3
   for RGB). Board crops are 256x256 RGB, cells indexed top-left first. */
import { Position, squareName, sameMove } from './rules.js';

export const CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
const CLASS_INDEX = Object.fromEntries(CLASSES.map((c, i) => [c, i]));
export const TILE = 32;
export const FPS = 3;
export const SEARCH_LINES = 720;  // bigger frames are searched shrunk to this; refine() works at full size

/* OpenCV's RGB -> grey, in the same fixed point */
const grey = (r, g, b) => (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;

/* numpy-style helpers */
function rint(x){  // round half to even
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
function linspace(a, b, n, endpoint = true){
  const div = endpoint ? n - 1 : n, step = (b - a) / div, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = i * step + a;
  if (endpoint && n > 1) out[n - 1] = b;
  return out;
}
function median(vals){
  const s = Float64Array.from(vals).sort(), n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function argmax(a, start = 0, n = a.length - start){
  let k = 0;
  for (let i = 1; i < n; i++) if (a[start + i] > a[start + k]) k = i;
  return k;
}

/* ---------------------------------------------------------------- board detection */
/* Best [score, x, y, cell] for an 8x8 checkerboard of whole-pixel cells. */
export function checkerSearch(px, W, H, ch = 4){
  const W1 = W + 1, ii = new Float64Array((H + 1) * W1);
  for (let y = 0; y < H; y++){
    let row = 0;
    for (let x = 0; x < W; x++){
      const o = (y * W + x) * ch;
      row += grey(px[o], px[o + 1], px[o + 2]);
      ii[(y + 1) * W1 + x + 1] = ii[y * W1 + x + 1] + row;
    }
  }
  let best = [0, 0, 0, 0];
  const lo = Math.max(6, Math.floor(Math.min(H, W) * 0.035)), hi = Math.floor(Math.min(H, W) / 8);
  const B = new Float64Array(H * W1), C = new Float64Array(H * W1), R = new Float64Array(H * W1);
  for (let c = lo; c <= hi; c++){
    // sum of every c x c box, indexed by its top-left corner
    const Bh = H + 1 - c, Bw = W + 1 - c;
    for (let y = 0; y < Bh; y++){
      const a = y * W1, b = (y + c) * W1, o = y * Bw;
      for (let x = 0; x < Bw; x++) B[o + x] = ii[b + x + c] - ii[a + x + c] - ii[b + x] + ii[a + x];
    }
    const ny = Bh - 7 * c, nx = Bw - 7 * c;
    if (ny <= 0 || nx <= 0) continue;
    // the checkerboard's +-1 pattern is (-1)^row * (-1)^file, so sum along the files
    // first and then down the rows: 16 passes over the image instead of 64
    for (let y = 0; y < Bh; y++){
      const o = y * Bw, q = y * nx;
      for (let x = 0; x < nx; x++){
        let s = 0;
        for (let f = 0; f < 8; f++) s += (f & 1 ? -1 : 1) * B[o + x + f * c];
        C[q + x] = s;
      }
    }
    let bi = 0, bv = -1;
    for (let y = 0; y < ny; y++){
      for (let x = 0; x < nx; x++){
        let s = 0;
        for (let r = 0; r < 8; r++) s += (r & 1 ? -1 : 1) * C[(y + r * c) * nx + x];
        s = Math.abs(s);
        if (s > bv){ bv = s; bi = y * nx + x; }
      }
    }
    const v = bv / (64 * c * c);  // mean light/dark luminance gap
    if (v > best[0]) best = [v, bi % nx, Math.floor(bi / nx), c];
  }
  return best;
}

/* Sample the board crop (x, y, w, h) scaled to 256x256 at one pixel, bilinearly, the way
   OpenCV's warpAffine does (1/32 pixel steps, black outside the frame); returns grey. */
function sampler(px, W, H, ch){
  return (sx, sy) => {
    const qx = Math.floor(sx * 32 + 0.5), qy = Math.floor(sy * 32 + 0.5);
    const x0 = qx >> 5, y0 = qy >> 5, fx = (qx & 31) / 32, fy = (qy & 31) / 32;
    const rgb = [0, 0, 0];
    for (let k = 0; k < 4; k++){
      const x = x0 + (k & 1), y = y0 + (k >> 1);
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const w = (k & 1 ? fx : 1 - fx) * (k >> 1 ? fy : 1 - fy), o = (y * W + x) * ch;
      rgb[0] += w * px[o]; rgb[1] += w * px[o + 1]; rgb[2] += w * px[o + 2];
    }
    return grey(Math.round(rgb[0]), Math.round(rgb[1]), Math.round(rgb[2]));
  };
}

/* How much colour varies along every cell's rim in the 256x256 crop of r: low when each
   rim sits inside one square. */
function ringVariance(sample, r){
  const [x, y, w, h] = r, kx = w / 256, ky = h / 256;
  let total = 0;
  const vals = new Float64Array(120);
  for (let cr = 0; cr < 8; cr++) for (let cc = 0; cc < 8; cc++){
    let n = 0;
    const at = (i, j) => sample(x + (cc * 32 + j) * kx, y + (cr * 32 + i) * ky);
    for (let j = 1; j < 31; j++) vals[n++] = at(1, j);
    for (let j = 1; j < 31; j++) vals[n++] = at(30, j);
    for (let i = 1; i < 31; i++) vals[n++] = at(i, 1);
    for (let i = 1; i < 31; i++) vals[n++] = at(i, 30);
    let m = 0;
    for (const v of vals) m += v;
    m /= 120;
    let s = 0;
    for (const v of vals) s += (v - m) * (v - m);
    total += s / 120;
  }
  return total;
}

/* Nudge each edge (sub-pixel) to minimise colour variation along every cell's rim. */
export function refine(px, W, H, ch, r){
  const sample = sampler(px, W, H, ch), o = r.slice();
  const lim = 0.05 * Math.max(r[2], r[3]);
  let best = ringVariance(sample, r);
  for (const frac of [0.02, 0.01, 0.005, 0.0025]){
    const step = Math.max(0.25, frac * Math.max(o[2], o[3]));
    let improved = true;
    while (improved){
      improved = false;
      for (const edge of 'lrtbxy'){
        for (const s of [-1, 1]){
          let [x, y, w, h] = r;
          const d = s * step;
          if (edge === 'l'){ x += d; w -= d; }
          else if (edge === 'r') w += d;
          else if (edge === 't'){ y += d; h -= d; }
          else if (edge === 'b') h += d;
          else if (edge === 'x') x += d;
          else y += d;
          const t = [x, y, w, h];
          if (x < 0 || y < 0 || x + w > W || y + h > H || Math.max(...t.map((v, i) => Math.abs(v - o[i]))) > lim) continue;
          const v = ringVariance(sample, t);
          if (v < best * 0.998){ best = v; r = t; improved = true; }
        }
      }
    }
  }
  return r;
}

/* How plainly a 256x256 board crop shows the board it was found on: the mean grey gap between
   light and dark squares, read near each cell's corners, where pieces seldom reach. It falls
   when the board has gone, or has moved by more than a sliver of a square. */
export function boardGap(img){
  let s = 0;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++){
    let m = 0;
    for (const y0 of [4, 24]) for (const x0 of [4, 24])
      for (let y = y0; y < y0 + 4; y++) for (let x = x0; x < x0 + 4; x++){
        const o = ((r * 32 + y) * 256 + c * 32 + x) * 3;
        m += grey(img[o], img[o + 1], img[o + 2]);
      }
    s += ((r + c) & 1 ? -1 : 1) * m / 64;
  }
  return Math.abs(s) / 64;
}

/* found: [{score, box, t}] from frames spread through the video. Boxes that agree with
   many others win; returns the winner (to be refined on its frame) or null. */
export function pickBoard(found){
  if (!found.length) return null;
  const close = (a, b) => [0, 1, 2, 3].every(i => Math.abs(a[i] - b[i]) < 0.04 * a[2]);
  const votes = found.map(f => found.filter(g => close(f.box, g.box)).length + f.score / 1000);
  return found[argmax(votes)];
}

/* ---------------------------------------------------------------- tiles */
/* (256x256 RGB) -> 64 tiles of 32x32 RGB, top-left cell first, one after another */
export function boardTiles(img){
  const out = new Uint8Array(64 * TILE * TILE * 3);
  let o = 0;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++)
    for (let i = 0; i < TILE; i++){
      const s = ((r * TILE + i) * 256 + c * TILE) * 3;
      out.set(img.subarray(s, s + TILE * 3), o);
      o += TILE * 3;
    }
  return out;
}

const TILE_LEN = TILE * TILE * 3;
/* Tiles that differ from ref by more than video compression flicker. A piece, a highlight,
   an arrow or a fade all count; the same square re-encoded doesn't. */
export function changedTiles(tiles, ref){
  const out = [];
  for (let k = 0; k < 64; k++){
    let big = 0, sum = 0;
    for (let i = k * TILE_LEN, e = i + TILE_LEN; i < e; i++){
      const d = Math.abs(tiles[i] - ref[i]);
      sum += d;
      if (d > 24) big++;
    }
    if (big >= 3 || sum / TILE_LEN > 1.5) out.push(k);
  }
  return out;
}

/* cells: 64x13 probabilities, top-left first. +1 if White looks to be at the bottom, -1 if
   at the top, 0 if unclear. Pawns can't pass each other on a file, so on a file with pawns of
   both colours White's are below Black's when White is at the bottom: each such file counts
   twice. Where the pieces stand on average breaks a tie. */
export function orientationVote(cells){
  let wn = 0, ws = 0, bn = 0, bs = 0, pawns = 0;
  const lowP = new Array(8).fill(-1), highP = new Array(8).fill(8), lowp = new Array(8).fill(-1), highp = new Array(8).fill(8);
  for (let i = 0; i < 64; i++){
    const k = argmax(cells, i * 13, 13), conf = cells[i * 13 + k], row = i >> 3, col = i & 7;
    if (conf <= 0.7 || k === 0) continue;
    if (k <= 6){ wn++; ws += row; } else { bn++; bs += row; }
    if (k === 1){ lowP[col] = Math.max(lowP[col], row); highP[col] = Math.min(highP[col], row); }
    if (k === 7){ lowp[col] = Math.max(lowp[col], row); highp[col] = Math.min(highp[col], row); }
  }
  for (let c = 0; c < 8; c++){
    if (lowP[c] < 0 || lowp[c] < 0) continue;
    if (highP[c] > lowp[c]) pawns++;       // every White pawn below every Black one
    else if (lowP[c] < highp[c]) pawns--;  // every White pawn above
  }
  const spread = wn >= 3 && bn >= 3 ? Math.sign(ws / wn - bs / bn) : 0;
  return Math.sign(2 * pawns + spread);
}

/* cells: 64x13, top-left first. true if they show the starting position (or one a few moves
   into a game) with White at the bottom, false if with White at the top, null if neither.
   It goes by colour only (a White piece, a Black piece or nothing on each square), which
   reads more reliably than which piece it is: 16 of one colour on the top two rows and 16
   of the other on the bottom two can't happen the wrong way up in a game. */
const START_KINDS = 'b'.repeat(16) + '.'.repeat(32) + 'w'.repeat(16);
export function startOrientation(cells){
  let up = 0, down = 0;
  for (let i = 0; i < 64; i++){
    const o = i * 13, p = { '.': cells[o], w: 0, b: 0 };
    for (let k = 1; k <= 6; k++){ p.w += cells[o + k]; p.b += cells[o + 6 + k]; }
    up += p[START_KINDS[i]];
    down += p[START_KINDS[63 - i]];  // Black at the bottom: the board turned round
  }
  return up >= 48 && up - down >= 16 ? true : down >= 48 && down - up >= 16 ? false : null;
}

const cellSquare = (i, whiteBottom) => {
  const row = i >> 3, col = i & 7;
  return whiteBottom ? (7 - row) * 8 + col : row * 8 + (7 - col);
};
/* How many squares read (with some confidence) the same as board b, an array indexed by
   square a1..h8, and how many don't, reading the cells the given way up. By colour only (a
   White piece, a Black piece or nothing), which reads more reliably than which piece it is. */
function agreement(cells, whiteBottom, b){
  let same = 0, differ = 0;
  for (let i = 0; i < 64; i++){
    const o = i * 13;
    let w = 0, bl = 0;
    for (let k = 1; k <= 6; k++){ w += cells[o + k]; bl += cells[o + 6 + k]; }
    const seen = cells[o] > 0.5 ? '.' : w > 0.5 ? 'w' : bl > 0.5 ? 'b' : null;
    if (!seen) continue;
    const p = b[cellSquare(i, whiteBottom)];
    if (seen === (p === '.' ? '.' : p === p.toUpperCase() ? 'w' : 'b')) same++; else differ++;
  }
  return { same, differ };
}
export function cellsToSquares(cells, whiteBottom){
  const out = new Float32Array(64 * 13);
  for (let i = 0; i < 64; i++) out.set(cells.subarray(i * 13, i * 13 + 13), cellSquare(i, whiteBottom) * 13);
  return out;
}

/* ---------------------------------------------------------------- readings -> positions */
/* The follower does its arithmetic in float32 as numpy did: a square that has just changed
   reads almost exactly 0.55 after smoothing, right at a threshold, so rounding decides
   which frame a move is dated to. */
const f32 = Math.fround;
const F045 = f32(0.45), F055 = f32(0.55), F085 = f32(0.85), F015 = f32(0.15), F1E4 = f32(1e-4);
/* numpy's float32 sum: eight running sums, combined pairwise, then the leftovers */
function sum32(a, start, n){
  if (n < 8){ let s = 0; for (let i = 0; i < n; i++) s = f32(s + a[start + i]); return s; }
  const r = Array.from(a.subarray(start, start + 8));
  const m = n - (n % 8);
  for (let i = 8; i < m; i += 8) for (let j = 0; j < 8; j++) r[j] = f32(r[j] + a[start + i + j]);
  let s = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
  for (let i = m; i < n; i++) s = f32(s + a[start + i]);
  return s;
}

function chessSense(sm){
  const cp = Float32Array.from(sm);
  for (const K of [6, 12]){
    let keep = 0;
    for (let s = 1; s < 64; s++) if (cp[s * 13 + K] > cp[keep * 13 + K]) keep = s;
    const col = cp[keep * 13 + K];
    for (let s = 0; s < 64; s++) cp[s * 13 + K] = 0;
    cp[keep * 13 + K] = col;
  }
  for (let s = 0; s < 64; s++) if (s < 8 || s >= 56){ cp[s * 13 + 1] = 0; cp[s * 13 + 7] = 0; }
  for (let s = 0; s < 64; s++){
    const t = Math.max(sum32(cp, s * 13, 13), f32(1e-9));
    for (let k = 0; k < 13; k++) cp[s * 13 + k] = cp[s * 13 + k] / t;
  }
  return cp;
}

function inferCastling(b){
  let s = '';
  if (b[4] === 'K' && b[7] === 'R') s += 'K';
  if (b[4] === 'K' && b[0] === 'R') s += 'Q';
  if (b[60] === 'k' && b[63] === 'r') s += 'k';
  if (b[60] === 'k' && b[56] === 'r') s += 'q';
  return s || '-';
}

export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

export class Follower {
  constructor(startFen = START_FEN){
    this.board = Position.fromFen(startFen);
    this.hist = [this.board.copy()];
    this.sm = null;
    this.pend = new Array(64).fill(null);
    this.waited = 0;
    this.events = [];  // {t, fen, san, from, to}
  }

  _commit(D, lastColour, raw, t){
    const key = D.join(''), prev = this.board;
    const logp = raw.map(v => Math.log(f32(v + F1E4))), row = new Float32Array(64);
    const score = b => { for (let q = 0; q < 64; q++) row[q] = logp[q * 13 + CLASS_INDEX[b[q]]]; return sum32(row, 0, 64); };
    // every position one move away (from either side to move) and two moves away, scored
    // together; the earliest best wins
    const bases = [prev];
    if (!prev.isCheck()){
      const sw = prev.copy();
      sw.turn = sw.turn === 'w' ? 'b' : 'w'; sw.ep = -1;
      if (sw.isValid()) bases.push(sw);
    }
    let best = null;
    const consider = (v, steps) => { if (!best || v > best.v) best = { v, steps }; };
    bases.forEach((base, i) => {
      for (const mv of base.legalMoves()) consider(score(base.push(mv, true).b) - (i ? 0.5 : 0), [mv]);
    });
    for (const mv of prev.legalMoves()){
      const mid = prev.push(mv);
      for (const mv2 of mid.legalMoves()) consider(score(mid.push(mv2, true).b) - 1, [mv, mv2]);
    }
    const llD = score(D);
    let histI = -1;
    for (let i = this.hist.length - 2; i >= 0; i--) if (this.hist[i].b.join('') === key){ histI = i; break; }
    if (histI >= 0 && (!best || best.v < llD)){
      this.back(t, histI);
      return;
    }
    if (best && best.v >= llD - 4){
      // replay to get each intermediate position
      let cur = prev.copy();
      if (best.steps.length === 1 && !cur.legalMoves().some(m => sameMove(m, best.steps[0]))){
        cur = prev.copy();
        cur.turn = cur.turn === 'w' ? 'b' : 'w'; cur.ep = -1;
      }
      for (const mv of best.steps){
        const san = cur.san(mv);
        cur = cur.push(mv);
        this.hist.push(cur.copy());
        this.events.push({ t, fen: cur.fen(), san, from: squareName(mv.from), to: squareName(mv.to) });
      }
      this.board = cur.copy();
      return;
    }
    const turn = lastColour ? (lastColour === 'w' ? 'b' : 'w') : prev.turn;
    const nb = Position.fromFen(`${new Position(D).placement()} ${turn} ${inferCastling(D)} - 0 ${prev.full}`);
    this.board = nb;
    this.hist.push(nb.copy());
    this.events.push({ t, fen: nb.fen(), san: null, from: null, to: null, jump: true });
  }

  /* back to the position hist[i], forgetting the ones after it */
  back(t, i){
    this.hist = this.hist.slice(0, i + 1);
    this.board = this.hist[i].copy();
    this.events.push({ t, fen: this.board.fen(), san: null, from: null, to: null });
  }

  /* p: 64x13 probabilities indexed by square a1..h8 */
  feed(t, p){
    this.sm = this.sm ? this.sm.map((v, i) => f32(F045 * v) + f32(F055 * p[i])) : Float32Array.from(p);
    const raw = this.sm, cp = chessSense(raw), D = this.board.b.slice();
    const ready = [];
    let settling = false;
    for (let sq = 0; sq < 64; sq++){
      const k = argmax(cp, sq * 13, 13), pr = cp[sq * 13 + k], cls = CLASSES[k];
      const occupied = cls !== '.' && raw[sq * 13] < F015;
      if (pr < F055 && !occupied){ this.pend[sq] = null; continue; }
      if (cls === D[sq]){ this.pend[sq] = null; continue; }
      let pd = this.pend[sq];
      if (pd && pd.cls === cls) pd.n++;
      else this.pend[sq] = pd = { cls, n: 1, t0: t };
      if (pd.n >= (pr > F085 ? 2 : 4)) ready.push([sq, cls, pd.t0]);
      else settling = true;
    }
    if (ready.length && settling && this.waited < 3){ this.waited++; return; }
    this.waited = 0;
    if (!ready.length) return;
    let lastColour = null;
    for (const [sq, cls] of ready){
      D[sq] = cls; this.pend[sq] = null;
      if (cls !== '.') lastColour = cls === cls.toUpperCase() ? 'w' : 'b';
    }
    // date the change to when it first appeared, not when it was confirmed
    this._commit(D, lastColour, raw, Math.min(...ready.map(r => r[2])));
  }
}

/* ---------------------------------------------------------------- analysis marks
   Arrows, circles and coloured squares drawn on the board, found with plain image
   processing. Lichess and chess.com draw every arrow from the centre of one square to the
   centre of another (chess.com draws knight arrows as an L), so instead of looking for
   arbitrary shapes we test each possible arrow against a mask of "overlay-coloured" pixels:
   the line between the two centres must be covered, the strip beside it must not be (so a
   tinted last-move square isn't mistaken for a short arrow), and one end must be wider than
   the other (the head). */
const S = 32;          // cell size in the 256x256 crop
const EXCESS = 22;     // colour a pixel needs beyond "its square's colour mixed with grey"
const MAX_ARROWS = 12; // more than this in one frame means we are reading noise, not arrows

function chroma3(r, g, b){ const m = (r + g + b) / 3; return [r - m, g - m, b - m]; }

/* How far a colour is from anything a mix of its square's colour and a grey (a piece, a
   shadow, an anti-aliased edge) could produce. Arrows and circles are drawn in strong
   translucent colours, so they stand out; pieces and the board don't. */
function colourExcess(p, base){
  const cp = chroma3(p[0], p[1], p[2]), cb = chroma3(base[0], base[1], base[2]);
  const bb = cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2];
  let s = (cp[0] * cb[0] + cp[1] * cb[1] + cp[2] * cb[2]) / Math.max(bb, 1);
  s = Math.min(1, Math.max(0, s));
  const d0 = cp[0] - s * cb[0], d1 = cp[1] - s * cb[1], d2 = cp[2] - s * cb[2];
  return Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

/* colour of each square from its rim (pieces don't reach it, an arrow only crosses it), and
   the usual light/dark colours across the board */
function cellBases(img){
  const base = [], parity = [];
  const rimOf = (r, c) => {
    const px = [];
    const at = (i, j) => { const o = ((r * S + i) * 256 + c * S + j) * 3; px.push([img[o], img[o + 1], img[o + 2]]); };
    for (let j = 1; j < S - 1; j++) at(1, j);
    for (let j = 1; j < S - 1; j++) at(S - 2, j);
    for (let i = 1; i < S - 1; i++) at(i, 1);
    for (let i = 1; i < S - 1; i++) at(i, S - 2);
    return px;
  };
  const rims = [];
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++){
    const rim = rimOf(r, c);
    rims.push(rim);
    base.push([0, 1, 2].map(k => median(rim.map(p => p[k]))));
    parity.push((r + c) % 2);
  }
  const usual = [0, 1].map(k => [0, 1, 2].map(ch => median(base.filter((_, i) => parity[i] === k).map(b => b[ch]))));
  // a rim that doesn't agree with itself (a circle, several arrows) can't be trusted
  for (let i = 0; i < 64; i++){
    const b = base[i];
    const agree = rims[i].filter(p => f32(Math.hypot(p[0] - b[0], p[1] - b[1], p[2] - b[2])) < 30).length / rims[i].length;
    if (agree < 0.6) base[i] = usual[parity[i]];
  }
  return { base, usual, parity };
}

function overlayMask(img){
  const { base, usual, parity } = cellBases(img);
  const raw = new Uint8Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++){
    const o = (y * 256 + x) * 3, b = base[(y >> 5) * 8 + (x >> 5)];
    raw[y * 256 + x] = colourExcess([img[o], img[o + 1], img[o + 2]], b) > EXCESS ? 1 : 0;
  }
  // drop lone pixels (compression fringes around pieces); arrows are several pixels wide.
  // OpenCV's 2x2 opening: erode then dilate, both over the pixel and the ones up and left
  const er = new Uint8Array(256 * 256), mask = new Uint8Array(256 * 256);
  const near = (src, y, x, op) => {
    let v = src[y * 256 + x];
    for (const [dy, dx] of [[-1, 0], [0, -1], [-1, -1]]){
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || xx < 0) continue;
      v = op(v, src[yy * 256 + xx]);
    }
    return v;
  };
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) er[y * 256 + x] = near(raw, y, x, Math.min);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) mask[y * 256 + x] = near(er, y, x, Math.max);
  return { mask, base, usual, parity };
}

function dilate5(mask){
  const tmp = new Uint8Array(256 * 256), out = new Uint8Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++){
    let v = 0;
    for (let d = -2; d <= 2 && !v; d++){ const xx = x + d; if (xx >= 0 && xx < 256) v = mask[y * 256 + xx]; }
    tmp[y * 256 + x] = v;
  }
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++){
    let v = 0;
    for (let d = -2; d <= 2 && !v; d++){ const yy = y + d; if (yy >= 0 && yy < 256) v = tmp[yy * 256 + x]; }
    out[y * 256 + x] = v;
  }
  return out;
}

/* Name the overlay's colour. The pixel is roughly 0.6 overlay + 0.4 square, so take a share
   of the square's tint back out before reading the hue. */
function hueName(rgb, base){
  const cp = chroma3(...rgb), cb = chroma3(...base);
  const [r, g, b] = [0, 1, 2].map(k => cp[k] - 0.4 * cb[k]);
  let h = Math.atan2(Math.sqrt(3) * (g - b), 2 * r - g - b) * 180 / Math.PI;
  h = ((h % 360) + 360) % 360;
  if (h < 15 || h >= 330) return 'red';
  if (h < 65) return 'orange';
  if (h < 170) return 'green';
  return 'blue';  // also what an unreadable (NaN) colour comes out as, as in numpy
}

const centre = rc => [f32((rc[1] + 0.5) * S), f32((rc[0] + 0.5) * S)];  // x, y

/* Points at the given arc lengths along a polyline (extrapolating past either end), plus
   the left normal there. */
function polylineSamples(pts, offsets){
  const seg = [], lens = [], cum = [0];
  for (let i = 0; i + 1 < pts.length; i++){
    const d = [f32(pts[i + 1][0] - pts[i][0]), f32(pts[i + 1][1] - pts[i][1])];
    seg.push(d);
    lens.push(f32(Math.sqrt(f32(d[0] * d[0] + d[1] * d[1]))));
    cum.push(f32(cum[cum.length - 1] + lens[lens.length - 1]));
  }
  return Array.from(offsets, off => {
    let i = 0;
    while (i + 1 < cum.length && cum[i + 1] <= off) i++;
    i = Math.min(i, seg.length - 1);
    const d = [f32(seg[i][0] / lens[i]), f32(seg[i][1] / lens[i])];
    return { p: [pts[i][0] + d[0] * (off - cum[i]), pts[i][1] + d[1] * (off - cum[i])], n: [-d[1], d[0]] };
  });
}

const K_LINE = 24, K_SIDE = 12, K_SHAFT = 10, K_HEAD = 10, K_PROF = K_SHAFT + K_HEAD;
const W_STEPS = linspace(-0.4 * S, 0.4 * S, 27);  // across the line; the middle one is on it
const NW = W_STEPS.length;
const ix = v => Math.min(255, Math.max(0, rint(v)));

function buildCandidates(){
  const cand = [];  // {a, b, corner, pts}
  for (let ar = 0; ar < 8; ar++) for (let ac = 0; ac < 8; ac++) for (let br = 0; br < 8; br++) for (let bc = 0; bc < 8; bc++){
    const a = [ar, ac], b = [br, bc];
    if (ar !== br || ac !== bc) cand.push({ a, b, corner: null, pts: [centre(a), centre(b)] });
    const dr = Math.abs(br - ar), dc = Math.abs(bc - ac);
    if ((dr === 1 && dc === 2) || (dr === 2 && dc === 1)){  // chess.com: knight arrows as an L, long leg first
      const corner = dr === 2 ? [br, ac] : [ar, bc];
      cand.push({ a, b, corner, pts: [centre(a), centre(corner), centre(b)] });
    }
  }
  const N = cand.length;
  const line = new Uint8Array(N * K_LINE * 2), side = new Uint8Array(N * K_SIDE * 4), prof = new Uint8Array(N * K_PROF * NW * 2);
  const frac = new Float32Array(N * K_PROF), back = new Uint8Array(N * 2), over = new Uint8Array(N * 6);
  cand.forEach(({ pts }, j) => {
    let L = 0;
    for (let i = 0; i + 1 < pts.length; i++)
      L += f32(Math.sqrt(f32((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2)));
    polylineSamples(pts, linspace(0.2 * S, L - 0.3 * S, K_LINE)).forEach(({ p }, k) => {
      line[(j * K_LINE + k) * 2] = ix(p[0]); line[(j * K_LINE + k) * 2 + 1] = ix(p[1]);
    });
    // the strip beside the shaft should be clear (rules out tinted whole squares)
    polylineSamples(pts, linspace(0.2 * S, Math.max(0.2 * S, L - 0.6 * S), K_SIDE)).forEach(({ p, n }, k) => {
      for (const [h, sgn] of [[0, 1], [1, -1]]){
        const o = (j * K_SIDE * 2 + h * K_SIDE + k) * 2;
        side[o] = ix(p[0] + sgn * 0.42 * S * n[0]); side[o + 1] = ix(p[1] + sgn * 0.42 * S * n[1]);
      }
    });
    // cross-sections along the shaft and, closely spaced whatever the length, the head
    const at = new Float64Array(K_PROF);
    at.set(linspace(0.15 * S, L - 0.6 * S, K_SHAFT));
    linspace(0.55, 0.08, K_HEAD).forEach((v, k) => { at[K_SHAFT + k] = L - v * S; });
    polylineSamples(pts, at).forEach(({ p, n }, k) => {
      frac[j * K_PROF + k] = at[k] / L;
      for (let w = 0; w < NW; w++){
        const o = ((j * K_PROF + k) * NW + w) * 2;
        prof[o] = ix(p[0] + W_STEPS[w] * n[0]); prof[o + 1] = ix(p[1] + W_STEPS[w] * n[1]);
      }
    });
    // an arrow begins at a centre, so just behind the start should be clear; and it stops
    // at one, so the line shouldn't carry on for most of another square
    const [bk] = polylineSamples(pts, [-0.45 * S]);
    back[j * 2] = ix(bk.p[0]); back[j * 2 + 1] = ix(bk.p[1]);
    polylineSamples(pts, [0.45, 0.7, 0.95].map(v => L + v * S)).forEach(({ p }, k) => {
      over[j * 6 + k * 2] = ix(p[0]); over[j * 6 + k * 2 + 1] = ix(p[1]);
    });
  });
  return { cand, line, side, prof, frac, back, over };
}

let CAND = null;
const candidates = () => CAND || (CAND = buildCandidates());
const at = (mask, pts, o) => mask[pts[o + 1] * 256 + pts[o]];
const meanAt = (mask, pts, start, n) => { let s = 0; for (let k = 0; k < n; k++) s += at(mask, pts, start + 2 * k); return s / n; };

function distToPolyline(p, pts){
  let best = Infinity;
  for (let i = 0; i + 1 < pts.length; i++){
    const a = pts[i], b = pts[i + 1], ab = [b[0] - a[0], b[1] - a[1]];
    const t = Math.min(1, Math.max(0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1])));
    best = Math.min(best, Math.hypot(p[0] - (a[0] + t * ab[0]), p[1] - (a[1] + t * ab[1])));
  }
  return best;
}

/* cell m lies on the straight segment a-b, strictly inside */
function between(a, m, b){
  const [ar, ac] = a, [mr, mc] = m, [br, bc] = b;
  return (mr - ar) * (bc - ac) === (mc - ac) * (br - ar) && Math.min(ar, br) <= mr && mr <= Math.max(ar, br) &&
    Math.min(ac, bc) <= mc && mc <= Math.max(ac, bc) && !(mr === ar && mc === ac) && !(mr === br && mc === bc);
}

function findArrows(mask, wide){
  const { cand, line, side, prof, frac, back, over } = candidates();
  const N = cand.length, mid = (NW - 1) / 2;
  const T = [];  // per candidate that passes every shape test: {j, past, sharp}
  for (let j = 0; j < N; j++){
    if (meanAt(wide, line, j * K_LINE * 2, K_LINE) < 0.9) continue;
    const width = new Float64Array(K_PROF);
    for (let k = 0; k < K_PROF; k++){
      let l = 0, r = 0;
      const o = (j * K_PROF + k) * NW * 2;
      for (let w = 0; w < mid; w++) l += at(mask, prof, o + 2 * w);
      for (let w = mid + 1; w < NW; w++) r += at(mask, prof, o + 2 * w);
      // width counted only where the mask spreads to both sides of the line: an arrowhead
      // does, the corner of an L-shaped arrow or a crossing arrow doesn't
      width[k] = 2 * Math.min(l, r) + at(mask, prof, o + 2 * mid);
    }
    let head = 0;
    for (let k = K_SHAFT; k < K_PROF; k++) head = Math.max(head, width[k]);
    const shaft = median(width.subarray(0, K_SHAFT));
    // the head end is the heavier one; this is what tells a short arrow from its reverse
    let ew = 0, e = 0;
    for (let k = 0; k < K_PROF; k++){ const x = Math.max(width[k] - shaft, 0); ew += x * frac[j * K_PROF + k]; e += x; }
    const ok = meanAt(mask, side, j * K_SIDE * 4, K_SIDE * 2) <= 0.2 &&
      head >= 0.25 * S && head >= 1.5 * Math.max(shaft, 2) && ew > 0.5 * e && !at(mask, back, j * 2);
    if (!ok) continue;
    T.push({ j, past: meanAt(wide, over, j * 6, 3) === 1, sharp: meanAt(mask, line, j * K_LINE * 2, K_LINE) });
  }
  const id = rc => rc[0] * 8 + rc[1], keyOf = (a, b) => id(a) * 64 + id(b);
  let passed = new Map();
  for (const t of T){
    const k = keyOf(cand[t.j].a, cand[t.j].b);
    if (!passed.has(k) || t.sharp > passed.get(k).sharp) passed.set(k, t);
  }
  // the line may carry on past the tip only where another arrow starts there (a chain);
  // otherwise this is the front part of a longer arrow
  const starts = new Set([...passed.keys()].map(k => Math.floor(k / 64)));
  passed = new Map([...passed].filter(([k, t]) => !t.past || starts.has(k % 64)));
  // a straight arrow made of two others end to end is that chain read as one
  for (const [k, t] of [...passed]){
    const { a, b, corner } = cand[t.j];
    if (corner !== null) continue;
    for (const m of starts){
      const mc = [m >> 3, m & 7];
      if (between(a, mc, b) && passed.has(id(a) * 64 + m) && passed.has(m * 64 + id(b))){ passed.delete(k); break; }
    }
  }
  // the legs of an L-shaped arrow pass as arrows of their own (nothing is behind the corner
  // along the last leg, the corner reads as a small head): they are part of the L. An L
  // along a leg of a better one is made of that L and whatever carries on from it
  const used = new Set();
  const Ls = [...passed.values()].filter(t => cand[t.j].corner !== null).sort((x, y) => y.sharp - x.sharp);
  for (const t of Ls){
    const { a, b, corner: m } = cand[t.j];
    const legs = [[id(a), id(m)], [id(m), id(b)]].map(([x, y]) => Math.min(x, y) * 64 + Math.max(x, y));
    if (legs.some(l => used.has(l))){ passed.delete(keyOf(a, b)); continue; }
    legs.forEach(l => used.add(l));
    for (const [x, y] of [[a, m], [m, a], [m, b], [b, m]]) passed.delete(keyOf(x, y));
  }
  // a thick arrow also passes for its near neighbours (a1-b7 for a2-b7, the legs of an L):
  // take the best-aligned, longest first and drop whatever mostly runs along it
  const ptp = j => {
    let x0 = 255, x1 = 0, y0 = 255, y1 = 0;
    for (let k = 0; k < K_LINE; k++){
      const x = line[(j * K_LINE + k) * 2], y = line[(j * K_LINE + k) * 2 + 1];
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    return x1 - x0 + y1 - y0;
  };
  const r2 = v => rint(v * 100) / 100;
  const order = [...passed.values()].map(t => ({ t, s: r2(t.sharp), n: cand[t.j].pts.length, p: ptp(t.j) }))
    .sort((x, y) => (y.s - x.s) || (y.n - x.n) || (y.p - x.p)).map(o => o.t);
  const taken = [];
  for (const { j } of order){
    const ok = taken.every(k => {
      let near = 0;
      for (let q = 0; q < K_LINE; q++)
        if (distToPolyline([line[(j * K_LINE + q) * 2], line[(j * K_LINE + q) * 2 + 1]], cand[k].pts) < 0.25 * S) near++;
      return near / K_LINE < 0.5;
    });
    if (ok) taken.push(j);
  }
  return taken.map(j => [cand[j].a, cand[j].b, j]);
}

const CIRCLE = Object.fromEntries([0.40, 0.44, 0.47, 0.25].map(rad =>
  [rad, Array.from(linspace(0, 2 * Math.PI, 32, false), a => [Math.cos(a) * rad * S, Math.sin(a) * rad * S])]));

/* img: 256x256 RGB board crop. Returns marks in cell coordinates:
   ['arrow', fromRC, toRC, colour] / ['circle', rc, colour] / ['square', rc, colour] */
export function detectMarks(img){
  const { mask, base, usual, parity } = overlayMask(img);
  let on = 0;
  for (const v of mask) on += v;
  if (on / mask.length > 0.25) return [];  // a scene change or an overlay over the board, not analysis marks
  const wide = dilate5(mask);
  const arrows = findArrows(mask, wide);
  if (arrows.length > MAX_ARROWS) return [];
  const { line } = candidates();
  const medianColour = pts => [0, 1, 2].map(k => median(pts.map(([x, y]) => img[(y * 256 + x) * 3 + k])));
  const out = [];
  for (const [a, b, j] of arrows){
    const px = [];
    for (let k = 0; k < K_LINE; k++){
      const x = line[(j * K_LINE + k) * 2], y = line[(j * K_LINE + k) * 2 + 1];
      if (mask[y * 256 + x]) px.push([x, y]);
    }
    out.push(['arrow', a, b, hueName(medianColour(px), usual[parity[a[0] * 8 + a[1]]])]);
  }
  // circles (lichess): a ring just inside the square, with the inside left alone
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++){
    const ctr = centre([r, c]);
    const pts = Object.fromEntries(Object.entries(CIRCLE).map(([k, v]) =>
      [k, v.map(([dx, dy]) => [ix(ctr[0] + dx), ix(ctr[1] + dy)])]));
    const cover = (m, k) => pts[k].filter(([x, y]) => m[y * 256 + x]).length / pts[k].length;
    let bestR = '0.4';
    for (const k of ['0.44', '0.47']) if (cover(wide, k) > cover(wide, bestR)) bestR = k;
    if (cover(wide, bestR) >= 0.8 && cover(mask, '0.25') < 0.5){
      const ring = pts[bestR].filter(([x, y]) => mask[y * 256 + x]);
      out.push(['circle', [r, c], hueName(medianColour(ring), base[r * 8 + c])]);
    }
  }
  // whole squares painted red or blue (chess.com right-click); yellow/green tints are
  // last-move highlights, which the 3D board shows on its own
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++){
    const u = usual[parity[r * 8 + c]], b = base[r * 8 + c];
    if (colourExcess(b, u) > 40){
      const name = hueName(b, u);
      if (name === 'red' || name === 'blue') out.push(['square', [r, c], name]);
    }
  }
  return out;
}

export function marksToSquares(found, whiteBottom){
  const name = rc => squareName(cellSquare(rc[0] * 8 + rc[1], whiteBottom));
  return found.map(m => m[0] === 'arrow' ? ['arrow', name(m[1]), name(m[2]), m[3]] : [m[0], name(m[1]), m[2]]);
}

/* Smooths per-frame detections: a mark shows after two frames in a row and goes after two
   frames missing, so a passing cursor or a half-drawn arrow doesn't flicker. */
export class MarkTracker {
  constructor(fps){
    this.dt = 1 / fps;
    this.on = new Map();  // key -> colour
    this.streak = new Map();
    this.miss = new Map();
    this.events = [];     // {t, arrows: [...], circles: [...], squares: [...]}
  }

  feed(t, marks){
    const now = new Map(marks.map(m => [JSON.stringify(m.slice(0, -1)), m[m.length - 1]]));
    this.streak = new Map([...now.keys()].map(k => [k, (this.streak.get(k) || 0) + 1]));
    let changed = false;
    for (const [k, colour] of now){
      this.miss.delete(k);
      if (this.on.has(k)) this.on.set(k, colour);
      else if (this.streak.get(k) >= 2){ this.on.set(k, colour); changed = true; }
    }
    for (const k of [...this.on.keys()]){
      if (now.has(k)) continue;
      this.miss.set(k, (this.miss.get(k) || 0) + 1);
      if (this.miss.get(k) >= 2){ this.on.delete(k); this.miss.delete(k); changed = true; }
    }
    if (!changed) return;
    const ev = { t: Math.max(0, t - this.dt), arrows: [], circles: [], squares: [] };
    const cmp = (x, y) => { for (let i = 0; i < Math.max(x.length, y.length); i++){ if (x[i] !== y[i]) return x[i] === undefined ? -1 : y[i] === undefined ? 1 : x[i] < y[i] ? -1 : 1; } return 0; };
    for (const [k, colour] of [...this.on].map(([k, c]) => [JSON.parse(k), c]).sort((x, y) => cmp(x[0], y[0]))){
      if (k[0] === 'arrow') ev.arrows.push({ from: k[1], to: k[2], color: colour });
      else ev[k[0] + 's'].push({ sq: k[1], color: colour });
    }
    this.events.push(ev);
  }
}

/* ---------------------------------------------------------------- one frame at a time */
/* Reads a stream of 256x256 RGB board crops. predict(tiles, n) -> Promise of n x 13
   probabilities. Only squares that changed since they were last read go through the CNN,
   and the marks are only looked for again when something on the board changed. Each is
   compared with what it was last read from, so a slow fade still gets noticed.

   Which way up the board is can change during a video: chess.com and Lichess put the player's
   colour at the bottom, so a streamer playing several games shows some from White's side and
   some from Black's. Each game starts from the starting position, and that settles it: once
   it has been seen for two readings in a row, the board is read that way up until a game
   starts the other way round. Before the first one (a video that joins a game in progress),
   the guess from the whole video is used, and changed if the game being read keeps saying
   otherwise. The board can also be flipped mid-video (chess.com and Lichess do it with one
   click, and some videos animate it): every piece moves at once and the screen shows a position
   just read, turned round. Seen for two readings in a row, that turns the board too, whatever
   the position, and if the follower already took the upside-down board for a new position, it
   goes back. fixed: the viewer chose the orientation, so it never changes. Every position
   comes with wb, which way up the board was when it was read. */
export class Reader {
  constructor(predict, { whiteBottom = true, decided = false, fixed = false, fps = FPS } = {}){
    Object.assign(this, { predict, whiteBottom, decided, fixed, votes: 0, anchored: fixed, startSeen: null, startRun: 0, startT: 0, flipRun: 0, flipT: 0 });
    this.follower = new Follower();
    this.tracker = new MarkTracker(fps);
    this.ref = this.probs = this.markRef = this.found = null;
    this.sent = this.sentMarks = 0;
  }

  async frame(t, img){
    const tiles = boardTiles(img);
    if (!this.ref){
      this.ref = tiles.slice();
      this.probs = await this.predict(tiles, 64);
    } else {
      const todo = changedTiles(tiles, this.ref);
      if (todo.length){
        const part = new Uint8Array(todo.length * TILE_LEN);
        todo.forEach((k, i) => {
          const src = tiles.subarray(k * TILE_LEN, (k + 1) * TILE_LEN);
          part.set(src, i * TILE_LEN);
          this.ref.set(src, k * TILE_LEN);
        });
        const p = await this.predict(part, todo.length);
        todo.forEach((k, i) => this.probs.set(p.subarray(i * 13, i * 13 + 13), k * 13));
      }
    }
    if (!this.fixed) this.orient(t);
    if (this.decided){  // until then, wait to know which way up the board is
      this.follower.feed(t, cellsToSquares(this.probs, this.whiteBottom));
      if (!this.markRef || changedTiles(tiles, this.markRef).length){
        this.markRef = tiles;
        this.found = detectMarks(img);
      }
      this.tracker.feed(t, marksToSquares(this.found, this.whiteBottom));
    }
    const wb = this.whiteBottom;
    const events = this.follower.events.slice(this.sent).map(e => ({ ...e, wb })), marks = this.tracker.events.slice(this.sentMarks);
    this.sent = this.follower.events.length; this.sentMarks = this.tracker.events.length;
    return { events, marks, whiteBottom: this.whiteBottom, decided: this.decided };
  }

  orient(t){
    const start = startOrientation(this.probs);
    this.startRun = start !== null && start === this.startSeen ? this.startRun + 1 : 1;
    if (this.startRun === 1) this.startT = t;
    this.startSeen = start;
    const flip = this.flipped();
    this.flipRun = flip >= 0 ? this.flipRun + 1 : 0;
    if (this.flipRun === 1) this.flipT = t;
    if (start !== null && this.startRun >= 2){
      this.anchored = true;
      this.votes = 0;
      this.turn(start, this.startT);
      return;
    }
    if (this.flipRun >= 2){
      // anything read off the upside-down board is wrong: back to the position it showed
      const f = this.follower;
      if (flip < f.hist.length - 1){ f.hist = f.hist.slice(0, flip + 1); f.board = f.hist[flip].copy(); }
      this.votes = 0; this.flipRun = 0;
      this.turn(!this.whiteBottom, this.flipT);
      return;
    }
    if (this.anchored) return;
    // no game start seen yet: go by the pieces, and overrule the whole video's guess only
    // when this game keeps disagreeing with it
    this.votes += orientationVote(this.probs);
    const need = this.decided ? 8 : 3;
    if (Math.abs(this.votes) >= need){ this.turn(this.votes > 0, t); this.votes = 0; }
  }

  /* The board turned round: the screen shows one of the last few positions taken, upside down
     (and not as it is: lots moved at once). Its index in the follower's history, or -1. */
  flipped(){
    const hist = this.follower.hist;
    for (let i = hist.length - 1; i >= Math.max(0, hist.length - 12); i--){
      const turned = agreement(this.probs, !this.whiteBottom, hist[i].b);
      if (turned.same < 48 || turned.differ > 3) continue;
      if (agreement(this.probs, this.whiteBottom, hist[i].b).differ >= 6 + 2 * turned.differ) return i;
    }
    return -1;
  }

  turn(whiteBottom, t){
    if (this.decided && whiteBottom === this.whiteBottom) return;
    // the squares now map the other way round: forget the smoothing over the old mapping
    const f = this.follower;
    if (this.decided){ f.sm = null; f.pend.fill(null); }
    this.whiteBottom = whiteBottom; this.decided = true;
    // the screen shows the position already taken, now the other way up: say it again, so the
    // view turns with the video now, not at the next move
    if (agreement(this.probs, whiteBottom, f.board.b).differ <= 3){
      const last = f.events[f.events.length - 1], fen = f.board.fen();
      // keeping the last move's squares lit if it is the position that move made
      const lit = last && last.fen === fen ? { from: last.from, to: last.to } : { from: null, to: null };
      f.events.push({ t: last ? Math.max(t, last.t) : t, fen, san: null, ...lit });
    }
  }
}

/* True if White is at the bottom, by majority over board crops spread through the whole
   video, so an intro, a thumbnail or the first second of play can't decide it. null if
   unclear. */
export async function detectOrientation(predict, crops){
  if (!crops.length) return null;
  const tiles = new Uint8Array(crops.length * 64 * TILE_LEN);
  crops.forEach((img, i) => tiles.set(boardTiles(img), i * 64 * TILE_LEN));
  const probs = await predict(tiles, crops.length * 64);
  let votes = 0;
  for (let i = 0; i < crops.length; i++) votes += orientationVote(probs.subarray(i * 64 * 13, (i + 1) * 64 * 13));
  return Math.abs(votes) >= 2 ? votes > 0 : null;
}

/* the model.json weights (float16, base64) -> list of Float32Arrays with their shapes */
export function decodeWeights(m){
  const bin = Uint8Array.from(atob(m.data), c => c.charCodeAt(0));
  const half = new Uint16Array(bin.buffer, bin.byteOffset, bin.byteLength / 2);
  const flat = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++){
    const h = half[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    flat[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  let off = 0;
  return m.shapes.map(shape => {
    const n = shape.reduce((a, b) => a * b, 1);
    const w = flat.subarray(off, off + n);
    off += n;
    return { shape, data: w };
  });
}
