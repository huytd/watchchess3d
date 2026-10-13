"""Video -> timeline of chess positions.

1. find the board: search every size and position for the strongest 8x8 checkerboard
   pattern, then fine-tune the edges so each cell's rim sits inside one square
2. read the board ~3 times a second with the CNN
3. turn noisy per-frame readings into a clean sequence of positions, preferring
   legal moves (python-chess) so a shaky square can't derail the game
4. alongside, pick out the arrows and circles streamers draw on the board (marks.py)
"""
import json, math, os, subprocess, threading
import numpy as np
import cv2
import chess

import marks
from recognizer import Recognizer, board_tiles, CLASSES

FPS = 3.0
# ffmpeg starts a decoding thread, each with its own frames, per core it can see; in a
# container that is often every core of the host, whatever the memory limit
DECODE = ['-threads', str(min(4, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4))]
_local = threading.local()


def recognizer():
    # one per thread: an OpenCV network can't run two inputs at once (SHADOWBOARD_WORKERS > 1)
    if not hasattr(_local, 'rec'):
        _local.rec = Recognizer()
    return _local.rec


# ---------------------------------------------------------------- video helpers
def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height:format=duration', '-of', 'json', path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j['streams'][0]
    return int(s['width']), int(s['height']), float(j['format'].get('duration') or 0)


def frames(path, fps, crop=None, size=None):
    """Yield (t, rgb ndarray). crop=(x,y,w,h) and size=(W,H) are applied by ffmpeg."""
    w, h, _ = probe(path)
    vf = [f'fps={fps}']
    if crop:
        x, y, cw, ch = [int(round(v)) for v in crop]
        vf.append(f'crop={cw}:{ch}:{x}:{y}')
        w, h = cw, ch
    if size:
        vf.append(f'scale={size[0]}:{size[1]}:flags=area')
        w, h = size
    # frames nothing else is predicted from (most B-frames) needn't be decoded: that halves
    # the decoding work, and the frame read for each sample moves by a frame or two at most
    p = subprocess.Popen(['ffmpeg', '-v', 'error', *DECODE, '-skip_frame', 'noref', '-i', path, '-an', '-sn',
                          '-vf', ','.join(vf), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n, k = w * h * 3, 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        p.stdout.close()
        p.wait()


def frame_at(path, t):
    out = subprocess.run(['ffmpeg', '-v', 'error', *DECODE, '-ss', str(max(0, t)), '-i', path, '-frames:v', '1',
                          '-f', 'image2', '-c:v', 'mjpeg', '-q:v', '4', '-'], capture_output=True)
    return out.stdout


def rgb_at(path, t):
    """The frame at t as an RGB array (None if there is none)."""
    img = cv2.imdecode(np.frombuffer(frame_at(path, t), np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- board detection
SEARCH_LINES = 720  # bigger frames are searched shrunk to this; refine() then works at full size


def checker_search(rgb):
    """Best (score, x, y, cell) for an 8x8 checkerboard of integer cell size."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = lum.shape
    # (H+1, W+1); integers keep it exact and fast, up to 4K where the sums outgrow int32
    ii = cv2.integral(lum, sdepth=cv2.CV_32S if H * W * 255 < 2 ** 31 else cv2.CV_64F)
    best = (0.0, 0, 0, 0)
    for c in range(max(6, int(min(H, W) * 0.035)), min(H, W) // 8 + 1):
        # sum of every c x c box, indexed by its top-left corner
        B = ii[c:, c:] - ii[:-c, c:] - ii[c:, :-c] + ii[:-c, :-c]
        ny, nx = B.shape[0] - 7 * c, B.shape[1] - 7 * c
        if ny <= 0 or nx <= 0:
            continue
        # the checkerboard's +-1 pattern is (-1)^row * (-1)^file, so sum along the files
        # first and then down the rows: 16 passes over the image instead of 64
        C = B[:, :nx].copy()
        for f in range(1, 8):
            if f % 2: C -= B[:, f * c:f * c + nx]
            else: C += B[:, f * c:f * c + nx]
        R = C[:ny].copy()
        for r in range(1, 8):
            if r % 2: R -= C[r * c:r * c + ny]
            else: R += C[r * c:r * c + ny]
        i = int(np.abs(R, out=R).argmax())
        v = R.flat[i] / (64 * c * c)  # mean light/dark luminance gap
        if v > best[0]:
            y, x = divmod(i, nx)
            best = (float(v), x, y, c)
    return best


def ring_variance(img256):
    lum = cv2.cvtColor(img256, cv2.COLOR_RGB2GRAY).astype(np.float64)
    t = lum.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3)  # 8,8,32,32
    ring = np.concatenate([t[:, :, 1, 1:31], t[:, :, 30, 1:31], t[:, :, 1:31, 1], t[:, :, 1:31, 30]], axis=2)
    return float(ring.var(axis=2).sum())


def crop256(rgb, r):
    x, y, w, h = r
    M = np.array([[256 / w, 0, -x * 256 / w], [0, 256 / h, -y * 256 / h]])
    return cv2.warpAffine(rgb, M, (256, 256), flags=cv2.INTER_AREA)


def refine(rgb, r):
    """Nudge each edge (sub-pixel) to minimise colour variation along every cell's rim."""
    H, W = rgb.shape[:2]
    o = r
    lim = 0.05 * max(r[2], r[3])
    best = ring_variance(crop256(rgb, r))
    for frac in (0.02, 0.01, 0.005, 0.0025):
        step = max(0.25, frac * max(o[2], o[3]))
        improved = True
        while improved:
            improved = False
            for edge in 'lrtbxy':
                for s in (-1, 1):
                    x, y, w, h = r
                    d = s * step
                    if edge == 'l': x, w = x + d, w - d
                    elif edge == 'r': w += d
                    elif edge == 't': y, h = y + d, h - d
                    elif edge == 'b': h += d
                    elif edge == 'x': x += d
                    else: y += d
                    t = (x, y, w, h)
                    if x < 0 or y < 0 or x + w > W or y + h > H or max(abs(t[i] - o[i]) for i in range(4)) > lim:
                        continue
                    v = ring_variance(crop256(rgb, t))
                    if v < best * 0.998:
                        best, r, improved = v, t, True
    return r


def detect_board(path, duration, samples=12):
    """Look at frames spread through the video; return the most consistent strong board box."""
    found = []
    for i in range(samples):
        t = duration * (i + 0.5) / samples
        rgb = rgb_at(path, t)
        if rgb is None:
            continue
        H, W = rgb.shape[:2]
        if min(H, W) > SEARCH_LINES:
            k = SEARCH_LINES / min(H, W)
            rgb = cv2.resize(rgb, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
        kx, ky = rgb.shape[1] / W, rgb.shape[0] / H
        score, x, y, c = checker_search(rgb)
        if score > 12 and c:
            found.append((score, (x / kx, y / ky, 8.0 * c / kx, 8.0 * c / ky), t))
    if not found:
        return None
    # vote: boxes that agree with many others win; then refine the strongest of them
    def close(a, b):
        return all(abs(a[i] - b[i]) < 0.04 * a[2] for i in range(4))
    votes = [sum(close(f[1], g[1]) for g in found) + f[0] / 1000 for f in found]
    score, box, t = found[int(np.argmax(votes))]
    # only the winner's frame is needed again: fetch it rather than hold every full-size frame
    return refine(rgb_at(path, t), box)


# ---------------------------------------------------------------- readings -> positions
def chess_sense(sm):
    cp = sm.copy()
    for K in (6, 12):
        keep = int(cp[:, K].argmax())
        col = cp[keep, K]
        cp[:, K] = 0
        cp[keep, K] = col
    cp[:8, [1, 7]] = 0
    cp[56:, [1, 7]] = 0
    return cp / np.maximum(cp.sum(axis=1, keepdims=True), 1e-9)


def to_board(b):
    """64-char list (a1..h8) -> python-chess Board placement only."""
    bd = chess.Board(None)
    for sq, ch in enumerate(b):
        if ch != '.':
            bd.set_piece_at(sq, chess.Piece.from_symbol(ch))
    return bd


# (piece type, colour) in CLASSES[1:] order
PIECES = [(pt, colour) for colour in (chess.WHITE, chess.BLACK) for pt in range(chess.PAWN, chess.KING + 1)]


def bitboards(board):
    """One bitboard per class in CLASSES[1:] order."""
    return [board.pieces_mask(pt, colour) for pt, colour in PIECES]


def class_grid(bbs):
    """(N,12) bitboards -> (N,64) class index of each square, a1..h8."""
    bits = np.unpackbits(np.asarray(bbs, '<u8').reshape(-1, 12).view(np.uint8).reshape(-1, 12, 8), axis=2,
                         bitorder='little')  # N,12,64
    return (bits * np.arange(1, 13, dtype=np.uint8)[:, None]).sum(axis=1)


def key_of(board):
    out = ['.'] * 64
    for sym, (pt, colour) in zip(CLASSES[1:], PIECES):
        for sq in chess.scan_forward(board.pieces_mask(pt, colour)):
            out[sq] = sym
    return ''.join(out)


def infer_castling(b):
    s = ''
    if b[4] == 'K' and b[7] == 'R': s += 'K'
    if b[4] == 'K' and b[0] == 'R': s += 'Q'
    if b[60] == 'k' and b[63] == 'r': s += 'k'
    if b[60] == 'k' and b[56] == 'r': s += 'q'
    return s or '-'


class Follower:
    """Offline version of the browser's mirror mode."""

    def __init__(self, start_fen=chess.STARTING_FEN):
        self.board = chess.Board(start_fen)
        self.hist = [self.board.copy()]
        self.sm = None
        self.pend = [None] * 64
        self.waited = 0
        self.events = []  # (t, fen, san, from, to)

    def _ll(self, key, raw):
        idx = [CLASSES.index(c) for c in key]
        return float(np.log(raw[np.arange(64), idx] + 1e-4).sum())

    @staticmethod
    def _children(base, bbs, line, out):
        """Append the bitboards after every legal move from base to bbs, its moves to out."""
        for mv in list(base.legal_moves):
            base.push(mv)
            bbs.append(bitboards(base))
            base.pop()
            out.append(line + [mv])

    def _commit(self, D, last_color, raw, t):
        key = ''.join(D)
        prev = self.board
        best = None
        bases = [prev]
        if not prev.is_check():
            sw = prev.copy(stack=False)
            sw.turn = not sw.turn
            sw.ep_square = None
            if sw.is_valid():
                bases.append(sw)
        # every position one move away (from either side to move) and two moves away,
        # scored together; the earliest best wins, as when they were tried one by one
        bbs, lines, penalty = [], [], []
        for i, base in enumerate(bases):
            self._children(base, bbs, [], lines)
            penalty += [0.5 if i else 0.0] * (len(lines) - len(penalty))
        for mv in list(prev.legal_moves):
            prev.push(mv)
            self._children(prev, bbs, [mv], lines)
            prev.pop()
        penalty += [1.0] * (len(lines) - len(penalty))
        if lines:
            logp = np.log(raw + 1e-4)
            v = logp[np.arange(64), class_grid(bbs)].sum(axis=1).astype(np.float64) - penalty
            i = int(v.argmax())
            best = (float(v[i]), lines[i])
        llD = self._ll(key, raw)
        hist_i = next((i for i in range(len(self.hist) - 2, -1, -1) if key_of(self.hist[i]) == key), -1)
        if hist_i >= 0 and (best is None or best[0] < llD):
            self.hist = self.hist[:hist_i + 1]
            self.board = self.hist[-1].copy()
            self.events.append({'t': t, 'fen': self.board.fen(), 'san': None, 'from': None, 'to': None})
            return
        if best and best[0] >= llD - 4:
            v, steps = best
            # replay to get each intermediate position
            cur = prev.copy(stack=False)
            if len(steps) == 1 and steps[0] not in cur.legal_moves:
                cur.turn = not cur.turn
                cur.ep_square = None
            for mv in steps:
                san = cur.san(mv)
                cur.push(mv)
                self.hist.append(cur.copy(stack=False))
                self.events.append({'t': t, 'fen': cur.fen(), 'san': san,
                                    'from': chess.square_name(mv.from_square), 'to': chess.square_name(mv.to_square)})
            self.board = cur.copy(stack=False)
            return
        turn = ('b' if last_color == 'w' else 'w') if last_color else ('w' if prev.turn else 'b')
        fen = f"{''.join(self._placement(D))} {turn} {infer_castling(D)} - 0 {prev.fullmove_number}"
        nb = chess.Board(fen)
        self.board = nb
        self.hist.append(nb.copy(stack=False))
        self.events.append({'t': t, 'fen': nb.fen(), 'san': None, 'from': None, 'to': None, 'jump': True})

    @staticmethod
    def _placement(D):
        rows = []
        for r in range(7, -1, -1):
            s, e = '', 0
            for f in range(8):
                c = D[r * 8 + f]
                if c == '.':
                    e += 1
                else:
                    if e: s += str(e); e = 0
                    s += c
            if e: s += str(e)
            rows.append(s)
        return '/'.join(rows)

    def feed(self, t, p_sq):
        """p_sq: (64,13) probabilities indexed by square a1..h8."""
        self.sm = p_sq.copy() if self.sm is None else 0.45 * self.sm + 0.55 * p_sq
        raw, cp = self.sm, chess_sense(self.sm)
        D = list(key_of(self.board))
        ready, settling = [], False
        for sq in range(64):
            k = int(cp[sq].argmax()); p = cp[sq, k]; cls = CLASSES[k]
            occupied = cls != '.' and raw[sq, 0] < 0.15
            if p < 0.55 and not occupied:
                self.pend[sq] = None; continue
            if cls == D[sq]:
                self.pend[sq] = None; continue
            pd = self.pend[sq]
            if pd and pd['cls'] == cls: pd['n'] += 1
            else: self.pend[sq] = pd = {'cls': cls, 'n': 1, 't0': t}
            if pd['n'] >= (2 if p > 0.85 else 4): ready.append((sq, cls, pd['t0']))
            else: settling = True
        if ready and settling and self.waited < 3:
            self.waited += 1; return
        self.waited = 0
        if not ready:
            return
        last_color = None
        for sq, cls, _ in ready:
            D[sq] = cls; self.pend[sq] = None
            if cls != '.': last_color = 'w' if cls.isupper() else 'b'
        # date the change to when it first appeared, not when it was confirmed
        self._commit(D, last_color, raw, min(r[2] for r in ready))


def orientation_vote(cells):
    """cells: (64,13) top-left first. +1 if White looks to be at the bottom."""
    k = cells.argmax(1); conf = cells.max(1)
    rows = np.arange(64) // 8
    w = (k >= 1) & (k <= 6) & (conf > 0.7)
    b = (k >= 7) & (conf > 0.7)
    if w.sum() < 3 or b.sum() < 3:
        return 0
    return 1 if rows[w].mean() > rows[b].mean() else -1


def detect_orientation(path, region, duration, samples=16):
    """True if White is at the bottom, by majority over frames spread through the whole video,
    so an intro, a thumbnail or the first second of play can't decide it. None if unclear."""
    tiles = []
    for i in range(samples):
        rgb = rgb_at(path, duration * (i + 0.5) / samples)
        if rgb is not None:
            tiles.append(board_tiles(crop256(rgb, region)))
    if not tiles:
        return None
    probs = recognizer().predict(np.concatenate(tiles)).reshape(len(tiles), 64, 13)
    votes = sum(orientation_vote(cells) for cells in probs)
    return votes > 0 if abs(votes) >= 2 else None


def cells_to_squares(cells, white_bottom):
    out = np.empty_like(cells)
    for i in range(64):
        row, col = divmod(i, 8)
        sq = (7 - row) * 8 + col if white_bottom else row * 8 + (7 - col)
        out[sq] = cells[i]
    return out


def cell_name(rc, white_bottom):
    """(row, col) in the board crop, top-left first -> square name."""
    row, col = rc
    return chess.square_name((7 - row) * 8 + col if white_bottom else row * 8 + (7 - col))


def marks_to_squares(found, white_bottom):
    out = []
    for m in found:
        if m[0] == 'arrow':
            out.append(('arrow', cell_name(m[1], white_bottom), cell_name(m[2], white_bottom), m[3]))
        else:
            out.append((m[0], cell_name(m[1], white_bottom), m[2]))
    return out


def changed_tiles(tiles, ref):
    """(64,) bool: tiles that differ from ref by more than video compression flicker. A piece,
    a highlight, an arrow or a fade all count; the same square re-encoded doesn't."""
    d = cv2.absdiff(tiles.reshape(64, -1), ref.reshape(64, -1))
    return ((d > 24).sum(axis=1) >= 3) | (d.mean(axis=1) > 1.5)


def process(path, on_progress=None, region=None, orientation='auto', fps=FPS):
    """Reads the video and reports progress, new events and new marks through on_progress."""
    W, H, duration = probe(path)
    if region is None:
        region = detect_board(path, duration)
        if region is None:
            raise RuntimeError('no_board')
    if on_progress:
        on_progress({'region': region, 'width': W, 'height': H, 'duration': duration, 'progress': 0})
    rec = recognizer()
    follower = Follower()
    tracker = marks.MarkTracker(fps)
    votes = 0
    white_bottom = orientation != 'b'
    decided = orientation != 'auto'
    if not decided:
        found = detect_orientation(path, region, duration)
        if found is not None:
            white_bottom, decided = found, True
    # Most of the board stays put from one reading to the next, so only squares that changed
    # since they were last read go through the CNN, and the marks are only looked for again
    # when something on the board changed. Each is compared with what it was last read from,
    # so a slow fade still gets noticed.
    ref = probs = None        # tiles each square was last read from, and its reading
    mark_ref = found_marks = None
    sent = sent_marks = 0
    for t, img in frames(path, fps, crop=region, size=(256, 256)):
        tiles = board_tiles(img)
        if ref is None:
            ref, probs = tiles.copy(), rec.predict(tiles)  # ref is updated in place
        else:
            todo = np.nonzero(changed_tiles(tiles, ref))[0]
            if len(todo):
                ref[todo] = tiles[todo]
                probs[todo] = rec.predict(tiles[todo])
        if not decided:
            votes += orientation_vote(probs)
            if abs(votes) >= 3:
                white_bottom, decided = votes > 0, True
        if decided:  # until then, wait to know which way up the board is
            follower.feed(t, cells_to_squares(probs, white_bottom))
            if mark_ref is None or changed_tiles(tiles, mark_ref).any():
                mark_ref, found_marks = tiles, marks.detect(img)
            tracker.feed(t, marks_to_squares(found_marks, white_bottom))
        if on_progress:
            new, new_marks = follower.events[sent:], tracker.events[sent_marks:]
            sent, sent_marks = len(follower.events), len(tracker.events)
            on_progress({'events': new, 'marks': new_marks, 'white_bottom': white_bottom,
                         'progress': min(1.0, (t + 1 / fps) / max(duration, 1e-6))})
    if on_progress:
        on_progress({'progress': 1.0, 'done': True})
