"""Video -> timeline of chess positions.

1. find the board: search every size and position for the strongest 8x8 checkerboard
   pattern, then fine-tune the edges so each cell's rim sits inside one square
2. read the board ~3 times a second with the CNN
3. turn noisy per-frame readings into a clean sequence of positions, preferring
   legal moves (python-chess) so a shaky square can't derail the game
4. alongside, pick out the arrows and circles streamers draw on the board (marks.py)
"""
import json, math, subprocess
import numpy as np
import cv2
import chess

import marks
from recognizer import Recognizer, board_tiles, CLASSES

FPS = 3.0
_rec = None


def recognizer():
    global _rec
    if _rec is None:
        _rec = Recognizer()
    return _rec


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
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-vf', ','.join(vf), '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
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
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(max(0, t)), '-i', path, '-frames:v', '1',
                          '-f', 'image2', '-c:v', 'mjpeg', '-q:v', '4', '-'], capture_output=True)
    return out.stdout


# ---------------------------------------------------------------- board detection
def checker_search(rgb):
    """Best (score, x, y, cell) for an 8x8 checkerboard of integer cell size."""
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    H, W = lum.shape
    ii = cv2.integral(lum)  # (H+1, W+1)
    best = (0.0, 0, 0, 0)
    for c in range(max(6, int(min(H, W) * 0.035)), min(H, W) // 8 + 1):
        # sum of every c x c box, indexed by its top-left corner
        B = ii[c:, c:] - ii[:-c, c:] - ii[c:, :-c] + ii[:-c, :-c]
        ny, nx = B.shape[0] - 7 * c, B.shape[1] - 7 * c
        if ny <= 0 or nx <= 0:
            continue
        R = np.zeros((ny, nx))
        for r in range(8):
            for f in range(8):
                R += (1 if (r + f) % 2 == 0 else -1) * B[r * c:r * c + ny, f * c:f * c + nx]
        R = np.abs(R) / (64 * c * c)  # mean light/dark luminance gap
        i = int(R.argmax())
        y, x = divmod(i, nx)
        if R.flat[i] > best[0]:
            best = (float(R.flat[i]), x, y, c)
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
        jpg = frame_at(path, t)
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score, x, y, c = checker_search(rgb)
        if score > 12 and c:
            found.append((score, (float(x), float(y), 8.0 * c, 8.0 * c), rgb))
    if not found:
        return None
    # vote: boxes that agree with many others win; then refine the strongest of them
    def close(a, b):
        return all(abs(a[i] - b[i]) < 0.04 * a[2] for i in range(4))
    votes = [sum(close(f[1], g[1]) for g in found) + f[0] / 1000 for f in found]
    score, box, rgb = found[int(np.argmax(votes))]
    return refine(rgb, box)


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


def key_of(board):
    return ''.join((board.piece_at(sq).symbol() if board.piece_at(sq) else '.') for sq in range(64))


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

    def _legal(self, base):
        out = []
        for mv in list(base.legal_moves):
            san = base.san(mv)
            nb = base.copy(stack=False)
            nb.push(mv)
            out.append((nb, [(san, mv)]))
        return out

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
        for i, base in enumerate(bases):
            for nb, steps in self._legal(base):
                v = self._ll(key_of(nb), raw) - (0.5 if i else 0)
                if best is None or v > best[0]:
                    best = (v, nb, steps)
        for nb1, s1 in self._legal(prev):
            for nb2, s2 in self._legal(nb1):
                v = self._ll(key_of(nb2), raw) - 1
                if v > best[0]:
                    best = (v, nb2, s1 + s2)
        llD = self._ll(key, raw)
        hist_i = next((i for i in range(len(self.hist) - 2, -1, -1) if key_of(self.hist[i]) == key), -1)
        if hist_i >= 0 and (best is None or best[0] < llD):
            self.hist = self.hist[:hist_i + 1]
            self.board = self.hist[-1].copy()
            self.events.append({'t': t, 'fen': self.board.fen(), 'san': None, 'from': None, 'to': None})
            return
        if best and best[0] >= llD - 4:
            v, nb, steps = best
            # replay to get each intermediate position
            cur = prev.copy(stack=False)
            if len(steps) == 1 and steps[0][1] not in cur.legal_moves:
                cur.turn = not cur.turn
                cur.ep_square = None
            for san, mv in steps:
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
        img = cv2.imdecode(np.frombuffer(frame_at(path, duration * (i + 0.5) / samples), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            tiles.append(board_tiles(crop256(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), region)))
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


def process(path, on_progress=None, region=None, orientation='auto', fps=FPS, batch=8):
    """Generator of events; also reports progress. Returns final info dict via on_progress."""
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
    pending_t, pending_tiles, pending_imgs = [], [], []
    sent = sent_marks = 0

    def flush():
        nonlocal votes, white_bottom, decided, sent, sent_marks
        if not pending_tiles:
            return
        probs = rec.predict(np.concatenate(pending_tiles)).reshape(len(pending_tiles), 64, 13)
        for t, cells, img in zip(pending_t, probs, pending_imgs):
            if not decided:
                votes += orientation_vote(cells)
                if abs(votes) >= 3:
                    white_bottom, decided = votes > 0, True
                else:
                    continue  # wait until we know which way up the board is
            follower.feed(t, cells_to_squares(cells, white_bottom))
            tracker.feed(t, marks_to_squares(marks.detect(img), white_bottom))
        pending_t.clear(); pending_tiles.clear(); pending_imgs.clear()
        new = follower.events[sent:]
        sent = len(follower.events)
        new_marks = tracker.events[sent_marks:]
        sent_marks = len(tracker.events)
        if on_progress:
            on_progress({'events': new, 'marks': new_marks, 'white_bottom': white_bottom,
                         'progress': min(1.0, (t + 1 / fps) / max(duration, 1e-6))})

    for t, img in frames(path, fps, crop=region, size=(256, 256)):
        pending_t.append(t); pending_tiles.append(board_tiles(img)); pending_imgs.append(img)
        if len(pending_tiles) >= batch:
            flush()
    flush()
    if on_progress:
        on_progress({'progress': 1.0, 'done': True})
