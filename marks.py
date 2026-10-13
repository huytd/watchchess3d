"""Analysis marks drawn on the board (arrows, circles, coloured squares), found with plain
image processing: no model involved.

Lichess and chess.com draw every arrow from the centre of one square to the centre of
another (chess.com draws knight arrows as an L), so instead of looking for arbitrary
shapes we test each possible arrow against a mask of "overlay-coloured" pixels:
the line between the two centres must be covered, the strip beside it must not be
(so a tinted last-move square isn't mistaken for a short arrow), and one end must be
wider than the other (the head).

All of this works on the same 256x256 board crop the square classifier reads, with cells
indexed top-left first; the caller maps cells to squares once the orientation is known.
"""
import numpy as np
import cv2

S = 32          # cell size in the 256x256 crop
EXCESS = 22     # colour a pixel needs beyond "its square's colour mixed with grey"
MAX_ARROWS = 12  # more than this in one frame means we are reading noise, not arrows


def _chroma(x):
    return x - x.mean(axis=-1, keepdims=True)


def colour_excess(img, base):
    """How far each pixel's colour is from anything a mix of its square's colour and a grey
    (a piece, a shadow, an anti-aliased edge) could produce. Arrows and circles are drawn in
    strong translucent colours, so they stand out; pieces and the board don't.
    img, base: (..., 3) float RGB."""
    cp, cb = _chroma(img), _chroma(base)
    s = (cp * cb).sum(-1) / np.maximum((cb * cb).sum(-1), 1.0)
    return np.linalg.norm(cp - np.clip(s, 0, 1)[..., None] * cb, axis=-1)


def _cells(img):
    return img.reshape(8, S, 8, S, 3).transpose(0, 2, 1, 3, 4).astype(np.float32)


def cell_bases(img):
    """(8,8,3) colour of each square, from its rim (pieces don't reach it, an arrow only
    crosses it), and (2,3) the usual light/dark colours across the board."""
    t = _cells(img)
    rim = np.concatenate([t[:, :, 1, 1:S - 1], t[:, :, S - 2, 1:S - 1],
                          t[:, :, 1:S - 1, 1], t[:, :, 1:S - 1, S - 2]], axis=2)  # 8,8,120,3
    base = np.median(rim, axis=2)
    parity = (np.arange(8)[:, None] + np.arange(8)[None, :]) % 2
    usual = np.stack([np.median(base[parity == k], axis=0) for k in (0, 1)])
    # a rim that doesn't agree with itself (a circle, several arrows) can't be trusted
    agree = (np.linalg.norm(rim - base[:, :, None], axis=-1) < 30).mean(axis=2)
    base = np.where((agree < 0.6)[..., None], usual[parity], base)
    return base, usual, parity


def overlay_mask(img):
    base, usual, parity = cell_bases(img)
    full = np.repeat(np.repeat(base, S, axis=0), S, axis=1)
    m = colour_excess(img.astype(np.float32), full) > EXCESS
    # drop lone pixels (compression fringes around pieces); arrows are several pixels wide
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    return m, base, usual, parity


def hue_name(rgb, base):
    """Name the overlay's colour. The pixel is roughly 0.6 overlay + 0.4 square, so take a
    share of the square's tint back out before reading the hue."""
    c = _chroma(np.asarray(rgb, np.float32)) - 0.4 * _chroma(np.asarray(base, np.float32))
    r, g, b = c
    h = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    if h < 15 or h >= 330:
        return 'red'
    if h < 65:
        return 'orange'
    if h < 170:
        return 'green'
    return 'blue'


# ---------------------------------------------------------------- candidate arrows
def _centre(rc):
    return np.array([(rc[1] + 0.5) * S, (rc[0] + 0.5) * S], np.float32)  # x, y


def _polyline_samples(pts, offsets):
    """Points at the given arc lengths along a polyline (extrapolating past either end),
    plus the unit direction and left normal there."""
    pts = np.asarray(pts, np.float32)
    seg = np.diff(pts, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(lens)])
    i = np.clip(np.searchsorted(cum, offsets, side='right') - 1, 0, len(seg) - 1)
    d = seg[i] / lens[i, None]
    p = pts[i] + d * (offsets - cum[i])[:, None]
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return p, d, n


K_LINE, K_SIDE, K_SHAFT, K_HEAD = 24, 12, 10, 10
W_STEPS = np.linspace(-0.4 * S, 0.4 * S, 27)  # across the line; the middle one is on it


def _build_candidates():
    cand = []  # (from_cell, to_cell, corner_cell or None, polyline)
    cells = [(r, c) for r in range(8) for c in range(8)]
    for a in cells:
        for b in cells:
            if a != b:
                cand.append((a, b, None, [_centre(a), _centre(b)]))
            dr, dc = b[0] - a[0], b[1] - a[1]
            if {abs(dr), abs(dc)} == {1, 2}:  # chess.com: knight arrows as an L, long leg first
                corner = (b[0], a[1]) if abs(dr) == 2 else (a[0], b[1])
                cand.append((a, b, corner, [_centre(a), _centre(corner), _centre(b)]))
    # pixel coordinates are rounded as each candidate is built: kept as floats until the
    # end, the tables briefly take over 100 MB
    ix = lambda a: np.clip(np.rint(a), 0, 255).astype(np.int16)
    line, side, prof, frac, back, over = [], [], [], [], [], []
    for _, _, _, pts in cand:
        L = sum(float(np.linalg.norm(np.subtract(q, p))) for p, q in zip(pts, pts[1:]))
        p, _, _ = _polyline_samples(pts, np.linspace(0.2 * S, L - 0.3 * S, K_LINE))
        line.append(ix(p))
        # the strip beside the shaft should be clear (rules out tinted whole squares)
        p, _, n = _polyline_samples(pts, np.linspace(0.2 * S, max(0.2 * S, L - 0.6 * S), K_SIDE))
        side.append(ix(np.concatenate([p + 0.42 * S * n, p - 0.42 * S * n])))
        # cross-sections along the shaft and, closely spaced whatever the length, the head
        at = np.concatenate([np.linspace(0.15 * S, L - 0.6 * S, K_SHAFT), L - np.linspace(0.55, 0.08, K_HEAD) * S])
        p, _, n = _polyline_samples(pts, at)
        prof.append(ix(p[:, None] + W_STEPS[None, :, None] * n[:, None]))
        frac.append(at / L)
        # an arrow begins at a centre, so just behind the start should be clear; and it
        # stops at one, so the line shouldn't carry on for most of another square
        back.append(ix(_polyline_samples(pts, np.array([-0.45 * S]))[0]))
        over.append(ix(_polyline_samples(pts, L + np.array([0.45, 0.7, 0.95]) * S)[0]))
    return (cand, np.stack(line), np.stack(side), np.stack(prof), np.array(frac, np.float32), np.stack(back),
            np.stack(over))


_CAND = None


def _candidates():
    global _CAND
    if _CAND is None:
        _CAND = _build_candidates()
    return _CAND


def _at(mask, pts):
    return mask[pts[..., 1], pts[..., 0]]


def _circle_points():
    a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    out = {}
    for rad in (0.40, 0.44, 0.47, 0.25):
        out[rad] = np.stack([np.cos(a), np.sin(a)], axis=1) * rad * S
    return out


_CIRCLE = _circle_points()


def _dist_to_polyline(p, pts):
    best = np.full(len(p), np.inf)
    for a, b in zip(pts, pts[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def _between(a, m, b):
    """Cell m lies on the straight segment a-b, strictly inside."""
    (ar, ac), (mr, mc), (br, bc) = a, m, b
    return ((mr - ar) * (bc - ac) == (mc - ac) * (br - ar) and min(ar, br) <= mr <= max(ar, br)
            and min(ac, bc) <= mc <= max(ac, bc) and m not in (a, b))


def arrow_tests(mask, wide):
    """Run every candidate through the shape tests. Returns the indices that are covered
    at all and, for those, each test's result (kept separate to make tuning easier)."""
    cand, line, side, prof, frac, back, over = _candidates()
    cover = _at(wide, line).mean(axis=1)
    keep = np.nonzero(cover >= 0.9)[0]
    x = _at(mask, prof[keep])                       # n, K_SHAFT + K_HEAD, len(W_STEPS)
    mid = len(W_STEPS) // 2
    # width counted only where the mask spreads to both sides of the line: an arrowhead
    # does, the corner of an L-shaped arrow or a crossing arrow doesn't
    width = 2 * np.minimum(x[..., :mid].sum(-1), x[..., mid + 1:].sum(-1)) + x[..., mid]
    head = width[:, K_SHAFT:].max(axis=1)
    shaft = np.median(width[:, :K_SHAFT], axis=1)
    # the head end is the heavier one; this is what tells a short arrow from its reverse
    extra = np.maximum(width - shaft[:, None], 0)
    w = frac[keep]
    t = {
        'beside': _at(mask, side[keep]).mean(axis=1) <= 0.2,
        'head': (head >= 0.25 * S) & (head >= 1.5 * np.maximum(shaft, 2)),
        'heavy': (extra * w).sum(axis=1) > 0.5 * extra.sum(axis=1),
        'behind': ~_at(mask, back[keep])[:, 0],
    }
    t['ok'] = t['beside'] & t['head'] & t['heavy'] & t['behind']
    t['past'] = _at(wide, over[keep]).all(axis=1)
    t['sharp'] = _at(mask, line[keep]).mean(axis=1)  # how squarely the line sits on the shaft
    return keep, t


def find_arrows(mask, wide):
    """[(from_cell, to_cell, candidate index)]"""
    cand, line = _candidates()[:2]
    keep, t = arrow_tests(mask, wide)
    passed = {}
    for i in np.nonzero(t['ok'])[0]:
        a, b = cand[keep[i]][:2]
        if (a, b) not in passed or t['sharp'][i] > t['sharp'][passed[(a, b)]]:
            passed[(a, b)] = i
    # the line may carry on past the tip only where another arrow starts there (a chain);
    # otherwise this is the front part of a longer arrow
    starts = {a for a, b in passed}
    passed = {k: i for k, i in passed.items() if not t['past'][i] or k[1] in starts}
    # a straight arrow made of two others end to end is that chain read as one
    for a, b in list(passed):
        if cand[keep[passed[a, b]]][2] is None and any(
                (a, m) in passed and (m, b) in passed for m in starts if _between(a, m, b)):
            passed.pop((a, b))
    # a thick arrow also passes for its near neighbours (a1-b7 for a2-b7, the legs of an L):
    # take the best-aligned, longest first and drop whatever mostly runs along it
    order = sorted(passed.values(), key=lambda i: (-round(t['sharp'][i], 2), -len(cand[keep[i]][3]),
                                                   -np.ptp(line[keep[i]], axis=0).sum()))
    taken = []
    for i in order:
        j = keep[i]
        if all((_dist_to_polyline(line[j].astype(np.float32), np.asarray(cand[k][3])) < 0.25 * S).mean() < 0.5
               for k in taken):
            taken.append(j)
    return [(cand[j][0], cand[j][1], j) for j in taken]


def detect(img):
    """img: (256,256,3) uint8 RGB board crop. Returns a list of marks in cell coordinates:
    ('arrow', from_rc, to_rc, colour) / ('circle', rc, colour) / ('square', rc, colour)."""
    mask, base, usual, parity = overlay_mask(img)
    if mask.mean() > 0.25:  # a scene change or an overlay over the board, not analysis marks
        return []
    wide = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    arrows = find_arrows(mask, wide)
    if len(arrows) > MAX_ARROWS:
        return []
    line = _candidates()[1]
    out = []
    img_f = img.astype(np.float32)
    for a, b, j in arrows:
        px = line[j][_at(mask, line[j])]
        rgb = np.median(img_f[px[:, 1], px[:, 0]], axis=0)
        out.append(('arrow', a, b, hue_name(rgb, usual[parity[a]])))
    # circles (lichess): a ring just inside the square, with the inside left alone
    for r in range(8):
        for c in range(8):
            ctr = _centre((r, c))
            pts = {k: np.clip(np.rint(ctr + v), 0, 255).astype(np.int32) for k, v in _CIRCLE.items()}
            best = max((rad for rad in (0.40, 0.44, 0.47)), key=lambda k: _at(wide, pts[k]).mean())
            if _at(wide, pts[best]).mean() >= 0.8 and _at(mask, pts[0.25]).mean() < 0.5:
                ring = pts[best][_at(mask, pts[best])]
                rgb = np.median(img_f[ring[:, 1], ring[:, 0]], axis=0)
                out.append(('circle', (r, c), hue_name(rgb, base[r, c])))
    # whole squares painted red or blue (chess.com right-click); yellow/green tints are
    # last-move highlights, which the 3D board shows on its own
    for r in range(8):
        for c in range(8):
            u = usual[parity[r, c]]
            if colour_excess(base[r, c], u) > 40:
                name = hue_name(base[r, c], u)
                if name in ('red', 'blue'):
                    out.append(('square', (r, c), name))
    return out


# ---------------------------------------------------------------- over time
class MarkTracker:
    """Smooths per-frame detections: a mark shows after two frames in a row and goes after
    two frames missing, so a passing cursor or a half-drawn arrow doesn't flicker."""

    def __init__(self, fps):
        self.dt = 1.0 / fps
        self.on = {}       # key -> colour
        self.streak = {}
        self.miss = {}
        self.events = []   # {'t', 'arrows': [...], 'circles': [...], 'squares': [...]}

    def feed(self, t, marks):
        """marks: detections already in square names, e.g. ('arrow', 'e2', 'e4', 'green')."""
        now = {m[:-1]: m[-1] for m in marks}
        self.streak = {k: self.streak.get(k, 0) + 1 for k in now}
        changed = False
        for k, colour in now.items():
            self.miss.pop(k, None)
            if k in self.on:
                self.on[k] = colour
            elif self.streak[k] >= 2:
                self.on[k] = colour
                changed = True
        for k in [k for k in self.on if k not in now]:
            self.miss[k] = self.miss.get(k, 0) + 1
            if self.miss[k] >= 2:
                del self.on[k], self.miss[k]
                changed = True
        if changed:
            ev = {'t': max(0.0, t - self.dt), 'arrows': [], 'circles': [], 'squares': []}
            for k, colour in sorted(self.on.items()):
                if k[0] == 'arrow':
                    ev['arrows'].append({'from': k[1], 'to': k[2], 'color': colour})
                else:
                    ev[k[0] + 's'].append({'sq': k[1], 'color': colour})
            self.events.append(ev)
