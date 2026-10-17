"""Image-level helpers: quality scores, co-registration, radiometric normalisation, tile reading.
Shared by training preprocessing and predict.py so both paths are identical."""
import numpy as np, cv2, rasterio
from rasterio.windows import Window

TILE = 384
PAD = 16          # extra margin read around tile so shifts never leave the read window
QWIN = 256        # 128 m quality window
CORE = 128        # 64 m core window for valid fraction
REGWIN = 256
MAX_SHIFT = 12.0
MIN_RESP = 0.05


def read_window(src, col, row, size):
    """Read RGBA window of `size` centred at (col,row); out-of-raster pixels get alpha 0."""
    c0 = int(round(col)) - size // 2; r0 = int(round(row)) - size // 2
    w = Window(c0, r0, size, size)
    arr = src.read(window=w, boundless=True, fill_value=0)  # (4,H,W)
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


def center(a, size):
    h = a.shape[0]; o = (h - size) // 2
    return a[o:o + size, o:o + size]


def quality(rgba):
    """Quality metrics on a QWIN window. Only valid pixels contribute."""
    win = center(rgba, QWIN)
    valid = win[..., 3] > 0
    core_valid = center(rgba, CORE)[..., 3] > 0
    out = dict(valid_frac=float(valid.mean()), core_valid=float(core_valid.mean()))
    if valid.sum() < 100:
        out.update(lap_var=0.0, contrast=0.0, sat=0.0, bright=0.0, haze=1.0)
        return out
    rgb = win[..., :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    er = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    g = gray[valid] / 255.0
    sat = float(hsv[..., 1][valid].mean() / 255.0)
    bright = float(g.mean()); contrast = float(g.std())
    out.update(lap_var=float(lap[er].var()) if er.sum() > 50 else 0.0, contrast=contrast, sat=sat, bright=bright,
               haze=float((1 - sat) * bright * (1 - min(contrast * 4, 1.0))))
    return out


def flag_bad(q_list):
    """Given quality dicts of all dates of one point, return bool list (True = bad)."""
    lv = np.array([q["lap_var"] for q in q_list]); ct = np.array([q["contrast"] for q in q_list])
    hz = np.array([q["haze"] for q in q_list]); cv = np.array([q["core_valid"] for q in q_list])
    med_lv = np.median(lv[cv >= 0.7]) if (cv >= 0.7).any() else np.median(lv)
    med_ct = np.median(ct[cv >= 0.7]) if (cv >= 0.7).any() else np.median(ct)
    med_hz = np.median(hz[cv >= 0.7]) if (cv >= 0.7).any() else np.median(hz)
    bad = []
    for i in range(len(q_list)):
        hazy = (lv[i] < 0.25 * med_lv and ct[i] < 0.7 * med_ct) or (hz[i] > 1.6 * med_hz and ct[i] < 0.6 * med_ct) \
               or ct[i] < 0.03
        bad.append(bool(cv[i] < 0.7 or hazy))
    return bad


def badness(q):
    """Scalar badness for fallback ranking (lower is better)."""
    return (1 - q["core_valid"]) * 3 + q["haze"] - 0.5 * min(q["contrast"] * 5, 1)


def grad_mag(rgba):
    g = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.sqrt(gx * gx + gy * gy)
    m[rgba[..., 3] == 0] = 0
    return m


_HANN = cv2.createHanningWindow((REGWIN, REGWIN), cv2.CV_32F)


def phase_shift(ref_rgba, mov_rgba):
    """Return (dx, dy, response): mov content at p corresponds to ref content at p - (dx,dy).
    Sampling mov at p + (dx,dy) aligns it to ref."""
    a = grad_mag(center(ref_rgba, REGWIN)); b = grad_mag(center(mov_rgba, REGWIN))
    if (center(ref_rgba, REGWIN)[..., 3] > 0).mean() < 0.5 or (center(mov_rgba, REGWIN)[..., 3] > 0).mean() < 0.5:
        return 0.0, 0.0, 0.0
    (dx, dy), resp = cv2.phaseCorrelate(a, b, _HANN)
    return float(dx), float(dy), float(resp)


def shift_img(rgba, dx, dy):
    """Sample image at p + (dx,dy). Alpha uses nearest and is eroded where it was resampled."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = rgba.shape[:2]
    rgb = cv2.warpAffine(rgba[..., :3], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    al = cv2.warpAffine(rgba[..., 3], M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.dstack([rgb, al])
    out[al == 0, :3] = 0
    return out


def hist_match(after, before):
    """Histogram-match `after` RGB to `before`, using only pixels valid in both. Returns new after (uint8)."""
    v = (after[..., 3] > 0) & (before[..., 3] > 0)
    out = after.copy()
    if v.sum() < 200:
        return out
    for ch in range(3):
        a = after[..., ch][v]; b = before[..., ch][v]
        ah = np.bincount(a, minlength=256).astype(np.float64); bh = np.bincount(b, minlength=256).astype(np.float64)
        ac = np.cumsum(ah) / ah.sum(); bc = np.cumsum(bh) / bh.sum()
        lut = np.interp(ac, bc, np.arange(256)).clip(0, 255).astype(np.uint8)
        out[..., ch] = lut[after[..., ch]]
    out[after[..., 3] == 0, :3] = 0
    return out


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def build_point_tiles(srcs, col, row, max_shift=MAX_SHIFT, min_resp=MIN_RESP, verify=False, pad=PAD):
    """srcs: list of open rasterio datasets (one per date, same grid). Returns dict with tiles, raw tiles,
    quality, bad flags, shifts, reference index. verify=True (v2): accept a shift only if the gradient NCC of the
    central window increases after applying it (guards against false phase-correlation peaks)."""
    PAD = pad
    size = TILE + 2 * PAD
    big = [read_window(s, col, row, size) for s in srcs]
    qs = [quality(center(b, TILE)) for b in big]
    bad = flag_bad(qs)
    good = [i for i, b in enumerate(bad) if not b]
    cand = good if good else list(range(len(big)))
    ref = max(cand, key=lambda i: qs[i]["lap_var"] * (qs[i]["core_valid"] >= 0.7))
    shifts = np.zeros((len(big), 3), np.float32)
    tiles = np.zeros((len(big), TILE, TILE, 4), np.uint8)
    raw = np.zeros_like(tiles)
    for i, b in enumerate(big):
        raw[i] = center(b, TILE)
        if i == ref:
            tiles[i] = raw[i]; continue
        dx, dy, resp = phase_shift(center(big[ref], TILE), center(b, TILE))
        shifts[i] = (dx, dy, resp)
        ok = np.hypot(dx, dy) <= max_shift and resp >= min_resp
        if ok and verify:
            g0 = grad_mag(center(big[ref], REGWIN))
            before = _ncc(g0, grad_mag(center(b, REGWIN))); after = _ncc(g0, grad_mag(center(shift_img(b, dx, dy), REGWIN)))
            ok = after > before + 0.02
        if ok:
            tiles[i] = center(shift_img(b, dx, dy), TILE)
        else:
            tiles[i] = raw[i]; shifts[i, 2] = -abs(resp)  # negative response marks "not applied"
    return dict(tiles=tiles, raw=raw, quality=qs, bad=np.array(bad), shifts=shifts, ref=ref)
