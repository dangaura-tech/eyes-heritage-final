"""Pair construction, labels, sample weights, CV folds, point-level aggregation, decision rule."""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import CLASSES, C2I, NC, SITES, CACHE

CON, DES, SOIL = C2I["construction"], C2I["destruction"], C2I["soil_sealing"]


def point_dates(z):
    years = np.array([int(d[:4]) for d in z["dates"]])
    return years, z["bad"].astype(bool)


def valid_dates(bad, qual=None, min_n=2):
    good = np.where(~bad)[0]
    if len(good) >= min_n:
        return good, False
    # fallback: least-bad dates
    from imgproc import badness
    order = np.argsort([badness(q) for q in qual]) if qual is not None else np.arange(len(bad))
    return np.sort(order[:max(min_n, len(good))]), True


def all_pairs(idx):
    return [(int(i), int(j)) for a, i in enumerate(idx) for j in idx[a + 1:]]


# optional year-aware inference: point_id -> year; when set, only the core pair is scored for that point
CORE_YEAR = {}


def infer_pairs(pid, vi, years):
    """Pairs scored at inference. Default: every valid pair (incl. first/last). With a known year: core pair."""
    vi = list(vi)
    Y = CORE_YEAR.get(pid)
    if Y is not None and np.isfinite(Y):
        bef = [i for i in vi if years[i] < Y]; aft = [i for i in vi if years[i] >= Y]
        if bef and aft:
            return [(int(bef[-1]), int(aft[0]))]
    return all_pairs(vi)


# v4 labels: explicit labelled pairs per point from train_pairs.csv (loaded if cache/pairs_v4.parquet exists)
EXPLICIT = {}
try:
    import pandas as _pd
    _fn = CACHE / "pairs_explicit.parquet"
    if not _fn.exists():
        _fn = CACHE / "pairs_v4.parquet"
    if _fn.exists():
        for _r in _pd.read_parquet(_fn).itertuples():
            EXPLICIT.setdefault(_r.point_id, []).append((int(_r.i), int(_r.j), int(_r.label), float(_r.sample_weight),
                                                         bool(_r.adjacent)))
except Exception as _e:  # pragma: no cover
    print("explicit pairs not loaded:", _e)


def train_pairs_for(r, years, vi, ambiguous="exclude", reverse=False):
    """Training pairs of one point: explicit v4 pairs when available, else derived from the year labels.
    Explicit pairs keep the provided sample_weight; adjacent pairs act as 'core' pairs; optional time reversal."""
    lst = EXPLICIT.get(r.point_id)
    if lst is None:
        return label_pairs(r.y, r.year, years, list(vi), ambiguous=ambiguous, reverse=reverse,
                           year_before=r.get("year_before"))
    out = []
    for i, j, lab, sw, adj in lst:
        out.append((i, j, lab, sw, adj))
        if reverse and lab in (CON, DES):
            out.append((j, i, DES if lab == CON else CON, 0.5 * sw, False))
        elif reverse and lab == NC:
            out.append((j, i, NC, sw, False))
    return out


def label_pairs(cls, year, years, idx, ambiguous="exclude", reverse=False, year_before=None):
    """Return list of (i, j, label, weight, is_core) training pairs for one point.
    v1 labels: `year` = first year in which the change is visible (before: < year, after: >= year).
    v3 labels: `year_before` / `year` (= year_after) bracket the change (before: <= year_before, after: >= year_after)."""
    out = []
    pairs = all_pairs(idx)
    if cls == NC:
        for i, j in pairs:
            out.append((i, j, NC, 1.0, False))
            if reverse:
                out.append((j, i, NC, 1.0, False))
        return out
    Y = int(year)
    YB = int(year_before) if (year_before is not None and np.isfinite(year_before)) else Y - 1
    before = [i for i in idx if years[i] <= YB]; after = [i for i in idx if years[i] >= Y]
    core = (before[-1], after[0]) if before and after else None
    for i, j in pairs:
        span = years[i] <= YB and years[j] >= Y
        if span:
            out.append((i, j, cls, 1.0, (i, j) == core))
            if reverse and cls in (CON, DES):
                out.append((j, i, DES if cls == CON else CON, 0.5, False))
        elif ambiguous == "nochange":
            out.append((i, j, NC, 0.2, False))
    if not any(o[2] == cls for o in out) and len(idx) >= 2:
        # no spanning pair among valid dates (e.g. year outside the valid range): use first/last valid
        out.append((idx[0], idx[-1], cls, 0.5, True))
    return out


def point_weights(pts, use_label_w=True, balance=True):
    w = np.ones(len(pts))
    if use_label_w and "train_weight" in pts:      # v3 labels: reviewer-assigned reliability weights
        w *= pts.train_weight.fillna(1.0).to_numpy(float)
    elif use_label_w:
        ba = pts.best_accuracy.fillna(70).to_numpy(float)
        w *= np.clip((ba - 40) / 55, 0.3, 1)
        w *= np.where(pts.verified_ok.fillna(0).to_numpy() > 0, 1.3, 1.0)
        w *= np.where(pts.conflict.fillna(0).to_numpy() > 0, 0.5, 1.0)
        w *= np.where(pts.n_teams.fillna(1).to_numpy() >= 2, 1.2, 1.0)
    if balance:
        key = pts.site + "|" + pts["class"]
        freq = key.map(key.value_counts()).to_numpy(float)
        w *= 1 / np.sqrt(freq)
    return w / w.mean()


def make_folds(pts, seed=0):
    """Returns dict with 'in' (array fold id 0..4 by spatial blocks) and 'loso' (site index)."""
    strat = (pts.site + "|" + pts["class"]).to_numpy()
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    f_in = np.full(len(pts), -1)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(pts)), strat, pts.block_id.to_numpy())):
        f_in[te] = k
    f_loso = pts.site.map({s: i for i, s in enumerate(SITES)}).to_numpy()
    return {"in": f_in, "loso": f_loso}


# ---------------- aggregation & decision ----------------
def aggregate(pair_probs, pair_point, n_points, mode="max", k=2):
    """pair_probs (P,4), pair_point (P,) -> point probs (N,4)."""
    out = np.zeros((n_points, 4)); out[:, NC] = 1.0
    order = np.argsort(pair_point, kind="stable")
    pp = pair_point[order]; pr = pair_probs[order]
    bounds = np.searchsorted(pp, np.arange(n_points + 1))
    for n in range(n_points):
        a, b = bounds[n], bounds[n + 1]
        if a == b:
            continue
        P = pr[a:b]; pc = 1 - P[:, NC]
        top = np.argsort(-pc)[:k]
        if mode == "max":
            pch = pc.max()
        elif mode == "top2":
            pch = pc[top].mean()
        elif mode == "mean":
            pch = pc.mean()
        else:
            raise ValueError(mode)
        cd = P[top][:, :3].sum(0); cd = cd / max(cd.sum(), 1e-9)
        out[n, :3] = pch * cd; out[n, NC] = 1 - pch
    return out


def decide(P, t=0.5, b=(0, 0, 0)):
    pch = 1 - P[:, NC]
    lc = np.log(np.clip(P[:, :3], 1e-9, 1)) + np.asarray(b)[None]
    pred = lc.argmax(1)
    return np.where(pch >= t, pred, NC)


def tune_decision(P, y, grid_t=np.linspace(0.1, 0.98, 45), grid_b=np.linspace(-1.5, 1.5, 13), prior=0.01):
    """Tune t and per-class log biases maximising macro-F1 - prior*||b||^2. Coordinate ascent."""
    from common import macro_f1
    best = (0.5, np.zeros(3)); best_s = -1
    t, b = 0.5, np.zeros(3)
    for _ in range(3):
        for tt in grid_t:
            s = macro_f1(y, decide(P, tt, b)) - prior * (b ** 2).sum()
            if s > best_s: best_s, best = s, (tt, b.copy())
        t = best[0]; b = best[1].copy()
        for c in range(3):
            for v in grid_b:
                bb = b.copy(); bb[c] = v
                s = macro_f1(y, decide(P, t, bb)) - prior * (bb ** 2).sum()
                if s > best_s: best_s, best = s, (t, bb.copy())
            b = best[1].copy()
    return best[0], best[1], best_s
