"""v6 ensemble / decision evaluation, all nested (fit on held-in sites for M_out, on LOSO OOF outside the in-domain
fold for M_in). New methods (not the ones already rejected):
  combine : mean (baseline), geo (geometric mean of probabilities), caruana (greedy forward selection with replacement)
  decide  : tb (threshold + class biases, baseline), consensus (change iff >= q of members exceed their own threshold;
            type = mean change-type distribution + biases), split (gate = mean p_change; type = p_change-weighted mean
            of member type distributions + biases)
usage: v6_eval.py <member1,...> [--tag name]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
from common import *
from pairs import make_folds, tune_decision, decide
import numpy as np


def fast_f1(y, p, per_class=False):
    cm = np.bincount(np.asarray(y) * 4 + np.asarray(p), minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    return f if per_class else f.mean()


common.macro_f1 = fast_f1
members = sys.argv[1].split(",")
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "v6"
pts = load_points(); y = pts.y.to_numpy(); site = pts.site.to_numpy(); fin = make_folds(pts, 0)["in"]; N = len(y)
HAS0 = "y_orig" in pts.columns
ORG = (pts.y_orig >= 0).to_numpy() if HAS0 else np.ones(N, bool); Y0 = pts.y_orig.to_numpy() if HAS0 else y
OL = {m: np.load(CACHE / "oof" / f"{m}.npz")["oof_loso"].mean(0) for m in members}
OI = {m: np.load(CACHE / "oof" / f"{m}.npz")["oof_in"].mean(0) for m in members}
LOG = lambda P: np.log(np.clip(P, 1e-6, 1))


def norm(P):
    return P / P.sum(1, keepdims=True)


def combine(sel, kind_src, ix, how):
    S = OL if kind_src == "lo" else OI
    if how == "geo":
        return norm(np.exp(np.mean([LOG(S[m][ix]) for m in sel], 0)))
    return np.mean([S[m][ix] for m in sel], 0)


def caruana(hi, rounds=12):
    """Greedy forward selection with replacement on held-in LOSO OOF; score = macro-F1 after tuning t/b on held-in."""
    bag = []; best = -1
    for _ in range(rounds):
        cand = None
        for m in members:
            P = np.mean([OL[x][hi] for x in bag + [m]], 0)
            _, _, s = tune_decision(P, y[hi])
            if cand is None or s > cand[1]: cand = (m, s)
        if cand[1] < best - 1e-9: break
        bag.append(cand[0]); best = max(best, cand[1])
    return bag


def member_thresholds(sel, hi):
    """Per-member change threshold maximising change-vs-no_change F1 on held-in LOSO OOF."""
    ts = {}
    yb = (y[hi] != NC)
    for m in set(sel):
        pc = 1 - OL[m][hi][:, NC]; best = (0, 0.5)
        for t in np.linspace(0.1, 0.95, 35):
            p = pc >= t; tp = (p & yb).sum(); f = 2 * tp / max(p.sum() + yb.sum(), 1)
            if f > best[0]: best = (f, t)
        ts[m] = best[1]
    return ts


def decide_rule(rule, sel, how, hi, kind_src, ix):
    Pf = combine(sel, "lo", hi, how); Pa = combine(sel, kind_src, ix, how)
    if rule == "tb":
        t, b, _ = tune_decision(Pf, y[hi]); return decide(Pa, t, b)
    S_apply = OL if kind_src == "lo" else OI
    if rule == "consensus":
        ts = member_thresholds(sel, hi)
        vote = lambda S, idx: np.mean([(1 - S[m][idx][:, NC]) >= ts[m] for m in sel], 0)
        vf, va = vote(OL, hi), vote(S_apply, ix)
        best = (-1, None)
        for q in np.linspace(0.2, 0.8, 13):
            for b in [np.zeros(3)]:
                pf = np.where(vf >= q, (LOG(Pf[:, :3]) + b).argmax(1), NC); s = fast_f1(y[hi], pf)
                if s > best[0]: best = (s, q)
        q = best[1]
        _, bb, _ = tune_decision(Pf, y[hi])
        return np.where(va >= q, (LOG(Pa[:, :3]) + bb).argmax(1), NC)
    if rule == "split":
        def build(S, idx):
            pc = np.stack([1 - S[m][idx][:, NC] for m in sel], 1)             # (n, M)
            ty = np.stack([norm(S[m][idx][:, :3] + 1e-9) for m in sel], 1)   # (n, M, 3)
            w = pc / pc.sum(1, keepdims=True).clip(1e-9)
            td = (ty * w[..., None]).sum(1); g = pc.mean(1)
            return np.c_[td * g[:, None], 1 - g]
        Qf, Qa = build(OL, hi), build(S_apply, ix)
        t, b, _ = tune_decision(Qf, y[hi]); return decide(Qa, t, b)
    raise ValueError(rule)


def run(how, rule):
    po = np.zeros(N, int); pi = np.zeros(N, int); chosen = {}
    for s in SITES:
        te = np.where(site == s)[0]; hi = np.where(site != s)[0]
        sel = caruana(hi) if how == "caruana" else members
        chosen[s] = sel
        po[te] = decide_rule(rule, sel, "mean" if how == "caruana" else how, hi, "lo", te)
    for k in np.unique(fin):
        te = np.where(fin == k)[0]; hi = np.where(fin != k)[0]
        sel = caruana(hi) if how == "caruana" else members
        pi[te] = decide_rule(rule, sel, "mean" if how == "caruana" else how, hi, "in", te)
    Mi, Mo = fast_f1(y, pi), fast_f1(y, po)
    mi0, mo0 = fast_f1(Y0[ORG], pi[ORG]), fast_f1(Y0[ORG], po[ORG])
    return dict(combine=how, rule=rule, M_in=Mi, M_out=Mo, S=hm(Mi, Mo), S_org=hm(mi0, mo0),
                perclass_out=fast_f1(y, po, True).round(3).tolist(),
                per_site_out={s: round(float(fast_f1(y[site == s], po[site == s])), 3) for s in SITES},
                caruana_bags=({s: dict(zip(*np.unique(v, return_counts=True))) for s, v in chosen.items()} if how == "caruana" else None),
                pred_in=pi, pred_out=po)


res = []
for how in ("mean", "geo", "caruana"):
    for rule in ("tb", "consensus", "split"):
        r = run(how, rule); res.append(r)
        print(f"combine={how:8s} rule={rule:9s} | S {r['S']:.4f} M_in {r['M_in']:.4f} M_out {r['M_out']:.4f} | S_org {r['S_org']:.4f} | "
              f"aleppo {r['per_site_out']['aleppo']:.3f} | per-class {r['perclass_out']}", flush=True)
best = max(res, key=lambda r: r["S"])
cells = pts.cell_id.to_numpy(); uc = np.unique(cells); ix = {c: np.where(cells == c)[0] for c in uc}
rng = np.random.RandomState(0); B = []
for _ in range(500):
    sel = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])
    B.append(hm(fast_f1(y[sel], best["pred_in"][sel]), fast_f1(y[sel], best["pred_out"][sel])))
best["S_ci"] = np.percentile(B, [2.5, 97.5]).round(4).tolist()
clean = lambda r: {k: (v if not isinstance(v, dict) else {kk: (str(vv) if not isinstance(vv, (int, float, str, list)) else vv) for kk, vv in v.items()})
                   for k, v in r.items() if k not in ("pred_in", "pred_out")}
print("BEST:", clean(best))
json.dump(dict(members=members, results=[clean(r) for r in res], best=clean(best)), open(REPORTS / f"v6eval_{tag}.json", "w"),
          indent=1, default=str)
