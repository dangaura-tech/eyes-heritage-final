"""v2 ensemble-level fixes, all evaluated nested (fit on held-in sites / points only):
  base : equal weights, t/b nested (current submission protocol)
  s1   : per-site standardisation of logit(p_change) using only that site's own (unlabelled) predictions
  s2   : stacking meta-learner (multinomial LR on member log-probs + log #pairs + #valid dates) with inner LOSO
  s3   : pair-count correction logit(p_change) - beta * log(#pairs), beta fitted on held-in data
  auc  : LOSO AUROC / ECE of raw vs BCTS-calibrated p_change (output calibration)
usage: v2_ensemble.py <member1,...> [--tag name]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
from common import *
from pairs import make_folds, tune_decision, decide, valid_dates, all_pairs
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def fast_f1(y, p, per_class=False):
    cm = np.bincount(np.asarray(y) * 4 + np.asarray(p), minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    return f if per_class else f.mean()


common.macro_f1 = fast_f1
members = sys.argv[1].split(",")
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "v2"
pts = load_points(); y = pts.y.to_numpy(); site = pts.site.to_numpy(); fin = make_folds(pts, 0)["in"]
OL = np.stack([np.load(CACHE / "oof" / f"{m}.npz")["oof_loso"].mean(0) for m in members])
OI = np.stack([np.load(CACHE / "oof" / f"{m}.npz")["oof_in"].mean(0) for m in members])
npairs = np.zeros(len(pts)); nval = np.zeros(len(pts))
for k, (pid, s) in enumerate(zip(pts.point_id, pts.site)):
    z = np.load(tile_path(s, pid), allow_pickle=True)
    vi, _ = valid_dates(z["bad"].astype(bool), list(z["quality"])); npairs[k] = len(all_pairs(list(vi))); nval[k] = len(vi)
eps = 1e-6
logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))
sig = lambda z: 1 / (1 + np.exp(-z))


def with_pchange(P, pc_new):
    """Replace p_change of P (n,4) keeping the conditional change-class distribution."""
    cd = P[:, :3] / np.clip(P[:, :3].sum(1, keepdims=True), 1e-9, None)
    out = np.zeros_like(P); out[:, :3] = cd * pc_new[:, None]; out[:, 3] = 1 - pc_new
    return out


def eq(S, ix):
    return S[:, ix].mean(0)


# ---- transforms: f(P_fit, idx_fit, P_apply, idx_apply, site_apply) -> P_apply'
def t_base(Pf, ixf, Pa, ixa):
    return Pf, Pa


def site_std(P, ix):
    z = logit(1 - P[:, NC]); out = np.zeros_like(z)
    for s in np.unique(site[ix]):
        m = site[ix] == s; out[m] = (z[m] - z[m].mean()) / (z[m].std() + 1e-6)
    return out


def t_s1(Pf, ixf, Pa, ixa):
    # held-in: each site standardised on its own predictions; target likewise. Rescale to a common logit scale.
    zf = site_std(Pf, ixf); za = site_std(Pa, ixa)
    return with_pchange(Pf, sig(zf)), with_pchange(Pa, sig(za))


def t_s3(Pf, ixf, Pa, ixa):
    best = (-1, 0.0)
    for beta in np.linspace(0, 1.0, 11):
        Q = with_pchange(Pf, sig(logit(1 - Pf[:, NC]) - beta * np.log(npairs[ixf])))
        _, _, s = tune_decision(Q, y[ixf])
        if s > best[0]: best = (s, beta)
    b = best[1]
    f = lambda P, ix: with_pchange(P, sig(logit(1 - P[:, NC]) - b * np.log(npairs[ix])))
    return f(Pf, ixf), f(Pa, ixa)


def meta_X(Sst, ix):
    L = np.log(np.clip(Sst[:, ix], 1e-6, 1)).transpose(1, 0, 2).reshape(len(ix), -1)
    return np.c_[L, np.log(npairs[ix]), nval[ix]]


def t_s2_factory(S_apply):
    def t_s2(Pf, ixf, Pa, ixa):
        # inner LOSO over the held-in sites gives honest held-in stacked predictions for t/b tuning
        Qf = np.zeros((len(ixf), 4))
        for s in np.unique(site[ixf]):
            m = site[ixf] == s
            lr = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(meta_X(OL, ixf[~m]), y[ixf[~m]])
            Qf[m] = lr.predict_proba(meta_X(OL, ixf[m]))
        lr = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(meta_X(OL, ixf), y[ixf])
        return Qf, lr.predict_proba(meta_X(S_apply, ixa))
    return t_s2


def nested(transform_for):
    """transform_for(kind) -> transform; kind 'loso' applies to LOSO OOF, 'in' applies to in-domain OOF."""
    po = np.zeros(len(y), int); pi = np.zeros(len(y), int)
    for s in SITES:
        te = np.where(site == s)[0]; hi = np.where(site != s)[0]
        Pf, Pa = transform_for("loso")(eq(OL, hi), hi, eq(OL, te), te)
        t, b, _ = tune_decision(Pf, y[hi]); po[te] = decide(Pa, t, b)
    for k in np.unique(fin):
        te = np.where(fin == k)[0]; hi = np.where(fin != k)[0]
        Pf, Pa = transform_for("in")(eq(OL, hi), hi, eq(OI, te), te)
        t, b, _ = tune_decision(Pf, y[hi]); pi[te] = decide(Pa, t, b)
    Mi, Mo = fast_f1(y, pi), fast_f1(y, po)
    return dict(M_in=Mi, M_out=Mo, S=hm(Mi, Mo), perclass_out=fast_f1(y, po, True).round(3).tolist(),
                per_site_out={s: round(float(fast_f1(y[site == s], po[site == s])), 3) for s in SITES})


res = {}
res["base"] = nested(lambda kind: t_base)
res["s1_site_std_pchange"] = nested(lambda kind: t_s1)
res["s3_pair_count_corr"] = nested(lambda kind: t_s3)
res["s2_stacking_lr"] = nested(lambda kind: t_s2_factory(OL if kind == "loso" else OI))
for k, r in res.items():
    print(f"{k:24s} M_in {r['M_in']:.4f} M_out {r['M_out']:.4f} S {r['S']:.4f} aleppo {r['per_site_out']['aleppo']:.3f} "
          f"marrakesh {r['per_site_out']['marrakesh']:.3f} perclass {r['perclass_out']}", flush=True)

# ---- output calibration: BCTS on the equal-weight ensemble, nested; AUROC + ECE of p_change ----
from scipy.optimize import minimize
def bcts_fit(P, yy):
    L = np.log(np.clip(P, 1e-6, 1))
    def nll(th):
        Z = L / np.exp(th[0]) + np.r_[th[1:], 0.0]; Z -= Z.max(1, keepdims=True)
        Q = np.exp(Z); Q /= Q.sum(1, keepdims=True); return -np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-9, 1)).mean()
    return minimize(nll, np.zeros(4), method="L-BFGS-B").x
def bcts_apply(P, th):
    Z = np.log(np.clip(P, 1e-6, 1)) / np.exp(th[0]) + np.r_[th[1:], 0.0]; Z -= Z.max(1, keepdims=True)
    Q = np.exp(Z); return Q / Q.sum(1, keepdims=True)
def ece_bin(pc, yb, nb=10):
    e = 0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (pc > lo) & (pc <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(yb[m].mean() - pc[m].mean())
    return float(e)
Praw = eq(OL, np.arange(len(y))); Pcal = np.zeros_like(Praw)
for s in SITES:
    te = site == s; th = bcts_fit(Praw[~te], y[~te]); Pcal[te] = bcts_apply(Praw[te], th)
yb = (y != NC).astype(int)
cal = dict(auroc_raw=float(roc_auc_score(yb, 1 - Praw[:, NC])), auroc_bcts=float(roc_auc_score(yb, 1 - Pcal[:, NC])),
           ece_change_raw=ece_bin(1 - Praw[:, NC], yb), ece_change_bcts=ece_bin(1 - Pcal[:, NC], yb),
           brier_raw=float(((1 - Praw[:, NC]) - yb).__pow__(2).mean()), brier_bcts=float(((1 - Pcal[:, NC]) - yb).__pow__(2).mean()))
print("output calibration (LOSO, nested BCTS):", {k: round(v, 4) for k, v in cal.items()})
json.dump(dict(members=members, results=res, output_calibration=cal), open(REPORTS / f"v2ens_{tag}.json", "w"), indent=1)

# ---- monotone output calibration of p_change (Platt on logit p_change): keeps AUROC and decisions unchanged ----
def platt_fit(pc, yb):
    z = logit(pc)
    def nll(th):
        q = np.clip(sig(th[0] * z + th[1]), 1e-9, 1 - 1e-9); return -(yb * np.log(q) + (1 - yb) * np.log(1 - q)).mean()
    return minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B").x
pc_raw = 1 - Praw[:, NC]; pc_pl = np.zeros_like(pc_raw)
for s in SITES:
    te = site == s; th = platt_fit(pc_raw[~te], yb[~te]); pc_pl[te] = sig(th[0] * logit(pc_raw[te]) + th[1])
cal.update(auroc_platt=float(roc_auc_score(yb, pc_pl)), ece_change_platt=ece_bin(pc_pl, yb),
           brier_platt=float(((pc_pl - yb) ** 2).mean()), platt_all=platt_fit(pc_raw, yb).tolist())
print("monotone Platt on p_change (LOSO, nested):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cal.items()})
json.dump(dict(members=members, results=res, output_calibration=cal), open(REPORTS / f"v2ens_{tag}.json", "w"), indent=1)
