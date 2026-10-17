"""Exploratory data analysis + error analysis of the current submission -> reports/eda.md, reports/figs/eda_*.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from pairs import valid_dates, all_pairs, make_folds, decide
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

pts = load_points(); q = pd.read_parquet(CACHE / "quality.parquet")
figs = REPORTS / "figs"; figs.mkdir(exist_ok=True)
out = ["# Exploratory data analysis", ""]
md = lambda df: df.to_markdown()

# ---- 1. labels ----
ct = pd.crosstab(pts.site, pts["class"]).reindex(columns=CLASSES); ct["total"] = ct.sum(1)
out += ["## 1. Labels per site", md(ct), "",
        "Share of each class within a site:", md((pd.crosstab(pts.site, pts["class"], normalize="index")[CLASSES] * 100).round(1)), ""]

# ---- 2. change year vs. available dates ----
rows = []
for _, p in pts.iterrows():
    z = np.load(tile_path(p.site, p.point_id), allow_pickle=True)
    yrs = np.array([int(d[:4]) for d in z["dates"]]); bad = z["bad"].astype(bool)
    vi, fb = valid_dates(bad, list(z["quality"]))
    span = None
    if p.y != NC and np.isfinite(p.year):
        bef = [i for i in vi if yrs[i] < p.year]; aft = [i for i in vi if yrs[i] >= p.year]
        span = bool(bef and aft)
        gap = (yrs[aft[0]] - yrs[bef[-1]]) if (bef and aft) else np.nan
    else:
        gap = np.nan
    rows.append(dict(point_id=p.point_id, site=p.site, cls=p["class"], year=p.year, n_dates=len(yrs), n_valid=len(vi),
                     n_pairs=len(all_pairs(list(vi))), fallback=fb, spans=span, gap=gap,
                     y_first=yrs[vi[0]], y_last=yrs[vi[-1]]))
D = pd.DataFrame(rows)
yc = pd.crosstab(pts[pts.y != NC].year.astype(int), pts[pts.y != NC]["class"]).reindex(columns=CLASSES[:3]).fillna(0).astype(int)
out += ["## 2. Change year (changed classes)", md(yc), "",
        f"- `no_change` points with year = null: {int(((pts.y == NC) & pts.year.isna()).sum())} / {int((pts.y == NC).sum())}",
        f"- changed points whose year is **not spanned** by a valid before/after pair: {int((D.spans == False).sum())} "
        f"({(D.spans == False).mean() / max((D.spans.notna()).mean(), 1e-9) * 100:.1f} % of changed points); "
        "these can only be trained via the first/last fallback pair",
        f"- years between the last valid 'before' and first valid 'after' date (core pair): median {D.gap.median():.1f}, "
        f"max {D.gap.max():.0f}", ""]
out += ["Changed points per site whose year equals the **first** or **last** available year (hardest to see):",
        md(D[D.spans.notna()].assign(edge=lambda d: (d.year == d.y_first) | (d.year == d.y_last))
           .groupby("site").edge.agg(["sum", "mean"]).round(3)), ""]

# ---- 3. label quality ----
lq = pts.groupby("class").agg(n=("y", "size"), best_acc_mean=("best_accuracy", "mean"), best_acc_min=("best_accuracy", "min"),
                              conflict=("conflict", "sum"), verified=("verified_ok", "sum"),
                              multi_team=("n_teams", lambda s: int((s >= 2).sum())),
                              class_uncertain=("class_uncertain", "sum")).reindex(CLASSES).round(1)
out += ["## 3. Label quality (metadata; used for weights only)", md(lq), ""]
ls = pts.groupby("site").agg(best_acc_mean=("best_accuracy", "mean"), conflict=("conflict", "sum"),
                             verified=("verified_ok", "sum"), class_uncertain=("class_uncertain", "sum")).round(1)
out += [md(ls), ""]

# ---- 4. spatial structure ----
sp = []
for s in SITES:
    P = pts[pts.site == s][["utm_x", "utm_y"]].to_numpy(); t = cKDTree(P); d, _ = t.query(P, 2)
    same_cell = pts[pts.site == s].groupby("cell_id").size()
    sp.append(dict(site=s, n=len(P), nn_median_m=round(float(np.median(d[:, 1])), 1), nn_p10_m=round(float(np.percentile(d[:, 1], 10)), 1),
                   pts_within_20m=int((d[:, 1] < 20).sum()), cells=len(same_cell), max_pts_per_cell=int(same_cell.max()),
                   blocks=pts[pts.site == s].block_id.nunique()))
out += ["## 4. Spatial structure", md(pd.DataFrame(sp).set_index("site")), "",
        "Nearest-neighbour distances under ~20 m mean neighbouring points share most of a 128 m crop; spatial CV groups "
        "them by 200 m block.", ""]

# ---- 5. imagery quality & co-registration ----
qs = q.groupby("site").agg(dates=("t", "nunique"), bad_frac=("bad", "mean"), valid=("core_valid", "mean"),
                           contrast=("contrast", "mean"), haze=("haze", "mean")).round(3)
nr = q[~q.ref].assign(shift=lambda d: np.hypot(d.dx, d.dy))
qs["shift_median_px"] = nr.groupby("site")["shift"].median().round(2)
qs["shift_p90_px"] = nr.groupby("site")["shift"].quantile(0.9).round(2)
qs["shift_applied"] = nr.groupby("site").resp.apply(lambda r: (r > 0).mean()).round(3)
out += ["## 5. Imagery quality and co-registration", md(qs), "",
        md(D.groupby("site").agg(valid_dates_mean=("n_valid", "mean"), pairs_mean=("n_pairs", "mean"),
                                 fallback_points=("fallback", "sum")).round(2)), ""]
badtab = q.groupby(["site", "date"]).bad.mean().unstack(0).round(2)
out += ["Fraction of points flagged bad per date:", md(badtab), ""]

# ---- 6. domain shift between cities (colour statistics of the centre 64 m) ----
cs = []
for _, p in pts.iterrows():
    z = np.load(tile_path(p.site, p.point_id), allow_pickle=True)
    t = z["tiles"][int(z["ref"])][128:256, 128:256]; v = t[..., 3] > 0
    if v.sum() < 100: continue
    rgb = t[..., :3][v].astype(float) / 255
    cs.append(dict(site=p.site, R=rgb[:, 0].mean(), G=rgb[:, 1].mean(), B=rgb[:, 2].mean(), bright=rgb.mean(), contrast=rgb.mean(1).std()))
cs = pd.DataFrame(cs)
out += ["## 6. Domain shift: colour of the reference-date centre crop (64 m)", md(cs.groupby("site").mean().round(3)), ""]
emb = np.load(CACHE / "emb" / "dinov2-l_B1" / "center.npy", mmap_mode="r"); idx = pd.read_parquet(CACHE / "emb" / "dinov2-l_B1" / "index.parquet")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X = np.asarray(emb[:, 0, :1024], np.float32); ysite = idx.site.to_numpy()
acc = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), X[::3], ysite[::3], cv=5).mean()
out += [f"A linear probe recognises the **city** from a single centre-pooled DINOv2 embedding with **{acc * 100:.1f} %** accuracy "
        f"(chance {100 / 5:.0f} %): the feature space is strongly city-specific, which is the core M_out difficulty.", ""]

# ---- 7. errors of the current submission (nested equal-weight ensemble) ----
a = json.load(open(REPORTS / "audit_ens5.json"))
row = [r for r in a["ensembles"] if r["name"] == "a_equal_nested_tb"][0]
mem = a["members"]
Pl = np.mean([np.load(CACHE / "oof" / f"{m}.npz")["oof_loso"].mean(0) for m in mem], 0)
e = json.load(open(REPORTS / "ens_eq.json")); pred = decide(Pl, e["t"], np.array(e["b"]))
y = pts.y.to_numpy()
err = pd.DataFrame(dict(site=pts.site, cls=pts["class"], ok=pred == y, conflict=pts.conflict, ba=pts.best_accuracy,
                        uncertain=pts.class_uncertain, n_valid=D.n_valid, fallback=D.fallback, spans=D.spans))
err["ba_bin"] = pd.cut(err.ba, [0, 60, 70, 80, 100])
out += ["## 7. Where the current ensemble fails (LOSO OOF, deployed t/b)",
        md(err.groupby(["site", "cls"]).ok.mean().unstack().reindex(columns=CLASSES).round(2)), "",
        "Accuracy by annotation quality (best_accuracy bin) / conflict / uncertain flag:",
        md(err.groupby("ba_bin", observed=True).ok.agg(["mean", "size"]).round(3)),
        md(err.groupby("conflict").ok.agg(["mean", "size"]).round(3)),
        md(err.groupby("uncertain").ok.agg(["mean", "size"]).round(3)), "",
        "Accuracy by number of valid dates and fallback use:",
        md(err.groupby(pd.cut(err.n_valid, [0, 3, 5, 7, 10])).ok.agg(["mean", "size"]).round(3)),
        md(err.groupby("spans", dropna=False).ok.agg(["mean", "size"]).round(3)), "",
        f"Nested per-site macro-F1: {row['per_site_out']}", ""]
(REPORTS / "eda.md").write_text("\n".join(out))

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
(pd.crosstab(pts.site, pts["class"], normalize="index")[CLASSES] * 100).plot.bar(stacked=True, ax=ax[0], colormap="tab10", rot=0)
ax[0].set_ylabel("% of site"); ax[0].set_title("class mix per site"); ax[0].legend(fontsize=7)
yc.plot.bar(ax=ax[1], rot=0); ax[1].set_title("change year (changed classes)")
for s in SITES:
    ax[2].hist(pts[pts.site == s].best_accuracy, bins=np.arange(45, 100, 5), histtype="step", label=s, lw=1.5)
ax[2].set_title("annotation best_accuracy"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(figs / "eda_labels.png", dpi=100); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
badtab.T.plot(ax=ax[0], marker="o", legend=False); ax[0].set_title("fraction of points flagged bad per date (lines=dates)")
for s in SITES:
    c = cs[cs.site == s]; ax[1].scatter(c.bright, c.contrast, s=6, label=s, alpha=0.6)
ax[1].set_xlabel("brightness"); ax[1].set_ylabel("contrast"); ax[1].set_title("centre-crop colour per city"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(figs / "eda_imagery.png", dpi=100); plt.close()
print("\n".join(out))
