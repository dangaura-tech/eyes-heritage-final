"""Builds notebooks/heritage_submission_0642.ipynb (pure json, no nbformat dependency)."""
import json
from pathlib import Path

cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))

md(r"""
# Eyes on Heritage: submission pipeline (nested S_proxy = 0.642)

The notebook runs the submitted pipeline end to end:

| step | what | where |
|---|---|---|
| 1 | data audit + asserts, 100 m cell lattice from the black squares | `scripts/p0_audit.py` |
| 2 | per point × date quality flags, local co-registration, 384 px tiles | `scripts/p1_tiles.py`, `src/imgproc.py` |
| 3 | engineered pair features (Model A) + frozen DINOv2-L tokens, 8 dihedral views | `scripts/p3_featA.py`, `scripts/p4_embed.py` |
| 4 | validation protocol: spatial 5-fold (200 m blocks) + leave-one-site-out, leak assertions | `src/cv.py` |
| 5 | four members: **A** LightGBM per pair, **A-pt** LightGBM per point, **B** frozen-token head, **C** DINOv2-L LoRA | `src/model*.py` |
| 6 | **equal-weight** ensemble, decision rule (t, b) **fitted nested**, cell bootstrap CI | this notebook |
| 7 | deployment: t/b on pooled LOSO OOF, monotone Platt calibration of p_change | `models/final/ensemble.json` |
| 8 | final training on all 776 points (5 seeds each) and `predict.py` | `src/final.py`, `predict.py` |

**Modes**
* `MODE = "load"` reuses the cached out-of-fold predictions (`cache/oof/*.npz`) and reproduces the reported numbers in a few minutes.
* `MODE = "reproduce"` re-runs every step and every cross-validation (hours on one RTX 5000 Ada; Model C alone is about 3 h per seed).

Run on a machine with a CUDA GPU. No annotation metadata (year, teams, ...) is used as an input; metadata is used only for sample weights.
""")

code(r"""
MODE = "load"            # "load" | "reproduce"
TRAIN_FINAL = False      # retrain the 4 final members on all 776 points (5 seeds) -> models/final/
import os, sys, json, time
from pathlib import Path
ROOT = Path(os.environ.get("HERITAGE_ROOT", Path.home() / "heritage"))
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
import numpy as np, pandas as pd
import common
from common import *
seed_all(0)
print("root", ROOT, "| data", DATA_DIR)
try:
    import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except ImportError:
    print("torch not available (fine for MODE='load')")
""")

md(r"""
## 1. Data audit
Unzips the imagery, asserts the facts of the brief (776 points, class/site counts, 4-band uint8 RGBA at 0.5 m UTM, one grid per site) and derives the 100 m cell lattice from the withheld black squares (alpha = 0 holes, aligned to a 200 px lattice). Output: `cache/points.parquet` (point_id, cell_id, 200 m block_id) and `reports/data_audit.md`.
""")
code(r"""
if MODE == "reproduce" or not (CACHE / "points.parquet").exists():
    %run scripts/p0_audit.py
pts = load_points()
print(len(pts), "points |", pts.cell_id.nunique(), "cells |", pts.block_id.nunique(), "200 m blocks")
pd.crosstab(pts.site, pts["class"])[CLASSES]
""")

md(r"""
## 2. Quality flags, co-registration, tile cache
For every point and date, a 128 m window is scored for sharpness (Laplacian variance), RMS contrast, valid alpha and haze. A date is flagged **bad** if it is hazy or has < 70 % valid pixels in the core window. Each date is registered to the sharpest good date by Hanning-windowed phase correlation on gradient magnitude. The shift is applied only if it is ≤ 12 px and the peak response is reliable. A 384 × 384 px (192 m) crop with alpha is cached per point and date. Alpha = 0 pixels are never used or inpainted.
""")
code(r"""
if MODE == "reproduce" or not (CACHE / "quality.parquet").exists():
    %run scripts/p1_tiles.py
q = pd.read_parquet(CACHE / "quality.parquet")
q.groupby("site").agg(dates=("t", "nunique"), bad_frac=("bad", "mean")).round(3)
""")

md(r"""
## 3. Features
* **Model A features** on 16 / 32 / 64 m windows, before, after and their difference: Lab stats, change-vector magnitude, SSIM, ExG and green fraction, shadow fraction, Canny density, gradient-orientation entropy, LBP χ², bare-soil and paved fractions, histogram-matched colour differences, shift magnitude, time gap.
* **DINOv2-L tokens**: 128 m crop → 448 px → 32×32 tokens, pooled to 16×16 (8 m). Last layer and layer 16 are concatenated. Each crop is standardised per channel. 8 dihedral views are mapped back to the canonical frame, so before/after views can be mixed without breaking spatial correspondence. Cosine and L2 distances between dates (CLS and centre-pooled) become extra Model A features.
""")
code(r"""
if MODE == "reproduce" or not (CACHE / "featA_tiles_pairs.parquet").exists():
    %run scripts/p3_featA.py
if MODE == "reproduce" or not (CACHE / "dinopair_dinov2-l_B1.parquet").exists():
    %run scripts/p4_embed.py dinov2-l B1        # GPU, ~10 min; token cache ~50 GB (put cache/emb on a fast local disk)
print(pd.read_parquet(CACHE / "featA_tiles_pairs.parquet").shape, "pair rows")
""")

md(r"""
## 4. Validation protocol
* **In-domain proxy (M_in)**: `StratifiedGroupKFold(5)` with groups = 200 m blocks (2 × 2 cells), stratified by site × class.
* **Out-of-domain proxy (M_out)**: leave-one-site-out.
* **Early stopping / checkpoint selection** uses an inner split of the *training* part only: one other site for LOSO, 20 % of the training blocks in-domain. `cv.assert_split` asserts disjointness for every fold.
""")
code(r"""
from pairs import make_folds
from cv import inner_split, assert_split
folds = make_folds(pts, 0)
for proto in ("in", "loso"):
    for k in np.unique(folds[proto]):
        te = np.where(folds[proto] == k)[0]; tr = np.where(folds[proto] != k)[0]
        tr_in, va = inner_split(pts, tr, proto, int(k))
        print(proto, k, assert_split(pts, te, tr, tr_in, va, proto), "| test", len(te))
""")

md(r"""
## 5. Ensemble members (exact configurations of the submission)
* **A**: LightGBM on pair features + DINO distances. Trained on all spanning pairs, with label-quality × (site × class) weights. Prediction scores every valid pair and averages the top-2 most-changed pairs (year-free).
* **A-pt**: point-level LightGBM on max/mean/std of the pair features over all pairs, first-last pair and temporal std. Hyper-parameters come from a 30-trial Optuna search on LOSO S.
* **B**: frozen DINOv2-L tokens, then a 2-layer transformer head over [f_b, f_a, |f_a−f_b|, cos], with Gaussian centre attention pooling and focal loss. Trained without time reversal.
* **C**: siamese DINOv2-L with LoRA r=16 on q/k/v/o of the last 8 blocks and the same head. Image-space augmentation: dihedral, ±10 px misregistration, colour/gamma/haze/JPEG/sensor gains, scale, periphery cutout. 3 CV seeds.
""")
code(r"""
MEMBERS = {
 "A_lgb_top2_dino": dict(kind="A", seeds=[0, 1, 2], cfg={"algo": "lgb", "agg": "top2", "k": 2, "reverse": False, "ambiguous": "exclude",
                    "label_w": True, "balance": True, "variant": "tiles", "dino": "dinov2-l_B1", "threads": 4}),
 "Apt_opt": dict(kind="Apt", seeds=[0, 1, 2], cfg={"algo": "lgb", "agg": "top2", "k": 2, "reverse": False, "ambiguous": "exclude",
                    "label_w": True, "balance": True, "variant": "tiles", "dino": "dinov2-l_B1", "threads": 4, "point": True,
                    "lr": 0.0414605918232234, "leaves": 21, "min_leaf": 78, "ff": 0.7634362510345057, "l2": 6.9216962296422375,
                    "l1": 0.09339014290782491, "depth": 5}),
 "B_l_B1_norev": dict(kind="B", seeds=[0, 1, 2], cfg={"emb": "dinov2-l_B1", "loss": "focal", "gamma": 1.5, "smooth": 0.05, "lr": 3e-4,
                    "wd": 0.05, "epochs": 150, "bs": 64, "reverse": False, "ambiguous": "exclude", "label_w": True, "balance": True,
                    "tokdrop": 0.1, "mixup": 0.2, "agg": "top2", "k": 2, "tta": 8, "use_mid": True, "workers": 5}),
 "C_l_B2_lora3": dict(kind="C", seeds=[0, 1, 2], cfg={"loss": "focal", "gamma": 1.5, "smooth": 0.05, "lr": 3e-4, "lr_lora": 1e-4,
                    "epochs": 40, "bs": 32, "rank": 16, "last_k": 8, "reverse": True, "label_w": True, "balance": True,
                    "agg": "top2", "tta": 8, "workers": 5}),
}


def make_factory(name):
    kind, cfg = MEMBERS[name]["kind"], MEMBERS[name]["cfg"]
    if kind in ("A", "Apt"):
        from modelA import PairTable, ModelA
        from modelApt import PointTable, ModelApt
        T = PairTable(pts, pd.read_parquet(CACHE / "featA_tiles_unary.parquet"), pd.read_parquet(CACHE / "featA_tiles_pairs.parquet"),
                      pd.read_parquet(CACHE / f"dinopair_{cfg['dino']}.parquet"))
        if kind == "A":
            return lambda: ModelA(T, cfg)
        PT = PointTable(T); return lambda: ModelApt(PT, cfg)
    if kind == "B":
        from modelB import EmbStore, ModelB
        S = EmbStore((CACHE / "emb" / cfg["emb"]).resolve()); return lambda: ModelB(pts, S, cfg)
    from modelC import ModelC
    M = ModelC(pts, cfg)
    def f():
        m = ModelC.__new__(ModelC); m.__dict__.update(pts=M.pts, cfg=cfg, dev="cuda", meta=M.meta, tiles=M.tiles); return m
    return f


from cv import run_experiment
OOF = {}
for name, spec in MEMBERS.items():
    fn = CACHE / "oof" / f"{name}.npz"
    if MODE == "reproduce" or not fn.exists():
        print("running CV for", name, flush=True)
        run_experiment(name if name != "C_l_B2_lora3" else "C_l_B2_lora3", spec["cfg"], make_factory(name), pts, seeds=spec["seeds"])
    z = np.load(fn)
    OOF[name] = (z["oof_in"].mean(0), z["oof_loso"].mean(0))       # 3-seed averaged probabilities
    print(f"{name:18s} seeds={z['oof_in'].shape[0]}  OOF loaded")
""")

md(r"""
## 6. Ensemble and decision rule: fully nested evaluation
* Ensemble = **plain average** of the four members' 3-seed-averaged probabilities. No weights are fitted, because in the audit fitted weights scored lower out of sample.
* Decision rule: `p_change = 1 − p(no_change)`. If `p_change ≥ t`, predict the argmax of the three change classes after adding log-biases `b_c`, otherwise `no_change`.
* **Nested**: for **M_out**, t and b are fitted on the LOSO OOF of 4 sites and applied to the 5th. For **M_in**, they are fitted on the LOSO OOF of the points outside each in-domain test fold. Thresholds are never tuned on in-domain predictions.
""")
code(r"""
from pairs import tune_decision, decide

def fast_f1(y, p, per_class=False):
    cm = np.bincount(np.asarray(y) * 4 + np.asarray(p), minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); d = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    return f if per_class else f.mean()
common.macro_f1 = fast_f1            # identical to sklearn macro-F1 (labels 0..3), only faster inside the tuner

names = list(MEMBERS)
P_in = np.mean([OOF[n][0] for n in names], 0); P_lo = np.mean([OOF[n][1] for n in names], 0)
y = pts.y.to_numpy(); site = pts.site.to_numpy(); fin = folds["in"]

pred_out = np.zeros(len(y), int); pred_in = np.zeros(len(y), int)
for s in SITES:                                   # M_out
    te = site == s; t, b, _ = tune_decision(P_lo[~te], y[~te]); pred_out[te] = decide(P_lo[te], t, b)
for k in np.unique(fin):                          # M_in
    te = fin == k; t, b, _ = tune_decision(P_lo[~te], y[~te]); pred_in[te] = decide(P_in[te], t, b)
M_in, M_out = fast_f1(y, pred_in), fast_f1(y, pred_out)
print(f"M_in_cv {M_in:.4f} | M_out_cv {M_out:.4f} | S_proxy = HM = {hm(M_in, M_out):.4f}   (reported: 0.7202 / 0.5793 / 0.6421)")
""")
code(r"""
# cell-level paired bootstrap of S (decisions fixed)
cells = pts.cell_id.to_numpy(); uc = np.unique(cells); ix = {c: np.where(cells == c)[0] for c in uc}
rng = np.random.RandomState(0); B = []
for _ in range(1000):
    sel = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])
    B.append(hm(fast_f1(y[sel], pred_in[sel]), fast_f1(y[sel], pred_out[sel])))
print("S 95% CI:", np.percentile(B, [2.5, 97.5]).round(3))
res = pd.DataFrame({"F1 in-domain": fast_f1(y, pred_in, True), "F1 LOSO": fast_f1(y, pred_out, True)}, index=CLASSES).round(3)
display(res)
display(pd.Series({s: fast_f1(y[site == s], pred_out[site == s]) for s in SITES}, name="LOSO macro-F1 per site").round(3))
""")
code(r"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (pr, ttl) in zip(ax, [(pred_in, "in-domain (nested t/b)"), (pred_out, "leave-one-site-out (nested t/b)")]):
    cm = confusion_matrix(y, pr, labels=range(4)); a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, cm[i, j], ha="center", va="center")
    a.set_xticks(range(4)); a.set_xticklabels(["constr", "destr", "soil", "no_ch"]); a.set_yticks(range(4))
    a.set_yticklabels(["constr", "destr", "soil", "no_ch"]); a.set_title(ttl); a.set_xlabel("predicted"); a.set_ylabel("true")
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Deployment decision layer and probability calibration
For deployment, t and b are fitted once on the pooled LOSO OOF of all 5 sites. t is taken from the flat region of the S-vs-t curve rather than the exact peak. `p_change` is then calibrated **monotonically** (Platt on logit p_change). This leaves ranking and decisions unchanged. Nested LOSO: ECE 0.249 → 0.036, Brier 0.251 → 0.171.
""")
code(r"""
from cv import evaluate
t0, b0, _ = tune_decision(P_lo, y)
ts = np.round(np.arange(max(0.05, t0 - 0.1), min(0.99, t0 + 0.1) + 1e-9, 0.02), 3)
curve = [hm(fast_f1(y, decide(P_in, t, b0)), fast_f1(y, decide(P_lo, t, b0))) for t in ts]
sm = np.convolve(curve, np.ones(3) / 3, mode="same"); sm[0] = sm[-1] = -1
t_dep = float(ts[int(np.argmax(sm))])
print("deployed t =", t_dep, "| b =", np.round(b0, 2), "| S(t) around t:", dict(zip(ts.round(2), np.round(curve, 3))))

from scipy.optimize import minimize
pc = np.clip(1 - P_lo[:, NC], 1e-6, 1 - 1e-6); z = np.log(pc / (1 - pc)); yb = (y != NC).astype(float)
nll = lambda th: -np.mean(yb * np.log(1e-9 + 1 / (1 + np.exp(-(th[0] * z + th[1])))) + (1 - yb) * np.log(1e-9 + 1 - 1 / (1 + np.exp(-(th[0] * z + th[1])))))
platt = minimize(nll, [1.0, 0.0], method="L-BFGS-B").x
print("Platt (a, c) for p_change:", platt.round(4))
ensemble_cfg = dict(weights={n: 0.25 for n in names}, t=t_dep, b=list(map(float, b0)), pc_platt=list(map(float, platt)),
                    S_proxy_nested=float(hm(M_in, M_out)), M_in_nested=float(M_in), M_out_nested=float(M_out))
ensemble_cfg
""")

md(r"""
## 8. Final training (all 776 points) and prediction
Each member is retrained on all points with 5 seeds. The budget is the median CV-selected epoch or iteration: A 63 rounds, A-pt 93 rounds, B 40 epochs, C 25 epochs. Deep models use the same cosine schedule length as in CV and stop at that epoch.
""")
code(r"""
if TRAIN_FINAL:
    from final import train_member
    BUDGET = {"A_lgb_top2_dino": {"lgb": 63}, "Apt_opt": 93, "B_l_B1_norev": 40, "C_l_B2_lora3": 25}
    for n, spec in MEMBERS.items():
        train_member(dict(name=n, kind=spec["kind"], cfg=spec["cfg"], budget=BUDGET[n]), [0, 1, 2, 3, 4], MODELS / "final")
    json.dump(dict(ensemble_cfg, seeds=[0, 1, 2, 3, 4]), open(MODELS / "final" / "ensemble.json", "w"), indent=1)
print(sorted(p.name for p in (MODELS / "final").iterdir()))
""")
md(r"""
Prediction on a new site. The UTM zone is read from the GeoTIFF CRS, and sites with any number of dates work. Points whose dates are all bad fall back to the least-bad pair. Output columns are configured in the `OUTPUT` block of `predict.py`.
```bash
python predict.py --points test_points.geojson --imagery /path/to/damascus_imagery --out submission.csv
```
""")
code(r"""
# sanity check on one training site (scores must be high but not 100 %): uncomment to run (~20 min on a shared GPU)
# !python predict.py --points work/timbuktu_pts.geojson --imagery data/raw/timbuktu --site timbuktu --out work/pred_timbuktu.csv --work work/timbuktu
""")

md(r"""
## 9. Audit summary
| check | result |
|---|---|
| leak assertions (120 member × seed × fold splits) | all disjoint; early stopping never on the held-out site |
| label-shuffle test (all members retrained on permuted labels) | ensemble S 0.245 (chance ≈ 0.25) |
| fitted weights vs equal weights (nested) | 0.624 vs **0.642** → equal weights |
| fitting weights on all LOSO OOF (earlier claim) | 0.648, optimistic by ~0.03 (same +0.03 appears on shuffled labels) |
| not nested | member and hyper-parameter selection (~40 configurations on the same LOSO OOF) → treat 0.642 as mildly optimistic |
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
out = Path(__file__).with_name("heritage_submission_0642.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
