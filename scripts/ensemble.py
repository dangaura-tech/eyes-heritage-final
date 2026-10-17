"""P6: ensemble OOF predictions, tune decision rule on LOSO OOF, robustness curve, bootstrap CI.
usage: ensemble.py <run1,run2,...> [--out name]"""
import sys, itertools, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from cv import evaluate, bootstrap_S
from pairs import tune_decision, decide
import numpy as np

runs = sys.argv[1].split(",")
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "ens"
pts = load_points(); y = pts.y.to_numpy()
O = {r: np.load(CACHE / "oof" / f"{r}.npz") for r in runs}
OI = {r: O[r]["oof_in"].mean(0) for r in runs}; OL = {r: O[r]["oof_loso"].mean(0) for r in runs}
for r in runs:
    e = evaluate(OI[r], OL[r], pts)
    print(f"{r:28s} seed-avg  S={e['S']:.4f} M_in={e['M_in']:.4f} M_out={e['M_out']:.4f}")

step = 0.1 if len(runs) <= 4 else 0.2
if "--weights" in sys.argv:          # fixed weights, e.g. "eq" or "0.25,0.25,0.25,0.25" (no weight fitting)
    ws = sys.argv[sys.argv.index("--weights") + 1]
    grid = [tuple([1 / len(runs)] * len(runs)) if ws == "eq" else tuple(float(x) for x in ws.split(","))]
else:
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(runs)) if abs(sum(w) - 1) < 1e-6]
best = None
for w in grid:
    Pi = sum(wi * OI[r] for wi, r in zip(w, runs)); Pl = sum(wi * OL[r] for wi, r in zip(w, runs))
    e = evaluate(Pi, Pl, pts)
    if best is None or e["S"] > best[1]["S"]:
        best = (w, e)
w, e = best
print("best weights", dict(zip(runs, np.round(w, 2))), f"S={e['S']:.4f} M_in={e['M_in']:.4f} M_out={e['M_out']:.4f}")
Pi = sum(wi * OI[r] for wi, r in zip(w, runs)); Pl = sum(wi * OL[r] for wi, r in zip(w, runs))
t0, b0 = e["t"], np.array(e["b"])
# robustness: S vs t in +-0.1 around the tuned value, choose centre of the flat region
ts = np.round(np.arange(max(0.05, t0 - 0.1), min(0.99, t0 + 0.1) + 1e-9, 0.02), 3)
curve = []
for t in ts:
    r = evaluate(Pi, Pl, pts, t=t, b=b0); curve.append((float(t), hm(r["M_in"], r["loso"]["macroF1"]), r["M_in"], r["loso"]["macroF1"]))
Ss = np.array([c[1] for c in curve])
# flat-region choice: maximise the 3-point moving average
sm = np.convolve(Ss, np.ones(3) / 3, mode="same"); sm[0] = sm[-1] = -1
t_sel = float(ts[int(np.argmax(sm))])
final = evaluate(Pi, Pl, pts, t=t_sel, b=b0)
mS, ci = bootstrap_S(Pi, Pl, pts, t_sel, b0, n=1000)
res = dict(runs=runs, weights=list(map(float, w)), t=t_sel, b=b0.tolist(), S=final["S"], M_in=final["M_in"],
           M_out=final["M_out"], per_site=final["per_site"], perclass_in=final["in"]["perclass"],
           perclass_loso=final["loso_nested"]["perclass"], cm_in=final["in"]["cm"], cm_loso=final["loso"]["cm"],
           boot_mean=mS, boot_ci=ci, t_curve=curve, auroc_loso=final["loso"]["auroc"], auroc_in=final["in"]["auroc"])
json.dump(res, open(REPORTS / f"{out}.json", "w"), indent=1)
np.savez(CACHE / "oof" / f"{out}.npz", oof_in=Pi[None], oof_loso=Pl[None])
print(json.dumps({k: v for k, v in res.items() if k not in ("t_curve", "cm_in", "cm_loso")}, indent=1))
print("t curve:", [(c[0], round(c[1], 4)) for c in curve])
