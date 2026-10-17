"""P7: retrain every ensemble member on all 776 points (5 seeds, fixed budgets from CV) and write
models/final/ensemble.json. usage: train_final.py reports/<ens>.json [--seeds 0,1,2,3,4]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from final import train_member
import numpy as np

ens = json.load(open(sys.argv[1]))
seeds = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [0, 1, 2, 3, 4]
hold = sys.argv[sys.argv.index("--holdout") + 1] if "--holdout" in sys.argv else None   # LOSO sanity check
pts_all = load_points()
train_idx = np.where(pts_all.site != hold)[0] if hold else None
runs = {}
for line in open(REPORTS / "runs.jsonl"):
    r = json.loads(line); runs[r["name"]] = r          # last entry with a given name wins


def kind_of(name):
    """Member kind from the run name: Apt*, A*, B*, C* -> as is; D*/E* (temporal) -> D; F*/G* (U-Net / AdaptFormer) -> C."""
    if name.startswith("Apt"): return "Apt"
    return {"A": "A", "B": "B", "C": "C", "D": "D", "E": "D", "F": "C", "G": "C"}[name[0]]


def budget(r, kind):
    infos = []
    for res in r["results"]:
        fi = res.get("fit_info", {})
        for proto in ("inn", "loso"):
            infos += list(fi.get(proto, {}).values())
    if kind == "A":
        out = {}
        for k in ("lgb", "cat"):
            v = [i[f"{k}_iter"] for i in infos if f"{k}_iter" in i]
            if v: out[k] = int(np.median(v))
        return out
    if kind == "Apt":
        return int(np.median([i["iter"] for i in infos]))
    v = [i["best_ep"] for i in infos if i.get("best_ep")]
    return int(np.median(v)) if v else int(r["cfg"].get("epochs", 150))


out_root = MODELS / (f"loso_{hold}" if hold else "final"); out_root.mkdir(parents=True, exist_ok=True)
weights = {}
for name, w in zip(ens["runs"], ens["weights"]):
    if w <= 0:
        continue
    r = runs[name]; kind = kind_of(name)
    spec = dict(name=name, kind=kind, cfg=r["cfg"], budget=budget(r, kind), cv_hash=r["hash"])
    print("training", spec["name"], kind, "budget", spec["budget"], flush=True)
    train_member(spec, seeds, out_root, train_idx)
    weights[name] = float(w)
json.dump(dict(weights=weights, t=ens["t"], b=ens["b"], S_proxy=ens["S"], M_in=ens["M_in"], M_out=ens["M_out"],
               boot_ci=ens.get("boot_ci"), seeds=seeds, source=sys.argv[1]), open(out_root / "ensemble.json", "w"), indent=1)
print("final models written to", out_root)
