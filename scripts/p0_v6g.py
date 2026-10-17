"""v6 explicit, time-gap-balanced labelled pairs on the reviewed v3 labels (run with HERITAGE_ROOT=<root>).
usage: p0_v6g.py [--scene]   --scene also writes the scene-level exclusions (same rule as v5)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
from common import *
import numpy as np, pandas as pd

if "--scene" in sys.argv:
    q = pd.read_parquet(Path(__import__("os").environ.get("BASE_CACHE", CACHE)) / "quality.parquet")
    g = q.groupby(["site", "t"]).agg(bad=("bad", "mean"), contrast=("contrast", "median"), lap=("lap_var", "median"),
                                     haze=("haze", "median"))
    for c in ("contrast", "lap", "haze"):
        g[c + "_rel"] = g[c] / g.groupby(level=0)[c].transform("median")
    drop = g[(g.bad >= 0.8) | (g.contrast_rel < 0.55) | ((g.haze_rel >= 1.5) & (g.lap_rel < 0.55))]
    scene = {}
    for s, t in drop.index:
        scene.setdefault(s, []).append(int(t))
    json.dump(scene, open(CACHE / "scene_bad.json", "w"), indent=1)
    common.SCENE_BAD = {k: set(v) for k, v in scene.items()}
    print("scene exclusions:", scene)

from pairs import valid_dates, label_pairs, all_pairs
pts = load_points(); rows = []
for _, r in pts.iterrows():
    z = np.load(tile_path(r.site, r.point_id), allow_pickle=True)
    years = np.array([int(d[:4]) for d in z["dates"]]); fy = z["fyear"]
    vi, _ = valid_dates(point_bad(r.site, z), list(z["quality"])); vi = list(vi)
    adj = {(int(vi[a]), int(vi[a + 1])) for a in range(len(vi) - 1)}
    if r.y == NC:
        lp = [(i, j, NC, 1.0) for i, j in all_pairs(vi)]
    else:
        yb = r.year_before if pd.notna(r.year_before) else r.year - 1
        lp = [(i, j, l, w) for i, j, l, w, _ in label_pairs(r.y, r.year, years, vi, year_before=yb)]
        lp += [(i, j, NC, 0.5) for i, j in all_pairs(vi) if years[j] <= yb]
    tot = sum(w for *_, w in lp) or 1.0
    for i, j, l, w in lp:
        rows.append(dict(point_id=r.point_id, i=int(i), j=int(j), label=int(l), weight=w, pmass=w / tot,
                         adjacent=int((i, j) in adj), gap_years=float(fy[j] - fy[i])))
P = pd.DataFrame(rows)
P["gap_bucket"] = np.clip(np.ceil(P.gap_years), 1, 4).astype(int)
mass = P.groupby(["label", "gap_bucket"]).pmass.sum()
pb = P.groupby("gap_bucket").pmass.sum() / P.pmass.sum()
pbc = mass / mass.groupby(level=0).transform("sum")
mult = (pb.reindex(pbc.index.get_level_values(1)).to_numpy() / pbc.to_numpy()).clip(1 / 3, 3)
M = dict(zip(pbc.index, mult))
P["sample_weight"] = [w * M[(l, b)] for w, l, b in zip(P.weight, P.label, P.gap_bucket)]
P.to_parquet(CACHE / "pairs_explicit.parquet")
print(len(P), "pairs |", P.label.map(dict(enumerate(CLASSES))).value_counts().to_dict())
