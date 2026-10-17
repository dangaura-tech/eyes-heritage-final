"""Extract frozen backbone tokens for every point x date x 8 dihedral views (training points).
usage: p4_embed.py <backbone> <config> [tiles_variant]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from embed import extract_to_dir, center_pool, dino_pair_feats
import numpy as np, pandas as pd

bb, cfgn = sys.argv[1], sys.argv[2]
variant = sys.argv[3] if len(sys.argv) > 3 else "tiles"
name = f"{bb}_{cfgn}{'' if variant == 'tiles' else '_' + variant}"
out = CACHE / "emb" / name
seed_all(0)
if not (out / "done").exists():
    pts = load_points()
    index = []
    for _, p in pts.iterrows():
        z = np.load(tile_path(p.site, p.point_id), allow_pickle=True)
        index += [(p.point_id, p.site, t) for t in range(len(z["dates"]))]
    extract_to_dir(pd.DataFrame(index, columns=["point_id", "site", "t"]), out, bb, cfgn, variant,
                   log=lambda m: print(m, flush=True))
if not (out / "center.npy").exists():
    center_pool(out)
if not (CACHE / f"dinopair_{name}.parquet").exists():
    dino_pair_feats(out).to_parquet(CACHE / f"dinopair_{name}.parquet")
print("done", name)
