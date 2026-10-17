"""Compute Model A unary (point x date) and pairwise (point x i<j) features for all cached tiles."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from featA import unary, pairwise
import numpy as np, pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

VARIANT = os.environ.get("TILE_VARIANT", "tiles")   # 'tiles' (co-registered) or 'raw' (ablation)


def featurize_npz(args):
    pid, fn, variant = args
    z = np.load(fn, allow_pickle=True)
    T = z[variant]; sh = z["shifts"]; fy = z["fyear"]
    U, Hs = [], []
    for t in range(len(T)):
        f, h = unary(T[t]); f.update(point_id=pid, t=t); U.append(f); Hs.append(h)
    P = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            f = pairwise(T[i], T[j], Hs[i], Hs[j])
            f.update(point_id=pid, i=i, j=j, dt=float(fy[j] - fy[i]),
                     shift_i=float(np.hypot(*sh[i, :2])), shift_j=float(np.hypot(*sh[j, :2])),
                     shift_rel=float(np.hypot(*(sh[j, :2] - sh[i, :2]))),
                     resp_min=float(min(abs(sh[i, 2]) if sh[i, 2] != 0 else 1, abs(sh[j, 2]) if sh[j, 2] != 0 else 1)))
            P.append(f)
    return U, P


def run(pts_ids_files, out_prefix, variant=VARIANT, procs=None):
    procs = procs or int(os.environ.get("FEAT_PROCS", max(1, (os.cpu_count() or 4) - 1)))
    U, P = [], []
    with Pool(procs) as pool:
        for u, p in tqdm(pool.imap_unordered(featurize_npz, [(a, b, variant) for a, b in pts_ids_files], chunksize=2),
                         total=len(pts_ids_files)):
            U += u; P += p
    U = pd.DataFrame(U); P = pd.DataFrame(P)
    U.to_parquet(f"{out_prefix}_unary.parquet"); P.to_parquet(f"{out_prefix}_pairs.parquet")
    return U, P


if __name__ == "__main__":
    pts = load_points()
    items = [(p.point_id, tile_path(p.site, p.point_id)) for _, p in pts.iterrows()]
    out = CACHE / f"featA_{VARIANT}{os.environ.get('FEAT_SUFFIX', '')}"
    if not Path(f"{out}_pairs.parquet").exists():
        U, P = run(items, str(out))
        print(U.shape, P.shape)
