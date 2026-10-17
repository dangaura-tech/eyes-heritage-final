"""From a token cache, derive centre-pooled embeddings (per tile x view) and DINO pair distance features for Model A.
usage: p4b_center.py <emb_name>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np, pandas as pd, json

name = sys.argv[1]
d = CACHE / "emb" / name
meta = json.load(open(d / "meta.json")); G = meta["grid"]
tok = np.load(d / "tokens.npy", mmap_mode="r"); cls = np.load(d / "cls.npy", mmap_mode="r")
val = np.load(d / "valid.npy").astype(np.float32)
idx = pd.read_parquet(d / "index.parquet")
N = len(idx)
# centre radius in tokens: ~16 m for each config
tok_m = 128 / G if meta["config"] == "B1" else 64 / G
r = max(1, int(round(16 / tok_m)))
yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij"); c = (G - 1) / 2
cm = ((np.abs(yy - c) <= r) & (np.abs(xx - c) <= r)).flatten()
print("centre tokens", cm.sum(), "radius", r)
if not (d / "center.npy").exists():
    out = np.zeros((N, 8, tok.shape[-1]), np.float16)
    for s in range(0, N, 64):
        t = np.asarray(tok[s:s + 64][:, :, cm], np.float32)          # (b,8,n,2D)
        w = val[s:s + 64][:, cm][:, None, :, None] + 1e-3
        out[s:s + 64] = ((t * w).sum(2) / w.sum(2)).astype(np.float16)
    np.save(d / "center.npy", out)
cen = np.load(d / "center.npy").astype(np.float32)
D = cen.shape[-1] // 2
row = {(p, t): k for k, (p, t) in enumerate(zip(idx.point_id, idx.t))}


def cosd(a, b):
    return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-6)


rows = []
clsf = np.asarray(cls, np.float32).mean(1); cen_m = cen.mean(1)
for pid, g in idx.groupby("point_id"):
    ts = sorted(g.t)
    for a, i in enumerate(ts):
        for j in ts[a + 1:]:
            ki, kj = row[(pid, i)], row[(pid, j)]
            rows.append(dict(point_id=pid, i=i, j=j,
                             dino_cls_cos=float(cosd(clsf[ki], clsf[kj])),
                             dino_cen_cos=float(cosd(cen_m[ki, :D], cen_m[kj, :D])),
                             dino_cen_mid_cos=float(cosd(cen_m[ki, D:], cen_m[kj, D:])),
                             dino_cen_l2=float(np.linalg.norm(cen_m[ki, :D] - cen_m[kj, :D]) / np.sqrt(D))))
pd.DataFrame(rows).to_parquet(CACHE / f"dinopair_{name}.parquet")
print("pairs", len(rows))
