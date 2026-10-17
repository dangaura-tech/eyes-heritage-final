"""Model T (TabPFN v2 + GBDT ensemble on point features + DINOv3 PCA). usage: exp_T.py <name> '<json cfg>' [seeds]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelA import PairTable
from modelT import PointFeats, ModelT
from cv import run_experiment
import pandas as pd
name = sys.argv[1]
cfg = dict(dino="dinov3s-l_B1", emb="dinov3s-l_B1", pca=16, tabpfn=True, gbdt=True, w_pfn=0.5, pfn_estimators=8,
           label_w=True, balance=True)
cfg.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
seed_all(0)
pts = load_points()
T = PairTable(pts, pd.read_parquet(CACHE / "featA_tiles_unary.parquet"), pd.read_parquet(CACHE / "featA_tiles_pairs.parquet"),
              pd.read_parquet(CACHE / f"dinopair_{cfg['dino']}.parquet"))
F = PointFeats(T, (CACHE / "emb" / cfg["emb"]).resolve())
print("point features", F.X.shape, "embedding diffs", None if F.E is None else F.E.shape, flush=True)
run_experiment(name, cfg, lambda: ModelT(F, cfg), pts, seeds=seeds, notes=cfg.get("notes", ""))
