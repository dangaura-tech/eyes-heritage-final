"""Model A experiments. usage: exp_A.py <name> '<json cfg overrides>' [seeds]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelA import PairTable, ModelA
from cv import run_experiment
from modelApt import PointTable, ModelApt
import pandas as pd

name = sys.argv[1]
cfg = dict(algo="lgb", agg="max", k=2, reverse=False, ambiguous="exclude", label_w=True, balance=True,
           variant="tiles", dino=None, threads=4)
cfg.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
seed_all(0)
pts = load_points()
U = pd.read_parquet(CACHE / f"featA_{cfg['variant']}_unary.parquet")
P = pd.read_parquet(CACHE / f"featA_{cfg['variant']}_pairs.parquet")
extra = pd.read_parquet(CACHE / f"dinopair_{cfg['dino']}.parquet") if cfg.get("dino") else None
if cfg.get("sam"):
    sp = pd.read_parquet(CACHE / "sampair.parquet")
    extra = sp if extra is None else extra.merge(sp, on=["point_id", "i", "j"], how="left")
T = PairTable(pts, U, P, extra)
if cfg.get("site_norm"):
    T.enable_site_norm()
print("features", len(T.names), flush=True)
if cfg.get("point"):
    PT = PointTable(T)
    if cfg.get("site_norm") and not cfg.get("site_norm_pairs_only"):
        PT.site_normalise()
    print("point features", PT.X.shape, flush=True)
    run_experiment(name, cfg, lambda: ModelApt(PT, cfg), pts, seeds=seeds, notes=cfg.get("notes", ""))
else:
    run_experiment(name, cfg, lambda: ModelA(T, cfg), pts, seeds=seeds, notes=cfg.get("notes", ""))
