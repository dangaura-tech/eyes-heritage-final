"""Model B experiments. usage: exp_B.py <name> '<json cfg>' [seeds]   cfg must include emb (e.g. dinov2-l_B1)"""
import sys, json, os, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelB import EmbStore, ModelB
from cv import run_experiment

name = sys.argv[1]
cfg = dict(emb="dinov2-l_B1", loss="focal", gamma=1.5, smooth=0.05, lr=3e-4, wd=0.05, epochs=150, bs=64,
           reverse=True, ambiguous="exclude", label_w=True, balance=True, tokdrop=0.1, mixup=0.2, agg="max", k=2,
           tta=8, use_mid=True, workers=5)
cfg.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
seed_all(0)
src = CACHE / "emb" / cfg["emb"]
# cache/emb may be a symlink to a fast local disk (token caches are ~50 GB)
store = EmbStore(src.resolve(), site_norm=cfg.get("site_norm", False))
pts = load_points()
run_experiment(name, cfg, lambda: ModelB(pts, store, cfg), pts, seeds=seeds, notes=cfg.get("notes", ""))
