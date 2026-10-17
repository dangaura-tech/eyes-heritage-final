"""Model C (LoRA) experiments. usage: exp_C.py <name> '<json cfg>' [seeds]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelC import ModelC
from modelE import ModelE
from cv import run_experiment
name = sys.argv[1]
cfg = dict(loss="focal", gamma=1.5, smooth=0.05, lr=3e-4, lr_lora=1e-4, epochs=40, bs=32, rank=16, last_k=8,
           reverse=True, label_w=True, balance=True, agg="top2", tta=8, workers=5)
cfg.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
seed_all(0)
pts = load_points()
Cls = ModelE if cfg.get("arch") == "unet" else ModelC
M = Cls(pts, cfg)          # tiles loaded once, shared across folds
def make():
    m = Cls.__new__(Cls); m.__dict__.update(pts=M.pts, cfg=cfg, dev="cuda", meta=M.meta, tiles=M.tiles); return m
run_experiment(name, cfg, make, pts, seeds=seeds, notes=cfg.get("notes", ""))
