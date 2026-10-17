"""Model D experiments. usage: exp_D.py <name> '<json cfg>' [seeds]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from modelD import SeqStore, ModelD
from cv import run_experiment
name = sys.argv[1]
cfg = dict(emb="dinov2-l_B1", loss="focal", gamma=1.5, smooth=0.05, lr=3e-4, wd=0.05, epochs=200, bs=32,
           label_w=True, balance=True, d=256, layers=2, drop=0.1)
cfg.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
seed_all(0)
pts = load_points()
S = SeqStore(CACHE / "emb" / cfg["emb"], pts, [CACHE / "emb" / x for x in cfg.get("extra_emb", [])])
run_experiment(name, cfg, lambda: ModelD(pts, S, cfg), pts, seeds=seeds, notes=cfg.get("notes", ""))
