"""Merge OOFs + logged results of runs that differ only in seeds. usage: merge_runs.py <new_name> <run1,run2,...>"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from cv import log_result
import numpy as np
new, names = sys.argv[1], sys.argv[2].split(",")
runs = {}
for line in open(REPORTS / "runs.jsonl"):
    r = json.loads(line); runs[r["name"]] = r
Z = [np.load(CACHE / "oof" / f"{n}.npz") for n in names]
np.savez(CACHE / "oof" / f"{new}.npz", oof_in=np.concatenate([z["oof_in"] for z in Z]), oof_loso=np.concatenate([z["oof_loso"] for z in Z]))
res = sum([runs[n]["results"] for n in names], []); seeds = sum([runs[n]["seeds"] for n in names], [])
log_result(new, runs[names[0]]["cfg"], seeds, res, notes=f"merged {names}")
