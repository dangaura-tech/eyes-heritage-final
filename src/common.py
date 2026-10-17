"""Shared paths, constants and helpers for the Eyes on Heritage pipeline."""
import os, json, random, hashlib, datetime
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("HERITAGE_ROOT", Path.home() / "heritage"))
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "data" / "train_dataset"))
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "cache"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for p in (RAW, CACHE, REPORTS, MODELS, CACHE / "tiles"):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = ["construction", "destruction", "soil_sealing", "no_change"]
C2I = {c: i for i, c in enumerate(CLASSES)}
NC = C2I["no_change"]
SITES = ["agadez", "aleppo", "derna", "marrakesh", "timbuktu"]

PX = 0.5            # metres / pixel
CELL_PX = 200       # 100 m cell
TILE = 384          # 192 m tile
EXPECTED_COUNTS = {"no_change": 406, "construction": 243, "soil_sealing": 64, "destruction": 63}
EXPECTED_SITE = {
    "agadez": (73, 11, 12, 90), "aleppo": (76, 7, 20, 39), "derna": (14, 12, 21, 84),
    "marrakesh": (34, 22, 3, 119), "timbuktu": (46, 11, 8, 74)}
EXPECTED_DATES = {"agadez": 9, "aleppo": 10, "derna": 5, "marrakesh": 8, "timbuktu": 8}


TILE_DIR = Path(os.environ["HERITAGE_TILE_DIR"]) if os.environ.get("HERITAGE_TILE_DIR") else CACHE / "tiles"


def tile_path(site, pid):
    return TILE_DIR / site / f"{pid}.npz"


def set_tile_dir(p):
    global TILE_DIR
    TILE_DIR = Path(p)


# v5: scene-level exclusions (whole dates of a site), loaded from cache/scene_bad.json if present
SCENE_BAD = {}
try:
    _sb = CACHE / "scene_bad.json"
    if _sb.exists():
        SCENE_BAD = {k: set(v) for k, v in json.load(open(_sb)).items()}
except Exception:
    SCENE_BAD = {}


def point_bad(site, z):
    """Per point x date bad flags, OR-ed with scene-level exclusions for that site."""
    bad = z["bad"].astype(bool).copy()
    for t in SCENE_BAD.get(site, ()):
        if t < len(bad): bad[t] = True
    return bad


def seed_all(s=0):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def parse_date(fname):
    """'2023_20230914.tif' -> (datetime.date, fractional year)."""
    s = Path(fname).stem.split("_")[1]
    d = datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    fy = d.year + (d.timetuple().tm_yday - 1) / 365.25
    return d, fy


def site_dates(site_dir):
    fs = sorted(Path(site_dir).glob("*.tif"))
    return fs, [parse_date(f.name) for f in fs]


LABEL_COLS = ["class", "y", "year", "best_accuracy", "verified_ok", "conflict", "n_teams", "teams",
              "classes_seen", "class_uncertain"]


def load_points():
    """Points table. If HERITAGE_SHUFFLE=<seed> is set, all label/metadata columns are permuted jointly across
    points (label-shuffle sanity test); geometry, site, cells and imagery stay in place."""
    import pandas as pd
    df = pd.read_parquet(CACHE / "points.parquet")
    sh = os.environ.get("HERITAGE_SHUFFLE")
    if sh:
        perm = np.random.RandomState(int(sh)).permutation(len(df))
        df[LABEL_COLS] = df[LABEL_COLS].iloc[perm].to_numpy()
        df["y"] = df["y"].astype(int)
        df["year"] = df["year"].astype(float)
        for c in ("best_accuracy", "verified_ok", "conflict", "n_teams"):
            df[c] = pd.to_numeric(df[c])
    return df


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=range(4), average="macro", zero_division=0)


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)
