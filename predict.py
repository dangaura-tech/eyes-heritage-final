"""End-to-end prediction on new points / a new site.

usage:
  python predict.py --points points.geojson|points.csv --imagery <dir with YYYY_YYYYMMDD.tif | zip> \
                    --out predictions.csv [--site name] [--use-year] [--models models/final] [--work work_dir]

The points file needs lon/lat (CRS84) and an id column (default 'id'; falls back to row index). A 'site'
column is optional. Runs the same quality scoring, co-registration, tile extraction, feature extraction,
all final models with 8x dihedral TTA, the ensemble and the decision rule fitted in CV.
"""
import argparse, json, sys, zipfile, logging, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np, pandas as pd

# ---------------- output format: edit only this block to map the official submission format ----------------
OUTPUT = dict(
    id="id", p_change="p_change", pred_class="pred_class",
    probs={"construction": "p_construction", "destruction": "p_destruction",
           "soil_sealing": "p_soil_sealing", "no_change": "p_no_change"},
    class_names={"construction": "construction", "destruction": "destruction",
                 "soil_sealing": "soil_sealing", "no_change": "no_change"},
    sep=",", float_format="%.5f")
# -------------------------------------------------------------------------------------------------------------

import common
from common import CLASSES, NC, seed_all, site_dates, parse_date
from imgproc import build_point_tiles
from pairs import decide, aggregate, valid_dates, all_pairs

log = logging.getLogger("predict")


def read_points(fn, id_col, site):
    fn = Path(fn)
    if fn.suffix.lower() in (".geojson", ".json"):
        gj = json.load(open(fn)); rows = []
        for k, f in enumerate(gj["features"]):
            p = dict(f.get("properties") or {}); p["lon"], p["lat"] = f["geometry"]["coordinates"][:2]
            p.setdefault(id_col, f.get("id", k)); rows.append(p)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(fn)
        ren = {c: "lon" for c in df.columns if c.lower() in ("lon", "longitude", "x", "lng")}
        ren.update({c: "lat" for c in df.columns if c.lower() in ("lat", "latitude", "y")})
        df = df.rename(columns=ren)
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))
    if site:
        df["site"] = site
    elif "site" not in df.columns:
        df["site"] = "newsite"
    return df


def prepare_imagery(imagery, work):
    p = Path(imagery)
    if p.suffix == ".zip":
        out = work / "imagery"; out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as zf:
            for m in zf.infolist():
                if m.filename.endswith(".tif"):
                    (out / Path(m.filename).name).write_bytes(zf.read(m))
        return out
    if not list(p.glob("*.tif")):
        subs = [d for d in p.iterdir() if d.is_dir() and list(d.glob("*.tif"))]
        assert len(subs) == 1, f"no GeoTIFFs found in {p}"
        return subs[0]
    return p


def build_tiles(df, img_dir, work, id_col):
    import rasterio
    from pyproj import Transformer
    fs, dates = site_dates(img_dir)
    assert len(fs) >= 2, "need at least two dates"
    srcs = [rasterio.open(f) for f in fs]
    ref = srcs[0]
    for s in srcs[1:]:
        assert s.crs == ref.crs and s.transform == ref.transform and s.shape == ref.shape, "dates not on one grid"
    log.info("imagery %s: %d dates, CRS %s (zone taken from GeoTIFF), %dx%d", img_dir, len(fs), ref.crs, ref.width, ref.height)
    tr = Transformer.from_crs("OGC:CRS84", ref.crs, always_xy=True)
    tile_dir = work / "tiles"
    rows = []
    for k, r in df.iterrows():
        site = str(r.site); pid = f"{site}_{k:05d}"
        fn = tile_dir / site / f"{pid}.npz"; fn.parent.mkdir(parents=True, exist_ok=True)
        x, y = tr.transform(r.lon, r.lat)
        col, row = ~ref.transform * (x, y)
        if not fn.exists():
            t = build_point_tiles(srcs, col, row)
            np.savez_compressed(fn, tiles=t["tiles"], raw=t["raw"], bad=t["bad"], shifts=t["shifts"], ref=t["ref"],
                                quality=np.array(t["quality"], dtype=object),
                                dates=np.array([d.isoformat() for d, _ in dates]),
                                fyear=np.array([fy for _, fy in dates]))
            if (~t["bad"]).sum() < 2:
                log.warning("point %s: %d good dates -> falling back to least-bad pair", r[id_col], (~t["bad"]).sum())
        rows.append(dict(point_id=pid, site=site, src_id=r[id_col], lon=r.lon, lat=r.lat, col=col, row=row,
                         year=r.get("year", np.nan), y=NC, best_accuracy=70, verified_ok=0, conflict=0, n_teams=1,
                         **{"class": "no_change"}))
    for s in srcs: s.close()
    common.set_tile_dir(tile_dir)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True); ap.add_argument("--imagery", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--site", default=None)
    ap.add_argument("--id-col", default="id"); ap.add_argument("--use-year", action="store_true",
                    help="if the points file has a 'year' column, score only the core pair (last before / first after)")
    ap.add_argument("--models", default=str(common.MODELS / "final")); ap.add_argument("--work", default=None)
    ap.add_argument("--tta", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_all(0)
    work = Path(a.work or tempfile.mkdtemp(prefix="heritage_pred_")); work.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.models)
    ens = json.load(open(mdir / "ensemble.json"))
    df = read_points(a.points, a.id_col, a.site)
    img = prepare_imagery(a.imagery, work)
    pts = build_tiles(df, img, work, a.id_col)
    log.info("%d points tiled", len(pts))
    from final import predict_members
    probs = predict_members(pts, ens, mdir, work, use_year=a.use_year, tta=a.tta)
    P = sum(w * probs[m] for m, w in ens["weights"].items() if w > 0)
    P = P / P.sum(1, keepdims=True)
    pred = decide(P, ens["t"], np.array(ens["b"]))          # decision on the raw ensemble (as tuned in CV)
    if ens.get("pc_platt"):
        # monotone Platt calibration of p_change (fitted on pooled LOSO OOF): calibrated probabilities for output,
        # ranking and decisions unchanged; class split of p_change kept proportional
        a_, c_ = ens["pc_platt"]; pc = np.clip(1 - P[:, NC], 1e-6, 1 - 1e-6)
        pcc = 1 / (1 + np.exp(-(a_ * np.log(pc / (1 - pc)) + c_)))
        cd = P[:, :3] / np.clip(P[:, :3].sum(1, keepdims=True), 1e-9, None)
        P = np.c_[cd * pcc[:, None], 1 - pcc]
    out = pd.DataFrame({OUTPUT["id"]: pts.src_id, OUTPUT["p_change"]: 1 - P[:, NC],
                        OUTPUT["pred_class"]: [OUTPUT["class_names"][CLASSES[c]] for c in pred]})
    for c, name in OUTPUT["probs"].items():
        out[name] = P[:, CLASSES.index(c)]
    out.to_csv(a.out, index=False, sep=OUTPUT["sep"], float_format=OUTPUT["float_format"])
    log.info("wrote %s (%d rows); class counts %s", a.out, len(out), out[OUTPUT["pred_class"]].value_counts().to_dict())


if __name__ == "__main__":
    main()
