"""P1: per point x date quality, local co-registration, 384 px tile cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from imgproc import build_point_tiles, phase_shift, shift_img, center
import numpy as np, pandas as pd, rasterio
from tqdm import tqdm

seed_all(0)
pts = load_points()

# ---- self-test of shift sign convention on a real tile ----
def _selftest():
    site = pts.site.iloc[0]; fs, _ = site_dates(RAW / site)
    with rasterio.open(fs[-1]) as s:
        from imgproc import read_window
        a = read_window(s, pts.col.iloc[0], pts.row.iloc[0], 416)
    b = np.zeros_like(a); b[:, 5:] = a[:, :-5]; b[3:] = b[:-3].copy()     # content moved +5 x, +3 y
    dx, dy, r = phase_shift(center(a, 384), center(b, 384))
    assert abs(dx - 5) < 0.6 and abs(dy - 3) < 0.6, (dx, dy)
    c = shift_img(b, dx, dy)
    err = np.abs(center(c, 300)[..., :3].astype(int) - center(a, 300)[..., :3].astype(int)).mean()
    assert err < 3, err
    print("shift self-test ok", dx, dy, r, err)
_selftest()

qrows = []
for site in SITES:
    fs, dates = site_dates(RAW / site)
    out_dir = CACHE / "tiles" / site; out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(f) for f in fs]
    sp = pts[pts.site == site]
    for _, p in tqdm(sp.iterrows(), total=len(sp), desc=site):
        fn = out_dir / f"{p.point_id}.npz"
        if fn.exists():
            z = np.load(fn, allow_pickle=True); bad = z["bad"]; shifts = z["shifts"]; ref = int(z["ref"])
            qs = list(z["quality"])
        else:
            r = build_point_tiles(srcs, p.col, p.row)
            np.savez_compressed(fn, tiles=r["tiles"], raw=r["raw"], bad=r["bad"], shifts=r["shifts"], ref=r["ref"],
                                quality=np.array(r["quality"], dtype=object),
                                dates=np.array([d.isoformat() for d, _ in dates]),
                                fyear=np.array([fy for _, fy in dates]))
            bad, shifts, ref, qs = r["bad"], r["shifts"], r["ref"], r["quality"]
        for t, (d, fy) in enumerate(dates):
            q = dict(qs[t]); q.update(point_id=p.point_id, site=site, t=t, date=d.isoformat(), fyear=fy,
                                      bad=bool(bad[t]), ref=(t == ref), dx=float(shifts[t, 0]), dy=float(shifts[t, 1]),
                                      resp=float(shifts[t, 2]))
            qrows.append(q)
    for s in srcs: s.close()

q = pd.DataFrame(qrows)
q.to_parquet(CACHE / "quality.parquet")
q["shift"] = np.hypot(q.dx, q.dy)
with open(REPORTS / "data_audit.md", "a") as fh:
    fh.write("\n\n## Per-date quality (fraction of points flagged bad)\n\n")
    fh.write(q.groupby(["site", "date"]).bad.mean().round(3).unstack(0).to_markdown())
    fh.write("\n\n## Co-registration shift distribution (px, non-reference dates)\n\n")
    nr = q[~q.ref]
    applied = nr[nr.resp > 0]
    fh.write(f"applied: {len(applied)}/{len(nr)}; shift quantiles (applied) "
             f"{applied['shift'].quantile([.1,.5,.9,.99]).round(2).to_dict()}\n\n")
    fh.write(nr.groupby("site")["shift"].describe().round(2).to_markdown())
    ng = q.groupby("point_id").bad.apply(lambda b: (~b).sum())
    fh.write(f"\n\nPoints with <2 good dates: {(ng < 2).sum()}\n")
print(q.groupby(["site", "date"]).bad.mean().round(2))
