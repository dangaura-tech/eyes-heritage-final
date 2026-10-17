"""P0: unzip imagery, verify data facts, build points table with point_id / cell_id."""
import sys, zipfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy import ndimage

# ---------- unzip ----------
for z in sorted((DATA_DIR / "train_imagery").glob("*.zip")):
    site = z.stem.replace("train_imagery_", "").split("-")[0]
    out = RAW / site
    if out.exists() and len(list(out.glob("*.tif"))) == EXPECTED_DATES[site]:
        continue
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            if m.filename.endswith(".tif"):
                (out / Path(m.filename).name).write_bytes(zf.read(m))
    print("unzipped", site)

# ---------- annotations ----------
gj = json.load(open(DATA_DIR / "train_annotations.geojson"))
assert "CRS84" in gj["crs"]["properties"]["name"]
rows = []
for k, f in enumerate(gj["features"]):
    p = dict(f["properties"]); p["lon"], p["lat"] = f["geometry"]["coordinates"][:2]
    p["src_index"] = k
    rows.append(p)
df = pd.DataFrame(rows)
assert len(df) == 776, len(df)
assert df["class"].value_counts().to_dict() == EXPECTED_COUNTS, df["class"].value_counts()
for s, cnt in EXPECTED_SITE.items():
    got = tuple(int(((df.site == s) & (df["class"] == c)).sum()) for c in CLASSES)
    assert got == cnt, (s, got, cnt)
assert int(((df["class"] == "no_change") & df.year.isna()).sum()) == 247

# ---------- imagery audit + lattice ----------
audit = []
site_meta = {}
for site in SITES:
    fs, dates = site_dates(RAW / site)
    assert len(fs) == EXPECTED_DATES[site], (site, len(fs))
    prof0 = None
    alpha_and = None; alpha_or = None
    valid_pct = []
    for f in fs:
        with rasterio.open(f) as r:
            assert r.count == 4 and r.dtypes[0] == "uint8", (f, r.count, r.dtypes)
            assert abs(r.res[0] - 0.5) < 1e-6 and abs(r.res[1] - 0.5) < 1e-6, r.res
            assert r.crs.is_projected and "UTM" in r.crs.to_wkt(), r.crs
            meta = (r.crs.to_string(), r.transform, r.width, r.height)
            if prof0 is None:
                prof0 = meta
            assert meta == prof0, ("grid mismatch", f, meta, prof0)
            a = r.read(4) > 0
        valid_pct.append(a.mean() * 100)
        alpha_and = a if alpha_and is None else alpha_and & a
        alpha_or = a if alpha_or is None else alpha_or | a
    crs, tr, W, H = prof0
    # holes: alpha==0 everywhere in all dates, enclosed (not connected to border)
    inv = ~alpha_or
    lab, n = ndimage.label(inv)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    holes = [i for i in range(1, n + 1) if i not in border]
    sl = ndimage.find_objects(lab)
    # lattice offset from hole bounding boxes
    offs_c = [sl[i - 1][1].start % CELL_PX for i in holes] + [sl[i - 1][1].stop % CELL_PX for i in holes]
    offs_r = [sl[i - 1][0].start % CELL_PX for i in holes] + [sl[i - 1][0].stop % CELL_PX for i in holes]
    oc = int(np.bincount(offs_c, minlength=CELL_PX).argmax()) if holes else 0
    orr = int(np.bincount(offs_r, minlength=CELL_PX).argmax()) if holes else 0
    frac_aligned = np.mean([(o - oc) % CELL_PX == 0 for o in offs_c]) if holes else 0
    sizes = [((sl[i-1][0].stop - sl[i-1][0].start), (sl[i-1][1].stop - sl[i-1][1].start)) for i in holes]
    mult = np.mean([(h % CELL_PX == 0) and (w % CELL_PX == 0) for h, w in sizes]) if holes else 0
    site_meta[site] = dict(crs=crs, transform=list(tr)[:6], width=W, height=H, lat_off_col=oc, lat_off_row=orr,
                           dates=[d.isoformat() for d, _ in dates], files=[f.name for f in fs])
    audit.append(dict(site=site, crs=crs, res=0.5, width=W, height=H, n_dates=len(fs),
                      dates=", ".join(d.isoformat() for d, _ in dates),
                      valid_pct_min=round(min(valid_pct), 1), valid_pct_max=round(max(valid_pct), 1),
                      n_holes=len(holes), hole_px_total=int(sum((lab == i).sum() for i in holes)) if holes else 0,
                      holes_lattice_aligned=round(float(frac_aligned), 3), holes_mult200=round(float(mult), 3),
                      lattice_offset=(oc, orr)))
    print(audit[-1])

json.dump(site_meta, open(CACHE / "site_meta.json", "w"), indent=1)

# ---------- point pixel coords, ids, cells ----------
pc, pr, ux, uy, cid, inb = [], [], [], [], [], []
for i, row in df.iterrows():
    m = site_meta[row.site]
    t = Transformer.from_crs("OGC:CRS84", m["crs"], always_xy=True)
    x, y = t.transform(row.lon, row.lat)
    a, b, c, d, e, f = m["transform"]
    col = (x - c) / a; rr = (y - f) / e
    pc.append(col); pr.append(rr); ux.append(x); uy.append(y)
    cc = int(np.floor((col - m["lat_off_col"]) / CELL_PX)); cr = int(np.floor((rr - m["lat_off_row"]) / CELL_PX))
    cid.append(f"{row.site}_{cr}_{cc}")
    inb.append(0 <= col < m["width"] and 0 <= rr < m["height"])
df["col"], df["row"], df["utm_x"], df["utm_y"], df["cell_id"] = pc, pr, ux, uy, cid
assert all(inb), "points outside raster"
df["cell_r"] = [int(c.split("_")[1]) for c in cid]; df["cell_c"] = [int(c.split("_")[2]) for c in cid]
df["block_id"] = [f"{s}_{r//2}_{c//2}" for s, r, c in zip(df.site, df.cell_r, df.cell_c)]
df["point_id"] = [f"{s}_{k:04d}" for s, k in zip(df.site, df.src_index)]
df["y"] = df["class"].map(C2I)
df.to_parquet(CACHE / "points.parquet")

ad = pd.DataFrame(audit)
with open(REPORTS / "data_audit.md", "w") as fh:
    fh.write("# Data audit\n\nAll asserts in `scripts/p0_audit.py` passed: 776 points, class/site counts as briefed, "
             "4-band uint8 RGBA, 0.5 m UTM, identical grid across dates per site.\n\n")
    fh.write(ad.to_markdown(index=False))
    fh.write(f"\n\nPoints: {len(df)}; cells: {df.cell_id.nunique()}; 200 m blocks: {df.block_id.nunique()}\n")
print(ad.to_string()); print("cells", df.cell_id.nunique(), "blocks", df.block_id.nunique())
