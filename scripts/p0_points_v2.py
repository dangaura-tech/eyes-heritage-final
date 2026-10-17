"""v3 labels: build cache/points.parquet from training_points_reviewed.geojson (run with HERITAGE_ROOT=<v2 root>).
Organizer points keep their v1 point_id (tiles are reused); extra points get '<site>_x<id>'. Writes y (reviewed class),
y_orig (organizer class, -1 for extra points), year_before / year_after, train_weight, cells and 200 m blocks."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np, pandas as pd
from pyproj import Transformer

V1 = Path(__import__("os").environ.get("BASE_CACHE", CACHE)) / "points.parquet"
gj = json.load(open(DATA_DIR / "training_points_reviewed.geojson"))
old = pd.read_parquet(V1)
okey = {(round(a, 7), round(b, 7)): r for a, b, r in zip(old.lon, old.lat, old.itertuples())}
meta = json.load(open(CACHE / "site_meta.json"))
rows = []
for f in gj["features"]:
    p = dict(f["properties"]); lon, lat = f["geometry"]["coordinates"][:2]
    if not p.get("used_for_training", True):
        continue
    site = p["site"].split("-")[0]; assert site in SITES, site
    o = okey.get((round(lon, 7), round(lat, 7)))
    if p["source"] == "organizers":
        assert o is not None and o.site == site, ("organizer point not found in v1", p["point_id"])
    pid = o.point_id if o is not None else f"{site}_x{int(p['point_id']):05d}"
    m = meta[site]; t = Transformer.from_crs("OGC:CRS84", m["crs"], always_xy=True)
    x, y_ = t.transform(lon, lat); a, b, c, d, e, ff = m["transform"]
    col = (x - c) / a; row = (y_ - ff) / e
    assert 0 <= col < m["width"] and 0 <= row < m["height"], ("outside raster", pid)
    cr = int(np.floor((row - m["lat_off_row"]) / CELL_PX)); cc = int(np.floor((col - m["lat_off_col"]) / CELL_PX))
    rows.append(dict(point_id=pid, site=site, lon=lon, lat=lat, col=col, row=row, utm_x=x, utm_y=y_,
                     cell_id=f"{site}_{cr}_{cc}", cell_r=cr, cell_c=cc, block_id=f"{site}_{cr // 2}_{cc // 2}",
                     source=p["source"], review=p["review"], **{"class": p["class"]}, y=C2I[p["class"]],
                     y_orig=(o.y if o is not None else -1), class_original=p.get("class_original"),
                     year_before=p["year_before"], year_after=p["year_after"],
                     year=(float(p["year_after"]) if p["class"] != "no_change" and p["year_after"] is not None else np.nan),
                     train_weight=float(p["train_weight"]), src_index=int(p["point_id"])))
df = pd.DataFrame(rows)
assert df.point_id.is_unique
print(len(df), "points |", (df.source == "extra").sum(), "extra |", df["class"].value_counts().to_dict())
print("changed vs v1 organizer labels:", int(((df.y_orig >= 0) & (df.y != df.y_orig)).sum()))
print("year_after != v1 year (changed classes):", int(((df.source == "organizers") & df.year.notna() &
      (df.point_id.map(dict(zip(old.point_id, old.year))) != df.year)).sum()))
df.to_parquet(CACHE / "points.parquet")
print(pd.crosstab(df.site, df["class"])[CLASSES])
