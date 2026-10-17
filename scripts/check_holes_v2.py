"""Assert that no training point lies inside a withheld test cell (alpha = 0 in every date at the point)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np
pts = load_points(); bad = []
for pid, s, src in zip(pts.point_id, pts.site, pts.source):
    z = np.load(tile_path(s, pid), allow_pickle=True)["tiles"]
    c = z.shape[1] // 2
    centre = z[:, c - 4:c + 4, c - 4:c + 4, 3]
    if (centre == 0).all():
        bad.append((pid, src))
print("points inside withheld cells / outside AOI:", bad)
assert not bad
print("ok:", len(pts), "points, none inside alpha=0 holes")
