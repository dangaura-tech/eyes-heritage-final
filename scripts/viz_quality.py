import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pts = load_points(); (REPORTS/"figs").mkdir(exist_ok=True)
rng = np.random.RandomState(1)
for site in SITES:
    ids = rng.choice(pts[pts.site==site].point_id, 3, replace=False)
    zs = [np.load(CACHE/"tiles"/site/f"{i}.npz", allow_pickle=True) for i in ids]
    T = len(zs[0]["dates"])
    fig, ax = plt.subplots(3, T, figsize=(2*T, 6.4))
    for r, z in enumerate(zs):
        for t in range(T):
            ax[r,t].imshow(z["tiles"][t][...,:3]); ax[r,t].axis("off")
            q = z["quality"][t]
            ax[r,t].set_title(f"{z['dates'][t]}{' BAD' if z['bad'][t] else ''}\nlv{q['lap_var']:.0f} c{q['contrast']:.2f} h{q['haze']:.2f}", fontsize=6, color="red" if z["bad"][t] else "k")
    plt.tight_layout(); plt.savefig(REPORTS/"figs"/f"quality_{site}.png", dpi=80); plt.close()
