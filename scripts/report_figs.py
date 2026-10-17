"""Figures for the final report from an ensemble result. usage: report_figs.py <ens_name>"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
from pairs import decide, valid_dates
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1]
ens = json.load(open(REPORTS / f"{name}.json"))
z = np.load(CACHE / "oof" / f"{name}.npz")
Pi, Pl = z["oof_in"][0], z["oof_loso"][0]
pts = load_points(); y = pts.y.to_numpy()
t, b = ens["t"], np.array(ens["b"])
figs = REPORTS / "figs"; figs.mkdir(exist_ok=True)
short = ["constr", "destr", "soil", "no_ch"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, (P, ttl) in zip(ax, [(Pi, "In-domain spatial CV"), (Pl, "Leave-one-site-out")]):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, decide(P, t, b), labels=range(4))
    a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm[i].sum() * 0.6 else "white")
    a.set_xticks(range(4)); a.set_xticklabels(short); a.set_yticks(range(4)); a.set_yticklabels(short)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(ttl)
plt.tight_layout(); plt.savefig(figs / "confusion.png", dpi=110); plt.close()

# t robustness curve
c = np.array(ens["t_curve"])
plt.figure(figsize=(5, 3.2)); plt.plot(c[:, 0], c[:, 1], "o-", label="S (fixed t)"); plt.plot(c[:, 0], c[:, 2], label="M_in")
plt.plot(c[:, 0], c[:, 3], label="M_out (pooled)"); plt.axvline(t, ls="--", c="gray"); plt.xlabel("threshold t")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figs / "t_curve.png", dpi=110); plt.close()


def strip(rows, fn, title):
    fig, ax = plt.subplots(len(rows), 2, figsize=(6.0, 3.0 * len(rows)))
    for r, (k, P) in enumerate(rows):
        p = pts.iloc[k]
        zz = np.load(tile_path(p.site, p.point_id), allow_pickle=True)
        yrs = np.array([int(d[:4]) for d in zz["dates"]]); vi, _ = valid_dates(zz["bad"].astype(bool), list(zz["quality"]))
        if p.y != NC and np.isfinite(p.year):
            bef = [i for i in vi if yrs[i] < p.year]; aft = [i for i in vi if yrs[i] >= p.year]
            i, j = (bef[-1] if bef else vi[0]), (aft[0] if aft else vi[-1])
            if i == j:
                i, j = vi[0], vi[-1]
        else:
            i, j = vi[0], vi[-1]
        for c_, tt in enumerate((i, j)):
            img = zz["tiles"][tt][96:288, 96:288, :3]
            ax[r, c_].imshow(img); ax[r, c_].axis("off")
            ax[r, c_].plot([96], [96], "r+", ms=10)
            ax[r, c_].set_title(zz["dates"][tt], fontsize=7)
        pr = decide(P[None], t, b)[0]
        ax[r, 0].text(0, -40, f"{p.point_id} true={CLASSES[p.y]} pred={CLASSES[pr]} p_ch={1 - P[NC]:.2f}", fontsize=6.5)
    fig.suptitle(title, fontsize=9); plt.tight_layout(); plt.savefig(fn, dpi=95); plt.close()


pred = decide(Pl, t, b)
conf = np.where(pred != y, Pl.max(1), -1)
err = np.argsort(-conf)[:12]
strip([(k, Pl[k]) for k in err], figs / "confident_errors.png", "12 most confident LOSO errors (96 m crops)")
unc = np.argsort(np.abs((1 - Pl[:, NC]) - t))[:12]
strip([(k, Pl[k]) for k in unc], figs / "uncertain.png", "12 most uncertain LOSO cases (p_change closest to t)")
print("figures written to", figs)
