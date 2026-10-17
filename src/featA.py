"""Model A engineered features. Unary features per (point, date, window); pair features per (i, j)."""
import numpy as np, cv2
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity
from imgproc import center, hist_match

WINS = {"w16": 64, "w32": 128, "w64": 256}   # radius 16/32/64 m -> window side in px
LBP_P, LBP_R = 8, 1
NB_ORI = 12


def _masked(x, m):
    return x[m] if m.sum() > 0 else np.zeros(1, x.dtype)


def unary(rgba):
    """rgba: (384,384,4) uint8. Returns dict of features for each window + LBP hists (for chi2)."""
    feats, hists = {}, {}
    rgb = rgba[..., :3]; al = rgba[..., 3] > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    f = rgb.astype(np.float32) / 255.0
    s = f.sum(-1) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    exg = 2 * g - r - b
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy); ori = (np.arctan2(gy, gx) % np.pi)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform").astype(np.int32)
    V = hsv[..., 2] / 255; S = hsv[..., 1] / 255; H = hsv[..., 0] * 2
    shadow = (V < 0.25) & (S < 0.35)
    green = exg > 0.05
    bare = (H >= 15) & (H <= 45) & (S > 0.18) & (S < 0.6) & (V > 0.35)
    paved = (V > 0.8) & (S < 0.15)
    for wn, ws in WINS.items():
        m = center(al, ws)
        fr = {}
        L = center(lab, ws)
        for c, nm in enumerate("Lab"):
            v = _masked(L[..., c], m); fr[f"{nm}_mean"] = v.mean(); fr[f"{nm}_std"] = v.std()
        fr["exg"] = _masked(center(exg, ws), m).mean()
        fr["green_frac"] = _masked(center(green, ws), m).mean()
        fr["bright"] = _masked(center(V, ws), m).mean()
        fr["shadow_frac"] = _masked(center(shadow, ws), m).mean()
        fr["edge_dens"] = _masked(center(edges, ws), m).mean()
        mg = _masked(center(mag, ws), m); og = _masked(center(ori, ws), m)
        h, _ = np.histogram(og, bins=NB_ORI, range=(0, np.pi), weights=mg)
        h = h / max(h.sum(), 1e-9); fr["ori_entropy"] = -(h * np.log(h + 1e-12)).sum()
        fr["grad_mean"] = mg.mean()
        fr["bare_frac"] = _masked(center(bare, ws), m).mean()
        fr["paved_frac"] = _masked(center(paved, ws), m).mean()
        fr["valid"] = m.mean()
        lh = np.bincount(_masked(center(lbp, ws), m), minlength=LBP_P + 2)[:LBP_P + 2].astype(np.float32)
        hists[wn] = lh / max(lh.sum(), 1)
        for k, v in fr.items():
            feats[f"{wn}_{k}"] = float(v)
    return feats, hists


def pairwise(bef, aft, hb, ha):
    """bef/aft rgba tiles (aft already co-registered). Returns pairwise change features."""
    am = hist_match(aft, bef)
    vb = (bef[..., 3] > 0) & (am[..., 3] > 0)
    lb = cv2.cvtColor(bef[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    la = cv2.cvtColor(am[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    cva = np.sqrt(((la - lb) ** 2).sum(-1))
    gb = cv2.cvtColor(bef[..., :3], cv2.COLOR_RGB2GRAY); ga = cv2.cvtColor(am[..., :3], cv2.COLOR_RGB2GRAY)
    out = {}
    for wn, ws in WINS.items():
        m = center(vb, ws)
        c = _masked(center(cva, ws), m)
        out[f"{wn}_cva_mean"] = float(c.mean()); out[f"{wn}_cva_p90"] = float(np.percentile(c, 90))
        out[f"{wn}_cva_frac20"] = float((c > 20).mean())
        try:
            _, smap = structural_similarity(center(gb, ws), center(ga, ws), data_range=255, full=True, win_size=7)
            out[f"{wn}_ssim"] = float(_masked(smap, m).mean())
        except ValueError:
            out[f"{wn}_ssim"] = 1.0
        p, q = hb[wn], ha[wn]
        out[f"{wn}_lbp_chi2"] = float(0.5 * (((p - q) ** 2) / (p + q + 1e-9)).sum())
        # matched-colour difference of mean Lab (after histogram matching)
        for ch, nm in enumerate("Lab"):
            out[f"{wn}_dm_{nm}"] = float(_masked(center(la[..., ch] - lb[..., ch], ws), m).mean())
    return out
