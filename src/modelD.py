"""Model D: temporal transformer over all valid dates of a point (year-free, point-level logits)."""
import json, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from pairs import valid_dates, point_weights
from modelB import loss_fn


class SeqStore:
    """Per point: rows into the centre-pooled embedding table (valid dates), fractional years, quality features.
    extra_dirs: further embedding caches concatenated feature-wise (aligned by point_id, date)."""
    def __init__(self, emb_dir, pts, extra_dirs=()):
        d = Path(emb_dir)
        self.cen = np.load(d / "center.npy").astype(np.float32)            # (N,8,2D)
        idx = pd.read_parquet(d / "index.parquet")
        row = {(p, t): k for k, (p, t) in enumerate(zip(idx.point_id, idx.t))}
        for ed in extra_dirs:
            ed = Path(ed); c2 = np.load(ed / "center.npy", mmap_mode="r"); i2 = pd.read_parquet(ed / "index.parquet")
            r2 = {(p, t): k for k, (p, t) in enumerate(zip(i2.point_id, i2.t))}
            order = np.array([r2[(p, t)] for p, t in zip(idx.point_id, idx.t)])
            self.cen = np.concatenate([self.cen, np.asarray(c2[order], np.float32)], -1)
        self.seq, self.qual = {}, {}
        for pid, site in zip(pts.point_id, pts.site):
            z = np.load(tile_path(site, pid), allow_pickle=True)
            fy = z["fyear"]; bad = point_bad(site, z); Q = list(z["quality"])
            vi, _ = valid_dates(bad, Q)
            rows = [row[(pid, int(t))] for t in vi]
            self.seq[pid] = (np.array(rows), fy[vi].astype(np.float32))
            self.qual[pid] = np.array([[np.log1p(Q[t]["lap_var"]) / 10, Q[t]["contrast"] * 5, Q[t]["haze"] * 2, Q[t]["core_valid"]]
                                       for t in vi], np.float32)


class SeqNet(nn.Module):
    def __init__(self, d_in, d=256, layers=2, heads=4, drop=0.1, delta=False, n_date=3):
        super().__init__()
        self.delta = delta
        self.inp = nn.Sequential(nn.LayerNorm(d_in * (2 if delta else 1)), nn.Linear(d_in * (2 if delta else 1), d))
        self.denc = nn.Linear(n_date, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 4))

    def forward(self, x, dates, mask):
        # x (B,T,D) ; dates (B,T,n_date) ; mask (B,T) True=valid
        if self.delta:      # change w.r.t. the previous valid date (first date: zero)
            prev = torch.cat([x[:, :1], x[:, :-1]], 1)
            x = torch.cat([x, (x - prev) * mask[..., None]], -1)
        x = self.inp(x)
        x = x - (x * mask[..., None]).sum(1, keepdim=True) / mask.sum(1, keepdim=True)[..., None].clamp(min=1) * 0.5
        x = x + self.denc(dates)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        m = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        x = self.tf(x, src_key_padding_mask=~m)
        return self.out(x[:, 0])


class ModelD:
    def __init__(self, pts, store, cfg, device="cuda"):
        self.pts, self.S, self.cfg, self.dev = pts.reset_index(drop=True), store, cfg, device

    def _dims(self):
        q = 4 if self.cfg.get("qual") else 0
        return self.S.cen.shape[-1] + q, (4 if self.cfg.get("gapenc") else 3)

    def _batch(self, ks, train, view=None):
        Tm = max(len(self.S.seq[self.pts.point_id.iloc[k]][0]) for k in ks)
        D, nd = self._dims(); E = self.S.cen.shape[-1]
        X = np.zeros((len(ks), Tm, D), np.float32); Dt = np.zeros((len(ks), Tm, nd), np.float32)
        M = np.zeros((len(ks), Tm), bool)
        for b, k in enumerate(ks):
            rows, fy = self.S.seq[self.pts.point_id.iloc[k]]
            sel = np.arange(len(rows))
            if train and len(rows) > 3 and np.random.rand() < 0.7:
                n = np.random.randint(3, len(rows) + 1); sel = np.sort(np.random.choice(len(rows), n, replace=False))
            views = np.random.randint(8, size=len(sel)) if train else np.full(len(sel), view or 0, int)
            X[b, :len(sel), :E] = self.S.cen[rows[sel], views]
            if self.cfg.get("qual"):
                X[b, :len(sel), E:] = self.S.qual[self.pts.point_id.iloc[k]][sel]
            f = fy[sel]
            cols = [(f - 2022) / 4, np.sin(2 * np.pi * f), np.cos(2 * np.pi * f)]
            if self.cfg.get("gapenc"):
                cols.append(np.r_[0.0, np.diff(f)] / 3)
            Dt[b, :len(sel)] = np.stack(cols, 1)
            M[b, :len(sel)] = True
        return (torch.from_numpy(X).to(self.dev), torch.from_numpy(Dt).to(self.dev), torch.from_numpy(M).to(self.dev))

    def fit(self, tr, va, seed=0, all_tr=None):
        cfg = self.cfg; torch.manual_seed(seed); np.random.seed(seed)
        wp = point_weights(self.pts, cfg.get("label_w", True), cfg.get("balance", True))
        tr = np.array(tr); y = self.pts.y.to_numpy()
        D, nd = self._dims()
        net = SeqNet(D, d=cfg.get("d", 256), layers=cfg.get("layers", 2), drop=cfg.get("drop", 0.1),
                     delta=cfg.get("delta", False), n_date=nd).to(self.dev)
        E = cfg.get("epochs", 200); bs = cfg.get("bs", 32)
        spe = math.ceil(len(tr) / bs); total = E * spe
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 0.05))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.05 * total)) *
                                                  0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        cnt = np.bincount(y[tr], minlength=4) + 1
        prior = torch.tensor(cnt / cnt.sum(), dtype=torch.float32, device=self.dev)
        best = (1e9, None, 0)
        for ep in range(cfg.get("stop_at") or E):
            net.train(); perm = np.random.permutation(tr)
            for s in range(0, len(perm), bs):
                ks = perm[s:s + bs]
                X, Dt, M = self._batch(ks, True)
                lg = net(X, Dt, M)
                loss = loss_fn(lg, torch.from_numpy(y[ks]).to(self.dev), torch.from_numpy(wp[ks]).float().to(self.dev), cfg, prior)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if (ep + 1) % 10 == 0 and len(va):
                net.eval()
                with torch.no_grad():
                    X, Dt, M = self._batch(va, False)
                    l = F.cross_entropy(net(X, Dt, M), torch.from_numpy(y[va]).to(self.dev), reduction="none")
                    vl = float((l * torch.from_numpy(wp[va]).float().to(self.dev)).mean())
                if vl < best[0]:
                    best = (vl, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        if best[1] is not None and cfg.get("keep_best", True):
            net.load_state_dict(best[1])
        self.net = net
        return dict(best_ep=best[2])

    def fit_final(self, tr, seed, epochs):
        cfg = dict(self.cfg); cfg["stop_at"] = int(epochs); cfg["keep_best"] = False
        old = self.cfg; self.cfg = cfg
        self.fit(np.array(tr), np.array([], int), seed)
        self.cfg = old
        self.nets = getattr(self, "nets", []) + [self.net]

    def bind(self, pts, store):
        self.pts, self.S = pts.reset_index(drop=True), store

    @torch.no_grad()
    def predict(self, idx, tta=8):
        nets = getattr(self, "nets", None) or [self.net]
        out = 0
        for n in nets:
            self.net = n
            out = out + self._predict1(idx, tta)
        return out / len(nets)

    @torch.no_grad()
    def _predict1(self, idx, tta=8):
        self.net.eval(); out = 0
        for v in range(tta):
            P = []
            for s in range(0, len(idx), 128):
                ks = idx[s:s + 128]
                X, Dt, M = self._batch(ks, False, view=v)      # TTA: dihedral view v for every date
                P.append(torch.softmax(self.net(X, Dt, M), -1).cpu().numpy())
            out = out + np.concatenate(P)
        return out / tta
