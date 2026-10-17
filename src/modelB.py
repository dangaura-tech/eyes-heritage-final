"""Model B: frozen-backbone token cache + small bitemporal transformer head."""
import json, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from pairs import label_pairs, train_pairs_for, valid_dates, point_weights, aggregate, all_pairs, infer_pairs


class EmbStore:
    """Memory-mapped token cache for one backbone/config."""

    def __init__(self, path, site_norm=False):
        path = Path(path)
        self.meta = json.load(open(path / "meta.json"))
        self.tok = np.load(path / "tokens.npy", mmap_mode="r")
        self.cls = np.load(path / "cls.npy", mmap_mode="r")
        self.valid = np.load(path / "valid.npy").astype(np.float32)
        idx = pd.read_parquet(path / "index.parquet")
        self.row = {(p, t): k for k, (p, t) in enumerate(zip(idx.point_id, idx.t))}
        self.G = self.meta["grid"]; self.D2 = self.tok.shape[-1]
        self.site_of = idx.site.to_numpy(); self.norm = None
        if site_norm:
            self.norm = site_token_stats(self, path)

    def get(self, pid, t, view):
        k = self.row[(pid, t)]
        f = np.asarray(self.tok[k, view])
        if self.norm is not None:
            mu, sd = self.norm[self.site_of[k]]
            f = ((f.astype(np.float32) - mu) / sd).astype(np.float16)
        return f, self.valid[k]


def site_token_stats(store, path, n_tiles=400, seed=0):
    """AdaBN-style per-site channel statistics of the frozen tokens, from the site's *unlabelled* imagery
    (all tiles of that site, view 0, valid tokens only). Cached next to the token file."""
    fn = Path(path) / "site_stats.npz"
    if fn.exists():
        z = np.load(fn, allow_pickle=True); return z["stats"].item()
    rng = np.random.RandomState(seed); stats = {}
    for s in np.unique(store.site_of):
        rows = np.where(store.site_of == s)[0]; rows = np.sort(rng.choice(rows, min(n_tiles, len(rows)), replace=False))
        X = np.concatenate([np.asarray(store.tok[k, 0], np.float32)[store.valid[k] > 0.5] for k in rows])
        stats[s] = (X.mean(0), X.std(0) + 1e-3)
    np.savez(fn, stats=np.array(stats, dtype=object))
    return stats


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, items, store, train):
        self.items, self.S, self.train = items, store, train   # items: (pid, i, j, label, weight)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        pid, i, j, lab, w = self.items[n]
        if self.train:
            vb, va = np.random.randint(8), np.random.randint(8)
        else:
            vb = va = 0
        fb, mb = self.S.get(pid, i, vb); fa, ma = self.S.get(pid, j, va)
        return (torch.from_numpy(fb.copy()), torch.from_numpy(fa.copy()), torch.from_numpy(np.minimum(mb, ma)),
                lab, np.float32(w))


class Head(nn.Module):
    def __init__(self, d_in, G, d=256, layers=2, heads=4, sigma_tokens=2.0, drop=0.1, use_mid=True):
        super().__init__()
        self.G, self.use_mid = G, use_mid
        d_in_eff = d_in if use_mid else d_in // 2
        self.norm = nn.LayerNorm(d_in_eff)
        self.proj = nn.Linear(d_in_eff, d)
        self.fuse = nn.Sequential(nn.Linear(3 * d + 1, d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, G * G, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        self.att = nn.Linear(d, 1)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_tokens)))
        yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
        c = (G - 1) / 2
        self.register_buffer("d2", ((yy - c) ** 2 + (xx - c) ** 2).flatten().float())
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 4))

    def forward(self, fb, fa, mask):
        if not self.use_mid:
            h = fb.shape[-1] // 2; fb, fa = fb[..., :h], fa[..., :h]
        fb, fa = fb.float(), fa.float()
        D = fb.shape[-1] // (2 if self.use_mid else 1)
        cos = F.cosine_similarity(fb[..., :D], fa[..., :D], dim=-1).unsqueeze(-1)
        hb = self.proj(self.norm(fb)); ha = self.proj(self.norm(fa))
        x = self.fuse(torch.cat([hb, ha, (ha - hb).abs(), cos], -1)) + self.pos
        valid = mask > 0.5
        valid = valid | (~valid.any(1, keepdim=True))          # avoid all-masked rows
        x = self.tf(x, src_key_padding_mask=~valid)
        s = self.att(x).squeeze(-1) - self.d2[None] / (2 * torch.exp(self.log_sigma) ** 2)
        s = s.masked_fill(~valid, -1e4)
        a = torch.softmax(s, 1)
        return self.out((a.unsqueeze(-1) * x).sum(1))


def shift_tokens(fa, m, G):
    """Random +-1 token shift of the 'after' grid (batch-wise), wrapped tokens masked."""
    B = fa.shape[0]
    dy, dx = np.random.randint(-1, 2, 2)
    if dy == 0 and dx == 0:
        return fa, m
    f = fa.view(B, G, G, -1).roll((int(dy), int(dx)), (1, 2)); mm = m.view(B, G, G).roll((int(dy), int(dx)), (1, 2)).clone()
    if dy == 1: mm[:, 0] = 0
    if dy == -1: mm[:, -1] = 0
    if dx == 1: mm[:, :, 0] = 0
    if dx == -1: mm[:, :, -1] = 0
    return f.reshape(B, G * G, -1), mm.reshape(B, G * G)


def loss_fn(logits, y, w, cfg, prior):
    ls = cfg.get("smooth", 0.05)
    if cfg.get("loss", "focal") == "la":
        logits = logits + cfg.get("tau", 1.0) * torch.log(prior)[None]
        ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=ls)
    else:
        logp = F.log_softmax(logits, -1)
        tgt = F.one_hot(y, 4).float() * (1 - ls) + ls / 4
        p = logp.exp()
        g = cfg.get("gamma", 1.5)
        ce = -(tgt * (1 - p) ** g * logp).sum(-1)
    return (ce * w).sum() / w.sum()


class ModelB:
    def __init__(self, pts, store, cfg, device="cuda"):
        self.pts, self.S, self.cfg, self.dev = pts.reset_index(drop=True), store, cfg, device
        self.meta = {}
        for pid, site in zip(self.pts.point_id, self.pts.site):
            z = np.load(tile_path(site, pid), allow_pickle=True)
            self.meta[pid] = (np.array([int(d[:4]) for d in z["dates"]]), point_bad(site, z), list(z["quality"]))

    def items(self, idx, wp, per_point_sample=False):
        cfg = self.cfg; out = []
        for k in idx:
            r = self.pts.iloc[k]; years, bad, q = self.meta[r.point_id]
            vi, _ = valid_dates(bad, q)
            lp = train_pairs_for(r, years, vi, ambiguous=cfg.get("ambiguous", "exclude"),
                             reverse=cfg.get("reverse", True))
            if cfg.get("pairs") == "core":
                if r.y == NC:
                    cons = {(int(vi[a]), int(vi[a + 1])) for a in range(len(vi) - 1)}
                    lp = [l for l in lp if (l[0], l[1]) in cons]
                else:
                    lp = [l for l in lp if l[4]][:1]
            if not lp: continue
            tot = sum(l[3] * (2.0 if l[4] else 1.0) for l in lp)
            for i, j, lab, pw, core in lp:
                out.append((r.point_id, int(i), int(j), int(lab), float(wp[k] * pw * (2.0 if core else 1.0) / tot), k))
        return out

    def _train(self, items, val_items, epochs, seed, stop_at=None):
        cfg = self.cfg
        torch.manual_seed(seed); np.random.seed(seed)
        G = self.S.G
        net = Head(self.S.D2, G, d=cfg.get("d", 256), layers=cfg.get("layers", 2),
                   sigma_tokens=16.0 / (128.0 / G if self.S.meta["config"].startswith("B1") else 64.0 / G),
                   drop=cfg.get("drop", 0.1), use_mid=cfg.get("use_mid", True)).to(self.dev)
        # per-point sampling: every epoch sees each training point once (pair drawn by weight)
        by_pt = {}
        for it in items: by_pt.setdefault(it[5], []).append(it)
        pts_keys = list(by_pt)
        pt_w = {k: sum(i[4] for i in v) for k, v in by_pt.items()}
        mw = np.mean(list(pt_w.values()))
        bs = cfg.get("bs", 64)
        steps_per_ep = math.ceil(len(pts_keys) / bs)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("wd", 0.05))
        total = epochs * steps_per_ep
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.05 * total)) *
                                                  0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        cls_cnt = np.bincount([it[3] for it in items], minlength=4) + 1
        prior = torch.tensor(cls_cnt / cls_cnt.sum(), dtype=torch.float32, device=self.dev)
        hist = []; best = (1e9, None, 0)
        for ep in range(stop_at or epochs):
            net.train()
            ep_items = []
            for k in pts_keys:
                v = by_pt[k]; pw = np.array([i[4] for i in v])
                c = v[np.random.choice(len(v), p=pw / pw.sum())]
                ep_items.append((c[0], c[1], c[2], c[3], pt_w[k] / mw))
            dl = torch.utils.data.DataLoader(PairDataset(ep_items, self.S, True), batch_size=bs, shuffle=True,
                                             num_workers=cfg.get("workers", 8), drop_last=False,
                                             persistent_workers=False)
            for fb, fa, m, y, w in dl:
                fb, fa, m = fb.to(self.dev, non_blocking=True), fa.to(self.dev, non_blocking=True), m.to(self.dev)
                y, w = y.to(self.dev), w.to(self.dev).float()
                fa, m = shift_tokens(fa, m, G)
                if cfg.get("tokdrop", 0.1) > 0:
                    m = m * (torch.rand_like(m) > cfg.get("tokdrop", 0.1)).float()
                if cfg.get("mixup", 0.2) > 0 and np.random.rand() < 0.5:
                    # mixup within the same class
                    perm = torch.arange(len(y), device=self.dev)
                    for c in range(4):
                        ii = (y == c).nonzero().flatten()
                        if len(ii) > 1: perm[ii] = ii[torch.randperm(len(ii), device=self.dev)]
                    lam = float(np.random.beta(0.4, 0.4)); lam = max(lam, 1 - lam)
                    fb = (lam * fb.float() + (1 - lam) * fb[perm].float()); fa = (lam * fa.float() + (1 - lam) * fa[perm].float())
                    m = torch.minimum(m, m[perm])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(fb, fa, m)
                loss = loss_fn(logits.float(), y, w, cfg, prior)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if val_items is not None and ((ep + 1) % cfg.get("eval_every", 10) == 0 or ep == epochs - 1):
                vl = self._val_loss(net, val_items, prior)
                hist.append((ep + 1, vl))
                if vl < best[0]:
                    best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep + 1)
        if val_items is not None and best[1] is not None and cfg.get("keep_best", True):
            net.load_state_dict(best[1])
        return net, best[2], hist

    @torch.no_grad()
    def _val_loss(self, net, val_items, prior):
        net.eval()
        dl = torch.utils.data.DataLoader(PairDataset([v[:5] for v in val_items], self.S, False), batch_size=256,
                                         num_workers=self.cfg.get("workers", 8))
        L, W = 0.0, 0.0
        for fb, fa, m, y, w in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(fb.to(self.dev), fa.to(self.dev), m.to(self.dev)).float()
            y = y.to(self.dev); w = w.to(self.dev).float()
            L += float((F.cross_entropy(lg, y, reduction="none") * w).sum()); W += float(w.sum())
        return L / max(W, 1e-9)

    def fit(self, tr, va, seed=0, all_tr=None):
        wp = point_weights(self.pts, self.cfg.get("label_w", True), self.cfg.get("balance", True))
        it_tr = self.items(tr, wp); it_va = self.items(va, wp)
        E = self.cfg.get("epochs", 150)
        self.nets = []
        net, best_ep, hist = self._train(it_tr, it_va, E, seed)
        self.nets.append(net)
        return dict(best_ep=best_ep, hist=hist)

    def fit_final(self, tr, seed, epochs):
        wp = point_weights(self.pts, self.cfg.get("label_w", True), self.cfg.get("balance", True))
        # same cosine schedule length as in CV, stopped at the CV-selected epoch (mimics keep-best)
        net, _, _ = self._train(self.items(tr, wp), None, self.cfg.get("epochs", 150), seed, stop_at=int(epochs))
        self.nets = getattr(self, "nets", []) + [net]

    def bind(self, pts, store):
        """Re-point the model at a new set of points / embedding store (prediction on a new site)."""
        self.pts, self.S = pts.reset_index(drop=True), store
        self.meta = {}
        for pid, site in zip(self.pts.point_id, self.pts.site):
            z = np.load(tile_path(site, pid), allow_pickle=True)
            self.meta[pid] = (np.array([int(d[:4]) for d in z["dates"]]), point_bad(site, z), list(z["quality"]))

    @torch.no_grad()
    def pair_proba_point(self, pid, pairs, tta=8):
        """Probabilities (n_pairs,4) for the given (i,j) pairs of one point, averaged over nets and views."""
        ts = sorted({t for p in pairs for t in p})
        out = 0
        for v in range(tta):
            F_, M_ = {}, {}
            for t in ts:
                f, m = self.S.get(pid, t, v); F_[t] = torch.from_numpy(f.copy()); M_[t] = torch.from_numpy(m)
            fb = torch.stack([F_[i] for i, _ in pairs]).to(self.dev); fa = torch.stack([F_[j] for _, j in pairs]).to(self.dev)
            m = torch.stack([torch.minimum(M_[i], M_[j]) for i, j in pairs]).to(self.dev)
            for net in self.nets:
                net.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = out + torch.softmax(net(fb, fa, m).float(), -1)
        return (out / (tta * len(self.nets))).cpu().numpy()

    def predict(self, idx):
        Ps, owner = [], []
        for n, k in enumerate(idx):
            pid = self.pts.point_id.iloc[k]; years, bad, q = self.meta[pid]
            vi, _ = valid_dates(bad, q)
            pr = infer_pairs(pid, vi, years)
            Ps.append(self.pair_proba_point(pid, pr, self.cfg.get("tta", 8))); owner += [n] * len(pr)
        return aggregate(np.concatenate(Ps), np.array(owner), len(idx), mode=self.cfg.get("agg", "max"),
                         k=self.cfg.get("k", 2))
