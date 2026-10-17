"""Model A-pt: point-level gradient boosting on statistics of pair features over all valid pairs (year-free)."""
import numpy as np
from common import *
from pairs import valid_dates, point_weights, all_pairs


class PointTable:
    def __init__(self, pair_table):
        self.T = pair_table; pts = pair_table.pts
        X = []
        for pid in pts.point_id:
            years, bad, q = self.T.meta[pid]
            vi, _ = valid_dates(bad, q)
            pr = all_pairs(list(vi))
            V = np.stack([self.T.vec(pid, i, j) for i, j in pr])
            cons = np.stack([self.T.vec(pid, int(vi[a]), int(vi[a + 1])) for a in range(len(vi) - 1)])
            fl = self.T.vec(pid, int(vi[0]), int(vi[-1]))
            U = np.stack([self.T.U[(pid, int(t))] for t in vi])
            X.append(np.concatenate([V.max(0), V.mean(0), V.std(0), cons.max(0), fl, U.std(0),
                                     [len(vi)]]).astype(np.float32))
        self.X = np.stack(X)

    def site_normalise(self):
        """Per-site z-scoring of point features (statistics from the site's own unlabelled points)."""
        site = self.T.pts.site.to_numpy()
        for s in np.unique(site):
            m = site == s; mu = self.X[m].mean(0); sd = self.X[m].std(0) + 1e-6
            self.X[m] = (self.X[m] - mu) / sd


class ModelApt:
    def __init__(self, ptable, cfg):
        self.P, self.cfg = ptable, cfg
        self.pts = ptable.T.pts

    def fit(self, tr, va, seed=0, all_tr=None):
        import lightgbm as lgb
        cfg = self.cfg; y = self.pts.y.to_numpy(); n_drop = 0
        if cfg.get("cl"):
            from noise import detect_noisy
            g = self.pts.block_id.astype("category").cat.codes.to_numpy()
            wp0 = point_weights(self.pts, cfg.get("label_w", True), cfg.get("balance", True))
            base = np.asarray(all_tr if all_tr is not None else np.r_[tr, va])
            drop = detect_noisy(self.P.X, y, base, g, wp0, seed); n_drop = len(drop)
            keep = lambda ix: np.array([k for k in ix if k not in drop], int)
            tr, va = keep(tr), keep(va); all_tr = keep(all_tr) if all_tr is not None else None
        wp = point_weights(self.pts, cfg.get("label_w", True), cfg.get("balance", True))
        cw = np.array([1 / np.sqrt(max((y[tr] == c).sum(), 1)) for c in range(4)])
        w = wp * cw[y]; w = w / w[tr].mean()
        p = dict(objective="multiclass", num_class=4, learning_rate=cfg.get("lr", 0.03),
                 num_leaves=cfg.get("leaves", 7), min_data_in_leaf=cfg.get("min_leaf", 15),
                 feature_fraction=cfg.get("ff", 0.3), bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=cfg.get("l2", 10.0), lambda_l1=cfg.get("l1", 1.0), max_depth=cfg.get("depth", 4),
                 seed=seed, verbose=-1, num_threads=cfg.get("threads", 10))
        X = self.P.X
        m = lgb.train(p, lgb.Dataset(X[tr], y[tr], weight=w[tr]), 3000,
                      valid_sets=[lgb.Dataset(X[va], y[va], weight=w[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        at = all_tr if all_tr is not None else tr
        nb = max(50, int(m.best_iteration * 1.1))
        self.m = lgb.train(p, lgb.Dataset(X[at], y[at], weight=w[at]), nb)
        return dict(iter=nb, n_noisy_dropped=n_drop)

    def fit_final(self, tr, seed, n_iter):
        import lightgbm as lgb
        cfg = self.cfg; y = self.pts.y.to_numpy()
        wp = point_weights(self.pts, cfg.get("label_w", True), cfg.get("balance", True))
        cw = np.array([1 / np.sqrt(max((y[tr] == c).sum(), 1)) for c in range(4)])
        w = wp * cw[y]; w = w / w[tr].mean()
        p = dict(objective="multiclass", num_class=4, learning_rate=cfg.get("lr", 0.03),
                 num_leaves=cfg.get("leaves", 7), min_data_in_leaf=cfg.get("min_leaf", 15),
                 feature_fraction=cfg.get("ff", 0.3), bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=cfg.get("l2", 10.0), lambda_l1=cfg.get("l1", 1.0), max_depth=cfg.get("depth", 4),
                 seed=seed, verbose=-1, num_threads=cfg.get("threads", 10))
        self.m = lgb.train(p, lgb.Dataset(self.P.X[tr], y[tr], weight=w[tr]), int(n_iter))

    def predict(self, idx):
        return self.m.predict(self.P.X[idx])
