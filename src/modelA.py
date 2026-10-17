"""Model A: engineered pair features + gradient boosting (LightGBM / CatBoost)."""
import numpy as np, pandas as pd
from common import *
from pairs import label_pairs, train_pairs_for, valid_dates, point_weights, aggregate, all_pairs, infer_pairs


class PairTable:
    """Holds pair features for all points, builds training/inference matrices."""

    def __init__(self, pts, unary, pairs, extra_pair=None):
        self.pts = pts.reset_index(drop=True)
        self.ucols = [c for c in unary.columns if c not in ("point_id", "t")]
        self.pcols = [c for c in pairs.columns if c not in ("point_id", "i", "j")]
        self.U = {(r.point_id, r.t): np.asarray(row, np.float32)
                  for r, row in zip(unary[["point_id", "t"]].itertuples(), unary[self.ucols].to_numpy(np.float32))}
        if extra_pair is not None:
            pairs = pairs.merge(extra_pair, on=["point_id", "i", "j"], how="left")
            self.pcols = [c for c in pairs.columns if c not in ("point_id", "i", "j")]
        self.sign = np.array([-1.0 if ("_dm_" in c) else 1.0 for c in self.pcols], np.float32)
        self.P = {(r.point_id, r.i, r.j): row for r, row in
                  zip(pairs[["point_id", "i", "j"]].itertuples(), pairs[self.pcols].to_numpy(np.float32))}
        self.meta = {}
        for pid in self.pts.point_id:
            site = self.pts.loc[self.pts.point_id == pid, "site"].iloc[0]
            z = np.load(tile_path(site, pid), allow_pickle=True)
            self.meta[pid] = (np.array([int(d[:4]) for d in z["dates"]]), point_bad(site, z), list(z["quality"]))
        self.names = ([f"b_{c}" for c in self.ucols] + [f"a_{c}" for c in self.ucols] +
                      [f"d_{c}" for c in self.ucols] + self.pcols)
        self.site_of = dict(zip(self.pts.point_id, self.pts.site)); self.sn = None

    def enable_site_norm(self):
        """Per-site z-scoring of every pair feature, with statistics from all valid pairs of that site's points
        (unlabelled imagery only; at test time computed on the new site's points)."""
        self.sn = None; stats = {}
        for s in self.pts.site.unique():
            idx = np.where(self.pts.site.to_numpy() == s)[0]
            X, _ = self.infer_set(idx)
            stats[s] = (X.mean(0), X.std(0) + 1e-6)
        self.sn = stats

    def vec(self, pid, i, j):
        v = self._vec(pid, i, j)
        if self.sn is not None:
            mu, sd = self.sn[self.site_of[pid]]; v = (v - mu) / sd
        return v

    def _vec(self, pid, i, j):
        ui, uj = self.U[(pid, i)], self.U[(pid, j)]
        if i < j:
            p = self.P[(pid, i, j)]
        else:
            p = self.P[(pid, j, i)] * self.sign
        return np.concatenate([ui, uj, uj - ui, p])

    def train_set(self, idx, cfg, w_point):
        X, y, w, g = [], [], [], []
        for k in idx:
            r = self.pts.iloc[k]; years, bad, q = self.meta[r.point_id]
            vi, _ = valid_dates(bad, q)
            lp = train_pairs_for(r, years, vi, ambiguous=cfg.get("ambiguous", "exclude"),
                             reverse=cfg.get("reverse", False))
            if cfg.get("pairs") == "core":
                if r.y == NC:   # consecutive valid dates only
                    cons = {(int(vi[a]), int(vi[a + 1])) for a in range(len(vi) - 1)}
                    lp = [l for l in lp if (l[0], l[1]) in cons]
                else:
                    lp = [l for l in lp if l[4]][:1]
            if not lp:
                continue
            tot = sum(l[3] * (2.0 if l[4] else 1.0) for l in lp)
            for i, j, lab, pw, core in lp:
                X.append(self.vec(r.point_id, i, j)); y.append(lab)
                w.append(w_point[k] * pw * (2.0 if core else 1.0) / tot); g.append(k)
        w = np.array(w); w = w / w.mean()
        return np.array(X), np.array(y), w, np.array(g)

    def infer_set(self, idx):
        X, owner = [], []
        for n, k in enumerate(idx):
            r = self.pts.iloc[k]; years, bad, q = self.meta[r.point_id]
            vi, _ = valid_dates(bad, q)
            for i, j in infer_pairs(r.point_id, vi, years):
                X.append(self.vec(r.point_id, i, j)); owner.append(n)
        return np.array(X), np.array(owner)


class ModelA:
    def __init__(self, table, cfg):
        self.T, self.cfg = table, cfg

    def _wp(self):
        return point_weights(self.T.pts, self.cfg.get("label_w", True), self.cfg.get("balance", True))

    def fit_final(self, tr, seed, n_iter):
        """Train on all `tr` points with a fixed number of boosting rounds per algorithm (dict)."""
        cfg = self.cfg; wp = self._wp()
        X, y, w, _ = self.T.train_set(tr, cfg, wp)
        cw = {c: 1 / np.sqrt(max((y == c).sum(), 1)) for c in range(4)}
        s = np.mean([cw[c] for c in y]); w = w * np.array([cw[c] for c in y]) / s
        self.models = []
        if "lgb" in n_iter:
            import lightgbm as lgb
            self.models.append(("lgb", lgb.train(self._lgb_params(seed), lgb.Dataset(X, y, weight=w), int(n_iter["lgb"]))))
        if "cat" in n_iter:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(loss_function="MultiClass", iterations=int(n_iter["cat"]), learning_rate=cfg.get("cat_lr", 0.05),
                                   depth=cfg.get("cat_depth", 6), l2_leaf_reg=cfg.get("cat_l2", 8), random_seed=seed,
                                   verbose=0, thread_count=cfg.get("threads", 10), allow_writing_files=False)
            m.fit(X, y, sample_weight=w); self.models.append(("cat", m))

    def _lgb_params(self, seed):
        cfg = self.cfg
        return dict(objective="multiclass", num_class=4, learning_rate=cfg.get("lr", 0.03),
                    num_leaves=cfg.get("leaves", 15), min_data_in_leaf=cfg.get("min_leaf", 40),
                    feature_fraction=cfg.get("ff", 0.5), bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=cfg.get("l2", 5.0), lambda_l1=cfg.get("l1", 0.5), max_depth=cfg.get("depth", 6),
                    seed=seed, verbose=-1, num_threads=cfg.get("threads", 10))

    def _filter_noise(self, tr, va, all_tr, seed):
        from noise import detect_noisy
        from modelApt import PointTable
        if getattr(self.T, "_ptX", None) is None:
            self.T._ptX = PointTable(self.T).X
        pts = self.T.pts; y = pts.y.to_numpy()
        g = pts.block_id.astype("category").cat.codes.to_numpy()
        base = np.asarray(all_tr if all_tr is not None else np.r_[tr, va])
        drop = detect_noisy(self.T._ptX, y, base, g, self._wp(), seed)
        keep = lambda ix: np.array([k for k in ix if k not in drop], int)
        return keep(tr), keep(va), (keep(all_tr) if all_tr is not None else None), len(drop)

    def fit(self, tr, va, seed=0, all_tr=None):
        cfg = self.cfg; wp = self._wp()
        n_drop = 0
        if cfg.get("cl"):
            tr, va, all_tr, n_drop = self._filter_noise(tr, va, all_tr, seed)
        Xtr, ytr, wtr, _ = self.T.train_set(tr, cfg, wp)
        Xva, yva, wva, _ = self.T.train_set(va, cfg, wp)
        # class weights: inverse sqrt class freq at pair level on top of point weights
        cw = {c: 1 / np.sqrt(max((ytr == c).sum(), 1)) for c in range(4)}
        s = np.mean([cw[c] for c in ytr]); wtr = wtr * np.array([cw[c] for c in ytr]) / s
        wva = wva * np.array([cw[c] for c in yva]) / s
        self.models = []
        algo = cfg.get("algo", "lgb")
        Xall, yall, wall, _ = self.T.train_set(all_tr if all_tr is not None else tr, cfg, wp)
        wall = wall * np.array([cw[c] for c in yall]) / s
        info = {}
        if algo in ("lgb", "both"):
            import lightgbm as lgb
            p = dict(objective="multiclass", num_class=4, learning_rate=cfg.get("lr", 0.03),
                     num_leaves=cfg.get("leaves", 15), min_data_in_leaf=cfg.get("min_leaf", 40),
                     feature_fraction=cfg.get("ff", 0.5), bagging_fraction=0.8, bagging_freq=1,
                     lambda_l2=cfg.get("l2", 5.0), lambda_l1=cfg.get("l1", 0.5), max_depth=cfg.get("depth", 6),
                     seed=seed, verbose=-1, num_threads=cfg.get("threads", 10))
            m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(Xva, yva, weight=wva)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            nb = max(50, int(m.best_iteration * 1.1))
            m2 = lgb.train(p, lgb.Dataset(Xall, yall, weight=wall), num_boost_round=nb)
            self.models.append(("lgb", m2)); info["lgb_iter"] = nb
        if algo in ("cat", "both"):
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(loss_function="MultiClass", iterations=3000, learning_rate=cfg.get("cat_lr", 0.05),
                                   depth=cfg.get("cat_depth", 6), l2_leaf_reg=cfg.get("cat_l2", 8),
                                   random_seed=seed, verbose=0, thread_count=cfg.get("threads", 10), allow_writing_files=False,
                                   od_type="Iter", od_wait=150)
            m.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva))
            nb = max(50, int(m.get_best_iteration() * 1.1))
            m2 = CatBoostClassifier(loss_function="MultiClass", iterations=nb, learning_rate=cfg.get("cat_lr", 0.05),
                                    depth=cfg.get("cat_depth", 6), l2_leaf_reg=cfg.get("cat_l2", 8),
                                    random_seed=seed, verbose=0, thread_count=cfg.get("threads", 10), allow_writing_files=False)
            m2.fit(Xall, yall, sample_weight=wall)
            self.models.append(("cat", m2)); info["cat_iter"] = nb
        info["n_noisy_dropped"] = n_drop
        return info

    def pair_proba(self, X):
        ps = []
        for kind, m in self.models:
            ps.append(m.predict(X) if kind == "lgb" else m.predict_proba(X))
        return np.mean(ps, 0)

    def predict(self, idx):
        X, owner = self.T.infer_set(idx)
        P = self.pair_proba(X)
        # pair-level outputs for the change-point aggregator
        self.last_pairs = getattr(self, "last_pairs", {})
        o = 0
        for n, k in enumerate(idx):
            r = self.T.pts.iloc[k]; years, bad, q = self.T.meta[r.point_id]
            vi, _ = valid_dates(bad, q); pr = infer_pairs(r.point_id, vi, years)
            self.last_pairs[r.point_id] = (pr, P[o:o + len(pr)]); o += len(pr)
        return aggregate(P, owner, len(idx), mode=self.cfg.get("agg", "max"), k=self.cfg.get("k", 2))
