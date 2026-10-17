"""Model T (v4): point-level tabular model. Features per point: max/mean over all valid pairs and the first/last pair of
the engineered difference + pairwise features, plus PCA (fitted inside the fold) of DINOv3 centre-embedding differences
between dates. Classifier: TabPFN v2, averaged with a LightGBM + XGBoost + CatBoost ensemble."""
import numpy as np
from common import *
from pairs import valid_dates, all_pairs, point_weights


class PointFeats:
    def __init__(self, pair_table, emb_dir=None):
        T = pair_table; self.T = T; pts = T.pts
        nU = len(T.ucols); sel = np.r_[np.arange(2 * nU, 3 * nU), np.arange(3 * nU, 3 * nU + len(T.pcols))]
        X, E = [], []
        cen = row = None
        if emb_dir is not None:
            import pandas as pd
            cen = np.load(Path(emb_dir) / "center.npy", mmap_mode="r")
            idx = pd.read_parquet(Path(emb_dir) / "index.parquet")
            row = {(p, t): k for k, (p, t) in enumerate(zip(idx.point_id, idx.t))}
        for pid in pts.point_id:
            years, bad, q = T.meta[pid]
            vi, _ = valid_dates(bad, q); pr = all_pairs(list(vi))
            V = np.stack([T.vec(pid, i, j)[sel] for i, j in pr])
            fl = T.vec(pid, int(vi[0]), int(vi[-1]))[sel]
            X.append(np.concatenate([V.max(0), V.mean(0), fl, [len(vi)]]).astype(np.float32))
            if cen is not None:
                D = cen.shape[-1] // 2
                e = np.stack([np.asarray(cen[row[(pid, int(t))]], np.float32).mean(0)[:D] for t in vi])
                d = e[1:] - e[:-1]
                E.append(np.concatenate([np.abs(d).max(0), e[-1] - e[0]]))
        self.X = np.stack(X)
        self.E = np.stack(E) if E else None


class ModelT:
    def __init__(self, feats, cfg):
        self.F, self.cfg = feats, cfg
        self.pts = feats.T.pts

    def _design(self, tr, idx):
        X = self.F.X[idx]
        if self.F.E is not None and self.cfg.get("pca", 16) > 0:
            from sklearn.decomposition import PCA
            if not hasattr(self, "pca"):
                self.pca = PCA(self.cfg.get("pca", 16), random_state=0).fit(self.F.E[tr])
            X = np.c_[X, self.pca.transform(self.F.E[idx])]
        return X

    def fit(self, tr, va, seed=0, all_tr=None):
        cfg = self.cfg; y = self.pts.y.to_numpy()
        at = np.asarray(all_tr if all_tr is not None else np.r_[tr, va])
        X = self._design(at, at); self.models = []
        w = point_weights(self.pts, cfg.get("label_w", True), cfg.get("balance", True))[at]
        cw = np.array([1 / np.sqrt(max((y[at] == c).sum(), 1)) for c in range(4)]); w = w * cw[y[at]]; w = w / w.mean()
        if cfg.get("tabpfn", True):
            from tabpfn import TabPFNClassifier
            m = TabPFNClassifier(device="cuda", n_estimators=cfg.get("pfn_estimators", 8), random_state=seed,
                                 ignore_pretraining_limits=True)
            m.fit(X, y[at]); self.models.append(("pfn", m, cfg.get("w_pfn", 0.5)))
        if cfg.get("gbdt", True):
            import lightgbm as lgb
            if cfg.get("xgb", False): import xgboost as xgb
            from catboost import CatBoostClassifier
            g = []
            g.append(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                        random_state=seed, verbose=-1, n_jobs=3).fit(X, y[at], sample_weight=w))
            if cfg.get("xgb", False):   # xgboost's CUDA probe fails on this driver; off by default
                g.append(xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.5,
                                           reg_lambda=5.0, random_state=seed, n_jobs=3, tree_method="hist", device="cpu").fit(X, y[at], sample_weight=w))
            g.append(CatBoostClassifier(iterations=500, learning_rate=0.05, depth=5, l2_leaf_reg=8, random_seed=seed, verbose=0,
                                        thread_count=3, allow_writing_files=False).fit(X, y[at], sample_weight=w))
            self.models.append(("gbdt", g, 1 - cfg.get("w_pfn", 0.5) if cfg.get("tabpfn", True) else 1.0))
        return dict(n_train=int(len(at)))

    def predict(self, idx):
        X = self._design(None, idx); P = 0; W = 0
        for kind, m, w in self.models:
            p = m.predict_proba(X) if kind == "pfn" else np.mean([g.predict_proba(X) for g in m], 0)
            P = P + w * p; W += w
        return P / W
