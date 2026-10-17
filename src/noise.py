"""Confident-learning style label-noise filtering (Northcutt et al., JAIR 2021), applied inside a training fold only.
Out-of-fold probabilities on the training points come from an inner GroupKFold (200 m blocks) of a small point-level
LightGBM; a point is flagged when its given label has self-confidence below that class's average self-confidence
AND another class is predicted above *its* class threshold. At most `max_frac` of the training points are dropped."""
import numpy as np
from sklearn.model_selection import GroupKFold


def detect_noisy(X, y, tr, groups, w, seed=0, max_frac=0.08, n_splits=5):
    import lightgbm as lgb
    tr = np.asarray(tr); P = np.zeros((len(tr), 4))
    p = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=7, min_data_in_leaf=15,
             feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=seed, verbose=-1, num_threads=4)
    for a, b in GroupKFold(n_splits).split(tr, y[tr], groups[tr]):
        m = lgb.train(p, lgb.Dataset(X[tr[a]], y[tr[a]], weight=w[tr[a]]), 150)
        P[b] = m.predict(X[tr[b]])
    yt = y[tr]
    thr = np.array([P[yt == c, c].mean() if (yt == c).any() else 1.0 for c in range(4)])
    pg = P[np.arange(len(tr)), yt]
    other = P.copy(); other[np.arange(len(tr)), yt] = -1
    am = other.argmax(1)
    flag = (pg < thr[yt]) & (other[np.arange(len(tr)), am] >= thr[am])
    score = np.where(flag, thr[yt] - pg, -np.inf)
    k = min(int(flag.sum()), int(max_frac * len(tr)))
    drop = tr[np.argsort(-score)[:k]] if k > 0 else np.array([], int)
    return set(drop.tolist())
