"""Validation driver: in-domain spatial CV + leave-one-site-out, metrics, results logging.
A model is any object with fit(tr_idx, va_idx, seed) and predict(idx) -> (n,4) point probabilities."""
import json, time, datetime
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from common import *
from pairs import make_folds, tune_decision, decide


def inner_split(pts, tr, protocol, seed):
    """Validation subset of tr for early stopping. LOSO: one other site; in-domain: ~20% of blocks."""
    rng = np.random.RandomState(seed)
    if protocol == "loso":
        sites = sorted(pts.site.iloc[tr].unique())
        vs = sites[rng.randint(len(sites))]
        m = pts.site.iloc[tr].to_numpy() == vs
    else:
        blocks = np.array(pts.block_id.iloc[tr].unique()); rng.shuffle(blocks)
        vb = set(blocks[: max(1, len(blocks) // 5)])
        m = pts.block_id.iloc[tr].isin(vb).to_numpy()
    return tr[~m], tr[m]


def assert_split(pts, te, tr, tr_in, va, protocol):
    """Leak assertions: held-out points never in train or early-stopping sets; LOSO: no held-out-site point at all;
    in-domain: no 200 m block shared between test and train/val. Returns a description of the selection source."""
    te_s, tr_s, ti_s, va_s = set(te), set(tr), set(tr_in), set(va)
    assert not (te_s & tr_s), "test/train overlap"
    assert ti_s | va_s == tr_s and not (ti_s & va_s), "inner split is not a partition of train"
    pid = pts.point_id.to_numpy()
    assert not (set(pid[te]) & set(pid[tr])), "point_id overlap"
    if protocol == "loso":
        held = set(pts.site.iloc[te]); assert len(held) == 1
        assert not (held & set(pts.site.iloc[tr])), "held-out site in train/val"
        return f"val site={sorted(set(pts.site.iloc[va]))[0]} (held-out={sorted(held)[0]})"
    bt = set(pts.block_id.iloc[te])
    assert not (bt & set(pts.block_id.iloc[tr])), "block shared between test and train/val"
    return f"val=20% of training blocks ({len(set(pts.block_id.iloc[va]))} blocks)"


PAIRS_OOF = {}


def run_oof(make_model, pts, protocol, seed, folds=None, verbose=True):
    folds = folds or make_folds(pts, 0)
    f = folds[protocol]
    oof = np.zeros((len(pts), 4)); extra = {}
    for k in np.unique(f):
        te = np.where(f == k)[0]; tr = np.where(f != k)[0]
        tr_in, va = inner_split(pts, tr, protocol, seed + k)
        src = assert_split(pts, te, tr, tr_in, va, protocol)
        m = make_model()
        t0 = time.time()
        info = m.fit(tr_in, va, seed=seed + 100 * k, all_tr=tr)
        oof[te] = m.predict(te)
        if hasattr(m, "last_pairs"):        # pair-level probabilities for the change-point aggregator
            PAIRS_OOF.setdefault(protocol, {}).update(m.last_pairs)
        info = dict(info or {}); info["select_src"] = src
        extra[int(k)] = info
        if verbose:
            print(f"  [{protocol} fold {k}] n_tr={len(tr)} n_te={len(te)} {time.time()-t0:.0f}s "
                  f"argmaxF1={macro_f1(pts.y.iloc[te], oof[te].argmax(1)):.3f}", flush=True)
    return oof, extra


def nested_loso_decision(P, y, site):
    """Honest LOSO score: tune (t,b) on 4 sites' OOF, apply to the 5th."""
    pred = np.zeros(len(y), int)
    for s in np.unique(site):
        m = site == s
        t, b, _ = tune_decision(P[~m], y[~m])
        pred[m] = decide(P[m], t, b)
    return pred


def metrics(P, y, pred):
    yb = (y != NC).astype(int); pc = 1 - P[:, NC]
    out = dict(macroF1=macro_f1(y, pred),
               perclass=f1_score(y, pred, labels=range(4), average=None, zero_division=0).round(3).tolist(),
               auroc=float(roc_auc_score(yb, pc)) if 0 < yb.sum() < len(yb) else float("nan"),
               changeF1=float(f1_score(yb, (pred != NC).astype(int), zero_division=0)),
               cm=confusion_matrix(y, pred, labels=range(4)).tolist())
    return out


def evaluate(oof_in, oof_loso, pts, t=None, b=None):
    """Tune decision on pooled LOSO OOF (never in-domain), apply to both. Also nested-honest LOSO."""
    y = pts.y.to_numpy(); site = pts.site.to_numpy()
    if t is None:
        t, b, _ = tune_decision(oof_loso, y)
    p_in = decide(oof_in, t, b); p_lo = decide(oof_loso, t, b)
    p_lo_nested = nested_loso_decision(oof_loso, y, site)
    r = dict(t=float(t), b=np.round(b, 3).tolist())
    r["in"] = metrics(oof_in, y, p_in)
    r["loso"] = metrics(oof_loso, y, p_lo)
    r["loso_nested"] = metrics(oof_loso, y, p_lo_nested)
    r["per_site"] = {s: float(macro_f1(y[site == s], p_lo_nested[site == s])) for s in SITES}
    r["M_in"] = r["in"]["macroF1"]; r["M_out"] = r["loso_nested"]["macroF1"]
    r["S"] = hm(r["M_in"], r["M_out"])
    r["argmax_in"] = macro_f1(y, oof_in.argmax(1)); r["argmax_loso"] = macro_f1(y, oof_loso.argmax(1))
    return r


def bootstrap_S(oof_in, oof_loso, pts, t, b, n=1000, seed=0):
    """Cell-level paired bootstrap of S_proxy (decision fixed)."""
    y = pts.y.to_numpy(); cells = pts.cell_id.to_numpy()
    p_in = decide(oof_in, t, b); p_lo = decide(oof_loso, t, b)
    uc = np.unique(cells); idx = {c: np.where(cells == c)[0] for c in uc}
    rng = np.random.RandomState(seed); S = []
    for _ in range(n):
        sel = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
        S.append(hm(macro_f1(y[sel], p_in[sel]), macro_f1(y[sel], p_lo[sel])))
    return float(np.mean(S)), np.percentile(S, [2.5, 97.5]).round(4).tolist()


def log_result(name, cfg, seeds, res_list, notes=""):
    """Append one row to reports/results.md and the full json to reports/runs.jsonl."""
    S = [r["S"] for r in res_list]; Mi = [r["M_in"] for r in res_list]; Mo = [r["M_out"] for r in res_list]
    pc_in = np.mean([r["in"]["perclass"] for r in res_list], 0).round(3).tolist()
    pc_lo = np.mean([r["loso_nested"]["perclass"] for r in res_list], 0).round(3).tolist()
    aleppo = np.mean([r["per_site"]["aleppo"] for r in res_list])
    auroc = np.mean([r["loso"]["auroc"] for r in res_list])
    fn = REPORTS / "results.md"
    if not fn.exists():
        fn.write_text("# Experiment log\n\nS = HM(M_in_cv, M_out_cv). M_out_cv uses nested LOSO decision tuning "
                      "(t, b tuned on 4 sites, applied to the 5th). Per-class F1 order: construction, destruction, "
                      "soil_sealing, no_change.\n\n| time | run | hash | seeds | M_in_cv | M_out_cv | S_proxy | "
                      "Aleppo | LOSO AUROC | F1/class in | F1/class LOSO | notes |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(fn, "a") as fh:
        fh.write(f"| {datetime.datetime.now():%m-%d %H:%M} | {name} | {cfg_hash(cfg)} | {len(seeds)} | "
                 f"{np.mean(Mi):.4f}±{np.std(Mi):.4f} | {np.mean(Mo):.4f}±{np.std(Mo):.4f} | "
                 f"**{np.mean(S):.4f}**±{np.std(S):.4f} | {aleppo:.3f} | {auroc:.3f} | {pc_in} | {pc_lo} | {notes} |\n")
    with open(REPORTS / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(dict(name=name, cfg=cfg, hash=cfg_hash(cfg), seeds=seeds, results=res_list,
                                 time=str(datetime.datetime.now())), default=float) + "\n")
    print(f"== {name}: M_in {np.mean(Mi):.4f} M_out {np.mean(Mo):.4f} S {np.mean(S):.4f}±{np.std(S):.4f} "
          f"aleppo {aleppo:.3f}", flush=True)


def run_experiment(name, cfg, make_model, pts, seeds=(0, 1, 2), notes="", save_oof=True):
    folds = make_folds(pts, 0)
    res, oofs = [], []
    PAIRS_OOF.clear()
    for s in seeds:
        oi, ei = run_oof(make_model, pts, "in", s, folds)
        ol, el = run_oof(make_model, pts, "loso", s, folds)
        r = evaluate(oi, ol, pts); r["fit_info"] = dict(inn=ei, loso=el); res.append(r); oofs.append((oi, ol))
        print(f" seed {s}: S={r['S']:.4f} M_in={r['M_in']:.4f} M_out={r['M_out']:.4f}", flush=True)
    log_result(name, cfg, list(seeds), res, notes)
    if save_oof:
        d = CACHE / "oof"; d.mkdir(exist_ok=True)
        if PAIRS_OOF:
            import pickle
            pickle.dump(PAIRS_OOF, open(d / f"{name}_pairs.pkl", "wb"))
        np.savez(d / f"{name}.npz", oof_in=np.stack([o[0] for o in oofs]), oof_loso=np.stack([o[1] for o in oofs]))
    return res, oofs
