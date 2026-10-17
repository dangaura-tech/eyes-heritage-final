"""Audit of the ensemble: (1) leak assertions on every member/fold split, (2) ensemble recomputed as
(a) equal weights, (b) fully nested weights+t+b, (c) as reported, (4) member S recomputed from the same OOF files.
usage: audit_ensemble.py <member1,...> [--tag name]   (writes reports/audit_<tag>.json and .md)"""
import sys, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
from common import *
from cv import inner_split, assert_split, evaluate, metrics
from pairs import make_folds, tune_decision, decide
import numpy as np


# ---- fast macro-F1 (same definition as sklearn, labels 0..3, zero_division=0) used inside the tuners ----
def fast_f1(y, p, per_class=False):
    y = np.asarray(y); p = np.asarray(p)
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    return f if per_class else f.mean()


_yy = np.random.RandomState(0).randint(0, 4, 500); _pp = np.random.RandomState(1).randint(0, 4, 500)
assert abs(fast_f1(_yy, _pp) - common.macro_f1(_yy, _pp)) < 1e-12, "fast_f1 != sklearn"
common.macro_f1 = fast_f1
members = sys.argv[1].split(",")
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "ens"
pts = load_points(); y = pts.y.to_numpy(); site = pts.site.to_numpy(); cells = pts.cell_id.to_numpy()
folds = make_folds(pts, 0)
runs = {}
for line in open(REPORTS / "runs.jsonl"):
    r = json.loads(line); runs[r["name"]] = r
out = dict(members=members)

# ================= 1. leak assertions (splits are deterministic functions of (fold, seed)) =================
print("== 1. leak assertions", flush=True)
src_runs = {"C_l_B2_lora3": ["C_l_B2_lora", "C_l_B2_lora_s12"], "C_v2lab3": ["C_v2lab_s0", "C_v2lab_s12"],
            "C_v3lab_dinov3_3": ["C_v3lab_dinov3_s0", "C_v3lab_dinov3_s12"]}
leak_rows = []
for m in members:
    for rn in src_runs.get(m, [m]):
        r = runs[rn]
        for si, s in enumerate(r["seeds"]):
            fi = r["results"][si].get("fit_info", {})
            for proto, key in (("in", "inn"), ("loso", "loso")):
                f = folds[proto]
                for k in np.unique(f):
                    te = np.where(f == k)[0]; tr = np.where(f != k)[0]
                    tr_in, va = inner_split(pts, tr, proto, s + int(k))
                    src = assert_split(pts, te, tr, tr_in, va, proto)      # raises on any leak
                    info = fi.get(key, {}).get(str(int(k)), {})
                    sel = {kk: v for kk, v in info.items() if kk in ("best_ep", "lgb_iter", "cat_iter", "iter")}
                    leak_rows.append(dict(member=m, run=rn, seed=s, protocol=proto, fold=int(k),
                                          heldout=(SITES[int(k)] if proto == "loso" else f"in-fold {int(k)}"),
                                          n_train=len(tr_in), n_val=len(va), n_test=len(te), select_src=src,
                                          selected=sel))
print(f"  {len(leak_rows)} member x seed x fold splits checked: all disjoint", flush=True)
out["leak_checks"] = leak_rows

# ================= 4. member S from the same OOF files =================
print("== 4. member scores from OOF files", flush=True)
OI, OL, mem_rows = {}, {}, []
for m in members:
    z = np.load(CACHE / "oof" / f"{m}.npz")
    per_seed = [evaluate(z["oof_in"][i], z["oof_loso"][i], pts) for i in range(len(z["oof_in"]))]
    OI[m], OL[m] = z["oof_in"].mean(0), z["oof_loso"].mean(0)
    avg = evaluate(OI[m], OL[m], pts)
    logged = float(np.mean([x["S"] for x in runs[m]["results"]]))
    mem_rows.append(dict(member=m, n_seeds=len(per_seed), S_per_seed_mean=float(np.mean([x["S"] for x in per_seed])),
                         S_logged_results_md=logged, S_seedavg_probs=avg["S"], M_in_seedavg=avg["M_in"],
                         M_out_seedavg=avg["M_out"], aleppo_seedavg=avg["per_site"]["aleppo"],
                         perclass_loso_seedavg=avg["loso_nested"]["perclass"]))
    print(f"  {m:22s} per-seed mean S {mem_rows[-1]['S_per_seed_mean']:.4f} | results.md {logged:.4f} | "
          f"seed-averaged probs S {avg['S']:.4f}", flush=True)
out["members_check"] = mem_rows

# ================= 2. ensembles =================
grid = [np.array(w) for w in itertools.product(np.round(np.arange(0, 1.0001, 0.1), 1), repeat=len(members))
        if abs(sum(w) - 1) < 1e-6]
Lst = np.stack([OL[m] for m in members]); Ist = np.stack([OI[m] for m in members])
mix = lambda S, w: np.tensordot(w, S, 1)


def fit_layer(idx, weights=None):
    """Fit (w, t, b) on LOSO OOF of points idx only, objective = pooled macro-F1 after tuning t,b on idx."""
    best = None
    cands = [weights] if weights is not None else grid
    for w in cands:
        P = mix(Lst[:, idx], w)
        t, b, s = tune_decision(P, y[idx])
        if best is None or s > best[3]:
            best = (w, t, b, s)
    return best


def nested_eval(weights=None, name=""):
    t0 = time.time()
    pred_out = np.zeros(len(y), int); pred_in = np.zeros(len(y), int); fitted = []
    for s in SITES:                                         # M_out: fit on 4 sites, apply to the 5th
        te = site == s; w, t, b, _ = fit_layer(np.where(~te)[0], weights)
        pred_out[te] = decide(mix(Lst[:, te], w), t, b); fitted.append(("loso", s, w.tolist(), t, b.tolist()))
    fin = folds["in"]
    for k in np.unique(fin):                               # M_in: fit on LOSO OOF of points outside in-fold k
        te = fin == k; w, t, b, _ = fit_layer(np.where(~te)[0], weights)
        pred_in[te] = decide(mix(Ist[:, te], w), t, b); fitted.append(("in", int(k), w.tolist(), t, b.tolist()))
    print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)
    return pred_in, pred_out, fitted


HAS_ORIG = "y_orig" in pts.columns
ORG = (pts.source == "organizers").to_numpy() if HAS_ORIG else None
Y0 = pts.y_orig.to_numpy() if HAS_ORIG else None


def summarise(name, pred_in, pred_out, how, fitted=None):
    Mi, Mo = fast_f1(y, pred_in), fast_f1(y, pred_out)
    org = {}
    if HAS_ORIG:   # comparable with v1: organizer points only, scored against the original organizer labels
        mi0, mo0 = fast_f1(Y0[ORG], pred_in[ORG]), fast_f1(Y0[ORG], pred_out[ORG])
        org = dict(M_in_org=mi0, M_out_org=mo0, S_org=hm(mi0, mo0),
                   aleppo_org=float(fast_f1(Y0[ORG & (site == "aleppo")], pred_out[ORG & (site == "aleppo")])))
    uc = np.unique(cells); ix = {c: np.where(cells == c)[0] for c in uc}; rng = np.random.RandomState(0); B = []
    for _ in range(1000):
        sel = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])
        B.append(hm(fast_f1(y[sel], pred_in[sel]), fast_f1(y[sel], pred_out[sel])))
    return dict(name=name, how=how, M_in=Mi, M_out=Mo, S=hm(Mi, Mo), S_ci=np.percentile(B, [2.5, 97.5]).round(4).tolist(),
                perclass_in=fast_f1(y, pred_in, True).round(3).tolist(), perclass_out=fast_f1(y, pred_out, True).round(3).tolist(),
                per_site_out={s: float(fast_f1(y[site == s], pred_out[site == s])) for s in SITES}, fitted=fitted, **org)


print("== 2. ensembles", flush=True)
ens_rows = []
eq = np.ones(len(members)) / len(members)
# (a0) equal weights, nothing fitted at all (argmax of mean probabilities)
ens_rows.append(summarise("a0_equal_argmax", mix(Ist, eq).argmax(1), mix(Lst, eq).argmax(1),
                          "equal weights, argmax, no t/b"))
# (a) equal weights; t,b nested
pi, po, ft = nested_eval(eq, "a_equal_nested_tb")
ens_rows.append(summarise("a_equal_nested_tb", pi, po, "equal weights; t,b fitted on held-in sites/points only", ft))
# (b) fully nested weights + t + b
pi, po, ft = nested_eval(None, "b_fully_nested")
ens_rows.append(summarise("b_fully_nested", pi, po, "weights+t+b fitted on 4 sites (M_out) / on LOSO OOF outside the in-fold (M_in)", ft))
# (c) as reported (weights fitted on all LOSO OOF to maximise S; t chosen on all pooled LOSO; M_out with nested t,b only)
rep = json.load(open(REPORTS / f"{tag}.json")) if (REPORTS / f"{tag}.json").exists() else None
if rep and rep["runs"] == members:
    w = np.array(rep["weights"]); e = evaluate(mix(Ist, w), mix(Lst, w), pts, t=rep["t"], b=np.array(rep["b"]))
    from cv import nested_loso_decision
    po = nested_loso_decision(mix(Lst, w), y, site); pi = decide(mix(Ist, w), rep["t"], np.array(rep["b"]))
    ens_rows.append(summarise("c_as_reported", pi, po, "weights fitted on all 5 sites' LOSO OOF (S objective), t on pooled LOSO; only t,b nested for M_out",
                              [("all", w.tolist(), rep["t"], rep["b"])]))
# single members under the same nested-t,b protocol (fair comparison with (a)/(b))
for k, m in enumerate(members):
    w = np.eye(len(members))[k]
    pi, po, ft = nested_eval(w, m)
    ens_rows.append(summarise(f"single_{m}", pi, po, "single member (seed-averaged probs); t,b nested", ft))
out["ensembles"] = ens_rows
json.dump(out, open(REPORTS / f"audit_{tag}.json", "w"), indent=1, default=float)

# ---- markdown ----
L = ["| variant | M_in | M_out | S | S 95% CI (cell bootstrap) | F1 constr/destr/soil/no_ch (LOSO) | Aleppo | S organizer pts, original labels |",
     "|---|---|---|---|---|---|---|---|"]
for r in ens_rows:
    L.append(f"| {r['name']} | {r['M_in']:.4f} | {r['M_out']:.4f} | **{r['S']:.4f}** | {r['S_ci'][0]:.3f}–{r['S_ci'][1]:.3f} | "
             f"{r['perclass_out']} | {r['per_site_out']['aleppo']:.3f} | {r.get('S_org', float('nan')):.4f} |")
M = ["| member | seeds | S per-seed mean (recomputed) | S in results.md | S seed-averaged probs (what the ensemble reads) |", "|---|---|---|---|---|"]
for r in mem_rows:
    M.append(f"| {r['member']} | {r['n_seeds']} | {r['S_per_seed_mean']:.4f} | {r['S_logged_results_md']:.4f} | {r['S_seedavg_probs']:.4f} |")
sel_summary = {}
for r in leak_rows:
    sel_summary.setdefault((r["member"], r["protocol"]), set()).add(r["select_src"].split(" (")[0] if r["protocol"] == "loso" else "20% train blocks")
S = ["| member | protocol | early-stopping / checkpoint selection source |", "|---|---|---|"]
for (m, p), v in sel_summary.items():
    S.append(f"| {m} | {p} | {', '.join(sorted(v))} |")
(REPORTS / f"audit_{tag}.md").write_text("\n".join(["## Ensemble variants", *L, "", "## Members", *M, "",
                                                    f"## Leak checks: {len(leak_rows)} splits asserted disjoint", *S]) + "\n")
print((REPORTS / f"audit_{tag}.md").read_text())
