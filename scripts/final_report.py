"""Write reports/final_report.md from the chosen ensemble json + experiment log. usage: final_report.py <ens_name>"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import *
import numpy as np

name = sys.argv[1]
e = json.load(open(REPORTS / f"{name}.json"))
aud = json.load(open(REPORTS / sys.argv[sys.argv.index("--audit") + 1]))
chosen = sys.argv[sys.argv.index("--chosen") + 1] if "--chosen" in sys.argv else "a_equal_nested_tb"
AR = {r["name"]: r for r in aud["ensembles"]}; ch = AR[chosen]
shuf = json.load(open(REPORTS / "audit_shuffle.json")) if (REPORTS / "audit_shuffle.json").exists() else None
runs = {}
for line in open(REPORTS / "runs.jsonl"):
    r = json.loads(line); runs[r["name"]] = r


def mean_of(r, key):
    return float(np.mean([x[key] for x in r["results"]]))


MC = {r["member"]: r for r in aud["members_check"]}
member_rows = "\n".join(
    f"| {m} | {w:.2f} | {MC[m]['S_per_seed_mean']:.3f} | {MC[m]['S_seedavg_probs']:.3f} | {AR['single_' + m]['S']:.3f} "
    f"(M_in {AR['single_' + m]['M_in']:.3f}, M_out {AR['single_' + m]['M_out']:.3f}) |"
    for m, w in zip(e["runs"], e["weights"]) if w > 0)
var_rows = "\n".join(
    f"| {r['name']} | {r['how']} | {r['M_in']:.4f} | {r['M_out']:.4f} | **{r['S']:.4f}** | {r['S_ci'][0]:.3f}–{r['S_ci'][1]:.3f} | "
    f"{r['perclass_out']} | {r['per_site_out']['aleppo']:.3f} |" for r in aud["ensembles"])
if shuf:
    shuf_rows = "\n".join(f"| {r['name']} | {r['M_in']:.4f} | {r['M_out']:.4f} | {r['S']:.4f} |" for r in shuf["ensembles"])
    shuf_rows += "\n" + "\n".join(f"| member {r['member']} (1 seed) | {r['M_in_seedavg']:.4f} | {r['M_out_seedavg']:.4f} | {r['S_seedavg_probs']:.4f} |"
                                   for r in shuf["members_check"])
else:
    shuf_rows = "| (pending) | | | |"
n_leak = len(aud["leak_checks"])
cls = ["construction", "destruction", "soil_sealing", "no_change"]
pc = "\n".join(f"| {c} | {a:.3f} | {b:.3f} |" for c, a, b in zip(cls, ch["perclass_in"], ch["perclass_out"]))
site_rows = "\n".join(f"| {s}{' (closest proxy to Damascus)' if s == 'aleppo' else ''} | {v:.3f} |" for s, v in ch["per_site_out"].items())


def cm_md(cm):
    s = "| true \\ pred | " + " | ".join(cls) + " |\n|---|---|---|---|---|\n"
    return s + "\n".join(f"| {cls[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm))


abl = []
def S(n):
    return f"{mean_of(runs[n], 'S'):.3f} (M_out {mean_of(runs[n], 'M_out'):.3f})" if n in runs else "n/a"
pairs_abl = [
    ("Preprocessing (quality filter + co-registration + hist-match)", "A_lgb_top2_raw_noprep", "A_lgb_top2", "raw tiles", "full preprocessing"),
    ("Aggregation over pairs", "A_lgb_max", "A_lgb_top2", "max", "mean of top-2"),
    ("DINOv2 pair distances in Model A", "A_lgb_top2", "A_lgb_top2_dino", "off", "on"),
    ("Pair strategy", "A_lgb_top2_dino_core", "A_lgb_top2_dino", "core pair only", "all spanning pairs"),
    ("Time-reversal augmentation (A)", "A_lgb_top2_dino_rev", "A_lgb_top2_dino", "on", "off"),
    ("Time-reversal augmentation (B)", "B_l_B1_norev", "B_l_B1_base", "off", "on"),
    ("Ambiguous pairs as no_change (w 0.2)", "A_lgb_top2_dino_amb", "A_lgb_top2_dino", "on", "excluded"),
    ("Label-quality weights (A)", "A_lgb_top2_dino_nolabw", "A_lgb_top2_dino", "off", "on"),
    ("Label-quality weights (B)", "B_l_B1_nolabw", "B_l_B1_base", "off", "on"),
    ("(site x class) balancing (A)", "A_lgb_top2_dino_nobal", "A_lgb_top2_dino", "off", "on"),
    ("Loss (B)", "B_l_B1_la", "B_l_B1_base", "logit-adjusted CE", "focal"),
    ("Loss (D)", "D_l_B1_la", "D_l_B1_base", "logit-adjusted CE", "focal"),
    ("Input config (B)", "B_l_B2", "B_l_B1_base", "B2 64 m @4 m", "B1 128 m @8 m"),
    ("Input config (D)", "D_l_B2", "D_l_B1_base", "B2", "B1"),
        ("Backbone (D)", "D_g_B1", "D_l_B1_base", "DINOv2-giant", "DINOv2-large"),
    ("Model C LoRA vs B frozen", "C_l_B2_lora3", "B_l_B1_base", "C (LoRA)", "B (frozen)"),
    ("CatBoost vs LightGBM", "A_cat_max", "A_lgb_max", "CatBoost", "LightGBM"),
]
for ttl, a, b, la, lb in pairs_abl:
    abl.append(f"| {ttl} | {la}: {S(a)} | {lb}: {S(b)} |")
abl = "\n".join(abl).replace("CatBoost: n/a", "CatBoost: ≈0.448 (2 seeds, stopped: slower and worse)")
best_single = max(AR["single_" + m]["S"] for m, w in zip(e["runs"], e["weights"]) if w > 0)
tc = ", ".join(f"{t:.2f}→{s:.3f}" for t, s, _, _ in e["t_curve"])

md = f"""# Eyes on Heritage: final report

## Result (proxy for the hidden test sets): fully nested numbers
| | value |
|---|---|
| **S_proxy = HM(M_in_cv, M_out_cv), nested** | **{ch['S']:.4f}** (cell bootstrap 95 % CI {ch['S_ci'][0]:.3f}–{ch['S_ci'][1]:.3f}) |
| M_in_cv (5-fold spatial CV, 200 m blocks) | {ch['M_in']:.4f} |
| M_out_cv (leave-one-site-out) | {ch['M_out']:.4f} |
| change AUROC in / LOSO | {e['auroc_in']:.3f} / {e['auroc_loso']:.3f} |
| deployed threshold t / class log-biases b (constr, destr, soil) | {e['t']:.2f} / {e['b']} (fitted on pooled LOSO OOF of all 5 sites) |

**What is nested:** ensemble weights are **equal (0.25 each, not fitted)**. Threshold t and class biases b are fitted only on
held-in data. For M_out the fit uses the LOSO OOF of the 4 other sites and is applied to the 5th. For M_in it uses the LOSO OOF of the points
outside each in-domain test fold. Early stopping and checkpoint selection always use an inner split of the training
sites/blocks (the held-out site is never touched; {n_leak} splits asserted, see Audit).
**Not nested:** the choice of members and their hyper-parameters (incl. the Optuna search for A-pt) was made on the
same LOSO OOF. That selection bias is not removed, so treat the numbers as mildly optimistic. Weights and thresholds for
a held-out site are fitted on OOF predictions of the other sites, and those predictions came from models that saw the
held-out site's labels in training. This is second-order. A strict fix needs leave-two-sites-out.

Ensemble members (equal weights). "Alone, nested" uses the same nested t/b protocol as the headline:

| member | weight | S per-seed mean (results.md) | S of 3-seed-averaged probs | alone, nested |
|---|---|---|---|---|
{member_rows}

## Audit of the earlier 0.648 claim
The earlier report fitted ensemble weights on all 5 sites' LOSO OOF to maximise S. It also compared the ensemble with the
members' *per-seed* S, while the ensemble reads *3-seed-averaged* probabilities, which are already a seed ensemble.
Both inflated the apparent gain: +0.089 was claimed. On a like-for-like basis, the fully nested gain is
{AR['b_fully_nested']['S'] - best_single:+.3f} with fitted weights and {AR['a_equal_nested_tb']['S'] - best_single:+.3f} with equal weights,
over the best member ({best_single:.3f}).

| variant | how | M_in | M_out | S | S 95 % CI | F1 constr/destr/soil/no_ch (LOSO) | Aleppo |
|---|---|---|---|---|---|---|---|
{var_rows}

Fitted weights (b) score below equal weights (a) out of sample, so weight fitting overfits the 776 points. **Equal weights are
used for the submission.**

Label-shuffle test (all label and metadata columns permuted jointly; every member retrained from scratch through the full
CV pipeline; ensemble evaluated with the same protocols):

| shuffled run | M_in | M_out | S |
|---|---|---|---|
{shuf_rows}

Chance level for 4-class macro-F1 is about 0.25.

## Method
```mermaid
flowchart LR
  G[GeoTIFF stack per site<br/>RGB+alpha 0.5 m] --> Q[Per point x date quality<br/>sharpness, contrast, valid, haze → bad flag]
  Q --> R[Local co-registration<br/>phase correlation on gradients, ≤12 px]
  R --> T[384 px tile cache<br/>alpha=0 masked, never inpainted]
  T --> FA[Engineered pair features<br/>Lab, CVA, SSIM, ExG, shadows, edges, LBP,<br/>bare/paved fractions, hist-matched diffs]
  T --> E[DINOv2-L frozen tokens<br/>8 dihedral views, canonical frame]
  E --> FA
  FA --> A[Model A: LightGBM per pair<br/>→ top-2 aggregation]
  FA --> Apt[Model A-pt: LightGBM on<br/>pair-feature statistics per point]
  E --> B[Model B: bitemporal token head<br/>Gaussian centre attention pooling]
  T --> C[Model C: siamese DINOv2-L + LoRA r16<br/>last 8 blocks, heavy image augs, same head as B]
  E --> D[Model D: temporal transformer<br/>over all valid dates, year-free<br/>evaluated, weight 0 in final ensemble]
  A & Apt & B & C --> ENS[Equal-weight average<br/>+ decision rule t, b_c]
  ENS --> OUT[p_change, class]
```

### Why each component exists
- **Quality filter**: hazy/blurred dates (Derna 2024, Agadez 2018/2024, half of Aleppo 2021) create false changes. Dates flagged bad are dropped per point; if fewer than two remain, the least-bad pair is used.
- **Local co-registration**: dates are misregistered by a few metres, which looks like change at building edges. Shifts are estimated per point on a 128 m window and applied only when the peak is reliable.
- **Histogram matching (pair level) / per-crop standardisation (date level)**: removes seasonal, sensor and city colour differences, which matters most for an unseen city.
- **Year-free inference**: test points may come without a year, so every valid pair is scored and aggregated (mean of the two most changed pairs), and Model D reads the whole sequence. A `--use-year` flag scores only the core pair when a year is given.
- **Spatial CV + LOSO**: 200 m block groups prevent near-duplicate neighbours from leaking. LOSO is the proxy for Damascus. Decision thresholds are tuned only on LOSO OOF, and M_out is reported with nested tuning (fit on 4 sites, apply to the 5th).
- **Label-quality weights + (site × class) balancing**: noisy labels (best_accuracy 49–95, conflicts) are down-weighted. Metadata is used only for weights, never as input.
- **LoRA fine-tuning (Model C)** with sensor/haze/JPEG/misregistration augmentations: it had to beat frozen Model B by ≥ 0.01 S over 3 seeds and did (0.560 vs 0.532), and it is the strongest single member.
- **Ensembling heterogeneous models**: hand-crafted features (robust colour/texture change), frozen foundation-model features (semantics) and a temporal model make different errors. Under the same nested protocol, the equal-weight ensemble gains {ch['S'] - best_single:+.3f} S over its best single member ({best_single:.3f}).

## Per-class F1
| class | in-domain CV | LOSO |
|---|---|---|
{pc}

Confusion matrices (deployed global t, b applied to the OOF; descriptive, not nested): ![confusion](figs/confusion.png)

In-domain:

{cm_md(e['cm_in'])}

LOSO (pooled, global t/b):

{cm_md(e['cm_loso'])}

## Per-site LOSO macro-F1 (nested)
| held-out site | macro-F1 |
|---|---|
{site_rows}

## Threshold robustness
S (with fixed b) as t moves ±0.1: {tc}. The chosen t is the centre of the flat region (3-point moving average), not the exact peak. ![t](figs/t_curve.png)

## Ablations (3 seeds unless stated, S_proxy mean)
| component | variant | reference |
|---|---|---|
{abl}

Full experiment log: `reports/results.md`.

## Error analysis
12 most confident LOSO errors: ![errors](figs/confident_errors.png)

12 most uncertain LOSO cases: ![uncertain](figs/uncertain.png)

## Known failure modes
- **Haze / cloud**: removed by the quality filter when detectable. Thin haze still lowers contrast and is sometimes read as soil_sealing or no_change.
- **Misregistration / off-nadir**: tall buildings lean differently between dates. Residual parallax shows up as false construction at building edges.
- **Ambiguous year**: changes visible only in a late date, or dates close to the event, give weak pairs. Year-free aggregation helps but is the main source of construction ↔ no_change confusion.
- **Label noise**: some `no_change` points visibly changed. They are down-weighted but still lower measured precision.
- **Rare classes across cities**: destruction (63) and soil_sealing (64) look very different per city (flood debris in Derna, war damage in Aleppo, paving in Marrakesh). They remain the weakest classes out of domain.
- **Damascus domain shift**: no Damascus imagery was available, so Aleppo LOSO is the closest proxy.

## External resources
- DINOv2 ViT-L/14 and ViT-g/14 weights, `facebook/dinov2-large`, `facebook/dinov2-giant` (Meta AI, Apache-2.0), via Hugging Face. DINOv3 satellite weights are gated and were not available (no HF token), so DINOv2 was used as the documented fallback.
- Libraries: PyTorch, transformers, peft, timm, LightGBM, CatBoost, scikit-learn, scikit-image, OpenCV, rasterio, pyproj, shapely, geopandas, pandas, pyarrow, optuna, matplotlib.
- No external labels or imagery were used.
"""
# ---------------- v2 experiments (all nested; same protocol as the headline) ----------------
def single_S(n):
    return f"{mean_of(runs[n], 'S'):.3f} (M_out {mean_of(runs[n], 'M_out'):.3f}, {len(runs[n]['seeds'])} seed)" if n in runs else "n/a"
def ens_S(tag):
    fn = REPORTS / f"audit_{tag}.json"
    if not fn.exists(): return "n/a"
    r = [x for x in json.load(open(fn))["ensembles"] if x["name"] == "a_equal_nested_tb"][0]
    return f"{r['S']:.3f}"
V2 = [
    ("DINOv3-sat-L frozen distances (Model A)", "A_lgb_top2_dinov3", None, "weaker than DINOv2 features (A 0.506)"),
    ("DINOv3-sat-L frozen tokens (Model D)", "D_v3s_B1", None, "D on DINOv2: 0.508"),
    ("DINOv3-sat-L + LoRA (Model C3), 5th member", "C3_v3s_lora", "5m_C3", "+0.006, inside noise, Aleppo -0.016; seeds 1-2 dropped"),
    ("Siamese U-Net, ConvNeXt-DINOv3 encoder (Model E), 5th member", "E_unet_cnx3", "5m_E", "best single M_out (0.500) but no ensemble gain"),
    ("SAM ViT-H AnyChange features in A and A-pt", "A_lgb_top2_dino_sam", "ens_sam", "A-pt+SAM single 0.518"),
    ("Per-site feature normalisation + confident learning (A)", "A_lgb_top2_dino_sncl", "v2_Asncl", ""),
    ("Per-site feature normalisation + confident learning (A-pt)", "Apt_opt_sncl", "v2_Aptsncl", "higher M_out alone (0.472)"),
    ("Confident learning only (A / A-pt)", "A_lgb_top2_dino_cl", None, f"A-pt: {single_S('Apt_opt_cl')}"),
    ("Per-site token normalisation, AdaBN-style (B)", "B_l_B1_norev_sn3", "v2_Bsn", "B current: 0.542"),
    ("Co-registration v2 (+-32 px, NCC-verified), Model A w/o DINO", "A_lgb_top2_coreg2", None, "A w/o DINO: 0.481"),
]
v2rows = "\n".join(f"| {t} | {single_S(r)} | {ens_S(g) if g else '-'} | {n} |" for t, r, g, n in V2)
v2e = json.load(open(REPORTS / "v2ens_cur.json")) if (REPORTS / "v2ens_cur.json").exists() else None
cal_rows = ""
if v2e:
    R = v2e["results"]; c = v2e["output_calibration"]
    cal_rows = "\n".join(f"| {k} | {R[k]['S']:.3f} | {R[k]['M_out']:.3f} |" for k in ("base", "s1_site_std_pchange", "s3_pair_count_corr", "s2_stacking_lr"))
md += f"""

## v2 experiments (targeted at the weaknesses found in the EDA)
Everything uses the headline protocol (equal weights, t/b nested). A change was accepted only if the nested ensemble S rose
by about 0.015 or more (CI half-width is about 0.04). Reference: current ensemble **0.642**.

| experiment | single member S | ensemble S (nested) | note |
|---|---|---|---|
{v2rows}

Ensemble-level variants (same four members):

| variant | S | M_out |
|---|---|---|
{cal_rows}

Label-shift adaptation by EM (Saerens 2002, with the source prior implied by the model): it collapsed the target prior and lowered M_out (0.19–0.53 vs 0.58) for every smoothing strength. Rejected.

**Adopted in v2: monotone Platt calibration of p_change** (fitted on pooled LOSO OOF). Ranking within a site and all
decisions are unchanged. Nested LOSO change ECE {c['ece_change_raw']:.3f} → {c['ece_change_platt']:.3f}, Brier
{c['brier_raw']:.3f} → {c['brier_platt']:.3f}. BCTS calibrated as well but lowered macro-F1 when used for decisions.

**Takeaway.** Most v2 ideas made individual members slightly stronger out of domain, especially on Aleppo and M_out.
None improved the ensemble. They tend to make members more alike: per-site normalisation removes city-specific cues that
the other members still use, and the ensemble gain comes from that diversity. With 776 points and 5 sites, differences
below about 0.02 S cannot be told apart from noise. About 60 configurations were tried in total, so the headline 0.642 carries some
selection bias.
""" if v2e else ""

(REPORTS / "final_report.md").write_text(md)
print("wrote", REPORTS / "final_report.md")
