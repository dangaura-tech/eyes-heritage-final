# Eyes on Heritage: final model

Point-level change classification from multi-date 0.5 m RGB satellite imagery. For every point the model outputs
`p_change` and one class: `construction`, `destruction`, `soil_sealing` or `no_change`.

## Results (fully nested validation, 932 points)
S = harmonic mean of M_in (spatial 5-fold CV with 200 m blocks) and M_out (leave-one-site-out), macro-F1 over 4 classes.

| | this model | previous best |
|---|---|---|
| **S** | **0.670** (cell-bootstrap 95 % CI 0.629–0.706) | 0.637 |
| M_in | 0.717 | 0.692 |
| M_out (unseen city) | **0.628** | 0.590 |
| Aleppo held out | **0.771** | 0.684 |
| S on the organizers' points with their original labels | 0.671 | 0.632 |

Per-class F1, leave-one-site-out: construction 0.731, destruction 0.473, soil_sealing 0.506, no_change 0.801.

## Method
```
images + points
  └─ preprocessing: per point × date quality flags, local co-registration (phase correlation, ≤ 12 px), 192 m tiles
       ├─ B   frozen DINOv2-L tokens + bitemporal transformer head, trained on time-gap-balanced labelled pairs
       │      with scene-level exclusion of hazy / blurred dates
       ├─ C   siamese DINOv2-L, full fine-tuning (layer-wise LR decay), change gate + type heads, EMA
       ├─ C2  siamese DINOv2-L + LoRA (r 16)
       ├─ C3  siamese DINOv3-sat-L (SAT-493M) + LoRA (r 16)
       └─ F   siamese SatlasPretrain Swin-v2-B (aerial RGB) + U-Net decoder, full fine-tuning
  equal-weight average of the 5 members (5 seeds each, 8 dihedral views at test time)
  decision: p_change ≥ 0.62 → most likely change class, else no_change
  monotone Platt calibration of the reported p_change (LOSO ECE 0.15 → 0.03)
```
Every valid pair of dates is scored and the two most-changed pairs are averaged, so no change year is needed at test
time. Pixels with alpha = 0 (outside the area or withheld cells) are masked and never used. Annotation metadata is used
only for training weights, never as model input.

## Installation
Python 3.12, one CUDA GPU (about 12 GB is enough for inference).
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Weights
The trained weights (about 4 GB: 5 members × 5 seeds) are attached to the GitHub release `v6.0`. Download and unpack
them into `models/final/`:
```bash
gh auth login                      # once, the repository is private
bash scripts/download_weights.sh   # -> models/final/<member>/..., models/final/ensemble.json, checksums verified
```
The pretrained backbones (`facebook/dinov2-large`, `timm/vit_large_patch16_dinov3.sat493m`,
`torchgeo/swin_v2_b_naip_rgb_satlas`) are downloaded from the Hugging Face Hub on first use. Set `HF_HOME` to choose the
cache directory.

## Inference on the test set
```bash
export HERITAGE_ROOT=$PWD           # cache/, logs/ and work files are written here
python predict.py \
  --points /path/to/test_points.geojson \
  --imagery /path/to/test_site_imagery \
  --models models/final \
  --out submission.csv
```
* `--points`: GeoJSON or CSV with lon/lat (CRS84) and an id column (`--id-col`, default `id`).
* `--imagery`: folder or zip of `YYYY_YYYYMMDD.tif` GeoTIFFs on one grid (RGB + alpha). The UTM zone is read from
  the files, and any number of dates works. Points whose dates are all unusable fall back to the least-bad pair.
* Optional: `--site NAME`, `--work DIR` (tiles and embedding caches, ~10 GB per 150 points), `--tta 8` (lower is faster),
  `--use-year` (score only the pair around a given `year` column).
* Output columns: `id, p_change, pred_class, p_construction, p_destruction, p_soil_sealing, p_no_change`. They are set in
  the `OUTPUT` block at the top of `predict.py` to match the official submission format.
* Runtime: about 20 minutes for 140 points on one RTX 5000 Ada. Most of it is the fine-tuned members with 5 seeds ×
  8 views. `--tta 2` is roughly 4× faster.

Sanity check on a training site (in-sample): macro-F1 0.854, probabilities sum to 1, `p_change` spans 0.004–0.999.

## Reproduce training
`scripts/run.sh <log> <script> [args]` runs one step. Environment: `HERITAGE_ROOT` (working directory), `DATA_DIR`
(the unzipped training dataset), `BASE_CACHE` (optional, cache holding the original `points.parquet` and
`quality.parquet`, defaults to `$HERITAGE_ROOT/cache`).
```bash
python scripts/p0_audit.py                         # unzip imagery, asserts, points / cells / blocks
python scripts/p0_points_v2.py                     # reviewed labels (training_points_reviewed.geojson)
python scripts/p1_tiles.py                         # quality flags, co-registration, tiles
python scripts/check_holes_v2.py                   # no point inside a withheld cell
python scripts/p3_featA.py
python scripts/p4_embed.py dinov2-l B1             # frozen token caches (~60 GB each)
python scripts/p4_embed.py dinov3s-l B1
python scripts/p0_v6g.py --scene                   # time-gap-balanced labelled pairs + scene exclusions (for B)
python scripts/exp_B.py B6gs_gapscene '{"agg":"top2","reverse":false,"emb":"dinov2-l_B1"}' 0
python scripts/exp_C.py C6_v2_ft '{"full_ft":true,"lr_bb":3e-5,"llrd":0.8,"two_head":true,"tau":1.0,"ema":0.99,"epochs":40}' 0
python scripts/exp_C.py C_v2lab_s0 '{}' 0
python scripts/exp_C.py C_v3lab_dinov3_s0 '{"bb":"dinov3s-l"}' 0
python scripts/exp_C.py F6_satlas_unet '{"arch":"unet","enc":"satlas-swinv2b","lr_lora":5e-5,"lr":3e-4,"two_head":true,"tau":1.0,"ema":0.99,"epochs":40,"bs":16}' 0
M=B6gs_gapscene,C6_v2_ft,C_v3lab_dinov3_s0,C_v2lab_s0,F6_satlas_unet
python scripts/ensemble.py $M --weights eq --out ens_g1        # threshold on pooled LOSO OOF
python scripts/v2_ensemble.py $M --tag g1                      # p_change calibration
python scripts/audit_ensemble.py $M --tag g1                   # nested scores, leak assertions, bootstrap CI
python scripts/train_final.py reports/ens_g1.json              # all points, 5 seeds per member
```
Member and ensemble configurations are in `configs/`.

## Validation
* In-domain proxy: `StratifiedGroupKFold(5)` with 200 m block groups, stratified by site × class.
* Out-of-domain proxy: leave-one-site-out.
* Early stopping and checkpoint selection use the training sites only. Every split is asserted disjoint.
* Weights are equal, not fitted. The threshold is fitted only on held-in data, which makes the score nested.
* A label-shuffle test gives S ≈ 0.245, the chance level.

## External resources
* DINOv2 ViT-L/14 (`facebook/dinov2-large`, Apache-2.0)
* DINOv3 ViT-L/16 SAT-493M (`timm/vit_large_patch16_dinov3.sat493m`, DINOv3 licence)
* SatlasPretrain Swin-v2-B NAIP RGB (`torchgeo/swin_v2_b_naip_rgb_satlas`, Apache-2.0)
* AdaptFormer code (`src/third_party/adaptformer`, MIT, from `deepang/adaptformer-LEVIR-CD`), used in experiments only
* PyTorch, transformers, timm, peft, LightGBM, scikit-learn, scikit-image, OpenCV, rasterio, pyproj, shapely, pandas
