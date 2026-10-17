#!/bin/bash
# Download the final model weights (GitHub release assets) into models/final/<member>/.
# usage: bash scripts/download_weights.sh [release_tag] [models_dir]
# Requires the GitHub CLI (`gh auth login`) because the repository is private.
set -e
TAG=${1:-v6.0}; OUT=${2:-models/final}; REPO=dangaura-tech/eyes-heritage-final
TMP=$(mktemp -d)
gh release download "$TAG" -R "$REPO" -D "$TMP"
mkdir -p "$OUT"
for f in "$TMP"/*__*; do
  b=$(basename "$f"); member=${b%%__*}; file=${b#*__}
  mkdir -p "$OUT/$member"; mv "$f" "$OUT/$member/$file"
done
for m in B6gs_gapscene C6_v2_ft C_v3lab_dinov3_s0 C_v2lab_s0 F6_satlas_unet; do cp "configs/$m.json" "$OUT/$m/cfg.json"; done
cp configs/ensemble.json "$OUT/ensemble.json"
( cd "$OUT" && sha256sum -c "$TMP/SHA256SUMS" 2>/dev/null || shasum -a 256 -c "$TMP/SHA256SUMS" ) && echo "checksums ok"
rm -rf "$TMP"
echo "weights in $OUT"
