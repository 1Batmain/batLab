#!/usr/bin/env bash
# 8 seeds per arm from the checkpoints compare.sh produced, then the diversity
# measures of SCALE_UNET §4 (inter_seed_std / mean_pairwise_rmse /
# banding_ratio), plus the dataset reference.
#
# `--paths 1` is the honest setting: `paths=3` averages three trajectories and
# hides part of the collapse (SCALE_UNET §1.3 — 0.00329 with p=3 vs 0.00587 with
# p=1 on the same model). Both are produced so the numbers stay comparable with
# the earlier reports.
set -u
cd "$(dirname "$0")/../.."
BIN=./target/release/main
MODEL=${MODEL:-Greyscale_Diffusion_L}
OUT=${OUT:-weighting_samples}

for arm in uniform snr_g1; do
  ckpt="runs/${arm}.ckpt"
  [ -f "$ckpt" ] || { echo "missing $ckpt" >&2; exit 1; }
  for paths in 1 3; do
    mkdir -p "$OUT/${arm}_p${paths}"
    for s in 1 2 3 4 5 6 7 8; do
      $BIN --headless-sample "$MODEL" --checkpoint "$ckpt" \
           --seed "$s" --paths "$paths" --magnitude 0.3 \
           --out "$OUT/${arm}_p${paths}/seed_${s}.png" \
           --log "$OUT/${arm}_p${paths}/seed_${s}.jsonl" > /dev/null 2>&1 \
        || { echo "sampling failed: $arm p=$paths seed=$s" >&2; exit 1; }
    done
    python3 tools/sample_diversity.py \
        --glob "$OUT/${arm}_p${paths}/seed_*.png" --label "${arm} p=${paths}"
  done
done

python3 tools/sample_diversity.py --dataset datasets/cifar10_grey.batraw --n 16
