#!/usr/bin/env bash
# Paired SGD/Adam comparison on the Greyscale_Diffusion baseline.
#
# Pairing is structural, not a flag: main.rs derives every per-step random
# quantity from the step index alone — the batch seed is `(step as u64) << 32`
# (data order, diffusion timestep, noise) and the probe seed is
# `0x50B0_1234 ^ step`. Weight init draws from a fixed seed. So two runs with
# the same step count and batch size see byte-identical inputs, and the step-0
# batch loss (measured before any update) is bit-identical across arms.
# `analyse.py --check-pairing` asserts exactly that.
#
# Runs are STRICTLY SEQUENTIAL: the GPU is shared with other agents' runs, and
# two of ours in parallel would fight each other on top of that.
set -u
cd "$(dirname "$0")/../.."
. bench/optimizer/lib.sh
DATASET=/Users/bat/development/lab/batLab/datasets/cifar10_grey.batraw
STEPS=${STEPS:-1500}
BIN=./target/release/main
mkdir -p runs
ensure_binary

run() {
  local tag=$1 opt=$2 lr=$3
  shift 3
  echo "=== $tag (optimizer=$opt lr=$lr steps=$STEPS $*) start $(date -u +%H:%M:%S) ==="
  local t0=$(date +%s)
  $BIN --headless-train Greyscale_Diffusion --steps "$STEPS" --dataset "$DATASET" \
       --optimizer "$opt" --lr "$lr" "$@" --out "$PWD/runs/${tag}.ckpt" \
       > "$PWD/runs/${tag}.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$tag optimizer=$opt lr=$lr steps=$STEPS $* seconds=$((t1 - t0)) rc=$rc" \
    | tee -a "$PWD/runs/timings.txt"
}

run sgd_lr1e-3   sgd  1e-3
run adam_lr1e-3  adam 1e-3
run adam_lr3e-4  adam 3e-4
echo "ALL DONE $(date -u +%H:%M:%S)"
