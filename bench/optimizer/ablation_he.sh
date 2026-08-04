#!/usr/bin/env bash
# He-init ablation. The uniform arms come from compare.sh (adam_lr1e-3,
# sgd_lr1e-3); this script only adds the He arms, so `--weight-init uniform`
# vs `he` is a single-variable change at identical seeds and step counts.
#
# Adam first: it is the winner of the optimiser comparison, so its ablation is
# the one the model-L recommendation depends on. The SGD arm answers the
# secondary question "does a better init rescue SGD?" and can be dropped.
set -u
cd "$(dirname "$0")/../.."
DATASET=/Users/bat/development/lab/batLab/datasets/cifar10_grey.batraw
STEPS=${STEPS:-1500}
BIN=./target/release/main
mkdir -p runs

run() {
  local tag=$1 opt=$2 lr=$3
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  local t0=$(date +%s)
  $BIN --headless-train Greyscale_Diffusion --steps "$STEPS" --dataset "$DATASET" \
       --optimizer "$opt" --lr "$lr" --weight-init he \
       --out "$PWD/runs/${tag}.ckpt" > "$PWD/runs/${tag}.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$tag optimizer=$opt lr=$lr init=he steps=$STEPS seconds=$((t1 - t0)) rc=$rc" \
    | tee -a "$PWD/runs/timings.txt"
}

run adam_lr1e-3_he adam 1e-3
run sgd_lr1e-3_he  sgd  1e-3
echo "ABLATION DONE $(date -u +%H:%M:%S)"
