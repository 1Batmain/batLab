#!/usr/bin/env bash
# Serialises the remaining benchmark work behind a still-running run, so the GPU
# is never idle and never runs two of our jobs at once.
#
# Order is by priority, not by section number: the He ablation on the winning
# optimiser and the per-step cost measurement are required deliverables; the SGD
# He arm only answers the secondary "does a better init rescue SGD?" and is last
# so it can be dropped if time runs out.
set -u
cd "$(dirname "$0")/../.."
WAIT_PID=${WAIT_PID:-}
DATASET=/Users/bat/development/lab/batLab/datasets/cifar10_grey.batraw
STEPS=${STEPS:-1500}
BIN=./target/release/main
mkdir -p runs

if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  echo "pid $WAIT_PID finished $(date -u +%H:%M:%S)"
fi

run() {
  local tag=$1 opt=$2 lr=$3 init=$4
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  local t0=$(date +%s)
  $BIN --headless-train Greyscale_Diffusion --steps "$STEPS" --dataset "$DATASET" \
       --optimizer "$opt" --lr "$lr" --weight-init "$init" \
       --out "$PWD/runs/${tag}.ckpt" > "$PWD/runs/${tag}.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "$tag optimizer=$opt lr=$lr init=$init steps=$STEPS seconds=$((t1 - t0)) rc=$rc" \
    | tee -a "$PWD/runs/timings.txt"
}

run adam_lr1e-3_he adam 1e-3 he      # required: the He ablation on the winner
bash bench/optimizer/timing.sh       # required: per-step cost, contention-proof
run sgd_lr1e-3_he  sgd  1e-3 he      # bonus: does He rescue SGD?
echo "QUEUE DONE $(date -u +%H:%M:%S)"
