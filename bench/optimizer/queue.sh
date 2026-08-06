#!/usr/bin/env bash
# Serialises the remaining benchmark work, so the GPU is never idle and never
# runs two of our jobs at once (it is already shared with another agent's run).
#
# Order is by priority, not by section number: the He ablation on the winning
# optimiser and the per-step cost measurement are required deliverables; the
# lr = 3e-3 arm is the bonus and comes last.
set -u
cd "$(dirname "$0")/../.."
. bench/optimizer/lib.sh
WAIT_PID=${WAIT_PID:-}
DATASET=/Users/bat/development/lab/batLab/datasets/cifar10_grey.batraw
STEPS=${STEPS:-1500}
BIN=./target/release/batlab
mkdir -p runs
ensure_binary

if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

run() {
  local tag=$1 opt=$2 lr=$3 init=$4
  local log="$PWD/runs/${tag}.log"
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  local t0=$(date +%s)
  $BIN --headless-train Greyscale_Diffusion --steps "$STEPS" --dataset "$DATASET" \
       --optimizer "$opt" --lr "$lr" --weight-init "$init" \
       --out "$PWD/runs/${tag}.ckpt" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  assert_flag "$log" "optimizer=$opt, init=$init" || rc=99
  echo "$tag optimizer=$opt lr=$lr init=$init steps=$STEPS seconds=$((t1 - t0)) rc=$rc" \
    | tee -a "$PWD/runs/timings.txt"
}

run adam_lr1e-3_he adam 1e-3 he       # required: He ablation on the winner
bash bench/optimizer/timing.sh        # required: per-step cost, contention-proof
# Bracket the learning rate from above: 1e-3 beat 3e-4 on every metric, so the
# optimum is only bounded from below and the recommendation needs a number.
run adam_lr3e-3 adam 3e-3 uniform
echo "QUEUE DONE $(date -u +%H:%M:%S)"
