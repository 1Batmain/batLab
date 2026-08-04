#!/usr/bin/env bash
# Per-step cost of Adam vs SGD, measured so that neither process startup nor a
# shared GPU can fake the answer.
#
# Two problems with reading ms/step off the big comparison runs:
#   1. wall-clock includes process start, shader compilation and dataset load —
#      a fixed cost that dilutes a per-step difference;
#   2. the GPU is shared with other agents' training, and load drifts over the
#      ~25 min a 1500-step run takes. In the raw compare.sh timings the *Adam*
#      arm came out FASTER than SGD, which is impossible in compute terms —
#      that number is contention, not optimiser cost.
#
# So: for each arm run SHORT (50 steps) and LONG (250 steps) and take the slope
#   per_step = (t_long - t_short) / (250 - 50)
# which cancels the constant startup exactly. Arms are interleaved within a
# repetition (short/short then long/long, back to back) so both see the same
# neighbourhood of machine load, and the whole thing is repeated so the median
# survives a contention spike landing on one arm.
set -u
cd "$(dirname "$0")/../.."
DATASET=/Users/bat/development/lab/batLab/datasets/cifar10_grey.batraw
BIN=./target/release/main
SHORT=${SHORT:-50}
LONG=${LONG:-250}
REPS=${REPS:-3}
OUT=$PWD/runs/timing.txt
mkdir -p runs
: > "$OUT"

timed() {
  local opt=$1 steps=$2 rep=$3
  local t0=$(date +%s%N)
  $BIN --headless-train Greyscale_Diffusion --steps "$steps" --dataset "$DATASET" \
       --optimizer "$opt" --lr 1e-3 --out "$PWD/runs/timing_scratch.ckpt" \
       > "$PWD/runs/timing_${opt}_${steps}_${rep}.log" 2>&1
  local rc=$?
  local t1=$(date +%s%N)
  echo "rep=$rep optimizer=$opt steps=$steps ms=$(( (t1 - t0) / 1000000 )) rc=$rc" | tee -a "$OUT"
}

for rep in $(seq 1 "$REPS"); do
  timed sgd  "$SHORT" "$rep"
  timed adam "$SHORT" "$rep"
  timed sgd  "$LONG"  "$rep"
  timed adam "$LONG"  "$rep"
done
echo "TIMING DONE $(date -u +%H:%M:%S)"
