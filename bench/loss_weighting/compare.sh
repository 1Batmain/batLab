#!/usr/bin/env bash
# Paired uniform/snr comparison of the loss weighting, on Greyscale_Diffusion_L.
#
# Pairing is weaker here than in bench/optimizer by construction: the two arms
# differ precisely in WHICH timestep each sample gets, so their step-0 losses
# differ (1.0726 vs 1.1305) and `--check-pairing` does NOT apply. What is still
# shared: the dataset order, the weight init, the per-step noise seeds, and the
# probe (a fixed sample set at fixed timesteps, deliberately untouched by the
# weighting — it is the measuring instrument, not the objective).
#
# The two arms run in PARALLEL: the deadline is a fixed wall-clock (the night
# training run), and the loss curves are unaffected by GPU contention. The
# per-step TIMINGS from this script are therefore meaningless — see
# OPTIMIZER_ADAM.md §2.3 for the same caveat.
set -u
cd "$(dirname "$0")/../.."
. bench/optimizer/lib.sh
DATASET=${DATASET:-datasets/cifar10_grey.batraw}
MODEL=${MODEL:-Greyscale_Diffusion_L}
STEPS=${STEPS:-1500}
BIN=./target/release/main
mkdir -p runs
ensure_binary

run() {
  local tag=$1
  shift
  echo "=== $tag ($* steps=$STEPS) start $(date -u +%H:%M:%S) ==="
  $BIN --headless-train "$MODEL" --steps "$STEPS" --dataset "$DATASET" \
       --optimizer adam --lr 1e-3 --batch 16 "$@" \
       --out "$PWD/runs/${tag}.ckpt" > "$PWD/runs/${tag}.log" 2>&1 &
}

run uniform --loss-weighting uniform
run snr_g1  --loss-weighting snr --snr-gamma 1
wait

# The stale-binary guard of bench/optimizer/lib.sh, applied to the new flag: the
# banner re-emits the weighting the binary actually parsed.
assert_flag runs/uniform.log 'loss-weighting=uniform' || exit 1
assert_flag runs/snr_g1.log  'loss-weighting=snr(gamma=1)' || exit 1

echo "ALL DONE $(date -u +%H:%M:%S)"
python3 tools/compare_weighting.py \
    uniform:runs/uniform_metrics.jsonl snr:runs/snr_g1_metrics.jsonl
