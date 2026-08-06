#!/usr/bin/env bash
# Paired old-vs-new step timing, swept over the batch size.
#
# The question this benchmark exists to answer is NOT "is the new path faster"
# — it is "does the gain grow with the batch". Section 7 of PERF_CONVOLUTION.md
# claims the kernels are parallelism-bound because the batch was unrolled on the
# CPU; if that claim is right, the two paths must be indistinguishable at
# batch 1 (one sample, one submit either way) and diverge as the batch grows.
# Batch 1 is therefore not a data point to be embarrassed about, it is the
# INTERNAL CONTROL: a "speedup" there would mean the instrument is measuring
# something else.
#
# Protocol is the house one (PERF_GROUP_NORM.md, PERF_CONVOLUTION.md §5.1):
#
#   - paired and INTERLEAVED (old, new, old, new, ...) inside one sweep, so a
#     drift in machine load hits both arms alike instead of one of them;
#   - estimator is the MINIMUM over rounds, not the mean. Contention can only
#     ADD time, so the fastest round is the best estimate of the true cost;
#     taking it on both arms keeps the comparison fair. The spread is printed
#     so the reader can see how much was thrown away;
#   - a NULL CONTROL arm: the same binary against itself. It measures the
#     instrument's own noise floor, and any effect smaller than it is not an
#     effect. Publishing a speedup without it is publishing a guess.
#
# Usage:
#   bench/batch_dispatch/timing.sh [rounds] [steps]
#
# Requires the baseline worktree built:
#   git worktree add worktrees/batch-baseline <pre-batch-commit> --detach
#   (cd worktrees/batch-baseline && cargo build --release -p batlab)

set -uo pipefail

ROUNDS=${1:-5}
STEPS=${2:-60}
MODEL=Greyscale_Diffusion_L
BATCHES=(1 4 16 64)

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASELINE=$REPO/../batch-baseline
DATASET=${DATASET:-$REPO/datasets/cifar10_grey.batraw}
OUT=${OUT:-$(mktemp -d)}

NEW_BIN=$REPO/target/release/batlab
OLD_BIN=$BASELINE/target/release/batlab

for bin in "$NEW_BIN" "$OLD_BIN"; do
  [ -x "$bin" ] || { echo "FATAL: missing $bin — build both worktrees first" >&2; exit 1; }
done
[ -f "$DATASET" ] || { echo "FATAL: no dataset at $DATASET" >&2; exit 1; }

# One timed run. Echoes seconds; fails loudly if the banner does not echo the
# batch back — a binary that silently ignored --batch would produce a perfectly
# clean, perfectly meaningless arm (the stale-binary trap of
# bench/optimizer/lib.sh, which cost a whole He-init campaign).
run_one() {
  local bin=$1 dir=$2 batch=$3 log=$4
  local start end
  start=$(python3 -c 'import time;print(time.monotonic())')
  # `--out` is not optional here. Without it the run writes its metrics to a
  # path derived from the model name alone, so two arms of the same model —
  # which is exactly what a paired benchmark runs — write to the SAME file and
  # interleave into each other. That happened during this mission: a smoke test
  # of this very script corrupted a 600-step validation run's metrics, and the
  # only reason it was caught is that the file no longer parsed. A unique
  # checkpoint path per arm gives a unique metrics path per arm.
  ( cd "$dir" && "$bin" --headless-train "$MODEL" --steps "$STEPS" \
      --dataset "$DATASET" --lr 1e-3 --optimizer adam --batch "$batch" \
      --out "$OUT/$(basename "$log" .log).ckpt" ) \
      > "$log" 2>&1
  end=$(python3 -c 'import time;print(time.monotonic())')
  grep -q "batch=$batch" "$log" || {
    echo "FATAL: '$batch' missing from the banner of $log — stale binary" >&2
    exit 1
  }
  python3 -c "print(f'{$end - $start:.4f}')"
}

echo "batch-dispatch timing sweep — $ROUNDS rounds x $STEPS steps, model $MODEL"
echo "output: $OUT"
echo

for batch in "${BATCHES[@]}"; do
  olds=(); news=(); nulls=()
  for _ in $(seq "$ROUNDS"); do
    olds+=("$(run_one "$OLD_BIN" "$BASELINE" "$batch" "$OUT/old_b$batch.log")")
    news+=("$(run_one "$NEW_BIN" "$REPO"     "$batch" "$OUT/new_b$batch.log")")
  done
  # Null control: the NEW binary run twice, same arm both times. Its spread is
  # the floor below which nothing measured here means anything.
  for _ in $(seq 2); do
    nulls+=("$(run_one "$NEW_BIN" "$REPO" "$batch" "$OUT/null_b$batch.log")")
  done

  python3 - "$batch" "$STEPS" "${#olds[@]}" "${olds[@]}" "${news[@]}" "${nulls[@]}" <<'PY'
import sys
batch, steps, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
vals = [float(v) for v in sys.argv[4:]]
old, new, null = vals[:n], vals[n:2*n], vals[2*n:]
def ms(xs): return min(xs) / steps * 1000
o, w = ms(old), ms(new)
null_spread = (max(null) - min(null)) / min(null) * 100 if len(null) > 1 else float('nan')
print(f"batch {batch:>3} | old {o:8.2f} ms/step (spread {(max(old)-min(old))/min(old)*100:5.1f}%) "
      f"| new {w:8.2f} ms/step (spread {(max(new)-min(new))/min(new)*100:5.1f}%) "
      f"| speedup {o/w:5.2f}x | null control {null_spread:4.1f}%")
PY
done
