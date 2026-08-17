#!/usr/bin/env bash
# Balayage sur les VRAIS poids de la page : eleph_night/night.ckpt (BBCKPT3, EMA
# 0.999) == web/dist/weights.ckpt au bit près. L'axe poids est VIVANT (EMA défaut
# vs --raw-weights). Magnitudes étendues au-dessus de 1.0 (déficit de détail).
# Stage A : 1 trajectoire. Sortie : $DIR/m<mag>_<var>_<w>_p1_s<seed>.png
set -euo pipefail
BIN=/Users/bat/development/lab/batLab/worktrees/sampling/target/release/batlab
CKPT=/Users/bat/development/lab/batLab/eleph_night/night.ckpt
export BATLAB_ROOT=/Users/bat/development/lab/batLab
DIR="$1"
mkdir -p "$DIR"

SEEDS=(101 202 303 404 505 606)
MAGS=(0.3 0.5 0.7 0.85 1.0 1.2 1.5)
VARS=(beta posterior)
# weights: étiquette -> flag
run() { # $1=mag $2=var $3=wlabel $4=seed $5=extraflag
  local out="$DIR/m$1_$2_$3_p1_s$4.png"
  [ -f "$out" ] && return
  # shellcheck disable=SC2086
  "$BIN" --headless-sample Elephants_XL --checkpoint "$CKPT" \
    --seed "$4" --paths 1 --magnitude "$1" --variance "$2" $5 \
    --out "$out" --log "${out%.png}.jsonl" >/dev/null 2>&1
}

total=$(( ${#SEEDS[@]} * ${#MAGS[@]} * ${#VARS[@]} * 2 ))
i=0
for v in "${VARS[@]}"; do
  for m in "${MAGS[@]}"; do
    for s in "${SEEDS[@]}"; do
      run "$m" "$v" ema  "$s" ""            ; i=$((i+1))
      run "$m" "$v" raw  "$s" "--raw-weights"; i=$((i+1))
      if [ $((i % 28)) -eq 0 ]; then echo "generated $i/$total"; fi
    done
  done
done
echo "DONE $i/$total"
