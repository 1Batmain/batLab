#!/usr/bin/env bash
# Runner de la suite à l'aveugle du régime « flux ».
#   ./blind_tests/run.sh                  → génère les dumps puis évalue les 8 propriétés
#   BLIND_BASELINE=1 ./blind_tests/run.sh → ajoute la comparaison bit-à-bit avec le
#                                            binaire pré-flux (P7, non-régression forte)
#   ACTIONS=6000 ./blind_tests/run.sh     → horizon plus long
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Greyscale_Diffusion_L
CKPT=Models/$MODEL/pretrained_weights/night_run.ckpt
SRC_CKPT=/Users/bat/development/lab/batLab/Models/$MODEL/pretrained_weights/night_run.ckpt
OUT=blind_tests/out
ACTIONS=${ACTIONS:-3000}
BIN=./target/release/main
export ACTIONS

mkdir -p "$OUT"
if [ ! -f "$CKPT" ]; then
  echo "→ copie du checkpoint depuis $SRC_CKPT"
  mkdir -p "$(dirname "$CKPT")"
  cp "$SRC_CKPT" "$CKPT"
fi

echo "→ build"
cargo build --release -p main >/dev/null

perp() { # perp <regime> <sortie> <actions> [flags…]
  local regime=$1 out=$2 acts=$3; shift 3
  "$BIN" --headless-perpetual "$MODEL" --regime "$regime" --actions "$acts" \
    --dump "$OUT/$out.f32" --checkpoint "$CKPT" "$@" >/dev/null 2>&1
}

echo "→ génération des dumps (actions=$ACTIONS)"
perp flux        flux_main   "$ACTIONS" --seed 7
perp flux        flux_repeat "$ACTIONS" --seed 7      # reproductibilité (P5)
perp flux        flux_seed2  "$ACTIONS" --seed 99     # divergence (P5)
perp errance     errance     "$ACTIONS" --seed 7
perp respiration respiration "$ACTIONS" --seed 7

# OBS : le binaire annonce « frames → perpetual_samples/<regime> » ; qui écrit vraiment ?
rm -rf perpetual_samples/flux perpetual_samples/errance perpetual_samples/respiration
for regime in flux errance respiration; do
  "$BIN" --headless-perpetual "$MODEL" --regime "$regime" --actions 1500 --seed 7 \
    --frames 8 --checkpoint "$CKPT" >"$OUT/stdout_$regime.txt" 2>&1
done
{ printf '{'
  sep=""
  for regime in flux errance respiration; do
    shopt -s nullglob; pngs=("perpetual_samples/$regime"/*.png); n=${#pngs[@]}
    printf '%s"%s": %s' "$sep" "$regime" "$n"; sep=", "
  done
  printf '}\n'; } > "$OUT/png.json"

# P4 : les leviers annoncés pour t* changent-ils quoi que ce soit ?
perp flux knob_ref   400 --seed 7
perp flux knob_tstar 400 --seed 7 --t-star 30
perp flux knob_tr    400 --seed 7 --t-r 30
perp flux knob_bogus 400 --seed 7 --flag-inexistant-de-controle 30

if [ "${BLIND_BASELINE:-0}" = "1" ]; then
  echo "→ baseline pré-flux (P7)"
  bash blind_tests/baseline.sh || echo "  (baseline indisponible — P7 restera partiel)"
else
  rm -f "$OUT/baseline.json"
fi

echo "→ évaluation"
python3 blind_tests/checks.py "$OUT"
