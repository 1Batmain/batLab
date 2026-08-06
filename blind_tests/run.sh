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

# P4 : la loi d'amplitude, sur tout le cadran
rm -f "$OUT"/ts_*.f32
TSTARS=${TSTARS:-"8 16 32 64 96 160 224 250"}
for k in $TSTARS; do perp flux "ts_$k" "$((ACTIONS < 1500 ? ACTIONS : 1500))" --seed 7 --t-star "$k"; done

# P8 : bornes et diagnostics du CLI (contrat publié par --help)
rm -rf perpetual_samples/flux perpetual_samples/errance perpetual_samples/respiration
cli() { "$BIN" "$@" >"$OUT/_cli.txt" 2>&1 && echo 0 || echo $?; }
help_exit=$(cli --help); help_len=$(wc -c <"$OUT/_cli.txt" | tr -d ' ')
unk_exit=$(cli --headless-perpetual "$MODEL" --regime flux --actions 5 \
  --flag-inexistant-de-controle 30 --checkpoint "$CKPT"); unk_msg=$(tr -d '"\n' <"$OUT/_cli.txt")
ff_exit=$(cli --headless-perpetual "$MODEL" --regime flux --frames 8 \
  --checkpoint "$CKPT"); ff_msg=$(tr -d '"\n' <"$OUT/_cli.txt")
for regime in flux errance respiration; do
  "$BIN" --headless-perpetual "$MODEL" --regime "$regime" --actions 900 --seed 7 \
    --frames 8 --checkpoint "$CKPT" >"$OUT/stdout_$regime.txt" 2>&1 || true
done
shopt -s nullglob
png_json=""; sep=""
for regime in flux errance respiration; do
  pngs=("perpetual_samples/$regime"/*.png)
  png_json+="$sep\"$regime\": ${#pngs[@]}"; sep=", "
done
# --actions doit l'emporter sur --frames quand les deux sont donnés
perp flux both 700 --seed 7 --frames 2
both_n=$(python3 -c "import sys;sys.path.insert(0,'blind_tests');from dumpio import read_dump;print(read_dump('$OUT/both.f32')['n'])")
# le cadran répond-il à ses trois orthographes ?
dial_json=""; sep=""
for flag in --t-r --t-star --depth; do
  v=$("$BIN" --headless-perpetual "$MODEL" --regime flux --actions 2 "$flag" 137 \
      --checkpoint "$CKPT" 2>&1 | sed -n 's/.*t\*=\([0-9]*\).*/\1/p' | head -1)
  dial_json+="$sep\"$flag\": ${v:-null}"; sep=", "
done
cat > "$OUT/cli.json" <<JSON
{"help": {"exit": $help_exit, "len": $help_len},
 "unknown": {"exit": $unk_exit, "msg": "$unk_msg"},
 "flux_frames": {"exit": $ff_exit, "msg": "$ff_msg"},
 "png": {$png_json},
 "actions_n": 700, "both": {"frames": $both_n},
 "dial": {$dial_json}}
JSON

# hygiène : les runs ci-dessus réécrivent perpetual_samples/ (et le binaire peut toucher
# aux configs de Models/) — on remet l'arbre de travail dans l'état du dépôt.
git checkout -- perpetual_samples Models 2>/dev/null || true
git clean -fdq perpetual_samples 2>/dev/null || true

if [ "${BLIND_BASELINE:-0}" = "1" ]; then
  echo "→ baseline pré-flux (P7)"
  bash blind_tests/baseline.sh || echo "  (baseline indisponible — P7 restera partiel)"
else
  rm -f "$OUT/baseline.json"
fi

echo "→ évaluation"
python3 blind_tests/checks.py "$OUT"
