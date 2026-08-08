#!/usr/bin/env bash
# EMA contre poids bruts, depuis UN SEUL checkpoint (EMA.md §5).
#
# Les deux bras sortent du même fichier : `--raw-weights` choisit l'itéré brut,
# le défaut prend la moyenne. Rien d'autre ne change — mêmes graines, même
# magnitude, même schedule —, donc l'écart mesuré n'est imputable qu'au jeu de
# poids. C'est ce qu'un run par bras ne pourrait PAS dire : deux entraînements
# séparés diffèrent aussi par leur tirage de données.
#
# `--magnitude 1.0` : `inter_seed_std` est proportionnel à la magnitude
# (CLAUDE.md), il ne se lit qu'à magnitude fixée, et 1.0 est le réglage nominal.
# `--paths 1` : `paths=3` moyenne trois trajectoires et masque une partie de
# l'effondrement (SCALE_UNET §1.3).
set -u
cd "$(dirname "$0")/../.."
BIN=./target/release/batlab
MODEL=${MODEL:-Greyscale_Diffusion_L}
CKPT=${CKPT:-runs/ema_campaign/ema1500.ckpt}
OUT=${OUT:-runs/ema_campaign/samples}
SEEDS=${SEEDS:-16}
MAG=${MAG:-1.0}

[ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 1; }

for arm in ema raw; do
  case "$arm" in
    ema) flags="" ;;
    raw) flags="--raw-weights" ;;
  esac
  mkdir -p "$OUT/$arm"
  for s in $(seq 1 "$SEEDS"); do
    # shellcheck disable=SC2086
    $BIN --headless-sample "$MODEL" --checkpoint "$CKPT" \
         --seed "$s" --paths 1 --magnitude "$MAG" $flags \
         --out "$OUT/$arm/seed_$(printf '%02d' "$s").png" \
         --log "$OUT/$arm/seed_$(printf '%02d' "$s").jsonl" > /dev/null 2>&1 \
      || { echo "sampling failed: $arm seed=$s" >&2; exit 1; }
  done
done

for arm in ema raw; do
  python3 tools/sample_diversity.py --glob "$OUT/$arm/seed_*.png" --label "$arm"
done
python3 tools/sample_diversity.py --dataset datasets/cifar10_grey.batraw --n "$SEEDS"
python3 bench/ema/paired.py --a "$OUT/ema" --b "$OUT/raw"
