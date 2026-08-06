#!/usr/bin/env bash
# blind_tests_manager/run.sh — suite à l'aveugle du gestionnaire de modèles.
#
#   ./blind_tests_manager/run.sh              # tout
#   ./blind_tests_manager/run.sh t02 t05      # une sélection
#   BLIND_DATASET=/chemin/x.batraw ./blind_tests_manager/run.sh
#
# Aucun test ne touche au Models/ du dépôt : chacun travaille sur une racine
# BATLAB_ROOT jetable, peuplée en shell depuis les modèles du dépôt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO="$PWD"

export BT_REPO="$REPO"
export BT_BIN="${BT_BIN:-$REPO/target/release/batlab}"
export BT_WORK="${BT_WORK:-${TMPDIR:-/tmp}/batlab-blind-manager.run.$$}"
mkdir -p "$BT_WORK"

if [ ! -x "$BT_BIN" ]; then
  echo "binaire absent : $BT_BIN"
  echo "  → cargo build --release -p batlab"
  exit 2
fi
command -v tmux >/dev/null || { echo "tmux est requis (pilotage du TUI)"; exit 2; }

ALL=(t01_flow t02_rename t03_delete t04_navigation t05_checkpoints t06_test_hygiene t07_infer t08_hostile)
SEL=()
if [ "$#" -gt 0 ]; then
  for a in "$@"; do
    for t in "${ALL[@]}"; do case "$t" in "$a"*) SEL+=("$t");; esac; done
  done
else
  SEL=("${ALL[@]}")
fi

echo "racine de travail : $BT_WORK"
echo "binaire           : $BT_BIN"
echo

declare -a VERDICTS=()
RC=0
for t in "${SEL[@]}"; do
  LOG="$BT_WORK/$t.log"
  bash "$REPO/blind_tests_manager/$t.sh" 2>&1 | tee "$LOG"
  LINE="$(grep -- '──' "$LOG" | tail -1)"
  NF="$(printf '%s' "$LINE" | sed -n 's/.*, \([0-9]*\) FAIL.*/\1/p')"
  NG="$(printf '%s' "$LINE" | sed -n 's/.*, \([0-9]*\) écart.*/\1/p')"
  if [ "${NF:-1}" -eq 0 ]; then
    if [ "${NG:-0}" -gt 0 ]; then VERDICTS+=("PASS (avec ${NG} écart(s) de spec) — $t")
    else VERDICTS+=("PASS — $t"); fi
  else
    VERDICTS+=("FAIL — $t (${NF})"); RC=1
  fi
  echo
done

echo "════════════════════════════════════════════════════"
for v in "${VERDICTS[@]}"; do
  case "$v" in
    PASS*) printf '  \033[32m%s\033[0m\n' "$v" ;;
    *)     printf '  \033[31m%s\033[0m\n' "$v" ;;
  esac
done
echo "════════════════════════════════════════════════════"
echo "journaux : $BT_WORK"
exit "$RC"
