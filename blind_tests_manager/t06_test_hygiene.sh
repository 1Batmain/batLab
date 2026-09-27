#!/usr/bin/env bash
# P6 — Hygiène des tests : `cargo test --workspace` laisse `git status` propre.
# Spec : MODEL_MANAGER.md §3 « Preuve que cargo test ne pollue plus » et §7.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P6 Hygiène des tests"
bt_init_report
echo "== $BT_PROP =="

SPEC='MODEL_MANAGER.md §3 : « Critère de recette permanent, inscrit dans AGENTS.md — cargo test --workspace suivi d’un git status propre. […] Avant cette mission, la même commande laissait Models/Stable_Diffusion/config_file modifié dans le dépôt. La limite est levée. »'
SPEC_GREEN='MODEL_MANAGER.md §7 : « cargo test --workspace # 167 tests, 0 échec » — « Aucun test n’a été affaibli. »'

cd "$BT_REPO" || exit 1

DIRTY_BEFORE="$(git status --porcelain)"
if [ -n "$DIRTY_BEFORE" ]; then
  skip "l'arbre de travail n'était pas propre AVANT la mesure — résultat non concluant"
  printf '%s\n' "$DIRTY_BEFORE" | sed 's/^/       /'
fi

# empreinte de Models/ et datasets/ du DÉPÔT (pas d'une racine jetable) :
# la limite historique était une réécriture de Models/Stable_Diffusion/config_file
MODELS_BEFORE="$(bt_snapshot "$BT_REPO/Models")"

LOG="$BT_WORK/cargo-test.log"
echo "  … cargo test --workspace (peut prendre plusieurs minutes)"
cargo test --workspace > "$LOG" 2>&1
RC=$?

if [ "$RC" -eq 0 ]; then ok "cargo test --workspace : 0 échec"
else
  fail "cargo test --workspace a échoué (code $RC)" "$SPEC_GREEN"
  grep -E '^(test result|error|failures:|---- )' "$LOG" | head -30 | sed 's/^/       /'
fi
grep -E '^test result:' "$LOG" | sed 's/^/       /'
TOTAL="$(grep -Eo '^test result: ok\. [0-9]+ passed' "$LOG" | grep -Eo '[0-9]+' | paste -sd+ - | bc 2>/dev/null)"
[ -n "$TOTAL" ] && echo "       total : $TOTAL tests verts"

DIRTY_AFTER="$(git status --porcelain)"
if [ "$DIRTY_AFTER" = "$DIRTY_BEFORE" ]; then
  ok "git status est inchangé après cargo test --workspace"
else
  fail "cargo test --workspace a sali l'arbre de travail" "$SPEC"
  diff <(printf '%s\n' "$DIRTY_BEFORE") <(printf '%s\n' "$DIRTY_AFTER") | sed 's/^/       /'
fi
if [ -z "$DIRTY_BEFORE" ]; then
  if [ -z "$DIRTY_AFTER" ]; then ok "git status --porcelain est vide après la suite"
  else fail "git status --porcelain n'est pas vide après la suite" "$SPEC"
    printf '%s\n' "$DIRTY_AFTER" | sed 's/^/       /'
  fi
else
  skip "critère « git status vide » non évaluable : l'arbre était déjà sali avant la mesure (le delta, lui, est vide — voir ci-dessus)"
fi

MODELS_AFTER="$(bt_snapshot "$BT_REPO/Models")"
if [ "$MODELS_BEFORE" = "$MODELS_AFTER" ]; then
  ok "le Models/ du dépôt est bit-à-bit intact (la limite PERPETUAL_INFERENCE §5 est bien levée)"
else
  fail "cargo test a modifié des fichiers sous Models/ du dépôt" "$SPEC"
  diff <(printf '%s\n' "$MODELS_BEFORE") <(printf '%s\n' "$MODELS_AFTER") | sed 's/^/       /'
fi

bt_summary
