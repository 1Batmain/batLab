#!/usr/bin/env bash
# Runner de la suite à l'aveugle de la couche d'attention.
#
#   ./blind_tests_attention/run.sh            → construit puis déroule les 7 volets
#   ./blind_tests_attention/run.sh t4 t6      → n'en déroule que certains
#
# Tout se passe dans une racine de stockage jetable (BATLAB_ROOT =
# blind_tests_attention/work/, gitignorée) : le `Models/` du dépôt n'est jamais
# touché. Le checkpoint entraîné est copié depuis `.checkpoint_backup/` du dépôt
# principal et n'est jamais commité.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "→ build"
cargo build --release -p batlab >/dev/null || exit 1

cd blind_tests_attention
ALL=(t1_identity t2_trained t3_determinism t4_reference t5_batch t6_softmax t7_grid)
if [ $# -gt 0 ]; then
  SEL=(); for a in "$@"; do for t in "${ALL[@]}"; do [[ $t == $a* ]] && SEL+=("$t"); done; done
else
  SEL=("${ALL[@]}")
fi

fail=0
for t in "${SEL[@]}"; do
  echo
  echo "══ $t ═══════════════════════════════════════════════════════"
  python3 "$t.py" || fail=$((fail+1))
done

echo
if [ $fail -eq 0 ]; then echo "TOUS LES VOLETS PASSENT (${#SEL[@]}/${#SEL[@]})"; else
  echo "ÉCHEC : $fail volet(s) sur ${#SEL[@]}"; fi
exit $fail
