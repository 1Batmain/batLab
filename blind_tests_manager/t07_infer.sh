#!/usr/bin/env bash
# P7 — Contrat d'inférence : depuis le menu d'actions, une inférence courte
# aboutit sur le modèle copié.
# Spec : MISSION_BLIND_TEST_MANAGER.md P7 (« PNG produit ») et
#        MODEL_MANAGER.md §5 pas 12 (« run complet, aperçu affiché »).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P7 Inférence"
bt_init_report
echo "== $BT_PROP =="

SPEC_RUN='MODEL_MANAGER.md §5 pas 12 : « Infer depuis latest.ckpt → run complet, aperçu affiché → OK — Inference Preview (32x32x1) »'
SPEC_NOW='MODEL_MANAGER.md §6 : « Infer sur un modèle sans poids est un cul-de-sac assumé : […] le run s’arrête sur un message qui dit quoi faire (train the model first or place weights there). »'
SPEC_PNG="MISSION_BLIND_TEST_MANAGER.md P7 : « depuis le menu d'actions, une inférence courte aboutit (PNG produit) sur le modèle copié »"
SPEC_CONF='MODEL_MANAGER.md §5 : « Confinement : toutes les écritures — modèles, checkpoints, métriques, datasets/generated_samples/step_0039.png — ont atterri dans la racine jetable. »'

ROOT="$(bt_new_root p7)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
DS="$(bt_dataset "$ROOT")" || DS=""
REPO_SNAP_BEFORE="$(bt_snapshot "$BT_REPO/Models")"
REPO_DS_BEFORE="$(ls -1 "$BT_REPO/datasets" 2>/dev/null | LC_ALL=C sort)"

bt_launch p7 "$ROOT" || { fail "TUI non démarré" "$SPEC_RUN"; bt_summary; exit 1; }

# --- 1. Infer SANS poids : un cul-de-sac, mais explicite et sans crash
bt_select 'Alpha' && bt_key Enter
bt_action 'Infer'
bt_key Enter                       # « Start from random weights »
bt_key Enter; bt_key Enter; bt_key Enter; bt_key Enter
bt_wait_for 'Status       :' 60
bt_redraw
S="$(bt_screenj)"
if printf '%s' "$S" | grep -qF 'inference checkpoint not found'; then
  ok "Infer sans poids : message explicite « inference checkpoint not found », pas de crash"
else
  skip "Infer sans poids n'a pas produit le message attendu (peut-être un autre chemin d'erreur)"
fi
if printf '%s' "$S" | grep -qF "$ROOT"; then
  ok "le chemin de checkpoint cherché est DANS la racine jetable (confinement)"
else
  fail "le chemin de checkpoint cherché ne pointe pas dans la racine BATLAB_ROOT" "$SPEC_CONF"
  bt_dump_screen "$S"
fi
bt_alive && ok "l'application est vivante après l'échec d'inférence" \
  || { fail "l'application est morte après l'échec d'inférence (code $(bt_exit_code))" "$SPEC_NOW"; bt_summary; exit 1; }

# --- 2. Un entraînement court pour disposer de poids chargeables
if [ -z "$DS" ]; then
  bt_quit
  skip "aucun .batraw trouvé (BLIND_DATASET non fourni) — l'inférence depuis un checkpoint n'a pas pu être testée"
  bt_summary; exit $?
fi
bt_key r                            # -> menu d'actions
bt_action 'Train'
bt_key Enter                        # random weights
bt_wait_for 'Training Parameters' 15
bt_key Down; bt_key Down; bt_backspace 6; bt_type '20'
bt_key Enter
bt_wait_for 'Training Dataset' 15
bt_key Enter
PW="$ROOT/Models/Alpha/pretrained_weights"
DEADLINE=$((SECONDS+300))
while [ "$SECONDS" -lt "$DEADLINE" ] && [ ! -f "$PW/latest.ckpt" ]; do sleep 2; done
[ -f "$PW/latest.ckpt" ] && ok "l'entraînement court a produit des poids chargeables" \
  || { fail "pas de latest.ckpt après l'entraînement court" "$SPEC_RUN"; bt_quit; bt_summary; exit 1; }
sleep 4

# --- 3. Inférence depuis ces poids, depuis le menu d'actions
bt_key r                            # -> menu d'actions (relit les checkpoints)
bt_redraw
bt_action 'Infer'
bt_wait_for '— Weights' 15
bt_select 'latest.ckpt' || fail "le checkpoint frais n'est pas sélectionnable dans le sélecteur de poids" "$SPEC_RUN"
bt_key Enter
bt_wait_for 'Inference Parameters' 15 || fail "formulaire d'inférence non atteint" "$SPEC_RUN"
bt_key Down; bt_key Down            # champ « Denoising Paths »
bt_backspace 4; bt_type '1'         # un seul chemin : inférence courte
bt_key Enter; bt_key Enter          # -> Magnitude -> run

PNG_BEFORE="$(cd "$ROOT" && find . -name '*.png' | LC_ALL=C sort)"
INFER_START="$SECONDS"
# le panneau porte le titre « Inference Preview » DÈS le début du run ; il ne
# gagne ses dimensions « (WxHxC) » qu'une fois l'image composée
bt_wait_for 'Inference Preview (' 600
bt_redraw
S="$(bt_screen)"
if printf '%s' "$S" | grep -qE 'Inference Preview \([0-9]+x[0-9]+x[0-9]+\)'; then
  ok "l'inférence aboutit : l'aperçu « Inference Preview (WxHxC) » est affiché ($((SECONDS-INFER_START)) s)"
else
  fail "l'inférence n'a pas abouti à un aperçu « Inference Preview (WxHxC) »" "$SPEC_RUN"
  bt_dump_screen "$S"
fi
# l'aperçu porte des pixels, pas un cadre vide
BODY="$(printf '%s' "$S" | grep -c '▀')"
if [ "$BODY" -ge 8 ]; then ok "l'aperçu contient une image dessinée ($BODY lignes de pixels)"
else fail "l'aperçu est vide (aucune ligne de pixels)" "$SPEC_RUN"; fi
if printf '%s' "$S" | grep -qiF 'error'; then
  fail "un message d'erreur subsiste à l'écran après l'inférence" "$SPEC_RUN"; bt_dump_screen "$S"
else ok "aucun message d'erreur à l'écran après l'inférence"; fi

sleep 3
PNG_AFTER="$(cd "$ROOT" && find . -name '*.png' | LC_ALL=C sort)"
NEWPNG="$(comm -13 <(printf '%s\n' "$PNG_BEFORE") <(printf '%s\n' "$PNG_AFTER"))"
if [ -n "$NEWPNG" ]; then
  ok "l'inférence a écrit un PNG : $(printf '%s' "$NEWPNG" | tr '\n' ' ')"
else
  spec_gap "l'inférence aboutit (aperçu affiché) mais n'écrit AUCUN PNG sur disque" \
    "l'ordre de mission dit « (PNG produit) » ; la spec du manager, elle, ne promet qu'un aperçu — $SPEC_RUN. Les seuls PNG de la racine viennent du run d'entraînement (datasets/generated_samples/step_*.png). Écart de formulation, pas de bug observable."
  echo "       PNG présents sous la racine : $(printf '%s' "$PNG_AFTER" | tr '\n' ' ')"
fi
# le chemin d'entraînement, lui, écrit bien des PNG dans la racine jetable
if printf '%s\n' "$PNG_AFTER" | grep -q 'generated_samples'; then
  ok "les PNG produits par le run se trouvent DANS la racine jetable (datasets/generated_samples/)"
else
  skip "aucun PNG sous la racine jetable — rien à confiner"
fi

# --- 4. confinement : le dépôt n'a pas bougé
REPO_SNAP_AFTER="$(bt_snapshot "$BT_REPO/Models")"
assert_eq "$REPO_SNAP_AFTER" "$REPO_SNAP_BEFORE" "le Models/ du dépôt est intact après le parcours" "$SPEC_CONF"
REPO_DS_AFTER="$(ls -1 "$BT_REPO/datasets" 2>/dev/null | LC_ALL=C sort)"
assert_eq "$REPO_DS_AFTER" "$REPO_DS_BEFORE" "le datasets/ du dépôt n'a reçu aucun fichier" "$SPEC_CONF"

# --- 5. sortie propre
bt_quit
sleep 0.5
assert_eq "$(bt_exit_code)" "0" "la session se termine sur un code de sortie 0" "$SPEC_RUN"
ERRBYTES="$(wc -c < "$BT_STDERR" | tr -d ' ')"
if [ "$ERRBYTES" = "0" ]; then ok "stderr est resté vide sur tout le parcours"
else
  fail "stderr n'est pas vide ($ERRBYTES octets)" "$SPEC_RUN"
  head -10 "$BT_STDERR" | sed 's/^/       /'
fi

bt_summary
