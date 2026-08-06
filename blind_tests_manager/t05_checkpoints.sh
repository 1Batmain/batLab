#!/usr/bin/env bash
# P5 — Checkpoints : seuls les .ckpt non cachés sont listés et comptés ; après
# un run court, le menu d'actions relu propose le checkpoint frais.
# Spec : MODEL_MANAGER.md §2.5 et §4.1/§4.2.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P5 Checkpoints"
bt_init_report
echo "== $BT_PROP =="

SPEC_CKPT='MODEL_MANAGER.md §2.5 : « Un checkpoint est un fichier d’extension .ckpt dans Models/<nom>/pretrained_weights/, non caché. Le latest_metrics.jsonl que tout run d’entraînement dépose à côté n’en est pas un — ni dans le compteur « N checkpoints » de la liste, ni dans les poids que le sélecteur propose de charger. »'
SPEC_HIDDEN='MODEL_MANAGER.md §4.2 : « les fichiers cachés restent exclus par-dessus l’extension, parce que les AppleDouble de macOS (._latest.ckpt) passeraient le test d’extension sans être des poids. »'
SPEC_FRESH='MODEL_MANAGER.md §4.1 : « le menu annonçait « no checkpoints » pour un modèle dont le run qui venait de finir avait justement écrit latest.ckpt. […] App::enter_model_actions fait maintenant les deux d’un bloc : relire le disque, puis poser l’écran. »'

ROOT="$(bt_new_root p5)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
PW="$ROOT/Models/Alpha/pretrained_weights"
mkdir -p "$PW"
head -c 256 /dev/urandom > "$PW/real_one.ckpt"          # un vrai checkpoint
echo '{"step":0}'        > "$PW/foo_metrics.jsonl"      # leurre : journal de métriques
head -c 64 /dev/urandom  > "$PW/._x.ckpt"               # leurre : AppleDouble caché
head -c 64 /dev/urandom  > "$PW/._latest.ckpt"          # leurre : AppleDouble caché
echo 'notes'             > "$PW/README.txt"             # leurre : fichier quelconque
head -c 64 /dev/urandom  > "$PW/weights.ckpt.bak"       # leurre : extension décalée
mkdir -p "$PW/archive.ckpt"                             # leurre : un RÉPERTOIRE en .ckpt

bt_launch p5 "$ROOT" || { fail "TUI non démarré" "$SPEC_CKPT"; bt_summary; exit 1; }
S="$(bt_screen)"

assert_contains "$S" '1 checkpoint: real_one.ckpt' "la liste ne compte que le .ckpt véritable" "$SPEC_CKPT"
for decoy in 'foo_metrics.jsonl' 'README.txt' 'weights.ckpt.bak'; do
  assert_not_contains "$S" "$decoy" "le compteur de la liste ignore « $decoy »" "$SPEC_CKPT"
done
assert_not_contains "$S" '._x.ckpt'      "le compteur ignore l'AppleDouble ._x.ckpt" "$SPEC_HIDDEN"
assert_not_contains "$S" '._latest.ckpt' "le compteur ignore l'AppleDouble ._latest.ckpt" "$SPEC_HIDDEN"
assert_not_contains "$S" 'archive.ckpt'  "le compteur ignore un RÉPERTOIRE nommé archive.ckpt" "$SPEC_CKPT"

# --- le sélecteur de poids fait le même tri
bt_select 'Alpha' && bt_key Enter
bt_action 'Infer'
S="$(bt_screen)"
assert_contains "$S" '— Weights' "le sélecteur de poids s'ouvre" "$SPEC_CKPT"
assert_contains "$S" 'real_one.ckpt' "le sélecteur propose le .ckpt véritable" "$SPEC_CKPT"
for decoy in 'foo_metrics.jsonl' 'README.txt' 'weights.ckpt.bak' '._x.ckpt' '._latest.ckpt' 'archive.ckpt'; do
  assert_not_contains "$S" "$decoy" "le sélecteur de poids ne propose pas « $decoy »" "$SPEC_CKPT"
done
bt_key Escape

# --- après un run d'entraînement, le menu relu voit le checkpoint frais
DS="$(bt_dataset "$ROOT")" || DS=""
if [ -z "$DS" ]; then
  bt_quit
  skip "aucun .batraw trouvé (BLIND_DATASET non fourni) — le volet « checkpoint frais » de P5 est resté non testé"
  bt_summary; exit $?
fi
# le TUI ne rafraîchit pas la liste des datasets en cours de session : on
# ressort jusqu'à la liste des modèles puis on rouvre le modèle
bt_key Escape
bt_select 'Alpha' && bt_key Enter
bt_action 'Train'
bt_key Enter                      # « Start from random weights »
bt_wait_for 'Training Parameters' 10 || fail "formulaire d'entraînement non atteint" "$SPEC_FRESH"
bt_key Down; bt_key Down          # champ Steps
bt_backspace 6; bt_type '20'
bt_key Enter                      # -> Training Dataset
bt_wait_for 'Training Dataset' 10 || fail "écran dataset non atteint" "$SPEC_FRESH"
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '.batraw'; then ok "le dataset déposé est proposé"
else fail "le dataset déposé n'apparaît pas sur l'écran Training Dataset" "$SPEC_FRESH"; bt_dump_screen "$S"; fi
bt_key Enter                      # lancer le run

# attendre la fin du run : le checkpoint doit apparaître sur disque
DEADLINE=$((SECONDS+240))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  [ -f "$PW/latest.ckpt" ] && break
  sleep 2
done
if [ -f "$PW/latest.ckpt" ]; then ok "le run d'entraînement a écrit latest.ckpt"
else fail "aucun latest.ckpt écrit après le run d'entraînement" "$SPEC_FRESH"; fi
sleep 3
[ -f "$PW/latest_metrics.jsonl" ] && ok "le run dépose bien un latest_metrics.jsonl à côté (le leurre est réel)" \
  || skip "pas de latest_metrics.jsonl déposé par ce run"

# [r] new run : le menu d'actions doit relire le disque
bt_key r
bt_redraw
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF 'latest.ckpt'; then
  ok "le menu d'actions relu voit le checkpoint fraîchement écrit"
else
  fail "le menu d'actions ne voit pas latest.ckpt après le run qui vient de l'écrire" "$SPEC_FRESH"
  bt_dump_screen "$S"
fi
if printf '%s' "$S" | grep -qE '2 checkpoints: (latest\.ckpt, real_one\.ckpt|real_one\.ckpt, latest\.ckpt)'; then
  ok "le compte est de 2 checkpoints, le latest_metrics.jsonl exclu"
else
  fail "le compte de checkpoints après le run n'est pas « 2 checkpoints » (metrics compté ?)" "$SPEC_CKPT"
  bt_dump_screen "$S"
fi

# et le sélecteur de poids aussi
bt_action 'Infer'
S="$(bt_screen)"
assert_contains "$S" 'latest.ckpt' "le sélecteur de poids propose le checkpoint frais" "$SPEC_FRESH"
assert_not_contains "$S" 'latest_metrics.jsonl' "le sélecteur ne propose pas latest_metrics.jsonl" "$SPEC_CKPT"

bt_quit
bt_summary
