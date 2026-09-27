#!/usr/bin/env bash
# P1 — Flow : l'ouverture est la liste des modèles ; un modèle ouvre son menu d'actions.
# Spec : AGENTS.md « Ouverture → LISTE DES MODÈLES (l'écran d'accueil) » et
#        MODEL_MANAGER.md §1 / §2.1.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P1 Flow"
bt_init_report
echo "== $BT_PROP =="

SPEC_LIST='MODEL_MANAGER.md §1 : « Ouverture → LISTE DES MODÈLES (racine n°1) / chaque entrée : nom · géométrie · nb de couches · checkpoints / dernière ligne : « New model (from template) » »'
SPEC_ACT='MODEL_MANAGER.md §1 : « → Entrée sur un modèle → MENU D’ACTIONS / Train / Infer / Perpetual → … / Rename / Delete → le manager »'
SPEC_TITLE='MODEL_MANAGER.md §2.1 : Liste des modèles, titre affiché « batlab — Models » ; Menu d’actions, titre affiché « <nom du modèle> »'

ROOT="$(bt_new_root p1)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha_Model
bt_fixture "$ROOT" Greyscale_Diffusion_broken Beta_Model
mkdir -p "$ROOT/Models/Alpha_Model/pretrained_weights"
head -c 2048 /dev/urandom > "$ROOT/Models/Alpha_Model/pretrained_weights/one.ckpt"

bt_launch p1 "$ROOT" || { fail "le TUI n'a pas affiché la liste des modèles au démarrage" "$SPEC_TITLE"; bt_summary; exit 1; }
S="$(bt_screen)"

assert_contains "$S" 'batlab — Models'              "l'écran d'accueil est la liste des modèles" "$SPEC_TITLE"
assert_contains "$S" 'Alpha_Model'                  "la liste montre Alpha_Model"                "$SPEC_LIST"
assert_contains "$S" 'Beta_Model'                   "la liste montre Beta_Model"                 "$SPEC_LIST"
assert_contains "$S" 'New model (from template)'    "la dernière ligne est « New model (from template) »" "$SPEC_LIST"

# géométrie · couches · checkpoints, sur la ligne de détail de chaque modèle
if printf '%s' "$S" | grep -qE '[0-9]+x[0-9]+x[0-9]+ · [0-9]+ layers · '; then
  ok "chaque entrée porte géométrie · nb de couches · checkpoints"
else
  fail "aucune ligne « NxNxN · N layers · … » dans la liste" "$SPEC_LIST"; bt_dump_screen "$S"
fi
assert_contains "$S" '1 checkpoint: one.ckpt' "le compteur de checkpoints d'Alpha_Model est exact" "$SPEC_LIST"
assert_contains "$S" 'no checkpoints'         "Beta_Model, sans poids, est annoncé « no checkpoints »" "$SPEC_LIST"

# « New model (from template) » est bien la DERNIÈRE entrée
LASTENTRY="$(printf '%s' "$S" | grep -nE '(Alpha_Model|Beta_Model|New model \(from template\))' | tail -1)"
case "$LASTENTRY" in
  *"New model (from template)"*) ok "« New model (from template) » est la dernière ligne de la liste" ;;
  *) fail "« New model (from template) » n'est pas la dernière entrée (dernière vue : $LASTENTRY)" "$SPEC_LIST" ;;
esac

# --- Entrée sur un modèle → menu d'actions
bt_key Enter
S="$(bt_screen)"
assert_contains "$S" 'Alpha_Model' "le menu d'actions est titré du nom du modèle" "$SPEC_TITLE"
for a in Train Infer Perpetual Rename Delete; do
  assert_contains "$S" "$a" "le menu d'actions propose « $a »" "$SPEC_ACT"
done
assert_not_contains "$S" 'batlab — Models' "le menu d'actions a remplacé la liste" "$SPEC_ACT"

# le menu redit la géométrie du modèle ouvert
if printf '%s' "$S" | grep -qE '[0-9]+x[0-9]+x[0-9]+ · [0-9]+ layers'; then
  ok "le menu d'actions rappelle géométrie et nb de couches"
else
  fail "le menu d'actions ne rappelle pas la géométrie" "$SPEC_ACT"; bt_dump_screen "$S"
fi

# --- « New model (from template) » mène au sélecteur de templates
bt_key Escape
bt_key Down; bt_key Down
S="$(bt_screen)"
bt_key Enter
S2="$(bt_screen)"
if [ "$S" != "$S2" ] && ! printf '%s' "$S2" | grep -qF 'batlab — Models'; then
  ok "« New model (from template) » ouvre un écran de templates"
else
  fail "« New model (from template) » n'ouvre rien" "$SPEC_LIST"; bt_dump_screen "$S2"
fi

bt_quit
bt_summary
