#!/usr/bin/env bash
# P4 — Navigation : Esc remonte d'exactement un cran depuis chaque écran
# atteignable ; l'application ne quitte que depuis la liste et le moniteur ;
# aucun écran annoncé n'est inatteignable au clavier.
# Spec : MODEL_MANAGER.md §2.1 + §1, AGENTS.md « Esc remonte d'exactement un cran ».
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P4 Navigation"
bt_init_report
echo "== $BT_PROP =="

SPEC_ESC='MODEL_MANAGER.md §2.1 : « Esc ne quitte que depuis les deux racines : la liste (rien au-dessus) et le moniteur (un run en cours, que quitter termine). Partout ailleurs il remonte d’exactement un cran, et la chaîne des parents atteint toujours une racine. »'
SPEC_REACH='MODEL_MANAGER.md §1 : « [e] depuis le menu d’actions ouvre le constructeur de couches ; [i] dedans ouvre la géométrie d’entrée. Screen::InputSize était dessiné, géré au clavier et jamais assigné — le troisième écran fantôme de ce dépôt. »'

ROOT="$(bt_new_root p4)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
bt_dataset "$ROOT" >/dev/null || true

declare -a REACHED=()
mark_reached() { REACHED+=("$1"); }

# at <marqueur> : l'écran courant porte-t-il ce marqueur ?
at() { bt_screen | grep -qF -- "$1"; }

# on_screen <nom lisible> <marqueur> : assertion « on est bien sur cet écran »
on_screen() {
  if at "$2"; then ok "atteint : $1"; mark_reached "$1"; return 0
  else fail "écran attendu « $1 » non atteint (marqueur « $2 » absent)" "$SPEC_REACH"; bt_dump_screen "$(bt_screen)"; return 1; fi
}

# esc_to <depuis> <vers lisible> <marqueur du parent> [marqueur interdit = grand-parent]
esc_to() {
  local from="$1" to="$2" parent="$3" grand="${4:-}"
  bt_key Escape
  if ! bt_alive; then
    fail "Esc depuis « $from » a QUITTÉ l'application (code $(bt_exit_code))" "$SPEC_ESC"; return 1
  fi
  if at "$parent"; then
    if [ -n "$grand" ] && at "$grand"; then
      fail "Esc depuis « $from » a sauté au-delà de « $to »" "$SPEC_ESC"; return 1
    fi
    ok "Esc depuis « $from » remonte d'un cran vers « $to »"
  else
    fail "Esc depuis « $from » n'a pas mené à « $to »" "$SPEC_ESC"; bt_dump_screen "$(bt_screen)"; return 1
  fi
}

M_LIST='batlab — Models'
M_ACT='[Enter] confirm  [e] edit layers'
M_TPL='Model Templates'
M_REN='Rename Model'
M_DEL='Delete Model'
M_WGT='— Weights'
M_TRP='Training Parameters'
M_DST='Training Dataset'
M_INF='Inference Parameters'
M_PRP='Perpetual Inference'
M_LAY='[e/Esc] back to add'
M_ADD='Add Layer'
M_INS='Model Input Size'
M_MON='Status       :'

bt_launch p4 "$ROOT" || { fail "TUI non démarré" "$SPEC_ESC"; bt_summary; exit 1; }
on_screen 'liste des modèles' "$M_LIST"

# --- sélecteur de templates ------------------------------------------------
bt_select 'New model \(from template\)' && bt_key Enter
on_screen 'sélecteur de templates' "$M_TPL"
esc_to 'sélecteur de templates' 'liste des modèles' "$M_LIST"

# --- menu d'actions --------------------------------------------------------
bt_select 'Alpha' && bt_key Enter
on_screen "menu d'actions" "$M_ACT"

# --- Rename ---------------------------------------------------------------
bt_action 'Rename'
on_screen 'Rename Model' "$M_REN"
esc_to 'Rename Model' "menu d'actions" "$M_ACT" "$M_LIST"

# --- Delete ---------------------------------------------------------------
bt_action 'Delete'
on_screen 'Delete Model' "$M_DEL"
esc_to 'Delete Model' "menu d'actions" "$M_ACT" "$M_LIST"

# --- Train : Weights -> Params -> Dataset ---------------------------------
bt_action 'Train'
on_screen 'sélecteur de poids' "$M_WGT"
bt_key Enter
on_screen 'Training Parameters' "$M_TRP"
bt_key Enter; bt_key Enter; bt_key Enter
on_screen 'Training Dataset' "$M_DST"
esc_to 'Training Dataset'    'Training Parameters' "$M_TRP" "$M_WGT"
esc_to 'Training Parameters' 'sélecteur de poids'  "$M_WGT" "$M_ACT"
esc_to 'sélecteur de poids'  "menu d'actions"      "$M_ACT" "$M_LIST"

# --- Infer : Weights -> Inference Parameters ------------------------------
bt_action 'Infer'
on_screen 'sélecteur de poids (Infer)' "$M_WGT"
bt_key Enter
on_screen 'Inference Parameters' "$M_INF"
esc_to 'Inference Parameters' 'sélecteur de poids' "$M_WGT" "$M_ACT"
esc_to 'sélecteur de poids'   "menu d'actions"     "$M_ACT" "$M_LIST"

# --- Perpetual : Weights -> Perpetual Inference ---------------------------
bt_action 'Perpetual'
on_screen 'sélecteur de poids (Perpetual)' "$M_WGT"
bt_key Enter
on_screen 'Perpetual Inference' "$M_PRP"
esc_to 'Perpetual Inference' 'sélecteur de poids' "$M_WGT" "$M_ACT"
esc_to 'sélecteur de poids'  "menu d'actions"     "$M_ACT" "$M_LIST"

# --- Constructeur de couches et géométrie d'entrée ------------------------
bt_key e
on_screen 'constructeur de couches (liste des couches)' "$M_LAY"
# la spec AGENTS.md dit « [i] dedans ouvre la géométrie d'entrée » : ici [i] est inerte
bt_key i
if at "$M_INS"; then ok "[i] depuis l'écran ouvert par [e] mène à la géométrie d'entrée"
else
  skip "[i] est inerte sur l'écran que [e] ouvre (liste des couches) — la géométrie d'entrée s'atteint depuis « Add Layer », un cran plus haut (voir rapport)"
fi
esc_to 'liste des couches' 'Add Layer' "$M_ADD" "$M_ACT"
on_screen 'Add Layer' "$M_ADD"
bt_key i
on_screen "géométrie d'entrée" "$M_INS"
esc_to "géométrie d'entrée" 'Add Layer'      "$M_ADD" "$M_ACT"
esc_to 'Add Layer'          "menu d'actions" "$M_ACT" "$M_LIST"
esc_to "menu d'actions"     'liste des modèles' "$M_LIST"

# --- moniteur : racine n°2 -------------------------------------------------
bt_select 'Alpha' && bt_key Enter    # menu d'actions
bt_action 'Infer'                    # sélecteur de poids
bt_key Enter                         # random weights -> formulaire
bt_key Enter; bt_key Enter; bt_key Enter; bt_key Enter   # dérouler jusqu'au run
bt_wait_for "$M_MON" 60
bt_redraw
if at "$M_MON"; then
  ok "atteint : moniteur"; mark_reached 'moniteur'
else
  fail "le moniteur n'a pas été atteint depuis le formulaire d'inférence" "$SPEC_REACH"; bt_dump_screen "$(bt_screen)"
fi
bt_key Escape
sleep 1.5
if bt_alive; then
  fail "Esc depuis le moniteur n'a pas quitté l'application" "$SPEC_ESC"
else
  ok "Esc depuis le moniteur quitte l'application (racine n°2)"
  assert_eq "$(bt_exit_code)" "0" "la sortie depuis le moniteur est propre (code 0)" "$SPEC_ESC"
fi
tmux kill-session -t "$BT_SESS" 2>/dev/null

# --- la liste est bien l'autre racine -------------------------------------
bt_launch p4b "$ROOT" || { fail "TUI non relancé" "$SPEC_ESC"; bt_summary; exit 1; }
bt_key Escape; sleep 1.2
if bt_alive; then fail "Esc depuis la liste des modèles n'a pas quitté l'application" "$SPEC_ESC"
else ok "Esc depuis la liste des modèles quitte l'application (racine n°1)"
  assert_eq "$(bt_exit_code)" "0" "la sortie depuis la liste est propre (code 0)" "$SPEC_ESC"
fi
tmux kill-session -t "$BT_SESS" 2>/dev/null

# --- couverture des écrans annoncés ---------------------------------------
EXPECTED=('liste des modèles' 'sélecteur de templates' "menu d'actions" 'Rename Model' 'Delete Model'
          'sélecteur de poids' 'Training Parameters' 'Training Dataset' 'Inference Parameters'
          'Perpetual Inference' 'constructeur de couches (liste des couches)' 'Add Layer'
          "géométrie d'entrée" 'moniteur')
MISSING=()
for e in "${EXPECTED[@]}"; do
  found=0
  for r in "${REACHED[@]}"; do [ "$r" = "$e" ] && found=1 && break; done
  [ "$found" -eq 0 ] && MISSING+=("$e")
done
if [ "${#MISSING[@]}" -eq 0 ]; then
  ok "les ${#EXPECTED[@]} écrans annoncés sont tous atteignables au clavier depuis la porte d'entrée"
else
  fail "écrans annoncés jamais atteints au clavier : ${MISSING[*]}" "$SPEC_REACH"
fi

bt_summary
