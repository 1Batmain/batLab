#!/usr/bin/env bash
# t08 — Sondes hostiles : les coins que la spec affirme sans les dérouler.
# Racine vide, garde de run (§2.4), casse du système de fichiers, [r] refresh,
# « q » comme texte sur Rename.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="Sondes hostiles"
bt_init_report
echo "== $BT_PROP =="

SPEC_NOCREATE='MODEL_MANAGER.md §3 : « list_models ne crée plus rien : regarder une racine n’y laisse plus un Models/. » et §5 pas 1 : « Ouvrir sur une racine vide → liste, message « No models yet », aucun Models/ créé »'
SPEC_DSCREATE='MODEL_MANAGER.md §6 : « Ouvrir une racine y crée datasets/ alors que Models/ n’est créé qu’au premier modèle. […] Bénin, mais asymétrique. »'
SPEC_GUARD='MODEL_MANAGER.md §2.4 : « Rename et Delete refusent tous deux un modèle dont ce processus tient un run. Cette garde est de l’état TUI (running_model, posé au démarrage d’un run, levé quand le run se déclare fini). »'
SPEC_Q='MODEL_MANAGER.md §2.1 : « Sur les deux écrans du manager, q est du texte, pas une sortie […]. C’est Esc qui annule. »'
SPEC_REFRESH='MODEL_MANAGER.md §2.1 : « Liste des modèles […] r rafraîchir »'
SPEC_VALID='MODEL_MANAGER.md §2.3 verrou 2 : « Nom validable — alphanumérique plus -_., pas de . en tête, ≤ 64 caractères. »'
SPEC_DSFREEZE='MODEL_MANAGER.md §6 : « La liste des datasets ne se rafraîchit pas en cours de session : un .batraw déposé pendant que le TUI tourne reste invisible jusqu’à ce qu’on ressorte au moins jusqu’à la liste des modèles et qu’on rouvre le modèle. »'

# --- 1. Racine vide : ce que regarder coûte ------------------------------
ROOT="$(bt_new_root h1)"
rmdir "$ROOT/Models" "$ROOT/datasets"     # racine réellement vide
bt_launch h1 "$ROOT" || { fail "TUI non démarré sur racine vide" "$SPEC_NOCREATE"; bt_summary; exit 1; }
S="$(bt_screen)"
assert_contains "$S" 'batlab — Models' "sur une racine vide, la liste s'ouvre quand même" "$SPEC_NOCREATE"
if printf '%s' "$S" | grep -qiF 'No models yet'; then ok "la racine vide affiche « No models yet »"
else fail "pas de message « No models yet » sur une racine vide" "$SPEC_NOCREATE"; bt_dump_screen "$S"; fi
[ -d "$ROOT/Models" ] && fail "regarder une racine vide y a créé Models/" "$SPEC_NOCREATE" \
  || ok "regarder une racine vide n'y crée pas Models/"
if [ -d "$ROOT/datasets" ]; then
  ok "l'asymétrie annoncée en §6 est confirmée : datasets/ EST créé à l'ouverture"
else
  spec_gap "datasets/ n'a PAS été créé à l'ouverture" \
    "la limite §6 (« Ouvrir une racine y crée datasets/ ») ne se reproduit pas — la spec est peut-être en avance ou en retard sur le binaire."
fi
bt_quit

# --- 2. « q » est du texte sur Rename aussi ------------------------------
ROOT="$(bt_new_root h2)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
bt_launch h2 "$ROOT" || { fail "TUI non démarré" "$SPEC_Q"; bt_summary; exit 1; }
bt_select 'Alpha' && bt_key Enter
bt_action 'Rename'
bt_backspace 40; bt_type 'q'
if bt_alive; then ok "taper « q » sur l'écran Rename ne quitte pas l'application"
else fail "taper « q » sur Rename a quitté l'application (code $(bt_exit_code))" "$SPEC_Q"; bt_summary; exit 1; fi
S="$(bt_screen)"
assert_contains "$S" 'New name     : q' "le « q » tapé est bien du texte dans le champ" "$SPEC_Q"

# --- 3. Casse : renommer vers une variante de casse d'un nom pris --------
bt_key Escape; bt_key Escape
bt_key r                            # rafraîchir la liste
sleep 0.5
mkdir -p "$ROOT/Models/Beta"
cp "$BT_REPO/Models/Greyscale_Diffusion_broken/config_file" "$ROOT/Models/Beta/config_file"
head -c 99 /dev/urandom > "$ROOT/Models/Beta/PRECIEUX.bin"
BETA_SHA="$(shasum -a 256 "$ROOT/Models/Beta/PRECIEUX.bin" | cut -d' ' -f1)"
bt_key r; sleep 0.8
S="$(bt_screen)"
assert_contains "$S" 'Beta' "[r] rafraîchit la liste : un modèle déposé pendant la session apparaît" "$SPEC_REFRESH"

bt_select 'Alpha' && bt_key Enter
bt_action 'Rename'
bt_backspace 40; bt_type 'beta'; bt_key Enter; sleep 1
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then
  ok "renommer « Alpha » en « beta » (variante de casse de « Beta ») est refusé"
else
  fail "renommer vers une variante de casse d'un nom pris n'a pas été refusé" "$SPEC_VALID"
  bt_dump_screen "$S"
fi
if [ -f "$ROOT/Models/Beta/PRECIEUX.bin" ] && \
   [ "$(shasum -a 256 "$ROOT/Models/Beta/PRECIEUX.bin" | cut -d' ' -f1)" = "$BETA_SHA" ]; then
  ok "le modèle « Beta » n'a pas été écrasé par la variante de casse (FS insensible à la casse)"
else
  fail "le contenu de Models/Beta a été détruit par un renommage en « beta »" "$SPEC_VALID"
fi

# --- 4. Renommer un modèle vers son propre nom ---------------------------
bt_backspace 40; bt_type 'Alpha'; bt_key Enter; sleep 1
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then
  ok "renommer un modèle vers son propre nom est refusé (collision avec lui-même)"
elif printf '%s' "$S" | grep -qF '✓ Renamed'; then
  ok "renommer un modèle vers son propre nom est accepté et sans effet"
else
  spec_gap "renommer un modèle vers son propre nom ne dit rien du tout" \
    "la spec §2.2/§2.3 ne tranche pas ce cas ; l'écran ne renvoie ni ✓ ni ✗."
  bt_dump_screen "$S"
fi
[ -d "$ROOT/Models/Alpha" ] && ok "après « renommer vers soi-même », le modèle existe toujours" \
  || fail "« renommer vers soi-même » a fait disparaître le modèle" "$SPEC_VALID"
bt_quit

# --- 5. La garde de run : observable au clavier ? ------------------------
ROOT="$(bt_new_root h5)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
DS="$(bt_dataset "$ROOT")" || DS=""
bt_launch h5 "$ROOT" || { fail "TUI non démarré" "$SPEC_GUARD"; bt_summary; exit 1; }
if [ -n "$DS" ]; then
  bt_select 'Alpha' && bt_key Enter
  bt_action 'Train'
  bt_key Enter
  bt_wait_for 'Training Parameters' 15
  bt_key Down; bt_key Down; bt_backspace 8; bt_type '200000'   # un run qui dure
  bt_key Enter; bt_wait_for 'Training Dataset' 15; bt_key Enter
  bt_wait_for 'training: running' 60
  bt_redraw
  FOOT="$(bt_screen | tail -3)"
  if printf '%s' "$FOOT" | grep -qF '[r] new run'; then
    bt_key r; sleep 1; bt_redraw
    if bt_screen | grep -qF '[Enter] confirm  [e] edit layers'; then
      bt_action 'Rename'; bt_backspace 40; bt_type 'Renamed_Under_Run'; bt_key Enter; sleep 1
      S="$(bt_screen)"
      if printf '%s' "$S" | grep -qF '✗'; then ok "Rename refuse un modèle dont ce processus tient un run"
      else fail "Rename a accepté un modèle dont un run est en cours" "$SPEC_GUARD"; bt_dump_screen "$S"; fi
    else
      spec_gap "[r] pendant un run ne mène pas au menu d'actions" "$SPEC_GUARD"
    fi
  else
    spec_gap "la garde de run §2.4 n'est PAS observable au clavier dans un seul processus" \
      "tant qu'un run tourne (même mis en pause par [p]), le moniteur n'offre aucun chemin vers le menu d'actions : son pied de page est « [p] pause/resume  [t] tune params  [v] visualise  [s] save snapshot  [q] quit » — [r] new run n'apparaît qu'une fois le run terminé, donc la garde levée. Le refus annoncé par $SPEC_GUARD est du code défensif que le TUI ne laisse pas atteindre."
  fi
  # le run tourne encore : on quitte franchement
  tmux send-keys -t "$BT_SESS" q; sleep 2
  tmux kill-session -t "$BT_SESS" 2>/dev/null
else
  skip "pas de .batraw : garde de run non sondée"
  bt_quit
fi

# --- 6. La garde ne reste pas coincée après un run qui a échoué ----------
ROOT="$(bt_new_root h6)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha
bt_launch h6 "$ROOT" || { fail "TUI non démarré" "$SPEC_GUARD"; bt_summary; exit 1; }
bt_select 'Alpha' && bt_key Enter
bt_action 'Infer'                       # sans poids : le run échoue tout de suite
bt_key Enter
bt_key Enter; bt_key Enter; bt_key Enter; bt_key Enter
bt_wait_for 'Status       :' 60
bt_key r; sleep 1; bt_redraw
bt_action 'Rename'
bt_backspace 40; bt_type 'Alpha_After_Failed_Run'; bt_key Enter; sleep 1.5
S="$(bt_screen)"
if [ -d "$ROOT/Models/Alpha_After_Failed_Run" ]; then
  ok "après un run en échec, la garde est bien levée : Rename fonctionne"
else
  fail "après un run en échec, Rename reste bloqué — la garde de run ne se lève pas" "$SPEC_GUARD"
  bt_dump_screen "$S"
fi
bt_quit

bt_summary
