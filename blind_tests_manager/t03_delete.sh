#!/usr/bin/env bash
# P3 — Delete : re-frappe exacte exigée, frappe inexacte inoffensive, le
# répertoire entier part, rien d'autre sous la racine n'est touché ; « q » est
# du texte. Spec : MODEL_MANAGER.md §2.1 et §2.3.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P3 Delete"
bt_init_report
echo "== $BT_PROP =="

SPEC_EXACT='MODEL_MANAGER.md §2.3 verrou 1 : « Le nom doit être retapé à l’identique. Tant que ce n’est pas le cas, Enter refuse et affiche ✗ Type the model name exactly — ''<nom>'' — to confirm. »'
SPEC_ARMED='MODEL_MANAGER.md §2.3 verrou 1 : « La barre d’aide bascule sur [Enter] DELETE — no further prompt seulement quand la frappe correspond : l’écran dit lui-même quand il est armé. »'
SPEC_WHOLE='MODEL_MANAGER.md §2.3 : « La suppression efface ensuite le dossier entier — config, checkpoints, métriques. »'
SPEC_Q='MODEL_MANAGER.md §2.1 : « Sur les deux écrans du manager, q est du texte, pas une sortie : un modèle peut légitimement s’appeler q-experiment. C’est Esc qui annule. »'
SPEC_LINK='MODEL_MANAGER.md §2.3 verrou 3 : « le chemin résolu doit être un enfant direct de Models/, les deux côtés canonicalisés. Un dossier de modèle qui est un lien symbolique est refusé, pas suivi. »'

ROOT="$(bt_new_root p3)"
VICTIM='q-experiment.v1_2'
bt_fixture "$ROOT" Greyscale_Diffusion "$VICTIM"
bt_fixture "$ROOT" Greyscale_Diffusion_broken Bystander
mkdir -p "$ROOT/Models/$VICTIM/pretrained_weights"
head -c 512 /dev/urandom > "$ROOT/Models/$VICTIM/pretrained_weights/latest.ckpt"
echo '{"step":0}' > "$ROOT/Models/$VICTIM/pretrained_weights/latest_metrics.jsonl"
mkdir -p "$ROOT/Models/Bystander/pretrained_weights"
head -c 512 /dev/urandom > "$ROOT/Models/Bystander/pretrained_weights/keep.ckpt"
# des témoins ailleurs sous la racine
mkdir -p "$ROOT/datasets" "$ROOT/perpetual_samples"
echo 'témoin' > "$ROOT/datasets/witness.txt"
echo 'témoin' > "$ROOT/perpetual_samples/witness.txt"
python3 -c "import sys;json=__import__('json');p=sys.argv[1];d=json.load(open(p));d['model_name']='$VICTIM';json.dump(d,open(p,'w'),indent=2)" "$ROOT/Models/$VICTIM/config_file"

BEFORE="$(bt_snapshot "$ROOT")"

bt_launch p3 "$ROOT" || { fail "TUI non démarré" "$SPEC_EXACT"; bt_summary; exit 1; }

open_delete() {  # depuis la liste, sur le modèle victime (2e entrée, alpha : Bystander < q-…)
  bt_key Down            # Bystander -> victime
  bt_key Enter           # menu d'actions
  bt_key Down; bt_key Down; bt_key Down   # Infer -> Perpetual -> Rename -> Delete
  bt_key Enter
}
open_delete
S="$(bt_screen)"
assert_contains "$S" 'Delete Model' "l'action Delete ouvre l'écran « Delete Model »" "$SPEC_EXACT"
assert_contains "$S" "$VICTIM" "l'écran nomme le modèle visé" "$SPEC_EXACT"

# la barre d'aide n'est PAS armée tant que rien n'est tapé
assert_not_contains "$S" '[Enter] DELETE' "à l'ouverture, l'écran n'est pas armé" "$SPEC_ARMED"

# --- « q » est du texte, pas une sortie
bt_type 'q'
if bt_alive; then ok "taper « q » sur l'écran Delete ne quitte pas l'application"
else fail "taper « q » a quitté l'application (code $(bt_exit_code))" "$SPEC_Q"; bt_summary; exit 1; fi
S="$(bt_screen)"
assert_contains "$S" 'Delete Model' "après « q » on est toujours sur l'écran Delete" "$SPEC_Q"

# --- frappe inexacte : préfixe du nom
bt_type '-experiment.v1'   # champ = « q-experiment.v1 » : un préfixe
S="$(bt_screen)"
assert_not_contains "$S" '[Enter] DELETE' "sur une frappe incomplète l'écran reste désarmé" "$SPEC_ARMED"
bt_key Enter
sleep 0.5
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF "✗ Type the model name exactly"; then
  ok "une frappe incomplète est refusée avec le message attendu"
else
  fail "pas de message « ✗ Type the model name exactly — '<nom>' — to confirm. » sur frappe incomplète" "$SPEC_EXACT"
  bt_dump_screen "$S"
fi
[ -d "$ROOT/Models/$VICTIM" ] && ok "frappe incomplète : le modèle est intact" \
  || fail "frappe incomplète : le modèle a été supprimé" "$SPEC_EXACT"

# --- frappe inexacte : casse différente
bt_backspace 40; bt_type 'Q-EXPERIMENT.V1_2'
S="$(bt_screen)"
assert_not_contains "$S" '[Enter] DELETE' "une frappe de casse différente laisse l'écran désarmé" "$SPEC_ARMED"
bt_key Enter; sleep 0.5
[ -d "$ROOT/Models/$VICTIM" ] && ok "casse différente : le modèle est intact" \
  || fail "casse différente : le modèle a été supprimé" "$SPEC_EXACT"

# --- frappe inexacte : nom d'un AUTRE modèle
bt_backspace 40; bt_type 'Bystander'; bt_key Enter; sleep 0.5
[ -d "$ROOT/Models/Bystander" ] && ok "taper le nom d'un autre modèle ne le supprime pas" \
  || fail "taper le nom d'un autre modèle l'a supprimé" "$SPEC_EXACT"
[ -d "$ROOT/Models/$VICTIM" ] && ok "taper le nom d'un autre modèle ne supprime pas la victime non plus" \
  || fail "le modèle visé a été supprimé sur la frappe d'un autre nom" "$SPEC_EXACT"

MID="$(bt_snapshot "$ROOT")"
if [ "$BEFORE" = "$MID" ]; then ok "aucune frappe inexacte n'a modifié le disque"
else fail "une frappe inexacte a modifié le disque" "$SPEC_EXACT"
  diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$MID") | sed 's/^/       /' | head -20
fi

# --- Esc annule (et ne quitte pas)
bt_key Escape
S="$(bt_screen)"
if bt_alive; then ok "Esc depuis Delete ne quitte pas l'application" \
  ; else fail "Esc depuis Delete a quitté l'application" "$SPEC_Q"; bt_summary; exit 1; fi
assert_contains "$S" "$VICTIM" "Esc depuis Delete remonte au menu d'actions du modèle" "$SPEC_Q"

# --- frappe exacte
bt_key Down; bt_key Down; bt_key Down; bt_key Enter   # rouvrir Delete
bt_type "$VICTIM"
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '[Enter] DELETE — no further prompt'; then
  ok "sur la frappe exacte, la barre d'aide s'arme"
else
  fail "la barre d'aide ne bascule pas sur « [Enter] DELETE — no further prompt »" "$SPEC_ARMED"
  bt_dump_screen "$S"
fi
bt_key Enter; sleep 1.5
S="$(bt_screen)"
assert_contains "$S" 'batlab — Models' "après suppression on revient à la liste" "$SPEC_WHOLE"
if printf '%s' "$S" | grep -qF "✓ Deleted '$VICTIM'"; then ok "l'accusé « ✓ Deleted » est affiché"
else fail "pas d'accusé « ✓ Deleted '$VICTIM' »" "$SPEC_WHOLE"; bt_dump_screen "$S"; fi
assert_not_contains "$(printf '%s' "$S" | grep -v '✓')" "$VICTIM" "le modèle a disparu des entrées de la liste" "$SPEC_WHOLE"

# --- effets disque : le répertoire entier, et RIEN d'autre
[ -e "$ROOT/Models/$VICTIM" ] && fail "le dossier du modèle subsiste après suppression" "$SPEC_WHOLE" \
  || ok "le dossier du modèle a été effacé en entier (config, poids, métriques)"

AFTER="$(bt_snapshot "$ROOT")"
LOST="$(diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") | grep '^<' | grep -v "Models/$VICTIM/")"
GAINED="$(diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") | grep '^>')"
if [ -z "$LOST" ]; then ok "rien d'autre que le modèle visé n'a disparu de la racine"
else fail "des fichiers hors du modèle visé ont disparu" "$SPEC_WHOLE"; printf '%s\n' "$LOST" | sed 's/^/       /'; fi
if [ -z "$GAINED" ]; then ok "la suppression n'a rien créé sous la racine"
else fail "la suppression a créé des fichiers" "$SPEC_WHOLE"; printf '%s\n' "$GAINED" | sed 's/^/       /'; fi

bt_quit

# --- verrou 3 : un dossier de modèle qui est un lien symbolique
ROOT2="$(bt_new_root p3-symlink)"
OUTSIDE="$BT_WORK/p3-outside"
rm -rf "$OUTSIDE"; mkdir -p "$OUTSIDE/pretrained_weights"
cp "$BT_REPO/Models/Greyscale_Diffusion/config_file" "$OUTSIDE/config_file"
python3 -c "import sys,json;p=sys.argv[1];d=json.load(open(p));d['model_name']='Linked_Model';json.dump(d,open(p,'w'),indent=2)" "$OUTSIDE/config_file"
echo 'précieux' > "$OUTSIDE/precious.txt"
ln -s "$OUTSIDE" "$ROOT2/Models/Linked_Model"
OUT_BEFORE="$(bt_snapshot "$OUTSIDE")"

bt_launch p3sym "$ROOT2" || { fail "TUI non démarré (symlink)" "$SPEC_LINK"; bt_summary; exit 1; }
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF 'Linked_Model'; then
  bt_key Enter
  bt_key Down; bt_key Down; bt_key Down; bt_key Enter
  S="$(bt_screen)"
  if printf '%s' "$S" | grep -qF 'Delete Model'; then
    bt_type 'Linked_Model'; bt_key Enter; sleep 1.5
    S="$(bt_screen)"
    OUT_AFTER="$(bt_snapshot "$OUTSIDE")"
    if [ "$OUT_BEFORE" = "$OUT_AFTER" ]; then
      ok "un modèle-lien-symbolique : la cible hors de Models/ est intacte (le lien n'est pas suivi)"
    else
      fail "la suppression a suivi le lien symbolique et effacé des fichiers hors de Models/" "$SPEC_LINK"
      diff <(printf '%s\n' "$OUT_BEFORE") <(printf '%s\n' "$OUT_AFTER") | sed 's/^/       /'
    fi
    if printf '%s' "$S" | grep -qF '✗'; then
      ok "la suppression d'un modèle-lien-symbolique est refusée avec un message"
    elif [ -L "$ROOT2/Models/Linked_Model" ]; then
      fail "aucun message de refus, mais le lien est toujours là — le refus est muet" "$SPEC_LINK"
      bt_dump_screen "$S"
    else
      fail "le lien symbolique a été supprimé sans refus alors que la spec dit « refusé, pas suivi »" "$SPEC_LINK"
      bt_dump_screen "$S"
    fi
  else
    fail "impossible d'atteindre l'écran Delete pour le modèle-lien" "$SPEC_LINK"; bt_dump_screen "$S"
  fi
else
  skip "un dossier de modèle qui est un lien symbolique n'apparaît pas dans la liste — verrou 3 non atteignable au TUI"
fi
bt_quit

bt_summary
