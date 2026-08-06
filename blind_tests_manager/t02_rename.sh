#!/usr/bin/env bash
# P2 — Rename : renomme le dossier ET le model_name du config_file, de façon
# cohérente ; la liste reflète le nouveau nom ; validation des noms.
# Spec : MODEL_MANAGER.md §2.2 et §2.3 verrou 2.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="P2 Rename"
bt_init_report
echo "== $BT_PROP =="

SPEC_MOVE='MODEL_MANAGER.md §2.2 : « Déplace Models/<ancien>/ → Models/<nouveau>/. Et réécrit le config_file : model_name, plus tout chemin de checkpoint qui pointait à l’intérieur du dossier du modèle. »'
SPEC_PREFILL='MODEL_MANAGER.md §2.2 : « Le formulaire s’ouvre prérempli sur le nom courant. »'
SPEC_ACK='MODEL_MANAGER.md §2.2 : « Au retour, la liste affiche le nouveau nom et un accusé ✓ Renamed ''x'' → ''y''. »'
SPEC_VALID='MODEL_MANAGER.md §2.3 verrou 2 : « Nom validable — alphanumérique plus -_., pas de . en tête, ≤ 64 caractères. Une traversée de répertoire n’est même pas épelable. »'
SPEC_ATOMIC='MODEL_MANAGER.md §2.2 : « Si la réécriture du config échoue, le renommage est défait : jamais de paire dossier/config à moitié renommée. »'

ROOT="$(bt_new_root p2)"
bt_fixture "$ROOT" Greyscale_Diffusion Alpha_Model
bt_fixture "$ROOT" Greyscale_Diffusion_broken Occupied_Name
mkdir -p "$ROOT/Models/Alpha_Model/pretrained_weights"
head -c 1024 /dev/urandom > "$ROOT/Models/Alpha_Model/pretrained_weights/w.ckpt"
# un chemin de checkpoint qui pointe DANS le dossier du modèle : la spec exige
# qu'il soit réécrit par le renommage
python3 - "$ROOT/Models/Alpha_Model/config_file" "$ROOT" <<'PY'
import json,sys
p,root=sys.argv[1],sys.argv[2]
d=json.load(open(p))
d.setdefault('inference',{})['checkpoint']=f"{root}/Models/Alpha_Model/pretrained_weights/w.ckpt"
json.dump(d,open(p,'w'),indent=2)
PY

BEFORE="$(bt_snapshot "$ROOT")"
OCC_SHA="$(shasum -a 256 "$ROOT/Models/Occupied_Name/config_file" | cut -d' ' -f1)"
bt_launch p2 "$ROOT" || { fail "TUI non démarré" "$SPEC_MOVE"; bt_summary; exit 1; }

open_rename() {  # depuis la liste, sur le 1er modèle
  bt_key Enter          # menu d'actions
  bt_key Down; bt_key Down   # Infer -> Perpetual -> Rename (curseur initial : Infer)
  bt_key Enter
}
open_rename
S="$(bt_screen)"
assert_contains "$S" 'Rename Model' "l'action Rename ouvre l'écran « Rename Model »" "$SPEC_MOVE"
assert_contains "$S" 'New name     : Alpha_Model' "le champ s'ouvre prérempli sur le nom courant" "$SPEC_PREFILL"

# --- validation : traversée de répertoire
bt_backspace 40; bt_type '../evil'; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "'../evil' est refusé avec un message d'erreur"
else fail "'../evil' n'a pas été refusé" "$SPEC_VALID"; bt_dump_screen "$S"; fi
[ -d "$ROOT/Models/Alpha_Model" ] && ok "'../evil' : le modèle est intact sur disque" \
  || fail "'../evil' : le dossier du modèle a disparu" "$SPEC_VALID"
[ -e "$ROOT/evil" ] && fail "'../evil' a écrit hors de Models/ : $ROOT/evil existe" "$SPEC_VALID" \
  || ok "'../evil' n'a rien créé hors de Models/"

# --- validation : nom vide
bt_backspace 40; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "un nom vide est refusé"
else fail "un nom vide n'a pas été refusé" "$SPEC_VALID"; bt_dump_screen "$S"; fi

# --- validation : collision
bt_type 'Occupied_Name'; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "un nom déjà pris est refusé"
else fail "un nom déjà pris n'a pas été refusé" "$SPEC_VALID"; bt_dump_screen "$S"; fi
[ -d "$ROOT/Models/Alpha_Model" ] && ok "collision : le modèle source est intact" \
  || fail "collision : le dossier source a disparu" "$SPEC_VALID"
OCC_NOW="$(shasum -a 256 "$ROOT/Models/Occupied_Name/config_file" | cut -d' ' -f1)"
assert_eq "$OCC_NOW" "$OCC_SHA" "collision : le config_file du modèle homonyme est intact" "$SPEC_VALID"

# --- validation : nom > 64 caractères
LONG="$(printf 'a%.0s' $(seq 1 65))"
bt_backspace 40; bt_type "$LONG"; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "un nom de 65 caractères est refusé"
else fail "un nom de 65 caractères a été accepté" "$SPEC_VALID"; bt_dump_screen "$S"; fi

# --- validation : point en tête
bt_backspace 80; bt_type '.hidden'; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "un nom commençant par '.' est refusé"
else fail "un nom commençant par '.' a été accepté" "$SPEC_VALID"; bt_dump_screen "$S"; fi

# --- validation : séparateur de chemin nu
bt_backspace 40; bt_type 'a/b'; bt_key Enter
S="$(bt_screen)"
if printf '%s' "$S" | grep -qF '✗'; then ok "un nom contenant '/' est refusé"
else fail "un nom contenant '/' a été accepté" "$SPEC_VALID"; bt_dump_screen "$S"; fi
[ -e "$ROOT/Models/a/b" ] && fail "'a/b' a créé une arborescence imbriquée" "$SPEC_VALID" \
  || ok "'a/b' n'a créé aucune arborescence imbriquée"

# rien n'a bougé sur disque après six refus
AFTER_REFUS="$(bt_snapshot "$ROOT")"
if [ "$BEFORE" = "$AFTER_REFUS" ]; then ok "aucun refus n'a modifié le disque"
else fail "un nom refusé a tout de même modifié le disque" "$SPEC_VALID"
  diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER_REFUS") | sed 's/^/       /' | head -20
fi

# --- renommage valide
bt_backspace 40; bt_type 'q-experiment.v1_2'; bt_key Enter
sleep 1
S="$(bt_screen)"
assert_contains "$S" 'batlab — Models' "après renommage on revient à la liste" "$SPEC_ACK"
assert_contains "$S" 'q-experiment.v1_2' "la liste affiche le nouveau nom" "$SPEC_ACK"
# l'accusé cite légitimement l'ancien nom : on ne regarde que les entrées de la liste
ENTRIES="$(printf '%s' "$S" | grep -v '✓')"
assert_not_contains "$ENTRIES" 'Alpha_Model' "l'ancien nom a disparu des entrées de la liste" "$SPEC_ACK"
if printf '%s' "$S" | grep -qF "✓ Renamed 'Alpha_Model' → 'q-experiment.v1_2'"; then
  ok "l'accusé « ✓ Renamed 'x' → 'y' » est affiché"
else
  fail "pas d'accusé « ✓ Renamed 'Alpha_Model' → 'q-experiment.v1_2' »" "$SPEC_ACK"; bt_dump_screen "$S"
fi

# --- effets disque
[ -d "$ROOT/Models/q-experiment.v1_2" ] && ok "le dossier a été déplacé vers le nouveau nom" \
  || fail "Models/q-experiment.v1_2/ n'existe pas" "$SPEC_MOVE"
[ -e "$ROOT/Models/Alpha_Model" ] && fail "l'ancien dossier Models/Alpha_Model/ subsiste" "$SPEC_MOVE" \
  || ok "l'ancien dossier a disparu"
[ -f "$ROOT/Models/q-experiment.v1_2/pretrained_weights/w.ckpt" ] && ok "les poids ont suivi le dossier" \
  || fail "les poids n'ont pas suivi le dossier" "$SPEC_MOVE"

NEWNAME="$(bt_json_field "$ROOT/Models/q-experiment.v1_2/config_file" model_name)"
assert_eq "$NEWNAME" 'q-experiment.v1_2' "le config_file porte le nouveau model_name" "$SPEC_MOVE"

CKPT="$(bt_json_field "$ROOT/Models/q-experiment.v1_2/config_file" inference.checkpoint)"
case "$CKPT" in
  *"/Models/q-experiment.v1_2/pretrained_weights/w.ckpt") ok "le chemin de checkpoint interne a été réécrit" ;;
  *Alpha_Model*) fail "le chemin de checkpoint pointe encore l'ancien dossier : $CKPT" "$SPEC_MOVE" ;;
  *) fail "chemin de checkpoint inattendu après renommage : « $CKPT »" "$SPEC_MOVE" ;;
esac
if [ -n "$CKPT" ] && [ ! -e "$CKPT" ]; then
  fail "le chemin de checkpoint réécrit ne désigne aucun fichier : $CKPT" "$SPEC_MOVE"
elif [ -n "$CKPT" ]; then ok "le chemin de checkpoint réécrit désigne un fichier existant"; fi

# --- cohérence : le modèle renommé se rouvre et reste manipulable
bt_key Enter
S="$(bt_screen)"
assert_contains "$S" 'q-experiment.v1_2' "le modèle renommé se rouvre sous son nouveau nom" "$SPEC_MOVE"

# --- pas de résidu : rien d'autre n'a changé sous la racine
AFTER="$(bt_snapshot "$ROOT")"
UNEXPECTED="$(diff <(printf '%s\n' "$BEFORE" | sed 's|^\./Models/Alpha_Model/|@MODEL@/|'       | LC_ALL=C sort) \
                   <(printf '%s\n' "$AFTER"  | sed 's|^\./Models/q-experiment\.v1_2/|@MODEL@/|' | LC_ALL=C sort) \
              | grep -E '^[<>]' | grep -v 'config_file')"
if [ -z "$UNEXPECTED" ]; then ok "hors config_file, le contenu du modèle est bit-à-bit identique et rien d'autre n'a bougé"
else fail "des fichiers autres que le config_file ont changé pendant le renommage" "$SPEC_ATOMIC"
  printf '%s\n' "$UNEXPECTED" | sed 's/^/       /' | head -20
fi

bt_quit
bt_summary
