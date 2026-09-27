#!/usr/bin/env bash
# t09 — Le flux « New model (from template) » : la géométrie qu'il écrit sur
# disque respecte-t-elle l'invariant de conditionnement temporel ?
# Spec : AGENTS.md « Points d'attention » — « Un modèle de diffusion DOIT être
# conditionné sur le timestep : input_size.z > output.z ».
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
BT_PROP="Géométrie des templates"
bt_init_report
echo "== $BT_PROP =="

SPEC_T='AGENTS.md, Points d’attention : « Un modèle de diffusion DOIT être conditionné sur le timestep : input_size.z > output.z (les canaux excédentaires reçoivent l’embedding temporel). Sans ça, ε̂ dégénère et l’échantillonnage explose en blanc saturé (voir docs/reports/INSIGHTS_TRAINING.md). »'
SPEC_FLOW='MODEL_MANAGER.md §1 : « dernière ligne : « New model (from template) » → sélecteur de templates » ; §5 pas 2 : « Enter → template Greyscale → Enter → modèle créé sur disque, menu d’actions → OK ».'

geom() { # <config_file> -> "in_z out_z"
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
last=list(d['layers'][-1].values())[0]
print(d['input_size'][2], last.get('nb_kernel'))
PY
}

for TPL in 'Greyscale Diffusion' 'Stable Diffusion'; do
  ROOT="$(bt_new_root "t09-$(printf '%s' "$TPL" | tr ' ' '_')")"
  bt_launch "t09$(printf '%s' "$TPL" | tr -dc 'A-Za-z')" "$ROOT" || { fail "TUI non démarré" "$SPEC_FLOW"; continue; }
  bt_select 'New model \(from template\)' && bt_key Enter
  if ! bt_select "$TPL"; then fail "template « $TPL » introuvable dans le sélecteur" "$SPEC_FLOW"; bt_quit; continue; fi
  bt_key Enter; sleep 1.5
  S="$(bt_screen)"
  CFG="$(ls "$ROOT"/Models/*/config_file 2>/dev/null | head -1)"
  if [ -z "$CFG" ]; then fail "le template « $TPL » n'a créé aucun config_file sur disque" "$SPEC_FLOW"; bt_quit; continue; fi
  ok "le template « $TPL » crée bien un modèle sur disque ($(basename "$(dirname "$CFG")"))"
  [ -d "$(dirname "$CFG")/pretrained_weights" ] && ok "le template crée aussi pretrained_weights/" \
    || fail "le template ne crée pas pretrained_weights/" "$SPEC_FLOW"
  read -r INZ OUTZ <<<"$(geom "$CFG")"
  echo "       input_size.z = $INZ, canaux de sortie = $OUTZ"
  if [ "$INZ" -gt "$OUTZ" ]; then
    ok "« $TPL » : input_size.z ($INZ) > output.z ($OUTZ) — conditionnement temporel possible"
  else
    fail "« $TPL » crée un modèle NON conditionnable sur le timestep : input_size.z=$INZ, output.z=$OUTZ (l'invariant exige input_size.z > output.z)" "$SPEC_T"
  fi
  bt_quit
done

# Le repère qui rend le verdict lisible : le dépôt embarque un modèle que ses
# auteurs ont nommé « _broken », et sa géométrie est celle du template.
BROKEN="$BT_REPO/Models/Greyscale_Diffusion_broken/config_file"
if [ -f "$BROKEN" ]; then
  read -r BZ BOUT <<<"$(geom "$BROKEN")"
  echo "       repère : Models/Greyscale_Diffusion_broken → input_size.z=$BZ, sortie=$BOUT"
  read -r GZ GOUT <<<"$(geom "$BT_REPO/Models/Greyscale_Diffusion/config_file")"
  echo "       repère : Models/Greyscale_Diffusion        → input_size.z=$GZ, sortie=$GOUT"
fi

bt_summary
