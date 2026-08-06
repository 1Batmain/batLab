#!/usr/bin/env bash
# Non-régression forte (P7) : reconstruit le binaire du commit PRÉCÉDANT l'ajout du
# régime flux, rejoue errance et respiration avec la même graine, et compare les PNG
# produits octet à octet. Aucune lecture de source — on ne fait que builder et exécuter.
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
OUT=$HERE/blind_tests/out
MODEL=Greyscale_Diffusion_L
CKPT_SRC=$HERE/Models/$MODEL/pretrained_weights/night_run.ckpt

# commit parent de « feat(perpetual): le regime flux »
REV=${BLIND_BASELINE_REV:-12acad8}
REPO=$(git rev-parse --path-format=absolute --git-common-dir)
REPO=${REPO%/.git}
WT=$REPO/worktrees/blind-baseline-preflux

if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add "$WT" "$REV" --detach >/dev/null
fi
# La baseline reconstruit une arborescence ANTÉRIEURE à la restructuration, où le
# paquet binaire s'appelait encore `main` (il est sous crates/batlab depuis). On le
# déduit de l'arborescence plutôt que de le figer, pour que le script reste valable
# des deux côtés de la bascule et pour tout BLIND_BASELINE_REV.
pkg_of() { # pkg_of <racine> → nom du paquet/binaire de cette arborescence
  if [ -d "$1/crates/batlab" ]; then echo batlab; else echo main; fi
}

mkdir -p "$WT/Models/$MODEL/pretrained_weights"
cp -n "$CKPT_SRC" "$WT/Models/$MODEL/pretrained_weights/" 2>/dev/null || true
( cd "$WT" && cargo build --release -p "$(pkg_of "$WT")" >/dev/null )

# le binaire pré-flux ignore --actions et s'arrête après --frames cycles (8 par défaut) ;
# le binaire courant honore --actions : on lui en donne assez pour dépasser 8 cycles.
gen() { # gen <racine> <regime>
  local root=$1 regime=$2
  rm -rf "$root/perpetual_samples/$regime"
  ( cd "$root" && "./target/release/$(pkg_of "$root")" --headless-perpetual "$MODEL" \
      --regime "$regime" --actions 5000 --seed 7 --frames 8 \
      --checkpoint "Models/$MODEL/pretrained_weights/night_run.ckpt" >/dev/null 2>&1 )
}

python_json='{"rev": "'$REV'", "png": {'
first=1
for regime in errance respiration; do
  gen "$WT" "$regime"
  gen "$HERE" "$regime"
  [ $first -eq 1 ] || python_json+=', '
  first=0
  python_json+='"'$regime'": {'
  f1=1
  for i in 000 001 002 003 004 005 006 007; do
    a=$WT/perpetual_samples/$regime/$i.png
    b=$HERE/perpetual_samples/$regime/$i.png
    if [ -f "$a" ] && [ -f "$b" ] && cmp -s "$a" "$b"; then v=true; else v=false; fi
    [ $f1 -eq 1 ] || python_json+=', '
    f1=0
    python_json+='"'$i'": '$v
  done
  python_json+='}'
done
python_json+='}}'
echo "$python_json" > "$OUT/baseline.json"

# hygiène : les runs viennent de remplir perpetual_samples/ du worktree courant
# (répertoire de sortie, gitignoré — cf. blind_tests/run.sh)
rm -rf "$HERE/perpetual_samples"
