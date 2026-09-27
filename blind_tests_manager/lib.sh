# blind_tests_manager/lib.sh — harnais boîte noire du TUI batlab
#
# Pilote le VRAI binaire dans une session tmux détachée, sur une racine de
# stockage JETABLE (BATLAB_ROOT). Aucune connaissance de l'implémentation :
# on envoie des touches, on lit l'écran et le disque.
#
# Protocole d'envoi : cf. AGENTS.md — texte et Entrée en deux commandes,
# petite pause entre les deux (le bracketed paste du TUI gobe sinon le CR).

set -uo pipefail

BT_REPO="${BT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BT_BIN="${BT_BIN:-$BT_REPO/target/release/batlab}"
BT_WORK="${BT_WORK:-${TMPDIR:-/tmp}/batlab-blind-manager.$$}"
BT_COLS="${BT_COLS:-200}"
BT_ROWS="${BT_ROWS:-50}"
# délai standard après une touche, avant capture
BT_KEY_PAUSE="${BT_KEY_PAUSE:-0.6}"

# --- compteurs / rapport ------------------------------------------------

BT_PASS=0
BT_FAIL=0
BT_SKIP=0
BT_FAILLOG="${BT_FAILLOG:-$BT_WORK/failures.log}"

bt_init_report() { mkdir -p "$BT_WORK"; : > "$BT_FAILLOG"; }

ok()   { BT_PASS=$((BT_PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$1"; }
# fail <message> <citation de la spec>
fail() {
  BT_FAIL=$((BT_FAIL+1))
  printf '  \033[31mFAIL\033[0m %s\n' "$1"
  printf '       spec: %s\n' "${2:-<non citée>}"
  { echo "FAIL [${BT_PROP:-?}] $1"; echo "     spec: ${2:-<non citée>}"; } >> "$BT_FAILLOG"
}
skip() { BT_SKIP=$((BT_SKIP+1)); printf '  \033[33mskip\033[0m %s\n' "$1"; }

# Écart entre deux textes de spec, ou spec muette là où l'ordre de mission
# attend quelque chose : ce n'est pas un bug du code, on le consigne sans le
# compter en échec. Un aveugle signale, il ne tranche pas.
BT_GAP=0
spec_gap() {
  BT_GAP=$((BT_GAP+1))
  printf '  \033[36mSPEC\033[0m %s\n' "$1"
  printf '       %s\n' "${2:-}"
  { echo "SPEC-GAP [${BT_PROP:-?}] $1"; echo "     ${2:-}"; } >> "$BT_FAILLOG"
}

# assert_contains <écran> <motif> <message> <spec>
assert_contains() {
  if printf '%s' "$1" | grep -qF -- "$2"; then ok "$3"; else
    fail "$3 — motif absent de l'écran : « $2 »" "$4"
    bt_dump_screen "$1"
  fi
}
assert_not_contains() {
  if printf '%s' "$1" | grep -qF -- "$2"; then
    fail "$3 — motif présent alors qu'il ne devrait pas : « $2 »" "$4"
    bt_dump_screen "$1"
  else ok "$3"; fi
}
assert_eq() {
  if [ "$1" = "$2" ]; then ok "$3"; else
    fail "$3 — attendu « $2 », obtenu « $1 »" "$4"
  fi
}

bt_dump_screen() {
  printf '%s\n' "$1" | sed 's/^/       | /' | head -40
}

# --- cycle de vie d'une instance ---------------------------------------

# bt_new_root <nom> : crée une racine jetable, l'affiche
bt_new_root() {
  local root="$BT_WORK/$1"
  rm -rf "$root"
  mkdir -p "$root/Models" "$root/datasets"
  printf '%s' "$root"
}

# bt_fixture <racine> <nom du modèle> : copie un modèle du dépôt (shell only)
bt_fixture() {
  local root="$1" name="${2:-Greyscale_Diffusion}" as="${3:-$2}"
  cp -R "$BT_REPO/Models/$name" "$root/Models/$as"
  # pas de poids hérités : ils sont d'une ère antérieure du pipeline
  rm -rf "$root/Models/$as/pretrained_weights"
}

# bt_dataset <racine> : lie un .batraw si on en trouve un
bt_dataset() {
  local root="$1" ds
  ds="${BLIND_DATASET:-}"
  if [ -z "$ds" ]; then
    ds="$(ls "$BT_REPO/datasets"/*.batraw 2>/dev/null | head -1)"
  fi
  if [ -z "$ds" ]; then
    # le worktree n'a pas les .batraw (gitignorés) : chercher le dépôt principal
    ds="$(ls "$BT_REPO"/../../datasets/*.batraw 2>/dev/null | head -1)"
  fi
  [ -n "$ds" ] || return 1
  ln -sf "$ds" "$root/datasets/$(basename "$ds")"
  printf '%s' "$root/datasets/$(basename "$ds")"
}

# bt_launch <nom de session> <racine> : lance le TUI détaché
bt_launch() {
  BT_SESS="bt-$1"
  BT_ROOT="$2"
  BT_STDERR="$BT_WORK/$1.stderr"
  BT_EXIT="$BT_WORK/$1.exit"
  : > "$BT_STDERR"; rm -f "$BT_EXIT"
  tmux kill-session -t "$BT_SESS" 2>/dev/null
  tmux new-session -d -s "$BT_SESS" -x "$BT_COLS" -y "$BT_ROWS" -c "$BT_REPO" \
    "env BATLAB_ROOT='$BT_ROOT' '$BT_BIN' 2>'$BT_STDERR'; echo \$? > '$BT_EXIT'; sleep 900"
  bt_wait_for 'batlab — Models' 20 || return 1
}

bt_alive() { [ ! -f "$BT_EXIT" ]; }

bt_quit() {
  if bt_alive; then tmux send-keys -t "$BT_SESS" q 2>/dev/null; sleep 1.5; fi
  tmux kill-session -t "$BT_SESS" 2>/dev/null
  true
}

bt_exit_code() { cat "$BT_EXIT" 2>/dev/null || echo "<toujours vivant>"; }

# --- envoi de touches / lecture d'écran --------------------------------

# bt_key <touches tmux...> : Enter, Down, BSpace, Escape, q, r, e, i…
bt_key() {
  tmux send-keys -t "$BT_SESS" "$@"
  sleep "$BT_KEY_PAUSE"
}

# bt_type <texte littéral> — jamais interprété comme nom de touche
bt_type() {
  tmux send-keys -t "$BT_SESS" -l -- "$1"
  sleep "$BT_KEY_PAUSE"
}

bt_backspace() {
  local n="${1:-1}" i
  for ((i=0;i<n;i++)); do tmux send-keys -t "$BT_SESS" BSpace; sleep 0.08; done
  sleep 0.3
}

bt_screen()  { tmux capture-pane -t "$BT_SESS" -p 2>/dev/null; }
bt_screenj() { tmux capture-pane -t "$BT_SESS" -p -J 2>/dev/null; }

# le journal de métriques écrit sur stdout par-dessus le TUI ; un
# redimensionnement force ratatui à repeindre tout l'écran
bt_redraw() {
  tmux resize-window -t "$BT_SESS" -x $((BT_COLS-1)) -y "$BT_ROWS" 2>/dev/null
  sleep 0.4
  tmux resize-window -t "$BT_SESS" -x "$BT_COLS" -y "$BT_ROWS" 2>/dev/null
  sleep 0.6
}

# bt_wait_for <motif grep -F> [timeout s]
bt_wait_for() {
  local pat="$1" tmo="${2:-15}" t=0
  while [ "$t" -lt "$((tmo*4))" ]; do
    bt_screen | grep -qF -- "$pat" && return 0
    sleep 0.25; t=$((t+1))
  done
  return 1
}

# bt_select <libellé> [max] : descend le curseur « > » jusqu'à la ligne voulue.
# Les menus gardent leur curseur d'une visite à l'autre — on ne compte donc
# jamais les flèches, on regarde l'écran.
bt_select() {
  local want="$1" max="${2:-12}" i
  for ((i=0;i<max;i++)); do
    if bt_screen | grep -qE "^[^│]*│?[[:space:]]*>[[:space:]]+${want}[[:space:]]*(│|\$)"; then return 0; fi
    tmux send-keys -t "$BT_SESS" Down; sleep 0.25
  done
  # deuxième passe vers le haut, au cas où la cible serait au-dessus
  for ((i=0;i<max;i++)); do
    if bt_screen | grep -qE "^[^│]*│?[[:space:]]*>[[:space:]]+${want}[[:space:]]*(│|\$)"; then return 0; fi
    tmux send-keys -t "$BT_SESS" Up; sleep 0.25
  done
  return 1
}

# bt_action <libellé> : sélectionne puis valide une entrée du menu d'actions
bt_action() { bt_select "$1" && bt_key Enter; }

# --- empreinte disque ---------------------------------------------------

# bt_snapshot <racine> : chemin relatif + taille + sha, trié
bt_snapshot() {
  ( cd "$1" && find . -type f -o -type l | LC_ALL=C sort | while read -r f; do
      if [ -L "$f" ]; then printf '%s\tsymlink\t%s\n' "$f" "$(readlink "$f")"
      else printf '%s\t%s\t%s\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
      fi
    done )
}

bt_json_field() {
  python3 - "$1" "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
cur=d
for k in sys.argv[2].split('.'):
    cur = cur.get(k) if isinstance(cur,dict) else None
    if cur is None: break
print('' if cur is None else cur)
PY
}

bt_summary() {
  echo
  printf '  ── %s : %d ok, %d FAIL, %d skip, %d écart(s) de spec\n' \
    "${BT_PROP:-suite}" "$BT_PASS" "$BT_FAIL" "$BT_SKIP" "$BT_GAP"
  [ "$BT_FAIL" -eq 0 ]
}
