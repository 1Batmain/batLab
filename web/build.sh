#!/usr/bin/env bash
# File purpose: Reproducible build of the déposable web bundle in web/dist/.
#
# It compiles the engine to WebAssembly, runs wasm-bindgen to make it loadable,
# optionally shrinks it with wasm-opt, and copies the three assets the page
# fetches (the model config, the weights, and a handful of seed images). The
# result — web/dist/ — is a self-contained folder you can serve or drop into a
# site (see web/README.md).
#
# Nothing here is machine-specific except the default asset paths, which point at
# the main checkout; override them with the env vars below.
set -euo pipefail

WEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$WEB_DIR/.." && pwd)"
DIST="$WEB_DIR/dist"

# --- Assets (override via env) ----------------------------------------------
# Datasets and checkpoints are gitignored, so they live in the main working
# checkout rather than in this worktree; hence absolute defaults.
MODEL_DIR="${MODEL_DIR:-/Users/bat/development/lab/batLab/Models/Elephants_XL}"
CONFIG="${CONFIG:-$MODEL_DIR/config_file}"
# The model's own latest weights — not a path into some run directory. A build
# that pointed at `eleph_night/` broke the moment that campaign folder was tidied
# away, and it hid which model the page actually served. The checkpoint carries
# weights + Adam + EMA; the export step below strips it to what inference reads.
WEIGHTS="${WEIGHTS:-$MODEL_DIR/pretrained_weights/latest.ckpt}"
SEED_DATASET="${SEED_DATASET:-/Users/bat/development/lab/batLab/datasets/elephants256.batraw}"
SEED_COUNT="${SEED_COUNT:-32}"

# The export resolves the model's geometry by NAME through the storage layer, so
# it needs both the name and the root that holds Models/. Defaults derive from
# MODEL_DIR (…/<root>/Models/<name>); override either if your layout differs.
MODEL="${MODEL:-$(basename "$MODEL_DIR")}"
BATLAB_ROOT="${BATLAB_ROOT:-$(cd "$MODEL_DIR/../.." 2>/dev/null && pwd || echo "$REPO_DIR")}"

# What the export carries. RAW_WEIGHTS=1 ships the last iterate instead of the
# average. QUANTIZE=1 codes the weights to 8 bits — a quarter of the file again,
# but MEASURED NO-GO on this model (x₀-RMSE +14 %, visibly muddier; see the
# report in docs/reports/WEB_PORT.md and docs/gallery/web_slim_q8_nogo.png), so
# it is off by default: the page ships the f32 stripped file.
QUANTIZE="${QUANTIZE:-0}"
RAW_WEIGHTS="${RAW_WEIGHTS:-0}"

WASM_CRATE="batlab_web"
TARGET="wasm32-unknown-unknown"
WASM_IN="$REPO_DIR/target/$TARGET/release/${WASM_CRATE}.wasm"

say() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- Tool checks ------------------------------------------------------------
command -v cargo   >/dev/null || die "cargo not found"
command -v python3 >/dev/null || die "python3 not found"
if ! command -v wasm-bindgen >/dev/null; then
  die "wasm-bindgen not found. It is a system tool — add it declaratively:
     /etc/nix-darwin/flake.nix → environment.systemPackages: wasm-bindgen-cli
     then: sudo darwin-rebuild switch --flake /etc/nix-darwin#macbat
   Its version MUST match the wasm-bindgen crate ($(grep -A1 '^name = \"wasm-bindgen\"' "$REPO_DIR/Cargo.lock" | grep version | head -1 | cut -d'\"' -f2))."
fi

for f in "$CONFIG" "$WEIGHTS" "$SEED_DATASET"; do
  [ -f "$f" ] || die "missing asset: $f (set MODEL_DIR/CONFIG/WEIGHTS/SEED_DATASET)"
done

mkdir -p "$DIST"

# --- 1. Rust → wasm ---------------------------------------------------------
say "cargo build --release --target $TARGET -p $WASM_CRATE"
( cd "$REPO_DIR" && rustup target add "$TARGET" >/dev/null 2>&1 || true
  cargo build --release --target "$TARGET" -p "$WASM_CRATE" )
[ -f "$WASM_IN" ] || die "expected wasm not produced at $WASM_IN"

# --- version sanity: crate vs CLI ------------------------------------------
LOCKED="$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/[version = "]/,""); print; exit}' "$REPO_DIR/Cargo.lock" 2>/dev/null || true)"
CLI="$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')"
if [ -n "$LOCKED" ] && [ -n "$CLI" ] && [ "$LOCKED" != "$CLI" ]; then
  printf '\033[1;33m⚠ wasm-bindgen CLI %s ≠ crate %s — pin one to the other or the glue will not load.\033[0m\n' "$CLI" "$LOCKED"
fi

# --- 2. wasm-bindgen glue ---------------------------------------------------
say "wasm-bindgen --target web"
wasm-bindgen --target web --no-typescript --out-dir "$DIST" --out-name "$WASM_CRATE" "$WASM_IN"

# --- 3. wasm-opt (optional) -------------------------------------------------
BG="$DIST/${WASM_CRATE}_bg.wasm"
if command -v wasm-opt >/dev/null; then
  say "wasm-opt -Oz"
  wasm-opt -Oz -o "$BG.opt" "$BG" && mv "$BG.opt" "$BG"
else
  printf '\033[1;33m⚠ wasm-opt not found (nix: binaryen) — shipping the unoptimised wasm.\033[0m\n'
fi

# --- 4. Assets --------------------------------------------------------------
say "assets → dist"
cp "$CONFIG"  "$DIST/model.json"

# The weights the page downloads are the STRIPPED file, not the training
# checkpoint: the two Adam moments and the EMA trailer inflate the source four
# times over and inference reads none of them. `--export-weights` bakes in the
# set inference would select (the average by default) and re-serialises the
# weights alone — the same file, sampling bit-for-bit identically (proven by
# `an_export_samples_the_same_image_as_the_checkpoint_it_came_from`).
EXPORT_FLAGS=()
[ "$QUANTIZE" = "1" ]    && EXPORT_FLAGS+=(--quantize)
[ "$RAW_WEIGHTS" = "1" ] && EXPORT_FLAGS+=(--raw-weights)
say "export-weights $MODEL → dist/weights.ckpt ${EXPORT_FLAGS[*]:-(f32, EMA)}"
( cd "$REPO_DIR" && BATLAB_ROOT="$BATLAB_ROOT" cargo run --release -p batlab -- \
    --export-weights "$MODEL" --ckpt "$WEIGHTS" --out "$DIST/weights.ckpt" \
    ${EXPORT_FLAGS[@]+"${EXPORT_FLAGS[@]}"} )
[ -f "$DIST/weights.ckpt" ] || die "export produced no weights.ckpt"

python3 "$WEB_DIR/tools/make_seeds.py" "$SEED_DATASET" "$DIST/seeds.bin" --count "$SEED_COUNT"
cp "$WEB_DIR/index.html" "$DIST/index.html"

# --- 5. Report --------------------------------------------------------------
size() { du -h "$1" 2>/dev/null | cut -f1 | tr -d ' '; }
bytes() { wc -c <"$1" | tr -d ' '; }
total=0
for f in "$BG" "$DIST/weights.ckpt" "$DIST/seeds.bin" "$DIST/model.json" "$DIST/${WASM_CRATE}.js" "$DIST/index.html"; do
  [ -f "$f" ] && total=$(( total + $(bytes "$f") ))
done
src_bytes=$(bytes "$WEIGHTS")
ckpt_bytes=$(bytes "$DIST/weights.ckpt")
echo
say "web/dist/ built — download weight of the deposited page:"
printf '  %-22s %8s\n' "wasm (WebGPU engine)" "$(size "$BG")"
printf '  %-22s %8s  (stripped from %s, ×%s)\n' "weights.ckpt" "$(size "$DIST/weights.ckpt")" \
  "$(size "$WEIGHTS")" "$(awk -v s="$src_bytes" -v c="$ckpt_bytes" 'BEGIN{printf "%.1f", s/c}')"
printf '  %-22s %8s\n' "seeds.bin" "$(size "$DIST/seeds.bin")"
printf '  %-22s %8s\n' "model.json + js + html" "$(( $(bytes "$DIST/model.json") + $(bytes "$DIST/${WASM_CRATE}.js") + $(bytes "$DIST/index.html") )) B"
printf '  %-22s %8s\n' "TOTAL" "$(awk -v b="$total" 'BEGIN{printf "%.1f Mo", b/1048576}')"
echo
say "serve it (WebGPU needs a secure context — localhost counts):"
echo "    python3 -m http.server -d \"$DIST\" 8000   # then open http://localhost:8000"
