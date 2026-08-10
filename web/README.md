# batLab sur le web — diffusion sur le GPU du visiteur

Cette page fait tourner le moteur de diffusion de batLab **entièrement dans le
navigateur du visiteur, sur sa propre carte graphique**, via WebGPU. Aucun
serveur ne calcule : le serveur ne sert que des fichiers statiques. C'est l'usage
pour lequel la frontière moteur/interface du dépôt existe — le crate
`batlab_web` dépend de `batlab_core` et de la plateforme web, jamais l'inverse.

Deux modes, ceux du TUI :

- **Inférence** — génère une image depuis du bruit en montrant le débruitage se
  dérouler (256 pas).
- **Errance** — la dérive img2img perpétuelle : part d'une vraie image du dataset
  de graines et dérive sans fin (régimes *errance*, *respiration*, *flux* ;
  profondeur de re-bruitage `t_r` réglable).

Le modèle servi est **Elephants_XL** (1,19 M paramètres, entrée `[32,32,7]`,
sortie RGB `[32,32,3]`, attention 8×8).

---

## Ce qui se télécharge (le poids de la page)

| Asset | Taille | Rôle |
|---|---:|---|
| `weights.ckpt` | **13,6 Mo** (14 311 503 o) | les poids du modèle (f32) |
| `batlab_web_bg.wasm` | ~1,6 Mo brut, **~1,0–1,3 Mo** après `wasm-opt -Oz` | le moteur (WebGPU compilé) |
| `seeds.bin` | 96 Kio | 32 images de départ pour l'errance |
| `batlab_web.js` + `model.json` + `index.html` | ~40 Kio | glue + config + page |
| **Total** | **≈ 15–16 Mo** (non compressé) | |

**Les poids f32 dominent.** La quantification int8 (÷4 → ~3,5 Mo, total ≈ 5 Mo)
est la piste évidente pour alléger — voir « Ce qui reste » plus bas. Servez avec
gzip/brotli activé : le wasm se comprime bien (~450 Kio), les poids f32 peu.

---

## Prérequis (une fois)

Le build a besoin de trois outils. La cible Rust est gérée par `rustup` ; les
deux binaires sont des **paquets système** — sur cette machine gérée par
nix-darwin, ils s'ajoutent déclarativement, jamais par `brew`/`cargo install` :

```bash
# 1. la cible wasm (rustup, réversible)
rustup target add wasm32-unknown-unknown

# 2. wasm-bindgen-cli + binaryen (wasm-opt) — dans /etc/nix-darwin/flake.nix,
#    environment.systemPackages :  wasm-bindgen-cli   binaryen
#    puis :
sudo darwin-rebuild switch --flake /etc/nix-darwin#macbat
```

> **Version critique.** `wasm-bindgen-cli` doit correspondre **exactement** au
> crate `wasm-bindgen` verrouillé (`Cargo.lock` → actuellement **0.2.114**).
> Si `wasm-bindgen --version` diffère de ce que nixpkgs fournit, épinglez le
> crate à la version de la CLI dans `crates/batlab-web/Cargo.toml`
> (`wasm-bindgen = "=<version de la CLI>"`) puis rebuild. `build.sh` avertit en
> cas d'écart.

---

## Build

```bash
web/build.sh
```

Le script : compile `batlab_web` en `wasm32` (release), lance `wasm-bindgen`
(`--target web`), passe `wasm-opt -Oz` si disponible, puis copie les trois
assets (config, poids, seeds) dans `web/dist/`. Il affiche à la fin le poids
téléchargé, détaillé.

Variables d'environnement (défauts pointant sur le checkout principal, car
datasets et checkpoints sont gitignorés donc absents du worktree) :

```bash
MODEL_DIR=/chemin/vers/Models/Elephants_XL \
WEIGHTS=$MODEL_DIR/pretrained_weights/latest.ckpt \
SEED_DATASET=/chemin/vers/datasets/elephants256.batraw \
SEED_COUNT=32 \
web/build.sh
```

Le résultat, `web/dist/`, est **autonome et déposable** :

```
web/dist/
  index.html            la page
  batlab_web.js         glue wasm-bindgen (module ES)
  batlab_web_bg.wasm    le moteur
  model.json            le config_file du modèle
  weights.ckpt          les poids
  seeds.bin             les images de départ
```

---

## Servir

WebGPU exige un **contexte sécurisé** : `https://` **ou** `http://localhost`
(localhost est traité comme sécurisé). Un simple serveur de fichiers suffit —
rien ne s'exécute côté serveur.

```bash
python3 -m http.server -d web/dist 8000
# puis ouvrir http://localhost:8000
```

En production, servez `web/dist/` derrière HTTPS (n'importe quel hébergeur
statique : Netlify, Pages, nginx…). Activez la compression.

---

## Intégrer dans un site existant

1. Copiez le contenu de `web/dist/` dans votre site (par ex. sous `/batlab/`).
2. Le cœur tient en quelques lignes de module ES :

```html
<canvas id="view" width="32" height="32" style="image-rendering:pixelated;width:384px;height:384px"></canvas>
<script type="module">
  import init, { BatDiffusion } from '/batlab/batlab_web.js';
  await init();
  const bytes = u => fetch(u).then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
  const inst = await BatDiffusion.create(
    await bytes('/batlab/model.json'),
    await bytes('/batlab/weights.ckpt'),
    await bytes('/batlab/seeds.bin'),
  );
  const cv = document.getElementById('view'), ctx = cv.getContext('2d');
  const off = new OffscreenCanvas(inst.width(), inst.height());
  const octx = off.getContext('2d');
  inst.set_mode('errance');            // ou 'inference'
  (async function loop() {
    const px = await inst.step();       // Uint8Array, width*height*4 (RGBA)
    octx.putImageData(new ImageData(new Uint8ClampedArray(px), inst.width(), inst.height()), 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, inst.width(), inst.height(), 0, 0, cv.width, cv.height);
    requestAnimationFrame(loop);
  })();
</script>
```

**API JS** (`BatDiffusion`) :

| Méthode | Effet |
|---|---|
| `BatDiffusion.create(config, weights, seeds)` → `Promise` | construit le moteur (rejette si WebGPU absent) |
| `await inst.step()` → `Uint8Array` | avance d'une frame, rend x̂₀ en RGBA (`w*h*4`) |
| `inst.width()` / `inst.height()` | la géométrie de l'image |
| `inst.status()` → `String` | ligne d'état (mode, régime, `t_r`, compteur) |
| `inst.set_mode("inference"\|"errance")` | change de mode |
| `inst.set_regime("wander"\|"breathe"\|"flux")` | régime de dérive (errance) |
| `inst.nudge_depth(±1)` | bouge `t_r` d'un cran |
| `inst.reseed()` | nouveau bruit (inférence) / image suivante (errance) |

Les contrôles sont non bloquants : ils sont mis en file et appliqués au début de
la frame suivante, donc sûrs à appeler pendant qu'un `step()` est en cours.

---

## Navigateurs

WebGPU **uniquement** — pas de repli WebGL ni CPU. Un navigateur sans WebGPU
reçoit un message clair, pas une page blanche.

- **Chrome / Edge ≥ 113** (desktop) : OK.
- **Safari 18+** (macOS Sequoia, iOS 18) : WebGPU activé par défaut.
- **Firefox** : OK là où WebGPU est livré (Windows d'abord ; ailleurs via
  `dom.webgpu.enabled`).

Une machine sans GPU exploitable, ou un adaptateur qui refuse ses limites, fait
échouer `create()` — la page l'affiche.

---

## Architecture (et la frontière, respectée)

- **Le moteur ne connaît pas le web.** Tout le chemin d'inférence reste dans
  `batlab_core` ; `batlab_web` ne fait que le *piloter*. La dépendance ne pointe
  que `batlab_web → batlab_core`.
- **Profil de limites WebGPU.** Le contexte GPU est ouvert sous
  `GpuLimitsProfile::Web` — la base WebGPU (256 Mio/tampon, 128 Mio/binding),
  le même chemin de code que `--gpu-limits web` en natif, dont le dépôt garantit
  déjà qu'une inférence de ce modèle y passe au bit près.
- **Le latent fait l'aller-retour CPU↔GPU, mais en asynchrone.** La lecture
  synchrone du natif (`device.poll(wait)` + `block_on`) est impossible dans un
  navigateur : le thread qu'elle bloquerait est celui qui doit faire tourner la
  boucle d'événements pour que le GPU réponde. Le moteur a donc gagné
  `Model::predict_async` et `DriftWalk::advance_async`, qui *attendent* la
  lecture. Ils produisent, au bit près, les mêmes frames que les chemins
  synchrones — garanti côté natif par
  `advance_async_matches_advance_bit_for_bit`. **La dérive web ne diverge donc
  pas de la dérive native** : c'est le même `DriftWalk`, la même `LinearNoiseSchedule`,
  le même `reverse_step_from_epsilon`.

---

## Ce qui reste (mesuré, pas deviné)

- **Latent résident sur le GPU (tâche #19).** Le port actuel garde le latent
  côté CPU et relit ε̂ à chaque pas, en asynchrone. À la cadence d'une pièce
  contemplative (≈ 30 img/s), cet aller-retour n'est pas le goulot : une frame
  = un appel modèle (30 couches) qui domine largement les 12 Kio d'ε̂ relus.
  Le garder résident supprimerait cette relecture, **mais impose de réécrire
  l'arithmétique du schedule en WGSL** — exactement la divergence que le dépôt
  interdit sans test d'égalité bit-à-bit. C'est une optimisation à faire *avec*
  un tel garde-fou (comme `the_gpu_decode_agrees_with_the_cpu_one`), pas un
  prérequis. Laissée sur la table, chiffrée, non bâclée.
- **Quantification int8 des poids.** 13,6 Mo → ~3,5 Mo. À implémenter seulement
  une fois le reste validé en navigateur, et à mesurer avec `--eval` (l'écart de
  qualité) : un gain de taille qui casse les images est un NO-GO.
- **Vérification navigateur.** Voir le rapport de mission
  (`docs/reports/WEB_PORT.md`) pour ce qui a été observé et ce qui ne l'a pas
  été.
