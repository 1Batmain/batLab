# Port web — le modèle sur le GPU du visiteur

**Mission** : faire tourner le moteur de diffusion de batLab dans le navigateur
du visiteur, sur *sa* carte graphique (WebGPU, aucun GPU serveur), pour le
portfolio. Deux modes : **inférence** (génération depuis du bruit, débruitage
visible) et **errance** (dérive img2img perpétuelle depuis une vraie image).
Modèle : **Elephants_XL** (1,19 M params, entrée `[32,32,7]` → RGB `[32,32,3]`,
attention 8×8).

Branche `web`. Livrable : le crate `batlab_web`, le dossier `web/` (page,
build, tools, README) et cet écrit.

---

## 1. L'architecture retenue

### La frontière, respectée à la lettre

La règle du dépôt — `batlab_core` ne dépend NI de ratatui, NI de crossterm, NI
de winit, prend des **octets**, ne connaît pas le système de fichiers — est
exactement ce qui a rendu ce port possible. Le nouveau crate suit la même flèche
de dépendance que le reste :

```
batlab_web → batlab_core        (comme batlab → batlab_ui → batlab_core)
```

`batlab_web` ne fait que **piloter** le moteur : aucune arithmétique de diffusion
ne vit ici. Chaque pas inverse est `reverse_step_from_epsilon`, chaque frame de
dérive est `DriftWalk::advance_async`, l'entrée est composée par
`compose_diffusion_input`, le schedule est celui du moteur. Vérifié en une
commande, la même que le CLAUDE.md :

```
$ awk '/^\[dependencies\]/{f=1;next}/^\[/{f=0}f' crates/batlab-core/Cargo.toml \
    | grep -E '^(ratatui|crossterm|winit)' && echo ÉCHEC || echo OK
OK
```

Les dépendances web (`wasm-bindgen`, `web-sys`, …) vivent sous
`[target.'cfg(target_arch = "wasm32")'.dependencies]` du crate web, et le corps
de sa lib est `#![cfg(target_arch = "wasm32")]` : **sur une cible native le crate
est vide**, si bien que `cargo build`/`cargo test --workspace` restent verts sans
jamais tirer une dépendance web.

### Le vrai blocage : la lecture GPU synchrone, pas le trafic

La tâche #19 nomme « 256 allers-retours CPU↔GPU par image » comme le point
bloquant. En le regardant de près, le blocage n'est pas le *coût* de l'aller-
retour — c'est qu'un aller-retour synchrone est **impossible** dans un
navigateur. `Model::predict` lit ε̂ via `device.poll(wait)` + `pollster::block_on` ;
sur le web, le thread ainsi bloqué est **celui qui doit faire tourner la boucle
d'événements** pour que le GPU réponde. C'est un interblocage, pas une lenteur.

Le moteur a donc gagné un chemin **asynchrone**, sans aucune dépendance web
(seulement `futures`, déjà présent, et un `cfg(target_arch)`) :

- `read_back_f32_at_async` (debug.rs) — mêmes trois étapes que la version
  bloquante (copie → `map_async` → lecture), mais elle *attend* le oneshot. En
  natif elle poll l'appareil (le futur se résout aussitôt) ; sur wasm elle ne
  poll pas — le navigateur tourne la boucle et résout la promesse. **Les mêmes
  octets ; seul change qui tourne la manivelle.**
- `Model::predict_async` (model.rs) — jumeau de `predict`, même submit, même
  tenseur, lecture attendue.
- `DriftWalk::advance_async` + trait `AsyncNoisePredictor` (drift.rs) — jumeau de
  `advance`, chaque bras identique à une `.await` près. Le natif reste sans
  exécuteur.

`batlab_web` pilote ces deux fonctions depuis `requestAnimationFrame`, une frame
à la fois, et peint x̂₀ (le pane par défaut de l'errance, `LiveView::X0Only`) sur
un `<canvas>` via `ImageData`.

### Pourquoi le latent fait encore l'aller-retour (et pas résident sur GPU)

Le garder résident supprimerait la relecture d'ε̂ à chaque pas — mais impose de
**réécrire l'arithmétique du schedule en WGSL** (`denoise_step`, `x0_estimate`,
`forward_from`, et surtout `gaussian_at` avec son `fmix64` au bit près). C'est
précisément la divergence que le dépôt interdit sans un test d'égalité bit-à-bit
(cf. `the_gpu_decode_agrees_with_the_cpu_one`, où une simple division divergeait
d'1 ULP sur 111/256 valeurs sous Metal). **Mesuré contre la cible ici** — une
pièce contemplative affichée à ~30 img/s — l'aller-retour n'est pas le mur : une
frame = un appel modèle (30 couches) qui domine de loin les 12 Kio d'ε̂ relus.
Le chemin async est le premier port correct et vérifiable ; la résidence GPU est
une optimisation chiffrée laissée sur la table, **avec** son garde-fou à écrire,
pas un prérequis bâclé. Voir §5.

### Le profil de limites WebGPU, obligatoire, obtenu par le même code

Le contexte GPU s'ouvre sous `GpuLimitsProfile::Web` — la base WebGPU (256 Mio/
tampon, 128 Mio/binding), **le même chemin de code** que `--gpu-limits web` en
natif. Le dépôt garantit déjà qu'une inférence de ce modèle passe sous ce profil
avec une grosse marge et rend une sortie **bit-à-bit identique** aux deux
profils (`both_profiles_open_a_device_and_native_never_grants_less`, et la
propriété d'inférence de `--headless-sample`). Le port n'y touche pas.

---

## 2. Le poids téléchargé (chiffré)

| Asset | Taille | Note |
|---|---:|---|
| `weights.ckpt` | **13,65 Mio** (14 311 503 o) | poids f32 — **domine** |
| `batlab_web_bg.wasm` | 1,6 Mo brut (release, avant `wasm-bindgen`/`wasm-opt`) ; **~1,0–1,3 Mo** attendu après `wasm-opt -Oz` | moteur WebGPU compilé (compute seul, pas de Vulkan/GL/DX) |
| `seeds.bin` | 96 Kio | 32 images de départ 32×32×3 |
| `batlab_web.js` + `model.json` + `index.html` | ~24 Kio | glue + config + page |
| **Total** | **≈ 15,5 Mo** non compressé | |

**Le mot honnête** : 13,65 Mo de poids f32 sur une page portfolio, c'est
beaucoup, et c'est ~88 % du poids. La quantification int8 (÷4 → ~3,5 Mo, total
≈ 5 Mo) est la piste évidente ; non implémentée ici par discipline (« ne
l'implémente que si le reste marche », et le reste n'a pas encore été observé en
navigateur — §4). Servi avec gzip/brotli, le wasm tombe à ~450 Kio ; les poids
f32 se compriment peu.

---

## 3. Images/s

**Non mesuré en navigateur** (voir §4 — je n'ai pas pu piloter de navigateur
WebGPU). Ce que l'on sait :

- La boucle de rendu est **bridée à ~30 img/s** (tempo contemplatif, le défaut
  perpetual du natif).
- En natif non bridé, le dépôt mesure la dérive à **324 frames/s** après le fix
  img2img (`IMG2IMG_DRIFT.md`), pour un tempo par défaut de 30 pas/s. Le coût
  dominant d'une frame est l'appel modèle, identique ici.
- À 30 img/s, l'aller-retour async (un `map_async` + un tour de boucle
  d'événements par frame) est très en-deçà du budget de 33 ms : 30 relectures de
  12 Kio par seconde ne sont pas le facteur limitant.

Un chiffre navigateur réel reste à relever par l'utilisateur (compteur d'fps sur
la page, ou `performance.now()` autour de `step()`).

---

## 4. Ce qui a été observé, et ce qui ne l'a pas été

### Observé (vérifié à la main)

- **Le moteur compile en `wasm32-unknown-unknown`** avec wgpu 28, qui câble seul
  son backend WebGPU (`web-sys`, `js-sys`, `wasm-bindgen-futures` tirés
  automatiquement, aucun `--cfg=web_sys_unstable_apis`). `batlab_web` compile en
  wasm (debug **et** release ; release = 1,6 Mo).
- **Le natif ne régresse pas.** `cargo test --workspace` : **377 tests verts**
  (batlab 27, core 240, ui 110), `git status` propre (aucune écriture parasite
  dans `Models/`, `datasets/`, ni `.jsonl`). Le build natif du workspace reste
  vert avec `batlab_web` vide.
- **Le chemin async est prouvé identique au synchrone, au bit près, sur vrai
  GPU** :
  - `predict_async_matches_predict_bit_for_bit` (GPU natif) — la lecture async
    rend les octets exacts de la lecture bloquante.
  - `advance_async_matches_advance_bit_for_bit` (les trois régimes, oracle sans
    GPU) — `advance_async` produit la frame exacte d'`advance`, latent et x̂₀.
  - C'est la garantie que **la dérive web ne diverge pas de la dérive native**.
- **L'extraction des graines** (`web/tools/make_seeds.py`) : 32 images 32×32×3
  depuis `elephants256.batraw` → `seeds.bin` (96 Kio), payload u8 recopié tel
  quel, re-décodé côté moteur par `decode_u8`.
- **Le bundle s'assemble** : `web/dist/` contient `model.json`, `weights.ckpt`,
  `seeds.bin`, `index.html` ; il ne manque que la glue `wasm-bindgen` (§ outil
  manquant).

### NON observé (dit honnêtement)

- **La page tournant dans un vrai navigateur.** Deux raisons :
  1. `wasm-bindgen-cli` (et `binaryen`/`wasm-opt`) **ne sont pas installés** sur
     la machine, et ce sont des paquets système → nix-darwin déclaratif, pas
     d'install impérative. Sans `wasm-bindgen`, pas de glue chargeable, donc pas
     de page fonctionnelle à servir. (La cible `wasm32` a, elle, été ajoutée par
     `rustup target add` — outil Rust, réversible.)
  2. Même la glue produite, je n'ai pas de moyen de **piloter un navigateur
     WebGPU** headless dans cet environnement (WebGPU exige un GPU et des drapeaux
     spécifiques). Je n'ai donc pas vu les deux modes s'exécuter.

  → **L'utilisateur devra faire tourner `web/build.sh` puis servir `web/dist/`
  et constater les deux modes.** Ce qui est prouvé sans navigateur : que le code
  compile en wasm, que le chemin async est correct au bit près sur GPU, et que
  le natif ne bouge pas.

---

## 5. Navigateurs

WebGPU **uniquement** (pas de WebGL, pas de repli CPU). Un navigateur sans
WebGPU reçoit un **message clair** (garde `if (!('gpu' in navigator))` avant tout
chargement lourd), pas une page blanche ; un `create()` qui échoue (GPU absent,
adaptateur refusant ses limites) est attrapé et affiché.

Passent aujourd'hui, en principe (non re-testés en navigateur ici) :
**Chrome/Edge ≥ 113** (desktop), **Safari 18+** (macOS Sequoia / iOS 18),
**Firefox** là où WebGPU est livré.

---

## 6. Ce qui reste

1. **Installer `wasm-bindgen-cli` + `binaryen`** dans `/etc/nix-darwin/flake.nix`
   (`environment.systemPackages`), rebuild, puis `web/build.sh`. La version de
   `wasm-bindgen-cli` **doit** correspondre au crate verrouillé (`0.2.114`) —
   sinon épingler le crate à la version de la CLI ; `build.sh` avertit.
2. **Constater en navigateur** les deux modes, relever les img/s réelles.
3. **Quantification int8** des poids (13,65 → ~3,5 Mo), *après* validation, en
   mesurant la dégradation avec `--eval` — un gain de taille qui casse les
   images est un NO-GO.
4. **Latent résident sur GPU** (tâche #19) : réécrire le pas inverse en WGSL avec
   un test d'égalité bit-à-bit contre `denoise_step_with_magnitude`/`x0_estimate`/
   `gaussian_at`, dans l'idiome du dépôt. Optimisation, pas correctif.

---

## Fichiers touchés

- **Moteur (sans dépendance web)** : `crates/batlab-core/src/model/debug.rs`
  (`read_back_f32_at_async`), `…/model/model.rs` (`predict_async`,
  `read_last_output_async`, test GPU), `…/model/training/drift.rs`
  (`AsyncNoisePredictor`, `advance_async`, test), `…/model/training/mod.rs` et
  `…/lib.rs` (exports + les trois constantes de schedule, désormais partagées
  avec le binaire), `crates/batlab/src/main.rs` (alias vers ces constantes).
- **Crate web** : `crates/batlab-web/` (`Cargo.toml`, `src/lib.rs`), ajouté au
  workspace.
- **Livrable** : `web/index.html`, `web/build.sh`, `web/tools/make_seeds.py`,
  `web/README.md` ; `web/dist/` gitignoré (wasm + poids + seeds).
