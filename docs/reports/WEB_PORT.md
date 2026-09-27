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
commande, la même que l'AGENTS.md :

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

Le checkpoint d'un entraînement transporte **quatre fois** ce que l'inférence
lit. Sur `eleph_night/night.ckpt` (le modèle **EMA** que la page utilise,
`BBCKPT3`, 19 082 087 o) :

| Contenu du checkpoint | Poids | Utile en inférence |
|---|---:|:--:|
| les poids | 4,77 Mo (1 192 563 params × 4 o) | **oui** |
| les deux moments d'Adam | 9,54 Mo | non (entraînement seul) |
| la remorque EMA | 4,77 Mo | **c'est ELLE qu'on garde**, à la place des poids bruts |

D'où l'export **nu** (`--export-weights`, item 1) : relire le checkpoint et n'en
réécrire que les poids que l'inférence génèrerait — la **moyenne EMA** par
défaut, exactement la règle de sélection du sampler (`CheckpointWeights::Ema`,
non ré-implémentée) — dans un `BBCKPT2` ordinaire, sans moments ni remorque. Le
fichier échantillonne **au bit près** la même image que le checkpoint d'origine,
sur la même graine : prouvé, pas inspecté
(`an_export_samples_the_same_image_as_the_checkpoint_it_came_from`, et reconfirmé
par `--eval`, cf. plus bas).

`web/build.sh` lance cet export à la place de l'ancien `cp` (knobs `WEIGHTS`,
`MODEL`, `RAW_WEIGHTS`, `QUANTIZE`). Le bilan mesuré de la page déposée :

| Asset | Taille | Note |
|---|---:|---|
| `weights.ckpt` | **4,55 Mio** (4 770 507 o) | poids EMA f32, nus — dépouillés de 18,20 Mio (×4,0) |
| `batlab_web_bg.wasm` | 395 Kio (404 197 o) | moteur WebGPU, après `wasm-opt -Oz` |
| `seeds.bin` | 96 Kio (98 320 o) | 32 images de départ 32×32×3 |
| `batlab_web.js` + `model.json` + `index.html` | 63,6 Kio (65 089 o) | glue + config + page |
| **Total** | **5,09 Mio** (5 338 113 o) non compressé | contre ~18,7 Mo déposés — **×3,7** |

Le fichier reste un checkpoint ordinaire : l'inférence, le sélecteur de poids du
TUI et `--resume` le relisent tous. Le moteur web le charge par la **même**
fonction que le natif — `load_checkpoint_bytes_with(&weights,
CheckpointWeights::Ema)` (`crates/batlab-web/src/lib.rs`) : un `BBCKPT2` n'ayant
pas de remorque, `Ema` retombe sur ses entrées brutes, qui **sont** le jeu EMA
qu'on y a cuit — d'où l'égalité au bit près.

### La quantification 8 bits : MESURÉE NO-GO à ce jalon

Descendre encore ces 4,77 Mo à ~1,2 Mo en stockant les poids en **u8** (échelle
+ zéro par tenseur, déquantifiés au chargement par une table de 256 valeurs —
jamais une formule, même piège que `dataset_decode.wgsl`, ici sans morsure car
la déquant est côté hôte) est implémenté (`--export-weights --quantize`, module
`model::quant`, format `BBCKPTQ`) et **le fichier tient la cible** : 1 193 134 o.
Mais c'est un compromis, donc ça se mesure — `--eval` sur `elephants_all` (256
held-out, seed 7) et une planche seeds 101..606 (`docs/gallery/web_slim_q8_nogo.png`) :

| | f32 nu (= source, au bit près) | u8 quantifié | Δ |
|---|---:|---:|---:|
| taille des poids | 4 770 507 o | 1 193 134 o | ÷4,0 |
| ε-MSE (tout le schedule) | 0,0567 | 0,0571 | +0,7 % |
| **x₀-RMSE (tout)** | **0,3568** | **0,4079** | **+14,3 %** |
| x₀-RMSE tranche haute t[192,256) | 0,5756 | 0,6904 | +20 % |

+14,3 % de x₀-RMSE (et +20 % sur la tranche haute), **plus** une dégradation
**visible** sur la planche (images plus boueuses, détails perdus, teintes
décalées). Les deux critères de NO-GO sont franchis. **La page embarque donc
l'export nu f32**, pas l'u8 ; `--quantize` reste un opt-in documenté et
reproductible (`QUANTIZE=1` dans `build.sh`), jamais câblé au build par défaut.
Le dépôt a déjà deux NO-GO chiffrés (EMA court, pondération de loss) ; en voici
un troisième. Ce qui rachèterait peut-être la quantification — une échelle par
canal plutôt que par tenseur, ou un modèle moins sensible sur le haut-t — n'est
pas de ce jalon.

Servi avec gzip/brotli, le wasm tombe encore ; les poids f32 se compriment peu.

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

1. ~~**Installer `wasm-bindgen-cli` + `binaryen`**~~ — **fait** : la chaîne est en
   place (`wasm-bindgen 0.2.121`, `wasm-opt`, cible `wasm32-unknown-unknown`) et
   `web/build.sh` tourne de bout en bout ici. `build.sh` avertit toujours si la
   version de la CLI diverge du crate verrouillé.
2. **Constater en navigateur** les deux modes, relever les img/s réelles — non
   pilotable depuis cet agent (§4). La page se sert déjà (fenêtre tmux, port
   8080/8000) : ouvrir, vérifier que les éléphants s'affichent, que « Nouveau
   bruit » en ouvre un autre.
3. ~~**Quantification int8**~~ — **fait et MESURÉ NO-GO** (§2) : implémentée
   (`--quantize`, `BBCKPTQ`), le fichier tient 1,2 Mo, mais x₀-RMSE +14 % et
   images visiblement dégradées. La page reste sur l'export nu f32.
4. **Latent résident sur GPU** (tâche #19) : réécrire le pas inverse en WGSL avec
   un test d'égalité bit-à-bit contre `denoise_step_with_magnitude`/`x0_estimate`/
   `gaussian_at`, dans l'idiome du dépôt. Optimisation, pas correctif.

---

## 7. Le bug de l'inférence — le navigateur mangeait l'erreur

**Symptôme.** La page livrée charge, les deux modes s'animent, le compteur défile
— mais l'image est du **bruit RGB saturé** (chaque canal indépendant, couleurs
pures), et l'errance **dégrade** l'image de départ en ce même bruit. Console du
navigateur : **aucune erreur**.

**La signature.** Un bruit qui *s'installe* dans les deux modes est la récurrence
inverse qui réinjecte du bruit à chaque pas sans jamais en retirer : c'est ε̂ ≈ 0
(ou constant). Si ε̂ = 0, alors x̂₀ = x_t/√ᾱ — le latent gaussien brut, saturé à
*tous* les t — et comme chaque pas inverse ajoute encore σ·bruit, la sortie
*évolue* vers du bruit tout en restant morte. Le modèle ne calcule rien.

**Ce qui a été écarté, mesuré et non supposé** (le natif est l'oracle) :

- **Le profil de limites WebGPU** (« 8 storage buffers / 65 535 workgroups »).
  `--headless-sample Elephants_XL --gpu-limits web` rend une image **saine**,
  au bit près identique à `--gpu-limits native` (min/max/mean/std égaux). Sous
  wgpu natif, le profil web valide tout (création de pipeline **et** dispatch) et
  débruite parfaitement. Écarté.
- **L'arithmétique et la boucle.** La boucle d'inférence du crate web est
  bit-à-bit la descente de `sample_diffusion` (même seed de base, même path_seed,
  même ordre t décroissant). Vérifié, puis **verrouillé** par un test (voir plus
  bas).
- **La lecture async.** `predict_async` lit les mêmes octets que `predict`
  (`predict_async_matches_predict_bit_for_bit`).
- **Le repli EMA.** Le checkpoint ne porte pas d'EMA ; `CheckpointWeights::Ema`
  retombe alors sur les poids bruts (`use_ema = … && ema.is_some()`), gardé par
  `asking_for_an_average_a_checkpoint_lacks_falls_back_on_the_weights`. La page
  charge sans erreur — or le parseur est **strict** (il refuse toute longueur qui
  ne colle pas et tout octet en trop), donc des poids chargés = des poids de la
  bonne taille. Écarté.

**Le mécanisme du silence.** Le défaut est donc propre au couple **wasm +
navigateur** — un pipeline que le compilateur WGSL du navigateur (Tint/WebKit,
plus strict que `naga → Metal`) rejette, ou une limite tripotée au dispatch,
**invalide silencieusement** le pipeline : ses dispatches deviennent des no-ops,
le tampon de sortie garde ses zéros, ε̂ = 0. Et l'erreur était **mangée** :

- le crate web **n'installait aucun sink `log`** — tout ce que wgpu signale par
  `log::error!` (shader refusé, limite, erreur de device) était écrit *nulle
  part* sur wasm ;
- le gestionnaire `on_uncaptured_error` du moteur écrit par `eprintln!`, **perdu**
  sur wasm (aucun flux std n'est câblé à la console) ;
- **aucun error scope** n'entourait la construction du modèle ni les frames.

Une page qui peut manger une erreur de validation finit par en manger une. C'est
la classe de bug que le dépôt documentait déjà (échec silencieux, sorties à zéro
sur Metal) — ici rendue invisible une seconde fois par l'absence de journal.

**La correction — rendre l'erreur impossible à avaler, et la mesure visible :**

1. **Un sink console** (`console_log`) installé dans `start()` : ce que le moteur
   essayait déjà de dire arrive enfin dans la console.
2. **Un error scope de validation** (`GpuContext::guarded`) autour de la
   construction du modèle **et** de chaque frame. Un shader que le navigateur
   refuse devient un **message sur la page** (`create()` rejeté en nommant
   l'erreur du shader) au lieu d'un canvas qui anime un modèle mort ; une erreur
   au dispatch est attrapée par frame et affichée.
3. **Statistiques par frame** — ε̂ / x̂₀ (std, mean, min/max, NaN/Inf) — loggées et
   **affichées sous le canvas**. `ε̂ std ≈ 0` tranche l'hypothèse d'un coup d'œil ;
   `ε̂ std ≈ 1` l'élimine et déplace le soupçon vers l'affichage.
4. **Intégrité des poids** : octets reçus + hachage FNV-1a + compte de scalaires
   attendu (**1 192 563**) loggés, pour confirmer que le téléchargement qui
   atteint le wasm est le fichier sur disque.
5. **La boucle d'inférence ne re-dérive plus** la récurrence : elle passe par
   `reverse_step_async` (moteur), sibling async de `reverse_step`, exactement
   comme la dérive passe par `advance_async`.

**Le test qui manquait.** La dérive avait `advance_async_matches_advance` ;
l'inférence — le chemin qui a cassé — n'avait **aucun** équivalent, et re-dérivait
sa boucle dans le crate web où aucun test ne tourne.
`the_async_reverse_step_matches_the_sync_one` (GPU, `model/model.rs`) pilote la
descente async **comme le crate web la pilote** (une `reverse_step_async` par
frame) et l'égale à `sample_diffusion` au bit près. Une divergence de
composition/seed/ordre échoue désormais ici, pas dans le navigateur d'un visiteur.

**Observé / non observé (honnêteté du port, tenue).** Prouvé sans navigateur : le
crate web **compile en wasm** avec l'instrumentation ; `--gpu-limits web` en natif
débruite au bit près (donc ni les limites, ni la boucle, ni le repli EMA) ; la
descente async égale la synchrone (nouveau test) ; `cargo test --workspace` vert,
`git status` propre. **Non observé** : la page dans un vrai navigateur — je n'ai
pas de pilote WebGPU ici.

→ **Ce que l'utilisateur doit relever, page rechargée (`web/dist/` reconstruit) :**
la console imprime le profil obtenu, l'intégrité des poids, puis `[batlab] …
ε̂ std=…`. Deux issues, toutes deux désormais parlantes :
- un **message d'erreur** (page ou console) nommant un pipeline/shader refusé →
  la cause est là, dans ce shader, et le correctif sera dans son WGSL ;
- **pas d'erreur** mais `ε̂ std ≈ 0.000` → le forward est mort sans erreur de
  validation (un zéro numérique, pas un rejet), et la sonde par frame confirme
  l'hypothèse tout en excluant un simple bug d'affichage. `ε̂ std ≈ 1` avec une
  image toujours fausse pointerait, lui, vers l'affichage.

Dans tous les cas la page ne peut **plus** animer un modèle mort en silence.

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
- **Allègement des poids (§2)** : `crates/batlab-core/src/model/quant.rs`
  (quantification affine u8 + table de déquant), `…/model/model.rs`
  (`checkpoint_bytes_quantized`, magic `BBCKPTQ`, branche de chargement),
  `…/model/mod.rs` (module), `…/model/ema_tests.rs` (fidélité u8 au bit près),
  `crates/batlab/src/main.rs` (`--export-weights [--raw-weights] [--quantize]`,
  `stripped_checkpoint`, `an_export_samples_the_same_image_…`), `web/build.sh`
  (export à la place du `cp`), `docs/gallery/web_slim_q8_nogo.png` (planche NO-GO).
- **Le bug de l'inférence (§7)** : `crates/batlab-core/src/gpu_context.rs`
  (`GpuContext::guarded`, l'error scope de validation),
  `…/model/training/metrics.rs` (`reverse_step_async`, sibling async de
  `reverse_step`), `…/model/training/mod.rs` + `…/lib.rs` (export),
  `…/model/model.rs` (`the_async_reverse_step_matches_the_sync_one`) ;
  `crates/batlab-web/Cargo.toml` (`log`, `console_log`), `…/src/lib.rs`
  (sink console, error scopes, sonde ε̂/x̂₀ par frame, intégrité des poids) ;
  `web/index.html` (ligne de diagnostic sous le canvas).
