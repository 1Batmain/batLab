# Rapport — passe d'allègement des commentaires (branche `comments`)

## Décompte avant / après

**Global** (`crates/`, `tools/`, `bench/`, `blind_tests/`, fichiers `.rs` + `.wgsl`) :

| | Lignes de commentaire | % du code |
|---|---|---|
| Avant | 9 456 / 53 091 | 17,8 % |
| Après | 7 940 / 51 682 | 15,4 % |
| **Retiré** | **−1 516 (−16 %)** | |

Une passe qui *retire* des commentaires en écrivant des commentaires plus courts : le net est −1 516 lignes de commentaire, et le total de lignes ne baisse que de 1 409 (l'écart, ~107, ce sont des commentaires remplacés à volume égal mais mieux placés).

**Par fichier** (les 12 nommés par l'architecte comme « là où se joue la mission » — ils portaient la moitié du volume — plus les shaders) :

| Fichier | Avant | Après | Δ |
|---|---|---|---|
| `crates/batlab/src/main.rs` | 1009 | 610 | −399 |
| `crates/batlab-ui/src/tui/app.rs` | 666 | 439 | −227 |
| `crates/batlab-core/src/model/training/perpetual.rs` | 564 | 343 | −221 |
| `crates/batlab-ui/src/storage.rs` | 467 | 324 | −143 |
| `crates/batlab-core/src/model/model.rs` | 402 | 285 | −117 |
| `crates/batlab-core/src/model/training/schedule.rs` | 305 | 189 | −116 |
| `crates/batlab-core/src/resources/mod.rs` | 267 | 204 | −63 |
| `crates/batlab-ui/src/tui/ui.rs` | 237 | 197 | −40 |
| `crates/batlab-core/src/config.rs` | 243 | 199 | −44 |
| `crates/batlab-web/src/lib.rs` | 232 | 197 | −35 |
| `crates/batlab-ui/src/visualiser/mod.rs` | 222 | 154 | −68 |
| `crates/batlab-core/src/model/training/dataset.rs` | 220 | 181 | −39 |
| 7 shaders WGSL (`dataset_decode`, `loss`, `add`, `back_add`, `ema`, `sgd`, `adam`) | — | — | −56 |

Un commit atomique par fichier (ou par paire fwd/back de shaders), `On branch comments` / arbre propre après chacun.

## La règle appliquée (celle du brief, sans l'improviser)

> **Si un test garantit l'invariant, le commentaire devient un pointeur d'une ligne vers ce test. Si rien ne le garantit, le commentaire EST le garde-fou : il reste — ou mieux, il devient un test et disparaît.**

Concrètement, ce qui est parti :

- **Les récits de campagne** (chiffres avant/après en ms, noms de missions, dates) : ils vivent dans `docs/reports/`. Ex. le récit BATRAW3 (« disque, RAM, trafic ×4 ») en tête de `dataset_decode.wgsl` et de `main.rs`, le `−0,9 ms` du profil GPU, le sweep de `SAMPLING_SWEEP.md` recopié dans `batlab-web`.
- **Les paraphrases de signature** (`/// Charge le checkpoint` sur une fonction `load_checkpoint`) et les **blocs de bindings** des shaders qui recopiaient les déclarations juste en dessous.
- **Les redites** : la même explication en tête de module + en tête de fonction + sur la ligne. Deux redites franches trouvées et fusionnées (voir plus bas).

Ce qui a été **compressé mais gardé** : le « pourquoi » d'un choix contre-intuitif que rien ne teste, les pièges de plateforme, les invariants inter-fichiers — chacun pointé vers son test ou son rapport quand il en a un.

## Redites et périmés trouvés (le brief demandait de les signaler)

1. **`visualiser/mod.rs`** portait **deux headers module** : `//! File purpose: …` (ligne 1) suivi de `//! Live GPU visualiser…` (lignes 3-27). Fusionnés en un.
2. **`tui/app.rs`** portait **deux commentaires de merge quasi identiques** (~10 lignes) sur le probe de `device_profile`, l'un juste après l'autre. Fusionnés.
3. **Commentaire PÉRIMÉ corrigé** — `main.rs`, boucle `run_perpetual` : le commentaire affirmait « *A climb increment costs no model call* » alors que le fix IMG2IMG_DRIFT lui en fait coûter un désormais (le commentaire 25 lignes plus haut le dit explicitement). Réécrit factuellement, et la valeur `CLIMB_TEMPO_RATIO = 1.0` vérifiée dans le moteur avant d'affirmer quoi que ce soit.

## Commentaires convertis en tests

**Aucune conversion neuve.** La quasi-totalité des invariants non triviaux rencontrés **avaient déjà leur test** — c'est le cas médian de la règle (« devient un pointeur d'une ligne vers ce test »), pas le cas rare (« devient un test et disparaît »). Je les ai donc pointés plutôt que réécrits :

- `dataset_decode.wgsl` / `dataset.rs` : la table-vs-formule → `the_gpu_decode_agrees_with_the_cpu_one`.
- `schedule.rs` : `gaussian_at` (avalanche avant, addition et non XOR) → `injected_noise_over_reverse_chain_is_isotropic`.
- `storage.rs` : racine injectée → `project_root_is_the_workspace_that_holds_models_and_datasets`.
- `config.rs` / `app.rs` : conditionnement sur t → `every_model_created_from_a_template_is_conditionable_on_the_timestep`.
- `main.rs` / `perpetual.rs` : résolution unique de la graine → `every_path_that_starts_a_drift_resolves_its_seed_the_same_way` ; nom-du-frame → `DriftAction::phase`.
- `visualiser/mod.rs` : opt-out du vol de focus → le test source `the_event_loop_never_activates_the_app_over_the_terminal`.

Là où **aucun** test ne gardait l'invariant, je **n'ai pas supprimé** le commentaire (voir la liste ci-dessous) — c'est la moitié « il reste » de la règle. Je n'ai pas écrit de nouveau test dans cette passe : le mandat était d'alléger, pas d'ajouter une suite ; les créer aurait mérité leur propre jalon (avec la vérification par mutation qu'exige le brief).

## Ce que j'ai GARDÉ délibérément (la preuve que la passe a discriminé)

Chacun est un « pourquoi » ou un piège que **rien ne teste**, resserré à l'os mais conservé :

**Pièges de plateforme (faits externes, aucun test local ne les rend évidents)**
- `visualiser/mod.rs` — **le vol de focus macOS** : winit termine `applicationDidFinishLaunching` par `activateIgnoringOtherApps(true)` par défaut ; `ActivationPolicy::Accessory` ne le couvre pas ; `with_activate_ignoring_other_apps(false)` est la ligne à NE PAS retirer. (AGENTS.md l'exige aussi.)
- `visualiser/mod.rs` — l'event loop winit doit vivre sur le **main thread** (AppKit).
- `visualiser/mod.rs` — la couleur de letterbox est **linéaire** alors que la surface est **sRGB** : un `0,08` sort à `sRGB 80/255`.
- `batlab-web/lib.rs` — `predict` **bloque** sur son readback ε̂ et **deadlock** le thread qui doit tourner l'event loop du navigateur (d'où les variantes `*_async`).
- `batlab-web/lib.rs` — le scope de validation GPU : sans lui, un pipeline rejeté peint du ε̂ à zéro **en silence**.
- `dataset_decode.wgsl` / plusieurs shaders — Metal compile la division en réciproque approchée : la table de 256 valeurs n'est **pas** une formule à « simplifier ».

**« Pourquoi » d'un choix contre-intuitif, non testé**
- `back_add.wgsl` — une **seule** passe (pas deux) pour écrire les deux gradients : garde producteur et lecteur à un hazard, pas deux.
- `model.rs` — le trailer optimiseur : dropper m/v redémarre la correction de biais à t=1 (premier pas à ±lr).
- `schedule.rs` — `forward_from` et non `forward_step` pour la remontée perpetual : un champ par frame « seethes » (télé statique) ; un champ par cycle cohère.
- `dataset.rs` — le **hazard `write_buffer`** : encoder deux échantillons de deux chunks dans un seul encoder les fait lire tous deux le dernier chunk chargé (image silencieusement fausse) → grouper par chunk.
- `perpetual.rs` — l'oracle du gel est **délibérément sous-confiant (×0,9)** ; le ratio ᾱ formé depuis l'état de départ, pas re-lu comme image propre.
- `ema.wgsl` — la rampe de warmup vit sur le CPU ; ne pas la remettre dans le shader.

**Invariants inter-fichiers / budgets de dimensionnement**
- `resources/mod.rs` & `dataset.rs` — `select_chunk_bytes` est la **seule** règle, appelée par le dataset ET l'inventaire, sinon prédiction ≠ allocation ; budget résidence (gpu/4) vs streaming (gpu/8) ; la limite de **binding** (pas de buffer) est celle qui borne la résidence sur le profil web.
- `schedule.rs` — l'embedding temporel doit rester **identique** à `diffusion_prepare.wgsl` (gardé par test, pointé).
- `storage.rs` — `run-…_02.ckpt` avec underscore : `-` trie avant `.` et casserait « trier les noms = trier les runs » ; le refus de symlink dans le manager ; le lien dur (pas copie) de `latest.ckpt`.

## Vérification

- Compilation verte après chaque commit (`cargo check` sur le paquet touché).
- Édition commentaires **uniquement** : zéro changement de comportement (aucun code déplacé ; le `const HELP` de `main.rs`, sortie `--help` contractuelle, laissé intact ; les messages d'assertion, chaînes runtime, non touchés).
- Tests exécutés en fin de passe, tous verts : `batlab_core` (perpetual/schedule/config, 50), `batlab_ui` (storage/app/nav/ui, 110), `batlab` (main.rs, 11 ciblés). Doctests : 0 (le dépôt n'en a pas). `git status` propre.

## Restant (non fait — pour être honnête sur le périmètre)

Les 12 fichiers du cœur de mission sont faits. Restent, dans l'ordre du volume, des fichiers de 100-199 lignes de commentaire qui suivent les mêmes motifs (mêmes règles applicables) :
`batch_equivalence_tests.rs` (199), `gpu_context.rs` (178), `drift.rs` (171), `layer_types/convolution.rs` (168), `conv_equivalence_tests.rs` (158), `profile.rs` (156), `training/metrics.rs` (152), `model/layer.rs` (152), `live_frame.rs` (152), `attention_tests.rs` (152), `training/diffusion.rs` (144), plus les shaders denses restants (`back_convolution.wgsl`, `attention.wgsl`, `back_upsample_conv.wgsl`…). ~2 000 lignes de commentaire s'y trouvent encore ; la passe peut reprendre là avec la même règle.

---
*Passe menée sur la branche `comments`, rebasée sur `master` (fusion de la branche `audit`) avant de toucher aux fichiers partagés.*
