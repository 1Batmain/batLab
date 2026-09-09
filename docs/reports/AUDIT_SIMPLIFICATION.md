# Audit de simplification — tour d'ensemble après la campagne des deux jours

**Branche** : `audit`. **Statut** : diagnostic seul (aucune modification de code dans ce
commit). **Mission** : « chasse les doublons inutiles, évite les booléens à tout va,
simplifie le code au maximum, centralise toutes les variables vraiment utiles ».

Ce dépôt a reçu une dizaine de missions en deux jours (instrument, couches, port web,
export de poids, quantification, variance postérieure, balayage d'échantillonnage,
modèle 64×64). Chacune a ajouté ses flags, ses constantes, ses chemins. Personne n'avait
fait le tour de l'ensemble. Voici ce tour.

Chaque trouvaille est classée **DANGER** (deux vérités qui peuvent diverger — le risque
réel), **DETTE** (ça marche, c'est laid) ou **COSMÉTIQUE**. La dernière section dit
franchement ce que je propose de **NE PAS** faire. Rien n'a encore été touché : c'est au
propriétaire d'arbitrer avant la phase de correction.

## Ce que l'audit confirme comme SAIN (à ne pas « corriger »)

Pour cadrer : la campagne a plutôt bien tenu ses invariants. Sont déjà propres —
- Les constantes de schedule (`DIFFUSION_SCHEDULE_STEPS`, `DIFFUSION_BETA_START/END`)
  vivent une seule fois dans `core/lib.rs`, et le crate web **les lit** au lieu de les
  recopier (`web/lib.rs:49-51`). C'est le bon patron.
- La dérive perpétuelle résout sa graine par la porte unique
  `SeedImages::resolve`/`resolve_seed_dataset` sur les **deux** chemins (le trou
  historique `seed_dataset` est bouché — vérifié : main.rs:2127 et 3732).
- La dérivation de graine **par pas** est canonisée dans `reverse_step_seed()`
  (metrics.rs:348) — un seul endroit, un seul calcul.
- `PosteriorVariance` et `CheckpointWeights` sont déjà des **enums nommés**, pas des
  booléens. Le vœu « évite les booléens » est **déjà largement respecté** : il n'existe
  aucune fonction du dépôt à ≥2 booléens positionnels (balayage des trois crates).
- Aucun `#[allow(dead_code)]`, un seul warning de compilation.

Le travail restant est donc ciblé, pas une refonte.

---

# DANGERS — deux vérités qui peuvent diverger

## D1. Le fold de bruit de base n'est pas canonisé ; le crate web en tient une copie privée

**C'est le doublon de logique le plus coûteux du lot, et exactement la famille de bug
que la mission cite** (le web qui re-dérive sa boucle et masque une divergence).

La constante `0xa5a5_5a5a_0123_4567` — le fold qui transforme la graine d'un run en
graine du **latent d'ouverture** — est écrite **quatre fois**, en clair, sans source
commune :

| Emplacement | Forme | Rôle |
|---|---|---|
| `metrics.rs:533` | littéral `seed ^ 0xa5a5_5a5a_0123_4567` | le sampler de production (`sample_diffusion`) |
| `perpetual.rs:561` | `self.base_seed ^ 0xa5a5_5a5a_0123_4567` (`initial_noise_seed`) | l'ouverture de la dérive |
| `model.rs:2129` | `const BASE_NOISE_FOLD` **locale au test** | le test-garde de l'inférence async |
| `web/lib.rs:75` | `const BASE_NOISE_FOLD` **privée au crate** | l'ouverture de l'inférence web |

De même la **graine de chemin** `seed ^ (path_idx · GAMMA)` est un littéral nu dans le
sampler (`metrics.rs:537`), et le web la reproduit implicitement (path 0 = la graine
elle-même, `web/lib.rs:247-253`).

Le crate web relie ses copies au moteur **par un simple commentaire** :
`web/lib.rs:70-75` — *« Same fold as `batlab_core`'s reverse-chain seed »*. C'est de la
prose, pas une garantie. Et le test qui **devrait** attraper une divergence,
`the_async_inference_descent…` (model.rs:2110-2151), **redéfinit sa propre const locale**
`0xa5a5…` (ligne 2129) : il ne référence jamais la constante du crate web. Autrement dit,
si quelqu'un modifiait `web/lib.rs:75` ou `:72`, **aucun test n'échouerait** et la page
ouvrirait sur un autre champ de bruit que le sampler natif — silencieusement. C'est
littéralement la situation « le crate web a re-dérivé sa boucle et a masqué un bug pendant
tout un port ».

**Correction proposée** (une seule fonction, une seule fois) : exposer dans `batlab_core`,
à côté de `reverse_step_seed`, deux fonctions publiques —
`base_noise_seed(seed) -> u64` et `path_seed(seed, path_idx) -> u64` — et les appeler
depuis le sampler (`metrics.rs`), la dérive (`perpetual.rs`) et le web (`web/lib.rs`). Les
valeurs étant identiques, le run reste **bit-à-bit inchangé** (prouvable :
`--headless-sample` et le test async gardent la même sortie). Ajouter un test-garde qui
**balaye le code** pour tout littéral `0xa5a5_5a5a…` hors de la fonction canonique —
même patron que `every_path_that_starts_a_drift_resolves_its_seed_the_same_way` et
`nothing_in_the_engine_opens_an_untimed_pass`, deux gardes que le dépôt possède déjà.

## D2. `TrainingConfig.loss` — franc mensonge de configuration

`pub loss: LossMethod` (config.rs:1002) est un champ **obligatoire** (pas de
`#[serde(default)]`) : chaque `config_file` doit le porter. Il est écrit partout —
en dur `LossMethod::MeanSquared` par le TUI (app.rs:3148, 3462, 3600, 3628, 4273 ;
events.rs:792 ; storage.rs:1744), et round-trippé par headless-train (main.rs:943-944).

**Il n'atteint jamais le moteur.** Le seul point de création du modèle d'entraînement,
`build_execution_model` (main.rs:2784), ne prend même pas de paramètre `loss` : il code
la perte en dur —

```rust
// main.rs:2797
PLoss::MeanSquared,   // pas config.run.mode…loss
```

Impact **nul aujourd'hui** (l'enum n'a qu'un variant, config.rs:105-108). Mais c'est
structurellement **le schéma exact de `seed_dataset`/`optimizer` avant leur correction** :
un champ sérialisé, obligatoire, réécrit dans le `config_file`, qui n'atteint jamais le
moteur. Le jour où un second variant de perte est ajouté, il sera **silencieusement
ignoré** — et le `config_file` mentira sur ce que le modèle a appris.

**Correction proposée** : câbler `train_cfg.loss` jusqu'à `build_execution_model`
(comme `optimizer`, `weight_init`, `ema`). Comportement inchangé tant que l'enum reste
à un variant, mais le mensonge est levé et le futur second variant ne pourra plus se
perdre. (Alternative plus radicale — supprimer le champ jusqu'à ce qu'un second variant
existe — non recommandée : le câbler est plus proche de l'intention et suit ce que le
dépôt a fait pour les autres réglages.)

## D3. `denoise_magnitude` et `posterior_variance` : le web les code en dur au lieu de lire la config qu'il vient de charger

Le crate web fixe **en Rust** ses choix d'échantillonnage :

```rust
// web/lib.rs:61
const DENOISE_MAGNITUDE: f32 = 1.2;
// web/lib.rs:68
const POSTERIOR_VARIANCE: PosteriorVariance = PosteriorVariance::Beta;
```

Or il **charge déjà** le `config_file` du modèle (`build_engine`, web/lib.rs:647) qui
porte exactement ces deux réglages dans `InferenceConfig` — et le modèle déployé,
`Models/Elephants_XL/config_file`, dit **déjà** `denoise_magnitude: 1.2` et
`posterior_variance: "beta"`. Le web recopie donc en dur ce qu'il a sous la main.

C'est la même faute que D1, dans une autre monnaie : un choix qui a **un foyer**
(`config.inference`) est dupliqué dans le code du web. Le jour où l'auteur ajuste la
magnitude d'Elephants_XL dans son `config_file`, **la page l'ignore**.

**Correction proposée (sûre)** : faire lire au web `config.inference.denoise_magnitude`
et `config.inference.posterior_variance`, comme il lit déjà STEPS/BETA. Puisque le
`config_file` déployé porte déjà 1.2/beta, **le rendu de la page est bit-à-bit
identique** — c'est de la pure suppression de duplication. À vérifier avant : que le
`config_file` embarqué par `web/build.sh` est bien celui d'Elephants_XL (il l'est,
build.sh:21-22).

**Sous-cas à ARBITRER, pas à corriger d'office** : `--headless-sample` résout la
magnitude une **troisième** façon — `flag("--magnitude").unwrap_or(1.0)` (main.rs:1071),
qui **ignore** `config.inference.denoise_magnitude` (que l'inférence interactive, elle,
honore à main.rs:3634). La variance, dans la même fonction, retombe pourtant bien sur la
config (main.rs:1086-1089). C'est une asymétrie réelle, mais `--headless-sample` est un
outil DEV/CI et des bancs sous `bench/` peuvent compter sur le défaut 1.0 : **changer ce
défaut changerait une sortie observable**. Je le signale, je ne le corrige pas sans ton
feu vert.

## D4. Le lecteur d'en-tête `.batraw` existe en trois exemplaires Rust

L'en-tête `.batraw` (24 octets : magic `BATRAW{1,2,3}\0` + 4×u32) est parsé à trois
endroits :

| Site | Signature | Testé ? |
|---|---|---|
| `main.rs:1623` `read_batraw_header` | `-> Result<…, String>` | oui (test l.5801) |
| `storage.rs:1141` `read_batraw_header` | `-> Option<…>` | oui (test l.1247) |
| `main.rs:4239` inline dans `try_load_raw_dataset` | — | via les tests du chargeur |

Le point piquant : **`main.rs` importe et utilise déjà `storage::read_batraw_header`**
(main.rs:601) tout en gardant sa **propre** copie (main.rs:1623, pour `--resources`). Deux
fonctions homonymes, même en-tête, sémantique quasi identique (l'une renvoie `Result`
avec messages d'erreur, l'autre `Option`). Une évolution du format (un `BATRAW4`) devrait
être répétée à trois endroits — le genre d'oubli qui coûte une nuit.

**Correction proposée** : un seul lecteur. Le format est de l'octet pur (le décodage
`decode_u8` vit déjà dans `batlab_core`), donc un `BatrawHeader::parse(&[u8])` dans
`batlab_core` — qui **ne casse pas la frontière moteur/interface** (aucun `fs`, aucun
UI) — appelé par les trois sites. À défaut, faire pointer `resources_dataset` (main.rs)
sur `storage::read_batraw_header` et supprimer la copie de main.rs:1623. À arbitrer : le
foyer (core vs ui). **Note** : `SeedImages::parse` du web (web/lib.rs:114) n'est **pas**
un `.batraw` (en-tête 16 o, sans magic) — le laisser tel quel ; c'est en revanche le
**seul** des lecteurs sans test.

---

# DETTE — ça marche, c'est laid

## T1. `metrics.rs` contient tout le chemin d'inférence — le nom ne dit plus le contenu

`crates/batlab-core/src/model/training/metrics.rs` (678 l.) porte, sous un nom
« metrics », **tout le sampler public** : `compose_diffusion_input`, `predict_epsilon`,
`reverse_step`, `reverse_step_from_epsilon`, `reverse_step_async`, `sample_diffusion`,
`reverse_step_seed`, `log_trajectory`. Le CLAUDE.md affirme « tout le chemin d'inférence
vit dans `batlab_core` » — vrai, mais qui le chercherait dans `metrics.rs` ? Les vraies
métriques (`MetricsLogger`, `ProbeConfig`, `probe_diffusion`, `log_*`) y cohabitent.

**Correction proposée** : scinder en `inference.rs` (ou `sampler.rs`) pour le chemin
reverse/sample, et laisser `metrics.rs` aux métriques. Déplacement + réexports depuis
`lib.rs`/`mod.rs` : aucun changement de comportement. Cosmétique mais à haute valeur de
lisibilité.

### T1 — fait

Scindé en `model/training/sampler.rs`, en **deux commits**.

**Ce qui a déménagé dans `sampler.rs`** (le chemin d'inférence) : `compose_diffusion_input`,
`DenoiseFrame`, la dérivation des graines (`STEP_SEED_GAMMA` privé, `reverse_step_seed`,
`BASE_NOISE_FOLD`/`base_noise_seed`, `path_seed`), `ReverseStep`, `predict_epsilon`,
`reverse_step_from_epsilon`, `reverse_step_async`, `reverse_step`, `sample_diffusion`, et les
cinq tests qui les tiennent (dont `the_base_noise_fold_is_written_in_exactly_one_place` et
`the_canonical_seed_folds_match_the_old_literals`).

**Ce qui reste dans `metrics.rs`** (ce qui mesure) : `Stats`, `MetricsLogger`, `ProbeConfig`,
`BucketStat`, `probe_diffusion`, `log_probe`, `log_train_loss`, `log_trajectory`,
`DenoiseStepStat`, et les tests `stats_*`.

**Les deux cas limites, tranchés.**

- `DenoiseFrame` → **sampler**. C'est le paramètre du callback observateur dans la signature
  de `sample_diffusion` (`observer: Option<&mut dyn FnMut(&DenoiseFrame)>`) et la couture
  sampler↔visualiseur (`live_frame`). Il porte des tenseurs *empruntés* (`latent`, `x0_hat`),
  pas des statistiques : c'est un contrat du sampler, pas une métrique. Le laisser dans
  `metrics` ferait dépendre la signature du sampler d'un type d'instrumentation.
- `DenoiseStepStat` → **metrics**. C'est un *résumé statistique* par pas (trois champs `Stats`),
  produit dans l'unique but d'être remis à `log_trajectory` qui l'écrit en JSONL. Un enregistrement
  de métrique dans chacun de ses champs. `sample_diffusion` le *remplit* comme un out-param
  optionnel (`trajectory`) — exactement comme il ne remplit rien quand on ne le lui demande pas ;
  il reste donc avec la métrique qu'il alimente, et `sampler.rs` l'importe.
- `compose_diffusion_input` → **sampler**, comme le veut la mission : elle compose l'entrée
  `[signal | timestep]` du modèle, cœur de l'appel réseau. Appelants vérifiés avant de trancher :
  `predict_epsilon`/`reverse_step_async` (sampler), `probe_diffusion` (metrics), `eval.rs`
  (eval d'entraînement), `batlab_web`. Utilisée des deux côtés, mais elle est de nature
  *inférence* — c'est la primitive de composition du chemin d'appel du modèle.

**Couplage résiduel assumé** : `sample_diffusion` calcule `Stats::of(...)` en ligne et bâtit un
`DenoiseStepStat` quand une trajectoire est demandée, donc `sampler.rs` importe `Stats` et
`DenoiseStepStat` de `metrics.rs` ; réciproquement `metrics.rs` (via `probe_diffusion`) importe
`compose_diffusion_input` de `sampler.rs`. Les deux sont des modules frères d'un même crate, l'usage
mutuel est légal et sans cycle de crate. `Stats` est une structure pure (aucun `fs`, aucun GPU),
donc ce couplage ne blesse pas la portabilité wasm. Le supprimer (sortir le calcul de `Stats` de
`sample_diffusion`) serait un **changement de logique**, hors du périmètre de ce déplacement pur.

**Discipline des deux commits.**

1. *Déplacement pur.* Le code quitte `metrics.rs` pour `sampler.rs` verbatim ; `metrics.rs`
   re-exporte le tout (`pub use super::sampler::*`) et `mod sampler;` reste privé, si bien
   qu'**aucun appelant ne change d'import** — `metrics::…`, les réexports `mod.rs`/`lib.rs` et
   les paths internes (`drift.rs`, `eval.rs`) résolvent inchangés. Seule concession forcée par le
   déplacement : l'import `PosteriorVariance` de `metrics.rs`, devenu mort (son seul usager est
   parti), est retiré pour tenir le zéro-warning ; ce n'est pas de la logique.
2. *Imports et documentation.* `mod.rs` passe à `pub mod sampler;` et scinde ses `pub use` entre
   `metrics::{…}` et `sampler::{…}` selon le vrai foyer ; le pont `pub use super::sampler::*` est
   retiré et remplacé par l'unique `use …sampler::compose_diffusion_input` dont `probe_diffusion` a
   besoin ; `drift.rs` (`super::metrics::reverse_step_from_epsilon` → `super::sampler::…`, ×4) et
   `eval.rs` (`…metrics::compose_diffusion_input` → `…sampler::…`) sont repointés. `lib.rs` est
   **inchangé** : il source depuis `model::training::{…}`, niveau où les noms restent exposés à plat.
   Ce présent rapport et le CLAUDE.md (section frontière) sont mis à jour.

**Preuve d'innocuité** (aux deux commits) : `cargo build --workspace` sans warning (le seul warning
`mut` du build de test préexiste dans `attention_tests.rs:811`, fichier non touché) ; `cargo test
--workspace` vert (392) ; `git status` propre ; image `--headless-sample` identique au **SHA-256**
avant/après sur `Greyscale_Diffusion_L` et `Stable_Diffusion` (PNG **et** JSONL de métriques, donc
les deux chemins couverts) ; crate web toujours compilé en `wasm32-unknown-unknown` ;
`the_base_noise_fold_is_written_in_exactly_one_place` toujours vert (le `const BASE_NOISE_FOLD` et son
détecteur ont voyagé ensemble).

## T2. La version de checkpoint est un tuple `(bool, bool, bool)` aux combinaisons impossibles

`model.rs:1175-1186` décode le magic en
`(has_optimizer_trailer, has_ema_trailer, quantized)` :

```
V3 => (true,  true,  false)
V2 => (true,  false, false)
V1 => (false, false, false)
Q8 => (false, false, true)
```

Quatre combinaisons valides sur huit ; `(quantized=true, has_ema=true)` est **impossible**
mais représentable. C'est exactement le « couple de booléens dont une combinaison est
impossible » que la mission vise. Un `enum CheckpointVersion { V1, V2, V3, Q8 }` avec
accesseurs `has_optimizer_trailer()`, `has_ema_trailer()`, `quantized()` rendrait
l'impossible non-représentable.

**Confiance modérée** : le tuple est décodé une seule fois et consommé localement dans la
même fonction — le risque de divergence est faible. Bon candidat de lisibilité, pas une
urgence. À faire si on est déjà dans `model.rs`.

## T3. Trois fois la même description de poids (EMA/brut)

Le couple `(report.carries_ema, report.used_ema)` est traduit en phrase à **deux endroits
strictement identiques** (main.rs:716-723 et main.rs:761-768), et une logique voisine
décrit `CheckpointWeights` en 1369-1370. Extraire `fn describe_weights(&CheckpointLoad)
-> String`. Trivial, sans risque.

## T4. Trois flags parsés mais absents de `--help`

`--seed-noise` (main.rs:2073), `--single-view` (main.rs:2077) et `--seed-dataset`
(main.rs:2081), tous sur `--headless-perpetual`, sont acceptés par la liste blanche mais
**n'apparaissent nulle part dans la const `HELP`** (l.54-286). C'est précisément
l'anti-pattern que le dépôt documente lui-même (les doc-comments de `reject_unknown_flags`
et `dial_level` parlent de l'agent aveugle qui ne peut découvrir un flag caché).
Correction : trois lignes dans `HELP`. Sans risque.

## T5. `main.rs` fait 5939 lignes

Un seul fichier mêle 7 sous-commandes CLI, tout le chargement de datasets (l.4141-4827),
la gestion des checkpoints, l'orchestration training/inférence/perpétuel, et **~1105
lignes de tests** (l.4835-5939, ~19 % du fichier). Blocs cohérents et extractibles :
`dataset`/`image_io` (Dataset, load_dataset, try_load_*, conversions image↔tenseur),
`checkpoints` (weights_source, stripped_checkpoint, run_export_weights, partiels), et le
module tests vers un fichier `tests/`. **Gros chantier, purement mécanique, à faire par
petits commits prudents** — pas prioritaire, mais c'est le fichier le plus dur à relire du
dépôt.

## T6. Import inutilisé

`OptimizerKind` et `WeightInit` importés mais non utilisés (tui/app.rs:12) — le seul
warning du build. `cargo fix` d'une ligne.

## T7. ~16 fonctions `pub` jamais référencées dans le workspace

Le lint `dead_code` ne couvre pas les items `pub` d'une lib ; ces fonctions ne sont
référencées nulle part (tests compris) : `allocator_report` (resources/mod.rs:186),
`content_rms` (eval.rs:149), `with_live_frame` (resources/mod.rs:510),
`training_hyperparameters` (model.rs:437), `input_dim_str`/`output_dim_str` (config.rs:482/497),
`poll_loss_readback`/`request_loss_readback` (model.rs:1795/1740), `normalized_step`
(schedule.rs:184), `state_slots_per_parameter` (optimizer.rs:38), `without_bias`
(attention.rs:67), `input_elem_count`/`output_elem_count` (config.rs:1227/1232),
`type_name` (config.rs:468), `training_mode` (model.rs:739), et le **wrapper**
`estimated_prepare_gpu_bytes` (diffusion.rs:190) dont seule la variante `_for_batch` est
appelée.

**À arbitrer une par une, surtout pas en masse** : plusieurs sont vraisemblablement une
surface d'API destinée à `batlab-web` ou à un usage futur (accesseurs de dimensions,
readback de loss). Le plus clairement mort est le wrapper `estimated_prepare_gpu_bytes`.
Je propose de ne toucher qu'à ceux que tu confirmes.

---

# COSMÉTIQUE / RANGEMENT

## C1. Encombrement de la racine du dépôt

Cinq fichiers `.md` de mission/rapport traînent à la racine — `BLIND_TEST_ATTENTION.md`,
`BLIND_TEST_MANAGER.md`, `MISSION_BLIND_ATTENTION.md`, `MISSION_BLIND_TEST_MANAGER.md`,
`GO_NOGO.md` — au lieu de `docs/reports/` (qui a son `INDEX.md`). Et le dépôt porte
**trois** répertoires de tests à l'aveugle — `blind_tests/`, `blind_tests_attention/`,
`blind_tests_manager/` — quand le CLAUDE.md n'en documente qu'un (« la suite »).
Proposition : déplacer les `.md` sous `docs/reports/`, et soit documenter les deux suites
supplémentaires dans le CLAUDE.md, soit les ranger sous `blind_tests/`. Ce n'est pas
l'écran d'accueil du TUI, mais celui du dépôt : c'est ce que l'auteur voit en `ls`.

## C2. Checkpoints commités — politique incohérente

Cinq `.ckpt` sont commités sous `Models/` (~2,17 Mo), **aucun n'est fixture de test**
(tous les tests passent par `TempRoot`, vérifié) :

| Fichier | Taille |
|---|---|
| `Greyscale_Diffusion/pretrained_weights/latest.ckpt` | 76,6 ko |
| `Greyscale_Diffusion_L/pretrained_weights/scale_run.ckpt` | 1,86 Mo |
| `Stable_Diffusion/pretrained_weights/latest.ckpt` | 80,1 ko |
| `Stable_Diffusion/pretrained_weights/latest_backup.ckpt` | 80,1 ko |
| `Stable_Diffusion/pretrained_weights/one.ckpt` | 80,1 ko |

Les **trois** `.ckpt` de `Stable_Diffusion` font exactement 80 055 octets. Le
`.gitignore` n'ignore que `night_run.ckpt`, et le bloc `web/dist/` y **affirme** « tout
checkpoint est gitignoré » — ce qui est faux pour `Models/`.

> **Correction (phase 2, vérification avant action)** : les trois `.ckpt` ne sont **PAS**
> identiques — trois SHA-256 distincts (même *taille*, contenu différent) : ce sont des
> poids entraînés **distincts**, pas des copies redondantes comme ce diagnostic le
> supposait. **Aucun `.ckpt` n'a donc été supprimé** : ce sont des poids entraînés,
> non régénérables, que l'audit n'a pas créés. Reste, non traité, la seule incohérence
> réelle : le commentaire `web/dist/` du `.gitignore` qui parle de « tout checkpoint »
> — laissé à l'arbitrage, il ne casse rien.

## C3. Modèles de banc sur l'écran d'accueil du TUI

La liste des modèles (l'écran d'accueil du TUI) porte des entrées qui ne sont pas des
modèles de travail :
- `Archi32_A_Baseline`, `_B_TimeBias`, `_C_Wide`, `_D_Residual` — bancs **régénérables**
  par `tools/gen_archi32_bench.py`, configs sans poids.
- `Greyscale_Diffusion_broken` — copie de **diagnostic** (géométrie non-conditionnable),
  citée comme exemple documenté dans `config.rs:934` et `:1291`.
- `Stable_Diffusion` — c'est aussi une **clé de template built-in** (config.rs:959).

Proposition : sortir les `Archi32_*` du commit (régénérables à la demande) pour dégager la
liste d'accueil. **Attention avant de toucher** `Greyscale_Diffusion_broken` (référencé
par des commentaires de `config.rs`) et `Stable_Diffusion` (clé de template) — les sortir
demande de vérifier ces références. **À arbitrer** : c'est *ta* première vue du TUI, à toi
de dire lesquels tu veux y voir.

> **Fait (phase 2)** : les quatre `Archi32_*` sont **sortis** du commit — vérifié sans
> aucun consommateur (aucun banc, blind_test, script ou test ne les lit ; seul
> `gen_archi32_bench.py` les *écrit*, à l'identique, à la demande). Sa docstring, qui les
> disait « commitées », est corrigée. `Greyscale_Diffusion_broken` et `Stable_Diffusion`
> sont **laissés** (références documentées / clé de template), comme signalé.

---

# Ce que je propose de NE PAS faire (hiérarchiser, c'est aussi refuser)

1. **NE PAS collapser les huit occurrences de `0x9e3779b9_7f4a_7c15`.** Contrairement au
   fold de bruit de base (D1), cette constante de Weyl sert des flux **délibérément
   distincts** : le champ de bruit (`STREAM_GAMMA`, gaussian_at), la décorrélation
   Box-Muller (schedule.rs:501), et les flux perpétuels (`DESCENT_STREAM` vs
   `RENOISE_STREAM` vs `FLUX_PATH_STREAM`, avec commentaires explicites sur leur
   indépendance). Les unifier casserait l'isotropie que `ANISOTROPY_HUNT.md` a chèrement
   acquise. Seul le fold de bruit de base doit être canonisé.

2. **NE PAS convertir en enums les booléens uniques bien nommés** — `want_x0_hat`,
   `report_loss`, `maintain_latest`, `quantize`, `opens_cycle`. Un seul booléen lisible à
   l'appel n'a pas besoin d'un type ; un enum l'alourdirait. Seul le tuple `(bool,bool,bool)`
   de la version de checkpoint (T2) mérite un type.

3. **NE PAS changer le défaut `--headless-sample --magnitude` (1.0)** sans arbitrage :
   c'est du DEV/CI, des bancs peuvent en dépendre (D3, sous-cas).

4. **NE PAS supprimer en masse les fn `pub` de T7** : surface d'API potentielle pour le
   web/le futur. Une par une, sur confirmation.

5. **NE PAS réécrire `main.rs` d'un bloc** (T5) : extractions par petits commits, chacun
   prouvé neutre.

---

# Séquence de correction proposée (phase 2), du risque réel au cosmétique

Un commit par point, en commençant par les doublons de logique :

1. **D1** — canoniser `base_noise_seed`/`path_seed` dans `batlab_core`, câbler sampler +
   perpetual + web, ajouter le test-garde grep. *(preuve : sortie `--headless-sample` et
   test async inchangés au bit.)*
2. **D4** — un seul lecteur d'en-tête `.batraw`. *(preuve : tests de chargement inchangés.)*
3. **D2** — câbler `train_cfg.loss` jusqu'au moteur. *(preuve : loss inchangée, run apparié.)*
4. **D3 (sûr)** — le web lit `config.inference.{denoise_magnitude,posterior_variance}`.
   *(preuve : rendu bit-à-bit, la config déployée porte déjà 1.2/beta.)*
5. **T2, T3** — `enum CheckpointVersion`, `describe_weights`. *(preuve : formats et
   messages inchangés.)*
6. **T1** — ✅ fait — sampler sorti dans `model/training/sampler.rs`, `metrics.rs` aux
   métriques (voir §T1 — fait). *(preuve : SHA-256 image identique, tests verts, wasm32 OK.)*
7. **T4, T6** — flags dans `--help`, import inutilisé. *(trivial.)*
8. **T5, C1, C2, C3, T7** — rangements et découpes, sur arbitrage, en dernier.

Critère de recette permanent à chaque commit : `cargo test --workspace` vert **et**
`git status` propre (règle `TempRoot` du dépôt).
