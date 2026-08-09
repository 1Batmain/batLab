# SEED_DATASET — la graine d'une dérive, et d'où elle vient

Mission courte, née d'un défaut trouvé à l'usage : `Models/Elephants_XL`,
entraîné sur `datasets/elephants256.batraw`, lançait sa dérive perpetual
**depuis une photo CIFAR au hasard** — un camion, un chien. Pas faux au sens du
code, absurde au sens de l'utilisateur, et rien à l'écran ne prévenait.

---

## 1. La cause racine : un champ déclaré, documenté, jamais lu

`PerpetualConfig::seed_dataset` existait dans le format de config depuis la
mission img2img. Sa doc disait exactement ce qu'il fallait :

> *The dataset a run drifts away from — `Models/<name>/config_file` may name
> one, otherwise it is derived from the model's output channels.*

Trois choses étaient vraies en même temps :

1. **Le chemin TUI le lisait** (`run_perpetual`, `main.rs`).
2. **Le chemin headless ne le lisait pas** : `run_headless_perpetual` chargeait
   le `config_file` uniquement pour l'architecture, puis résolvait sa graine
   **à sa façon** — `--seed-dataset` s'il était passé, sinon la convention par
   canaux.
3. **Rien ne permettait de l'écrire** : aucun champ de formulaire, et le TUI se
   contentait de recopier ce qu'il avait lu. Aucun `config_file` du dépôt ne
   portait la clé. Elle n'a jamais pu être posée.

Le défaut n'est donc pas que le classement était faux — c'est qu'il **existait
en deux exemplaires**, dont un incomplet. Deux façons de répondre à une seule
question, c'est ainsi qu'elles divergent, et un relecteur qui lit l'une des deux
isolément ne voit rien d'anormal. C'est le même mécanisme que
`load_sampling_checkpoint` a fermé pour les poids.

---

## 2. Le contrat observable

### 2.1 Trois sources, un classement, une fonction

```
--seed-dataset <path>          (drapeau)      ─┐
ModelConfig::seed_dataset      (config)       ─┼─→ storage::Storage::resolve_seed_dataset
défaut par canaux de sortie    (convention)   ─┘
```

`Storage::resolve_seed_dataset(flag, configured, output_channels)` est **le seul
endroit qui connaît ce classement**. Elle rend un `SeedDatasetChoice { path,
source }` ; elle ne lit aucun octet et ne teste pas l'existence du fichier —
c'est un *nom*, et ce qu'un fichier manquant veut dire dépend de la source.

La convention, dernier recours, reste ce qu'elle était :
`default_seed_dataset_name(1) = cifar10_grey.batraw`,
`(3) = cifar10_rgb.batraw`, tout le reste `None`. Elle est dérivée des canaux de
**sortie** du modèle, pas devinée.

**Chemins relatifs — deux règles, et c'est voulu** :

| source | relatif à | pourquoi |
|---|---|---|
| drapeau | le répertoire courant | il est tapé dans un shell et garde le sens du shell |
| config | la racine du projet | il est **stocké** et relu depuis n'importe où : `datasets/elephants256.batraw` doit vouloir dire la même chose demain |
| convention | `<racine>/datasets/` | c'est là que cet hôte range ses datasets |

### 2.2 Ce qu'une résolution rend

`SeedImages::resolve(flag, &config, output_size)` — **la seule porte**, et elle
prend le `ModelConfig` **entier**, pas un chemin déjà choisi : un appelant ne
peut pas « oublier » de consulter le modèle.

| cas | résultat |
|---|---|
| trouvé, géométrie compatible | `Ok(Some(_))`, avec sa `source` |
| rien de nommé, rien à dériver, ou le fichier dérivé absent | `Ok(None)` → **repli sur bruit pur**, annoncé |
| **nommé** (drapeau ou config) et absent / vide / mauvais canaux | `Err(_)` |

La dissymétrie est le contrat : un dataset qu'un humain a **nommé** et qui
manque doit être une erreur, parce que retomber sur du bruit ressemblerait
exactement au réglage ignoré — le bug qu'on est en train de fermer.

### 2.3 Le garde-fou de géométrie : canaux refusés, taille rééchantillonnée

Un dataset dont les **canaux** ne collent pas au modèle est **refusé**, en
nommant les deux nombres. Largeur et hauteur, elles, sont rééchantillonnées.

Ce n'est pas une inconséquence :

- Réduire une photo 64×64 en 32×32 est une chose **visible** qu'on fait à une
  image ; c'est encore la même photo.
- Un fichier gris donné à un modèle couleur est **répliqué sur R, G et B** ; un
  fichier couleur donné à un modèle gris est **aplati**. Les deux
  *réussissent* — et c'est précisément ce qui les rend dignes d'être refusés.
  Le dépôt a déjà payé ce silence une fois, côté chargement de dataset.

Le refus tombe à **deux** endroits, et les deux comptent : dans le sélecteur du
TUI (sur l'écran qui vient de choisir, avec la raison affichée, sans rien
changer au réglage) et dans `SeedImages::resolve` (pour le drapeau, et pour un
`config_file` édité à la main). La convention ne peut jamais y tomber : elle est
dérivée de ces canaux-là.

### 2.4 Où le réglage vit : `ModelConfig`, pas `PerpetualConfig`

`seed_dataset` a été **remonté au niveau du modèle**, à côté d'`inference`.

C'est une propriété du **modèle** — « à quoi ressemblent ses images » — pas
d'un run. Rangé sous `RunMode::Perpetual`, il tenait dans `run.mode`, qui ne
garde que le **dernier** run : une seule nuit d'entraînement réécrivait le mode
et le réglage disparaissait, sans rien à l'écran pour le dire. C'est la même
raison qui fait d'`InferenceConfig` une section à part plutôt qu'une charge
utile de `RunMode::Infer`.

Deux conséquences testées :

- **Il survit à un run d'entraînement** — `compose_run_config` le recopie quel
  que soit le mode.
- **Il ne colle pas au modèle suivant** — il est lu **hors** du `match` sur
  `run.mode`, donc ouvrir un modèle qui n'en nomme aucun efface celui du
  précédent au lieu de le laisser traîner.

Aucune migration : **aucun `config_file` du dépôt ne portait la clé** (vérifié
sur les sept modèles de `Models/`), et serde ignore les champs inconnus — un
fichier qui l'aurait portée à l'ancien emplacement se relit sans erreur.

### 2.5 Le réglage dans le TUI

Le formulaire Perpetual gagne une ligne, **« Seed Dataset »**, en avant-dernière
position (index 5, avant « Regime ») :

```
Random Seed · Seed · Magnitude · Renoise Depth (t_r) · Steps / second · Seed Dataset · Regime
```

C'est une **porte vers un sélecteur**, pas un champ tapé — `Screen::SeedDatasetSelector`,
ouvert par `Entrée`, `→` ou `espace`. Trois raisons, dans l'ordre :

1. Un sélecteur peut montrer la **géométrie de chaque candidat** et marquer
   `✗ canaux` celui qui ne passera pas — c'est tout l'intérêt du garde-fou d'y
   voir avant de choisir.
2. Un chemin est long à taper.
3. Un champ de texte libre aurait volé `q` (règle du dépôt : sur un écran de
   saisie, toute touche imprimable est du texte). Le formulaire Perpetual garde
   donc son `q` = quitter, parce qu'aucun chemin ne s'y tape.

La **première ligne** du sélecteur rend la décision à la convention — et **nomme
le fichier** qu'elle prendra (`(défaut) cifar10_rgb.batraw`). « Défaut » sans
nommer le fichier est exactement le mot qui a laissé un modèle couleur dériver
depuis des camions CIFAR sans que rien n'ait l'air anormal.

`Esc` revient au formulaire sans rien changer ; une sélection acceptée avance le
curseur d'une ligne, pour qu'un `Entrée` de plus traverse le formulaire au lieu
de rouvrir le sélecteur qu'il vient de fermer.

L'entrée d'aide correspondante est dans `tui/help.rs`, sourcée sur ce rapport.

### 2.6 La bannière et le panneau nomment la source

Les deux affichaient déjà « origine → … ». Ils disent maintenant **laquelle des
trois sources a gagné** :

```
origine → image du dataset /…/datasets/elephants256.batraw (256 images) [config_file] — dérive img2img
origine → image du dataset /…/datasets/cifar10_rgb.batraw (50000 images) [défaut par canaux] — dérive img2img
origine → bruit pur en haut du schedule (--seed-noise)
origine → bruit pur en haut du schedule (aucun dataset de graine trouvable)
```

Panneau du moniteur, même chose en court : `elephants256.batraw [config_file]`.

Le nom du fichier seul ne suffisait pas : la question qu'a un utilisateur
surpris n'est pas « quel dataset ? » mais « **pourquoi celui-là ?** ». Et les
deux derniers cas de bruit pur étaient auparavant un seul message, qui
attribuait à `--seed-noise` un repli qui pouvait être un dataset introuvable.

---

## 3. Ce qui empêche la rechute

`every_path_that_starts_a_drift_resolves_its_seed_the_same_way` (`main.rs`) lit
**la source du binaire**, la découpe en fonctions de premier niveau, et exige
que toute fonction mentionnant `PerpetualDrift::from_image` — le constructeur
qui veut dire « pars d'une image » — appelle aussi `SeedImages::resolve`.

Même discipline que `nothing_in_the_engine_opens_an_untimed_pass`. Vérifié par
mutation : en remplaçant la résolution de `run_headless_perpetual` par une
graine fabriquée sur place, le test échoue en **nommant la fonction fautive** —

```
these start a perpetual drift without going through SeedImages::resolve, …: ["run_headless_perpetual"]
```

Il refuse aussi de passer si le balayage ne voit plus **au moins deux** fonctions
ouvrant une dérive : un détecteur qui ne détecte plus rien passerait sinon en
silence.

Deuxième garde, structurel : la signature. `resolve` prend `&ModelConfig`, pas un
`Option<&str>` déjà choisi — c'est ce qui rend « oublier la config » impossible
à écrire plutôt que seulement détectable.

---

## 4. La suite de tests

| test | fichier | ce qu'il tient |
|---|---|---|
| `the_seed_dataset_ranking_is_flag_then_config_then_convention` | `storage.rs` | le classement des trois sources, et les deux règles de chemin relatif |
| `the_convention_follows_the_models_output_channels` | `storage.rs` | 1 → grey, 3 → rgb, le reste rien |
| `the_flag_beats_the_config_which_beats_the_convention` | `main.rs` | le même classement **à travers un vrai chargement**, sur trois fichiers distinguables |
| `a_seed_dataset_of_the_wrong_channels_is_refused_not_resized` | `main.rs` | le refus, sur les deux sources nommées, en nommant les deux comptes |
| `every_path_that_starts_a_drift_resolves_its_seed_the_same_way` | `main.rs` | une seule résolution (balayage de source, vérifié par mutation) |
| `a_seed_dataset_chosen_in_the_form_survives_being_written_and_reopened` | `app.rs` | le tour complet : sélecteur → `config_file` → réouverture |
| `a_training_run_does_not_erase_the_models_seed_dataset` | `app.rs` | il survit à un changement de `run.mode` |
| `opening_a_model_without_a_seed_dataset_clears_the_previous_ones` | `app.rs` | il ne colle pas au modèle suivant |
| `the_chooser_refuses_a_dataset_of_the_wrong_channels` | `app.rs` | le refus côté TUI, réglage inchangé, on reste sur l'écran |
| `the_default_row_clears_the_choice_and_names_what_takes_over` | `app.rs` | la ligne « défaut » nomme le fichier qu'elle rend à la convention |
| `a_seed_dataset_that_is_missing_…` (existant, adapté) | `main.rs` | nommé+absent = erreur, dérivé+absent = repli |

Plus les gardes existantes qui ont tenu sans être touchées :
`every_help_entry_sits_on_the_field_it_describes` (elle a **échoué** dès l'ajout
de la ligne au formulaire, tant que l'entrée d'aide manquait),
`a_cited_source_is_a_report_that_exists` (elle a exigé ce fichier-ci), et
`every_screen_is_reachable` de `nav_tests` (elle a exigé le câblage de
`Screen::SeedDatasetSelector`).

---

## 5. Deux autres champs de la même classe

Un balayage de tous les champs sérialisés du `config_file` en a trouvé deux
autres. La classe est la même : la doc promet un comportement que rien
n'honore.

### 5.1 `TrainingConfig::{optimizer, weight_init, loss_weighting}` — CORRIGÉ

Le pire des deux, et la même forme d'un cran plus loin : les trois champs **sont
lus** par la boucle d'entraînement — mais la seule façon interactive de lancer un
run les détruisait d'abord. `App::finish_dataset_selector` construisait sa
`TrainingConfig` avec :

```rust
optimizer: OptimizerKind::default(),   // = Sgd
weight_init: WeightInit::default(),
loss_weighting: LossWeighting::default(),
```

Aucun de ces trois n'a de champ de formulaire. Donc un modèle configuré par
`--optimizer adam` **repassait en SGD** dès qu'on le relançait depuis le TUI —
et la config était réécrite ensuite, rendant la perte définitive. Adam converge
~20× plus vite en nombre de pas (`OPTIMIZER_ADAM.md`) : la note se paie en
heures de GPU.

Aggravant : la page Resources, un écran plus tôt, lit la valeur du **fichier**
(`self.resources_optimizer = train.optimizer`, avec le commentaire « Adam double
les postes de paramètres… on ne peut pas honnêtement deviner »). Elle
dimensionnait donc un run Adam pendant que le même écran en lançait un SGD.

Corrigé : `App` porte les trois valeurs chargées et les recopie dans la
`TrainingConfig` du run. Tenu par
`starting_a_run_from_the_form_keeps_the_optimizer_the_config_asked_for`.
(La version plus large — leur donner des champs de formulaire — n'est pas faite.)

### 5.2 `TrainingConfig::loss` — SIGNALÉ, non corrigé

Champ obligatoire du schéma, affiché sur le moniteur, recopié à l'identique par
le CLI — et **jamais consommé**. Les deux seuls appels à `Model::new_training*`
passent un littéral `PLoss::MeanSquared`, pas le champ.

Inerte aujourd'hui parce que `config::LossMethod` n'a qu'une variante : c'est un
piège latent, pas un bug vivant. Ajouter une seconde variante ne ferait
silencieusement rien. Deux issues, aucune triviale — le brancher jusqu'à
`build_execution_model`, ou supprimer le champ. **Non corrigé ici**, à décider.

### 5.3 Hors config : `storage::read_batraw_header` était aveugle à BATRAW3 — CORRIGÉ

Trouvé en chemin, parce que le sélecteur de graine en dépend pour lire la
géométrie des candidats. La copie `batlab-ui` du lecteur d'en-tête n'acceptait
que `BATRAW2`/`BATRAW1` — soit **aucun fichier écrit par les convertisseurs
actuels**, qui produisent tous du `BATRAW3` depuis la mission u8.

Conséquence : `Storage::dataset_spec` rendait `None` pour tous les datasets
réels, et la page Resources affichait *« streamed — nothing: this inventory was
asked without a dataset »* là où il y avait des gigaoctets. Un **sous-rapport
silencieux** : rien ne disait que l'en-tête avait été refusé.

Deuxième moitié, plus discrète : `sample_bytes` multipliait par `4` en dur.
Ajouter la magie sans toucher à ça aurait **sur-rapporté** la résidence d'un
BATRAW3 d'un facteur 4 — un échantillon u8 compté comme du f32. La largeur du
payload est donc rendue par le lecteur, pas supposée.

La copie du crate binaire (`main.rs`) gérait les trois magies et **était
testée** ; celle de `batlab-ui` ne l'était pas, d'où la dérive. Les deux copies
restent dupliquées entre crates — la correction durable serait une seule
implémentation dans `batlab_core`. Tenu ici par
`the_header_reader_accepts_every_magic_and_reports_the_payload_width`.

---

## 6. Limites connues

- **`PerpetualConfig::tempo` n'a pas d'équivalent headless.** Sa doc dit
  « reverse steps per second the run is paced to » ; c'est vrai du chemin TUI
  seul, la boucle headless court à la vitesse du sampler. Signalé, pas corrigé —
  ce n'est pas un champ non lu, c'est un champ dont la portée n'est pas dite.
- **Le sélecteur ne liste que `datasets/`.** Un chemin ailleurs se pose par
  `--seed-dataset` ou en éditant le `config_file` ; les deux sont honorés et
  validés, mais le TUI ne sait pas les choisir.
- **La géométrie n'est vérifiée que sur les `.batraw`.** Un dossier d'images
  n'a pas d'en-tête à lire, et ses canaux sont décidés par le chargeur
  exprès : ces candidats passent, sans caution.
