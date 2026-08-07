# EMA des poids, `--resume`, checkpoints périodiques

Branche `ema`. Trois items complémentaires autour de la qualité et de la
robustesse d'entraînement, plus un bug préexistant exhumé en chemin.

**Résumé** — à compléter §5 après la campagne de mesure.

---

## 1. Ce qui est implémenté

### 1.1 La passe EMA (`crates/batlab-core/src/model/shader/ema.wgsl`)

Une passe compute par couche entraînable, encodée **après** la passe
d'optimiseur, dans le même encodeur :

```
ema_weights[i] ← d·ema_weights[i] + (1−d)·weights[i]
ema_bias[i]    ← d·ema_bias[i]    + (1−d)·bias[i]
```

Bindings : `[0]` weights (lecture), `[1]` bias (lecture), `[2]` ema_weights
(rw, persistant), `[3]` ema_bias (rw, persistant), `[4]` l'uniforme `EmaSpecs`.
Deux buffers de la taille des poids par couche, alloués avec les passes
d'optimiseur, exactement la plomberie des moments `m`/`v` d'Adam
(`OptimizerBindings`, `create_opt_pass`).

**Passe séparée plutôt que fondue dans `adam.wgsl`/`sgd.wgsl`.** La récurrence
est la même quel que soit l'optimiseur qui a produit les poids ; une passe à
elle seule est une passe qu'on peut lire, tester et éteindre à elle seule. La
barrière implicite que wgpu insère entre deux passes compute d'un même encodeur
est ce qui fait que l'EMA lit les poids **de ce pas-ci** et pas ceux du pas
précédent — le test l'exige (voir la mutation n° 4, §4.2).

### 1.2 Le warmup : le choix, et pourquoi l'autre a été écarté

`ema ← d·ema + (1−d)·w` avec d = 0,999 a une constante de temps de mille pas.
Partie de l'initialisation aléatoire, la moyenne serait encore majoritairement
du bruit mille pas plus tard — elle serait **pire** que les poids bruts sur
tout le début du run.

Deux mécanismes, **tous deux appliqués** :

1. **Le shadow est semé avec les poids** au moment où la passe est construite
   (`Layer::create_ema_pass`), jamais avec des zéros. Il ne contient donc jamais
   rien que le modèle n'ait tenu d'abord.
2. **La décroissance est rampée** : `d(t) = min(decay, (1+t)/(10+t))`.

La rampe est la règle `ExponentialMovingAverage(num_updates=t)` de TensorFlow.
Elle a été préférée à la **correction de biais** (`ema / (1 − dᵗ)`, le tour
d'Adam appliqué à un shadow initialisé à zéro) pour une raison précise : la
correction de biais *suppose* un shadow parti de zéro, donc elle **interdit** de
le semer sur les poids — et un shadow parti de zéro est un shadow dont tous les
checkpoints des premiers milliers de pas génèrent du bruit. La rampe se
**compose** avec le semis au lieu de le contredire.

Repères : `d(1) = 0,182` · `d(10) = 0,55` · `d(100) = 0,918` ·
`d(1000) = 0,991` ; le 0,999 nominal prend la main vers **t ≈ 9990**.

> **À retenir pour un run court.** Sur 1200 pas, `d(1200) = 0,9926 < 0,999` :
> c'est la **rampe** qui lie tout du long, quelle que soit la valeur nominale
> demandée. La fenêtre effectivement moyennée fait ~135 pas en fin de course,
> pas 1000 — et `--ema 0.9999` y donnerait exactement le même résultat que
> `--ema 0.999`. La valeur nominale ne commence à distinguer les runs qu'au-delà
> de ~10 000 pas.

Le decay effectif est calculé **sur le CPU en f64** et passé en uniforme, comme
les corrections de biais d'Adam et pour la même raison : le compteur de pas vit
là, et le ratio est à un cheveu de 1 sur l'essentiel d'un run.

### 1.3 Le format de checkpoint : `BBCKPT3`

Troisième version, même patron que `BBCKPT2` (`OPTIMIZER_ADAM`) et `BATRAW2` :

```
BBCKPT1 : entrées poids/biais
BBCKPT2 : BBCKPT1 + remorque d'état d'optimiseur (tag, t, m/v par couche)
BBCKPT3 : BBCKPT2 + remorque EMA (tag, decay, [ema_weights, ema_bias] par couche)
```

**`BBCKPT3` n'est écrit que par un run qui garde une moyenne.** Un run sans
`--ema` produit toujours l'octet pour octet le `BBCKPT2` qu'il produisait avant
que ce fichier existe — la version est décidée par ce qu'il y a à écrire, pas
par ce que le code sait écrire.

V1 et V2 restent lisibles. Charger un V1/V2 dans un run qui moyenne **resème le
shadow sur les poids chargés** ; le laisser sur le tirage aléatoire de la
construction est le mode de défaillance qui compte ici — le run aurait l'air
sain pendant des heures en écrivant des moyennes moitié-poids-entraînés,
moitié-bruit.

### 1.4 Quel jeu de poids atterrit où

`CheckpointWeights::{Raw, Ema}` est un paramètre du **chargement**, décidé au
site d'appel, pas un état du modèle :

- **l'entraînement reprend sur l'itéré** (`Raw`) — les moments d'Adam décrivent
  cet itéré, et la moyenne n'est pas un point que l'optimiseur ait jamais
  visité ;
- **la génération prend la moyenne** (`Ema`) quand le fichier en porte une,
  parce que c'est le jeu pour lequel la moyenne existe.

Les quatre chemins de génération (les deux headless et les deux du TUI) passent
par `load_sampling_checkpoint` : une décision en un endroit au lieu de quatre
qui peuvent diverger. `CheckpointLoad` rapporte `carries_ema` **et** `used_ema`,
et leur écart est exactement le cas — « j'ai demandé la moyenne, le fichier n'en
a pas » — qui transformerait une comparaison EMA/brut en comparaison d'une chose
avec elle-même. Les deux chemins **l'annoncent à l'écran**.

### 1.5 `--resume` (item 2)

Le chemin headless forçait `load_checkpoint = false` : impossible de continuer
un entraînement en ligne de commande. `--resume <ckpt>` charge poids + moments
Adam + compteur de pas `t` + EMA et poursuit. Il écrit dans `--out`, donc un run
repris **n'écrase jamais le fichier dont il vient** sauf si on le lui demande.
Il prend le pas sur le mode de chargement du `config_file` : c'est une
instruction explicite de ligne de commande contre un défaut.

### 1.6 `--checkpoint-every N` (item 3)

**Rotation, pas historique** : un seul fichier `<out stem>.partial.ckpt`,
réécrit à chaque fois. Le besoin servi est « un run de dix heures qui meurt à la
neuvième n'est pas perdu », pas « garder tous les intermédiaires » — dix-sept
fichiers de 7 Mo pour une nuit d'entraînement serait une autre fonctionnalité,
et un moins bon défaut.

L'écriture est **atomique** : fichier temporaire à côté de la cible, puis
`rename`. Un checkpoint pèse des dizaines de mégaoctets ; un kill au milieu d'un
`fs::write` laisserait un fichier tronqué qui finit quand même en `.ckpt`, et le
run qui le reprend échouerait sur une longueur — au mieux. Avec le rename, le
partiel est toujours soit le précédent en entier, soit le nouveau en entier.

Un échec d'écriture **n'est jamais fatal** (un run ne meurt pas d'un disque
plein au pas 5000) mais il est bruyant à chaque occurrence.

C'est un `.ckpt` ordinaire, et c'est voulu : `--resume` et le sélecteur de poids
du TUI le voient tous les deux — le but est de pouvoir rattraper un run mourant.

### 1.7 Le TUI

Une cinquième ligne au formulaire d'entraînement, « EMA decay », entre « Steps »
et le bascule « Start from random ». **Vide = pas de moyenne**, affiché `off`.
1,0 fige la moyenne sur les poids initiaux pour toujours et 0,0 en fait une
copie des poids : deux réglages silencieusement inutiles plutôt que bruyamment
faux, donc refusés **sur le formulaire** et pas six heures plus tard. Entrée
d'aide sourcée sur ce rapport (`help.rs`).

---

## 2. Le bug préexistant exhumé

`resize_batch` — changer le batch d'un run **en cours** depuis le moniteur —
paniquait sur « at least one layer required for training ». Cause : `build()` et
`build_model()` appelaient `clear()` quand le modèle était déjà construit, et
`clear()` vide `self.layers`. Le **second** `build()` de la vie d'un modèle
échouait donc, et `resize_batch` n'a que ce chemin.

Sans rapport avec l'EMA — reproduit sans elle. Exhumé en écrivant le test qui
vérifie que le shadow survit à un resize. Le panneau d'aide promettait
l'inverse (« la reconstruction préserve poids, biais, moments Adam et compteur
de pas ») et CLAUDE.md le documente comme un contrat.

Corrigé : `discard_built_state()` jette tout ce que `build()` a produit en
**gardant la liste des couches** ; `clear()` garde sa sémantique destructrice
pour les chemins qui re-spécifient le modèle. Deux tests le tiennent.

---

## 3. Le contrat observable des flags

Ce que promet le binaire, pour un futur test aveugle. `--help` est la source
de vérité et il porte tout ce qui suit.

### `--ema F` (entraînement)

| Situation | Comportement observable |
| --- | --- |
| absent | Aucune moyenne. Poids **bit à bit** identiques au run sans ce flag, et checkpoint **octet pour octet** identique (magic `BBCKPT2`). |
| `F` hors de `]0, 1[`, ou non numérique | Erreur avant tout travail GPU, message nommant le flag et l'intervalle. |
| `F` valide | Magic du checkpoint = `BBCKPT3`. Le corps du `BBCKPT2` du même run en est un **préfixe exact**. Bannière : `ema=F`. |

### `--raw-weights` (échantillonnage, perpetual)

| Situation | Comportement observable |
| --- | --- |
| absent, fichier avec EMA | Génère depuis la **moyenne**. Ligne `[weights] <path> → EMA weights (decay D)`. |
| absent, fichier sans EMA | Génère depuis les poids. Ligne `… → raw weights (the file carries no EMA)`. Pas d'erreur. |
| présent, fichier avec EMA | Génère depuis les poids. Ligne `… → raw weights (the file also carries an EMA)`. |
| — | C'est un flag **nu** : il n'avale pas l'argument suivant. |

### `--resume <ckpt>` (entraînement)

| Situation | Comportement observable |
| --- | --- |
| absent | From scratch, comme avant. Les bancs existants ne bougent pas. |
| fichier absent | Erreur `--resume: no such checkpoint: <path>`, avant tout travail GPU. |
| autre géométrie | Erreur nommant la couche, la longueur attendue et la reçue. Rien n'est tronqué. |
| compatible | Le compteur de pas repart de sa valeur sauvegardée (annoncé : `optimiser step N`). Écrit dans `--out`, jamais dans le fichier repris. |
| fichier avec EMA + `--ema` | `EMA restored (file decay X, this run Y)`. |
| fichier avec EMA, sans `--ema` | **AVERTISSEMENT** explicite : la moyenne ne sera pas reportée dans le checkpoint que ce run écrit. |
| fichier sans EMA + `--ema` | `no EMA in the file: the shadow starts on these weights`. |

### `--checkpoint-every N` (entraînement)

| Situation | Comportement observable |
| --- | --- |
| absent | Sauvegarde en fin de run uniquement. Inchangé. |
| `N = 0` ou non numérique | Erreur avant tout travail GPU. |
| `N` valide | Un fichier `<out stem>.partial.ckpt`, **réécrit** tous les N pas, jamais un par pas. Ligne `[checkpoint] step N → <path> (X MB, Y ms)`. Pas de `.ckpt.tmp` résiduel après un pas réussi. Le dernier pas n'en écrit pas (la sauvegarde finale le couvre). |
| échec d'écriture | Message sur stderr, le run **continue**. |

---

## 4. Preuves

### 4.1 Suite de tests

`crates/batlab-core/src/model/ema_tests.rs` et `model/ema.rs` :

| Test | Ce qu'il tient |
| --- | --- |
| `the_shadow_follows_its_f64_reference_step_by_step` | Oracle CPU **f64** de la récurrence **avec la rampe**, comparé pas à pas sur 12 pas. L'oracle dérive la rampe du compteur lui-même au lieu de recevoir le decay effectif — un off-by-one dans `publish_optimizer_specs` y apparaît. Decay 0,9 : à t=12 c'est encore la rampe qui lie (13/22 = 0,591), donc une implémentation qui ignore le warmup est à un facteur > 10 du résidu. |
| `the_warmup_ramp_matches_its_closed_form` / `…only_ever_rises` | Les chiffres cités dans ce rapport, contre la forme close ; monotonie sur 5000 pas. |
| `the_shadow_starts_on_the_weights_not_on_zero` | Vérifié **au pas 0**, avant tout entraînement. |
| `a_higher_decay_holds_the_shadow_further_back` | La propriété pour laquelle la fonctionnalité existe, énoncée sans référence à l'implémentation. |
| `a_run_without_an_ema_is_bit_identical_and_writes_a_v2_checkpoint` | Poids bit à bit identiques à un jumeau sans la passe ; **le V2 est un préfixe exact du V3**. |
| `a_checkpoint_round_trips_both_weight_sets` | Les deux jeux + le compteur de pas ; puis le même fichier chargé en `Ema` met bien la moyenne dans les buffers de poids. |
| `asking_for_an_average_a_checkpoint_lacks_falls_back_on_the_weights` | Le repli est annoncé, pas silencieux. |
| `a_v2_checkpoint_reseeds_the_shadow_instead_of_leaving_it_random` | Le mode de défaillance de §1.3. |
| `a_v1_checkpoint_still_loads_into_an_averaging_run` | Compatibilité descendante complète. |
| `a_checkpoint_of_another_geometry_is_refused` | Ce sur quoi `--resume` s'appuie. |
| `resizing_the_batch_preserves_the_shadow` | Et le bug de §2. |
| TUI : `the_ema_row_starts_empty_…`, `a_decay_typed_on_the_form_reaches_…`, `the_form_refuses_a_decay_outside_…`, `typing_on_the_ema_row_leaves_the_dataset_path_alone` | Le formulaire, dont le piège d'index de ce dépôt (voir §6). |

### 4.2 Validation par mutation

Six mutants, **six tués** :

| # | Mutation | Tests tués |
| --- | --- | --- |
| 1 | Shader : `decay` et `(1−decay)` échangés | `the_shadow_follows_its_f64_reference_step_by_step`, `a_higher_decay_holds_the_shadow_further_back` |
| 2 | Rampe de warmup supprimée (decay nominal partout) | `the_warmup_ramp_matches_its_closed_form`, `the_shadow_follows_its_f64_reference_step_by_step` |
| 3 | Shadow non semé (part de zéro) | `the_shadow_starts_on_the_weights_not_on_zero`, `the_shadow_follows_its_f64_reference_step_by_step` |
| 4 | Passe EMA encodée **avant** l'optimiseur | `the_shadow_follows_its_f64_reference_step_by_step` |
| 5 | Off-by-one : `effective_decay(t + 1)` | `the_shadow_follows_its_f64_reference_step_by_step` |
| 6 | Remorque EMA ignorée à la lecture | `a_checkpoint_round_trips_both_weight_sets`, `resizing_the_batch_preserves_the_shadow`, `a_checkpoint_of_another_geometry_is_refused` |

### 4.3 Équivalence bit-à-bit, vérifiée sur les fichiers entiers

Au-delà de l'unitaire, trois runs de 12 pas sur `Greyscale_Diffusion_L`
(`adam`, lr 1e-3, batch 8, CIFAR-10 gris) :

```
plain_a.ckpt  vs  plain_b.ckpt   (deux runs sans --ema)   → IDENTIQUES
plain_a.ckpt  (BBCKPT2, 5 573 679 o)
withema.ckpt  (BBCKPT3, 7 431 647 o)
corps du V2 == préfixe du V3        → vrai
remorque EMA                        → 1 857 968 o (+33,3 %)
```

Le run avec `--ema` produit donc **exactement les mêmes poids** que le run sans,
sur toute la longueur du fichier, plus la remorque.

---

## 5. La mesure : EMA contre poids bruts

*(à compléter)*

---

## 6. Le piège d'index du formulaire

`TrainingParamsState::fields` était `[lr, batch, steps, dataset_path]`, avec le
bascule « Start from random » partageant l'index 3 avec le chemin du dataset —
et une backspace non gardée y avait **déjà** mangé ce chemin un caractère par
frappe, depuis un écran où rien ne bouge (le commentaire du code le raconte).

Insérer une ligne devant le dataset est exactement l'édition qui recrée ce bug.
`fields` devient donc `[lr, batch, steps, ema, dataset_path]` : `field_idx`
indexe **directement** `fields` pour les quatre lignes tapées, et les quinze
`fields[3]` deviennent `fields[TRAINING_DATASET_FIELD]` — nommé, plus jamais un
nombre nu. `typing_on_the_ema_row_leaves_the_dataset_path_alone` le tient, sur
la ligne EMA **et** sur le bascule.

`nav_tests` a signalé la ligne ajoutée immédiatement : la suite parcourt tous
les écrans à la touche depuis la porte d'entrée, et un Entrée de plus était
nécessaire pour traverser le formulaire.
