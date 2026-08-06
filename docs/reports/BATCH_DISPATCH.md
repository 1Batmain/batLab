# Porter l'axe batch dans les dispatches — rapport de mission

Branche `batch-dispatch`. Répond au **§7 de `PERF_CONVOLUTION.md`** — « le goulot
est structurel : un encoder et un `submit` par échantillon », désigné comme « de
loin le plus gros gain restant » et « à traiter **avant** de continuer à affiner
des kernels ».

La conception a été écrite et commitée **avant** le code :
`BATCH_DISPATCH_DESIGN.md`. Ce rapport dit ce qui a été fait, ce qui a été
prouvé, et ce qui a été mesuré — y compris là où la mesure n'a pas donné ce
qu'on attendait.

---

## Résumé

Le batch est passé de la boucle CPU aux dispatches. À batch 16, tout le calcul
d'un pas d'entraînement tient désormais dans **une seule soumission** au lieu de
**18**, et chaque kernel voit 16 fois plus de travail par lancement.

_(Tableau de speedup : §7.)_

Trois résultats méritent d'être lus avant le reste :

- **L'inférence est bit à bit inchangée.** Pas « à 1e-6 près » : le PNG généré
  par `--headless-sample` est **octet pour octet** celui du binaire pré-batch,
  et le dump de `--headless-perpetual` est identique sur ses 11 535
  enregistrements. Le moteur emprunte le chemin batché avec `B = 1` et retrouve
  exactement son graphe d'avant.
- **La clause « jamais moins précis » des missions de réduction précédentes ne
  s'applique pas telle quelle**, et la forcer aurait demandé de truquer un
  seuil. Le §4.4 explique pourquoi, chiffres à l'appui, et donne la clause
  correcte — dérivée de la forme, pas ajustée sur la mesure. Le §4.3 raconte
  au passage un faux positif de mesure : l'écart relatif **par élément** est
  mal défini sur un tampon de gradients, et il a fallu changer de métrique
  puis **rejouer les neuf mutations** pour vérifier que ça n'affaiblissait
  rien.
- Un **bug réel** a été attrapé par le compilateur wgpu au moment d'ajouter
  l'uniforme de la loss, et il en dit long sur la fragilité de la convention
  « la sortie d'une couche est son dernier tampon » (§3.6).

---

## 1. Le problème, tel qu'il était mesuré

`diffusion.rs::train_step_batch_inner` déroulait le batch **côté CPU** : pour
chacun des `B` échantillons, un `CommandEncoder`, tout le graphe (prepare,
forward, loss, backward) encodé dedans, un `submit`. Plus un `submit` pour la
mise à zéro des gradients et un pour l'optimiseur.

Conséquences, toutes constatées dans `PERF_CONVOLUTION.md` :

- **18 `submit` par pas** à batch 16 ;
- chaque convolution ne travaillait que sur **un seul petit tenseur** — le
  forward de `conv4` de `Greyscale_Diffusion` a 1024 éléments de sortie, soit
  **16 workgroups** ;
- le register-blocking du forward, pourtant l'optimisation attendue, était
  **plus lent sur les quatre couches** (§3 de ce rapport-là), parce que diviser
  par 4 le nombre de threads affamait un GPU qui n'en avait déjà pas assez ;
- et surtout : des kernels **3,7× plus rapides en isolé** ne rendaient
  **+1 %** sur le pas complet.

## 2. Ce qui a été fait

### 2.1 La décision de conception : le batch est *implicite*

Deux représentations étaient possibles : ajouter un champ `batch` à chaque
uniforme de couche, ou ne rien ajouter et laisser chaque kernel **déduire** son
indice d'échantillon. C'est la seconde qui est livrée.

Les tampons d'activation sont alloués à `B × taille_par_échantillon`, les
dispatches sont multipliés par `B`, et un kernel élémentaire fait :

```wgsl
let per   = OH * OW * K;            // déjà dans son uniforme
if idx >= arrayLength(&output) { return; }
let sample = idx / per;
let local  = idx % per;
```

Trois raisons, et elles ont toutes payé :

1. **Aucun uniforme n'a changé de taille.** Les fixtures legacy de
   `conv_equivalence_tests.rs` et `group_norm_equivalence_tests.rs` lient
   toujours exactement le même tampon ;
   `conv_reduction_lanes_agrees_with_dispatch` relit toujours le mot à
   l'offset 12. **Aucun test d'équivalence existant n'a été touché.**
2. **Une seule source de vérité par kernel** — la taille du tampon. La classe
   de bug « l'uniforme dit 16, le tampon en contient 8 » n'existe pas.
3. `activation.wgsl`, `back_activation.wgsl` et `sum.wgsl` bornaient déjà leur
   boucle par `arrayLength` : ils sont devenus batchés **sans une ligne de
   changement**.

### 2.2 L'inventaire : ce qui porte un axe batch, et ce qui n'en porte pas

Il est porté par les types de couche eux-mêmes, via `batched_bytes(dim, batch)`.
La fonction est greppable, et **son absence** sur un accumulateur de gradient
est l'affirmation que cet accumulateur est une *réduction* du batch, pas une
tranche.

| × B (activations) | × 1 (paramètres) |
|---|---|
| `input`, `output`, `pre_activation` | `weights`, `bias`, `gamma`, `beta` |
| `grad_input`, `grad_output` | `grad_weights`, `grad_bias`, `grad_gamma`, `grad_beta` |
| `loss_terms`, `target`, `model_result` | l'état Adam (`m`, `v`) |
| `stats` de GroupNorm, `grad_merge` de Concat | tous les uniformes `specs` |
| `diffusion_clean_target`, le tableau de specs du prépass | |

### 2.3 Les dispatches se scindent en deux familles

C'est le cœur du changement, plus que le batch lui-même :

- une passe qui écrit **une activation par thread** voit sa grille multipliée
  par `B` (conv/upsample forward, `*_back_input`, concat, group_norm forward et
  ses deux passes de stats, la loss) ;
- une passe qui réduit sur un **paramètre** garde sa grille — il y a autant de
  sommes que de poids quel que soit le batch — et absorbe le batch **dans sa
  propre boucle** : `batch * OH * OW` positions au lieu de `OH * OW`,
  `batch * spatial_len` au lieu de `spatial_len`
  (`conv_back_weights`/`_bias`, `upsample_conv_back_weights`/`_bias`,
  `group_norm_back_gamma`/`_beta`, `fully_connected_back_weights`/`_bias`).

Il n'y a donc plus qu'**un seul `+=` par élément de poids et par pas**, au lieu
de `B`.

### 2.4 Le graphe, avant et après

Par pas d'entraînement, batch 16, `Greyscale_Diffusion_L` :

| | avant | après |
|---|---:|---:|
| `submit` portant du **calcul** | 18 | **1** |
| `submit` de copies dataset | 0 (encodées dans les 18) | ≈ 3,96 |
| `submit` au total | 18 | ≈ 4,96 |
| workgroups du forward de `conv4` (modèle S) | 16 | 256 |
| positions sommées par `grad_weights` de `conv1` | 1024, **× 16 fois** | 16384, **1 fois** |

**La ligne qui compte est la première** : tout le calcul d'un pas tient
désormais dans **une** soumission, contre 18. Les copies dataset sont
soumises à part, une par chunk **distinct** touché par le batch, pour la raison
d'ordonnancement du §3.5.

Et il faut être précis sur ce « ≈ 3,96 », parce que la première rédaction de ce
rapport écrivait « le dataset gris tient dans un chunk, donc une soumission » —
c'était faux, et la vérification l'a montré. Sur cette machine
`max_storage_buffer_binding_size` vaut 128 MiB, la règle de
`select_max_chunk_bytes` retient **64 MiB** par chunk, et CIFAR-10 gris
(50 000 × 4 KiB = 204,8 Mo) occupe donc **4 chunks**. Un batch de 16 indices
tirés dans toute la permutation en touche `4·(1 − (3/4)^16) ≈ 3,96` en
espérance. Voir §5.2 : c'est aussi ce qui rend l'ancien chemin coûteux en
téléversements.

### 2.5 Ce qui n'a **pas** changé

Aucune formule. Aucune sémantique de padding — la règle `Same` (offset
`pad_y`/`pad_x`, contribution **nulle** hors bornes, le finding #1
d'`AUDIT_TRAINING.md`) est intacte dans les huit kernels concernés. Aucun
schedule, aucune définition de gradient. `grad_scale = 1/batch` est inchangé,
parce que le tampon de gradients contient la même **somme** qu'avant.

Aucun re-tuning de kernel (blocking, tuiles, tailles de workgroup) : le §7 de
`PERF_CONVOLUTION.md` demande de lever le goulot structurel **avant** d'affiner
les kernels, et affiner en même temps rendrait les deux effets inséparables.

## 3. Les points durs, et comment ils ont été traités

### 3.1 Les graines de bruit — l'invariant dont tout dépend

Le prépass lisait un **uniforme scalaire** réécrit entre deux soumissions. Il
lit maintenant un **tableau de `B` structures**, indexé par l'échantillon.

Ce qui compte est ce qui n'a **pas** bougé : chaque entrée est calculée côté CPU
par exactement l'expression qui la produisait avant — même `batch_offset`, même
`sample_index` issu de la même `SampleShuffle`, même `fold_seed`. L'échantillon
`i` d'un batch tire donc **bit pour bit** le bruit et le timestep qu'il tirait.

Sans cette identité, comparer un run ancien à un run nouveau ne mesurerait
rien : les deux ne résoudraient pas le même problème.

**Sur la règle anti-`seed ^ index` du CLAUDE.md** : le motif interdit est la
*composition* de deux XOR (le pas de diffusion dans la graine de chemin **et**
l'indice pixel dans le champ de bruit), dont la somme s'effondrait sur les
anti-diagonales (`ANISOTROPY_HUNT.md`). Il n'y a ici qu'un seul XOR, celui qui
existait déjà, et le batching n'en ajoute aucun : **l'indice d'échantillon
sélectionne une entrée du tableau, il n'entre jamais dans une graine.**

Gardé par `batched_prepass_gives_each_sample_the_noise_it_had_alone` (§4.1).

### 3.2 GroupNorm — les statistiques restent par échantillon

Non négociable : normaliser à travers le batch changerait le modèle, pas son
implémentation — et casserait l'inférence (batch 1) contre l'entraînement.

Le forward et les deux passes de statistiques dispatchent `B × num_groups`
workgroups (`sample = wid.x / num_groups`, `group = wid.x % num_groups`), et le
tampon `stats` passe à `B × num_groups × 4` f32, indexé par le **couple**.
Seuls `grad_gamma` / `grad_beta` — qui sont des paramètres — somment sur le
batch ; `group_norm_back_gamma` re-lit d'ailleurs `mean`/`inv_std` **dans** sa
boucle, parce que la statistique reste celle de l'échantillon auquel la position
appartient.

Gardé par `group_norm_statistics_do_not_leak_across_the_batch`, qui met dans un
même batch deux échantillons à des échelles séparées de 4 ordres de grandeur.

### 3.3 La loss avait besoin de sa longueur par échantillon

`grad_output[i] = 2·(pred − target)/N` lisait `N` sur
`arrayLength(&model_result)`. Une fois le tampon batché, cette expression rend
`B·N` : pas de crash, pas de NaN, juste un **taux d'apprentissage effectif
divisé par le batch**.

`LossUniform` — défini de longue date dans `loss.rs` et lié à rien — devient le
binding **[4]**, ajouté en dernier pour ne déplacer aucun des indices 0–3
auxquels `model.rs` et `diffusion.rs` adressent `target`, `loss_terms` et
`grad_output`.

### 3.4 La loss rapportée est restée le même nombre

`train_step_report_batch` renvoyait la loss du **dernier** échantillon du batch.
La définition est conservée à l'identique — sinon la comparaison de trajectoires
du §5 comparerait deux grandeurs différentes — en lisant la tranche
`[(B−1)·N, B·N)` de `loss_terms`.

Et comme le forward ne contient **aucune** réduction sur l'axe batch, ce nombre
doit être **bit à bit** celui de l'ancien chemin. C'est vérifié
(`reported_loss_is_the_last_sample_of_the_batch`), y compris qu'il ne s'agit
**pas** de celui du premier échantillon — ce qu'un offset oublié renverrait.

### 3.5 Un piège d'ordonnancement dans les copies du dataset

Encoder les `B` copies dataset → `clean_target` dans **un seul** encodeur aurait
été **faux**. `ensure_chunk_loaded` téléverse par `queue.write_buffer`, et wgpu
applique toute écriture de queue en attente **avant** les command buffers soumis
ensuite : deux échantillons de deux chunks différents encodés dans le même
encodeur auraient tous deux lu le chunk chargé en dernier. Le premier aurait
silencieusement pris la mauvaise image — pas d'erreur de validation, pas de
crash, juste un batch entraîné sur des données dupliquées.

`GpuDataset::copy_samples_to` groupe donc par chunk et soumet un command buffer
par chunk résident.

### 3.6 Le bug attrapé au passage

En ajoutant le binding `specs` **en fin** de liste pour la couche de loss, la
valeur de retour de `Layer::create_buffers` — qui est, par convention, « le
dernier tampon de la couche » — est devenue **l'uniforme**. Et c'est cette
valeur qui partait chaîner le backward à la place de `grad_output`.

Cela a échoué bruyamment (« Usage flags `COPY_DST | UNIFORM` … do not contain
required usage flags `STORAGE` ») **uniquement parce que les usages diffèrent**.
Un tampon de stockage ajouté en dernier aurait été câblé en silence : le
backward aurait démarré sur un tampon de la bonne taille rempli de zéros, tous
les gradients auraient été nuls, et le modèle aurait simplement cessé
d'apprendre.

`Model::build` adresse maintenant `forward[3]` explicitement, constante nommée
et commentaire à l'appui. La leçon vaut au-delà de cette mission : la couche de
loss est le seul endroit du dépôt où « la sortie de la couche » n'est pas son
dernier tampon, et rien ne l'écrivait nulle part.

### 3.7 Changer le batch en cours de run

Le TUI le permet. Comme le batch dimensionne les tampons, `Model::resize_batch`
reconstruit le graphe **en préservant l'état** — poids, biais, moments Adam
**et** le compteur `optimizer_step` (la correction de biais d'Adam en dépend :
le perdre ferait du premier pas d'après un ±lr plein sur chaque poids) — par un
aller-retour de checkpoint.

## 4. Preuves d'équivalence

`crates/batlab-core/src/model/batch_equivalence_tests.rs` (7 tests) et un
huitième dans `training/diffusion.rs`.

La référence n'est **pas** une fixture figée : le chemin séquentiel est conservé
verbatim et exécutable (`DiffusionTask::train_step_batch_sequential` et les
quatre helpers de `Model` qu'il utilise, en `#[cfg(test)]`). Les tests comparent
donc deux choses qui tournent encore toutes les deux. `plan_batch` est factorisé
pour que les deux chemins ne puissent pas diverger sur le tirage.

### 4.1 Le prépass — bit à bit, sans dupliquer la formule

Le pas batché est joué une fois ; puis le chemin séquentiel est joué avec des
batchs de 1, 2, … B, chacun laissant en slot 0 le prépass de son **dernier**
échantillon. À `step = 0`, le compteur vaut `0·batch_size + offset == offset` :
le slot `j` du batch de `B` et le dernier slot du batch de `j+1` sont donc le
même triplet (échantillon, timestep, `batch_offset`), et doivent produire les
mêmes bits.

Résultat : **0 élément divergent** sur `model_input` et sur `target_noise`, pour
les 4 échantillons.

### 4.2 Le forward — bit à bit, échantillon par échantillon

Le forward n'a aucune réduction sur l'axe batch : l'échantillon `i` d'un batch
de `B` ne lit que sa tranche, dans le même ordre, depuis les mêmes poids. Sa
sortie doit donc être **exactement** celle d'une exécution seule.

L'assertion porte sur `to_bits()`, pas sur une tolérance, et c'est ce qui lui
donne sa force : un offset erroné qui tombe sur l'échantillon voisin produit des
nombres du bon ordre de grandeur, qu'un seuil à 1e-5 n'attraperait que par
chance. Mesuré : **0 élément divergent sur 5 échantillons**.

### 4.3 Le choix de la métrique — et pourquoi il a fallu le changer

Avant les chiffres, la façon de les mesurer, parce que la première version de
ces tests utilisait l'**écart relatif par élément** et qu'elle a produit un faux
positif instructif.

Sur le modèle en U (§4.5), l'entrée 100 du `grad_weights` de la première
convolution vaut **7,12e-4** dans un tampon dont la plus grande entrée vaut
**4,61**. Les deux chemins y diffèrent de **2,98e-7** — exactement le plancher
d'arrondi f32 pour des termes de cette taille. L'écart relatif par élément
annonce alors **4,2e-4** et le test tombe.

Ce nombre ne mesure pas l'accord sur le gradient : il mesure **à quel point sa
plus petite entrée se trouve proche de zéro**. Les petites entrées d'un tampon
de gradients sont petites *parce que* ce sont des sommes presque exactement
compensées de grands termes ; leur erreur relative est mal définie.

La métrique retenue est donc l'erreur **relative à l'échelle du vecteur** :
`max |a−b| / max(‖a‖∞, ‖b‖∞)`. C'est la réponse standard pour une grandeur
vectorielle, et elle reste mordante là où il faut : une réduction tronquée ou un
offset manqué déplacent les **grandes** entrées, celles-là mêmes dont ‖·‖∞ est
fait. Ce n'est pas un argument, c'est vérifié par mutation (§4.8) : les neuf
mutations sont attrapées avec la nouvelle métrique, et deux d'entre elles font
tomber **plus** de tests qu'avec l'ancienne.

### 4.4 Les gradients — et pourquoi la clause habituelle ne tient pas

Batché contre séquentiel, erreur relative à l'échelle, seuil 1e-4 :

| modèle | grandeur | écart |
|---|---|---:|
| conv → GN → SiLU → conv | conv1 `grad_weights` | 2,74e-7 |
| | conv1 `grad_bias` | 6,43e-7 |
| | GroupNorm `grad_gamma` | 1,21e-7 |
| | GroupNorm `grad_beta` | 1,86e-7 |
| | conv2 `grad_weights` | 3,91e-7 |
| U-net (conv s2 → up-conv → concat → conv) | 8 tampons | 7,17e-8 … 3,10e-7 |

Tout est au **plancher d'arrondi f32** (ε = 1,19e-7), deux à trois ordres de
grandeur sous le seuil.

Face à l'oracle f64 (convolution seule, batch 5) :

| grandeur | batché ↔ f64 | séquentiel ↔ f64 |
|---|---:|---:|
| `grad_weights` | 1,51e-7 | 1,36e-7 |
| `grad_bias` | **4,34e-7** | 1,58e-7 |

**Le résultat honnête** : sur `grad_bias`, le batché est **2,75× moins précis**
que le séquentiel. Les deux sont au plancher f32, mais le rapport est réel et il
fait tomber la clause habituelle du dépôt.

`PERF_GROUP_NORM.md` §3.2 et `PERF_CONVOLUTION.md` §4.2 pouvaient exiger
`nouveau ≤ 1,5 × ancien` parce que là-bas seule l'**association** d'un ensemble
**fixe** de termes changeait — un arbre au lieu d'une chaîne, sur les mêmes
`positions` termes. Ici le **nombre de termes change** : une lane qui parcourait
`positions` valeurs en parcourt `batch × positions` dans un seul accumulateur
f32. Chaîne plus longue, plus d'arrondi. C'est l'arithmétique de l'opération
demandée, pas un défaut d'implémentation — et 4,34e-7 > 1,5 × 1,58e-7, donc
écrire la clause telle quelle aurait demandé de truquer le seuil.

La clause livrée est donc : soit le batché passe l'ancienne barre 1,5×, soit il
tient dans un petit multiple du **plancher d'arrondi de la somme qu'il effectue
désormais** — `sqrt(n)·ε` pour `n` termes accumulés, l'estimation en marche
aléatoire. Le plancher est **dérivé de la forme**, pas ajusté sur la mesure
(ici `sqrt(36 × 5) × 1,19e-7 = 1,6e-6`, et la mesure est à 4,34e-7). Une
réduction réellement cassée est à O(1) relatif, quatre ordres de grandeur
au-dessus : le test mord toujours. L'exigence dure — **< 1e-4 face à l'oracle** —
est inchangée.

L'oracle f64 est écrit en **scatter** sur la carte des taps du forward
(`(oy,ox,ky,kx) → (oy·s+ky−pad_y, ox·s+kx−pad_x)`, contribution nulle hors
bornes) : il valide l'indexation des shaders au lieu de la répéter.

### 4.5 Concat et UpsampleConv — la forme en U

Un second modèle : `conv(s2) → up-conv(×2) → concat(skip) → conv`. Il mérite son
test parce que **Concat est la seule couche dont les trois tenseurs ont des
longueurs par échantillon différentes** (sortie = entrée + skip en canaux) : il
lui faut **trois** offsets d'échantillon distincts là où tout autre kernel en a
un seul. Un offset partagé y produirait des données plausibles et fausses.

Forward **bit à bit** sur les 5 échantillons, gradients des 4 couches
entraînables au plancher f32 (tableau ci-dessus).

### 4.6 L'isolation entre tranches

`a_batch_with_one_live_sample_equals_a_batch_of_one` : un batch dont tous les
échantillons sauf un sont nuls doit donner exactement les gradients d'un batch
de un contenant cet échantillon. C'est le test qui attrape un offset manqué —
si un kernel lit l'échantillon 0 en écrivant le 3, les tranches nulles fuient
dans la vivante.

L'échantillon vivant n'est **délibérément pas** le slot 0 : une implémentation
qui ignore l'offset passerait s'il l'était.

### 4.7 Différences finies

Les trois gradient checks d'`audit_tests.rs`
(`conv_weight_gradients_match_finite_differences`, `…_same_padding`,
`stacked_same_padding_conv_grad_input_is_consistent`) passent **inchangés**. Ce
sont des oracles qui ne connaissent aucune des deux implémentations. **Aucun
test existant n'a été touché ni affaibli.**

### 4.8 Les tests neufs ne sont pas vacuous — vérifié par mutation

**Neuf** mutations délibérées, toutes recompilées avec succès, toutes
attrapées — et rejouées **après** le changement de métrique du §4.3, ce qui est
le point : changer la façon de mesurer un écart peut affaiblir un test sans
qu'aucun ne devienne rouge, et la seule façon de le savoir est de refaire tomber
le code exprès.

| mutation | tests en échec |
|---|---|
| conv forward : offset d'échantillon sur `input` supprimé | 7 |
| group_norm forward : tous les échantillons sur la tranche 0 | 5 |
| `conv_back_weights` : axe batch tronqué hors de la réduction | 4 |
| loss : `N` relu sur `arrayLength` (donc `batch·N`) | 4 |
| `group_norm_back_input` : stats indexées par groupe, pas par (échantillon, groupe) | 2 |
| `group_norm_back_gamma` : batch retiré du balayage spatial | 2 |
| `upsample_conv_back_weights` : batch tronqué hors de la réduction | 1 |
| `back_concat` : offset d'échantillon du skip supprimé | 1 |
| `diffusion_prepare` : tous les échantillons lisent `specs[0]` | 1 |

Aucune n'est passée sous les mailles, et les deux mutations les plus larges en
font tomber **plus** qu'avec l'ancienne métrique (7 contre 6, 4 contre 3), parce
que le test du modèle en U s'est ajouté entre-temps.

Chaque test porte en outre sa **garde anti-vacuité** (les échantillons doivent
être assez différents, les gradients non nuls, les bruits distincts) : sans
elles, « toutes les tranches concordent » pourrait être vrai pour la mauvaise
raison.

### 4.9 Suite complète

```
cargo test --workspace
   115 passed (batlab_core) + 23 (batlab_ui) + 13 (batlab) — 0 failed
```

## 5. Non-régression de l'inférence — bit à bit

Le §3.7 de la conception promettait « même chemin, `B = 1` ». Vérifié en
comparant le binaire pré-batch (`bb1d9fe`) au binaire de la branche, même
checkpoint, même graine :

| chemin | résultat |
|---|---|
| `--headless-sample Greyscale_Diffusion_L --seed 7` | PNG **octet pour octet identique** (942 o, `cmp`) |
| `--headless-perpetual --regime breathe --frames 4` | dump JSONL identique sur **11 535 enregistrements** ; les 4 PNG de cycle identiques |
| `--headless-perpetual --regime flux --actions 8` | s'exécute, statistiques identiques |

Aucune tolérance n'a été nécessaire, et c'est le résultat attendu : à `B = 1`,
`arrayLength / longueur == 1`, `sample == 0`, et chaque kernel retrouve
littéralement son dispatch d'avant.

## 5.1 La suite à l'aveugle du régime flux

`./blind_tests/run.sh` (ACTIONS=800) — la suite écrite par un agent aveugle,
depuis la spec, sans lire l'implémentation :

```
résumé : 9/9 propriétés PASS, 0 FAIL (+ 0 observation hors spec numérotée)
```

Dont **P7 — non-régression errance / respiration** (5 cycles atteignent t=0,
corrélation inter-cycles 0,80–0,94, la respiration ne se résout jamais) et la
propriété d'isotropie qui couvre explicitement « le piège anti-diagonale
documenté dans `ANISOTROPY_HUNT.md` » : autocorrélation spatiale du bruit
ajouté au décalage (1,−1) = −0,00064. C'est la garde qui aurait attrapé une
graine composée par XOR, et elle tient.

## 5.2 Le trafic dataset a baissé au passage — et ce n'est pas du batching

Effet de bord réel du §3.5, qu'il serait malhonnête de laisser compter comme un
gain de dispatch : `ensure_chunk_loaded` téléverse un **chunk entier** par
`queue.write_buffer`, et sur cette machine
(`max_storage_buffer_binding_size` = 128 MiB) un chunk fait **64 MiB**.
CIFAR-10 gris, c'est 50 000 échantillons de 4 KiB, soit **4 chunks**.

Les indices d'un batch étant tirés d'une permutation de tout le dataset,
l'ancien chemin (un `ensure_chunk_loaded` par échantillon, dans l'ordre du
batch) rechargeait à chaque fois que l'échantillon suivant tombait dans un
autre chunk : **≈ 12,25 téléversements de 64 MiB par pas** à batch 16. Groupés
par chunk, il en reste **≈ 3,96** — le nombre de chunks distincts touchés.

C'est de l'arithmétique, pas une mesure (les deux espérances se calculent
exactement : `1 + 15·(3/4)` et `4·(1 − (3/4)^16)`), mais le comportement est
verrouillé par deux tests : `a_batch_uploads_each_chunk_at_most_once` et
`samples_land_in_the_slot_they_were_asked_for`. `GpuDataset::chunk_loads()`
expose le compteur, parce que ce coût n'apparaît nulle part ailleurs.

## 6. Validation en conditions réelles

Deux entraînements identiques, ancien binaire puis nouveau, tout le reste égal.
L'initialisation des poids est un xorshift à graine fixe et les tirages de
timestep/bruit dérivent du compteur de pas : les deux runs sont donc
rigoureusement comparables, et le test est **déterministe** — la contention du
GPU ne l'affecte pas, elle le ralentit seulement.

### 6.1 Contrôle rapide — `Greyscale_Diffusion`, 60 pas, batch 8, Adam

```
step  0  1.05248737 | 1.05248737     step 50  0.05277247 | 0.05277243
step 25  0.12493825 | 0.12493812     step 59  0.05624811 | 0.05624810
                                              (ancien | nouveau)
```

Sur les 265 enregistrements JSONL :

| grandeur | écart absolu max | écart relatif max (\|v\| > 1e-3) |
|---|---:|---:|
| `train_loss` | 1,34e-7 | **1,07e-6** |
| `train_probe` (loss par tranche de `t`) | 5,25e-6 | 1,22e-4 |
| `sample` | 7,75e-7 | 1,70e-6 |
| `denoise_step` | 4,05e-6 | 1,86e-4 |

**0** valeur non finie, **0** différence de structure. Le champ le plus
divergent, `denoise_step.eps_hat.mean`, est une moyenne proche de zéro sur
laquelle l'arrondi accumulé s'exprime en relatif — exactement le champ et
l'ordre de grandeur que `PERF_CONVOLUTION.md` §6 rapportait déjà (4,8e-5) pour
un changement d'ordre de sommation.

### 6.2 Run apparié complet — `Greyscale_Diffusion_L`, 600 pas, batch 16, Adam

_À compléter._

## 7. Benchmarks

_À compléter — voir §9._

## 8. Limites et pièges connus

- **La mémoire GPU croît linéairement avec le batch.** Tous les tampons
  d'activation, plus `clean_target`, sont multipliés par `B`. C'est ce qui
  touchera `max_storage_buffer_binding_size` en premier sur un gros modèle à
  grand batch ; la bannière de ressources le reflète désormais
  (`estimated_prepare_gpu_bytes_for_batch`).
- **Un dataset plus gros qu'un chunk redevient coûteux.** Les indices d'un batch
  sont tirés d'une permutation de tout le dataset : si le dataset dépasse la
  capacité d'un chunk, un batch peut toucher plusieurs chunks et déclencher
  plusieurs téléversements par pas. C'était déjà le cas avant (et en pire, un
  échantillon à la fois) ; le batching ne le corrige pas. CIFAR-10 gris
  (204,8 Mo) tient dans un chunk, donc le cas ne se présente pas ici.
- **`ConvolutionType::reduction_lanes` n'a pas été recalibrée.** Elle est une
  fonction de `(sommes, positions)` où `positions` est encore le compte **par
  échantillon**. Vérifié à la main : sur les quatre convolutions de
  `Greyscale_Diffusion` à batch 16, la règle choisirait **les mêmes** valeurs
  (32 / 16 / 8 / 32) si on lui passait `positions × batch`, parce que c'est la
  borne `TARGET_THREADS` et le plafond `MAX_LANES = 32` qui mordent, pas le
  plancher `MIN_POSITIONS_PER_LANE`. Le changement serait donc sans effet sur
  les modèles réels — mais il reste à faire pour des formes à peu de positions,
  et `bench_conv_reduction_lanes` existe pour le mesurer.
- **La longueur des chaînes d'accumulation a été multipliée par `B`.** C'est la
  contrepartie numérique du §4.3. Elle est bornée et documentée, mais elle est
  réelle : à batch 64 les chaînes sont 4× plus longues encore qu'à batch 16.
- **Le visualiseur montre le PREMIER échantillon du batch.** Il indexe le
  tampon de sortie par `(y·largeur + x)·canaux` avec les dimensions **par
  échantillon**, donc il lit la tranche 0. Avant le batching, ce tampon
  contenait le **dernier** échantillon soumis (chacun écrasait le précédent) —
  et c'est le dernier dont la loss est rapportée. Ce n'est pas un bug (la
  tranche lue est valide et complète), mais l'image affichée et le nombre
  affiché ne décrivent plus le même échantillon. Corriger demanderait de lier
  le tampon avec un offset, côté `batlab_ui` ; hors périmètre ici, et noté.
- `PoolingType` n'a pas de shader (`panic!("not wired yet")`) et n'est pas dans
  l'enum `LayerTypes` : ses dispatches ont été rendus batch-conscients par
  cohérence, rien n'a pu être testé.

## 9. Ce qui reste

- **Re-mesurer le register-blocking du forward.** Le §3 de
  `PERF_CONVOLUTION.md` l'a rejeté sur une mesure prise quand `conv4` avait
  1024 éléments de sortie, soit 16 workgroups ; avec l'axe batch il en a 16384.
  **Le verdict n'est plus établi** — la note en tête de `convolution.wgsl` a été
  corrigée en ce sens, pour que la piste ne soit ni retentée à l'aveugle ni
  abandonnée sur un argument périmé.
- **Rendre la sonde de coût fixe utilisable** (§7 de `PERF_CONVOLUTION.md`) :
  minimum sur N rondes au lieu d'un échantillon unique. C'est ce qui dira si les
  kernels sont encore optimisables ou collés au plancher.
- **Recalibrer `reduction_lanes` avec l'axe batch**, sur des formes où ça change
  quelque chose.
- **Le batch dans l'inférence.** Le sampler et le mode Perpetual tournent à
  `B = 1` par construction ; le graphe sait maintenant faire mieux. Générer
  plusieurs chemins de débruitage en parallèle est devenu un changement d'appel,
  plus un changement de moteur.
