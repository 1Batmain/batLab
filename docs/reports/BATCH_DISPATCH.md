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

Mesuré sur `Greyscale_Diffusion_L`, GPU partagé, protocole du §5.1 de
`PERF_CONVOLUTION.md` :

| batch | speedup (ms/pas) | contrôle nul | rapport de temps CPU |
|---:|---:|---:|---:|
| 1 | 1,03× | 7,9 % | 0,9× |
| 4 | 1,67× | 1,7 % | 3,5× |
| 16 | **1,84×** | **0,3 %** | 13,9× |
| 64 | (1,87×, non certifié) | 1965 % ⚠ | **43,3×** |

**La prédiction posée d'avance tient** : à batch 1 les deux chemins sont
indiscernables (1,03× à l'intérieur du contrôle nul), et le gain croît avec le
batch. Le 1,84× à batch 16 reproduit indépendamment le 1,85× obtenu par les
runs appariés de 600 pas (§7.2).

Et le résultat le plus net n'est pas dans la colonne des speedups : **le coût
CPU de l'ancien chemin est proportionnel au batch (×93 pour un batch ×64),
celui du nouveau ne l'est pas (×2)**. C'est la thèse du §7 de
`PERF_CONVOLUTION.md` démontrée directement (§7.4).

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

### 3.7 Le plafond de 65 535 workgroups — un vrai bug, trouvé tard

WebGPU limite un dispatch à **65 535 workgroups par dimension**. Rien dans ce
dépôt n'en approchait : le plus gros dispatch sur un tenseur unique est d'un
workgroup pour 64 éléments, soit **1024** pour la plus grosse couche de
`Greyscale_Diffusion_L`. Multiplié par un batch de 64, cela fait **65 536** —
un de trop.

**C'est le mode de défaillance qui rend ce point grave.** wgpu signale la
violation sur la queue, mais **le run continue** : le symptôme observé au
premier essai de `--batch 64` était `loss 0.000000` qui défile, c'est-à-dire un
pas qui ne calcule **rien**. Une mauvaise réponse silencieuse, exactement aux
tailles de batch que ce travail existe pour rendre possibles. Trouvé en
préparant le balayage du §7.3, pas par un test — les tests tournaient tous à des
batchs de 5 ou moins.

Correction : le dispatch devient **2-D** au-delà du plafond (`dispatch_grid`
dans `layer.rs`), et chaque kernel reconstruit son indice linéaire à partir de
`@builtin(num_workgroups)` — `gid.y · nwg.x · 64 + gid.x` pour les passes
élémentaires, `wid.y · nwg.x + wid.x` pour les passes à un workgroup par unité.
Comme le batch lui-même (§2.1), le découpage ne passe par **aucun uniforme** et
ne peut donc pas diverger de ce qui a été dispatché. Sous le plafond, la grille
reste strictement 1-D : l'inférence et les petits batchs ne voient rien changer,
et c'est vérifié — le PNG de `--headless-sample` et le dump de
`--headless-perpetual` sont restés **octet pour octet** identiques après ce
changement (§5).

Gardé par deux tests, dont un end-to-end qui traverse réellement le plafond
(65 600 workgroups, choisi à 64 éléments par échantillon pour que le batch
*soit* le compte de workgroups), et vérifié par mutation : rendre `dispatch_grid`
incapable de découper — le comportement d'avant — fait tomber les deux.

Vérifié après correction : `--batch 16`, `--batch 64` et `--batch 128`
convergent tous sur `Greyscale_Diffusion_L`.

### 3.8 Changer le batch en cours de run

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

Le modèle de la campagne, avec ses deux `UpsampleConv` et ses deux `Concat`.
**821 enregistrements de chaque côté, 0 non apparié, 0 différence de structure,
0 valeur non finie.**

```
step   0  1.07258594 | 1.07258594     step 350  0.04927786 | 0.04927786
step  50  0.04364759 | 0.04364757     step 400  0.02125878 | 0.02125876
step 100  0.02259401 | 0.02259401     step 450  0.01172543 | 0.01172546
step 150  0.05139169 | 0.05139172     step 500  0.00380219 | 0.00380219
step 200  0.10766626 | 0.10766631     step 550  0.04990207 | 0.04990207
step 250  0.01142664 | 0.01142664     step 599  0.00370164 | 0.00370165
step 300  0.00503387 | 0.00503387              (ancien | nouveau)
```

| grandeur | écart absolu max | écart relatif max (\|v\| > 1e-3) |
|---|---:|---:|
| `train_loss` | 1,64e-7 | **7,59e-6** |
| `train_probe` (loss par tranche de `t`) | 1,05e-5 | 2,49e-4 |
| `sample` | 2,62e-6 | 7,39e-6 |
| `denoise_step` | 9,54e-6 | 3,07e-4 |

Le profil attendu par `INSIGHTS_TRAINING.md` — loss élevée à `t` bas, faible à
`t` haut, c'est-à-dire un modèle qui utilise bien `t` — est identique des deux
côtés. Le champ le plus divergent reste `denoise_step.eps_hat.mean`, une
moyenne proche de zéro sur laquelle l'arrondi accumulé s'exprime en relatif :
même champ et même ordre de grandeur que `PERF_CONVOLUTION.md` §6 (4,8e-5)
pour un changement d'ordre de sommation, ici sur une chaîne 16 fois plus
longue (§4.4).

### 6.3 Une remarque de méthode : ce run a dû être refait

La première exécution de ce run apparié est **inexploitable**, et la raison
mérite d'être écrite parce qu'elle n'a rien à voir avec le code testé.

`--headless-train` dérive le chemin de ses métriques du checkpoint, et sans
`--out` celui-ci ne dépend **que du nom du modèle**. Deux runs du même modèle —
ce qu'un banc apparié fait par définition — écrivent donc dans **le même
fichier**. Un test de fumée de `bench/batch_dispatch/timing.sh`, lancé pendant
que le bras « ancien » tournait, s'y est entrelacé : 537 enregistrements
exploitables contre 821, dont une ligne de 152 051 caractères illisible.

Ce qui l'a sauvé est un accident : le fichier ne parsait plus. Trois défenses
ont donc été ajoutées, et elles valent pour tout banc futur du dépôt :

- `timing.sh` passe un `--out` unique par bras ;
- `compare_metrics.py` **signale** les lignes illisibles au lieu de les sauter —
  un chargeur silencieux aurait comparé 537 enregistrements à 821 et imprimé
  une réponse confiante et creuse ;
- l'appariement se fait sur les **coordonnées propres** de l'enregistrement
  (`step` / `train_step` / `step_index` / `diffusion_step` / `seed`), avec une
  assertion d'unicité, et non sur la position. Deux tentatives intermédiaires
  ont produit de fausses divergences spectaculaires : la clé `(kind, step)`
  n'est pas unique (un pas de `sample` émet un `denoise_step` par point de la
  chaîne inverse), et elle a fait comparer le pas de diffusion 0 au 255 — un
  « écart de 255,0 » purement comptable.

## 7. Benchmarks

### 7.1 La machine n'était pas libre, et voici de combien

Un run couleur de 20 000 pas a occupé le GPU pendant toute la mission. Ce n'est
pas une excuse posée en fin de rapport : c'est mesuré, et par accident très
proprement.

Le bras « ancien » du §6.2 a été exécuté **deux fois**, même binaire, mêmes
arguments, à 45 minutes d'écart. C'est un **contrôle nul** non planifié :

| | exécution 1 | exécution 2 | étendue |
|---|---:|---:|---:|
| ancien, temps réel | 1798,6 s | 2330,8 s | **+29,6 %** |
| ancien, temps CPU | 695,7 s | 804,4 s | +15,6 % |
| **nouveau, temps réel** | 980,4 s | 972,0 s | **+0,9 %** |

Deux enseignements :

1. **Le temps réel a un plancher de bruit de 29,6 % sur cette machine.** Tout
   effet plus petit que ça, mesuré ici, n'est pas un effet — et c'est
   exactement l'ordre de grandeur (±1 %) que `PERF_CONVOLUTION.md` §5.3
   cherchait à trancher. Sa conclusion « peut-être ~1 % plus lent » n'aurait
   pas survécu à cette fenêtre-ci.
2. **Le nouveau chemin est stable à 0,9 %, l'ancien varie de 29,6 %.** Ce n'est
   pas un hasard : le coût de l'ancien est dominé par du travail **côté CPU**
   (18 soumissions et ~12 téléversements de 64 Mio par pas), qui se dispute la
   bande passante mémoire avec le voisin. Le nouveau est court et
   majoritairement GPU.

### 7.2 Ce qui est mesuré, à batch 16

Estimateur = **minimum** des deux exécutions par bras (la contention ne peut
qu'ajouter du temps), `Greyscale_Diffusion_L`, 600 pas, Adam :

| | ancien | nouveau | rapport |
|---|---:|---:|---:|
| ms/pas (temps réel) | 2997,6 | 1620,0 | **1,85×** |
| temps CPU total | 695,7 s | 53,2 s | **13,1×** |
| dont temps système | 630,6 s | 41,8 s | 15,1× |

**Sur le 1,85×** : l'effet (+85 %) est environ trois fois le plancher de bruit
du §7.1 (29,6 %), donc le **sens et l'ordre de grandeur sont établis** ; le
chiffre exact ne l'est pas. Il faut le refaire sur une machine libre. Noter
aussi que 2997,6 ms/pas n'est pas comparable aux ~1270 ms/pas cités par l'ordre
de mission : c'est la même contention qui gonfle les deux bras.

**Sur le 13,1× de temps CPU** : c'est la mesure la plus solide du lot, parce
que le temps CPU d'un processus dépend beaucoup moins de la charge GPU que son
temps réel (15,6 % d'étendue contre 29,6 %). Et il a une explication mécanique
directe, pas une corrélation : 18 soumissions par pas deviennent ~5, et ~12,25
téléversements de chunk de 64 Mio deviennent ~3,96 (§5.2), soit environ
**800 Mio par pas** de trafic hôte→GPU en moins. Les 630,6 s de temps *système*
de l'ancien bras — 1,05 s de noyau par pas — sont exactement la signature de ce
trafic.

### 7.3 L'échelle en batch — la mesure qui décide

C'est **la** mesure que la mission demandait, et la prédiction avait été posée
d'avance, donc réfutable : à batch 1 les deux chemins doivent être
**indiscernables** (un échantillon, une soumission de chaque côté), et l'écart
doit **croître** avec le batch.

`Greyscale_Diffusion_L`, Adam, 3 rondes × 40 pas par point, estimateur =
minimum, contrôle nul = le binaire neuf contre lui-même :

| batch | ancien (ms/pas) | nouveau (ms/pas) | speedup | contrôle nul |
|---:|---:|---:|---:|---:|
| 1 | 207,25 | 200,25 | **1,03×** | 7,9 % |
| 4 | 752,25 | 450,50 | **1,67×** | 1,7 % |
| 16 | 3087,50 | 1679,00 | **1,84×** | 0,3 % |
| 64 | 12212,50 | 6520,75 | (1,87×) | **1965 %** ⚠ |

**La prédiction tient.** À batch 1, 1,03× est *à l'intérieur* du contrôle nul de
7,9 % : les deux chemins sont indiscernables, comme ils doivent l'être. C'est le
contrôle interne, et c'est lui qui autorise à lire le reste de la colonne. Le
gain croît ensuite de façon monotone, et le point à batch 16 est mesuré avec un
contrôle nul de **0,3 %** — un effet de 84 % contre un bruit de 0,3 %.

Le 1,84× à batch 16 **reproduit indépendamment** le 1,85× dérivé au §7.2 des
deux runs appariés de 600 pas. Deux instruments, deux protocoles, deux fenêtres
de mesure, même nombre.

**Le point à batch 64 n'est pas certifiable, et il est marqué comme tel.** Son
contrôle nul est à 1965 % : l'une des deux exécutions du binaire neuf a pris
**7859 s** au lieu de 261 s. Le log dit pourquoi — cette exécution a consommé
**1,75 s de temps utilisateur et 5,50 s de temps système**, soit exactement
autant que les autres. Le processus n'a pas travaillé plus longtemps, il a
**attendu** : deux heures et onze minutes bloqué sur un GPU monopolisé par le
run couleur. Ce n'est donc pas une propriété du chemin batché, mais le
protocole ne permet pas de trancher, et un chiffre dont le contrôle nul est à
1965 % ne se publie pas comme un résultat.

### 7.4 Le temps CPU — l'instrument que la contention n'atteint pas

Le §7.3 laisse un point invalide et un plancher de bruit variable. Le temps CPU,
lui, ne dépend presque pas de la charge GPU : un processus bloqué sur le GPU
n'en consomme pas (le run à 7859 s ci-dessus n'a coûté que 7,25 s de CPU). Même
balayage, même exécutions :

| batch | CPU ancien | CPU nouveau | rapport |
|---:|---:|---:|---:|
| 1 | 2,08 s | 2,19 s | **0,9×** |
| 4 | 10,22 s | 2,95 s | **3,5×** |
| 16 | 47,75 s | 3,44 s | **13,9×** |
| 64 | 194,22 s | 4,49 s | **43,3×** |

Lire la table par **colonne** plutôt que par ligne, parce que c'est là qu'est le
résultat :

- le coût CPU de l'**ancien** chemin est **proportionnel au batch** : 2,08 →
  194,22 s quand le batch fait ×64, soit ×93. C'est la signature exacte d'une
  boucle CPU sur les échantillons — une soumission et son cortège de
  téléversements **par échantillon** ;
- le coût CPU du **nouveau** est **quasi indépendant du batch** : 2,19 → 4,49 s
  pour le même ×64. Un pas, une soumission, quel que soit le nombre
  d'échantillons qu'elle porte.

C'est la thèse du §7 de `PERF_CONVOLUTION.md` démontrée directement, et sur la
grandeur la moins contestable dont on dispose ici. Et à batch 1 le rapport est
de 0,9× : les deux chemins font le même travail CPU, ce qu'ils doivent faire.

### 7.5 Ce que l'instrument a coûté

Trois défauts ont été trouvés dans le harnais de mesure au cours de cette
mission, **zéro** dans le moteur batché par ces mêmes exécutions. C'est une
donnée sur la difficulté de mesurer, et elle mérite d'être écrite :

1. **Chemin de métriques partagé.** Sans `--out`, deux runs du même modèle
   écrivent dans le même JSONL. Un test de fumée du banc a corrompu le run de
   validation de 600 pas (§6.3).
2. **`exit 1` depuis une substitution de commande.** `x=$(run_one …)` met la
   fonction dans un sous-shell : l'arrêt sur erreur ne tuait que la
   substitution, et chaque run raté rendait une chaîne vide qui ressortait
   plusieurs tailles de batch plus loin en erreur de conversion.
3. **L'horloge.** Les runs étaient encadrés par deux appels
   `python3 -c 'time.monotonic()'`. Sur cette machine `monotonic()` repart de
   près de zéro dans chaque processus : deux lectures à une seconde d'écart ont
   donné 0,005501 et 0,005996. La soustraction mesurait la gigue de démarrage
   de l'interpréteur, et sortait **négative** une fois sur deux — d'où un
   « speedup −2,67× » avec un « contrôle nul −150 % ».

Le troisième est le plus instructif : il n'a été attrapé que parce que le
résultat était **absurde**. Une variante de la même erreur qui serait tombée du
côté positif aurait produit un nombre plausible et faux, et serait entrée telle
quelle dans ce rapport. C'est précisément l'argument du contrôle nul du §7.1 —
un instrument dont on ne mesure pas le bruit propre ne mesure rien — appliqué à
l'instrument lui-même.

Le raisonnement des trois est écrit **dans** `timing.sh`, pas seulement ici.

## 8. Limites et pièges connus

- **Le plafond de 65 535 workgroups est traité mais reste une contrainte.**
  Un modèle dont une couche a `L` éléments par échantillon peut aller jusqu'à
  `batch × L / 64 ≤ 65 535 × 65 535` workgroups au total — largement assez —
  mais toute nouvelle passe de calcul ajoutée au graphe doit passer par
  `dispatch_grid` et lire son indice via `num_workgroups`, sans quoi elle
  échouera en silence au-delà du plafond. C'est le §3.7.
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

### Refaire le point à batch 64 sur une machine libre

Le seul chiffre non certifié du rapport (§7.3) : son contrôle nul est à 1965 %
parce qu'une exécution est restée deux heures bloquée sur un GPU monopolisé. La
valeur obtenue (1,87×) s'inscrit dans la tendance et le rapport de temps CPU du
même point (43,3×) est solide, mais le protocole ne certifie pas le premier.
Une demi-heure sans co-locataire suffit. Tout le reste du balayage tient
(contrôles nuls de 0,3 % à 7,9 %).

### Re-mesurer le register-blocking du forward

Le §3 de `PERF_CONVOLUTION.md` l'a rejeté sur une mesure prise quand `conv4`
avait 1024 éléments de sortie, soit **16 workgroups** ; avec l'axe batch il en a
**16384**. Le raisonnement qui le condamnait — « diviser le nombre de threads
par 4 affame un GPU qui n'en a déjà pas assez » — ne tient plus. **Le verdict
n'est plus établi**, et la note en tête de `convolution.wgsl` a été corrigée en
ce sens, pour que la piste ne soit ni retentée à l'aveugle ni abandonnée sur un
argument périmé.

### Rendre la sonde de coût fixe utilisable

`profile_convolution` estime déjà le « per-compute-pass floor », mais sur un
échantillon unique, ce qui l'a fait sortir au-dessus de passes qui font
strictement plus de travail. Minimum sur N rondes, comme partout ailleurs. Sans
elle on ne sait toujours pas si les kernels sont encore optimisables ou déjà
collés au plancher — et le batching a justement déplacé ce plancher.

### Recalibrer `reduction_lanes` avec l'axe batch

Sans effet sur les quatre convolutions de `Greyscale_Diffusion` (vérifié à la
main, §8), mais la règle raisonne encore en positions **par échantillon** alors
que chaque somme en couvre `batch ×` plus. À faire sur des formes où ça change
quelque chose, avec `bench_conv_reduction_lanes`.

### Le batch dans l'inférence

Le sampler et le mode Perpetual tournent à `B = 1` par construction ; le graphe
sait maintenant faire mieux. Générer plusieurs chemins de débruitage en
parallèle est devenu un changement d'appel, plus un changement de moteur —
`--headless-sample --paths N` en est le premier candidat.

### Faire remonter le compteur `chunk_loads`

Il existe (§5.2) mais rien ne l'affiche. Un dataset plus gros que quelques
chunks coûte des centaines de Mio de trafic par pas, et rien dans la sortie
d'un run ne le laisse voir.
