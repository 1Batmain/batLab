# Optimisation de la convolution — rapport de mission

Branche `perf-convolution`. Fait suite à `PERF_GROUP_NORM.md`, qui avait
déplacé le goulot du pas d'entraînement de `group_norm` vers la convolution.

**Résumé** : les quatre convolutions de `Greyscale_Diffusion` passent de
**2,57 ms à 0,69 ms par échantillon** (couches isolées, forward + backward,
mesure appariée) — **3,7×**. Les maths sont inchangées : sur 600 pas
d'entraînement réel, la trajectoire de loss est identique à toutes les
décimales affichées, l'écart relatif max sur `train_loss` est de **5,6e-7** et
l'écart absolu max sur les 821 enregistrements JSONL de **2,1e-6**.

Deux résultats méritent d'être lus avant le reste :

- Le coût était à **90 % dans le backward**, pas dans le forward. C'est le
  profilage qui l'a établi, contre l'intuition de départ.
- L'approche attendue par la mission — **tuiles / réutilisation par thread** —
  a été implémentée, mesurée, et **rejetée** : elle est plus lente sur les
  quatre couches. Le §3.3 explique pourquoi, chiffres à l'appui.

---

## 1. Ancienne approche, et ce que le profilage a montré

Le premier profilage donnait un forward de `conv1` à 0,43 ms, très au-dessus
des autres. C'était un artefact : sur Metal, le **premier dispatch d'un
pipeline paie sa compilation**, et le banc l'imputait à la première couche
mesurée. Après échauffement de tous les pipelines avant toute mesure, le total
passe de 3,79 à 2,35 ms/échantillon et la répartition change complètement.

Profil réel (ms/échantillon, médiane de 3 rondes, tous pipelines échauffés) :

| couche | forward | back_input | back_weights | back_bias | total |
|---|---:|---:|---:|---:|---:|
| conv1 32×32×3 → 16, s1 | 0,060 | 0,101 | 0,283 | 0,245 | 0,68 |
| conv2 32×32×16 → 32, s2 | 0,045 | 0,390 | 0,098 | 0,076 | 0,61 |
| conv3 16×16×32 → 32, s1 | 0,096 | 0,285 | 0,144 | 0,078 | 0,60 |
| conv4 32×32×32 → 1, s1 | 0,042 | 0,054 | 0,278 | 0,101 | 0,47 |
| **total** | **0,24** | **0,83** | **0,80** | **0,50** | **2,35** |
| **part** | 10 % | 35 % | 34 % | 21 % | |

**Le forward pesait 10 %.** Les trois passes du backward se partageaient le
reste à parts presque égales. Chacune avait un défaut distinct :

- **`grad_weights` et `grad_bias`** sommaient sur les `OH*OW` positions de
  sortie avec **un seul thread par somme**. La passe `grad_bias` de `conv1`
  faisait donc tourner **16 threads** (un par noyau) sur 1024 positions
  chacune, et sa passe `grad_weights` 432 threads — 7 workgroups — sur 1024
  positions. Sous-parallélisme massif, sur un GPU qui veut des milliers de
  threads.
- **`grad_input`** avait, lui, un thread par élément d'entrée (parallélisme
  correct), mais sa boucle `k` était **à l'extérieur** de `ky`/`kx` alors que
  toute la géométrie des taps — carte inverse du forward, test de divisibilité
  par le stride, contrôles de bornes — ne dépend **que** de `(ky, kx)`. Elle
  était donc recalculée `K` fois : 288 divisions et modulos entiers par thread
  sur les couches 3×3 à 32 noyaux, là où 9 suffisent.

## 2. Nouvelle approche

### 2.1 `grad_weights` / `grad_bias` — réductions par workgroup

Un workgroup de 64 threads porte désormais `slots` sommes indépendantes de
`lanes` threads coopérants (`lanes * slots == 64`). Chaque lane parcourt l'axe
des positions par pas de `lanes`, puis les lanes d'un même slot sont réduites
en arbre.

Les slots consécutifs couvrent des `kz` (resp. `k`) consécutifs — l'axe le plus
rapide de `fwd_input` et de `grad_output`. Les threads d'un même groupe lisent
donc des adresses contiguës et partagent la même valeur `grad_output`.

`lanes == 1` **dégénère exactement en un thread par somme**, c'est-à-dire en
l'ancien schéma : c'est le bon choix pour les couches qui ont déjà des milliers
de sommes indépendantes.

### 2.2 Le nombre de lanes vit dans l'uniforme, et il est calibré

Première version : `lanes` était dérivé de la forme par une règle **dupliquée**
entre le WGSL et le Rust. Deux problèmes. D'abord la duplication elle-même.
Ensuite et surtout, la règle était fausse : elle visait 16 384 threads et
découpait donc `conv3` en 2 lanes alors que l'optimum mesuré est 8 — et sur
`conv2`/`conv3` la première règle (16 lanes) était carrément **plus lente que
l'ancien code**.

`lanes` est passé dans **le mot de l'uniforme jusqu'ici nommé `_padding`** :
taille et offsets inchangés, donc les fixtures legacy lient toujours
exactement le même buffer. La règle n'est plus dupliquée, et devient réglable —
d'où `bench_conv_reduction_lanes`, qui balaie chaque nombre de lanes légal,
couche par couche, en ne réécrivant que ce mot entre deux points de mesure
(même couche, mêmes buffers, même pipeline).

Balayage sur `grad_weights` (ms, minimum de 7 rondes) :

| lanes | conv1 (432 sommes, 1024 pos.) | conv2 (4608, 256) | conv3 (9216, 256) | conv4 (288, 1024) |
|---:|---:|---:|---:|---:|
| 1 | 0,4348 | 0,0901 | 0,1275 | 0,4319 |
| 2 | 0,1723 | 0,0727 | 0,1056 | 0,1786 |
| 4 | 0,0981 | **0,0576** | 0,1043 | 0,0889 |
| 8 | 0,0504 | 0,0611 | **0,0965** | 0,0486 |
| 16 | 0,0292 | 0,0584 | 0,1126 | 0,0296 |
| 32 | **0,0291** | — | — | **0,0220** |
| 64 | 0,0290 | — | — | 0,0297 |

Deux enseignements que seule la mesure donnait :

- **64 lanes n'est jamais optimal.** À 64, un workgroup ne porte qu'un seul
  slot : les threads consécutifs ne parcourent plus des `kz` consécutifs et la
  coalescence que la disposition en slots existe pour obtenir disparaît. Sur
  `conv4` cela coûte 35 % (0,0220 à 32 lanes contre 0,0297 à 64). Le plafond
  est donc fixé à 32.
- Le gain est **plat au voisinage de l'optimum** mais s'effondre à faible
  `lanes` quand il y a peu de sommes (`conv1`, `conv4` : ×15 entre 1 et 32).

Règle retenue (`ConvolutionType::reduction_lanes`) : le moins de lanes qui
mette encore ~65 536 threads en vol, sans jamais descendre sous 16 positions
par lane ni dépasser 32. Elle choisit 32 / 16 / 8 / 32, soit **à moins de 1,4 %
de l'optimum par couche** sur les quatre convolutions.

### 2.3 `grad_input` — la boucle `k` pivote vers l'intérieur

`ky`/`kx` passent à l'extérieur, `k` à l'intérieur. La géométrie des taps est
calculée 9 fois au lieu de 288, et il reste un produit scalaire compact.
`grad_output` est alors parcouru de façon **contiguë** en `k` (axe le plus
rapide du layout HWK), et les threads voisins — `iz` voisins — lisent des poids
voisins : les deux accès coalescent.

Contrepartie assumée : sur `conv4`, `K == 1`, donc la boucle interne ne porte
qu'une itération — le pivot n'enlève aucun calcul et ajoute la mise en place de
boucle. C'est une régression de 0,007 ms, contre 0,385 ms gagnées sur
`conv2` + `conv3`.

### 2.4 Forward — ce qui est gardé, et ce qui est rejeté

Voir §3.3. Résumé : le register-blocking a été mesuré puis retiré ; seul le
hoisting d'adresses est conservé, et il est **bit à bit identique** à l'ancien
kernel.

### 2.5 Ce qui n'a pas changé

Aucune formule. Aucune sémantique de padding. En particulier la règle `Same`
— offset `pad_y`/`pad_x`, contribution **nulle** hors bornes (et non un clamp)
— est préservée à l'identique dans les quatre kernels ; c'est le fix
durement acquis du finding #1 de `AUDIT_TRAINING.md`, et le §4.5 montre qu'une
mutation qui le casse est bien détectée. L'accumulation `+=` de `grad_weights`
et `grad_bias` entre échantillons d'un batch est préservée (seule la lane 0
écrit). Les bind groups, les indices de bindings et la taille de l'uniforme
sont inchangés.

Sur `grad_weights` / `grad_bias` / `grad_input`, **seul l'ordre de sommation
flottant diffère**. Le forward, lui, est inchangé au bit près.

## 3. Le register-blocking du forward : implémenté, mesuré, rejeté

La mission proposait des tuiles en mémoire partagée / de la réutilisation par
thread. La variante la plus adaptée à ces formes a été écrite : chaque thread
produit un segment de `OUT_X_BLOCK` pixels de sortie adjacents en x pour un
noyau, ce qui charge chaque poids **une fois pour 4 sorties** au lieu d'une
fois par sortie, et fait se recouvrir les champs réceptifs (à stride 1, un
noyau de largeur 3 sur 4 sorties adjacentes couvre 6 colonnes, pas 12).

Mesures appariées (ms, minimum de 9 rondes entrelacées, même process) :

| couche | naïf | bloc ×4 | bloc ×2 |
|---|---:|---:|---:|
| conv1 | 0,0288 | 0,0280 | 0,0281 |
| conv2 | 0,0358 | 0,0445 | 0,0450 |
| conv3 | 0,0603 | 0,0753 | 0,0760 |
| conv4 | 0,0386 | 0,0814 | **0,0813 (0,47×)** |

**Ces dispatches sont limités par le parallélisme, pas par la bande passante.**
Le batch est déroulé échantillon par échantillon côté CPU (`model.rs` : un
encoder et un `submit` par échantillon), donc une convolution travaille ici sur
un seul petit tenseur. Le forward de `conv4` n'a que **1024 éléments de sortie
au total**, soit 16 workgroups. Diviser le nombre de threads par 4 pour
économiser de la bande passante affame le GPU bien plus vite que le trafic
épargné ne le rembourse.

Le raisonnement et ces chiffres sont conservés en commentaire dans
`convolution.wgsl`, pour que la piste ne soit pas retentée à l'aveugle.

**Ce qui est gardé** est la part gratuite : sortir les offsets de padding et
les adresses de base des boucles internes, et rejeter une ligne de noyau
entièrement dans le padding **une fois par ligne** au lieu d'une fois par
colonne. Mêmes taps, même ordre, mêmes sauts.

## 4. Preuves d'équivalence

Fichier : `bat_building/src/model/conv_equivalence_tests.rs`.

Les shaders d'avant l'optimisation sont conservés **verbatim** comme fixtures
(`shader/legacy/convolution_naive.wgsl`, `back_convolution_naive.wgsl`,
extraits de `d73ed32`). Chaque test construit un vrai modèle avec les nouveaux
shaders, l'exécute, puis redispatche les kernels legacy sur **exactement le
même bind group** — mêmes buffers, mêmes poids, mêmes entrées — avec les
**comptes de workgroups de l'époque** (un thread par élément), et compare.

Neuf formes sont couvertes : les quatre convolutions du modèle, plus padding
`Valid`, un stride qui ne divise pas l'entrée, un noyau 1×1, un noyau 5×5, et
une entrée plus petite qu'un workgroup.

### 4.1 Forward — égalité **bit à bit**

L'optimisation du forward ne réassocie que de l'arithmétique **entière**
d'indices : chaque tap est visité dans le même ordre, à partir du même biais.
L'accord doit donc être exact, et le test l'exige — il compare les `to_bits()`,
pas une tolérance. Un seuil à 1e-5 aurait laissé passer un changement d'ordre
des taps ; l'égalité binaire non. Mesuré : **0 élément divergent sur les 9
formes**.

### 4.2 Backward — ancien vs nouveau, arbitré par un oracle f64

| grandeur | écart relatif max nouveau ↔ naïf | seuil du test |
|---|---:|---:|
| `grad_input` | < 1e-4 | 1e-4 |
| `grad_weights` | < 1e-4 | 1e-4 |
| `grad_bias` | < 1e-4 | 1e-4 |

L'oracle f64 (`reference_conv`) est écrit en **scatter** sur la carte des taps
du forward — `(oy,ox,ky,kx) → (oy·s+ky−pad_y, ox·s+kx−pad_x)`, la définition du
finding #1. Or `conv_back_input` **inverse** cette carte (`oy = (iy+pad_y−ky)/s`
avec un test de divisibilité). L'oracle **valide donc l'inversion au lieu de la
répéter** : c'est une vérification indépendante, pas une reformulation.

Le test exige, comme pour `group_norm` :

- écart nouveau ↔ oracle < 1e-4 ;
- **`erreur(nouveau) ≤ 1,5 × erreur(naïf)`** face à l'oracle — la réduction en
  arbre n'est jamais *moins* précise que la somme séquentielle qu'elle remplace.

### 4.3 Différences finies

Les trois gradient checks de `audit_tests.rs`
(`conv_weight_gradients_match_finite_differences`, `…_same_padding`,
`stacked_same_padding_conv_grad_input_is_consistent`) passent inchangés. Ce
sont des oracles qui ne connaissent **ni l'une ni l'autre** implémentation.
**Aucun test existant n'a été touché ni affaibli.**

### 4.4 Le compte de lanes ne peut pas dériver du dispatch

`conv_reduction_lanes_agrees_with_dispatch` relit le mot de l'uniforme à
l'offset 12 et vérifie, sur chaque forme, qu'il est une puissance de deux ≤ 32,
qu'il laisse ≥ 16 positions par lane, et que le dispatch couvre **exactement**
tous les poids et tous les biais. Une dérive entre les deux laisserait
silencieusement des gradients jamais écrits.

### 4.5 Les tests ne sont pas vacuous — vérifié par mutation

Cinq mutations délibérées, toutes recompilées avec succès (une sixième,
rejetée, ne compilait pas — son « échec » aurait été un faux positif et n'est
pas comptée) :

| mutation | tests qui échouent |
|---|---|
| forward : offset `pad_x` du mode `Same` supprimé | 5 |
| backward : `grad_weights` lit en **clamp** au lieu de zero-padding | 3 |
| backward : la boucle `k` de `grad_input` s'arrête un cran trop tôt | 2 |
| backward : la réduction en arbre saute le partenaire de la lane 0 | 1 |
| backward : `grad_weights` biaisé de +0,1 % | 1 |

Les deux premières touchent la **sémantique du padding `Same`** — le finding #1
d'`AUDIT_TRAINING.md` — et sont attrapées à la fois par les tests
d'équivalence et par les différences finies d'`audit_tests`.

Les deux dernières illustrent la complémentarité des oracles : un biais de
1e-3 et une réduction partiellement fausse passent **sous** la tolérance des
différences finies (5e-2, limitée par la précision de la loss lue en un seul
f32) mais très au-dessus de celle de la comparaison ancien/nouveau (1e-4).
Aucun des deux oracles seul ne suffirait.

### 4.6 Suite complète

```
cargo test --workspace
   56 passed (bat_building) + 6 passed (main) — 0 failed
```

## 5. Benchmarks

### 5.1 Méthodologie — le GPU est partagé

Un autre agent entraînait un modèle plus gros (`--headless-train
Greyscale_Diffusion_L`) sur le **même GPU** pendant toute la mission, aux côtés
d'Ableton Live et de Spotlight. Les mesures absolues prises à des moments
différents ne sont donc **pas** comparables — c'est vérifié, pas supposé : le
même kernel `back_bias` de `conv1`, inchangé entre deux exécutions du banc, est
passé de 0,031 à 0,134 ms.

Protocole retenu :

- **Appariement et entrelacement** : chaque passe est mesurée
  ancien/nouveau/ancien/nouveau… dans **un seul process**, sur **les mêmes
  buffers**, 200 dispatches par point, 9 rondes.
- **Estimateur = le minimum**, pas la médiane. La contention ne peut
  qu'**ajouter** du temps ; la ronde la plus rapide est donc la meilleure
  estimation du coût propre du kernel, et la prendre des deux côtés garde la
  comparaison équitable. La médiane ne suffit pas ici : une rafale du voisin
  dure couramment plus qu'une ronde entière, et elle déforme **inégalement**
  les deux côtés, parce que les kernels optimisés sont assez petits (~0,03 ms)
  pour qu'une rafale les triple, tandis que les naïfs (~0,35 ms) bougent à
  peine. Le banc imprime l'étendue min–max des deux côtés pour que cette
  déformation reste visible.
- **Échauffement de tous les pipelines avant toute mesure** (cf. §1).

### 5.2 Couches isolées

TABLE_ISOLATED

### 5.3 Pas d'entraînement complet

TABLE_ENDTOEND

## 6. Validation en conditions réelles

Deux runs de **600 pas** (batch 16, lr 1e-3, `cifar10_grey.batraw`), ancienne
puis nouvelle implémentation, tout le reste identique. L'initialisation des
poids est un xorshift à graine fixe et les tirages de timestep/bruit dérivent
du compteur de pas : les deux runs sont donc rigoureusement comparables. La
loss finale du run ancien (0,074075) reproduit d'ailleurs à l'identique celle
de `PERF_GROUP_NORM.md`.

### Trajectoire de loss — identique à toutes les décimales affichées

```
step   0  1.12194014 | 1.12194014     step 300  0.10513614 | 0.10513613
step  25  0.73080879 | 0.73080868     step 375  0.43441018 | 0.43441010
step  50  0.50129014 | 0.50129002     step 450  0.07953402 | 0.07953400
step 100  0.25294417 | 0.25294426     step 525  0.08750758 | 0.08750759
step 200  0.27752286 | 0.27752286     step 599  0.07407522 | 0.07407522
                                                 (ancien | nouveau)
```

Écart relatif max sur `train_loss`, sur les 25 points : **5,6e-7**.

### Loss par tranche de `t` (`train_probe`)

C'est le diagnostic qui compte pour un modèle de diffusion (cf.
`INSIGHTS_TRAINING.md` : une loss batch qui décroît ne suffit pas).

| pas | tranche | ancien | nouveau | écart rel. |
|---|---|---:|---:|---:|
| 0 | t 0–64 | 1,13167381 | 1,13167381 | 0 |
| 0 | t 192–256 | 1,18465424 | 1,18465424 | 0 |
| 300 | t 0–64 | 0,71501017 | 0,71501017 | 0 |
| 300 | t 192–256 | 0,10651554 | 0,10651554 | 0 |
| 599 | t 0–64 | 0,68143141 | 0,68143141 | 0 |
| 599 | t 192–256 | 0,07151438 | 0,07151438 | 0 |

**Écart relatif max sur les 25 sondes × 4 tranches : 1,2e-7.** Le profil attendu
(loss élevée à t bas, faible à t haut — le modèle utilise bien `t`) est
identique dans les deux runs.

### Toutes métriques confondues

Sur les 821 enregistrements JSONL (`train_loss`, `train_probe`, `sample`,
`denoise_step`) :

- écart **absolu** max, tous champs : **2,1e-6** (`train_probe.eps_hat.max`) ;
- écart **relatif** max sur les champs de magnitude > 1e-3 : **4,8e-5**
  (`denoise_step.eps_hat.mean`, divergence d'arrondi accumulée sur 600 pas) ;
- **0** valeur non finie, **0** différence de structure : mêmes nombres et
  mêmes types d'enregistrements.

## 7. Ce qui reste

CEQUIRESTE

---

## Fichiers modifiés

| fichier | changement |
|---|---|
| `bat_building/src/model/shader/convolution.wgsl` | hoisting d'adresses ; note sur le blocking rejeté |
| `bat_building/src/model/shader/back_convolution.wgsl` | réductions par workgroup, pivot de la boucle `k` |
| `bat_building/src/model/layer_types/convolution.rs` | `reduction_lanes`, dispatches, mot d'uniforme |
| `bat_building/src/model/mod.rs` | + module de tests |
| `bat_building/src/model/conv_equivalence_tests.rs` | **nouveau** — tests + 3 bancs |
| `bat_building/src/model/shader/legacy/*_naive.wgsl` | **nouveau** — fixtures de test |
