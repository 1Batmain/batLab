# Optimisation de la convolution — rapport de mission

Branche `perf-convolution`. Fait suite à `PERF_GROUP_NORM.md`, qui avait
déplacé le goulot du pas d'entraînement de `group_norm` vers la convolution.

**Résumé** : les quatre convolutions de `Greyscale_Diffusion` passent de
**2,30 ms à 0,62 ms par échantillon** (couches isolées, forward + backward,
mesure appariée, minimum sur 36 rondes) — **3,7×**. Les maths sont inchangées :
sur 600 pas d'entraînement réel, la trajectoire de loss est identique à toutes
les décimales affichées, l'écart relatif max sur `train_loss` est de **5,6e-7**
et l'écart absolu max sur les 821 enregistrements JSONL de **2,1e-6**.

**Mais le pas d'entraînement complet, lui, ne bouge pas** — la seule mesure
bout-en-bout exploitable dont on dispose le donne même **~1 % plus lent**
(§5.3). C'est le résultat le plus important du rapport, et il est **inexpliqué**.
Le §5.4 pose l'hypothèse, dit franchement pourquoi elle n'est pas démontrée, et
donne le test qui trancherait. Contrairement à `group_norm`, où le banc isolé
était un *minorant* du gain réel, la convolution ne transmet pas son gain au pas.

Trois autres résultats méritent d'être lus avant le reste :

- Le coût était à **90 % dans le backward**, pas dans le forward. C'est le
  profilage qui l'a établi, contre l'intuition de départ.
- L'approche attendue par la mission — **tuiles / réutilisation par thread** —
  a été implémentée, mesurée, et **rejetée** : elle est plus lente sur les
  quatre couches. Le §3.3 explique pourquoi, chiffres à l'appui.
- Un **retuning** du nombre de lanes, motivé par le résultat bout-en-bout, a été
  écrit puis **retiré faute de mesure reproductible** (§5.4).

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

Quatre balayages complets de `bench_convolution_isolated` (9 rondes chacun,
200 dispatches par point), tous sur la configuration livrée. Le tableau donne le
**minimum par cellule sur les 36 rondes** : la contention ne pouvant qu'ajouter
du temps, c'est la meilleure estimation du coût propre de chaque kernel.

| couche | passe | naïf (ms) | nouveau (ms) | speedup |
|---|---|---:|---:|---:|
| conv1 32×32×3 → 16, s1 | `forward` | 0,0198 | 0,0281 | 0,70× |
| conv1 32×32×3 → 16, s1 | `back_input` | 0,0572 | 0,0271 | **2,11×** |
| conv1 32×32×3 → 16, s1 | `back_weights` | 0,3820 | 0,0313 | **12,20×** |
| conv1 32×32×3 → 16, s1 | `back_bias` | 0,2460 | 0,0302 | **8,15×** |
| conv2 32×32×16 → 32, s2 | `forward` | 0,0289 | 0,0289 | 1,00× |
| conv2 32×32×16 → 32, s2 | `back_input` | 0,4456 | 0,0533 | **8,36×** |
| conv2 32×32×16 → 32, s2 | `back_weights` | 0,0719 | 0,0604 | 1,19× |
| conv2 32×32×16 → 32, s2 | `back_bias` | 0,0725 | 0,0218 | **3,33×** |
| conv3 16×16×32 → 32, s1 | `forward` | 0,0449 | 0,0456 | 0,98× |
| conv3 16×16×32 → 32, s1 | `back_input` | 0,3510 | 0,0579 | **6,06×** |
| conv3 16×16×32 → 32, s1 | `back_weights` | 0,0821 | 0,0983 | 0,84× |
| conv3 16×16×32 → 32, s1 | `back_bias` | 0,0721 | 0,0217 | **3,32×** |
| conv4 32×32×32 → 1, s1 | `forward` | 0,0362 | 0,0368 | 0,98× |
| conv4 32×32×32 → 1, s1 | `back_input` | 0,0231 | 0,0292 | 0,79× |
| conv4 32×32×32 → 1, s1 | `back_weights` | 0,2671 | 0,0235 | **11,37×** |
| conv4 32×32×32 → 1, s1 | `back_bias` | 0,0973 | 0,0222 | **4,38×** |
| **total / échantillon** | | **2,2977** | **0,6163** | **3,73×** |

**Comment lire les lignes proches de 1,00×.** Le forward est **bit à bit
identique** au legacy (§4.1) : il ne change que de l'arithmétique entière
d'indices. Ses quatre lignes ne mesurent donc **que le bruit de l'instrument**,
et le fait qu'elles s'étalent de 0,70× à 1,00× donne l'ordre de grandeur de ce
bruit sur les passes courtes. Le gain réel du forward, mesuré au moment du
commit `b5465b7` par un banc dédié moins bruité, va de 1,00× à 1,23× selon la
couche. Même lecture pour `conv3 back_weights` (0,84×) et `conv4 back_input`
(0,79×) : sous le bruit, et cohérent avec les contreparties déjà assumées aux
§2.3 et §2.2.

Ce qui **est** au-dessus du bruit, d'un facteur 3 à 12, ce sont exactement les
trois passes que la mission visait : `back_weights`, `back_bias` et le
`back_input` de conv2/conv3.

Totaux des quatre balayages pris séparément : **3,88× / 3,72× / 3,57× / 3,36×**.
Le côté « nouveau » est très stable (0,6909 à 0,7011 ms, ±0,7 %) ; c'est le côté
naïf qui bouge (2,36 à 2,71 ms, ±7 %), ses kernels étant assez longs pour
absorber les rafales du voisin. Fourchette honnête : **3,4× à 3,9×**.

### 5.3 Pas d'entraînement complet

`Greyscale_Diffusion`, batch 16, `--headless-train`, 100 pas, binaires release.
Huit exécutions **entrelacées ABBA dans une même fenêtre calme**, plus un
**contrôle nul** : la même paire rejouée avec **deux copies du même binaire**,
pour établir le plancher de bruit de l'instrument.

| série | ancien (s) | nouveau (s) | écart |
|---|---:|---:|---:|
| moyenne des 4 | 54,86 | 55,48 | **+1,13 %** |
| minimum des 4 | 54,56 | 55,37 | **+1,48 %** |
| **contrôle nul** (binaire identique) | 54,49 | 54,44 | −0,09 % |

Soit **549 → 555 ms/pas**. Le démarrage est négligeable : la pente entre 100 et
300 pas donne 634–652 ms/pas là où 100 pas coûtent 62,5 s — moins d'une seconde
de mise en route, identique des deux côtés.

Le contrôle nul à 0,09 % dit que l'instrument résout **bien mieux** que l'effet
mesuré, et les étendues des deux bras ne se recouvrent pas (54,56–55,06 contre
55,37–55,57). **La conclusion est donc que l'ensemble des optimisations de la
convolution ne fait gagner aucun temps sur le pas d'entraînement, et lui en
coûte peut-être ~1 %.**

Deux réserves, à charge :

- c'est **une seule fenêtre de mesure**, celle de la session précédente ;
- ma tentative de la reproduire a **échoué faute de machine disponible**, et a
  même donné l'écart de **signe opposé** (−0,72 %) sur la seule paire de rondes
  dont le contrôle nul était acceptable (0,53 %). Voir §5.5.

Une troisième série, `lanes = 1` (réduction neutralisée, cf. §5.4) contre
l'ancien, donne −1,06 % au minimum et −2,09 % en moyenne — mais sa fenêtre
dérivait de 54,9 à 60,6 s en cours de série : **indicative, pas concluante**.

### 5.4 Pourquoi le gain isolé ne se transmet pas — hypothèse, et ce qui manque

Le banc isolé économise 2,30 − 0,62 = **1,68 ms/échantillon**, soit **27 ms** sur
un pas de batch 16. Le pas en coûte ~550 : on devrait voir **−5 %**. On mesure
**+1 %**. Il manque donc un facteur — et le précédent de `group_norm` interdit de
l'attribuer à la prudence du banc, puisque là-bas le banc isolé était un
*minorant* du gain réel (§4.3 de `PERF_GROUP_NORM.md`).

Ce qui est **établi** :

- **Le nombre de dispatches ne change pas.** Les lanes ne changent que le nombre
  de workgroups *par* dispatch. Pour `grad_weights`, sur les quatre couches et
  par échantillon : **228** workgroups à `lanes = 1`, **2664** avec la règle
  livrée. Le surcoût éventuel n'est donc **pas** un surcoût de lancement.
- Le graphe d'un échantillon est une **chaîne sérielle de dispatches minuscules
  séparés par des barrières** : `diffusion.rs:250-277` crée un `CommandEncoder`
  par échantillon, y encode tout le graphe et le **soumet** — **18 `submit` par
  pas** à batch 16. Après optimisation, une passe conv coûte 0,02 à 0,10 ms.

L'**hypothèse** : les kernels optimisés sont descendus au niveau du **coût fixe
d'une passe de calcul**, et le banc isolé le masque parce qu'il enchaîne 200
copies de la *même* passe dans un seul encoder — ce qui les laisse se recouvrir —
alors que le vrai graphe le paie une fois par dispatch, entre deux barrières.

Ce qui **manque pour la démontrer** : une mesure fiable de ce coût fixe.
`profile_convolution` l'estime déjà (« per-compute-pass floor »), mais **en un
seul échantillon non répliqué**. Il est sorti à **0,0854 ms** en fenêtre calme et
à **0,6235 ms** cette nuit — dans les deux cas **au-dessus de passes qui font
strictement plus de travail** (0,0217 ms pour `conv3 back_bias`). Un plancher
supérieur à ce qu'il est censé minorer n'est pas un plancher : la sonde est
inutilisable en l'état. **L'hypothèse reste une hypothèse.**

#### Le retuning des lanes : implémenté, non livré

Si l'hypothèse est vraie, découper les sommes en 2664 workgroups au lieu de 228
coûte, dans le vrai graphe, plus que ça ne rapporte. Une règle plus conservatrice
(`MAX_LANES = 8`, soit 666 workgroups) a donc été écrite et mesurée — mais dans
une fenêtre où le voisin dérivait de **92 à 140 s** par exécution, ce qui rend
ses chiffres inexploitables : l'ordre des deux bras **s'inverse** selon qu'on
prend le minimum ou la moyenne.

Elle **n'est pas livrée**. La règle calibrée en isolé (§2.2) est conservée, pour
trois raisons : le seul argument contre elle n'est pas reproductible ; le gain
isolé qu'elle apporte est solide et d'un facteur 12 ; et il redeviendra
pleinement exploitable dès que le goulot structurel du §7 sera levé. Changer un
réglage sur une mesure qu'on ne sait pas reproduire serait précisément l'erreur
que la méthode de ce dépôt cherche à éviter.

### 5.5 La re-mesure : pourquoi elle n'a rien donné

Trois entraînements concurrents d'autres agents (le modèle `L` à 10 000 pas, et
un comparatif SGD/Adam qui lançait ses propres bancs) occupaient le GPU. Effets
mesurés, pas supposés :

- le banc apparié isolé donne **0,65× sur `conv2 forward`** — un kernel pourtant
  **bit à bit identique** au legacy. Un kernel identique ne peut pas être 35 %
  plus lent : l'instrument ne lisait plus que du bruit ;
- le total isolé tombe à **1,93×** (contre 3,4–3,9× en fenêtre calme), et même en
  cumulant 45 rondes de minimum les petits kernels restent 3 à 4 fois au-dessus
  de leur coût connu (`conv4 forward` : 0,143 ms contre 0,037) ;
- en bout-en-bout, le **contrôle nul** — deux copies du même binaire — s'est
  écarté de **4,06 %** sur une paire de rondes, soit quatre fois l'effet
  recherché ; les deux paires se contredisent en signe.

Décision (architecte) : ne pas re-bencher, conserver les mesures de la fenêtre
calme — traçables commit par commit — et documenter la contention. Le **test de
correction du §6 a bien été rejoué**, lui : il est déterministe, donc insensible
à la contention, seulement plus lent.

## 6. Validation en conditions réelles

Deux runs de **600 pas** (batch 16, lr 1e-3, `cifar10_grey.batraw`), ancienne
puis nouvelle implémentation, tout le reste identique. L'initialisation des
poids est un xorshift à graine fixe et les tirages de timestep/bruit dérivent
du compteur de pas : les deux runs sont donc rigoureusement comparables. La
loss finale du run ancien (0,074075) reproduit d'ailleurs à l'identique celle
de `PERF_GROUP_NORM.md`.

Cette section est la seule du rapport à avoir été **entièrement rejouée dans la
session suivante**, sur des binaires reconstruits des deux côtés : le test est
déterministe, donc la contention du GPU ne l'affecte pas (elle le ralentit
seulement). Les deux exécutions indépendantes donnent **exactement les mêmes
chiffres**, jusqu'au dernier de ceux cités ci-dessous — ce qui vérifie du même
coup que la configuration livrée est bien celle qui a été validée.

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

### Le goulot est structurel : un encoder et un `submit` par échantillon

C'est la découverte la plus utile de la mission, et elle déborde largement la
convolution. `diffusion.rs:250-277` déroule le batch **côté CPU** : pour chaque
échantillon un `CommandEncoder` est créé, tout le graphe (forward, loss,
backward) y est encodé, et il est **soumis**. À batch 16 cela fait **18 `submit`
par pas** et, surtout, chaque convolution ne travaille jamais que sur **un seul
petit tenseur** — le forward de `conv4` a 1024 éléments de sortie, soit **16
workgroups**.

Trois faits déjà constatés dans ce rapport en découlent :

- le **register-blocking du forward est plus lent** (§3), parce que diviser le
  nombre de threads par 4 affame un GPU qui n'en avait déjà pas assez ;
- les réductions `grad_weights` / `grad_bias` n'avaient à l'origine que **5 à
  144 workgroups** à faire tourner (§1) — d'où leurs 55 % du coût ;
- et le gain isolé ne se transmet pas au pas complet (§5.4).

**Porter la dimension batch dans le dispatch** — un seul encoder par pas, les 16
échantillons portés par une dimension du workgroup — attaquerait les trois d'un
coup : les kernels deviendraient assez gros pour être limités par le *débit* et
non par le parallélisme, le coût fixe par passe serait amorti sur 16 fois plus de
travail, et le blocking comme les découpages en lanes agressifs redeviendraient
rentables. C'est de loin le plus gros gain restant. Hors périmètre de cette
mission, et à traiter **avant** de continuer à affiner des kernels.

### Avant cela : rendre la sonde de coût fixe utilisable

Une ligne à changer dans `profile_convolution` — minimum sur N rondes au lieu
d'un échantillon unique, l'estimateur déjà utilisé partout ailleurs (§5.1). Sans
elle, le §5.4 reste une hypothèse, et on ne sait pas si les kernels actuels sont
encore optimisables ou déjà collés au plancher.

### Refaire la mesure bout-en-bout sur une machine libre

Le §5.3 repose sur une fenêtre unique et le §5.4 sur une hypothèse non
démontrée : la question « les optimisations de la convolution font-elles gagner
ou perdre ~1 % au pas ? » est **ouverte**. Elle se tranche en une demi-heure sur
un GPU sans co-locataire, avec le protocole du §5.1 et les quatre bras déjà
outillés (`base` / lanes livrées / `MAX_LANES = 8` / `lanes = 1`).

### Ce qui reste dans la convolution elle-même

La passe la plus chère après optimisation est `conv3 back_weights` (0,0983 ms,
1,12× seulement) : 9216 sommes de 256 positions, un cas où le découpage en lanes
n'apporte presque rien parce que le parallélisme était déjà là. Toutes les autres
sont entre 0,022 et 0,060 ms, c'est-à-dire dans la zone où le coût fixe par passe
domine probablement. Les optimiser davantage avant d'avoir levé le goulot
structurel serait prématuré.

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
