# Les ressources GPU, rendues visibles — rapport de mission

Branche `gpu-resources`. Répond à trois questions posées ensemble : **combien
pèse le modèle sur le GPU**, **est-ce que ça tiendrait sur une autre machine**,
et **charge-t-on tout d'un coup ou partie par partie ?**

Les deux premières sont maintenant calculables sans GPU, à partir d'un
`config_file` seul. La troisième est **mesurée**, pas affirmée — et sa réponse
n'est pas celle qu'on attendait des deux côtés à la fois.

Tout ce qui suit est reproductible :

```bash
cargo run --release -p batlab -- --resources <modèle> [--batch N] [--dataset <path>] [--measure]
```

et la même page est à l'écran, `[r]` depuis le menu d'actions d'un modèle.

---

## Résumé — les trois réponses

1. **Le modèle est résident, en un bloc, une fois pour toutes.** Poids,
   gradients, moments d'Adam, activations : tout est alloué au `build()` et rien
   ne remonte ensuite. Mesuré : un `build` de `Greyscale_Diffusion_L` téléverse
   **1,8 MiB**, c'est-à-dire *exactement* ses poids, et un pas d'entraînement
   n'en téléverse plus un octet. Rien n'est paginé couche par couche.
2. **Le dataset, lui, est streamé — et bien plus cher que prévu.** Un pas
   d'entraînement monte **195,3 MiB** (CIFAR-10 gris) ou **585,9 MiB**
   (CIFAR-10 couleur) hôte→GPU. C'est le **dataset entier, à chaque pas**, et
   c'est **indépendant du batch** (§5.2). Le plus gros gain restant du dépôt est
   là, pas dans les kernels.
3. **L'inférence paie un aller-retour CPU↔GPU par pas de débruitage** — 256 par
   image, mesurés. Les octets sont dérisoires (6 MiB par image au total), le
   coût est la **latence** : 256 `map_async` + `poll(wait)` bloquants, un par
   pas, parce que le latent vit côté CPU entre deux appels au modèle.

Et le chiffre de capacité, sur ce Mac (Apple M5 Pro, Metal, mémoire unifiée) :

| modèle | inférence | train b16 | b32 | b64 | plafond de batch |
|---|---:|---:|---:|---:|---:|
| `Greyscale_Diffusion_L` (464 k param.) | **4,5 MiB** | 100,0 MiB | 193,0 MiB | 378,9 MiB | **512** |
| `Color_Diffusion_XL` (1,19 M param.) | **8,9 MiB** | 167,5 MiB | 316,9 MiB | 615,5 MiB | **341** |

(Résident, dataset exclu ; l'entraînement ajoute **128 MiB** de chunk de
dataset, soit 228,0 / 321,0 / 506,9 MiB en tout pour le gris et 295,5 / 444,9 /
743,5 MiB pour le couleur.)

**L'inférence tient dans 9 MiB.** C'est la ligne qui compte pour l'objectif
wasm : le graphe d'inférence de `Color_Diffusion_XL` tient sans effort dans les
limites que WebGPU accorde par défaut à une page, sur la carte du visiteur.

---

## 1. L'appareil, et ce qui borne vraiment

```
DEVICE  Apple M5 Pro (this machine)
  Metal · IntegratedGpu · unified memory (host and GPU share it)
  max_buffer_size                                  256.0 MiB
  max_storage_buffer_binding_size                  128.0 MiB
  adapter would allow, per buffer                   28.08 GiB
  max_compute_workgroups/dimension                     65535
  max_storage_buffers/stage                                8
  memory budget — not exposed
```

Quatre remarques, parce que chacune change la lecture de tout le reste.

**Le moteur tourne volontairement au contrat WebGPU, sur une machine qui
offrirait 100× plus.** L'adaptateur annonce 28,08 GiB par tampon (à peu près la
mémoire unifiée de la machine) ; le *device* en accorde 256 MiB, parce que
`request_device(&Default::default())` demande les limites par défaut. Ce sont
ces dernières que le graphe rencontre. Ce n'est pas une limitation subie : c'est
exactement ce qu'un navigateur accordera à la page, donc développer dessous,
c'est développer sur la cible.

Les deux nombres s'appellent tous les deux `max_buffer_size` dans ce dépôt, et
la page les affiche côte à côte pour cette raison — les confondre a coûté un
bug, §7.

**Il n'y a pas de budget mémoire.** wgpu n'en expose pas, et WebGPU non plus —
une page ne peut pas demander combien de VRAM elle a le droit de prendre. La
page le dit au lieu d'inventer un pourcentage de remplissage. C'est aussi
pourquoi le verdict par défaut porte sur les **limites par binding** et non sur
un total : `--vram N` est la façon de poser la question du total, et c'est une
donnée du lecteur, pas une mesure.

**C'est le plafond par binding qui mord en premier, pas la mémoire.** Le
plafond de batch de 512 (gris) et 341 (couleur) ne vient pas d'une mémoire
pleine : il vient d'**un seul tampon** d'activation qui atteint
`max_storage_buffer_binding_size` = 128 MiB. `Greyscale_Diffusion_L` a une
activation de 256 KiB par échantillon (128 MiB / 256 KiB = 512) ;
`Color_Diffusion_XL` en a une de 384 KiB (128 MiB / 384 KiB = 341). Un total
confortable et un build refusé est un mode de défaillance réel, et c'est celui
qu'un navigateur produit.

**La mémoire est unifiée ici**, et ça flatte la machine : un « téléversement
hôte→GPU » y est une copie dans la même mémoire. Les 585,9 MiB par pas du §4
coûteraient bien plus cher sur une carte discrète, où ils traverseraient le bus.
C'est précisément pour ça que la page nomme le type d'adaptateur.

---

## 2. L'inventaire : ce qui est alloué, et pourquoi on peut le croire

L'inventaire n'est **pas une formule**. Il rejoue `Model::build` : pour chaque
couche il demande aux types de couche les mêmes listes de bindings
(`get_forward_buffer_bindings`, `get_back_buffer_bindings`) que
`Layer::create_buffers` transforme en `wgpu::Buffer`, et il reproduit les règles
de **partage** — l'entrée d'une couche EST la sortie de la précédente, le
backward relie des tampons du forward, un `Concat` relie la sortie sauvegardée
d'une couche antérieure. Ce sont ces règles qui décident si deux lignes du
tableau sont une allocation ou deux.

Un inventaire qui ment est pire que pas d'inventaire, d'où le test qui porte
tout le reste :

> `the_predicted_inventory_matches_a_real_model` — égalité **à l'octet** entre
> l'inventaire prédit et `Model::estimated_gpu_bytes()` d'un modèle réellement
> construit sur l'appareil, à quatre configurations (batch 1 / 4 / 7, SGD et
> Adam, avec et sans EMA), plus le même test pour le graphe d'inférence.

Vérifié par mutation, parce qu'un test qui passe du premier coup ne prouve rien :

| mutation | tests qui tombent |
|---|---:|
| partage `PreviousOutput` supprimé (chaque entrée réallouée) | 3 |
| Adam ne garde que `m`, pas `v` | 2 |
| le backward réalloue les tampons du forward qu'il relie | 2 |
| le backward réalloue le gradient entrant | 1 |
| le chunk de dataset dimensionné sur la limite du *device* (§7) | 1 |

Et le compte de paramètres est croisé deux fois : dérivé des **tampons de
poids** ici, dérivé du **config** par `LayerDraft::parameter_count`, lui-même
déjà épinglé sur les scalaires d'un vrai checkpoint par `config::tests`.

### 2.1 `Color_Diffusion_XL`, entraînement, batch 32

```
WORKLOAD  training · batch 32 · Adam · diffusion prepass
  1 192 563 trainable scalars, 4.5 MiB of weights

GPU MEMORY  444.9 MiB total · 316.9 MiB resident · 128.0 MiB streamed
  activation gradients      ×batch  150.9 MiB   33.9%
  activations               ×batch  133.3 MiB   30.0%
  dataset chunk                     128.0 MiB   28.8%
  attention scratch         ×batch   13.0 MiB    2.9%
  optimizer state                     9.1 MiB    2.0%
  weights                             4.5 MiB    1.0%
  param gradients                     4.5 MiB    1.0%
  loss scratch              ×batch    1.1 MiB    0.3%
  diffusion prepass         ×batch  385.0 KiB    0.1%
  uniforms                            2.1 KiB    0.0%
  → 298.7 MiB scales with the batch, 18.2 MiB does not.
```

La colonne `×batch` **est** la courbe de mémoire, et c'est la seule chose à
retenir du tableau : **67 % du total bouge avec le batch, 4 % ne bouge pas** (le
reste est le chunk de dataset, qui ne dépend ni de l'un ni de l'autre).
Les poids, leurs gradients, l'état d'Adam et l'EMA sont des constantes ; tout le
reste est linéaire.

Le rapport poids/optimiseur se lit directement : `optimizer state` = 2 × `weights`
(Adam garde `m` et `v`), `param gradients` = 1 × `weights`. Avec `--ema`, une
copie de plus. Un run Adam+EMA coûte donc **4× ses poids** en postes fixes, ce
qui reste ici 18,2 MiB — invisible à côté des activations, mais pas sur un
modèle 100× plus gros.

### 2.2 L'échelle en batch, mesurée sur les deux modèles

| modèle | b16 | b32 | b64 | fixe |
|---|---:|---:|---:|---:|
| `Greyscale_Diffusion_L` | 100,0 MiB | 193,0 MiB | 378,9 MiB | 7,1 MiB |
| `Color_Diffusion_XL` | 167,5 MiB | 316,9 MiB | 615,5 MiB | 18,2 MiB |

Linéaire au terme fixe près, comme annoncé. Doubler le batch double très
exactement les activations : c'est aussi une assertion de test
(`the_batch_moves_the_activations_and_leaves_the_parameters_alone`).

### 2.3 L'attention, et la résolution à laquelle elle devient impossible

`Color_Diffusion_XL` attend sur une carte 8×8, soit **N = 64 positions** : les
lignes de softmax font `N² = 4096` flottants par échantillon, 16 KiB, donc
512 KiB à batch 32. Négligeable — mais quadratique, et c'est le chiffre à
projeter avant de déplacer un bloc d'attention vers une carte plus fine :

| carte attendue | N | `probs` / échantillon | à batch 32 |
|---|---:|---:|---:|
| 8×8 (aujourd'hui) | 64 | 16 KiB | 512 KiB |
| 16×16 | 256 | 256 KiB | 8 MiB |
| **32×32** | **1024** | **4 MiB** | **128 MiB** |

À 32×32, le seul tampon `probs` **atteint exactement**
`max_storage_buffer_binding_size`. Une attention pleine résolution sur cette
géométrie n'est donc pas une question de mémoire disponible : elle est refusée
par le contrat WebGPU, sur ce binding-là. C'est la contrainte que
`ATTENTION.md` annonçait qualitativement, ici en octets.

---

## 3. Le modèle d'exécution : résident contre streamé

La question « charge-t-on tout d'un coup, ou partie par partie ? » a **deux
réponses différentes** pour les deux moitiés du problème, et c'est ce qui rendait
la question difficile à trancher de tête.

### 3.1 Le modèle : tout d'un coup, une fois

Alloué à `Model::build()`, jamais rechargé. Les poids sont écrits une fois par
`write_buffer` au moment de l'initialisation (ou du chargement de checkpoint) et
restent là. Mesuré, en isolant la fenêtre du build :

| modèle | poids annoncés par l'inventaire | octets réellement montés au build |
|---|---:|---:|
| `Greyscale_Diffusion_L` | 1,8 MiB | **1,8 MiB** |
| `Color_Diffusion_XL` | 4,5 MiB | **4,5 MiB** |

Deux chemins indépendants — un walk de graphe sans GPU, un compteur sur la
queue — sur le même nombre. C'est aussi la preuve directe qu'il n'y a pas de
pagination : s'il y en avait une, le trafic par pas contiendrait des poids.

### 3.2 Le dataset : partie par partie, un morceau à la fois

`GpuDataset` alloue **un** tampon de chunk et y téléverse le chunk qui contient
l'échantillon demandé. Sur cette machine `select_chunk_bytes` retient
**128 MiB**, donc :

| dataset | taille | chunks | résident à la fois |
|---|---:|---:|---:|
| CIFAR-10 gris | 195,3 MiB | 2 | **128,0 MiB** |
| CIFAR-10 couleur | 585,9 MiB | 5 | **128,0 MiB** |

Un dataset de 585,9 MiB coûte donc 128 MiB de GPU. C'est le poste « streamé » de
la page, et le seul.

Le chunk fait 128 MiB et non 64 : `select_chunk_bytes` vise le huitième de la
capacité **de l'adaptateur** (28 GiB ici, donc 3,5 GiB), puis se fait plafonner
par `min(max_buffer_size, max_storage_buffer_binding_size) = 128 MiB`. Le
plancher de 64 MiB ne mord que sur un adaptateur annonçant moins de 1 GiB. Ce
n'est pas ce que `BATCH_DISPATCH.md` §5.2 avait calculé — voir §7.

---

## 4. Les transferts, mesurés

Les compteurs vivent sur `GpuContext` : `write_buffer`, `submit` et
`record_readback` remplacent les appels directs à la queue dans tout le moteur.
Quatre `fetch_add(Relaxed)` sur un chemin qui parle déjà à un driver — rien à
activer, rien à désactiver.

Le compteur n'est vrai que **s'il n'y a pas de seconde porte**, alors la règle
est vérifiée mécaniquement : `nothing_in_the_engine_bypasses_the_counted_queue`
relit les sources du moteur et échoue sur tout `queue.write_buffer` /
`queue.submit` hors de `gpu_context.rs`. Il a servi immédiatement — quatre sites
écrits sur deux lignes (`gpu.queue\n    .write_buffer(...)`) avaient échappé au
routage textuel, et le « build » annonçait alors 4,4 KiB au lieu de 1,8 MiB.
Sans ce test, le rapport aurait publié ce 4,4 KiB.

Protocole : le **premier pas est un warmup, exclu** — il monte le premier chunk
et amorce les pipelines ; l'amortir sur la moyenne attribuerait un chunk entier
à chaque pas pour toujours.

### 4.1 Par pas d'entraînement

| modèle | batch | hôte→GPU mesuré | prédit | GPU→hôte | aller-retours | submits |
|---|---:|---:|---:|---:|---:|---:|
| `Greyscale_Diffusion_L` | 16 | **195,3 MiB** | 195,2 | 4,0 KiB | 1,0 | 4,0 |
| | 32 | **195,3 MiB** | 195,3 | 4,0 KiB | 1,0 | 4,0 |
| | 64 | **195,3 MiB** | 195,3 | 4,0 KiB | 1,0 | 4,0 |
| `Color_Diffusion_XL` | 16 | **576,7 MiB** | 567,5 | 12,0 KiB | 1,0 | 6,9 |
| | 64 | **585,9 MiB** | 585,9 | 12,0 KiB | 1,0 | 7,0 |

La colonne « prédit » est la formule fermée du §5.1, calculée **sans GPU** par
`plan_dataset`. Elle tombe à 0,05 % de la mesure sur le gris et à 1,6 % sur le
couleur — l'écart restant est la différence entre une espérance et un tirage de
huit pas.

Le `submits` se décompose exactement, et reproduit la structure décrite par
`BATCH_DISPATCH.md` §2.4 : **une** soumission de calcul pour tout le pas, plus
une par chunk distinct touché, plus une pour la lecture de la loss. Gris :
1 + 2,00 + 1 = 4,0 ✓. Couleur à batch 16 : 1 + 4,81 + 1 = 6,81 contre 6,9
mesurés. Le `GPU→hôte` est cette loss seule, une tranche de la sortie — 4 KiB en
gris, 12 KiB en couleur.

Un pas de remontée fait 2 soumissions pour la même raison : le `predict` et la
lecture de son résultat.

### 4.2 Par pas de débruitage (inférence)

| modèle | hôte→GPU | GPU→hôte | aller-retours | submits |
|---|---:|---:|---:|---:|
| `Greyscale_Diffusion_L` | 20,0 KiB | 4,0 KiB | **1,0** | 2,0 |
| `Color_Diffusion_XL` | 28,0 KiB | 12,0 KiB | **1,0** | 2,0 |

**Une image = 256 pas = 256 allers-retours CPU↔GPU**, pour 6 MiB (gris) ou
10 MiB (couleur) déplacés au total. Le §5.2 dit pourquoi ce n'est pas une
question de bande passante.

---

## 5. Ce que l'inventaire révèle

Trois pistes, par ordre de gain estimé. Aucune n'est implémentée ici — cette
mission rend le coût **visible**, et ces trois-là ne sont visibles que
maintenant.

### 5.1 Le dataset est re-téléversé en entier à chaque pas — le plus gros gain restant

C'est la ligne la plus dure du rapport. À batch 16 et au-delà, un pas
d'entraînement monte **tout le dataset**.

Le mécanisme est arithmétique, pas accidentel. Les indices d'un batch sont tirés
d'une permutation du dataset **entier** ; un chunk qui détient une part `p` du
dataset est téléversé sauf si les `B` tirages l'ont tous manqué, soit
`1 − (1 − p)^B`. Sommé sur les chunks :

| dataset | chunks | batch | chunks touchés | trafic prédit | mesuré |
|---|---:|---:|---:|---:|---:|
| gris | 2 | 16 | 2,00 / 2 | 195,2 MiB | **195,3 MiB** |
| couleur | 5 | 16 | 4,81 / 5 | 567,5 MiB | **576,7 MiB** |
| couleur | 5 | 64 | 5,00 / 5 | 585,9 MiB | **585,9 MiB** |

La somme est faite **par chunk** et non avec le `C·(1 − ((C−1)/C)^B)` qui vient
à l'esprit : cette forme-là suppose des chunks égaux, et le dernier ne l'est
jamais (17 232 échantillons contre 32 768 sur le gris). Le facturer plein
sur-prédisait le trafic de 31 %.

**Conséquence contre-intuitive : le trafic ne dépend pas du batch.** 195,3 MiB
par pas à batch 16, 32 et 64. Donc le trafic *par échantillon* est divisé par
quatre entre batch 16 et 64 — un argument pour les gros batchs qui n'apparaît
dans aucune mesure de vitesse existante.

Deux corrections possibles, l'une triviale :

- **Rendre le dataset entièrement résident quand il tient.** Plusieurs tampons
  de chunk au lieu d'un seul : CIFAR-10 gris = 195,3 MiB en 2 tampons de
  ≤ 128 MiB, chacun exactement au plafond de binding. Le trafic par pas tombe à
  **zéro** après la première époque. Coût : 67 MiB de GPU en plus (le second
  chunk), à comparer aux 193,0 MiB du modèle à batch 32 — sur mémoire unifiée,
  rien. Ne marche pas pour le couleur (585,9 MiB en 5 tampons), qui a besoin de
  la seconde piste.
- **Un échantillonnage conscient des chunks** : parcourir les chunks dans un
  ordre permuté et tirer les batchs *à l'intérieur* du chunk résident. Un batch
  touche alors un chunk, donc ≈ 1 téléversement de 128 MiB par ~10 000 pas au
  lieu de 5 par pas. Ce n'est **pas** le même échantillonnage — la corrélation
  intra-chunk devient réelle — et il faudrait vérifier que la trajectoire de
  loss ne bouge pas avant d'y toucher. Le protocole existe :
  `compare_metrics.py` et les runs appariés de `BATCH_DISPATCH.md` §6.

`BATCH_DISPATCH.md` §7.4 avait déjà vu l'ombre de ce coût — 630 s de temps
**système** sur l'ancien chemin, « exactement la signature de ce trafic ». Ce
rapport en donne le chiffre : il n'a pas disparu avec le batching, il a été
divisé par ~3.

### 5.2 Le latent de la chaîne inverse vit sur le CPU — 256 stalls par image

`sample_diffusion` garde `latent: Vec<f32>` et rappelle `predict()` à chaque
pas. Chaque `predict` est : `write_buffer` de l'entrée composée, un `submit`,
puis `read_last_output` — qui copie dans un staging buffer, appelle `map_async`
et **bloque sur `device.poll(wait)`**. Le GPU se vide, le CPU attend, 256 fois
par image.

La page le dit dans ces termes parce que les octets induisent en erreur : 6 MiB
par image, c'est trois millisecondes de bande passante. Le coût est la
**sérialisation** — aucun pas ne peut être encodé pendant que le précédent
s'exécute.

La correction est structurelle : garder le latent dans un tampon GPU et écrire
`reverse_step` en shader (l'échantillonnage postérieur est une combinaison
linéaire plus un bruit gaussien, et `gaussian_at` existe déjà côté WGSL pour le
prépass). La chaîne entière tiendrait alors dans un ou deux `submit`, et le seul
aller-retour serait la lecture de l'image finale. Deux gardes existent déjà pour
prouver qu'on n'a rien changé en le faisant : la non-régression **octet pour
octet** du PNG de `--headless-sample` (`BATCH_DISPATCH.md` §5) et le dump de
`--headless-perpetual`.

À noter : le mode Perpetual paie la même chose par frame, et il vise 30 pas/s.

### 5.3 Les activations dominent, et ce sont elles qui fixent le plafond

75 à 85 % du total, dans les deux modèles, aux trois batchs. Si la mémoire
devenait contraignante, c'est le seul poste dont la réduction change quelque
chose — et il y a du gras identifiable : `pre_activation` est conservé par les
couches d'activation pour le backward, `fwd_scratch` par l'attention. Un
recalcul (« gradient checkpointing ») les échangerait contre du temps.

Mais il faut être honnête sur la priorité : ce dépôt n'est **pas** contraint par
la mémoire aujourd'hui (615 MiB pour le plus gros run, plafond de batch à 341
alors qu'on entraîne à 16-32). Cette piste est à garder pour le jour où un
modèle 10× plus grand la rendra vraie, pas à faire maintenant.

---

## 6. Répondre pour une autre machine

`--vram N` pose un budget, `--device <nom>` / `--no-gpu` répondent avec les
**limites WebGPU par défaut** — celles qu'un navigateur accorde à une page sans
rien demander (128 MiB par binding de stockage, 256 MiB par tampon).

```
$ batlab --resources Color_Diffusion_XL --batch 64 --no-gpu --vram 0.5 --dataset …
GPU MEMORY  679.5 MiB total · 615.5 MiB resident · 64.0 MiB streamed
VERDICT  DOES NOT FIT
  · total 679.5 MiB exceeds the 512.0 MiB budget
BATCH CEILING  46
```

(679,5 et non 743,5 MiB : sur un appareil hypothétique l'adaptateur ne promet
rien de plus que le device, donc la règle de chunk retombe sur son plancher de
64 MiB. Le chunk est une propriété de la machine visée, pas du modèle.)

Le verdict **nomme la limite** rencontrée, et le plafond de batch qui en découle
est exact et non échantillonné : chaque poste croît linéairement ou pas du tout,
donc « ça tient » est monotone en batch et la bissection tranche
(`the_batch_ceiling_is_the_last_batch_that_fits` vérifie que le batch suivant ne
tient pas).

Quelques réponses, pour `Color_Diffusion_XL` :

| machine | plafond de batch |
|---|---:|
| ce Mac (pas de budget exposé — binding 128 MiB) | 341 |
| carte 8 GiB | 341 *(le binding mord avant la mémoire)* |
| carte 2 GiB | 210 |
| carte 512 MiB | 46 |

(`Greyscale_Diffusion_L` : 512 sur ce Mac, même mécanisme.)

Et la réponse qui compte pour l'objectif navigateur : **l'inférence de
`Color_Diffusion_XL` tient dans 8,9 MiB**, dont 4,5 de poids, sur les limites
WebGPU par défaut. Le graphe d'inférence n'est pas ce qui empêchera batLab de
tourner chez le visiteur.

---

## 7. Deux nombres appelés `max_buffer_size` — et une correction à `BATCH_DISPATCH.md`

Le seul bug de cette mission, trouvé en poursuivant un écart entre une
prédiction et une mesure, et il mérite d'être écrit parce que la prédiction
était **plausible**.

`--measure` annonçait 576,7 MiB par pas sur le couleur là où `plan_dataset`
prédisait 521,4 MiB. 10 % d'écart : assez pour ne pas être du bruit, assez peu
pour être mis sur le compte d'un tirage. Le compteur `chunk_loads` a tranché :
4,88 chargements par pas, pour 5 chunks — alors que la prédiction en supposait
10.

**Cause.** Deux champs de ce dépôt s'appellent `max_buffer_size` :

| source | valeur ici | ce que c'est |
|---|---:|---|
| `device.limits().max_buffer_size` | 256 MiB | ce que le graphe peut allouer |
| `adapter.limits().max_buffer_size`, exposé par `GpuSpecs::memory_size()` | **28,08 GiB** | ce que le matériel accepterait |

`GpuDataset` dimensionne ses chunks à partir du **second** ; `DeviceProfile` ne
portait que le premier. Le prédicteur calculait donc des chunks de 64 MiB là où
le dataset en allouait de 128, et **tout ce qui en découle était divisé par
deux** : la taille du chunk résident, le nombre de chunks, le trafic prédit.

**Correction.** `DeviceProfile` porte les deux, sous deux noms qui ne peuvent
plus être confondus, et la page les affiche côte à côte. Gardé par
`the_predicted_chunk_plan_is_the_one_the_dataset_allocates`, qui compare le plan
prédit à la capacité d'un `GpuDataset` **réellement construit**.

Ce test a demandé une leçon de plus. Sa première version utilisait des fixtures
de 64 échantillons et **laissait passer la mutation** : `select_chunk_bytes`
finit par `target.min(dataset_total)`, donc sur un dataset plus petit qu'un
chunk *toutes* les règles concordent — le chunk est le dataset. C'est
exactement pourquoi le bug avait survécu à une suite entière. Le cas qui mord
alloue maintenant 137 MiB pour de vrai, et sans lui le test est décoratif.

**Conséquence pour `BATCH_DISPATCH.md` §5.2.** Ce rapport écrit : « sur cette
machine `max_storage_buffer_binding_size` vaut 128 MiB, la règle de
`select_max_chunk_bytes` retient **64 MiB** par chunk, et CIFAR-10 gris occupe
donc **4 chunks** », d'où ses « ≈ 3,96 téléversements » à batch 16 et « ≈ 12,25 »
sur l'ancien chemin. La règle n'a pas changé depuis (elle a seulement été
extraite en fonction pure), et elle rend **128 MiB** dès que l'adaptateur
annonce plus de 1 GiB par tampon. Les bons chiffres sont **2 chunks de
128 MiB**, soit **≈ 2,00 téléversements** par pas à batch 16 — et **195,3 MiB**,
mesurés ici.

La *conclusion* de ce rapport-là est intacte, et même renforcée : le trafic
dataset est de l'ordre du dataset entier par pas, et c'est ce qui explique ses
630 s de temps système. Seuls les nombres intermédiaires sont à relire.
Ni `CLAUDE.md` ni `INDEX.md` ne reprennent le chiffre — seul
`BATCH_DISPATCH.md` le porte, en §2.4, §5.2 et §8, et il n'a pas été réécrit :
les rapports de ce dépôt sont des archives (`INDEX.md`), et c'est ce
paragraphe-ci qui fait foi.

---

## 8. Limites, et ce que ce rapport ne dit pas

- **Le budget mémoire n'est pas mesuré, il est déclaré.** Aucune API portable
  n'en donne un. `DeviceProfile::allocator_report` interroge l'allocateur de
  wgpu quand il y en a un — il rend `None` sur Metal. Un verdict « tient » sans
  `--vram` porte donc uniquement sur les limites par binding et par tampon.
- **L'inventaire ne compte pas ce que le driver ajoute** : alignements internes,
  tampons de staging transitoires (`read_back_f32` en alloue un par lecture),
  pipelines compilés. Il compte les `wgpu::Buffer` que le graphe demande, ce qui
  est exactement ce que `estimated_gpu_bytes` compte — et rien d'autre.
- **Le prépass de diffusion est compté, la `LiveFrame` ne l'est que sur
  demande** (`live_frame: true` dans la requête). Le CLI ne la demande pas : une
  page qui suppose une fenêtre ouverte sur-compterait tout run headless.
- **`--measure` mesure quelques pas, pas un run.** Les moyennes sur 6 à 10 pas
  sont stables ici parce que le trafic est déterministe (le tirage des indices
  l'est), mais ce ne serait pas vrai d'une mesure de *temps*.
- **Le trafic mesuré est une borne basse du coût réel sur carte discrète.**
  Cette machine a une mémoire unifiée ; les mêmes octets traverseraient un bus
  PCIe ailleurs. Le rapport ne prétend pas prédire ce coût-là, seulement le
  volume qui le causerait.
- **La page ne mesure pas le temps.** Rien ici ne dit combien coûtent les
  585,9 MiB par pas en millisecondes : ce serait un banc, avec contrôle nul, et
  `BATCH_DISPATCH.md` §7.1 rappelle qu'un temps réel mesuré sur GPU partagé a un
  plancher de bruit de 30 %. Le volume, lui, est exact et reproductible.

## 9. Ce qui reste

- **Faire tomber le §5.1.** Le dataset entièrement résident quand il tient est
  une poignée de lignes dans `GpuDataset` (un `Vec<Buffer>` au lieu d'un
  `Buffer`) et supprime la totalité du trafic d'entraînement pour le modèle
  gris. Le cas couleur demande la seconde correction, qui change
  l'échantillonnage et doit donc passer par un run apparié.
- **Garder le latent sur le GPU** (§5.2). Chantier plus gros, garde-fou déjà
  écrit (non-régression octet pour octet du PNG).
- **Un poste « temps » sur la page.** Les compteurs comptent des octets et des
  soumissions ; ils ne chronomètrent rien. Un `TransferCounters` qui accumulerait
  aussi le temps passé bloqué dans `poll(wait)` rendrait le §5.2 chiffrable en
  secondes, ce qu'il n'est pas encore.
- **Faire remonter `chunk_loads` dans le moniteur d'un run.** `BATCH_DISPATCH.md`
  §9 le demandait déjà ; les compteurs de cette mission le rendent trivial, mais
  le moniteur d'entraînement ne l'affiche toujours pas.
