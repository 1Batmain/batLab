# Self-attention spatiale au goulot — conception, preuves, mesures

Mission de nuit du 6 août 2026. Objectif : une couche de self-attention
spatiale au goulot 8×8 du U-Net (64 positions), prouvée et intégrée à
`Color_Diffusion_XL`, pour qu'un run parte dans la nuit.

---

## 1. Ce que fait la couche

`y = x + W_o · softmax(qᵀk / √d) · v`

La séquence, ce sont les `H*W` positions spatiales ; la dimension de features,
le nombre de canaux. Single-head, `d = C = 192` au goulot. La couche est
**shape-preserving** et le **résiduel est DANS la couche** : le graphe reste une
chaîne, aucune chirurgie. La GroupNorm pré-attention est la couche GroupNorm
existante placée devant dans la config, pas une normalisation interne.

`W_o` s'initialise à **zéro** : au pas 0 la couche est l'identité exacte, donc
l'insérer dans un réseau qui marche ne peut être que neutre ou meilleur.

## 2. Les deux décisions de conception qui portent tout

### 2.1 Les quatre projections partagent UN tenseur de poids

`OptimizerBindings` ne nomme qu'un tenseur de poids et un tenseur de biais par
couche. Adam et ses moments, la mise à jour SGD, le décodage de checkpoint et la
remise à zéro des gradients sont tous bâtis là-dessus.

Q, K, V et W_o sont donc concaténés dans un seul buffer `4·C·C`, leurs biais
dans un `4·C`. Chacun de ces chemins traite alors l'attention **exactement comme
une convolution** : pas de plomberie d'optimiseur nouvelle, pas de cinquième
chemin à tenir synchrone. Les shaders adressent une projection par son offset
`which * C * C` ; cette arithmétique est le coût total du choix.

### 2.2 Le forward est multi-passes

L'attention exige une barrière **globale** en plein forward : la ligne `n` des
scores lit les `k[m]` et `v[m]` de toutes les autres lignes, produites par
d'autres workgroups. La seule synchronisation disponible à cette échelle est
celle que wgpu insère entre deux passes de compute d'un même encodeur.

`Pipelines::forward` passe donc de `Option<ComputePipeline>` à la même liste
`(pipeline, workgroups)` que le backward. **Toute autre couche déclare un seul
entry point et atterrit dans une liste à un élément** : le cas courant est une
liste de longueur 1, pas un cas particulier.

Découpage : `attn_qkv` → `attn_scores` → `attn_context` → `attn_out`.
Backward : sept passes, de `attn_back_ctx` à `attn_back_bias`.

## 3. Le piège de la nuit : la limite de 8 storage buffers

**C'est la découverte principale de cette mission, et elle dépasse l'attention.**

Le backward liait douze storage buffers. WebGPU en garantit **huit** par étage
(`max_storage_buffers_per_shader_stage`). Symptôme observé :

```
grad_weights = [0,0,0,…]   nonzero = 0
grad_input   = [0,0,0,…]
fwd qkv/probs/ctx/output : TOUS nuls — alors que les biais sont non nuls
```

Le forward ne tournait pas non plus. Raison : le bind group layout du backward
est rejeté, ce qui invalide le **command buffer entier** — et forward et
backward sont encodés dans le même encodeur (`encode_train_graph_without_opt`).
Rien ne lève d'exception. Seul `device.on_uncaptured_error` écrit une ligne sur
stderr, que le harnais de test n'affiche pas. C'est exactement le mode d'échec
que `dispatch_grid` documente déjà : « la passe ne calcule rien, loss 0.000000 ».

**Correction : pas relever la limite au `request_device`.** Le moteur doit
tourner sur le GPU du visiteur via WebGPU, et 8 est le défaut de la spec ;
demander plus échangerait la portabilité contre du confort. Les scratch
fusionnent en **deux tampons jumeaux de même disposition** :

```
fwd_scratch  [ q | k | v | ctx | probs ]     pas = 4·N·C + N²
grad_scratch [ gq| gk| gv| gctx| gscores ]
               0  NC  2NC  3NC   4NC
```

Le gradient d'un bloc vit à l'offset du bloc : `block_base(b, block)` adresse
les deux tampons. Bilan : forward 5 storage (au lieu de 7), backward 8 (au lieu
de 12), et moins de mémoire.

Gardé par `neither_bind_group_exceeds_the_webgpu_storage_buffer_limit`, qui
compte les liaisons storage des deux bind groups. C'est un **test** et non une
note parce que la panne est muette.

## 4. Preuves

### 4.1 Référence CPU f64 indépendante

`reference_attention` dans `attention_tests.rs` est écrite **depuis la formule**,
pas depuis le shader : ses propres boucles, ses propres `Vec`, en double
précision. Comparée sur trois formes, dont **(9, 8, 3) — 72 positions**, plus
que les 64 threads du workgroup de softmax : la boucle à pas exercée. Écart
relatif < 2e-4.

### 4.2 Différences finies

Sur les quatre projections (un seul balayage, le tenseur est packé), sur les
quatre biais, et sur **l'entrée** (avec la branche résiduelle). `eps = 2e-3`,
tolérance 3 % — le plancher de quantification f32 impose ce compromis, comme
dans `audit_tests.rs`.

### 4.3 L'axe batch — le risque spécifique de cette couche

L'attention est la seule couche du graphe où une position en lit une autre :
un offset par échantillon manquant laisserait l'image 3 attendre l'image 4.
Ça ne plante pas, ça n'a même pas l'air faux — la loss continue de descendre.

- `a_batched_pass_equals_the_same_samples_run_separately` : B échantillons
  batchés == B passes séparées, avec des **échelles différentes par échantillon**
  (0,4 / 1,2 / 2,0 / 2,8 / 3,6) pour qu'un mélange ne puisse pas se cacher sous
  une tolérance relative.
- `one_samples_data_cannot_reach_another_samples_output` : **chaque** échantillon
  prend son tour comme perturbé, et tous les autres doivent sortir **bit à bit**
  identiques. La première version ne perturbait que l'échantillon 2 — elle
  prouvait seulement que *lui* ne fuit pas vers l'extérieur, et une mutation qui
  lisait les k/v d'un échantillon 0 codé en dur y **survivait**. C'est la
  validation par mutation qui l'a exhumée.
- `batched_parameter_gradients_sum_the_per_sample_gradients` : les gradients de
  paramètres d'un batch sont la **somme** des gradients par échantillon.

### 4.4 Identité résiduelle

`W_o = 0` → sortie == entrée **bit à bit** (`assert_eq`, pas une tolérance). Et
`a_freshly_built_attention_layer_starts_as_the_identity` vérifie la couche telle
qu'elle est réellement **construite** : Q/K/V non nuls (une couche toute à zéro
passerait aussi le test d'identité et serait morte), W_o exactement nul.

### 4.5 Validation par mutation — 16 mutations, 16 attrapées

| Mutation | Test qui l'attrape |
|---|---|
| `grad_weights = acc * 2` (sanity) | différences finies poids |
| `grad_bias = acc * 2` (sanity) | différences finies biais |
| `grad_input = acc * 2` (sanity) | différences finies entrée |
| softmax sans soustraction du max | débordement `exp` |
| Q/K transposés au forward | référence f64 |
| scale `1/√d` omis | référence f64 |
| fuite : k du sample 0 | anti-fuite |
| fuite : v du sample 0 | anti-fuite |
| résiduel absent au forward | identité |
| grad_q / grad_k échangés | différences finies |
| softmax backward sans le terme soustrait | différences finies |
| résiduel absent de grad_input | différences finies entrée |
| réduction du batch tronquée | somme des gradients batchés |
| grad_v : probs non transposé | différences finies |
| repli des workgroups excédentaires (course sur la ligne) | grille 2-D |
| `unit` lu depuis `wid.x` seul | grille 2-D |

Les trois « sanity » sont là pour une raison : **la première version de la suite
passait 15/15 alors que le backward ne s'exécutait pas du tout**. Analytique = 0
et numérique = 0 (la loss ne bougeait plus avec les poids) donnent une erreur
relative de 0. Une suite verte du premier coup méritait cette vérification.

### 4.6 La grille 2-D, franchie pour de vrai

Les passes de lignes sont dispatchées **un workgroup par (échantillon, ligne)**,
donc elles atteignent la limite de 65 535 quand `seq · batch > 65 535` — au
goulot 8×8, à partir d'un batch de 1 024.

La première implémentation « repliait » les workgroups excédentaires sur la
dernière ligne, en pensant que la passe était idempotente. **Elle ne l'est pas** :
la softmax écrit le score, le relit pour l'exponentier, relit pour normaliser.
Deux workgroups sur la même ligne s'entrelacent et l'un lit une valeur que
l'autre a déjà transformée. Ce n'est pas « le même travail fait deux fois »,
c'est de la corruption — et uniquement au-delà de 65 535 workgroups, là où
personne ne regarde.

Corrigé en **retour anticipé**, légal précisément parce que la condition est
*uniforme au workgroup* (`unit` vient de `wid` et `nwg`, identiques pour toutes
les invocations) : toutes prennent la même branche et les barrières restent en
flot de contrôle uniforme. Une condition par thread ne pourrait pas.

`the_row_passes_survive_the_two_dimensional_dispatch_grid` construit un batch de
1 024 à 8×8 — **65 536 workgroups, un de trop** — et compare cinq échantillons
sonde (dont le premier et le dernier) à la référence f64. Rien en dessous de
cette taille de batch ne l'aurait attrapé.

### 4.7 Deux pièges de méthode, pour la suite

1. **`cargo` ne suit pas les `.wgsl` inclus par `include_str!`** dans ce dépôt :
   modifier un shader seul ne déclenche PAS de recompilation. Toute campagne de
   mutation sur un shader doit `touch` le `.rs` qui l'inclut, sinon les tests
   tournent contre l'ancien binaire et **toutes les mutations survivent**.
2. **Commiter avant de muter.** Un harnais qui restaure avec `git checkout --`
   écrase les modifications non commitées.

## 5. Coût, et pourquoi c'est une couche de goulot

Le scratch `probs` est en `N²` par échantillon : 64 positions au goulot 8×8 font
4 096 flottants, la **même** couche en 32×32 en ferait 1 048 576 — 256× plus.
L'attention n'est abordable que là où la grille spatiale est petite.

La passe dominante est `attn_back_weights` : `4·C·C = 147 456` threads, chacun
sommant sur `batch·N` positions. C'est la première cible d'optimisation si le
coût gêne.

## 6. Intégration

`gen_unet_config.py --attention` insère `GroupNorm + Attention` après le bloc du
goulot. `Color_Diffusion_XL` passe de 28 à 30 couches ; l'invariant temporel
tient (`input.z = 7 > output.z = 3`).

`LayerDraft::Attention` traverse config, CLI et TUI. La couche n'a aucun champ
de forme propre — séquence et features sont hérités — donc son formulaire n'a
qu'un « Save As ».

## 7. Mesures

`Color_Diffusion_XL` (48/96/192) contre le même sans les deux couches
`GroupNorm + Attention`. Adam, lr 1e-3, CIFAR-10 RGB, 100 pas, médiane des
intervalles en régime établi.

| batch | sans attention | avec attention | surcoût |
|---:|---:|---:|---:|
| 16 | 2 240 ms/pas — 140,0 ms/éch. | 2 240 ms/pas — 140,0 ms/éch. | 0 % (sous la résolution) |
| 32 | *(non mesuré)* | 4 417 ms/pas — 138,0 ms/éch. | ≈ 0 % |
| 64 | 8 792 ms/pas — 137,4 ms/éch. | 17 360 ms/pas — 271,2 ms/éch. | **+97 %** |

**L'attention est gratuite jusqu'à batch 32 et double le pas à batch 64.** Le
U-Net seul est parfaitement linéaire (140,0 → 137,4 ms/éch. de 16 à 64) ; seule
l'attention décroche, et d'un coup — c'est une falaise entre 32 et 64, pas une
croissance.

Non expliqué. L'attention pèse ~11 MMACs/échantillon sur les 238 du réseau
(~5 %), toutes ses passes sont linéaires en batch, aucun compte de workgroups
n'approche 65 535 (36 864 au plus), et le calcul reste juste (loss identique aux
deux configurations). Hypothèse non vérifiée : un seuil d'occupation ou de
pression mémoire franchi entre 32 et 64. Premières cibles à profiler :
`attn_back_weights` (147 456 threads sommant chacun sur `batch·N`) et
`attn_back_bias` (768 threads — 12 workgroups, occupation très faible).

Verdict et commande de run : `GO_NOGO.md` (déplacé ici depuis la racine).

## 8. Ce qui n'a pas été fait

- **Multi-head** : le single-head est prouvé, le multi-head n'a pas été tenté
  (l'ordre de mission le conditionnait au temps restant).
- **Optimisation de `attn_back_weights`** : aucune passe de perf n'a été faite.
  Le découpage en lanes de `ConvolutionType::reduction_lanes` est le modèle à
  suivre si ça devient le goulot.
- **Test à l'aveugle** : cette suite est co-écrite avec l'implémentation. Le
  protocole du dépôt demande un agent de test indépendant ; la validation par
  mutation ci-dessus est un substitut partiel, pas un remplacement.
