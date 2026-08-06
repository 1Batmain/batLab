# Porter l'axe batch dans les dispatches — conception

Branche `batch-dispatch`. Répond au §7 de `PERF_CONVOLUTION.md` (« le goulot est
structurel : un encoder et un `submit` par échantillon »).

Ce document est écrit **avant** le code. Il fixe la représentation, liste les
points durs et dit, pour chacun, ce qui doit rester invariant et ce qui a le
droit de bouger. Le rapport final (`BATCH_DISPATCH.md`) dira ce qui a
effectivement été mesuré.

---

## 1. L'état des lieux, en une phrase

`diffusion.rs::train_step_batch_inner` boucle sur les `B` échantillons du batch
**côté CPU**. Chaque tour crée un `CommandEncoder`, y encode le graphe complet
(prepare, forward, loss, backward) et le **soumet**. À `B = 16` : 18 `submit`
par pas, et chaque kernel ne voit qu'un seul petit tenseur — le forward de
`conv4` de `Greyscale_Diffusion` a 1024 éléments de sortie, soit 16 workgroups.

## 2. La représentation retenue : le batch est *implicite*

Le choix central. Deux options étaient ouvertes :

- **(A)** ajouter un champ `batch` à chaque uniforme de couche, et faire lire
  aux shaders `layer_spec.batch` ;
- **(B)** ne rien ajouter aux uniformes : allouer les tampons d'activation à
  `B × taille_par_échantillon`, dispatcher `B ×` plus de workgroups, et laisser
  chaque kernel **déduire** son indice d'échantillon de son indice global.

**On retient (B).** Raisons :

1. Les uniformes des couches gardent **exactement** leur taille et leurs
   offsets. Les fixtures legacy de `conv_equivalence_tests.rs` et de
   `group_norm_equivalence_tests.rs` lient toujours le même tampon, et
   `conv_reduction_lanes_agrees_with_dispatch` continue de relire le mot à
   l'offset 12. **Aucun test d'équivalence existant n'a besoin d'être touché.**
2. Le batch n'a alors qu'**une seule** source de vérité par kernel — la taille
   du tampon, via `arrayLength()`. Il n'y a pas de champ d'uniforme qui puisse
   dériver de la taille réellement allouée. La classe de bug « l'uniforme dit 16,
   le tampon en contient 8 » n'existe pas.
3. La moitié des shaders (`activation`, `back_activation`, `sum`) bornent déjà
   leur boucle par `arrayLength(&input)` et deviennent batchés **sans une seule
   ligne de changement**.

Concrètement, pour un kernel élémentaire :

```wgsl
let per   = OH * OW * K;              // taille par échantillon, déjà dans l'uniforme
let total = arrayLength(&output);     // = per * B
if idx >= total { return; }
let sample = idx / per;
let local  = idx % per;
// puis : input[sample * IN_LEN + …], output[idx]
```

### Ce qui est « par échantillon » et ce qui est « partagé »

C'est l'inventaire qui compte, et il est porté par les types de couche
eux-mêmes : `get_buffers_specs(batch)` / `get_back_buffers_specs(batch)`
reçoivent le batch et multiplient **elles-mêmes** les tampons concernés.

| par échantillon (× B) | partagé (× 1) |
|---|---|
| `input`, `output`, `pre_activation` | `weights`, `bias`, `gamma`, `beta` |
| `grad_input`, `grad_output` | `grad_weights`, `grad_bias`, `grad_gamma`, `grad_beta` |
| `loss_terms`, `target`, `model_result` | les uniformes `specs` |
| le tampon `stats` de GroupNorm (4 f32 / groupe / échantillon) | l'état Adam (`m`, `v`) |
| le tampon `grad_merge` de Concat | — |
| `diffusion_clean_target` | — |

La colonne de droite est exactement la liste des **paramètres du modèle et de
leurs gradients** : ils ne portent pas d'axe batch, ils en sont la *réduction*.

## 3. Les points durs

### 3.1 L'accumulation des gradients — là où l'ordre de sommation change

Aujourd'hui : `begin_batch_accumulation` met `grad_weights`/`grad_bias` à zéro,
puis chacun des `B` passages fait un `+=` (somme séquentielle en `B` étapes de
sommes déjà réduites en arbre par échantillon), puis `finish_batch_accumulation`
dispatche l'optimiseur avec `grad_scale = 1/B`.

En batché, l'axe batch **entre dans le kernel de réduction**. Pour
`conv_back_weights`, la boucle de positions passe de `OH*OW` à `B*OH*OW` :

```wgsl
let positions = OH * OW;
let batch     = arrayLength(&grad_output) / (OH * OW * K);
for (var p: u32 = lane; p < positions * batch; p += lanes) {
    let sample = p / positions;
    let pos    = p % positions;
    …
}
```

Il n'y a donc plus qu'**un seul `+=` par élément de poids et par pas**, au lieu
de `B`. Le `+=` est conservé (et non remplacé par `=`) pour que la mise à zéro
initiale garde son sens et qu'un appelant puisse encore accumuler plusieurs
micro-batches.

**Ce qui change, et c'est assumé** : l'ordre de sommation flottant. Avant :
`((s₀ + s₁) + s₂) + …`, chaque `sᵢ` étant lui-même une réduction en arbre sur
les positions. Après : une seule réduction en arbre sur l'axe `batch × positions`
entrelacé par `lanes`. C'est exactement la situation traitée par
`PERF_GROUP_NORM.md` §3.2 et `PERF_CONVOLUTION.md` §4.2, et elle sera bornée
de la même façon :

- oracle **f64** sur les mêmes entrées ;
- exigence `erreur(batché) ≤ 1,5 × erreur(séquentiel)` face à l'oracle — la
  réduction en arbre ne doit jamais être *moins* précise que la somme
  séquentielle qu'elle remplace (elle devrait l'être davantage : l'arbre a une
  profondeur logarithmique là où la boucle sur `B` était linéaire).

Même traitement pour `conv_back_bias`, `group_norm_back_gamma`,
`group_norm_back_beta`, `upsample_conv_back_weights/bias` et
`fully_connected_back_weights/bias`.

**Le nombre de workgroups de ces passes ne change pas** : le nombre de sommes
est celui des poids, qui ne dépend pas du batch. Seul le travail par somme est
multiplié par `B`. C'est le sens même de l'opération.

> Note de calibration : `ConvolutionType::reduction_lanes` a été calibrée sur
> `positions` par échantillon. Avec l'axe batch, chaque somme couvre `B ×` plus
> de positions et l'optimum de `lanes` peut se déplacer. La règle est laissée
> **inchangée** dans un premier temps (elle ne touche pas la correction), et le
> re-balayage est un point explicite du plan de mesure — `bench_conv_reduction_lanes`
> existe déjà pour ça.

### 3.2 Les graines de bruit — sémantique par échantillon strictement préservée

C'est la contrainte la plus dure du lot, et c'est ce qui rend la validation
appariée possible : **chaque échantillon du batch doit recevoir exactement le
même bruit et le même timestep qu'avant.**

Aujourd'hui `diffusion_prepare` lit un **uniforme scalaire** réécrit entre deux
soumissions :

```rust
let step_seed = seed ^ ((batch_offset as u64) << 32) ^ sample_index as u64;
DiffusionPrepareUniform { alpha_bar, step, seed: fold_seed(step_seed), … }
```

En batché il n'y a plus qu'une soumission : l'uniforme devient un **tableau de
`B` structures dans un tampon de stockage**, indexé par `sample`. Le calcul
CPU de chaque entrée est **repris verbatim** — mêmes `batch_offset`, même
`sample_index` issu de la même `SampleShuffle`, même `fold_seed`. Le shader
lit `specs[sample]` au lieu de `specs`, et rien d'autre ne bouge :

```wgsl
let noise = gaussian_from_seed(specs[sample].seed ^ clean_idx);
```

Sur la règle anti-`seed ^ index` du CLAUDE.md : le motif interdit est la
**composition** de deux XOR (`path_seed ^ diffusion_step` côté sampler *et*
`index ^ …` côté champ de bruit), qui faisait s'effondrer la somme sur les
anti-diagonales (`ANISOTROPY_HUNT.md`). Ici il n'y a qu'un seul XOR, celui qui
existe déjà, et **le batching n'en ajoute aucun** : l'indice d'échantillon
n'entre pas dans la graine par XOR, il **sélectionne** une entrée du tableau.
La graine par échantillon est bit-à-bit celle d'avant. C'est précisément
l'invariant qui rend le §5 (trajectoires appariées) exploitable.

Un test dédié vérifiera cette identité : `B` exécutions séquentielles du
prépass contre une exécution batchée, comparaison **bit à bit** de
`model_input` et de `target_noise`.

### 3.3 GroupNorm — les statistiques restent par échantillon

Non négociable : normaliser à travers le batch changerait le modèle, pas son
implémentation. La règle est mécanique dans la représentation retenue :

- le forward dispatche `B × num_groups` workgroups ;
  `sample = wid.x / num_groups`, `group = wid.x % num_groups` ;
- chaque workgroup ne lit que la tranche `[sample*len, (sample+1)*len)` ;
- le tampon `stats` passe à `B × num_groups × 4` f32 : `mean`, `inv_std`,
  `sum_dxhat`, `sum_dxhat_xhat` **par (échantillon, groupe)**.

Seuls `grad_gamma` / `grad_beta` — qui sont des *paramètres*, pas des
statistiques — réduisent sur l'axe batch (§3.1).

Un test le garde : deux échantillons de statistiques très différentes dans un
même batch doivent produire la même sortie que passés séparément.

### 3.4 La loss a besoin de sa longueur par échantillon

`loss.wgsl` calcule `grad_output[i] = 2·(pred−target)/N` avec
`N = arrayLength(&model_result)`. Tampon batché, `arrayLength` vaudrait `B·N` et
diviserait le gradient par `B` de trop.

`LossType` possède déjà un `LossUniform` (`dim_input`, `dim_output`) qui n'était
lié à rien. Il devient le **binding [4]**, ajouté **en dernier** pour ne
déplacer aucun des indices 0–3 auxquels `model.rs` adresse `forward[1]`
(`target`), `forward[2]` (`loss_terms`) et `forward[3]` (`grad_output`).
`N = dim_input.length()`, indépendant du batch. La formule est inchangée.

### 3.5 La loss rapportée doit rester le même nombre

`train_step_report_batch` renvoie aujourd'hui la loss du **dernier** échantillon
du batch (`is_last && report_last_loss`). On garde cette définition à
l'identique — sinon la comparaison de trajectoires du §5 compare deux grandeurs
différentes. La lecture se fait donc sur la tranche `[(B−1)·N, B·N)` de
`loss_terms`.

Et comme le forward ne contient **aucune** réduction sur l'axe batch, cette
valeur doit être **bit à bit** celle de l'ancien chemin. C'est une assertion
forte, et c'est le test le plus discriminant du lot.

### 3.6 Les copies du dataset — un piège d'ordonnancement

`GpuDataset::copy_sample_to` appelle `ensure_chunk_loaded`, qui fait un
`queue.write_buffer` sur le tampon de chunk. Encoder les `B` copies dans **un
seul** encodeur serait faux : `queue.write_buffer` est appliqué *avant* les
command buffers soumis ensuite, donc deux chargements de chunks différents dans
le même encodeur feraient lire à la première copie les données de la seconde.

Un `copy_samples_to(indices, destination)` groupe donc les indices **par chunk**
et soumet un command buffer par chunk résident. Sur un dataset qui tient dans un
chunk (le cas courant) c'est un seul groupe, une seule soumission.

### 3.7 Inférence, Perpetual, sonde : batch = 1, même chemin

Le chemin d'inférence ne change pas de forme — il emprunte le **même** code avec
`B = 1`. `Model<Infer>` construit ses tampons à `batch = 1`, donc
`arrayLength / per == 1`, `sample == 0`, et chaque kernel retrouve exactement
son dispatch d'avant. `PerpetualDrift`, `sample_diffusion`, `reverse_step` et
`compose_diffusion_input` ne voient rien.

Cas particulier : `predict()` sur un modèle **d'entraînement** (c'est ce que
fait `probe_diffusion`). Les tampons y sont dimensionnés pour `B`, mais
`predict` n'écrit que la première tranche et ne lit que la première. Le dispatch
forward est donc paramétré par le batch **au moment de l'encodage**
(`encode_pass_with_batch(encoder, 1)`) : la sonde ne paie pas `B ×` le travail
pour une seule image, et la troncature est correcte parce que `sample = idx/per`
donne 0 sur toute la plage dispatchée.

### 3.8 Changer la taille de batch en cours de run

Le TUI le permet (`main.rs` → `Model::set_batch_size`). Comme les tampons sont
dimensionnés à la construction, le changement impose une reconstruction. Elle
est faite **en préservant l'état** : `checkpoint_bytes()` → `build()` →
`load_checkpoint_bytes()`, ce qui restaure poids, biais, moments Adam **et** le
compteur `optimizer_step` (la correction de biais d'Adam en dépend). Opération
rare — une frappe utilisateur — et explicite.

## 4. Le graphe après changement

Par pas d'entraînement, `B = 16` :

| | avant | après |
|---|---:|---:|
| `CommandEncoder` | 18 | 2 (copies dataset + graphe) |
| `queue.submit` | 18 | 2 |
| dispatches | 16 × (graphe) + 2 | 1 × (graphe) + 1 |
| workgroups du forward `conv4` | 16 | 256 |
| positions sommées par `grad_weights` | 1024, ×16 fois | 16384, 1 fois |

## 5. Plan de preuve

Dans l'ordre, du plus mécanique au plus coûteux :

1. **Prépass bit à bit.** `B` exécutions séquentielles contre une batchée :
   `model_input` et `target_noise` identiques au bit près (§3.2).
2. **Forward bit à bit.** Le forward n'a aucune réduction sur l'axe batch : la
   sortie de l'échantillon `i` d'un batch de `B` doit être **exactement** celle
   d'une exécution seule. Comparaison sur `to_bits()`, comme
   `PERF_CONVOLUTION.md` §4.1 — un seuil laisserait passer un décalage d'offset.
3. **Gradients : batché vs séquentiel, arbitrés par un oracle f64.** Le chemin
   séquentiel est conservé comme **référence de test** (`train_step_sequential`,
   `#[cfg(test)]`-visible), donc la comparaison porte sur du code réellement
   exécutable, pas sur une fixture figée. Seuil relatif 1e-4, plus la clause
   `erreur(batché) ≤ 1,5 × erreur(séquentiel)`.
4. **Isolation du batch.** Un batch de `B` échantillons où seul l'échantillon
   `j` est non nul doit donner exactement les gradients d'un batch de 1
   contenant `j` — aucune fuite entre tranches. C'est le test qui attrape les
   offsets manqués, classe de bug numéro un de ce refactor.
5. **GroupNorm ne normalise pas à travers le batch** (§3.3).
6. **Différences finies** : les trois gradient checks d'`audit_tests.rs` passent
   inchangés. Interdiction de les affaiblir.
7. **Validation par mutation** des tests neufs : au moins un offset de tranche
   supprimé, une réduction batch tronquée à `B−1`, une graine partagée entre
   échantillons — chacune doit faire tomber un test nommé.
8. **Run apparié 600 pas** (`Greyscale_Diffusion_L`, Adam) ancien vs nouveau :
   trajectoire de `train_loss` et loss par tranche de `t`. Tolérance justifiée
   par l'ordre de sommation (§3.1), pas choisie après coup.

## 6. Plan de mesure

Protocole de `PERF_GROUP_NORM.md` / `PERF_CONVOLUTION.md` §5.1 : appariement,
entrelacement dans un seul process, **minimum** sur N rondes, échauffement de
tous les pipelines, contrôle nul (deux copies du même binaire).

- ms/pas avant/après à `B = 16` ;
- **échelle en batch** : 1 / 4 / 16 / 64. C'est la mesure qui décide si la
  thèse du §7 de `PERF_CONVOLUTION.md` est juste — le gain doit **croître**
  avec le batch. À `B = 1` les deux chemins doivent être équivalents (un seul
  échantillon, un seul submit des deux côtés) : c'est le contrôle interne.
- re-balayage de `reduction_lanes` avec l'axe batch (§3.1).

Contention GPU : un run couleur occupe la machine jusqu'à ~14h10. Toute la
correction est faite d'abord ; les chronos définitifs sont pris sur GPU libre,
et la contention est notée sinon — pas de chiffre publié sans son contrôle nul.

## 7. Ce qui n'est délibérément pas fait

- Aucune formule de couche n'est touchée. Aucune sémantique de padding, aucune
  définition de gradient, aucun schedule.
- Pas de re-tuning de kernel dans cette mission (blocking, tuiles, tailles de
  workgroup). Le §7 de `PERF_CONVOLUTION.md` demande de lever le goulot
  structurel **avant** d'affiner les kernels ; affiner en même temps rendrait
  les deux effets inséparables. Le batching rend ces pistes rentables — elles
  restent pour la mission d'après.
- Pas de fusion de dispatches entre couches (un `submit` par pas suffit à
  répondre à la question posée).
