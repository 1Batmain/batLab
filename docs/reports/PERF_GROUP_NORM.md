# Optimisation de `group_norm` — rapport de mission

Branche `perf-group-norm`. Répond au **finding #5** de `AUDIT_TRAINING.md`
(« `group_norm` recalcule moyenne et variance par élément »).

**Résumé** : les statistiques de groupe sont désormais calculées une seule fois
par groupe, par réduction en mémoire partagée, au lieu d'être recalculées
intégralement dans chaque invocation. Les couches `GroupNorm` isolées sont
**10,6× plus rapides**, le pas d'entraînement complet **1,63× plus rapide**
(498,5 → 305,8 ms/pas à batch 16). Les maths sont inchangées : sur 600 pas
d'entraînement réel, la trajectoire de loss est **identique à toutes les
décimales affichées**, et l'écart maximal sur l'ensemble des métriques JSONL
est de 3,8e-6 en absolu.

---

## 1. Ancienne approche

`compute_group_mean()` et `compute_group_inv_std()` parcouraient **tout le
groupe** et étaient appelées **à chaque invocation**, soit une fois par élément
du tenseur — un coût quadratique en la taille du groupe.

```wgsl
// group_norm.wgsl, avant
@compute @workgroup_size(64)
fn group_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    ...
    let mean    = compute_group_mean(group);      // boucle sur les 4096 éléments
    let inv_std = compute_group_inv_std(group, mean); // re-boucle sur les 4096
    output[index] = ((input[index] - mean) * inv_std) * gamma[channel] + beta[channel];
}
```

Sur la couche 32×32×32 à 8 groupes de `Greyscale_Diffusion` (`spatial_len` =
1024, `channels_per_group` = 4, `group_len` = 4096) : 2 × 4096 = 8192 lectures
par élément, 32 768 éléments → **2,7e8 lectures** pour une passe qui en demande
32 768. Le backward faisait pire : `group_norm_back_input` refaisait ces deux
réductions **plus** une troisième boucle sur le groupe (`sum_dxhat`,
`sum_dxhat_xhat`), et `group_norm_back_gamma` refaisait les deux réductions par
canal.

## 2. Nouvelle approche

### Forward — un workgroup par groupe, kernel unique

`get_forward_workgroup_count()` (nouvelle méthode du trait `LayerType`, dont le
défaut reproduit la convention élémentaire existante) fait dispatcher
`num_groups` workgroups de 256 threads au lieu de `length()/64`. Chaque
workgroup traite un groupe entier en trois phases, séparées par des réductions
en arbre sur `var<workgroup> partial: array<f32, 256>` :

1. somme des éléments du groupe → `mean` ;
2. somme des écarts centrés au carré → `variance`, `inv_std` ;
3. écriture de `(x - mean) * inv_std * gamma + beta`.

Les threads parcourent le groupe en *strides* de 256, donc la taille du groupe
n'a pas besoin d'être un multiple de la taille du workgroup.

Le calcul de la variance conserve délibérément la **forme centrée**
`Σ(x - mean)²` de l'original, et non le raccourci `E[x²] - E[x]²`, qui aurait
changé la stabilité numérique.

### Backward — deux passes de réduction préalables

`Layer::encode_back_pass` encode déjà les sous-passes en séquence avec des
barrières implicites entre elles ; il suffisait d'en ajouter deux devant. Le
backward passe de 3 à 5 dispatches :

| # | entry point | dispatch | rôle |
|---|---|---|---|
| 1 | `group_norm_stats` | 1 wg / groupe | `mean`, `inv_std` |
| 2 | `group_norm_grad_stats` | 1 wg / groupe | `sum_dxhat`, `sum_dxhat_xhat` |
| 3 | `group_norm_back_input` | 1 thread / élément | `grad_input` — **O(1) par élément** |
| 4 | `group_norm_back_gamma` | 1 wg / canal | `grad_gamma` (réduction spatiale) |
| 5 | `group_norm_back_beta` | 1 wg / canal | `grad_beta` (réduction spatiale) |

Les résultats des passes 1–2 transitent par un buffer `stats` de 4 f32 par
groupe (`[mean, inv_std, sum_dxhat, sum_dxhat_xhat]`), **ajouté en dernière
position** (binding 7) du bind group backward pour ne perturber aucun des
indices auxquels les autres bindings sont adressés (`BackwardBufferSource`,
`get_back_grad_input_index`, `OptimizerBindings`).

`grad_gamma` / `grad_beta` gagnent au passage un vrai parallélisme : avant,
`dim_input.z` threads (16 ou 32) faisaient chacun une boucle séquentielle de
`spatial_len` itérations ; maintenant un workgroup de 256 threads par canal
réduit l'axe spatial en arbre. L'accumulation `+=` entre échantillons d'un
batch est préservée (seul le thread 0 écrit).

### Ce qui n'a pas changé

Aucune formule. `grad_input` reste
`inv_std / M * (M·dxhat - sum_dxhat - x̂·sum_dxhat_xhat)`, `grad_gamma` reste
`Σ_s grad_output · x̂`, etc. **Seul l'ordre de sommation flottant diffère** —
une réduction en arbre au lieu d'une somme séquentielle.

## 3. Preuves d'équivalence

Fichier : `bat_building/src/model/group_norm_equivalence_tests.rs` (5 tests).

### 3.1 Ancienne vs nouvelle implémentation, sur les mêmes buffers

Les shaders d'avant l'optimisation sont conservés **verbatim** comme fixtures de
test (`bat_building/src/model/shader/legacy/*_naive.wgsl`, extraits de
`bc7bff7`). Chaque test construit un vrai modèle avec les nouveaux shaders,
l'exécute, puis redispatche les kernels legacy sur **exactement le même bind
group** — mêmes buffers, mêmes entrées, mêmes gamma/beta — et compare.

Six formes couvertes, dont les trois `GroupNorm` de `Greyscale_Diffusion`, un
groupe plus petit que le workgroup (`group_len` = 64), un groupe unique avec une
étendue spatiale non puissance de deux, et un canal par groupe.

| grandeur | écart relatif max mesuré (nouveau vs naïf) | seuil du test |
|---|---:|---:|
| forward `output` | < 1e-5 | 1e-5 |
| backward `grad_gamma` | 1,55e-6 | 1e-4 |
| backward `grad_beta` | 1,64e-6 | 1e-4 |
| backward `grad_input` | 1,59e-5 | 1e-4 (voir ci-dessous) |

### 3.2 Oracle f64 — qui a raison quand les deux f32 diffèrent ?

`grad_input` évalue `M·dxhat - sum_dxhat - x̂·sum_dxhat_xhat` : une soustraction
de termes individuellement ~M fois plus grands que le résultat. Toute
réassociation de `sum_dxhat` (une somme de 4096 f32) est donc **amplifiée par
l'annulation catastrophique** — l'accord ancien/nouveau ne peut pas descendre au
niveau de l'arrondi de la sortie, et un simple seuil serré aurait été un test
mal posé plutôt qu'une preuve.

Un oracle f64 indépendant (`reference_backward` : forward + gradient MSE +
backward en double précision) tranche. Le test exige :

- l'écart nouveau ↔ oracle < 1e-4 (précision absolue) ;
- **`erreur(nouveau) ≤ 1,5 × erreur(naïf)`** face à l'oracle — c'est-à-dire que
  la réduction en arbre n'est **pas moins précise** que la somme séquentielle
  qu'elle remplace.

Le verdict de l'oracle sur les deux plus grosses couches (`grad_input`) :

| couche | nouveau vs f64 | naïf vs f64 | écart nouveau↔naïf |
|---|---:|---:|---:|
| 32×32×32 / 8 | **1,51e-6** | 1,71e-5 | 1,59e-5 |
| 16×16×32 / 8 | **1,81e-6** | 1,17e-5 | 1,19e-5 |
| 32×32×16 / 4 | **1,13e-6** | 3,04e-6 | 2,73e-6 |

Autrement dit, la quasi-totalité de l'écart entre les deux versions est
**l'erreur de l'ancienne** : sur la couche 32×32×32, la nouvelle réduction est
**11× plus proche de la vérité f64**. C'est la propriété attendue d'une
réduction en arbre face à une somme séquentielle de 4096 termes f32 — mais elle
est ici mesurée, pas supposée. Le même classement vaut sur `grad_gamma`
(9,98e-8 vs 1,64e-6) et `grad_beta` (8,67e-8 vs 1,69e-6).

### 3.3 Différences finies

Oracle qui ne connaît ni l'une ni l'autre implémentation :

- `gamma_beta_gradients_match_finite_differences` — gradients analytiques vs
  différences centrées de la loss, erreur relative < 5e-2 ;
- `input_gradients_match_finite_differences` — idem sur `grad_input`.

**Note sur le plancher de quantification** : la loss revient du GPU en un seul
f32, donc `lp - lm` est quantifié à `ulp(loss)` et la différence finie ne peut
rien résoudre sous `ulp(loss) / (2·eps)`. Avec `loss ≈ 2,72` et `eps = 1e-3`,
ce plancher vaut 1,19e-4 — la première version de ce test échouait sur un
élément dont le gradient analytique était −3,5e-4, soit **3 ulps de signal**
(`lp - lm` valait exactement 1 ulp) : elle mesurait l'arrondi du readback, pas
le gradient. Le gradient en question était correct — il coïncide avec l'oracle
f64. Le test sonde maintenant avec `eps = 1e-2` (plancher ramené à 1,62e-5) et
**exclut explicitement** les éléments sous 20× ce plancher, avec un garde-fou
qui le fait échouer si le filtre avalait plus de la moitié des sondes. Rien n'a
été affaibli : le seuil d'erreur reste 5e-2, **25 des 26 sondes** sont retenues,
et l'erreur max mesurée est 2,47e-2.

### 3.4 Propriété définitionnelle

`forward_normalises_each_group` : avec γ = 1 et β = 0, chaque groupe de la sortie
a une moyenne nulle (|µ| < 1e-4) et une variance unitaire (|σ² − 1| < 1e-3),
vérifié en f64 sur le CPU. Indépendant des deux implémentations.

### 3.5 Les tests ne sont pas vacuous — vérifié par mutation

Trois mutations délibérées introduites dans les nouveaux shaders, puis
restaurées :

| mutation | tests qui échouent |
|---|---|
| forward : canal mal mappé en phase 3 (`i % cpg` → `0`) | 4 / 5 |
| backward : terme `x̂ · sum_dxhat_xhat` retiré de `grad_input` | 2 / 5 (dont les différences finies) |
| backward : `grad_gamma` biaisé de +0,1 % | 1 / 5 |

La troisième illustre la complémentarité des deux oracles : un biais de 1e-3 est
sous la tolérance des différences finies (5e-2, limitée par la précision de la
loss) mais très au-dessus de celle de la comparaison ancien/nouveau et de
l'oracle f64 (1e-4). Aucun des deux oracles seul ne suffirait.

### 3.6 Suite complète

```
cargo test --workspace
   52 passed (bat_building) + 6 passed (main) — 0 failed
```

Les gradient checks par différences finies de `audit_tests.rs`
(`conv_weight_gradients_match_finite_differences`,
`..._same_padding`, `stacked_same_padding_conv_grad_input_is_consistent`)
passent — aucun test existant n'a été touché ni affaibli.

## 4. Benchmarks

### 4.1 Couches isolées

Les deux implémentations chronométrées dans le **même processus** et sur les
**mêmes buffers**, 200 dispatches par mesure après échauffement :

```
cargo test --release -p bat_building --lib bench_group_norm_isolated -- --ignored --nocapture
```

| couche | passe | naïf (ms) | nouveau (ms) | speedup |
|---|---|---:|---:|---:|
| 32×32×16 / 4 groupes | forward  | 1,2660 | 0,1026 | **12,3×** |
| 32×32×16 / 4 groupes | backward | 2,0064 | 0,4085 | **4,9×** |
| 16×16×32 / 8 groupes | forward  | 0,2357 | 0,0370 | **6,4×** |
| 16×16×32 / 8 groupes | backward | 0,3563 | 0,1550 | **2,3×** |
| 32×32×32 / 8 groupes | forward  | 2,1840 | 0,0320 | **68,3×** |
| 32×32×32 / 8 groupes | backward | 3,4265 | 0,1560 | **22,0×** |
| **total / échantillon** | | **9,4749** | **0,8910** | **10,6×** |

### 4.2 Pas d'entraînement complet

`Greyscale_Diffusion`, batch 16, `--headless-train`, binaire release, mesuré
entre l'horodatage du pas 0 et celui du pas 99 :

| | ms / pas | total 600 pas |
|---|---:|---:|
| naïf | **498,5** | 5 min 04 s |
| nouveau | **305,8** | 3 min 07 s |
| **speedup** | **1,63×** | **1,63×** |

Les deux séries de mesures (99 pas chronométrés finement, 600 pas au temps
mural) concordent à 1 % près.

### 4.3 Écart entre les deux mesures — pourquoi

L'économie prédite par le banc isolé est 8,58 ms × 16 échantillons =
137 ms/pas ; l'économie observée est 192,7 ms/pas. Le banc isolé est donc un
**minorant**, pour deux raisons : il enchaîne 200 dispatches dans un seul
encoder (pipelining GPU idéal, sans les barrières inter-couches du vrai
graphe), et le pas d'entraînement contient des forwards supplémentaires que le
banc ne compte pas — les sondes `probe_diffusion` et les trajectoires
`sample_diffusion` de l'instrumentation JSONL, qui traversent elles aussi les
couches `GroupNorm`.

### 4.4 Ce qui reste

`GroupNorm` pèse désormais ~0,89 ms × 16 = **14 ms sur les ~306 ms du pas**,
soit 4,6 %. Le goulot est passé ailleurs (convolution / upsample_conv), hors
périmètre de cette mission.

Une marge subsiste néanmoins sur le forward : avec un workgroup par groupe, la
couche 32×32×16 à **4 groupes** ne lance que 4 workgroups et sous-occupe le GPU
— elle est plus lente (0,1026 ms) que la couche 32×32×32 qui fait pourtant deux
fois plus de travail mais dispose de 8 groupes (0,0320 ms). Un découpage en deux
dispatches (sommes partielles par tuile, puis combinaison) lèverait cette
limite. Gain plafond : ~1 ms/pas sur 306. Non poursuivi.

## 5. Validation en conditions réelles

Deux runs de **600 pas** (batch 16, lr 1e-3, `cifar10_grey.batraw`), ancienne
puis nouvelle implémentation, tout le reste identique.

### Trajectoire de loss — identique à toutes les décimales affichées

```
step   0  1.121940 | 1.121940      step 300  0.105136 | 0.105136
step  25  0.730809 | 0.730809      step 375  0.434410 | 0.434410
step  50  0.501290 | 0.501290      step 450  0.079534 | 0.079534
step 100  0.252944 | 0.252944      step 525  0.087508 | 0.087508
step 200  0.277523 | 0.277523      step 599  0.074075 | 0.074075
                                              (ancien | nouveau)
```

Les 25 points relevés coïncident. Sur les valeurs pleine précision du JSONL,
l'écart relatif max sur `train_loss` est de **7,1e-7**.

### Loss par tranche de t (`train_probe`)

C'est le diagnostic qui compte pour un modèle de diffusion (cf.
`INSIGHTS_TRAINING.md` : une loss batch qui décroît ne suffit pas). Sur 25
sondes × 4 tranches :

| pas | tranche | ancien | nouveau | écart rel. |
|---|---|---:|---:|---:|
| 0 | t bas | 1,13167393 | 1,13167381 | 1,1e-7 |
| 0 | t haut | 1,18465436 | 1,18465424 | 1,0e-7 |
| 300 | t bas | 0,71501011 | 0,71501017 | 8,3e-8 |
| 300 | t haut | 0,10651555 | 0,10651554 | 7,0e-8 |
| 599 | t bas | 0,68143141 | 0,68143141 | 0 |
| 599 | t haut | 0,07151438 | 0,07151438 | 0 |

**Écart relatif max sur l'ensemble des tranches : 2,9e-7.** Le profil attendu
(loss élevée à t bas, faible à t haut — le modèle utilise bien t) est identique
dans les deux runs.

### Toutes métriques confondues

Sur les 821 enregistrements JSONL (`train_loss`, `train_probe`, `sample`,
`denoise_step`) :

- écart **absolu** max, tous champs : **3,8e-6** ;
- écart relatif max sur les champs de magnitude > 1e-3 : **1,5e-4** (divergence
  d'arrondi accumulée sur 600 pas) ;
- aucune valeur non finie, mêmes nombres et mêmes types d'enregistrements.

---

## Fichiers modifiés

| fichier | changement |
|---|---|
| `bat_building/src/model/shader/group_norm.wgsl` | réécrit — réduction par workgroup |
| `bat_building/src/model/shader/back_group_norm.wgsl` | réécrit — 5 passes, buffer `stats` |
| `bat_building/src/model/layer_types/group_norm.rs` | dispatches, entry points, buffer `stats` |
| `bat_building/src/model/layer_types/mod.rs` | + `LayerType::get_forward_workgroup_count()` |
| `bat_building/src/model/layer.rs` | `Layer::new` utilise la nouvelle méthode |
| `bat_building/src/model/mod.rs` | + module de tests |
| `bat_building/src/model/group_norm_equivalence_tests.rs` | **nouveau** — 5 tests + banc |
| `bat_building/src/model/shader/legacy/*_naive.wgsl` | **nouveau** — fixtures de test |
