# Audit du pipeline d'entraînement — batLab

Branche : `audit-training`. Livrable : diagnostic uniquement. Aucune modification
du code de production (seul ajout : un module `#[cfg(test)]`).

**Tests diagnostiques** : `bat_building/src/model/audit_tests.rs`, activé par
`#[cfg(test)] mod audit_tests;` en fin de `bat_building/src/model/mod.rs`.
Ces tests sont écrits pour **échouer** contre l'implémentation actuelle —
l'échec *est* le finding. Deux d'entre eux passent volontairement : ce sont des
contrôles négatifs qui délimitent la zone saine.

```
cargo test -p bat_building audit_tests
```

| Test | Attendu | Rôle |
|---|---|---|
| `same_padding_conv_is_centered_and_zero_padded` | ÉCHOUE | finding #1 (forward) |
| `stacked_same_padding_conv_grad_input_is_consistent` | ÉCHOUE | finding #1 (backward) |
| `diffusion_timestep_is_decorrelated_from_sample_index` | ÉCHOUE | finding #2 |
| `schedule_reaches_approximately_pure_noise_at_final_step` | ÉCHOUE | finding #3 |
| `conv_weight_gradients_match_finite_differences` | PASSE | contrôle négatif |
| `conv_weight_gradients_match_finite_differences_same_padding` | PASSE | contrôle négatif |

---

## Finding #1 — `PaddingMode::Same` : convolution décalée, bords écrasés, gradients faux en profondeur — **CRITIQUE**

**Fichiers** : `bat_building/src/model/shader/convolution.wgsl:19,49-51`,
`bat_building/src/model/shader/back_convolution.wgsl:63-71,105-107`,
`bat_building/src/model/layer_types/convolution.rs:172,185`.

### Mécanisme

`ConvolutionType::set_dim_output()` calcule correctement la dimension de sortie
du mode `Same` (`dim_input.x.div_ceil(stride)`), mais le shader **n'applique
jamais l'offset de padding correspondant**. Le commentaire du shader assume
explicitement — et à tort — que le problème se règle au niveau des dimensions :

```wgsl
// convolution.wgsl:19
padding_mode: u32, // 0 = Valid, 1 = Same (unused in kernel — dims already computed)
...
// convolution.wgsl:49-51
let iy    = oy * s + ky;      // ← devrait être  oy*s + ky - pad_y
let ix    = ox * s + kx;      // ← devrait être  ox*s + kx - pad_x
let in_i  = iy * IW * IC + ix * IC + kz;
```

Deux conséquences :

1. **Décalage spatial.** Pour un noyau 3×3 stride 1, la sortie `(oy,ox)` agrège
   les entrées `(oy..oy+2, ox..ox+2)` au lieu de `(oy-1..oy+1, ox-1..ox+1)`.
   La carte de features glisse de `(KH/2, KW/2)` **à chaque couche `Same`**.
2. **Débordement silencieux.** `iy` monte jusqu'à `IH + KH - 2`, hors bornes.
   WGSL applique la robustesse d'accès : les lectures sont *clampées* sur le
   dernier élément au lieu de renvoyer 0. Les bords bas/droit sont remplis par
   répétition du dernier pixel — ni zero-padding, ni edge-padding cohérent.

`upsample_conv.wgsl:34-46,72-73` **implémente correctement** `pad_y()`/`pad_x()`.
Les deux couches convolutives divergent sur le même concept ; seule
`Convolution` est cassée.

### Preuves

**(a) Forward.** `same_padding_conv_is_centered_and_zero_padded` — conv 3×3
`Same` stride 1, entrée 4×4 = `1..16`, noyau dont **seul le tap central vaut 1**
(une conv `Same` correcte est alors l'identité) :

```
attendu : [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]   (identité)
obtenu  : [ 6, 7, 8, 9,10,11,12,13,14,15,16,16,16,16,16,16]
```

Décalage de +5 (une ligne + une colonne) puis saturation à 16 : décalage **et**
clamp hors-bornes confirmés sur GPU réel.

**(b) Backward — le point le plus grave.** Deux contrôles délimitent le
problème :

- Sur **une seule** couche, `conv_weight_gradients_match_finite_differences_same_padding`
  **passe**. Le backward est donc le gradient *exact* du forward (faux), y
  compris son clamping. Pris isolément, l'opérateur est auto-cohérent.
- Sur **deux couches empilées**, `stacked_same_padding_conv_grad_input_is_consistent`
  **échoue lourdement** :

```
analytic (GPU backward): [-0.1078, -0.1163, -0.1238, -0.1422, ...]
numeric  (central diff): [-0.5764, -0.5890, -0.6004, -0.6283, ...]
worst relative error: 0.8129
```

Le gradient reçu par la première couche est **~5× trop petit** (81 % d'erreur
relative). Raison : sous clamping, plusieurs positions de sortie lisent le
*même* élément d'entrée clampé ; un `grad_input` correct devrait **sommer** ces
contributions, or `conv_back_input` (`back_convolution.wgsl:63-71`) ne modélise
pas du tout le clamping. L'erreur se compose avec la profondeur.

### Impact entraînement

C'est la cause racine la plus probable du symptôme rapporté. Dans un U-Net,
`Concat` exige des dimensions spatiales identiques entre branche descendante et
skip : `Same` est donc quasi obligatoire, et ce bug est presque certainement
actif sur les configurations réelles. Résultat : le modèle apprend une cible
spatialement décalée, avec des bords corrompus, **et** les couches profondes
reçoivent un gradient atténué d'un facteur qui croît avec la profondeur. La
perte descend (le réseau minimise honnêtement un mauvais opérateur) mais les
échantillons ne peuvent pas converger.

### Fix proposé

Aligner `convolution.wgsl` sur `upsample_conv.wgsl` :

```wgsl
fn pad_y() -> i32 { if layer_spec.padding_mode == 1u { return i32(layer_spec.dim_kernel.x / 2u); } return 0; }
fn pad_x() -> i32 { if layer_spec.padding_mode == 1u { return i32(layer_spec.dim_kernel.y / 2u); } return 0; }

let iy = i32(oy * s) + i32(ky) - pad_y();
let ix = i32(ox * s) + i32(kx) - pad_x();
if iy < 0 || iy >= i32(IH) || ix < 0 || ix >= i32(IW) { continue; }  // zero-padding explicite
```

Le même offset **et le même test de bornes** doivent être répliqués à l'identique
dans `back_convolution.wgsl` :
- `conv_back_input` : `oy = (iy + pad_y - ky) / s`, avec tests de divisibilité et
  de bornes ;
- `conv_back_weights` : `iy = oy*s + ky - pad_y`, en sautant les positions hors
  bornes (contribution nulle au gradient).

Corriger le forward seul rendrait le gradient faux. Les deux tests empilés
ci-dessus servent de garde-fou après correction.

---

## Finding #2 — Timestep de diffusion entièrement corrélé à l'échantillon — **MAJEUR**

**Fichier** : `bat_building/src/model/training/diffusion.rs:172-173`.

```rust
let sample_index   = (step * batch_size + batch_offset) % sample_count;
let diffusion_step = diffusion_step_for(step, batch_size, batch_offset, schedule_len);
//                 = (step * batch_size + batch_offset) % schedule_len
```

### Mécanisme

Les deux quantités dérivent du **même compteur linéaire**
`n = step*batch_size + batch_offset`. Le timestep n'est pas échantillonné : il
est une fonction déterministe de l'indice d'image. L'image `i` reçoit toujours
le timestep `i % 256`, à chaque époque, indéfiniment.

Le nombre de timesteps distincts qu'une image donnée verra sur *toute* la durée
de l'entraînement vaut `gcd(sample_count, schedule_len)`. Pour CIFAR-10
(50 000 images) et 256 pas : **`gcd(50000, 256) = 16`**. Mesuré sur 20 époques
simulées, l'échantillon 0 ne voit que
`{0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240}` —
**16 timesteps sur 256, soit 6 % du schedule**.

Le fix de la PR #6 (`57f31c6`) a bien supprimé l'aliasing *intra-batch* (tous
les éléments d'un batch partageaient un timestep) et le test
`diffusion_step_progression_does_not_alias_single_batch` couvre ce cas précis.
Mais le fix est **incomplet** : il n'a pas rompu la corrélation entre le
timestep et l'échantillon, qui est le problème de fond. DDPM exige
`t ~ U{0, T-1}` **indépendant** de `x_0`.

Corollaire : `sample_index` étant strictement séquentiel, **le dataset n'est
jamais mélangé** — l'ordre de présentation est identique à chaque époque.

### Impact

La perte optimisée n'est pas `E_{x,t,ε}[·]` mais une somme sur un couplage
dégénéré `(x_i, t_{i mod 256})`. Le réseau peut mémoriser la corrélation
« ce contenu d'image ⇒ ce niveau de bruit » au lieu d'apprendre le débruitage
conditionné au timestep. À l'échantillonnage, où l'on parcourt les 256 timesteps
sur une même image, cette hypothèse apprise est violée à chaque pas.

### Fix proposé

Découpler timestep et compteur : le tirer d'un hash du seed.

```rust
let diffusion_step = if schedule_len == 0 { 0 } else {
    (fold_seed(seed ^ 0xD1B5_4A32 ^ ((batch_offset as u64) << 16)) as usize) % schedule_len
};
```

Ajouter un mélange du dataset par époque (permutation dérivée du seed d'époque).
Conserver le test anti-aliasing de la PR #6 en le renforçant : vérifier que les
timesteps d'un batch sont distincts **et** qu'un échantillon donné couvre
uniformément `[0, T)` sur N époques.

---

## Finding #3 — Le schedule n'atteint jamais le bruit pur — **MAJEUR**

**Fichier** : `bat_building/src/model/training/schedule.rs:11-39`, appelé avec
`num_steps = 256`, `beta ∈ [1e-4, 0.02]`.

### Mécanisme

Les bornes `beta_start = 1e-4`, `beta_end = 0.02` sont celles du papier DDPM,
mais y sont calibrées pour **T = 1000**. Le projet les réutilise telles quelles
à **T = 256**, sans recalibrage :

| T | `alpha_bar_T` | `sqrt(alpha_bar_T)` | signal résiduel en `x_T` |
|---|---|---|---|
| 1000 (DDPM) | 4.04e-5 | 0.0064 | 0.6 % |
| **256 (ici)** | **0.0750** | **0.2739** | **27.4 %** |

À l'issue du forward process, `x_T` conserve **27 % du signal d'origine** :
`q(x_T)` est très loin de `N(0, I)`.

### Impact

L'échantillonnage part de bruit gaussien pur, alors que le modèle n'a jamais vu,
à `t = T`, autre chose qu'une image encore largement reconnaissable. Décalage
train/inférence sur le pas le plus critique de la chaîne, qui se propage à toute
la trajectoire de débruitage.

### Fix proposé

Au choix : recalibrer `beta_end` pour T = 256 afin d'obtenir `alpha_bar_T ≈ 1e-4`
(≈ `beta_end = 0.06` en linéaire) ; passer à un cosine schedule
(Nichol & Dhariwal), plus robuste aux petits T ; ou revenir à T = 1000.

Ajouter une assertion de construction `alpha_bar(len-1) < 1e-3` pour que toute
future combinaison (T, betas) incohérente échoue bruyamment.

---

## Finding #4 — Données en [0,1] alors que la diffusion suppose [-1,1] — **MAJEUR**

**Fichiers** : `main/src/main.rs:908,925` (`pixel as f32 / 255.0`),
`main/src/main.rs:1098` (`to_u8` : `clamp(0,1) * 255`), format `.batraw`
documenté « normalised [0, 1] » (`main/src/main.rs:590`).

### Mécanisme

Les pixels sont normalisés dans `[0, 1]`, de moyenne ≈ 0.5. Le forward process
`x_t = sqrt(ᾱ)·x_0 + sqrt(1-ᾱ)·ε` suppose `x_0` **centré** — c'est précisément
pourquoi DDPM normalise en `[-1, 1]`. Avec `x_0 ∈ [0,1]`, `x_t` conserve un
biais de moyenne `0.5·sqrt(ᾱ_t)` à tous les timesteps.

Ce finding **compose** avec le #3 : à `t = T`, `E[x_T] ≈ 0.5 × 0.274 ≈ 0.137`
avec une variance réduite, tandis que l'échantillonneur démarre de `N(0,1)`
(moyenne 0, variance 1).

### Impact

Le réseau apprend à prédire `ε` à partir d'entrées systématiquement décalées vers
le positif. À l'inférence, l'entrée initiale est hors distribution : les premiers
pas de débruitage partent dans une direction arbitraire et l'erreur s'accumule
le long de la chaîne.

### Fix proposé

Normaliser en `[-1, 1]` au chargement (`v / 127.5 - 1.0`) et dénormaliser
symétriquement à l'écriture d'image (`to_u8` en `main.rs:1098` → `(v + 1) * 127.5`).
Le format `.batraw` existant devient incompatible : bumper le magic header ou
convertir à la lecture.

---

## Finding #5 — `group_norm` recalcule moyenne et variance par élément — **MAJEUR (performance)**

**Fichiers** : `bat_building/src/model/shader/group_norm.wgsl:22-48,57-58`,
`bat_building/src/model/shader/back_group_norm.wgsl:24-50,61-62,90-91`.

### Mécanisme

`compute_group_mean()` et `compute_group_inv_std()` parcourent **tout le groupe**
et sont appelées **à chaque invocation**, c'est-à-dire une fois par élément du
tenseur. Le coût est quadratique en la taille du groupe.

Pour une couche 32×32×128 à 8 groupes (`spatial_len = 1024`,
`channels_per_group = 16`) :

| | lectures |
|---|---|
| par invocation | 2 × 1024 × 16 = 32 768 |
| forward complet | 131 072 × 32 768 = **4.29e9** |
| optimal (une passe + réduction) | ~131 072 |

Soit un facteur **32 768×**. Le backward est pire encore :
`group_norm_back_input` refait ces deux réductions *plus* une troisième boucle
sur le groupe.

### Impact

Correct numériquement, mais domine très probablement le temps de pas et donc le
débit d'entraînement. À l'échelle d'un U-Net comportant plusieurs `GroupNorm`,
c'est l'écart entre un entraînement exploitable et un entraînement inutilisable.

### Fix proposé

Précalculer `mean` et `inv_std` par groupe dans une passe de réduction séparée
(un workgroup par groupe, réduction en mémoire partagée), les stocker dans un
buffer `[num_groups]`, puis les lire en O(1) dans la passe de normalisation et
dans les trois passes backward. Le buffer de statistiques est également
réutilisable entre forward et backward sur un même pas.

---

## Finding #6 — L'échantillonnage multi-chemins part toujours du même latent — **MINEUR**

**Fichier** : `main/src/main.rs:998,1002`.

```rust
let base_latent = schedule.sample_noise(output_len, base_noise_seed);
for path_idx in 0..path_count {
    let mut latent = base_latent.clone();   // ← identique pour tous les chemins
```

`path_seed` n'influence que le bruit stochastique réinjecté par
`denoise_step_with_magnitude`. Les `N` chemins partent donc du **même** point de
départ et sont moyennés en fin de course (`main.rs:1022-1029`). Moyenner des
échantillons de diffusion produit mécaniquement une image floue — l'inverse de
l'effet recherché par un ensemble.

**Fix** : dériver le latent initial de `path_seed`
(`schedule.sample_noise(output_len, path_seed)`), et présenter les chemins comme
des échantillons distincts plutôt que de les moyenner ; ou retirer l'option.
Sans effet à `denoising_paths = 1` (le défaut).

---

## Finding #7 — La perte rapportée n'échantillonne qu'un couple (image, timestep) — **MINEUR**

**Fichiers** : `bat_building/src/model/training/diffusion.rs:187-192`,
`main/src/main.rs:271-282`.

Seul le **dernier** élément du batch produit une valeur de perte, et uniquement
tous les `LOSS_REPORT_INTERVAL_STEPS` pas. Or la MSE de diffusion varie de
plusieurs ordres de grandeur selon `t` (le bruit à prédire est trivial à petit
`t`, difficile à grand `t`). La courbe affichée mélange donc le signal de
convergence avec la variance du timestep tiré.

Combiné au finding #2, c'est pire : le timestep du dernier élément du batch est
lui-même déterministe, donc la courbe de perte suit un motif cyclique de
période 256 qui n'a rien à voir avec l'apprentissage.

**Fix** : accumuler `loss_terms` sur tout le batch (réduction GPU) et rapporter
la moyenne ; ou tenir une moyenne glissante par tranche de `t`.

---

## Zones vérifiées et jugées saines

Contrôles positifs — utile pour cadrer où *ne pas* chercher :

- **Accumulation de gradients sur le batch** (`model.rs:268-296`,
  `layer.rs:612-625`) : `begin_batch_accumulation` remet à zéro une fois,
  chaque backward accumule en `+=` (vérifié dans les six shaders `back_*.wgsl`),
  puis `finish_batch_accumulation` applique SGD avec `lr / batch_size`.
  Sémantiquement équivalent à un gradient moyenné. Correct.
- **Backward de la convolution en mode `Valid`** : validé par différences finies
  (`conv_weight_gradients_match_finite_differences`, erreur relative < 5e-2).
  La mécanique d'autodiff (loss → `grad_output` → `conv_back_weights` → buffer
  de gradient → SGD) est saine ; le problème du finding #1 est strictement
  l'indexation du padding.
- **Routage des gradients de skip connections** (`model.rs:171-194`) : les
  gradients `Concat` sont bien fusionnés (`sum.wgsl`) dans la couche qui a
  produit le skip, avec détection de doublon.
- **Embedding de timestep** : le shader (`diffusion_prepare.wgsl:41-55`) et la
  version CPU utilisée à l'échantillonnage (`schedule.rs:69-90`) produisent la
  même disposition entrelacée `[sin(f0), cos(f0), sin(f1), …]` et le même
  dénominateur pour toute valeur de `half`. Cohérents.
- **Disposition mémoire `compose_diffusion_input`** (`main.rs:931-958`) :
  identique à celle produite par `diffusion_prepare.wgsl`. Cohérente.
- **`group_norm`, divisibilité** : `set_dim_output` (`group_norm.rs:154-161`)
  rejette `channels % num_groups != 0`. Pas de lecture hors groupe.
- **Comptes de workgroups backward** : vérifiés couche par couche contre les
  gardes de borne des shaders. Cohérents.
- **Ordonnancement `queue.write_buffer` / `submit`** dans la boucle de batch
  (`diffusion.rs:171-199`) : chaque itération soumet son propre encodeur, les
  écritures en attente sont donc appliquées avant les commandes qui les lisent.
  Correct.

---

## Synthèse par sévérité

| # | Sévérité | Finding | Fichier principal |
|---|---|---|---|
| **1** | **CRITIQUE** | `PaddingMode::Same` : conv décalée de `(KH/2, KW/2)` par couche, bords clampés hors bornes, et `grad_input` faux (~5× trop petit sur 2 couches, erreur croissante avec la profondeur) | `shader/convolution.wgsl:49-51` + `shader/back_convolution.wgsl:63-71` |
| **2** | **MAJEUR** | Timestep de diffusion = fonction déterministe de l'indice d'image ; chaque image ne voit que `gcd(50000,256) = 16` des 256 timesteps. Dataset jamais mélangé. Fix PR #6 incomplet | `training/diffusion.rs:172-173` |
| **3** | **MAJEUR** | Betas DDPM (T=1000) réutilisés à T=256 → `ᾱ_T = 0.075`, 27 % de signal résiduel en `x_T` ; `q(x_T) ≠ N(0,I)` | `training/schedule.rs:11-39` |
| **4** | **MAJEUR** | Données normalisées `[0,1]` au lieu de `[-1,1]` ; `x_0` non centré, biais à tous les `t`, décalage train/inférence (compose avec #3) | `main/src/main.rs:908,925,1098` |
| **5** | **MAJEUR** (perf) | `group_norm` recalcule moyenne et variance par élément → ~32 768× le travail nécessaire sur une couche 32×32×128 | `shader/group_norm.wgsl:22-48` |
| 6 | mineur | Échantillonnage multi-chemins : tous les chemins partent du même latent puis sont moyennés → flou | `main/src/main.rs:998-1002` |
| 7 | mineur | Perte rapportée sur un seul couple (image, timestep), lui-même déterministe → courbe cyclique non informative | `training/diffusion.rs:187-192` |

### Ordre de correction recommandé

1. **#1** — seul finding qui casse la sémantique même de l'opérateur convolutif
   et corrompt les gradients en profondeur. Rien d'autre ne peut être évalué
   tant qu'il est présent. Corriger forward **et** backward ensemble.
2. **#4** puis **#3** — peu coûteux, et ils conditionnent la validité de tout ce
   que produit l'échantillonnage. À faire avant toute mesure de qualité.
3. **#2** — nécessaire pour que la perte optimisée soit bien l'objectif DDPM.
4. **#5** — sans effet sur la correction, mais conditionne le débit
   d'entraînement et donc la capacité à itérer sur les points précédents.
5. **#6**, **#7** — confort d'observation et de diagnostic.
