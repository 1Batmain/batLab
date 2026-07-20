# Audit du pipeline d'entraînement — batLab

Branche : `audit-training`. Livrable : diagnostic uniquement, aucune modification
du code de production (hors ajout d'un module de tests `#[cfg(test)]`).

Tests diagnostiques : `bat_building/src/model/audit_tests.rs`
(activés par `#[cfg(test)] mod audit_tests;` en fin de `bat_building/src/model/mod.rs`).
Ces tests sont écrits pour **échouer** contre l'implémentation actuelle — l'échec
*est* le finding.

---

## Finding #1 — `PaddingMode::Same` : convolution décalée et bords écrasés — **CRITIQUE**

**Fichiers** : `bat_building/src/model/shader/convolution.wgsl:49-51`,
`bat_building/src/model/shader/back_convolution.wgsl:63-71` et `:105-107`,
`bat_building/src/model/layer_types/convolution.rs:172,185`.

### Mécanisme

`ConvolutionType::set_dim_output()` calcule bien la dimension de sortie du mode
`Same` (`dim_input.x.div_ceil(stride)`), mais le shader **n'applique jamais
l'offset de padding correspondant**. Le commentaire du shader l'assume
explicitement — et à tort :

```wgsl
// convolution.wgsl:19
padding_mode: u32, // 0 = Valid, 1 = Same (unused in kernel — dims already computed)
...
// convolution.wgsl:49-51
let iy    = oy * s + ky;      // ← devrait être  oy*s + ky - pad_y
let ix    = ox * s + kx;      // ← devrait être  ox*s + kx - pad_x
let in_i  = iy * IW * IC + ix * IC + kz;
```

Deux conséquences, toutes deux destructrices :

1. **Décalage spatial.** Pour un noyau 3×3 stride 1, le pixel de sortie `(oy,ox)`
   agrège les entrées `(oy..oy+2, ox..ox+2)` au lieu de `(oy-1..oy+1, ox-1..ox+1)`.
   La carte de features glisse de `(KH/2, KW/2)` **à chaque couche `Same`**. Sur
   un U-Net à N convolutions, la sortie est décalée de N pixels en diagonale par
   rapport à la cible.
2. **Débordement mémoire silencieux.** `iy` monte jusqu'à `IH + KH - 2`, hors des
   bornes de `input`. WGSL applique la robustesse d'accès : les lectures sont
   *clampées* sur le dernier élément au lieu de renvoyer 0. Les lignes/colonnes
   basses et droites sont donc remplies par répétition du dernier pixel — ce
   n'est ni du zero-padding, ni du edge-padding cohérent.

C'est d'autant plus net que `upsample_conv.wgsl:34-46,72-73` **implémente
correctement** `pad_y()` / `pad_x()`. Les deux couches convolutives du framework
divergent sur le même concept ; seule `Convolution` est cassée.

### Preuve

`audit_tests.rs::same_padding_conv_is_centered_and_zero_padded`. Conv 3×3
`Same`, stride 1, entrée 4×4 = `1..16`, noyau dont **seul le tap central vaut 1**
(une convolution `Same` correcte est alors l'identité) :

```
attendu : [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]   (identité)
obtenu  : [ 6, 7, 8, 9,10,11,12,13,14,15,16,16,16,16,16,16]
```

Décalage de +5 (= une ligne + une colonne) puis saturation à 16 : le décalage
*et* le clamp hors-bornes sont confirmés sur GPU réel.

### Impact entraînement

Le modèle apprend à prédire un bruit **spatialement décalé** de sa cible. Comme
le décalage est systématique et que les bords sont constants, le réseau peut
réduire la MSE en apprenant un flou global, mais ne peut jamais converger vers
un débruitage correct. Dans un U-Net, `Concat` exige des dimensions spatiales
identiques entre branche descendante et skip — `Same` est donc quasi obligatoire :
ce bug est presque certainement actif sur les configurations réelles.

### Fix proposé

Aligner `convolution.wgsl` sur `upsample_conv.wgsl` :

```wgsl
fn pad_y() -> i32 { if layer_spec.padding_mode == 1u { return i32(layer_spec.dim_kernel.x / 2u); } return 0; }
fn pad_x() -> i32 { if layer_spec.padding_mode == 1u { return i32(layer_spec.dim_kernel.y / 2u); } return 0; }

let iy = i32(oy * s) + i32(ky) - pad_y();
let ix = i32(ox * s) + i32(kx) - pad_x();
if iy < 0 || iy >= i32(IH) || ix < 0 || ix >= i32(IW) { continue; }  // zero-padding explicite
```

Le même offset et le même test de bornes doivent être répliqués **à l'identique**
dans `back_convolution.wgsl` :
- `conv_back_input` : `oy = (iy + pad_y - ky) / s`, avec test de divisibilité et de bornes ;
- `conv_back_weights` : `iy = oy*s + ky - pad_y`, en sautant les positions hors bornes
  (elles contribuent 0 au gradient).

Sans cette symétrie, corriger le forward seul rendrait le gradient faux.

---

## Finding #2 — Timestep de diffusion parfaitement corrélé à l'échantillon — **MAJEUR**

**Fichier** : `bat_building/src/model/training/diffusion.rs:172-173`.

```rust
let sample_index    = (step * batch_size + batch_offset) % sample_count;
let diffusion_step  = diffusion_step_for(step, batch_size, batch_offset, schedule_len);
//                  = (step * batch_size + batch_offset) % schedule_len
```

### Mécanisme

Les deux quantités dérivent du **même compteur linéaire** `n = step*batch_size + batch_offset`.
Le timestep n'est donc pas échantillonné — il est une fonction déterministe de
l'indice d'échantillon. L'image `i` reçoit toujours le timestep `i % 256`, à
chaque époque, pour toujours.

Le nombre de timesteps distincts qu'une image donnée verra sur *toute* la durée
de l'entraînement vaut `gcd(sample_count, schedule_len)`. Pour CIFAR-10
(50 000 images) et 256 pas : **`gcd(50000, 256) = 16`**. Vérifié
numériquement : l'échantillon 0 ne voit jamais que les timesteps
`{0, 48, 64, 80, 144, 160, 224, 240, ...}` — 16 valeurs sur 256.

La correction de la PR #6 (`57f31c6`) a bien supprimé l'aliasing *intra-batch*
(tous les éléments d'un batch partageaient un timestep), et le test
`diffusion_step_progression_does_not_alias_single_batch` couvre ce cas. Mais le
fix est **incomplet** : il n'a pas rompu la corrélation entre le timestep et
l'échantillon, qui est le problème plus profond. DDPM exige `t ~ U{0, T-1}`
**indépendant** de `x_0`.

### Impact

L'espérance de la perte optimisée n'est pas `E_{x,t,ε}[...]` mais une somme sur
un couplage dégénéré `(x_i, t_{i mod 256})`. Le réseau peut mémoriser la
corrélation « ce contenu d'image ⇒ ce niveau de bruit » au lieu d'apprendre le
débruitage conditionné au timestep. À l'échantillonnage, où l'on parcourt les
256 timesteps sur une *même* image, cette hypothèse apprise est violée à chaque
pas.

### Fix proposé

Découpler les deux : tirer le timestep d'un hash du seed, pas du compteur.

```rust
let diffusion_step = if schedule_len == 0 { 0 } else {
    (fold_seed(seed ^ 0xD1B5_4A32 ^ ((batch_offset as u64) << 16)) as usize) % schedule_len
};
```

Et conserver le test anti-aliasing intra-batch en le reformulant : vérifier que
les timesteps d'un même batch sont *distincts* et que, sur N époques, un
échantillon donné couvre ~uniformément `[0, T)`.

---

## Finding #3 — Le schedule n'atteint jamais le bruit pur — **MAJEUR**

**Fichiers** : `bat_building/src/model/training/schedule.rs:11-39` (construction),
appelants avec `num_steps = 256`, `beta ∈ [1e-4, 0.02]`.

### Mécanisme

Les bornes `beta_start = 1e-4`, `beta_end = 0.02` sont celles du papier DDPM —
mais elles y sont calibrées pour **T = 1000**. Ce projet utilise **T = 256** avec
les mêmes bornes, sans recalibrage :

| T | `alpha_bar_T` | `sqrt(alpha_bar_T)` | signal résiduel en `x_T` |
|---|---|---|---|
| 1000 (DDPM) | 4.04e-5 | 0.0064 | 0.6 % |
| **256 (ici)** | **0.0750** | **0.2739** | **27.4 %** |

À l'issue du forward process, `x_T` conserve **27 % du signal de l'image
d'origine**. `q(x_T)` est très loin de `N(0, I)`.

### Impact

L'échantillonnage part de bruit gaussien pur, alors que le modèle n'a jamais vu,
à `t = T`, autre chose qu'une image encore largement reconnaissable. C'est un
décalage train/inférence sur le pas le plus critique de la chaîne, et il se
propage à toute la trajectoire de débruitage. Les échantillons générés restent
structurellement bruités quel que soit le niveau de convergence de la perte.

### Fix proposé

Au choix :
- réétalonner `beta_end` pour T = 256 afin que `alpha_bar_T ≈ 1e-4`
  (≈ `beta_end = 0.06` avec un schedule linéaire), ou
- passer à un cosine schedule (Nichol & Dhariwal), plus robuste aux petits T, ou
- garder T = 1000.

Ajouter une assertion de construction : `alpha_bar(len-1) < 1e-3`, pour que toute
future combinaison (T, betas) incohérente échoue bruyamment.

---

## Finding #4 — Données en [0,1] alors que la diffusion suppose [-1,1] — **MAJEUR**

**Fichiers** : `main/src/main.rs:908` et `:925` (`pixel as f32 / 255.0`),
`main/src/main.rs:1098` (`to_u8` : `clamp(0,1) * 255`), commentaire du format
`.batraw` `main/src/main.rs:590` (« normalised [0, 1] »).

### Mécanisme

Les pixels sont normalisés dans `[0, 1]`, de moyenne ≈ 0.5. Le forward process
`x_t = sqrt(a_bar)·x_0 + sqrt(1-a_bar)·ε` suppose `x_0` **centré** — c'est la
raison pour laquelle DDPM normalise en `[-1, 1]`. Avec `x_0 ∈ [0,1]`, `x_t`
conserve un biais de moyenne `0.5·sqrt(alpha_bar_t)` à tous les timesteps.

Ce finding **compose** avec le #3 : à `t = T`, `E[x_T] ≈ 0.5 × 0.274 ≈ 0.137`
avec une variance réduite, tandis que l'échantillonneur démarre de `N(0,1)`
(moyenne 0, variance 1). L'écart entre la distribution d'entraînement et la
distribution d'échantillonnage est franc.

### Impact

Le réseau apprend à prédire `ε` à partir d'entrées systématiquement décalées
vers le positif. À l'inférence, l'entrée initiale est hors de la distribution
vue à l'entraînement : les premiers pas de débruitage partent dans une direction
arbitraire, et l'erreur s'accumule le long de la chaîne.

### Fix proposé

Normaliser en `[-1, 1]` au chargement (`v / 127.5 - 1.0`) et dénormaliser
symétriquement à l'écriture d'image (`(v + 1) * 127.5`, `to_u8` à ajuster en
`main.rs:1098`). Attention : le format `.batraw` existant devient incompatible —
soit bumper le magic header, soit convertir à la lecture.

---

*Audit en cours — findings suivants à venir (optimizer, group_norm, concat,
boucle d'entraînement).*
