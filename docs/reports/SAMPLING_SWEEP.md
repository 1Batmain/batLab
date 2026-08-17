# SAMPLING_SWEEP — trouver les réglages d'échantillonnage les plus nets

**Modèle** : `Elephants_XL` (1,19 M params, entrée `[32,32,7]`, sortie `[32,32,3]`).
**Poids servis par la page** : `eleph_night/night.ckpt` = `web/dist/weights.ckpt`
**au bit près** — un `BBCKPT3` qui porte une **EMA (decay 0,999)**. **Dataset
(cible)** : `datasets/elephants256.batraw`, 256 images 32×32 RGB. **Rien n'a été
réentraîné**. Graines des planches : **101, 202, 303, 404, 505, 606** (celles de
toutes les planches déjà faites, donc comparables).

L'instrument de jugement n'est pas « je trouve ça joli » : c'est la distance des
statistiques d'un lot généré à celles des **vraies** photos du dataset, sur le
**vecteur complet** (`intra_image_std`, `grad_energy`, `laplacian_var`,
`saturation`, `pixel_mean`, `banding_ratio`). Trop lisse ET trop bruité s'en
écartent ; une mesure de netteté qui récompense le bruit est un piège (le bruit
maximise l'énergie de gradient), donc **`grad_energy`/`laplacian_var` se lisent
contre la cible, jamais en valeur absolue**. Contrôle de non-vacuité : du bruit
blanc pur et un aplat gris doivent tous deux obtenir un mauvais score — vérifié
explicitement (§2).

Métrique validée contre une mesure indépendante de l'architecte : reproduite au
4ᵉ décimale sur le réglage actuel de la page (EMA : 0,1667 / 0,1159 / 0,0267 /
0,0279 / 0,3686) et son itéré brut (0,1563 / 0,1054 / 0,0210 / 0,0436 / 0,3089).

## 0. Le cadrage : un DÉFICIT DE DÉTAIL, pas un excès de bruit

Mesuré sur les 6 graines de référence au réglage **actuel de la page**
(`night.ckpt`, EMA, magnitude 1,0, 1 trajectoire, variance beta) :

| mesure | cible (dataset) | page actuelle (EMA) | itéré brut |
|---|---|---|---|
| intra_image_std | 0,2025 | 0,1667 | 0,1563 |
| grad_energy | 0,1442 | 0,1159 | 0,1054 |
| laplacian_var | 0,0543 | 0,0267 | 0,0210 |
| saturation | 0,0524 | 0,0279 | 0,0436 |
| pixel_mean | 0,4395 | 0,3686 | 0,3089 |

**La génération est SOUS la cible sur TOUTES les mesures, et de moitié sur le
laplacien.** Le défaut n'est donc pas un excès de bruit mais un **manque de
détail** — trop lisse. Trois conséquences, prises en compte dans le balayage :

1. **Baisser la magnitude ou passer en variance postérieure va lisser ENCORE** :
   ces deux axes iront probablement dans le mauvais sens. Testés quand même, et
   le balayage explore **AUSSI au-dessus de 1,0** (1,2 et 1,5) — c'est une mesure,
   pas un dogme.
2. **`laplacian_var` et `grad_energy` se truquent au bruit blanc.** Le critère
   reste donc la distance au **vecteur complet**, plus le contrôle de non-vacuité.
3. **EMA vs itéré brut sont DEUX axes qui ne vont pas ensemble** : l'EMA est plus
   NETTE (laplacien 0,0267 > 0,0210) mais MOINS saturée (0,0279 < 0,0436). On les
   rapporte séparément plutôt que de les fondre.

### Prémisses de la mission corrigées

- L'axe EMA/brut est **vivant** sur `night.ckpt` (le `BBCKPT3` déployé), pas mort.
  Le `latest.ckpt` de `Models/Elephants_XL/pretrained_weights/` est un **autre run**
  (`BBCKPT2`, sans EMA) et n'est **pas** ce que la page sert — ne pas le confondre.
- La config `Elephants_XL` porte `denoising_paths: 10`, mais **la page fait 1
  trajectoire** (la boucle `reverse_step_async` du crate web) : côté page, l'axe
  trajectoires est déjà à 1.

## 1. La variance postérieure, en option (Item 1)

La chaîne inverse ajoute `sigma_t · N(0,1)` à la moyenne postérieure. DDPM admet
deux `sigma_t²` :

- **`beta`** — `sigma_t² = beta_t`. L'actuelle, la plus grande. Défaut.
- **`posterior`** — `sigma_t² = beta_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)`, la variance
  du vrai postérieur `q(x_{t-1} | x_t, x_0)`. Strictement plus petite ; le rapport
  `sigma_post/sigma_beta = √((1-ᾱ_{t-1})/(1-ᾱ_t))` vaut **0,600 à t=1**, 0,824 à
  t=4, ~0,95 en moyenne sur t<64, ~0,99 sur toute la chaîne. **C'est donc la toute
  fin de chaîne qui est sur-bruitée** en `beta` — là où se décide la netteté.

Le calcul vit **une seule fois** : `LinearNoiseSchedule::posterior_sigma`, le seul
site de la formule (`denoise_step_with_magnitude`). L'enum `PosteriorVariance`
(défaut `Beta`) se propage par la récursion partagée
(`reverse_step_from_epsilon` → `reverse_step{,_async}` → `sample_diffusion`) ; la
dérive/perpetual garde explicitement `Beta` (bit-à-bit inchangée — c'est un choix
d'inférence, pas de dérive). Exposé en flag CLI (`--headless-sample
--variance beta|posterior`) et dans la config (`InferenceConfig.posterior_variance`,
serde default `Beta`), repris à l'identique par le crate web (constante
`POSTERIOR_VARIANCE`).

**Le défaut ne change RIEN au bit près**, prouvé à trois niveaux :
`the_beta_variance_is_the_historical_sigma_to_the_bit` (`sigma == √beta` à chaque
pas), `a_config_without_a_variance_field_deserialises_to_beta`, et au binaire la
sortie par défaut est **identique octet pour octet** à `--variance beta` (et
diffère de `posterior`).

## 2. Le balayage (Item 2)

Axes croisés : **magnitude** ∈ {0,3 · 0,5 · 0,7 · 0,85 · 1,0}, **variance** ∈
{beta · posterior}, **trajectoires** ∈ {1 · 2 · 4}. (L'axe poids est mort, cf. §0.)
30 configs × 6 graines = 180 échantillons. Chaque config est mesurée sur ses 6
graines et classée par **distance à la cible** = erreur relative moyenne sur
`intra_image_std`, `grad_energy`, `laplacian_var`, `saturation`, `pixel_mean`.

### Cible mesurée sur les 256 vraies images

| intra_image_std | grad_energy | laplacian_var | saturation | pixel_mean | banding_ratio |
|---|---|---|---|---|---|
| **0,2025** | **0,1442** | **0,0543** | **0,0524** | **0,4395** | **0,916** |

`intra_image_std` s'approche **par le bas** (le dépasser = bruit ajouté).

### Contrôle de non-vacuité (le test qui protège la métrique)

`grad_energy` et `laplacian_var` se truquent au bruit. Deux extrêmes doivent donc
mal scorer, sinon la distance est vide de sens :

| source | distance |
|---|---|
| bruit blanc pur (6 champs uniformes) | **2,58** |
| aplat gris à la bonne moyenne | **∞** (banding indéfini, gradient nul) |
| **pire config réelle du balayage** | **0,645** |

Les deux contrôles sont **loin pires** que n'importe quelle vraie config → la
métrique n'est pas truquable au bruit. (Test refait à chaque analyse.)

### Tableau classé (1 trajectoire ; 28 configs, extrait)

distance = erreur relative moyenne au vecteur cible. Plus petit = plus proche des
vraies photos.

| # | magnitude | variance | poids | **dist** | intra | grad | lap | sat | mean |
|--:|---|---|---|--:|--:|--:|--:|--:|--:|
| **1** | **1,2** | **beta** | **brut** | **0,098** | 0,217 | 0,144 | 0,045 | 0,053 | 0,322 |
| 2 | 1,2 | posterior | brut | 0,126 | 0,214 | 0,136 | 0,039 | 0,052 | 0,322 |
| 3 | 1,2 | beta | EMA | 0,139 | 0,234 | 0,149 | 0,049 | 0,037 | 0,353 |
| 4 | 1,2 | posterior | EMA | 0,164 | 0,232 | 0,142 | 0,042 | 0,035 | 0,353 |
| 5 | 1,5 | posterior | brut | 0,191 | 0,270 | 0,178 | 0,071 | 0,058 | 0,393 |
| … | | | | | | | | | |
| 8 | **1,0** | **beta** | **EMA** | **0,253** | 0,167 | 0,116 | 0,027 | 0,028 | 0,369 |
| 9 | 1,0 | beta | brut | 0,267 | 0,156 | 0,105 | 0,021 | 0,044 | 0,309 |
| … | | | | | | | | | |
| 28 | 0,3 | posterior | EMA | 0,645 | 0,028 | 0,008 | 0,000 | 0,018 | 0,467 |

**Ligne 8 = le réglage actuel de la page.** Le meilleur (ligne 1) est à **0,098**
contre **0,253** — 2,6× plus proche du dataset. Tableau complet :
`scratchpad/night/results.json` (harnais sous `bench/sampling/`).

### Axe trajectoires (au réglage gagnant, magnitude 1,2 beta)

| trajectoires | poids brut (dist) | poids EMA (dist) |
|---|--:|--:|
| **1** | **0,098** | **0,139** |
| 2 | 0,250 | 0,246 |
| 4 | 0,320 | 0,332 |

Moyenner **lisse** : le laplacien tombe de 0,045 (1 traj) à 0,016 (4 traj), un
tiers de la cible. **Une seule trajectoire bat deux, qui bat quatre** — et donc
bat les dix de la config. Confirme la mesure de la mission ; côté page c'était
déjà 1, côté config c'était 10.

### Ce que chaque axe fait

- **magnitude** — le grand levier. En dessous de 1,0 la génération s'effondre en
  **aplats de couleur** (planche §1 : 0,3 et 0,5 n'ont plus d'éléphant — exactement
  ce que l'utilisateur décrivait). 1,2 remonte le détail jusqu'à poser
  `grad_energy` **sur** la cible ; 1,5 le **dépasse** et bascule en mouchetis
  bruité (grad 0,19 > 0,14, lap 0,082 > 0,054). 1,2 est le point d'équilibre.
- **variance** — `posterior` va dans le **mauvais sens** : plus petite, elle
  lisse encore (lap 0,039 < 0,045 de beta à 1,2). L'option existe et est mesurée,
  mais pour ce modèle-là c'est `beta` qui gagne. (Prédit par le cadrage.)
- **poids EMA vs brut** — deux axes qui ne vont pas ensemble, comme annoncé :
  l'EMA est un cheveu plus **nette** (lap 0,049 vs 0,045) mais nettement moins
  **saturée** (0,037 vs 0,053 ; cible 0,052). Le brut **colle la saturation** de
  la cible ; sur des images qui lisaient « gris boueux », ce gain-là se voit
  (planche §2). Le vecteur complet tranche pour le brut.
- **trajectoires** — 1, sans appel (ci-dessus).

## 3. Recommandation et application (Item 3)

**Réglage recommandé : magnitude 1,2 · variance beta · 1 trajectoire · itéré brut**
(distance 0,098 contre 0,253 pour l'actuel ; et le meilleur à l'œil, planche §2 —
formes d'éléphants nettes, couleur revenue). L'EMA au même réglage (0,139) est le
dauphin : plus lisse mais plus terne — à préférer si on tient à la douceur EMA.

Appliqué :

- **`Models/Elephants_XL/config_file`** (bloc `inference`) : `denoise_magnitude`
  1,0 → **1,2**, `denoising_paths` 10 → **1**, `posterior_variance` explicité à
  **`beta`**. (Le sélecteur EMA/brut n'est pas dans `InferenceConfig` — voir §4 —
  donc l'inférence native TUI tourne en EMA : mag 1,2 beta 1 traj = 0,139, déjà un
  gros gain sur les dix trajectoires d'avant ; `--raw-weights` en CLI donne le
  0,098 exact.)
- **`crates/batlab-web/src/lib.rs`** : `DENOISE_MAGNITUDE` 1,0 → **1,2**,
  `POSTERIOR_VARIANCE` = **`Beta`**, et le chargement des poids passe de
  `CheckpointWeights::Ema` à **`Raw`**. La page fait déjà 1 trajectoire. C'est
  l'application intégrale de la recommandation.

## 4. Ce que le balayage n'a PAS pu améliorer — et où est vraiment le plafond

Le meilleur réglage reste à **distance 0,098**, pas 0. Ce qui reste n'est **pas**
réglable au sampler :

- **La luminosité.** `pixel_mean` du meilleur réglage vaut 0,322 (brut) / 0,353
  (EMA) contre **0,439** pour le dataset — un manque de ~25 % qui pèse pour ~la
  moitié de la distance résiduelle. Aucune magnitude ne le comble : monter la
  magnitude ajoute du contraste, pas de la clarté moyenne. Les éléphants générés
  sont **plus sombres** que les vrais ; c'est une propriété du modèle (ou du
  déséquilibre clair/sombre du corpus), pas du tirage.
- **La finesse du trait.** Même à 1,2, `laplacian_var` (0,045) reste **sous** la
  cible (0,054) sans que `grad_energy` puisse monter davantage sans virer au
  bruit. À 32×32 sur un modèle de 1,2 M de paramètres, le détail fin que portent
  les vraies photos n'est tout simplement pas dans ce que le réseau produit.

**Verdict honnête** : le sampler avait une marge réelle et bien nette — la page
passe de 0,253 à 0,098, un facteur 2,6, visible à l'œil (couleur et détail
reviennent). Mais le **plafond qui reste est le modèle et la résolution 32×32**,
pas l'échantillonnage. Le prochain gain se gagne à l'entraînement (le levier
qualité mesuré ces deux jours) ou en montant la résolution — pas en continuant à
tourner les boutons du tirage. Deux réglages (variance postérieure, trajectoires
> 1) ont été mesurés **NO-GO** pour ce modèle et le sont pour de bon.
