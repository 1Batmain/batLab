# LOSS_WEIGHTING — pondérer la loss de diffusion par tirage biaisé des timesteps

Branche `loss-weighting`. `SCALE_UNET.md` §5 conclut que la cause du mode
collapse est la **pondération de l'objectif** : la MSE sur `ε` non pondérée pèse
implicitement l'erreur `x₀` par `SNR(t) = ᾱ/(1−ᾱ)`, qui couvre ~1e8 sur le
schedule T = 256. Cette mission implémente la pondération, la teste, et la mesure
sur un bras apparié.

**Statut : (verdict en §6).**

---

## 1. Ce qui est implémenté

### 1.1 Pondérer sans toucher au moindre shader

Au lieu de pondérer la loss dans `loss.wgsl` ou le gradient dans `optimizer.wgsl`,
l'entraînement change la **distribution de tirage du timestep**. Au lieu de
`t ~ U{0, T−1}`, il tire `t ~ p(t) ∝ w(t)`. En espérance c'est exactement une
loss pondérée par `w(t)`, à la constante `Z = Σ w(t)` près :

```
E_{t ~ p}[ L_t ] = Σ_t (w(t)/Z) · L_t = (1/Z) · E_{t ~ U}[ T · w(t) · L_t ]
```

La constante est sans effet sous Adam (dont la mise à jour est invariante à un
rééchelonnage du gradient, cf. `OPTIMIZER_ADAM.md` §1.2) et n'est qu'un facteur
de learning rate sous SGD. **Aucun shader, aucune passe arrière, aucun format de
checkpoint ne change** — c'est le point de la méthode : le risque
d'implémentation est confiné à une trentaine de lignes de CPU.

`bat_building/src/model/training/weighting.rs`, appelé depuis
`DiffusionTask::train_step_batch_inner`.

### 1.2 La distribution retenue

`w(t) = min(1, γ/SNR(t)) = min(1, γ·(1−ᾱ_t)/ᾱ_t)` — la forme min-SNR-γ
(Hang et al. 2023), qui est la formulation « `(1−ᾱ)/ᾱ` plafonné » de la mission :
proportionnelle à l'inverse du SNR — ce qui **aplatit à une constante le poids
implicite sur l'erreur `x₀`** — écrêtée à 1 pour que le bas de la plage garde une
part finie des tirages.

**γ = 1**, c'est-à-dire l'écrêtage au point où signal et bruit portent la même
puissance : le seul point distingué du schedule, donc aucune constante ajustée
n'entre dans le défaut. Le choix est chiffré en §2.

**Mélange uniforme de 5 %** : `p(t) = 0,95·w(t)/Z + 0,05/T`. Sans lui,
`p(0) = 1,87e-6` sur le schedule de production, soit **0,045 tirage attendu** sur
un run de 1500 pas à batch 16 — les premiers pas de la chaîne inverse ne
seraient jamais entraînés. Avec, `p(0) = 1,97e-4`, soit 4,7 tirages attendus, et
la variance de l'échantillonnage d'importance est bornée. Le coût est nul :
les masses par tranche bougent de moins d'un point.

### 1.3 Le chemin uniforme reste bit à bit l'ancien

`LossWeighting::Uniform` est le défaut et **ne passe pas par la table** : il
appelle `diffusion_step_for` inchangé. Une table uniforme aurait consommé le mot
aléatoire différemment (inverse-CDF au lieu de modulo) et aurait silencieusement
cassé l'appariement avec **tous** les runs enregistrés avant ce changement —
`OPTIMIZER_ADAM.md`, `SCALE_UNET.md`. Le test
`uniform_sampler_is_bit_identical_to_the_legacy_draw` fige la propriété.

### 1.4 Le probe n'est pas touché — c'est l'instrument de mesure

`probe_diffusion` (`metrics.rs:213-221`) tire ses timesteps de façon déterministe
et **uniforme dans chaque tranche** (`t = t_lo` et `t_lo + span/2`), sur un jeu de
sonde fixe. Il ne consulte pas le sampler d'entraînement et n'a pas été modifié :
biaiser l'instrument avec la même distribution que l'objectif aurait rendu les
deux bras incomparables — chacun aurait été mesuré là où il s'entraîne.

### 1.5 Interface

```bash
--loss-weighting uniform|snr    # défaut: uniform
--snr-gamma F                   # défaut: 1.0
```

Exposés en headless, sérialisés dans `TrainingConfig` (`#[serde(default)]`, donc
tout config antérieur se relit en `uniform`). La bannière du run réémet la
pondération **réellement parsée** et la masse par tranche — c'est le garde-fou du
binaire périmé de `OPTIMIZER_ADAM.md` §2.4, appliqué au nouveau drapeau :

```
[weighting] snr(gamma=1) — timestep mass per bucket: 0.0918 0.3025 0.3029 0.3029
```

### 1.6 Tests

`cargo test --workspace --release` : **83 passés, 0 échec** (81 avant, +8 ajoutés,
−6 de recomptage). Les nouveaux :

| test | ce qu'il fige |
|---|---|
| `uniform_sampler_is_bit_identical_to_the_legacy_draw` | non-régression : mêmes tirages que l'ancien code, sur 5 pas × 64 offsets |
| `snr_empirical_draws_match_the_target_distribution` | 200 000 tirages, chaque tranche à moins de 6σ binomiaux de `p(t)`, et aucun timestep jamais tiré |
| `snr_weight_matches_the_min_snr_formula` | `w = min(1, γ/SNR)` à 1e-12, sur 3 γ × 5 timesteps |
| `snr_probabilities_shift_mass_towards_high_t` | masse basse < 0,12, masse haute > 0,28, monotonie, plancher uniforme |
| `invalid_gamma_falls_back_to_uniform` | γ ≤ 0 ou NaN → repli sur uniforme plutôt que CDF cassée |
| `serde_roundtrip_and_missing_field_default` | config sans le champ → `uniform` |

---

## 2. Pourquoi γ = 1 — la justification chiffrée

Sur le schedule de production (T = 256, betas rescalés) :

| t | ᾱ_t | SNR(t) | w(t) = min(1, 1/SNR) | p(t) |
|---:|---:|---:|---:|---:|
| 0 | 0,99961 | 2559 | 3,91e-4 | 1,97e-4 |
| 16 | 0,95296 | 20,26 | 4,94e-2 | 4,19e-4 |
| 32 | 0,83992 | 5,247 | 0,191 | 1,06e-3 |
| 64 | 0,51483 | 1,061 | 0,942 | 4,47e-3 |
| 128 | 0,07415 | 0,0801 | 1,0 | 4,73e-3 |
| 255 | 3,29e-5 | 3,29e-5 | 1,0 | 4,73e-3 |

Masse par tranche de sonde :

| pondération | t[0-64) | t[64-128) | t[128-192) | t[192-256) |
|---|---:|---:|---:|---:|
| uniforme | 0,2500 | 0,2500 | 0,2500 | 0,2500 |
| **snr γ=1** | **0,0918** | **0,3025** | **0,3029** | **0,3029** |
| snr γ=5 (valeur de la littérature) | 0,179 | 0,274 | 0,274 | 0,274 |
| snr γ=0,3 | 0,029 | 0,273 | 0,349 | 0,349 |

### Pourquoi pas γ = 5, la valeur standard

Parce que sur **ce** schedule elle ne fait presque rien : `SNR(t) ≤ 5` dès
`t = 33`, donc `w = 1` sur 87 % du schedule et la redistribution se limite à
retirer 29 % de la masse de la tranche basse (0,250 → 0,179). γ = 1 en retire
**63 %**.

### Ce que γ = 1 vise, chiffré sur les gradients

L'amplitude du gradient de sortie est `2(ε̂−ε)`, donc proportionnelle à la RMS de
l'erreur. Avec les erreurs mesurées sur L (`SCALE_UNET` §3) —
√0,621 = 0,79 / √0,070 = 0,26 / √0,028 = 0,17 / √0,021 = 0,14 — le **budget de
gradient** par tranche vaut `p(tranche) × RMS(tranche)`, normalisé :

| | t[0-64) | t[64-128) | t[128-192) | t[192-256) |
|---|---:|---:|---:|---:|
| uniforme | **0,59** | 0,19 | 0,13 | 0,10 |
| snr γ=1 | **0,26** | 0,29 | 0,20 | 0,16 |

C'est l'objectif de la manœuvre : faire passer la tranche basse — celle dont
`SCALE_UNET` §3.4 dit qu'elle a un **plancher irréductible** et qu'il ne faut pas
piloter dessus — de 59 % à 26 % du budget de gradient, au profit des tranches où
le contenu se décide.

---

## 3. Le comparatif apparié

(rempli en fin de run — voir §4)

---

## 4. Résultats

(à compléter)

---

## 5. Un confondant trouvé en chemin : la diversité mesurée **est** le bruit du sampler

`SCALE_UNET` §1.3 a écarté le sampler comme cause du collapse en contrôlant
`paths` (3 → 1). Le second réglage d'inférence, **`magnitude = 0.3`** — qui
réduit à 30 % le bruit `σ_t·z` réinjecté à chaque pas de la chaîne inverse — n'a
jamais été contrôlé. Il l'est ici, sur **le checkpoint L 10 000 pas déjà
publié**, sans réentraîner : 8 seeds, `paths=1`, seule la magnitude varie.

| magnitude | `inter_seed_std` | `inter_seed_std` / magnitude | `intra_image_std` | `banding_ratio` | `pixel_mean` |
|---|---:|---:|---:|---:|---:|
| 0,3 (réglage du config) | 0,00589 | 0,0196 | 0,1078 | 2,875 | 0,289 |
| 0,6 | 0,01154 | 0,0192 | 0,2135 | 5,588 | 0,664 |
| 1,0 (chaîne DDPM nominale) | 0,01745 | 0,0175 | 0,2828 | 5,966 | 0,754 |
| dataset réel | 0,2316 | — | 0,2061 | 1,074 | 0,441 |

Le `0,00589` à magnitude 0,3 **reproduit exactement** le chiffre publié dans
`SCALE_UNET` §4 (0,00589) : le pipeline de mesure est le même.

Deux lectures, et la seconde annule la première :

1. **Naïvement**, la diversité inter-seeds triple (0,0059 → 0,0175) en remettant
   simplement la chaîne à son bruit nominal. Le « 39× sous le dataset » de
   `SCALE_UNET` §1.3 devient 13×. Un réglage d'inférence pesait donc un facteur 3
   sur la métrique qui sert de verdict à toute la série de rapports.

2. **Mais `inter_seed_std / magnitude` est constant à ±6 %** (0,0196 / 0,0192 /
   0,0175). La diversité inter-seeds est donc **proportionnelle au bruit injecté**
   — c'est-à-dire qu'elle *est* ce bruit, transmis jusqu'à l'image finale, et non
   du contenu produit par le modèle. Confirmation indépendante :
   `intra_image_std` passe de 0,108 à **0,283**, soit **au-dessus** du dataset
   réel (0,206), et le `banding_ratio` **double** (2,88 → 5,97). À magnitude 1,0
   les images ne sont pas plus variées : elles sont plus **bruitées**.

**Conséquences pratiques :**

- La conclusion de `SCALE_UNET` (« le modèle ne produit pas de contenu ») en
  ressort **renforcée**, pas affaiblie : le peu de variation inter-seeds qu'on
  mesurait était déjà du bruit résiduel non débruité.
- **`inter_seed_std` seul n'est pas un critère de succès valide.** Il faut le
  lire à magnitude fixée, et toujours accompagné de `intra_image_std` (qui doit
  s'approcher de 0,206 par le bas, pas le dépasser) et de `banding_ratio`. Le
  critère « diversité inter-seeds en hausse » de cette mission est appliqué à
  magnitude constante (0,3) pour cette raison.

---

## 6. Verdict

(à compléter)
