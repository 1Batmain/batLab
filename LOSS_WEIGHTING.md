# LOSS_WEIGHTING — pondérer la loss de diffusion par tirage biaisé des timesteps

Branche `loss-weighting`. `SCALE_UNET.md` §5 conclut que la cause du mode
collapse est la **pondération de l'objectif** : la MSE sur `ε` non pondérée pèse
implicitement l'erreur `x₀` par `SNR(t) = ᾱ/(1−ᾱ)`, qui couvre ~1e8 sur le
schedule T = 256. Cette mission implémente la pondération, la teste, et la mesure
sur un bras apparié.

---

## TL;DR

- **La pondération est implémentée, testée et livrée** (`--loss-weighting
  uniform|snr`, `--snr-gamma`), sans toucher un seul shader : c'est le tirage du
  timestep qui est biaisé, ce qui équivaut en espérance à pondérer la loss.
  Le chemin `uniform` reste **bit à bit** l'ancien. 83 tests passent.
- **Elle marche dans le bon sens, et l'ampleur n'y est pas.** Bras apparié
  1500 pas sur L + Adam : haut-`t` **×1,20** en faveur de `snr` (×1,36 au
  meilleur point), tranche basse ×0,92 en sa défaveur, banding **−29 %**.
  Diversité inter-seeds +23 % brut, mais **+4 % une fois normalisée** du bruit
  résiduel — non établie sur 8 seeds.
- **Verdict : NO-GO** sur les critères de la mission (attendu : sous 0,0004, ou
  ×10 ; obtenu : 0,00242 et ×1,20). **Le run de nuit part sur L + Adam lr 1e-3
  sans pondération**, exactement la config d'`OPTIMIZER_ADAM` §6.
- **Ce que le résultat négatif apprend** : les deux bras **plafonnent au même
  endroit dès le pas ~600**. La pondération n'était pas le facteur limitant. Le
  « déséquilibre 1e5 » de `SCALE_UNET` §5.3 est un déséquilibre en unités `x₀` ;
  en unités `ε`, où le gradient est calculé, il vaut **≈ 5,7×** — et on récupère
  exactement ce qu'une correction de 5,7× peut donner. Piste suivante, non
  mesurée : le réseau n'a **aucune route bon marché vers l'identité** (une
  `GroupNorm` derrière chaque convolution), or `ε̂ ≈ x_t` est ce qu'il faut à `t`
  élevé → **connexion résiduelle globale** ou paramétrisation `x₀`/`v`. §6.2.
- **Trouvaille annexe, indépendante de la mission** : `inter_seed_std` est
  **proportionnel à `--magnitude`** (constant à ±6 % sur 0,3 / 0,6 / 1,0). La
  diversité qu'on mesurait depuis `SCALE_UNET` est du **bruit non débruité**, pas
  du contenu. Le critère « diversité en hausse » n'est lisible qu'à magnitude
  fixée et accompagné de `intra_image_std`. §5.

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

## 3. Le comparatif apparié — protocole

Deux bras, 1500 pas, `Greyscale_Diffusion_L`, `--optimizer adam --lr 1e-3
--batch 16`, `cifar10_grey.batraw`, from scratch. Seule différence :
`--loss-weighting uniform` contre `--loss-weighting snr --snr-gamma 1`.

**L'appariement est plus faible que dans `OPTIMIZER_ADAM` §2.1, par construction.**
Là-bas les deux bras voyaient des entrées octet pour octet identiques et la loss
au pas 0 était bit à bit égale. Ici les bras diffèrent **précisément par le
timestep attribué à chaque échantillon** — c'est l'intervention elle-même. Les
loss au pas 0 diffèrent donc (1,0726 contre 1,1305), et c'est la **preuve que le
drapeau a mordu** (le piège du binaire périmé de `OPTIMIZER_ADAM` §2.4 aurait
donné deux valeurs identiques). Restent partagés : l'ordre du dataset,
l'initialisation des poids, les graines de bruit par pas, et **le probe**.

Le probe est l'instrument : jeu de sonde fixe, timesteps fixes (`t_lo` et
`t_lo + 32` par tranche), bruit déterministe par pas — identique dans les deux
bras, et délibérément non biaisé (§1.4). C'est la seule métrique comparable.

Les deux bras ont tourné **en parallèle** (contrainte de wall-clock : un run de
nuit partait derrière). Les courbes de loss n'en sont pas affectées — les runs
sont déterministes — mais **les temps par pas de ce rapport ne veulent rien
dire**, même caveat que `OPTIMIZER_ADAM` §2.3.

---

## 4. Résultats

### 4.1 Loss de sonde par tranche de t (moyenne des 5 derniers points, pas 1400–1499)

| tranche | `uniform` | `snr γ=1` | rapport | trivial `ε̂ = x_t` |
|---|---:|---:|---:|---:|
| t[0-64) | **0,54620** | 0,59214 | 0,92 | 0,8735 |
| t[64-128) | 0,04662 | **0,04649** | 1,00 | 0,1402 |
| t[128-192) | 0,00965 | **0,00934** | 1,03 | 0,0114 |
| **t[192-256)** | 0,00291 | **0,00242** | **1,20** | **0,0004** |

Meilleur point atteint sur tout le run :

| tranche | `uniform` | `snr γ=1` | rapport |
|---|---:|---:|---:|
| t[0-64) | **0,53269** | 0,58132 | 0,92 |
| t[64-128) | 0,04456 | **0,04295** | 1,04 |
| t[128-192) | 0,00873 | **0,00831** | 1,05 |
| **t[192-256)** | 0,00234 | **0,00172** | **1,36** |

Aucun `ε̂` non fini dans aucun bras (`non_finite: 0` sur les 61 points de sonde
des deux runs) : la redistribution des tirages n'a introduit aucune instabilité.

**Lecture.** La pondération fait exactement ce que la théorie annonce, dans le
bon sens et sur les bonnes tranches — le haut-`t` gagne 20 % (36 % au meilleur
point), le milieu 3–5 %, et la tranche basse paie 8 %, ce qui est le prix
attendu quand on lui retire 63 % de ses tirages. Le signe est net et le
classement est stable sur toute la seconde moitié du run.

**Mais l'ordre de grandeur n'y est pas.** Les critères de la mission étaient :
`< 0,0004` (le prédicteur trivial) **ou** 10× mieux que le bras uniforme. Le
bras `snr` fait **0,00242**, soit **6,1× au-dessus du trivial** et **1,20×**
mieux que `uniform`. Ce n'est pas 10×, et ce n'est pas sous le trivial.

### 4.2 Trajectoire haut-`t` (un point de sonde sur 4, pas 0 → 1499)

```
uniform  0.8071 0.0250 0.0114 0.0063 0.0048 0.0034 0.0081 0.0026 0.0025 0.0025 0.0028 0.0030 0.0026 0.0037 0.0031 0.0028
snr g=1  0.9800 0.0237 0.0099 0.0062 0.0042 0.0029 0.0037 0.0026 0.0021 0.0023 0.0018 0.0034 0.0027 0.0039 0.0024 0.0019
```

Les deux bras **plafonnent** vers 0,002–0,003 dès le pas ~600 et n'en bougent
plus : ce n'est pas une convergence plus lente qu'un run plus long réglerait, le
plateau est atteint. C'est le fait le plus important du rapport, voir §6.

### 4.3 Diversité inter-seeds — 8 seeds (1..8), `magnitude = 0.3`

| | `inter_seed_std` | `mean_pairwise_rmse` | `intra_image_std` | `banding_ratio` |
|---|---:|---:|---:|---:|
| `uniform` p=1 | 0,004188 | 0,006220 | 0,1565 | 8,339 |
| **`snr γ=1` p=1** | **0,005143** (+22,8 %) | 0,007640 | 0,1677 | **5,822** (−30 %) |
| `uniform` p=3 | 0,002473 | 0,003615 | 0,0620 | 5,845 |
| **`snr γ=1` p=3** | **0,003077** (+24,4 %) | 0,004490 | 0,0741 | **4,227** (−28 %) |
| *repère* L 10 000 pas (SGD) p=1 | 0,005885 | 0,008540 | 0,1078 | 2,875 |
| **dataset réel** | **0,2316** | 0,3247 | 0,2061 | 1,074 |

Planche : `weighting_samples/comparison.png`.

Le `+23 %` est **cohérent entre deux réglages de sampler indépendants** (p=1 et
p=3), ce qui écarte le hasard d'un réglage. Mais §5 interdit de le lire tel quel :
`intra_image_std` monte aussi (+7 % à p=1, +19 % à p=3), or la diversité
inter-seeds mesurée est en grande partie du **bruit résiduel non débruité**.
Rapporté à `intra_image_std`, le gain tombe à **+4 %** (p=1) et **+4 %** (p=3) —
sur 8 seeds, ce n'est pas distinguable de zéro.

**Le gain de diversité n'est donc pas établi.** Les deux bras restent à **45–75×
sous le dataset**, comme la baseline et comme L à 10 000 pas.

Le seul effet propre est le **banding, −28 à −30 %** — le même type de gain, et
le même ordre de grandeur, que celui qu'avait apporté l'agrandissement du
modèle (`SCALE_UNET` §4 : −33 %). Réel, reproductible, et sans rapport avec le
problème principal.

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

### 6.1 Sur les critères de la mission : **NO-GO**

| critère | seuil | mesuré | |
|---|---|---:|:--|
| MSE(ε̂,ε) haut-`t`, bras `snr` | < 0,0004 | **0,00242** | ✗ (6,1× au-dessus) |
| …ou ≥ 10× mieux que `uniform` | ×10 | **×1,20** | ✗ |
| diversité inter-seeds en hausse, 8 seeds | mesurable | +23 % brut, **+4 % normalisé** | ✗ non établi |

Ce que la pondération **fait bien** : le signe est correct sur toutes les
tranches, stable sur toute la seconde moitié du run, cohérent entre deux
réglages de sampler, et sans aucune instabilité numérique. L'implémentation est
saine. Ce n'est pas un bug.

Ce qu'elle **ne fait pas** : bouger l'ordre de grandeur. Retirer 63 % des
tirages à la tranche basse et faire passer son budget de gradient de 59 % à 26 %
achète **20 %** de haut-`t`. On attendait un facteur, on a obtenu une marge.

### 6.2 Ce que ce résultat négatif apprend — la pondération n'était pas le facteur limitant

C'est le point à retenir pour la suite, et il est plus solide que le verdict
lui-même, parce qu'il repose sur §4.2 : **les deux bras plafonnent au même
endroit dès le pas ~600** et n'en bougent plus jusqu'à 1500. Un objectif
gradient-affamé ne plafonne pas — il descend lentement. Un objectif qui plafonne
en même temps avec 3× moins et 3× plus de tirages sur la tranche concernée bute
sur autre chose que la quantité de gradient qu'il y reçoit.

`SCALE_UNET` §5.3 imputait le collapse à un déséquilibre de pondération de ~1e5.
Le calcul est juste, mais **il mesure le poids implicite sur l'erreur `x₀`, pas
la quantité de gradient reçue par tranche**. En unités `ε` — celles où le
gradient est réellement calculé — le déséquilibre entre tranches n'est pas 1e5,
c'est le rapport des RMS d'erreur, soit **≈ 5,7×** (0,79 contre 0,14, §2). Un
facteur 5,7 se corrige par une redistribution de tirages ; c'est ce qui a été
fait, et on récupère bien l'ordre de grandeur attendu d'une telle correction
(20 %), pas plus. **Le « 1e5 » ne s'est jamais traduit en 1e5 de gradient.**

**Hypothèse pour la suite (non mesurée ici, à tester en premier).** Atteindre
0,0004 à `t[192-256)` exige `ε̂ ≈ x_t` à 2 % près, c'est-à-dire une **application
quasi-identité** du canal de signal vers la sortie. Or **chaque convolution de
cette architecture est suivie d'une `GroupNorm`**, qui retire à chaque étage la
moyenne et l'échelle par groupe : le réseau n'a aucune route bon marché vers
l'identité, et doit la reconstruire à travers 28 couches normalisées. Cela
expliquerait un plancher indépendant de la pondération. Deux corrections
testables, par ordre de coût :

1. **Connexion résiduelle globale** `ε̂ = x_t + f(x_t, t)` — le prédicteur trivial
   devient l'initialisation du modèle au lieu d'être son concurrent. Quelques
   lignes (une couche `Concat`/somme en sortie), aucun changement d'objectif.
2. **Changer de paramétrisation** (`x₀` ou `v`, `SCALE_UNET` §6 options 2 et 3) :
   la cible à `t` élevé devient l'image, bornée et structurée, au lieu d'une
   quasi-copie de l'entrée. Plus invasif.

Noter aussi que **0,0004 est le score d'un modèle qui prédirait `x̂₀ = image
moyenne`** (`trivial_baselines.py` : la colonne « ε̂ via moyenne » vaut aussi
0,0004 à cette tranche, et `SNR × E[x₀²] = 1,6e-3 × 0,234 = 3,8e-4`). Le seuil
de la mission n'est donc pas « produire du contenu », c'est « être au moins
aussi bon qu'un modèle effondré sur la moyenne ». Les deux bras en sont à 6× —
ils sont, à `t` élevé, **moins bons que le mode collapse lui-même**.

### 6.3 Recommandation pour le run de nuit : **L + Adam, pondération `uniform`**

```bash
cargo build --release -p main      # PAS `cargo build --release` seul
./target/release/main --headless-train Greyscale_Diffusion_L \
    --steps N --dataset datasets/cifar10_grey.batraw \
    --optimizer adam --lr 1e-3 --batch 16 --weight-init uniform
```

C'est-à-dire **exactement la config de `OPTIMIZER_ADAM` §6, inchangée** — le
défaut `--loss-weighting uniform` s'applique sans rien passer. Raisons :

- rien de mesuré ici ne justifie de changer la base de comparaison d'un run
  long : ×1,20 sur une tranche, ×0,92 sur une autre, diversité non établie ;
- le bras `uniform` reste directement comparable à `SCALE_UNET` et
  `OPTIMIZER_ADAM`, ce qui a de la valeur pour un run de nuit qui servira de
  référence ;
- le plateau de §4.2 dit qu'un run plus long ne convertira pas ces 20 % en
  davantage.

**Si le budget permet un second bras**, c'est celui-ci — le seul gain propre
mesuré (banding −29 %) y est, et le coût par pas est nul (le tirage est du CPU) :

```bash
    --loss-weighting snr --snr-gamma 1
```

**Ne pas** utiliser γ = 5 (la valeur de la littérature) sur ce schedule : §2
montre qu'elle n'y redistribue presque rien.

### 6.4 Ce que ce rapport ne prouve pas

- **Que la pondération est inutile en général.** Un seul γ (1) a été testé, sur
  un seul run de 1500 pas par bras, sur une seule architecture. γ = 0,3 (masse
  basse à 0,029 au lieu de 0,092) n'a pas été tenté faute de temps GPU avant le
  run de nuit — c'est le bras manquant le plus évident, ~35 min.
- **Que l'hypothèse GroupNorm/identité de §6.2 est la bonne.** Elle est
  cohérente avec le plateau et avec le fait que la cible haut-`t` est une
  quasi-copie de l'entrée, mais elle n'est **pas mesurée**. Le test décisif est
  la connexion résiduelle globale.
- **Que la diversité ne bougerait pas avec un run plus long.** Les deux bras sont
  à 1500 pas ; L à 10 000 pas (SGD) faisait 0,00589 contre 0,00514 ici. Rien
  n'indique une divergence des courbes, mais rien ne l'exclut non plus.
- **Le coût par pas de la pondération.** Non mesuré (les bras ont tourné en
  parallèle, §3). Le tirage est une recherche binaire dans 256 f64 par
  échantillon, sur un pas dominé par des passes GPU de ~1,3 s : il est
  structurellement négligeable, mais ce rapport ne le chiffre pas.

---

## 7. Reproduire

```bash
bash bench/loss_weighting/compare.sh      # les deux bras, 1500 pas, en parallèle
bash bench/loss_weighting/samples.sh      # 8 seeds x 2 bras x {p=1, p=3} + diversité
python3 tools/compare_weighting.py \
    uniform:runs/uniform_metrics.jsonl snr:runs/snr_g1_metrics.jsonl

# l'ablation de magnitude du §5, sur le checkpoint L 10k déjà publié
for m in 0.3 0.6 1.0; do
  for s in 1 2 3 4 5 6 7 8; do
    ./target/release/main --headless-sample Greyscale_Diffusion_L \
      --checkpoint Models/Greyscale_Diffusion_L/pretrained_weights/scale_run.ckpt \
      --seed $s --paths 1 --magnitude $m --out weighting_samples/ablate_mag_$m/seed_$s.png
  done
  python3 tools/sample_diversity.py --glob "weighting_samples/ablate_mag_$m/seed_*.png"
done
```

`runs/` et `*_metrics.jsonl` sont gitignorés : les scripts sont versionnés, les
métriques brutes non.
