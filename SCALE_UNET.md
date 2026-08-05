# SCALE_UNET — agrandir le U-Net de diffusion (Greyscale_Diffusion_L)

Branche `unet-scale`. Le pipeline est sain (`AUDIT_TRAINING.md`, `FIX_TRAINING.md`,
`INSIGHTS_TRAINING.md`) : un run de 10 000 pas converge, la loss batch tombe à
0,0152 et les images sont bornées. Mais le modèle **ne génère qu'une seule
image** — la « moyenne du dataset » — quel que soit le seed, avec un banding
horizontal. Hypothèse à tester : le modèle manque de capacité.

---

## TL;DR

- **L'hypothèse capacité est réfutée.** Un U-Net **11,7× plus gros** en calcul,
  **24× plus gros** en paramètres, avec un étage de plus et un champ réceptif qui
  couvre enfin l'image, entraîné 10 000 pas dans les mêmes conditions, donne une
  **diversité inter-seeds strictement identique** : 0,00299 contre 0,00329 pour
  la baseline (dataset : 0,2316). Il est même **moins bon sur les quatre tranches
  de `t`**.
- **Un seul gain réel** : le **banding chute de 33 %** (ratio 2,41 → 1,62 ;
  4,65 → 2,88 sans moyennage). C'est exactement ce que prédisait l'argument du
  champ réceptif (13 px → 35 px) — donc cette partie de l'hypothèse tient, mais
  elle ne gouvernait pas la diversité.
- **La vraie cause, mesurée** : à `t` élevé — là où la chaîne de génération
  **commence** et où le contenu global de l'image se décide — **l'erreur du
  modèle vaut 6,1× l'amplitude totale du signal de contenu** présent dans la
  cible (7,4× pour L). Les deux modèles y sont **plus mauvais qu'un prédicteur
  trivial à zéro paramètre** `ε̂ = x_t` (0,0142 et 0,0209 contre 0,0004, soit 36×
  et 52× pire). Ils n'apportent aucune information exploitable au moment où le
  contenu se choisit ; quand la chaîne atteint le `t` bas où ils excellent
  (erreur = 0,05× signal), l'image est déjà figée sur la moyenne.
- **Le « haut-`t` quasi parfait » de `INSIGHTS_TRAINING` est un artefact de
  métrique**, pas un apprentissage : à `t` élevé recopier l'entrée suffit à
  obtenir une loss minuscule. Et la loss « catastrophique » à `t` bas (0,61)
  correspond en réalité à une reconstruction de l'image à **1,1 % près** — c'est
  le meilleur régime des deux modèles, pas le pire.
- **Prochaine itération** : ne pas ré-agrandir. Corriger la **pondération de la
  loss** (ou changer de paramétrisation : `x₀` ou `v`) pour que le haut-`t` porte
  enfin un gradient utile, et remplacer le **SGD nu** par un optimiseur à
  momentum. Détail en §6.

---

## 1. La baseline, chiffrée

### 1.1 Convergence (`fable_run_metrics.jsonl`, 10 000 pas, lr 1e-3, batch 16)

| step | t[0-64) | t[64-128) | t[128-192) | t[192-256) | ε̂ std (haut-t) |
|-----:|--------:|----------:|-----------:|-----------:|----------------:|
| 0    | 1,1317  | 1,1628    | 1,1754     | 1,1847     | 0,496 |
| 1000 | 0,6646  | 0,1023    | 0,0560     | 0,0530     | 0,989 |
| 3000 | 0,6273  | 0,0839    | 0,0343     | 0,0287     | 0,986 |
| 6000 | 0,6047  | 0,0744    | 0,0267     | 0,0196     | 0,972 |
| 9999 | **0,6103** | 0,0644 | 0,0219     | 0,0142     | 0,980 |

La tranche basse plafonne à ~0,61 après ~4000 pas et **remonte même** entre 7000
et 10 000 (0,6009 → 0,6103). Interprétation retenue à l'époque : saturation de
capacité. **§5 montre que cette lecture est fausse.**

### 1.2 Le collapse de diversité est une mesure, pas une impression

8 échantillons, seeds 1..8, réglages d'inférence du config (`paths=3`,
`magnitude=0.3`). Repère : 16 images **réelles** du dataset.

| mesure | baseline 10k | dataset CIFAR-10 | rapport |
|---|---:|---:|---:|
| `inter_seed_std` | **0,00329** | 0,2316 | **70× moins** |
| `mean_pairwise_rmse` | 0,00479 | 0,3247 | 68× moins |
| `intra_image_std` | 0,0794 | 0,2061 | 2,6× moins |
| `banding_ratio` (Δrangées/Δcolonnes) | **2,414** | 1,074 | banding horizontal |

### 1.3 Contrôle : ce n'est pas le sampler

Hypothèse concurrente écartée avant d'accuser le modèle. `paths=3` moyenne trois
trajectoires et `magnitude=0.3` réduit le bruit injecté à 30 % — les deux
pourraient écraser la diversité par construction. Lecture du code
(`metrics.rs:332-337`) : le seed fixe le latent initial `x_T`, et les trois
chemins partagent **ce même `x_T`**. Vérifié plutôt que supposé, mêmes seeds à
`paths=1` :

| | `inter_seed_std` | `banding_ratio` |
|---|---:|---:|
| `paths=3` | 0,00329 | 2,414 |
| `paths=1` | 0,00587 | 4,655 |
| dataset | 0,2316 | 1,074 |

Sans moyennage la diversité ne gagne qu'un facteur 1,8 et reste **39× sous le
dataset** : le collapse est bien le modèle. Le banding, lui, **empire** — le
moyennage en cachait une partie.

---

## 2. L'architecture L et pourquoi

`Models/Greyscale_Diffusion_L/config_file`, **généré** par
`tools/gen_unet_config.py` (le modèle baseline n'est pas touché).

### 2.1 Pourquoi un générateur plutôt qu'un JSON écrit à la main

`Layer::new` propage la sortie de la couche précédente comme `dim_input`, donc
les `dim_input` du JSON sont documentaires — **sauf `dim_kernel.z`** :

- `ConvolutionType::set_dim_output` rejette l'écart au build
  (`KernelChannelMismatch`).
- **`UpsampleConvType::set_dim_output` ne le rejette pas.** Son shader indexe les
  poids avec `IC = dim_input.z` (`upsample_conv.wgsl:62,81`) : un noyau mal
  profilé passe le build et **corrompt silencieusement** la convolution.

Le générateur dérive `dim_kernel.z` de la dim courante et vérifie les contraintes
(`dim.z % num_groups == 0`, concordance spatiale des `Concat`).

### 2.2 Ce qui change

| | baseline | L | facteur |
|---|---:|---:|---:|
| échelles | 32 → 16 | 32 → 16 → **8** | +1 étage |
| noyaux | 16 / 32 | **32 / 64 / 128** | 2–4× |
| canaux d'embedding temporel | 2 | **4** (entrée `[32,32,5]`) | 2× |
| skips `Concat` | 1 | **2** | |
| couches | 12 | 28 | 2,3× |
| MACs/échantillon | 9,0 M | **105,6 M** | **11,7×** |
| taille du checkpoint | 77 Ko | **1,86 Mo** | **24×** |
| débit mesuré (batch 16) | 316 ms/pas | 1299 ms/pas | 4,1× |

Le temps (4,1×) croît bien plus lentement que le calcul (11,7×) : à cette taille
le pas reste dominé par l'overhead de dispatch.

### 2.3 Justification — le champ réceptif

`tools/receptive_field.py` déroule la récurrence RF/jump sur les deux configs :

```
baseline : 3 -> 5 -> 9 -> 11 -> 13 px    -> 13 px  NE COUVRE PAS l'image (32 px)
L        : 3 -> 5 -> 9 -> 13 -> 21 -> 25 -> 29 -> 31 -> 33 -> 35 px
                                         -> 35 px  couvre l'image
```

**Chaque pixel de sortie de la baseline est décidé par une fenêtre de 13 px sur
32.** Un modèle qui ne voit pas globalement ne peut pas *choisir* un contenu
global : il applique le même prior local partout, et le minimiseur MSE de ce
prior est la moyenne du dataset. L'argument prédisait aussi le banding (sans vue
d'ensemble, la seule structure possible est un gradient basse-fréquence).

**Ce raisonnement s'est avéré à moitié juste** : il explique le banding (§4),
pas la diversité (§5).

Les deux autres leviers : 2–4× de noyaux (16 descripteurs au premier étage, c'est
peu pour CIFAR-10) et 4 canaux temporels au lieu de 2 (deux paires
`[sin(2ⁱπτ), cos(2ⁱπτ)]` au lieu d'une, pour lire `t` plus finement).

---

## 3. Entraînement comparatif

10 000 pas, lr 1e-3, batch 16, from scratch, `cifar10_grey.batraw` — **exactement
les hyperparamètres de la baseline**.

| tranche | baseline (fin) | L (fin) | baseline (meilleur) | L (meilleur) |
|---|---:|---:|---:|---:|
| t[0-64)    | **0,6103** | 0,6211 | **0,5951** | 0,6038 |
| t[64-128)  | **0,0644** | 0,0696 | **0,0636** | 0,0696 |
| t[128-192) | **0,0219** | 0,0276 | **0,0211** | 0,0273 |
| t[192-256) | **0,0142** | 0,0209 | **0,0138** | 0,0201 |

**L est moins bon partout**, à la fin comme au meilleur de chaque run. L'écart
s'installe dès ~500 pas et reste ensuite quasi constant (t0-64 : ~0,009 ;
haut-`t` : facteur ~1,5). L n'est jamais en train de rattraper.

Deux éléments de la machinerie d'entraînement expliquent cette lenteur — constats
de lecture de code, sans modification (un autre agent travaille sur les shaders) :

1. **Optimiseur = SGD nu** (`sgd.wgsl` : `w -= lr * grad`). Pas de momentum, pas
   d'adaptatif, lr fixe. Les gradients sont bien moyennés sur le batch
   (`model.rs:289`). C'est le régime où la profondeur coûte le plus cher en
   vitesse de convergence : 28 couches contre 12.
2. **Initialisation aveugle au fan-in** (`layer.rs:744`) : uniforme(±0,1), écart-type
   0,0577 quelle que soit la couche. Rapport à He (`√(2/fan_in)`) — baseline :
   0,21× 0,49× 0,69× 0,69× 0,69× ; L : 0,27× 0,69× 0,98× 0,98× **1,39×** …
   Les `GroupNorm` absorbent l'essentiel côté forward ; l'échelle des gradients,
   elle, n'est pas renormalisée.

**Mais ces deux points n'expliquent pas le résultat de §4 et §5** : même en
supposant L simplement sous-entraîné, la baseline, elle, est bel et bien
convergée — et c'est *elle* qui définit le plafond que L n'a pas franchi.

---

## 4. Qualité et diversité — le résultat

8 échantillons par configuration, seeds 1..8. Planche :
**`scale_samples/comparison.png`** (baseline / L / baseline `p=1` / L `p=1` /
images réelles).

| | `inter_seed_std` | `mean_pairwise_rmse` | `banding_ratio` | `intra_image_std` |
|---|---:|---:|---:|---:|
| baseline `p=3` | 0,00329 | 0,00479 | 2,414 | 0,0794 |
| **L `p=3`** | **0,00299** | **0,00441** | **1,623** | 0,0521 |
| baseline `p=1` | 0,00587 | 0,00848 | 4,655 | 0,1134 |
| **L `p=1`** | **0,00589** | **0,00854** | **2,875** | 0,1078 |
| dataset réel | 0,2316 | 0,3247 | 1,074 | 0,2061 |

**Diversité : aucun gain.** 0,00299 contre 0,00329 (et 0,00589 contre 0,00587 à
`paths=1` — identique à la troisième décimale). Toujours **~70× sous le dataset**.
Sur la planche, les huit colonnes de la rangée L sont indiscernables entre elles,
exactement comme la baseline.

**Banding : gain net et reproductible.** 2,414 → 1,623 (−33 %) et 4,655 → 2,875
(−38 %), dans le sens du dataset (1,074). C'est le seul effet mesurable de
l'agrandissement, et il **valide l'argument du champ réceptif** de §2.3 : avec une
vue globale, le réseau cesse de ne produire que des bandes horizontales. Il ne
produit pas pour autant du contenu.

---

## 5. Pourquoi — ce que la loss ε mesure réellement

L'agrandissement n'ayant rien donné sur la diversité, la question devient : que
mesure-t-on exactement ? Deux calculs, l'un analytique, l'autre empirique,
suffisent à retourner la lecture des métriques.

### 5.1 La loss ε est une loupe à facteur variable (`tools/eps_metric_analysis.py`)

Identité exacte : puisque `ε = (x_t − √ᾱ·x₀)/√(1−ᾱ)`, une erreur de
reconstruction `x̂₀ − x₀` produit

```
MSE(ε̂, ε) = [ ᾱ / (1−ᾱ) ] · MSE(x̂₀, x₀)
```

Le facteur explose quand `t → 0`. Avec le schedule de production (T=256) et le
protocole de tirage du probe (`metrics.rs:213-221`, t = t_lo et t_lo+span/2) :

| tranche | `t` tirés | amplification moyenne | loss baseline | **RMSE(x₀) implicite** |
|---|---|---:|---:|---:|
| t[0-64)    | 0, 32   | **1282,1** (dont t=0 : 2559) | 0,6103 | **0,0218 → 1,1 % de la plage** |
| t[64-128)  | 64, 96  | 0,7 | 0,0644 | 0,308 → 15,4 % |
| t[128-192) | 128,160 | 0,05 | 0,0219 | 0,670 → 33,5 % |
| t[192-256) | 192,224 | 0,003 | 0,0142 | 2,947 → 147 % |

**La lecture habituelle est inversée.** La tranche « catastrophique » à 0,61
correspond à une reconstruction de l'image propre à **1,1 % près** : c'est le
meilleur régime du modèle. Les tranches « quasi parfaites » à 0,014 correspondent
à une connaissance de `x₀` d'erreur **147 %** — c'est-à-dire aucune.

### 5.2 Les modèles sont battus par un prédicteur à zéro paramètre (`tools/trivial_baselines.py`)

Calculé sur 256 vraies images du dataset, même schedule, même protocole de tirage :

| tranche | baseline | L | `ε̂ = x_t` (0 param.) | `ε̂` via image moyenne |
|---|---:|---:|---:|---:|
| t[0-64)    | **0,6103** | 0,6211 | 0,8735 | 294,7 |
| t[64-128)  | **0,0644** | 0,0696 | 0,1402 | 0,1562 |
| t[128-192) | 0,0219 | 0,0276 | **0,0114** | 0,0112 |
| t[192-256) | 0,0142 | 0,0209 | **0,0004** | 0,0004 |

Aux deux tranches hautes, **les deux modèles entraînés sont moins bons que
recopier l'entrée** : 0,0142 contre 0,0004, soit **36× pire** (52× pour L). Le
« haut-`t` quasi parfait » célébré dans `INSIGHTS_TRAINING` §4.1 ne démontre donc
**aucun apprentissage** — c'est simplement le régime où `x_t ≈ ε`.

### 5.3 Le mécanisme du collapse

Dans la cible `ε`, la part qui porte le **contenu** de l'image vaut
`√ᾱ·x₀/√(1−ᾱ)`. C'est tout ce qu'un modèle peut extraire à ce `t`. Comparé à
l'erreur effective des modèles :

| tranche | RMS du signal de contenu | erreur baseline | erreur L |
|---|---:|---:|---:|
| t[0-64)    | 17,30  | **0,05× le signal** | 0,05× |
| t[64-128)  | 0,398  | 0,64× | 0,66× |
| t[128-192) | 0,107  | 1,39× | 1,56× |
| t[192-256) | 0,0195 | **6,10× le signal** | **7,40×** |

**Voilà le collapse.** L'échantillonnage part de `t = 255` et descend : c'est
pendant les premières étapes, à `t` élevé, que le contenu global de l'image se
choisit. Or à `t` élevé l'erreur du modèle vaut **6 fois l'intégralité du signal
de contenu** disponible — sa sortie n'y contient rien d'exploitable sur *quelle*
image produire. Il ne peut que suivre le seul attracteur compatible avec toutes
les données : la moyenne. Quand la chaîne atteint enfin le `t` bas où le modèle
est excellent (erreur 0,05× signal), le contenu est déjà figé, et le modèle ne
fait plus qu'affiner proprement… une image moyenne.

Et **L est pire que la baseline exactement là où ça compte** (7,40× contre
6,10×) : agrandir a dégradé le régime critique.

La raison en est structurelle et indépendante de l'architecture : la loss MSE sur
`ε` non pondérée accorde à la tranche `t[192-256)` un poids ~0,003 fois celui de
la tranche `t[0-64)` (§5.1). **Le gradient qui devrait apprendre le contenu de
l'image est écrasé d'un facteur ~10⁵ par rapport au gradient qui affine des
détails déjà acquis.** Aucune quantité de capacité ne compense une pondération
pareille — ce qui est précisément ce que ce run démontre.

---

## 6. Verdict et prochaine itération

### Verdict

**L'hypothèse « le modèle manque de capacité » est réfutée.** 11,7× de MACs, 24×
de paramètres, un étage de plus et un champ réceptif porté de 13 à 35 px ne
changent **rien** à la diversité (0,00299 vs 0,00329, à ~70× du dataset) et
dégradent la loss sur les quatre tranches. Le seul gain est le **banding, −33 %**,
qui confirme la partie « champ réceptif » du raisonnement sans toucher au
problème principal.

Le facteur limitant est la **pondération de l'objectif**, pas la capacité :
à `t` élevé — où le contenu se décide — les deux modèles sont plus mauvais qu'un
prédicteur à zéro paramètre, avec une erreur valant 6 à 7× le signal de contenu.

`Models/Greyscale_Diffusion_L/` est conservé (config + checkpoint 10 000 pas)
comme témoin de ce résultat négatif. La baseline reste intacte.

### Prochaine itération — par ordre de rapport gain/coût

1. **Pondérer la loss par tranche de `t`, ou changer de paramétrisation.**
   C'est la cause prouvée. Trois options, de la moins à la plus invasive :
   - pondération `min(SNR, γ)` ou simplement `1/amplification(t)` appliquée au
     gradient de la loss — quelques lignes dans `optimizer.wgsl`/le calcul de
     `grad_output`, aucun changement d'architecture ;
   - **prédiction de `x₀`** au lieu de `ε` : la cible devient l'image, dont le
     signal est d'amplitude constante à tout `t`. Supprime l'amplification par
     construction, et supprime aussi l'amplification `1/√α` à l'échantillonnage
     déjà identifiée dans `INSIGHTS_TRAINING` §2.2 ;
   - **prédiction de `v`** (`v = √ᾱ·ε − √(1−ᾱ)·x₀`), le compromis standard.

   Critère de succès, mesurable avec l'outillage de ce rapport : l'erreur à
   `t[192-256)` doit passer **sous 1× le signal de contenu**
   (`tools/trivial_baselines.py`), et battre `ε̂ = x_t`.

2. **Remplacer le SGD nu par un optimiseur à momentum** (`sgd.wgsl` +
   un buffer de vitesse par couche). Indépendant du point 1, et prérequis pour
   que tout modèle profond soit entraînable ici dans un budget de pas raisonnable.

3. **Initialisation dépendante du fan-in** (He) dans `Layer::init_random_weights` —
   correction à trois lignes.

4. **Ne pas ré-agrandir tant que 1 et 2 ne sont pas faits.** Une fois le haut-`t`
   porteur de gradient, l'architecture L (champ réceptif 35 px, banding déjà
   −33 %) redevient le bon candidat — mais le tester avant serait refaire ce run.

### Ce que ce rapport ne prouve pas

- Que L serait mauvais avec un objectif corrigé : il a été entraîné sous
  l'objectif défaillant, comme la baseline. Son désavantage à budget de pas égal
  (§3) est vraisemblablement dû au SGD, pas à l'architecture.
- Que la pondération suffira. Elle traite la cause mesurée ici ; elle ne dit rien
  d'un éventuel second facteur limitant qui n'apparaîtra qu'une fois celui-ci levé.

---

## 7. Reproduire

```bash
# architecture (dims dérivées, pas écrites à la main)
python3 tools/gen_unet_config.py > Models/Greyscale_Diffusion_L/config_file
python3 tools/receptive_field.py Models/Greyscale_Diffusion/config_file \
                                 Models/Greyscale_Diffusion_L/config_file

# entraînement (~3 h 20 à 1,2 s/pas)
cargo run --release -p main -- --headless-train Greyscale_Diffusion_L \
    --steps 10000 --dataset datasets/cifar10_grey.batraw --lr 1e-3 --batch 16 \
    --out Models/Greyscale_Diffusion_L/pretrained_weights/scale_run.ckpt

# 8 échantillons de seeds différents
for s in 1 2 3 4 5 6 7 8; do
  cargo run --release -p main -- --headless-sample Greyscale_Diffusion_L \
      --checkpoint Models/Greyscale_Diffusion_L/pretrained_weights/scale_run.ckpt \
      --seed $s --paths 3 --magnitude 0.3 --out scale_samples/L/seed_$s.png
done

# mesures
python3 tools/sample_diversity.py --glob 'scale_samples/L/seed_*.png' --label 'L 10k'
python3 tools/sample_diversity.py --dataset datasets/cifar10_grey.batraw --n 16
python3 tools/compare_metrics.py baseline:<base_metrics.jsonl> \
    L:Models/Greyscale_Diffusion_L/pretrained_weights/scale_run_metrics.jsonl
python3 tools/eps_metric_analysis.py --label baseline --losses 0.6103 0.0644 0.0219 0.0142
python3 tools/trivial_baselines.py          # prédicteurs 0-paramètre + signal/erreur
python3 tools/compare_plate.py --out scale_samples/comparison.png \
    --row 'baseline 10k:scale_samples/baseline/seed_*.png' \
    --row 'L 10k:scale_samples/L/seed_*.png' \
    --dataset-row 'CIFAR-10 reel:datasets/cifar10_grey.batraw'
```

> Note sur l'outillage : les lecteurs `.batraw` de `tools/` lisent le payload en
> **f32** (`main.rs:936-941`) et appliquent `v*2−1` pour `BATRAW1`, comme le
> chargeur de production. Une première version les lisait en `u8` — les repères
> dataset en étaient faussés (détecté parce que la rangée dataset de la planche
> sortait en bruit ; 204 800 024 o = 50000·1024·**4** + 24). Les mesures sur les
> échantillons, lues depuis les PNG, n'étaient pas concernées.
