# SCALE_UNET — agrandir le U-Net de diffusion (Greyscale_Diffusion_L)

Branche `unet-scale`. Le pipeline est sain (`AUDIT_TRAINING.md`, `FIX_TRAINING.md`,
`INSIGHTS_TRAINING.md`) : un run de 10 000 pas converge, la loss batch tombe à
0,0152 et les images sont bornées. Mais le modèle **ne génère qu'une seule
image** — la « moyenne du dataset » — quel que soit le seed, avec un banding
horizontal. Ce rapport agrandit l'architecture et mesure si la capacité était
bien le facteur limitant.

> **Statut : run L en cours.** Les sections 3 à 6 sont remplies au fil des
> mesures ; les sections 1 et 2 (baseline chiffrée, architecture) sont finales.

---

## 1. La baseline, chiffrée

### 1.1 Convergence (`fable_run_metrics.jsonl`, 10 000 pas, lr 1e-3, batch 16)

La loss batch (0,0152) ne dit rien d'utile ; le signal est la loss `MSE(ε̂, ε)`
**par tranche de `t`** (cf. `INSIGHTS_TRAINING` §2.1) :

| step | t[0-64) | t[64-128) | t[128-192) | t[192-256) | ε̂ std (haut-t) |
|-----:|--------:|----------:|-----------:|-----------:|----------------:|
| 0    | 1,1317  | 1,1628    | 1,1754     | 1,1847     | 0,496 |
| 1000 | 0,6646  | 0,1023    | 0,0560     | 0,0530     | 0,989 |
| 3000 | 0,6273  | 0,0839    | 0,0343     | 0,0287     | 0,986 |
| 6000 | 0,6047  | 0,0744    | 0,0267     | 0,0196     | 0,972 |
| 9999 | **0,6103** | 0,0644 | 0,0219     | 0,0142     | 0,980 |

Le haut-`t` est quasi résolu (0,0142). **La tranche basse plafonne à ~0,61 et ne
bouge plus après ~4000 pas** — elle remonte même légèrement entre 7000 et 10000
(0,6009 → 0,6103). C'est le symptôme d'une **saturation de capacité**, pas d'un
manque de pas : à `t` bas le réseau doit distinguer une structure d'image fine,
ce que 16/32 noyaux sur une seule échelle ne permettent pas.

### 1.2 Le collapse de diversité est une mesure, pas une impression

8 échantillons, seeds 1..8, réglages d'inférence du config (`paths=3`,
`magnitude=0.3`), via `--headless-sample` sur le checkpoint 10 000 pas.
Repère : les mêmes statistiques sur 16 images **réelles** du dataset.

| mesure | baseline 10k | dataset CIFAR-10 | rapport |
|---|---:|---:|---:|
| `inter_seed_std` (écart-type pixel-à-pixel entre seeds) | **0,00329** | 0,2170 | **66× moins** |
| `mean_pairwise_rmse` | 0,00479 | 0,3559 | 74× moins |
| `min_pairwise_rmse` | 0,00337 | 0,2917 | — |
| `intra_image_std` (contraste interne) | 0,0794 | 0,2735 | 3,4× moins |
| `banding_ratio` (Δrangées / Δcolonnes) | **2,414** | 0,877 | banding horizontal |

Lecture :

- **Diversité ≈ 0.** Deux échantillons de seeds différents diffèrent de 0,5 % de
  la dynamique. Le modèle a appris **une** sortie, pas une distribution.
- **Banding objectivé.** Sur des images naturelles, les variations rangée-à-rangée
  et colonne-à-colonne sont du même ordre (ratio 0,88). Ici les variations
  verticales sont **2,4×** les horizontales : la sortie est structurée en bandes
  horizontales. `intra_image_std` = 0,079 dont l'essentiel est ce banding — le
  peu de « détail » produit **est** l'artefact.

Outils : `tools/sample_diversity.py`, images dans `scale_samples/baseline/`.

---

## 2. L'architecture L et pourquoi

`Models/Greyscale_Diffusion_L/config_file`, **généré** par
`tools/gen_unet_config.py` (le modèle baseline n'est pas touché).

### 2.1 Pourquoi un générateur plutôt qu'un JSON écrit à la main

`Layer::new` propage la sortie de la couche précédente comme `dim_input`, donc
les `dim_input` du JSON sont surtout documentaires — **sauf `dim_kernel.z`**, qui
doit égaler les canaux d'entrée réels :

- `ConvolutionType::set_dim_output` rejette l'écart au build
  (`KernelChannelMismatch`) — ce garde-fou existe parce que ce piège exact avait
  rendu inopérant le premier essai de conditionnement temporel.
- **`UpsampleConvType::set_dim_output` ne le rejette pas.** Son shader indexe les
  poids avec `IC = dim_input.z` (`upsample_conv.wgsl:62,81`) : un noyau mal
  profilé y passe le build et **corrompt silencieusement** la convolution.

Le générateur dérive `dim_kernel.z` de la dim courante et vérifie les contraintes
(`dim.z % num_groups == 0`, concordance spatiale des `Concat`), ce qui rend ces
deux erreurs impossibles par construction.

### 2.2 Ce qui change

| | baseline | L | facteur |
|---|---:|---:|---:|
| échelles | 32 → 16 | 32 → 16 → **8** | +1 étage |
| noyaux | 16 / 32 | **32 / 64 / 128** | 2–4× |
| canaux d'embedding temporel | 2 | **4** | entrée `[32,32,5]` |
| skips `Concat` | 1 | **2** | |
| couches | 12 | 28 | 2,3× |
| MACs / échantillon (forward) | 9,0 M | **105,6 M** | **11,7×** |
| débit mesuré (batch 16) | 316 ms/pas | **1299 ms/pas** | 4,1× |

Le coût en temps (4,1×) croît bien plus lentement que le coût arithmétique
(11,7×) : à cette taille le pas est encore largement dominé par l'overhead de
dispatch, pas par le calcul. Agrandir le modèle est donc « bon marché » ici.

Graphe (28 couches, blocs pré-activation `GroupNorm → SiLU → Conv` comme la
baseline) :

```
[32,32,5]  Conv 32                      -> [32,32,32]
           GN(8) -> SiLU                              == skip32
           Conv 64 s2                   -> [16,16,64]
           GN -> SiLU -> Conv 64        -> [16,16,64]
           GN -> SiLU                                 == skip16
           Conv 128 s2                  -> [8,8,128]
           GN -> SiLU -> Conv 128       -> [8,8,128]   (bottleneck)
           GN -> SiLU
           UpsampleConv x2 -> 64        -> [16,16,64]
           Concat skip16                -> [16,16,128]
           GN -> SiLU -> Conv 64        -> [16,16,64]
           UpsampleConv x2 -> 32        -> [32,32,32]
           Concat skip32                -> [32,32,64]
           GN -> SiLU -> Conv 32        -> [32,32,32]
           GN -> SiLU -> Conv 1         -> [32,32,1]
```

Justification des trois leviers :

1. **Un étage de plus (8×8) — l'argument central.** `tools/receptive_field.py`
   calcule le champ réceptif le long des deux configs :

   ```
   baseline : 3 -> 5 -> 9 -> 11 -> 13 px   -> 13 px  NE COUVRE PAS l'image (32 px)
   L        : 3 -> 5 -> 9 -> 13 -> 21 -> 25 -> 29 -> 31 -> 33 -> 35 px
                                            -> 35 px  couvre l'image
   ```

   **Chaque pixel de sortie de la baseline est décidé par une fenêtre de 13 px
   sur 32** — moins d'un tiers de l'image, jamais l'image entière. Un modèle qui
   ne voit pas globalement ne peut pas *choisir* un contenu global : il ne peut
   qu'appliquer le même prior local partout, et le minimiseur MSE de ce prior est
   la moyenne du dataset. C'est l'explication la plus directe et la plus
   structurelle du collapse — et elle prédit aussi le **banding** : sans vue
   d'ensemble, la seule structure que le réseau peut poser est un gradient
   basse-fréquence, qui apparaît en bandes.

   L'étage 8×8 ajouté porte le champ réceptif à 35 px, au-delà de l'image.
2. **2–4× de noyaux.** 16 canaux au premier étage, c'est un budget de 16
   descripteurs pour toute la variété de CIFAR-10 ; la sortie moyenne est le
   minimiseur MSE quand on ne peut pas représenter mieux.
3. **4 canaux temporels au lieu de 2.** Les paires sont `[sin(2ⁱπτ), cos(2ⁱπτ)]` ;
   avec 2 canaux, une seule paire, donc une résolution grossière de `t`. Passer
   à 4 ajoute la paire de fréquence double — utile précisément là où `t` doit
   être lu finement (bas `t`, où `√(1−ᾱ)` varie vite).

Les skips sont pris **après l'activation** (et non après la conv comme la
baseline) : le décodeur reçoit ainsi la même représentation que celle qui a servi
au downsampling.

---

## 3. Entraînement comparatif

*(en cours — 10 000 pas, lr 1e-3, batch 16, mêmes hyperparamètres que la
baseline, dataset `cifar10_grey.batraw`, from scratch)*

---

## 4. Qualité et diversité

*(à venir)*

---

## 5. Verdict

*(à venir)*

---

## 6. Reproduire

```bash
# architecture (regénère le config, dims dérivées)
python3 tools/gen_unet_config.py > Models/Greyscale_Diffusion_L/config_file

# entraînement
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
python3 tools/compare_metrics.py \
    baseline:<baseline_metrics.jsonl> \
    L:Models/Greyscale_Diffusion_L/pretrained_weights/scale_run_metrics.jsonl
```
