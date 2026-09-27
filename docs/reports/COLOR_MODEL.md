# COLOR_MODEL — premier modèle de diffusion couleur (CIFAR-10 RGB 32×32)

Branche `color-model` · Plateforme macOS (Darwin 25.4.0, Apple M5 Pro) · 5 août 2026

## TL;DR

- **Le chemin couleur marchait déjà de bout en bout — zéro ligne de pipeline à
  écrire.** Chargeur `.batraw`, `diffusion_prepare`, loss, chaîne inverse, encodage
  PNG : tout est piloté par `Dim3.z`, aucune constante « 1 canal » nulle part. Ce
  qui manquait, c'était un **config** correct (le seul modèle « couleur »
  préexistant, `Stable_Diffusion`, a **0 canal temporel** — le cas dégénéré que
  `AGENTS.md` interdit).
- **Architecture retenue : `Color_Diffusion_L` (32/64/128)**, entrée `[32,32,7]`
  (3 image + 4 temps), sortie `[32,32,3]`, champ réceptif **35 px ≥ 32**.
  Mesuré contre une variante XL (48/96/192) : XL coûte **2,18× par pas** pour
  l'axe « capacité » que `SCALE_UNET.md` a déjà **réfuté**. L achète 2,2× plus de
  pas d'optimisation pour le même mur d'horloge.
- **Débit : 1271 ms/pas** (pur) / **1280 ms/pas** (effectif, échantillonnage
  périodique inclus) à batch 16 — soit **+0 % par rapport au greyscale L**
  (1299 ms/pas dans `SCALE_UNET.md`). La couleur est gratuite à cette taille.
- **Smoke 600 pas : sain sur les quatre tranches, et déjà meilleur que le
  greyscale L à 10 000 pas** sur les quatre — c'est Adam, pas la couleur.
  `non_finite` = 0 partout, images bornées, trois canaux réellement distincts.
- **Run de nuit recommandé : 20 000 pas ≈ 7 h 07.** Commande exacte en §6.

---

## 1. Dataset

`datasets/cifar10_rgb.batraw` — **50 000 × 32 × 32 × 3**, magic `BATRAW1`
(payload en `[0,1]`, rééchelonné en `[-1,1]` au chargement par
`main.rs:1869-1873`), **585,9 MiB**, hors git (`datasets/*.batraw` est déjà dans
le `.gitignore`).

```bash
python3 datasets/cifar_to_raw.py --cifar-dir datasets/cifar-10-batches-bin \
        --mode rgb --out datasets/cifar10_rgb.batraw
```

**`BATRAW1` plutôt que `BATRAW2`** : le rééchelonnage `v*2−1` est appliqué par le
chargeur de production et couvert par un test existant
(`raw_dataset_legacy_unit_payload_is_rescaled_to_signed_range`). Produire du
`BATRAW2` aurait demandé de toucher le script d'export pour un résultat
strictement identique en mémoire — changement gratuit, risque non nul, la veille
d'un run long. Écarté.

**Le script était trop lent en RGB** : 153,6 M de f32 construits par 3072 `append`
Python par image × 50 000. Ajout d'un chemin numpy (calcul en float64 puis
abaissement en float32, comme `struct.pack`), **vérifié bit-à-bit identique** au
chemin pur Python sur 20 records en `grey` et en `rgb`. Le pur Python reste le
fallback si numpy est absent. **0,5 s** au lieu de plusieurs minutes.

Repères du dataset couleur (16 images, `tools/sample_diversity.py`) — c'est
l'échelle contre laquelle juger tout échantillon :

| mesure | CIFAR-10 RGB |
|---|---:|
| `inter_seed_std` | 0,2457 |
| `mean_pairwise_rmse` | 0,3464 |
| `intra_image_std` | 0,2251 |
| `banding_ratio` | 1,085 |
| **`chroma_std`** | **0,0707** |
| moyennes R/G/B | 0,447 / 0,446 / 0,402 |

---

## 2. Architecture

`Models/Color_Diffusion_L/config_file`, **généré** par
`tools/gen_unet_config.py` (rendu paramétrable : `--name`,
`--signal-channels`, `--widths` ; ses défauts reproduisent
`Greyscale_Diffusion_L` **à l'octet près**, vérifié par diff JSON).

```bash
python3 tools/gen_unet_config.py --name Color_Diffusion_L \
        --signal-channels 3 --widths 32 64 128 > Models/Color_Diffusion_L/config_file
python3 tools/receptive_field.py Models/Color_Diffusion_L/config_file
```

| | greyscale L | **Color_Diffusion_L** | Color_Diffusion_XL |
|---|---:|---:|---:|
| entrée | `[32,32,5]` | **`[32,32,7]`** | `[32,32,7]` |
| sortie | `[32,32,1]` | **`[32,32,3]`** | `[32,32,3]` |
| largeurs | 32/64/128 | **32/64/128** | 48/96/192 |
| MMACs/échantillon | 105,6 | **106,8** | 238,0 |
| champ réceptif | 35 px | **35 px** | 35 px |
| **ms/pas mesuré** | 1299 | **1271** | **2768** |

Le générateur dérive `dim_kernel.z` de la dim courante et vérifie
`dim.z % num_groups == 0` — c'est ce qui rend impossibles à la fois le
`KernelChannelMismatch` (rejeté au build côté `Convolution`) et la **corruption
silencieuse** côté `UpsampleConv`, qui ne rejette rien et indexe ses poids avec
`IC = dim_input.z`.

### Pourquoi L et pas XL

XL a été généré **et mesuré**, pas écarté sur intuition :

1. **Le prix est réel** : 2768 vs 1271 ms/pas, soit **2,18×** — proche du rapport
   de MACs (2,23×). Contrairement au greyscale L, XL n'est plus dominé par
   l'overhead de dispatch : on paie le calcul plein pot.
2. **L'axe est déjà réfuté.** `SCALE_UNET.md` a agrandi 11,7× en MACs et 24× en
   paramètres : **diversité inchangée** (0,00299 vs 0,00329) et loss dégradée sur
   les quatre tranches. Son §6.4 dit explicitement de **ne pas ré-agrandir** tant
   que la pondération de la loss n'est pas réglée — or `LOSS_WEIGHTING.md` est
   revenu **NO-GO**. Ce prérequis est **toujours ouvert**.
3. **Le seul gain démontré de l'agrandissement (banding −33 %) venait du champ
   réceptif**, et il est **déjà acquis en L** : 35 px ≥ 32.
4. **Les pas valent mieux que les paramètres ici** : à mur d'horloge égal, L offre
   **2,2× plus de pas** d'Adam.
5. **La couleur est la variable nouvelle.** Changer la géométrie en même temps
   confondrait les deux effets.

`Models/Color_Diffusion_XL/config_file` est **conservé** : il est prêt si le run
de nuit plafonne pour une raison de capacité plutôt que d'objectif.

---

## 3. Chemin couleur — ce qui a été vérifié

**Aucune correction de pipeline n'a été nécessaire.** Vérifications faites :

- **Convention mémoire.** `convolution.wgsl` est **HWC / z-le-plus-rapide**
  (`index = iy*W*C + ix*C + iz`), exactement la disposition entrelacée du
  `.batraw` et de `tensor_to_rgb_pixels`. C'est **la** propriété qu'un modèle à
  1 canal ne peut pas exercer : en C=1, entrelacé et planaire sont les mêmes
  octets. Vérifiée en lisant le shader, puis **figée par un test** (§5).
- **Chargeur** : `(width,height,channels) == output_size` → le payload est utilisé
  **verbatim**, sans permutation (`main.rs:1879-1884`).
- **Conditionnement temporel** : `diffusion_prepare.wgsl` prend `signal_channels`
  et `input_channels` comme **uniformes**, et compose `[R,G,B,t₀,t₁,t₂,t₃]` par
  pixel. `input.z − output.z = 7 − 3 = 4` canaux temporels.
- **Chaîne inverse** : les 256 pas de `--headless-sample` opèrent sur des tenseurs
  de **3072** éléments avec `non_finite: 0` à chaque pas.
- **PNG** : sortie `mode=RGB`, trois canaux mesurés distincts.
- **Bruit** : `gaussian_at` indexe le tenseur **plat**, donc chaque canal d'un
  pixel reçoit un tirage indépendant — comportement DDPM correct en C=3.

**Hors scope, noté comme demandé** : la vue live / `LiveFrame` n'a pas été
retouchée. Elle n'est de toute façon pas greyscale-only — `live_frame.rs` prend
`channels` en paramètre et `visualiser/shader.wgsl` bascule en RGB dès
`channels >= 3` ; et `--headless-sample` passe `observer = None`, donc aucun
`LiveFrame` n'est construit pendant un run headless.

**Piège préexistant signalé** : `Models/Stable_Diffusion/config_file` a
`input_size [32,32,3]` pour une sortie à 3 canaux — soit **zéro canal temporel**.
C'est le cas dégénéré documenté dans `AGENTS.md` (ε̂ dégénère, échantillonnage
saturé en blanc). **Il n'est pas utilisable comme modèle couleur.** Même défaut
dans `tui/app.rs::diffusion_template()`. Non corrigé ici : hors mission, et le
toucher la veille d'un run long n'apporte rien.

---

## 4. Smoke test — 600 pas, Adam lr 1e-3, batch 16, from scratch

```bash
./target/release/main --headless-train Color_Diffusion_L --steps 600 \
    --dataset datasets/cifar10_rgb.batraw --lr 1e-3 --batch 16 --optimizer adam \
    --out Models/Color_Diffusion_L/pretrained_weights/smoke600.ckpt
```

### 4.1 Loss par tranche de `t` — décroissance saine sur les quatre

| step | t[0-64) | t[64-128) | t[128-192) | t[192-256) | ε̂ std (haut-t) |
|-----:|--------:|----------:|-----------:|-----------:|---------------:|
| 0   | 1,2055 | 1,0642 | 1,0714 | 1,0611 | 0,442 |
| 100 | 0,6118 | 0,0972 | 0,0642 | 0,0622 | 0,948 |
| 200 | 0,5542 | 0,0660 | 0,0363 | 0,0327 | 0,972 |
| 400 | 0,5104 | 0,0490 | 0,0185 | 0,0140 | 0,976 |
| **599** | **0,4879** | **0,0423** | **0,0136** | **0,0094** | **0,992** |

Les quatre tranches décroissent **monotonement**, `non_finite` = 0 sur toutes les
sondes, et `std(ε̂) → 0,99` (cible 1,0).

**Repère** — greyscale L après **10 000** pas de SGD :
0,6211 / 0,0696 / 0,0276 / 0,0209. Le modèle couleur est **meilleur sur les
quatre tranches après 600 pas**. Ce n'est pas un effet de la couleur : c'est Adam
(`OPTIMIZER_ADAM.md` annonce ~20× en nombre de pas), et ça confirme surtout que
rien n'est cassé.

> Lecture obligatoire, rappelée ici pour la relecture de demain matin : d'après
> `SCALE_UNET.md` §5, ces chiffres en unités ε **inversent** l'importance des
> tranches. La tranche « catastrophique » à 0,49 est le **meilleur** régime du
> modèle (reconstruction de x₀ au %), et les tranches « excellentes » à 0,009
> sont celles où il ne sait **rien**. Convertir avec
> `tools/eps_metric_analysis.py` avant toute conclusion.

### 4.2 Échantillons — bornés, sans NaN, trois canaux distincts

4 seeds, `--paths 3 --magnitude 0.3`. Planche : `color_samples/smoke_plate.png`.

| mesure | smoke 600 pas | CIFAR-10 RGB |
|---|---:|---:|
| `final image` min / max | −0,93 / +0,35 | — |
| `intra_image_std` | 0,1211 | 0,2251 |
| **`chroma_std`** | **0,1119** | 0,0707 |
| moyennes R/G/B | 0,455 / 0,184 / 0,342 | 0,447 / 0,446 / 0,402 |
| `banding_ratio` | 0,682 | 1,085 |

**Les trois critères du smoke sont remplis** : pas de NaN (`non_finite` = 0 sur
les 256 pas de la chaîne), pas de saturation (max +0,35, loin de +1), et trois
canaux **réellement** distincts (`chroma_std` = 0,112, très au-dessus de 0).

Les images sont un mauve uniforme — l'attracteur « moyenne », en couleur. C'est
attendu à 600 pas et **ce n'est pas un verdict de qualité**.

**Un point à surveiller demain matin** — le canal vert est nettement en retrait
(0,184 contre 0,447 et 0,342), alors que le dataset a R et G quasi égaux. À 600
pas c'est du sous-entraînement. **Si le déséquilibre persiste à 20 000 pas, c'est
un signal à instruire**, pas une bizarrerie cosmétique.

### 4.3 Débit

Mesuré en annulant l'overhead fixe de démarrage : `(T₁₂₀ − T₂₀)/100`.

| modèle | T₂₀ | T₁₂₀ | **ms/pas** | overhead fixe |
|---|---:|---:|---:|---:|
| Color_Diffusion_L | 26,6 s | 153,6 s | **1271** | 1,1 s |
| Color_Diffusion_XL | 57,2 s | 334,0 s | **2768** | 1,8 s |

**Cadence effective, échantillonnage périodique compris : 1280 ms/pas**, mesurée
entre les PNG `step_0200` et `step_0599` du smoke (399 pas en 510,6 s). Un
échantillon périodique coûte ~0,5 s : négligeable, parce que c'est une chaîne
avant sans rétropropagation.

**Conditions de mesure** : runs strictement séquentiels, rien d'autre sur le GPU.
Les deux runs de L (1271) et la cadence effective du smoke (1280) concordent à
0,7 % près, à 15 minutes d'intervalle — la mesure est stable.

---

## 5. Ce qui a été corrigé (au-delà du strict nécessaire)

Trois défauts trouvés en route, tous susceptibles de **fausser silencieusement**
le run de nuit ou sa relecture :

1. **`main.rs` — l'inadéquation de géométrie `.batraw`/modèle était muette.**
   `try_load_raw_dataset` répare tout écart par un aller-retour 8 bits. Pratique
   pour redimensionner, mais un dataset **à 1 canal** donné à un modèle **à 3
   canaux** « marche » : chaque image devient du gris répliqué sur R, G et B, et
   **une nuit entière s'entraîne sur des données incolores sans une seule
   erreur**. Désormais annoncé, avec mention explicite quand c'est le nombre de
   canaux qui diffère. Vérifié dans les deux sens : le piège déclenche
   l'avertissement, le chemin correct reste silencieux.

2. **`tools/sample_diversity.py` — comparait du rouge à de la luminance.** Les PNG
   étaient lus en `convert("L")` (mélange des canaux) et le dataset via
   `[..., 0]`, c'est-à-dire le seul canal **rouge**. Comparer un modèle couleur à
   son dataset revenait à comparer deux grandeurs différentes. Corrigé, plus
   ajout de `chroma_std` et des stats par canal — **un modèle couleur peut
   converger vers du gris tout en affichant d'excellentes statistiques
   monochromes ; seule la chroma le dit.**

3. **`tools/compare_plate.py` — même défaut**, il aurait décoloré la planche de
   comparaison d'un modèle couleur.

Non-régression des deux outils vérifiée sur `scale_samples/L/seed_*.png` : les
**11 champs sont identiques au bit près** aux valeurs publiées dans
`SCALE_UNET.md` (0,00299 / 1,623 / 0,0521).

**Test ajouté** — `rgb_dataset_sample_survives_to_png_with_channels_unswapped`
(`main/src/main.rs`) : fige la convention HWC en C=3, du header `.batraw`
jusqu'aux pixels encodés, avec un pixel par primaire saturée. C'est le test
qu'aucun modèle greyscale ne pouvait produire. `cargo test -p main` : **10 verts**.

---

## 6. Run de nuit — commande exacte

```bash
cd /Users/bat/development/lab/batLab/worktrees/color-model

cargo run --release -p main -- --headless-train Color_Diffusion_L \
    --steps 20000 \
    --dataset datasets/cifar10_rgb.batraw \
    --lr 1e-3 --batch 16 \
    --optimizer adam --weight-init uniform --loss-weighting uniform \
    --out Models/Color_Diffusion_L/pretrained_weights/night_run.ckpt
```

**Durée : 20 000 × 1,280 s = 25 600 s ≈ 7 h 07.**

| lancement | fin estimée | marge avant 08:00 |
|---|---|---|
| 23:15 | **06:22** | 1 h 38 |
| 00:00 | 07:07 | 53 min |
| **00:53** | **08:00** | 0 — *heure limite de lancement* |

Le choix de 20 000 pas est **2,5× le minimum demandé** (8000) tout en gardant
plus d'une heure de marge à un lancement vers 23:15. Pour plus de marge encore :
**16 000 pas ≈ 5 h 41** (lançable jusqu'à 02:19).

Pendant le run :

- métriques → `Models/Color_Diffusion_L/pretrained_weights/night_run_metrics.jsonl`
- échantillons → `datasets/generated_samples/step_NNNN.png`, **un tous les 200
  pas** (100 PNG RGB sur le run), déjà gitignorés. C'est de quoi suivre la nuit
  sans le TUI.

`--headless-train` ne réécrit jamais le `config_file` du modèle et force
`load_checkpoint=false` : le run part bien de zéro.

### Au réveil

```bash
# 8 échantillons de seeds différents
for s in 1 2 3 4 5 6 7 8; do
  cargo run --release -p main -- --headless-sample Color_Diffusion_L \
      --checkpoint Models/Color_Diffusion_L/pretrained_weights/night_run.ckpt \
      --seed $s --paths 3 --magnitude 0.3 --out color_samples/night/seed_$s.png
done

python3 tools/sample_diversity.py --glob 'color_samples/night/seed_*.png' --label 'Color_L 20k'
python3 tools/sample_diversity.py --dataset datasets/cifar10_rgb.batraw --n 16
python3 tools/compare_plate.py --out color_samples/night_plate.png \
    --row 'Color_L 20k:color_samples/night/seed_*.png' \
    --dataset-row 'CIFAR-10 RGB reel:datasets/cifar10_rgb.batraw'
```

**Ce qu'il faudra regarder, dans l'ordre :**

1. **`chroma_std`** — s'il s'effondre vers 0, le modèle a convergé vers du gris et
   la couleur n'a rien apporté. Cible : l'ordre de 0,0707 (le dataset).
2. **L'équilibre R/G/B** — le vert en retrait du smoke doit s'être résorbé.
3. **`inter_seed_std`** — c'est là que la campagne greyscale a buté (~70× sous le
   dataset). **Aucune raison d'attendre mieux en couleur** : la cause mesurée dans
   `SCALE_UNET.md` §5 (pondération de l'objectif) est intacte, et ce run ne la
   traite pas. Le lire à `--magnitude` fixée, avec `intra_image_std` et
   `banding_ratio`, comme le rappelle `AGENTS.md`.
4. **Les tranches de `t`**, converties en unités x₀ via
   `tools/eps_metric_analysis.py` avant toute conclusion.

**Ce que ce run ne prouvera pas** : il ne teste ni la pondération de la loss, ni
la paramétrisation x₀/v — les deux pistes que `SCALE_UNET.md` §6 désigne comme la
cause réelle du collapse de diversité. Il établit **la couleur comme acquise**, et
donne une baseline RGB propre pour les tester ensuite.
