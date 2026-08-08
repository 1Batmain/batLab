# Entraîner sur ses propres images

*Rapport de mission — branche `custom-dataset`, 8 août 2026.*

Trois convertisseurs, un format qui a changé, des limites GPU devenues un choix,
et le mode d'emploi du fine-tuning. Le fil conducteur : rendre praticable un
corpus qu'on choisit soi-même — un dossier de photos, ou ImageNet 32×32 comme
fondation générale — là où le format précédent rendait le second impossible et
le premier laborieux.

---

## 1. Les trois commandes

### Un dossier d'images à soi

```bash
python3 tools/images_to_raw.py mes_photos/ \
    --out datasets/elephants.batraw \
    --size 32 --mode rgb --min-size 32 --dedup \
    --contact-sheet docs/gallery/elephants.png
```

Récursif, tous les JPEG/PNG/WebP/BMP/GIF/TIFF. Par image : orientation EXIF
appliquée, carré (`--crop center`, le plus grand carré centré — ou `--crop fit`,
l'image entière avec des marges en gris neutre), redimensionnement Lanczos,
`--mode rgb|grey`.

**Toujours passer `--contact-sheet`.** C'est la seule façon de voir ce sur quoi
on va réellement entraîner *après* recadrage. Les erreurs qui coûtent une nuit
ne lèvent aucune exception : un dossier qui contenait surtout des vignettes, un
`center` qui décapite tous les sujets, un `--mode grey` demandé par mégarde. Le
script le rappelle quand on ne le demande pas.

Les autres options existent parce qu'un scrape est sale :

| option | ce qu'elle règle |
|---|---|
| *(rien)* | fichier illisible/tronqué → sauté et **compté** |
| `--min-size N` | écarte les images dont le plus petit côté est sous N. Agrandir du 20×15 en 32×32 donne de la bouillie que le modèle apprendra consciencieusement. Sans le flag, le nombre d'images agrandies est imprimé en ATTENTION |
| `--dedup` | compare l'**échantillon produit**, pas le fichier : la paire PNG/JPEG du même cliché que tout scrape ramène est un seul échantillon, et c'est celui-là que le modèle verrait deux fois |
| — | l'ordre vient du **chemin relatif** trié : deux exécutions donnent le même fichier, octet pour octet, et déplacer le dossier n'y change rien |

Deux choix de pixels, pour la même raison : les marges de `--crop fit` et le fond
des PNG détourés sont l'octet **128**, c'est-à-dire 0 dans la plage [-1,1] du
modèle. Du noir vaudrait −1 : un bord dur que le réseau apprendrait comme une
propriété du dataset.

### CIFAR-10

```bash
cd datasets && python3 cifar_to_raw.py --mode rgb --out cifar10_rgb.batraw
```

Inchangé, sauf qu'il écrit BATRAW3 par défaut (`--format batraw2` rend l'ancien).
En RGB c'est exact — l'archive CIFAR *est* 8 bits. En niveaux de gris la
luminance BT.601 de trois octets n'est pas un octet et est désormais arrondie :
±1/255 par pixel, dit dans le script. Les fichiers existants continuent de se
charger ; rien n'a besoin d'être reconverti.

### ImageNet 32×32 — la fondation

```bash
python3 tools/imagenet32_to_raw.py ~/Downloads/Imagenet32_train \
    --out datasets/imagenet32_rgb.batraw

# un sous-ensemble, pour un premier essai qui ne coûte pas 4 Go :
python3 tools/imagenet32_to_raw.py ~/Downloads/Imagenet32_train \
    --out datasets/imagenet32_small.batraw --limit 50000
```

**Le script ne télécharge rien.** Il attend, dans le dossier donné,
`train_data_batch_1` … `train_data_batch_10` — **sans extension**, ce sont des
pickles Python et non des `.npz`. Chacun porte un `data` uint8 de forme
`(128116, 3072)`. Absents, le script dit lesquels il attendait et s'arrête.

Le piège est le même que pour CIFAR, et il ne lève aucune erreur : les 3072
octets d'une image sont **trois plans** de 1024 (R entier, puis G, puis B), pas
des pixels entrelacés. Lu tel quel, le dataset donne des images au tiers haut
rouge, tiers médian vert, tiers bas bleu, et l'entraînement se déroule
normalement. La conversion se fait dans le script ; `test_the_three_planes_become_pixels`
la tient, et la planche de contrôle la montre à l'œil.

1 281 160 images, **3,94 Go** en BATRAW3 (15,7 Go en f32).

### Les tests

```bash
python3 tools/test_dataset_tools.py      # 25 tests, images synthétiques, aucun réseau
cargo test --workspace                   # 331 tests
```

---

## 2. Ce qui a changé sous les convertisseurs

### `BATRAW3` — le payload en u8

Le format stockait des f32 pour des images 8 bits à la source : quatre octets
dont trois se déduisaient du premier. `BATRAW3` garde le même en-tête et met le
payload en u8 ; l'élargissement en [-1,1] se fait **sur le GPU**
(`training/shader/dataset_decode.wgsl`), donc le fichier, la RAM hôte *et* le
transfert sont tous au quart.

|  | f32 (BATRAW2) | u8 (BATRAW3) |
|---|---|---|
| CIFAR-10 RGB | 586 Mio | **146 Mio** |
| ImageNet 32×32 | 15,7 Go | **3,94 Go** |

`BATRAW1` et `BATRAW2` restent lisibles. Un fichier 8 bits dont la géométrie ne
colle pas au modèle retombe sur le chemin f32 — le rééchantillonnage se fait sur
l'hôte de toute façon.

**Le décodage est exact, et il a failli ne pas l'être.** Écrit comme la formule
`f32(byte) / 127.5 - 1.0` — la même qu'au CPU, au caractère près — le shader
répondait autre chose sur **111 valeurs sur 256**, d'un ULP chacune : Metal
compile une division flottante en multiplication par une réciproque approchée.
Un ULP n'intéresse aucun réseau, mais il rendait fausse la propriété qui
justifie tout le format — « BATRAW3 est un ré-encodage *sans perte* ». Le shader
lit donc une **table de 256 f32** remplie par `decode_u8` : plus d'arithmétique,
donc rien qu'un compilateur puisse réassocier. Ne pas la « simplifier » en une
formule.

Trois garde-fous, à trois hauteurs :

- `the_gpu_decode_agrees_with_the_cpu_one` — les 256 valeurs, au bit ;
- `an_8_bit_file_decodes_to_the_same_bits_as_the_f32_one_it_replaces` — le
  fichier 8 bits contre le fichier f32 issu des mêmes images ;
- `an_8_bit_dataset_trains_exactly_like_the_f32_one_it_replaces` — 8 pas
  d'entraînement dont les pertes doivent être identiques bit à bit.

### `--gpu-limits native|web` — les limites demandées deviennent un choix

`request_device` ne donne pas ce que l'adaptateur sait faire : il donne ce qu'on
**demande**, et ne rien demander revenait à demander la base WebGPU (256 Mio par
tampon, 128 Mio par binding de stockage). C'est juste pour l'**inférence**, dont
la cible est le navigateur du visiteur ; c'est faux pour l'**entraînement**, qui
tourne ici, sur un adaptateur qui autorise 28 Gio par tampon et 4 Gio par
binding.

- `native` (**défaut**) demande à l'adaptateur ses propres limites ;
- `web` demande la base — la façon de vérifier qu'un modèle passerait encore
  dans un navigateur.

Le dimensionnement des chunks a désormais deux cas. **Résident** : le dataset
tient dans un tampon que l'appareil accepte de lier *et* sous le budget de
résidence (`gpu_cap/4`) → on prend tout, un seul envoi pour la vie du run.
**Streamé** : sinon, budget `gpu_cap/8`, exactement comme avant. Les deux
fractions diffèrent exprès : un chunk streamé se repaie à chaque raté, un
dataset résident se paie une fois.

C'est ce qui fait entrer ImageNet 32×32 u8 (3,94 Go, sous les 4 Gio de binding)
**en entier** — zéro transfert dataset par pas.

---

## 3. Les chiffres

Tous mesurés sur `Color_Diffusion_XL`, batch 32, `cifar10_rgb`, via
`--resources … --measure` (compteurs réels, pas une estimation).

### Trafic hôte→GPU par pas d'entraînement

| dataset | limites | trafic/pas | chunks | plafond de batch |
|---|---|---|---|---|
| f32 (586 Mio) | web | 585,9 Mio | 5 | 341 |
| **u8 (146 Mio)** | web | **146,5 Mio** | 2 | 341 |
| **u8 (146 Mio)** | **native** | **1,9 Kio** | **1 (résident)** | **4096** |

Le passage f32 → u8 divise le trafic par **4,00** exactement. Le passage web →
native le supprime : il ne reste que le tableau de specs du batch.

### Ce que ça change au temps de calcul : rien, sur cette machine

| | 10 pas (batch 32) | coût fixe (chargement + build) |
|---|---|---|
| native | 46,0 s | 5,92 s |
| web | 46,0 s | 5,93 s |

**À 0,1 % près, identique.** La mémoire est unifiée : un `write_buffer` de
146 Mio est un memcpy à l'échelle de la milliseconde, contre un pas de ~4,4 s.
Le trafic n'était pas le goulot ici.

Il faut le dire clairement, parce que la tentation d'annoncer une accélération
est forte et elle serait fausse. Ce que ce travail achète **réellement sur cette
machine** :

- le **plafond de batch**, 341 → 4096 — c'était la limite de *binding* qui
  tranchait, pas la mémoire ;
- la **RAM hôte** : 3,94 Go au lieu de 15,7 Go pour ImageNet, ce qui est la
  différence entre « ça tient » et « ça ne tient pas » ;
- le **disque**, ×4 ;
- et la capacité à garder un corpus entier résident.

Le gain de trafic, lui, se paierait sur une **carte discrète**, derrière un
PCIe : 585,9 Mio par pas à ~10 Go/s, c'est ~60 ms ajoutées à chaque pas.

### Équivalence

Quarante pas de `Color_Diffusion_XL` sur `cifar10_rgb.batraw` dans les deux
encodages impriment les **mêmes** pertes :

```
step 0   loss 1.426182      step 0   loss 1.426182
step 25  loss 0.151379      step 25  loss 0.151379
step 39  loss 0.109517      step 39  loss 0.109517
```

Et `--headless-sample` rend des statistiques d'image identiques sous les deux
profils de limites (`min=-0.9999 max=0.2748 mean=-0.5989 std=0.2249`).

---

## 4. Fine-tuner sur ses propres images

Le chemin complet, du dossier de photos au modèle qui n'en dessine plus que.

### 4.1 Dupliquer le modèle — d'abord

Depuis le TUI : ouvrir la fondation, `Duplicate` au menu d'actions, « config +
poids » (le défaut). Ça écrit `Models/<copie>/` avec les derniers poids sous leur
nom daté et `latest.ckpt` reposé dessus.

**Ce n'est pas une précaution facultative.** Fine-tuner écrivait dans le dossier
de la fondation, et après assez de pas sur un corpus étroit le modèle général
avait disparu — c'est la raison d'être de la fonctionnalité
(`docs/reports/DUPLICATE_MODEL.md`). On duplique, puis on fine-tune la copie.

### 4.2 Le dataset

```bash
python3 tools/images_to_raw.py mes_photos/ --out datasets/elephants.batraw \
    --size 32 --mode rgb --min-size 32 --dedup \
    --contact-sheet docs/gallery/elephants.png
```

Regarder la planche. Vraiment.

### 4.3 Le run

```bash
cargo run --release -p batlab -- \
    --headless-train Elephants \
    --dataset datasets/elephants.batraw \
    --resume Models/Elephants/pretrained_weights/latest.ckpt \
    --out Models/Elephants/pretrained_weights/finetune.ckpt \
    --steps 20000 --batch 32 --lr 1e-4 --optimizer adam \
    --checkpoint-every 500
```

- `--resume` reprend poids, moments Adam, compteur de pas et EMA. Il **écrit
  dans `--out`** : un run repris n'écrase jamais le fichier dont il vient.
- `--checkpoint-every 500` écrit un partiel en rotation (un seul fichier,
  temporaire + `rename` atomique). Un run tué à la neuvième heure laisse son
  partiel comme poids les plus récents. Sur une nuit, c'est ce qui sépare
  « on a un modèle » de « on n'a rien ».
- `--lr 1e-4` — un dixième du taux d'entraînement initial. Un fine-tune part
  d'un optimum ; le taux de départ le quitte.

### 4.4 Les conseils honnêtes

**Sous ~1000 images, le modèle mémorise.** Il rendra des variations serrées de
vos images plutôt que de nouvelles images de la même famille. Ce n'est **pas
forcément un défaut** : pour une pièce artistique — une vidéo de mode flux qui ne
montre que des éléphants — c'est même souvent ce qu'on veut. Mais il faut le
savoir, parce que la sortie ressemblera au dataset et qu'on croira à un succès de
généralisation.

**Partir du checkpoint CIFAR (ou ImageNet) plutôt que de zéro.** Un run
from-scratch sur 500 images n'a pas assez de signal pour apprendre à quoi
ressemble une image ; il apprend d'abord la statistique locale des couleurs et
n'en sort pas. La fondation apporte cette part-là, et le fine-tune n'a plus qu'à
déplacer la distribution.

**`--magnitude 1.0` à la génération.** `inter_seed_std` est proportionnel à la
magnitude et ne se lit pas seul (voir `CLAUDE.md`) ; à magnitude réduite on croit
gagner en netteté ce qu'on perd en diversité.

**Le nombre de pas est le levier.** Pas le taux, pas l'EMA (mesurée NO-GO à 1500
pas, `docs/reports/EMA.md` §5), pas la pondération de la loss (NO-GO,
`docs/reports/LOSS_WEIGHTING.md`). Un fine-tune utile se compte en dizaines de
milliers de pas, pas en centaines.

**Vérifier la loss par tranche de t, pas la loss batch.** Une loss batch qui
décroît ne prouve rien ; une loss élevée à t bas est un modèle qui n'utilise pas
t (`docs/reports/INSIGHTS_TRAINING.md`). Le `*_metrics.jsonl` déposé à côté du
checkpoint la porte.

### 4.5 Un run de fondation sur ImageNet

```bash
cargo run --release -p batlab -- \
    --headless-train Color_Diffusion_XL \
    --dataset datasets/imagenet32_rgb.batraw \
    --out Models/Color_Diffusion_XL/pretrained_weights/imagenet.ckpt \
    --steps 200000 --batch 32 --lr 1e-3 --optimizer adam \
    --checkpoint-every 1000
```

Sous `--gpu-limits native` (le défaut), la bannière doit annoncer :

```
[gpu] limits=native (buffer 28.08 GiB, storage binding 4.00 GiB) · dataset 3.67 GiB
      in ONE resident chunk — uploaded once, no dataset traffic per step
```

Si elle annonce des chunks, c'est que le dataset dépasse ce que l'appareil
accepte de lier : réduire avec `--limit` à la conversion, ou accepter le
streaming.

---

## 5. Bout en bout, déroulé

Un mini-dataset synthétique de 200 formes colorées (tailles et ratios variés,
plus un doublon et un JPEG tronqué délibérés) :

```
200 image(s) écrite(s) (32×32×3, batraw3) → shapes.batraw  [0.6 MiB]
  202 candidat(s) · 1 illisible(s) · 0 trop petite(s) · 1 doublon(s)
  NOTE : 200 images, c'est sous le seuil (~1000) où le modèle se met à mémoriser
```

Converti en BATRAW3, puis 100 pas de fine-tune depuis les poids réels de
`Color_Diffusion_XL` :

```
[resume] latest.ckpt — optimiser step 8500
[gpu] limits=native (buffer 28.08 GiB, storage binding 4.00 GiB) · dataset 600.0 KiB
      in ONE resident chunk — uploaded once, no dataset traffic per step
step 0   loss 0.002248
step 25  loss 0.001632
[checkpoint] step 50 → shapes_finetune.partial.ckpt (14.3 MB)
step 50  loss 0.000927
step 75  loss 0.000639
[checkpoint] final → shapes_finetune.ckpt
```

La loss bouge, le compteur de pas reprend à 8500 (donc `--resume` a bien lu les
moments Adam et pas seulement les poids), le partiel tourne, et le checkpoint
produit échantillonne : `--headless-sample` en tire une image
(`min=-0.6470 max=0.6088 mean=-0.0417 std=0.2912`). La chaîne entière tient, du
dossier d'images à l'image générée.

Et la fondation, sur les vrais fichiers : 1 281 167 images converties en 3,94 Go,
puis

```
resident — the dataset too. All 3.67 GiB of it sits in one buffer, uploaded once
  at the first step and never again: a batch of 32 costs NO host→GPU dataset traffic.
train step    1.9 KiB up   12.0 KiB down   1.0 rt   3.0 sub
```

---

## 6. Ce qui reste ouvert

- **Pas de `--eval <ckpt>`.** Juger deux checkpoints demande de relancer un run ;
  c'est déjà ce qui a empêché de trancher proprement sur l'EMA (`EMA.md` §5).
- **`--limit` d'ImageNet prend le préfixe**, pas un échantillon réparti. Pour un
  sous-ensemble représentatif, les batches de l'archive sont déjà mélangés, donc
  le préfixe est acceptable — mais ce n'est pas la même chose qu'un tirage.
- **La planche de contrôle n'est pas exposée en CLI pour un `.batraw` déjà
  écrit** : il faut la ligne `python3 -c` que le convertisseur ImageNet imprime.
- **Le gain de trafic n'a pas été mesuré sur carte discrète**, faute d'en avoir
  une. Le raisonnement PCIe du §3 est une déduction, pas une mesure, et il est
  présenté comme telle.
