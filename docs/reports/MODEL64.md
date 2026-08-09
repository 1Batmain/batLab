# MODEL64 — passer batLab en 64×64 avec un modèle plus grand

Mission : porter le générateur d'architecture et le pipeline en 64×64, concevoir
un modèle **plus grand** que l'actuel `Color_Diffusion_XL` (32×32), sous une
contrainte dure de l'utilisateur :

> « Il faudrait que l'inférence passe toujours en web, mais soit à la limite. »

Toutes les mesures ci-dessous sont prises sur **Apple M5 Pro**, profil natif
(tampon 28,08 Gio, binding stockage 4,00 Gio) pour l'entraînement, profil **web**
(la base WebGPU : 256 Mio/tampon, 128 Mio/binding de stockage, 8 storage buffers
par étage, 65 535 workgroups/dimension) pour l'inférence. Faute de dataset 64×64
natif, les runs tournent sur `cifar10_rgb` (32×32) **rééchantillonné en 64×64 par
le chargeur** — plomberie, pas qualité.

---

## Le résultat qui recadre tout : le web ne mord pas

La contrainte « passe en web à la limite » s'est révélée **non contraignante à
l'inférence**. À batch 1 (le sampler tire un chemin à la fois), le plus gros
tampon de stockage d'un candidat 64×64 est de **2 à 9 Mio**, contre un plafond de
binding de **128 Mio** — un facteur **14× à 58×** de marge. Le plus gros dispatch
est de 4 096 workgroups contre 65 535. Pour *approcher* la limite web il faudrait
un modèle 15 à 25× plus large que tout ce qui est praticable à l'entraînement.

**Donc le web n'est pas le facteur limitant. Les deux vrais arbitrages sont :**

1. **le temps d'entraînement sur cette machine** (cible ≤ 500 ms/pas à batch 32) ;
2. **le poids du checkpoint** que le visiteur télécharge sur la page portfolio.

Vérifié, pas déduit : `Color_Diffusion_64` (le retenu) produit une inférence
**bit-à-bit identique** sous `--gpu-limits web` et `--gpu-limits native`
(`min=-0.9915 max=0.9017 mean=-0.1424 std=0.2846`, seed 7, les deux profils), et
un `--resources … --inference --gpu-limits web` sur le vrai M5 Pro **obtient**
bien les limites web (256/128 Mio) et rend le graphe avec un tampon de 5,1 Mio.

---

## Le tableau des candidats

Quatre étages (64→32→16→8), attention au goulot **8×8** — la garder à 8×8 est ce
qui maintient le scratch d'attention N² (64 positions) **identique à XL** ; le
monter à 16×16 le multiplierait par 16. Champ réceptif **75 px pour tous** (même
structure spatiale, seules les largeurs changent) : il **couvre** les 64 px, la
leçon de `RECEPTIVE_FIELD`. Généré par `gen_unet_config.py --size 64 --widths …
--attention`, donc toutes dims dérivées.

| candidat | largeurs | params | **params/px** | poids f32 (int8) | inf. web b1 : total / + gros tampon | verdict web | entr. b32 native | **ms/pas b32** | ms/frame → ms/image |
|---|---|---:|---:|---:|---|---|---:|---:|---|
| **XL 32² (réf.)** | 48/96/192 | 1,19 M | **1165** | 4,8 Mo (1,2) | 8,9 Mio / 1,3 Mio | PASSE | — | ~265¹ | — |
| A | 32/64/128/256 | 2,21 M | 539 | 8,8 Mo (2,2) | 21,1 Mio / 2,2 Mio (58×) | PASSE | 1,46 Gio | **427** | 7,0 ms → 1,8 s |
| D | 24/64/128/256 | 2,19 M | 534 | 8,8 Mo (2,2) | 19,4 Mio / 2,2 Mio (58×) | PASSE | 1,35 Gio | **401** | 8,0 ms → 2,1 s |
| **B ✅** | **48/96/192/384** | **4,96 M** | **1210** | **19,8 Mo (5,0)** | 37,8 Mio / 5,1 Mio (25×) | PASSE | 1,92 Gio | **851** | 11,0 ms → 2,8 s |
| E | 64/128/256/384 | 6,72 M | 1642 | 26,9 Mo (6,7) | 50,5 Mio / 5,1 Mio (25×) | PASSE | 2,34 Gio | **1381** | n.m. |
| C | 64/128/256/512 | 8,81 M | 2150 | 35,2 Mo (8,8) | 58,8 Mio / 9,0 Mio (14×) | PASSE | 2,39 Gio | **1496** | n.m. |

¹ `Color_Diffusion_XL` : 265 ms/pas b32 vient de `docs/reports/GPU_PROFILE.md`
(après le fix upsample), pas re-mesuré ici. « n.m. » = non mesuré (pas de
checkpoint entraîné pour ces largeurs ; le ms/image suit le coût forward, donc
E/C sont au-dessus de B). Le ms/frame de débruitage = ms/image ÷ 256 : c'est le
coût d'**une** frame perpetual.

**Poids f32** = params × 4 o, ce qu'un export d'inférence embarque ; l'**int8**
(÷4) est possible plus tard, non implémenté ici. Le « 14 Mo » de XL cité en
mission est le checkpoint d'**entraînement** (poids + moments Adam m,v = 3×), pas
ce qu'on sert au web — le web ne prend que les poids.

---

## Pourquoi B, et pourquoi « plus grand » n'est pas « plus de params »

À 64×64 il y a **4× plus de pixels** qu'à 32×32. La bonne mesure de capacité
n'est donc pas le nombre de paramètres brut mais **params/pixel** :

- A et D (2,2 M, ~535 params/px) sont **plus gros que XL en absolu** mais, par
  pixel, **deux fois plus maigres** (539 vs 1165). C'est une **régression par
  pixel déguisée en montée en gamme** : sur 4× la surface, chaque pixel dispose
  de moins de capacité qu'à 32×32. Ce n'est pas le « modèle plus grand » demandé.
- **B (48/96/192/384) rend 1210 params/px — la parité avec XL.** C'est le
  **plancher** d'un modèle qui soit vraiment plus grand : il restaure la densité
  de capacité de XL sur l'image 4× plus grande. 4,96 M params = 4,2× XL.
- E et C dépassent XL par pixel (1642, 2150) mais à un coût d'entraînement qui
  grimpe **superlinéairement** (851 → 1381 → 1496 ms/pas), pour un gain incertain
  tant que l'architecture n'est pas corrigée (voir *À ne pas figer* ci-dessous).

**Recommandation : B**, l'arbitrage explicite étant :

- **Entraînement** : 851 ms/pas à b32, **au-dessus** de la cible aspirationnelle
  de 500 ms — mais tenable : 2000 pas ≈ 28 min, une nuit en écrit des dizaines de
  milliers. La cible 500 ms n'est atteinte que par A/D, au prix de la régression
  par pixel. C'est l'arbitrage à trancher : **densité de capacité (B) contre
  débit d'entraînement (A/D)**. Recommandation : la capacité, B.
- **Téléchargement** : 19,8 Mo f32, **5,0 Mo en int8** — raisonnable pour une
  page portfolio (XL faisait 4,8 Mo f32).
- **Fluidité perpetual** : 11 ms par frame de débruitage sur M5 Pro → ~90 fps non
  throttlé, très au-dessus du tempo 30/s par défaut. Sur une machine ~3× plus
  lente, 11 → 33 ms tombe pile sur la ligne des 30 fps : B reste fluide, c'est la
  borne. E/C, plus lourds par frame, la franchiraient sur machine faible.
- **Web** : PASSE avec 25× de marge sur le binding — non contraignant, comme tous.

Config écrite dans **`Models/Color_Diffusion_64/config_file`**. Vérifiée : build
+ 30 pas (loss 1,48 → ~0,16–0,46), RF 75 px, parité web/native bit-à-bit, dérive
perpetual `flux` depuis une image du dataset rééchantillonnée en 64×64.

### À ne pas figer — la largeur sera re-dérivée après `archi32`

Une mission parallèle (branche `archi32`) établit que **l'architecture actuelle
sous-apprend** : pas de blocs résiduels (aucune addition dans `LayerKind`),
timestep injecté **à la seule entrée**, ~30× plus petite que le DDPM de référence
à 32×32. Ses correctifs rendront les paramètres **plus efficaces** — le point
d'équilibre capacité/coût se déplacera. **B est un placeholder provisoire** :
après le merge d'`archi32`, re-dériver la largeur (une architecture résiduelle
bien conditionnée sur t atteindra la même qualité avec moins de params, ou
justifiera d'aller vers E à coût égal). Ne pas traiter 48/96/192/384 comme figé.

---

## Item 1 — le générateur passe à N étages

`gen_unet_config.py` était câblé à trois étages 32→16→8. `build()` dérive
maintenant sa pile d'un `--widths` de **longueur libre** (une largeur par étage,
la dernière au goulot) et d'un `--size` quelconque (divisible par 2^(len−1), sinon
refus net). Les skips sont nommés par résolution (`skip64`/`skip32`/…), ce qui
garde **le défaut identique octet-pour-octet** à `Greyscale_Diffusion_L` (et à
XL avec `--widths 48 96 192 --attention`). Toutes les dims restent dérivées :
`dim_kernel.z` suit le canal courant à chaque couche — la corruption silencieuse
de l'`UpsampleConv` est ce que le script existe pour empêcher.

`receptive_field.py`, déjà résolution-générique, donne 75 px sur un 4-étages
64×64 : il couvre l'image.

Gardes ajoutées (`tools/test_dataset_tools.py`, `build_config()` isolé pour le
test) : reproduction octet-pour-octet du défaut, `dim_kernel.z == dim_input.z` à
chaque couche d'une pile 4 étages, refus d'une taille non divisible, et
+1 encodeur/+1 décodeur par étage ajouté.

---

## Item 3 — ce que 64×64 casse ailleurs

**Le moteur est résolution-agnostique à l'exécution.** Sampler, live_frame,
visualiseur, chargeur de dataset et planificateur de ressources dérivent tous
leurs dims soit du modèle chargé (`input_dim()`/`output_dim()`), soit de
l'en-tête `.batraw` — jamais d'un littéral. Les points de rupture réels sont donc
les **générateurs de config** et les **convertisseurs de données**, pas le code
de calcul. Ce qui a été traité :

- **`imagenet32_to_raw.py` → `imagenet_to_raw.py`** (renommé) : la release
  « Downsampled ImageNet » existe en 32×32 **et 64×64**, même format de pickle —
  seule la longueur de ligne change (3072 = 32²·3, 12288 = 64²·3). Le
  convertisseur **déduit** le côté de `data.shape[1]`, refuse une ligne qui n'est
  pas trois plans carrés, et interdit qu'un dossier mêle deux tailles. Tests
  ajoutés (détection 64, refus longueur non-carrée, refus dossier mixte).
- **`sample_diversity.py`** : suppression du `size=32` mort (la géométrie venait
  déjà de l'en-tête ; l'argument induisait en erreur).
- **Le visualiseur winit** est déjà résolution-indépendant :
  `initial_window_size()` s'échelonne sur n'importe quelle frame (une frame 64 px
  ouvre à ×8). Le mode **Perpetual** et sa graine marchent en 64×64 — vérifié :
  l'image du seed dataset est rééchantillonnée à la résolution du modèle, la
  bannière nomme la source, la frame sort en 64×64.

**Laissés délibérément** (légitimement 32×32) : `cifar_to_raw.py` et le décodeur
CIFAR (`main.rs`) — le format CIFAR-10 *est* 32×32, le chargeur rééchantillonne
vers la taille du modèle ; le template TUI `config.rs:diffusion_unet` (32×32,
mono-niveau) — c'est un gabarit de départ, l'écran « InputSize » laisse déjà
saisir n'importe quelle géométrie ; `docs/reports/CUSTOM_DATASET.md` cite encore
l'ancien nom `imagenet32_to_raw.py` — c'est une **archive** (politique : non
réécrite ; table de correspondance dans `INDEX.md`).

---

## Reste à faire

1. **Un dataset 64×64 natif.** Aujourd'hui tout tourne sur `cifar10_rgb` upscalé
   (le chargeur le prévient : « file is 32x32x3, model expects 64x64x3 »). Deux
   voies, toutes deux prêtes : `imagenet_to_raw.py` sur la release **ImageNet
   64×64** (l'utilisateur téléchargera l'archive — le convertisseur détecte la
   taille tout seul), ou `images_to_raw.py --size 64` sur un corpus propre.
   Attention : ImageNet 64×64 en u8 fait ~15,7 Go, **au-delà du binding 4 Gio** →
   il sera streamé par chunks (regarder la bannière), pas résident.
2. **La quantification int8 pour le web** (÷4 : B passerait de 19,8 à 5,0 Mo).
   Possible, non implémentée ici — il n'existe pas encore d'export « poids
   d'inférence seuls » (le checkpoint actuel embarque aussi les moments Adam). Un
   export web écrirait le premier bloc du checkpoint (les poids) quantifié.
3. **Re-dériver la largeur après `archi32`** (cf. *À ne pas figer*).

## Reproduire les mesures

```bash
export BATLAB_ROOT=<un dossier avec Models/C64_x/config_file>
BIN=target/release/batlab
# verdict web (papier, contre les limites WebGPU) :
$BIN --resources C64_B --inference --no-gpu
# graphe web obtenu sur le vrai GPU forcé aux limites web :
$BIN --resources C64_B --inference --gpu-limits web
# mémoire d'entraînement native b32 :
$BIN --resources C64_B --batch 32 --dataset datasets/cifar10_rgb.batraw
# temps GPU réel d'un pas :
$BIN --profile-step C64_B --dataset datasets/cifar10_rgb.batraw --batch 32
# parité inférence web vs native (doit être bit-à-bit) :
$BIN --headless-sample C64_B --checkpoint <ck> --paths 2 --seed 7 --gpu-limits web
```
