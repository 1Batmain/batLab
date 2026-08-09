# Où partent les 4,4 secondes d'un pas — rapport de mission

Branche `gpu-profile`. Mission : instrumenter le pas d'entraînement passe par
passe, diagnostiquer, puis corriger **ce que la mesure désigne, et seulement ça**.

---

## Résumé

Le pas de `Color_Diffusion_XL` (1,19 M paramètres, batch 32, Adam) passe de
**4 400,7 ms à 265,1 ms** de temps GPU mesuré — **16,6×**. Aucune formule n'a
changé ; 300 pas d'entraînement appariés donnent la même trajectoire de loss à
**7,3e-7** près.

Trois résultats méritent d'être lus avant le reste, et les trois contredisent
l'hypothèse de départ de la mission.

- **Il n'y a rien entre les passes.** `span − Σ passes = 0,0 ms` à batch 16, 32
  et 64. La latence de dispatch, les barrières et le coût fixe par passe — le
  « per-compute-pass floor » que `PERF_CONVOLUTION.md` §5.4 posait en hypothèse
  et n'a jamais su mesurer — valent **zéro** sur ce graphe. C'est la première
  chose que l'instrument a dite, et elle a fermé une piste entière.
- **94,1 % du pas était dans DEUX passes sur 151**, et la cause n'était ni le
  calcul, ni la bande passante, ni l'occupation : le kernel
  `upsample_conv_back_input` faisait **256 fois trop d'itérations**. Il balayait
  la carte de sortie entière pour chaque élément d'entrée et jetait tout ce qui
  ne tombait pas dans sa fenêtre. §3.
- **Le coût par échantillon plat de 32 à 64 n'est pas une saturation qui
  s'installe à 32** : il est déjà plat à batch **8** (140,9 / 138,5 / 137,5 /
  137,2 ms par échantillon), avant comme après la correction. Il n'y a rien à
  amortir — 151 dispatches quel que soit le batch, et 2 à 4 ms de coût fixe par
  pas. §5.4.

Ce qui reste après correction est un pas dont **48 % est dans six passes
`*_back_weights`** qui font exactement le même nombre de MAC que leur forward et
mettent 4,1× plus longtemps. §6.

---

## 1. L'instrument — `--profile-step`

`wgpu::Features::TIMESTAMP_QUERY` permet à une passe de calcul d'écrire
l'horloge du GPU à son entrée et à sa sortie. Sur ce backend
`TIMESTAMP_QUERY_INSIDE_PASSES` n'est **pas** disponible (vérifié sur
l'adaptateur : `TIMESTAMP_QUERY: true`, `TIMESTAMP_QUERY_INSIDE_PASSES: false`,
`TIMESTAMP_QUERY_INSIDE_ENCODERS: true`) — ce qui aurait été bloquant si le
moteur encodait tous ses dispatches dans une seule passe. Il n'en encode pas :
`layer.rs` ouvre **une compute pass par dispatch** depuis toujours, pour la
barrière implicite que wgpu insère entre deux passes d'un même encodeur. Le
profilage est donc une mesure **directe**, sans aucun changement de structure du
graphe : ni passe ajoutée, ni passe fusionnée, ni submit supplémentaire.

```
cargo run --release -p batlab -- --profile-step <model> --dataset <path> \
    [--batch N] [--rounds N] [--warmup N] [--top N]
```

Trois grandeurs sortent d'un pas armé, et ce sont trois choses différentes :

| grandeur | définition | ce qu'elle répond |
|---|---|---|
| `nanos` par passe | `end − begin` de cette passe | quelle passe coûte |
| `span` | dernier `end` moins premier `begin` | la durée GPU du pas entier |
| `span − Σ nanos` | | ce que le pas passe **entre** ses passes |
| `mur − span` | horloge hôte autour du pas, `poll(Wait)` compris | ce qui n'est pas dans l'encodeur profilé |

### 1.1 Discipline : pas de seconde porte

Toute `begin_compute_pass` du moteur passe par `GpuContext::compute_pass`, et
`nothing_in_the_engine_opens_an_untimed_pass` échoue sur tout site qui
contournerait la porte — exactement la règle, et pour exactement la raison, de
`nothing_in_the_engine_bypasses_the_counted_queue` (`GPU_RESOURCES.md` §4). Une
passe absente du tableau ne planterait pas, ne préviendrait pas, et ne se verrait
nulle part : **elle se relirait comme du temps entre les passes**, c'est-à-dire
comme la grandeur sur laquelle tout ce rapport repose.

Coût quand c'est éteint : la feature n'est pas demandée, le query set n'est pas
alloué, le descripteur est le `Default::default()` d'avant, et le libellé est une
**closure** jamais appelée — donc aucune `String` construite sur les 151 passes
d'un pas.

### 1.2 Deux défauts de l'instrument, trouvés en le lisant

Ils sont documentés ici parce que les deux produisaient des nombres *plausibles*.

**Metal ne remplit pas toujours les compteurs.** Environ un pas armé sur quatre,
trois des vingt passes Adam reviennent `begin = 0, end = 0`. Zéro n'est pas une
lecture plausible d'une horloge qui affiche 3,76e15. Non traité : `min(begin) = 0`
faisait du span la valeur **absolue** de l'horloge (`span across all rounds …
3 761 240 032 ms`), et surtout le temps manquant se serait relu comme du temps
entre les passes. Les paires non échantillonnées sont maintenant exclues du span,
comptées et annoncées, et une ronde où une passe n'a pas été échantillonnée ne
contribue pas au minimum de cette passe — sans quoi le minimum choisirait
précisément cette ronde-là, courte d'un dispatch entier.

**Le budget mélangeait trois rondes.** Σ passes, span et mur étaient minimisés
indépendamment ; le rapport a imprimé « −0,9 ms outside the GPU span », soit une
horloge hôte qui finit avant que le GPU commence. Ce n'est pas une découverte sur
la latence, ce sont deux pas différents soustraits l'un de l'autre. Les trois
lignes viennent maintenant d'**une** ronde, la plus rapide au mur, et l'étendue
du span sur toutes les rondes est imprimée à côté.

Il reste un plancher : l'horloge hôte et l'horloge GPU ne sont pas la même, et
`mur − span` s'étale de −0,8 à +6,2 ms sur les mesures de ce rapport. Sur un pas
de 265 ms cela borne la résolution de cette ligne-là à ±0,3 % ; les autres lignes
viennent toutes de l'horloge GPU seule.

### 1.3 Protocole

Estimateur = **minimum** sur les rondes armées (la contention ne peut
qu'ajouter du temps — `PERF_CONVOLUTION.md` §5.1). 3 pas de chauffe avant toute
mesure : sur Metal le premier dispatch d'un pipeline paie sa compilation, ce qui
avait inversé tout un profil dans la mission convolution. Le harnais appelle la
**vraie** fonction de production (`DiffusionTask::train_step_batch`) et rien
d'autre — ni relecture de loss, ni sonde par tranche de `t`, ni échantillonnage
périodique, qui soumettent tous du travail GPU qui serait profilé comme s'il
faisait partie du pas.

Machine : Apple M5 Pro, Metal, mémoire unifiée, `--gpu-limits native`, dataset
`cifar10_rgb` **résident** (donc zéro trafic dataset par pas ; cf.
`GPU_RESOURCES.md` §5.1). GPU par ailleurs libre.

---

## 2. Le profil de départ — batch 32, 151 passes

```
   min ms     % Σ  max/min  workgroups    n  pass
 3303.899   75.1%    1.00×       12288    1  L22 UpsampleConv · upsample_conv_back_input
  835.832   19.0%    1.00×        6144    1  L17 UpsampleConv · upsample_conv_back_input
   31.755    0.7%    1.02×        1296    1  L26 Convolution · conv_back_weights
   30.060    0.7%    1.02×        2592    1  L21 Convolution · conv_back_weights
   22.869    0.5%    1.00×         648    1  L22 UpsampleConv · upsample_conv_back_weights
   19.891    0.5%    1.01×        2592    1  L17 UpsampleConv · upsample_conv_back_weights
   15.495    0.4%    1.01×        1296    1  L06 Convolution · conv_back_weights
   14.499    0.3%    1.01×        5184    1  L12 Convolution · conv_back_weights
    8.015    0.2%    1.00×       49152    1  L26 Convolution · conv_back_input
    8.006    0.2%    1.02×        1296    1  L03 Convolution · conv_back_weights
    7.896    0.2%    1.01×           1    1  L22 UpsampleConv · upsample_conv_back_bias
    7.586    0.2%    1.00×       24576    1  L22 UpsampleConv · forward
   92.562    2.1%                       139  … 139 more passes

BUDGET  (ronde 3 sur 5, la plus rapide au mur)
  Σ pass minima                  4398.4 ms   151 passes chronométrées
  Σ passes, cette ronde          4401.2 ms
  GPU span, cette ronde          4400.7 ms   → 0.0 ms (0.0%) ENTRE les passes
  host wall, cette ronde         4406.9 ms   → 6.2 ms hors du span GPU
  dont encodage CPU                 3.7 ms
  span sur toutes les rondes     4399.9 … 4411.2 ms  (étendue 0,3 %)
```

Le tableau se lit en une ligne : **deux passes sur 151 valent 94,1 % du pas.**

---

## 3. Le diagnostic

### 3.1 Dans les passes, ou entre elles ? — 100,0 % dedans

| batch | Σ passes | span GPU | entre les passes | mur − span | encodage CPU |
|---:|---:|---:|---:|---:|---:|
| 8 | 1124,8 ms | 1127,2 ms | **2,4 ms (0,2 %)** | 0,0 ms | 2,5 ms |
| 16 | 2217,4 | 2216,6 | **0,0 ms (0,0 %)** | 6,2 | 4,1 |
| 32 | 4401,2 | 4400,7 | **0,0 ms (0,0 %)** | 6,2 | 3,7 |
| 64 | 8780,6 | 8779,7 | **0,0 ms (0,0 %)** | 5,6 | 3,9 |

Après correction, sur un pas 16× plus court, le même verdict tient : 0,0 ms à
batch 16/32/64, et 2,3 ms (3,1 %) à batch 8 — c'est-à-dire que le coût fixe par
passe ne devient visible qu'une fois que les passes elles-mêmes sont descendues
sous le demi-milliseconde.

**Conséquence directe : l'hypothèse du §5.4 de `PERF_CONVOLUTION.md` est
tranchée, et par la négative.** Ce rapport-là supposait que les kernels
optimisés étaient descendus au niveau du coût fixe d'une passe, et que c'était
pour ça qu'un gain isolé de 3,7× ne rendait rien sur le pas. Le coût fixe d'une
passe, ici, est **inférieur à 20 µs** (2,4 ms / 151 à batch 8, et
indiscernable de zéro au-dessus). Ce n'est pas là que le temps allait.

Et le temps CPU le confirme du dehors : **2 à 4 ms d'encodage** pour un pas qui
en dure 4 400 — le fix d'orchestration de `BATCH_DISPATCH.md` a bien fait ce
qu'il annonçait, et il ne reste rien à y gagner.

### 3.2 Calcul, bande passante, ou latence ? — le test qui les sépare

Le discriminant ne demande aucune connaissance du pic de la machine, et il est
dans le tableau du §2. Pour une convolution, **les trois passes contractent les
mêmes tenseurs et font exactement le même nombre de MAC** — le forward somme sur
`(ky,kx,kz)`, `grad_input` sur `(ky,kx,k)`, `grad_weights` sur `(b,oy,ox)`, et
les trois produits triples ont le même cardinal. Sur `L26`
(32×32×96 → 32×32×48, noyau 3×3×96, batch 32), c'est **1 359,0 MMAC** dans les
trois cas :

| passe de L26 | ms | MMAC | GFLOP/s effectifs | GiB chargés à 8 B/MAC | GiB/s effectifs |
|---|---:|---:|---:|---:|---:|
| `forward` | 7,26 | 1359,0 | **375** | 10,12 | 1394 |
| `conv_back_input` | 8,02 | 1359,0 | **339** | 10,12 | 1262 |
| `conv_back_weights` | 30,09 | 1359,0 | **90** | 10,12 | 336 |

**Arithmétique identique, 4,1× le temps.** Le temps de ces passes n'est donc pas
fixé par l'arithmétique — le calcul est éliminé comme facteur limitant sans
qu'on ait eu à mesurer le pic de l'appareil.

Ce qui distingue les trois est **le chemin mémoire**. L'intensité arithmétique
nominale est la même et elle est catastrophique — 1 MAC pour 2 chargements, soit
**0,25 FLOP/octet** — mais la réutilisation, elle, ne l'est pas : dans le
forward, les threads voisins (mêmes `oy,ox`, `k` différents) relisent
**exactement** la même fenêtre d'entrée, et le cache la sert (1394 GiB/s
effectifs, très au-dessus de toute DRAM). Dans `grad_weights`, chaque thread
parcourt l'axe des positions à grandes foulées dans deux grands tableaux, et la
réutilisation s'effondre (336 GiB/s).

Verdict : **le pas est limité par la hiérarchie mémoire**, pas par le calcul et
pas par la latence de dispatch. Réserve honnête : je n'ai pas mesuré la bande
passante DRAM ni le pic FLOP de cette machine, donc la position exacte sur le
roofline est **encadrée, pas épinglée** — ce qui est établi, c'est le sens
(mémoire, pas calcul) et le rapport entre passes.

### 3.3 …sauf que les deux passes qui dominaient n'étaient dans aucune des trois catégories

`upsample_conv_back_input` n'était limitée ni par le calcul, ni par la bande
passante, ni par la latence : elle était limitée par **des itérations de boucle
qui ne servaient à rien**.

Le kernel, pour chaque élément d'entrée, balayait `K × OH × OW × KH × KW` taps de
la carte de sortie et faisait `continue` sur tout ce qui ne tombait pas dans sa
fenêtre d'agrandissement `scale × scale` :

| couche | itérations/thread | dont utiles | gaspillage | itérations totales | temps | débit d'itérations |
|---|---:|---:|---:|---:|---:|---:|
| L22 (16×16×96 → 32×32×48) | 442 368 | 1 728 | **256×** | 3,48e11 | 3 303,9 ms | **105 G it/s** |
| L17 (8×8×192 → 16×16×96) | 221 184 | 3 456 | **64×** | 8,70e10 | 835,8 ms | **104 G it/s** |

Les deux passes tournaient au **même** débit d'itérations, et leurs temps sont
dans le rapport exact de leurs comptes d'itérations. C'est la signature d'un
kernel limité par l'émission d'instructions, et c'est ce qui explique le facteur
4 entre L22 et L17 : le gaspillage vaut `OH·OW / scale²`, donc il **empire avec
la résolution de la couche**. Le U-net remonte vers 32×32 ; la couche la plus
haute payait le plus cher.

---

## 4. La correction n°1 — inverser la carte au lieu de la balayer

La fenêtre est connue en forme fermée. Pour l'élément d'entrée `(iy, ix, iz)`,
`up_y` parcourt exactement `[iy·scale, (iy+1)·scale)` — donc dans `[0, up_h)` par
construction, ce qui est pourquoi les deux tests de bornes sur `up_y`/`up_x` ont
**disparu** et pas seulement été déplacés — et `oy = up_y + pad_y − ky` inverse
la carte du forward.

`k` passe à l'intérieur, comme dans `conv_back_input` depuis
`PERF_CONVOLUTION.md` §2.3 et pour les deux mêmes raisons : la géométrie des taps
ne dépend pas de `k` et serait recalculée `K` fois, et `grad_output` est alors
parcouru de façon contiguë (`k` est l'axe rapide du layout HWK).

| passe | avant | après | facteur |
|---|---:|---:|---:|
| `L22 upsample_conv_back_input` | 3 303,9 ms | **7,57 ms** | **437×** |
| `L17 upsample_conv_back_input` | 835,8 ms | **7,34 ms** | **114×** |
| le pas complet (span GPU) | 4 400,7 ms | **277,9 ms** | **15,8×** |

### 4.1 Preuves d'équivalence

Fichier : `crates/batlab-core/src/model/upsample_conv_equivalence_tests.rs`.
Même standard que `conv_equivalence_tests`.

1. **Ancien contre nouveau, sur les mêmes tampons.**
   `shader/legacy/back_upsample_conv_naive.wgsl` conserve le shader d'avant,
   verbatim. Chaque test construit un vrai modèle avec le nouveau shader,
   l'exécute, puis redispatche le kernel legacy sur **exactement le même bind
   group**, avec les comptes de workgroups de l'époque.
2. **Un oracle f64** écrit en **scatter** sur la carte du forward
   (`(oy,ox,ky,kx) → up_y = oy+ky−pad`, puis `iy = up_y/scale`). Le nouveau
   kernel **inverse** cette carte : l'oracle valide donc l'inversion au lieu de
   la répéter.
3. **Différences finies** sur la loss, qui ne savent rien de l'agrandissement.

Huit formes : les deux géométries de `Color_Diffusion_XL` (à canaux réduits pour
que l'oracle f64 reste un test et pas un banc), padding `Valid`, `scale` 3,
`scale` **1** (la fenêtre se réduit à un seul `up_y` — c'est là qu'un off-by-one
se cacherait), noyau 1×1 (`pad = 0`), noyau 5×5 (`oy` sort des bornes des deux
côtés), et une entrée plus petite qu'un workgroup.

| grandeur | écart max nouveau ↔ legacy | erreur max face à f64 | seuil |
|---|---:|---:|---:|
| `grad_input` | 8,8e-7 | 6,4e-7 | 1e-4 |
| `grad_weights` | 0,0 (kernel inchangé) | 9,5e-7 | 1e-4 |
| `grad_bias` | 0,0 (kernel inchangé) | 9,6e-7 | 1e-4 |

La clause « jamais moins précis » (`erreur(nouveau) ≤ 1,5 × erreur(ancien)`) tient
sur les huit formes. **Marge la plus serrée : 1,43× sur la forme 5×5**, où les
deux implémentations sont à 4 ULP du plancher d'arrondi f32 (5,5e-7 contre
3,9e-7) — c'est dit ici parce que c'est la seule assertion du lot qui ne passe
pas avec un ordre de grandeur de marge.

### 4.2 Les tests ne sont pas vacuous — six mutations, six attrapées

| mutation | tests qui échouent |
|---|---:|
| l'offset de padding supprimé | 2 |
| la fenêtre élargie d'une ligne | 2 |
| la foulée du poids figée | 2 |
| le test de borne inférieure supprimé | 2 |
| `ky`/`kx` échangés dans l'index de poids | 2 |
| le gradient biaisé de +0,1 % | 1 |

La première n'était attrapée que par **un** test au premier passage : le contrôle
par différences finies tournait sur une forme à padding `Valid`, où l'offset vaut
zéro et où le supprimer ne change rien. Le contrôle tourne maintenant sur les
deux modes de padding. C'est un renforcement que seule la mutation a révélé, et
il est noté dans le fichier de test.

### 4.3 En conditions réelles

300 pas d'entraînement appariés sur `Color_Diffusion_XL` (batch 4, Adam, lr 1e-3,
`cifar10_rgb`), ancien puis nouveau shader, tout le reste identique — l'init des
poids est à graine fixe et les tirages de timestep/bruit dérivent du compteur de
pas, donc les deux runs résolvent le même problème.

```
step   0  1.37026012 | 1.37026012      step 150  0.05403091 | 0.05403088
step  25  0.23124193 | 0.23124212      step 200  0.04283151 | 0.04283150
step  50  0.09855663 | 0.09855662      step 250  0.59533805 | 0.59533817
step 100  0.08717153 | 0.08717152      step 299  0.02813937 | 0.02813936
                                                     (ancien | nouveau)
```

Sur les 540 enregistrements JSONL : écart **relatif** max sur `train_loss`
**9,7e-7**, sur `train_probe` (la loss par tranche de `t`, le diagnostic qui
compte pour un modèle de diffusion) 2,3e-4, **0** valeur non finie, **0**
différence de structure.

Et le même run : **180,50 s → 16,85 s**.

---

## 5. Ce que le profil dit une fois la première correction faite

### 5.1 Le nouveau profil — batch 32 (les deux corrections)

```
   min ms     % Σ  max/min  workgroups    n  pass
   30.086   11.4%    1.00×        5184    1  L26 Convolution · conv_back_weights
   29.618   11.2%    1.01×        5184    1  L21 Convolution · conv_back_weights
   22.518    8.5%    1.00×         648    1  L22 UpsampleConv · upsample_conv_back_weights
   19.871    7.5%    1.00×        2592    1  L17 UpsampleConv · upsample_conv_back_weights
   14.878    5.6%    1.00×        5184    1  L06 Convolution · conv_back_weights
   14.500    5.5%    1.01×        5184    1  L12 Convolution · conv_back_weights
    8.022    3.0%    1.01×       49152    1  L26 Convolution · conv_back_input
    7.818    3.0%    1.07×           1    1  L22 UpsampleConv · upsample_conv_back_bias
    7.583    2.9%    1.00×       24576    1  L22 UpsampleConv · forward
    7.563    2.9%    1.00×       12288    1  L22 UpsampleConv · upsample_conv_back_input
    7.527    2.8%    1.01×        5184    1  L03 Convolution · conv_back_weights
    7.416    2.8%    1.00×        5184    1  L09 Convolution · conv_back_weights
    7.361    2.8%    1.01×        6144    1  L17 UpsampleConv · upsample_conv_back_input
    7.287    2.8%    1.00×       24576    1  L21 Convolution · conv_back_input
    7.257    2.7%    1.01×       24576    1  L26 Convolution · forward
    …
    0.020    0.0%    1.16×        2592    1  L21 Convolution · adam
    0.006    0.0%    1.24×         648    1  L22 UpsampleConv · adam

BUDGET  (ronde 2 sur 5, la plus rapide au mur)
  Σ pass minima                   264.4 ms   151 passes chronométrées
  Σ passes, cette ronde           266.1 ms
  GPU span, cette ronde           265.1 ms   → 0.0 ms (0.0%) ENTRE les passes
  host wall, cette ronde          263.8 ms   → -1.3 ms hors du span GPU
  dont encodage CPU                 2.4 ms
  span sur toutes les rondes      260.6 … 265.1 ms  (étendue 1,7 %)
```

Deux choses valent d'être notées au passage. **Les vingt passes de l'optimiseur
Adam pèsent ensemble moins de 0,2 ms** — le « candidat évident au regroupement »
qu'un comptage de dispatches désignerait n'existe pas dans la mesure. Et le
`upsample_conv_back_bias` de L22 coûte **7,8 ms avec UN seul workgroup** : c'est
la seule passe du graphe réellement affamée de parallélisme, et elle ne pèse que
3 %.

Le pas est devenu **plat** : plus de passe dominante, 48 % dans six passes
`*_back_weights`, et une queue de 139 passes à 35 %.

### 5.2 Le débit effectif, avant et après

Travail réel d'un pas, en ne comptant que les MAC **utiles** : 238 MMAC par
échantillon en avant, et les trois passes d'une convolution ayant le même
cardinal, ≈ 3× pour l'ensemble, soit **45,7 GFLOP par pas** à batch 32.

| | temps GPU | GFLOP/s utiles | trafic à 8 B/MAC |
|---|---:|---:|---:|
| avant | 4 400,7 ms | **10,4** | 38,7 GiB/s |
| après | 265,1 ms | **172** | 642 GiB/s |

Les 10 GFLOPS de l'ordre de mission sont donc confirmés au chiffre près — et
expliqués : ce n'étaient pas 10 GFLOPS de calcul lent. Les deux passes du §3.3
exécutaient à elles seules **4,35e11 itérations de boucle** pour n'en rendre que
2,7e9 en produits utiles — **99,4 % de leurs itérations se terminaient par un
`continue`** — et ces deux passes valaient 94,1 % du pas.

### 5.3 Ce qui n'a PAS changé, et c'était la vraie question

Le trafic hôte→GPU est **nul par pas** (dataset résident, `--gpu-limits native`).
Le CPU encode le pas en 2 à 4 ms. Les 151 dispatches ne bougent pas. Aucune de
ces trois grandeurs n'était le problème, et aucune ne l'est devenue.

### 5.4 Le coût par échantillon plat de 32 à 64 — ce qui sature vraiment

| batch | avant : span | ms/échantillon | après la correction n°1 : span | ms/échantillon |
|---:|---:|---:|---:|---:|
| 8 | 1 127,2 ms | **140,9** | 73,3 ms | **9,16** |
| 16 | 2 216,6 | **138,5** | 141,1 | **8,82** |
| 32 | 4 400,7 | **137,5** | 277,9 | **8,68** |
| 64 | 8 779,7 | **137,2** | 558,1 | **8,72** |

(Balayage pris avant la correction n°2 ; celle-ci baisse la colonne de droite de
4 à 6 % de façon uniforme — cf. le tableau du §6 — sans changer la platitude,
qui est ce que ce paragraphe mesure.)

La platitude ne commence pas à 32 : elle est déjà là à **batch 8**, à 3 % près,
avant comme après la correction. Il n'y a donc pas de seuil de saturation entre
32 et 64 — l'ordre de mission lisait ces deux points comme le signe qu'on
saturait quelque chose *à cet endroit-là*, et l'instrument montre que la courbe
est plate partout où on la regarde.

L'explication est structurelle, et c'est un acquis de `BATCH_DISPATCH.md` : le
graphe encode **151 dispatches quel que soit le batch**, et chacun fait un
travail exactement proportionnel au batch. Le seul coût fixe est l'encodage CPU
(2 à 4 ms), soit 1,4 % du pas à batch 32 après correction et 0,08 % avant. **Il
n'y a rien à amortir.** Un coût par échantillon plat est ici la signature d'un
graphe déjà saturé en travail à batch 8, pas d'une limite atteinte à 32.

Corollaire pratique : sur cette machine, **augmenter le batch n'achète pas de
débit**. Ce qu'il achète est ailleurs — moins de rechargements de dataset sur un
corpus streamé (`GPU_RESOURCES.md` §5.1), et un gradient moins bruité.

---

## 6. La correction n°2 — recalibrer les lanes de réduction

Le profil désigne les six passes `conv_back_weights` / `upsample_conv_back_weights`
(48 % du pas). Le §3.2 dit que leur handicap est le chemin mémoire ; le seul
réglage existant qui touche à ce chemin est `ConvolutionType::reduction_lanes`,
dont la constante `TARGET_THREADS` avait été calibrée sur
`bench_conv_reduction_lanes` — c'est-à-dire **un kernel dispatché 200 fois en
boucle, sur `Greyscale_Diffusion`, avant que l'axe batch existe**.

Re-balayée avec `--profile-step`, donc sur le pas réel :

| `TARGET_THREADS` | XL b8 | XL b32 | XL b64 | `Color_Diffusion_L` b32 |
|---:|---:|---:|---:|---:|
| 65 536 (avant) | 70,7 | 277,5 | 554,3 | 135,3 |
| 131 072 | 69,0 | 270,3 | 546,8 | 129,3 |
| **262 144** | **67,7** | **264,5** | **524,4** | **127,3** |
| 524 288 | 67,0 | 265,4 | 545,8 | 136,9 |
| 1 048 576 | — | 290,6 | — | — |

−4,2 % à −5,9 %, sur **deux modèles et trois tailles de batch**, avec un optimum
intérieur net : au-delà, le `conv_back_weights` de la plus grosse couche double
(31,7 → 59,4 ms à 20 736 workgroups) — la réduction en arbre et ses barrières
coûtent alors plus que la somme qu'elles découpent.

Preuves : les tests d'équivalence de la convolution sont **pilotés par la forme**,
donc ils exercent automatiquement les nouveaux comptes de lanes — neuf formes,
oracle f64, clause « jamais moins précis », et
`conv_reduction_lanes_agrees_with_dispatch` qui vérifie que le dispatch couvre
exactement tous les poids et tous les biais. Aucun test n'a été touché. Un run
apparié de 300 pas donne un écart relatif max de **7,3e-7** sur `train_loss`.

Ce que ce changement **n'est pas** : le point structurel de `BATCH_DISPATCH.md`
§9. `positions` reste le compte **par échantillon** alors que la boucle en
couvre `batch ×` plus ; le faire proprement demande de faire descendre le batch
jusqu'au mot de lanes de l'uniforme, et reste ouvert.

---

## 7. Le gain final, bout en bout

Trois mesures, trois instruments, même conclusion.

### 7.1 Le pas lui-même, chronométré par le GPU (batch 32)

| | span GPU par pas | facteur |
|---|---:|---:|
| avant | 4 400,7 ms | |
| + inversion de la carte (§4) | 277,9 ms | 15,8× |
| + recalibrage des lanes (§6) | **265,1 ms** | **16,6×** |

### 7.2 Le processus complet, même commande que la ligne de départ de la mission

`--headless-train Color_Diffusion_XL --steps 11 --batch 32 --optimizer adam
--lr 1e-3 --dataset cifar10_rgb.batraw`, temps réel du processus :

```
avant : 50,62 s        après : 4,94 s        10,2×
```

Le facteur au mur (10,2×) est **inférieur** à celui du pas (16,6×), et c'est
attendu : les ~2,0 s restantes sont du démarrage de processus, le téléversement
unique du dataset de 586 Mio et les deux sondes par tranche de `t` — un coût
fixe qui pesait 4 % du run avant et en pèse 41 % maintenant. Le dire dans l'autre
sens : le calcul du pas ne domine plus un run court.

### 7.3 Le recalibrage des lanes seul, apparié, avec contrôle nul

100 pas, batch 32, exécutions entrelacées A/B/B/A dans une même fenêtre, plus le
même binaire contre lui-même pour établir le plancher de bruit :

| série | temps réel (s) |
|---|---|
| base (correction n°1 seule) | 30,30 · 30,31 · 30,32 · 30,30 |
| + lanes (correction n°2) | 28,89 · 29,01 · 28,99 · 28,87 |
| **contrôle nul** (base contre base) | 30,38 / 30,28 → **0,33 %** |

Minimum contre minimum : **30,30 → 28,87 s, soit −4,7 %** — quatorze fois le
contrôle nul, et les étendues des deux bras ne se recouvrent pas (30,30–30,32
contre 28,87–29,01). Le chiffre reproduit **exactement** le −4,7 % que le
profileur donnait à batch 32 sur la seule somme des passes (§6). Deux
instruments indépendants, même nombre.

### 7.4 Et 300 pas à batch 4

180,50 s → 16,85 s (§4.3), avec la trajectoire de loss identique à 9,7e-7.


---

## 8. Ce qui n'a pas été corrigé, et pourquoi

**Les tuiles / le register-blocking du forward.** `PERF_CONVOLUTION.md` §3 les a
mesurés et rejetés ; `BATCH_DISPATCH.md` §9 note que le verdict a expiré, parce
qu'il reposait sur des kernels à 16 workgroups. Le profil actuel dit que le
forward pèse **7,6 ms sur 264** pour la plus grosse couche et qu'il tourne déjà à
1335 GiB/s effectifs, c'est-à-dire largement servi par le cache. La piste
existe, mais elle vise 3 % du pas : elle n'est pas la prochaine.

**Fusionner GroupNorm + SiLU + conv.** Le profil dit que les 9 GroupNorm et les 8
Activation pèsent ensemble moins que la seule `L26 conv_back_weights`. Fusionner
supprimerait des passes — or il n'y a **rien entre les passes** (§3.1). Le gain
serait le trafic d'activation épargné, pas le lancement ; sur mémoire unifiée et
avec les tampons concernés déjà en cache, il n'est pas évident, et rien dans la
mesure ne le désigne. Non fait.

**fp16 pour les activations.** C'est la piste que le diagnostic désigne
vraiment — le pas est limité par la hiérarchie mémoire, et l'intensité
arithmétique est de 0,25 FLOP/octet, donc halver l'octet est la seule chose qui
déplace le plafond. `SHADER_F16` est disponible sur cet adaptateur (vérifié).
Elle n'est pas faite ici parce qu'elle touche à la **numérique** de tout le
moteur : plage dynamique de `grad_output` sur un modèle de diffusion, chaînes
d'accumulation déjà multipliées par `B` (`BATCH_DISPATCH.md` §8), format de
checkpoint. C'est une mission, pas un patch en fin de mission.

**Donner des lanes de réduction à `upsample_conv_back_weights`.** Elle n'en a
pas (un thread par poids, 32 768 positions chacun) et pèse 22,5 + 19,9 = 42,4 ms,
soit 16 % du pas. Mais la mesure décourage la piste plutôt qu'elle ne l'appelle :
à 648 workgroups sans lanes, elle rend **60 GMAC/s**, contre 43 GMAC/s pour
`L26 conv_back_weights` qui, lui, a des lanes. Le découpage n'est pas ce qui
manque à ces passes. Non fait, et la raison est chiffrée plutôt que devinée.

**`--profile` sur `--headless-train`.** Une sous-commande dédiée a été préférée :
la boucle d'entraînement rapporte une loss, lance une sonde par tranche de `t` et
échantillonne une image selon un calendrier, et chacun de ces trois soumet du
travail GPU. Un « profil d'un pas » qui aurait silencieusement inclus une chaîne
de débruitage de 256 pas tous les 200 pas aurait été le profil d'autre chose.

---

## 9. Ce qui reste

- **fp16 pour les activations** — la seule piste que le diagnostic désigne pour
  un gain d'un facteur, et une mission à elle seule (§8).
- **Rendre `reduction_lanes` conscient du batch** — `BATCH_DISPATCH.md` §9,
  toujours ouvert ; le §6 n'a recalibré que la constante.
- **Le pic de la machine n'est pas mesuré.** Le §3.2 encadre la position sur le
  roofline au lieu de l'épingler. Un micro-banc de bande passante et un de FLOP
  la fixeraient, et diraient combien il reste à prendre.
- **Profiler l'inférence.** `--profile-step` ne couvre que le pas
  d'entraînement. La chaîne inverse fait 256 allers-retours CPU↔GPU par image
  (`GPU_RESOURCES.md` §5.2) et n'a jamais été chronométrée passe par passe.
- **Le plancher de 20 µs par passe** (§3.1) ne se voit qu'à batch 8. Sur un
  modèle plus petit ou une inférence à batch 1, il redeviendra une part
  significative, et c'est là que la fusion de passes retrouverait un sens.

---

## Fichiers modifiés

| fichier | changement |
|---|---|
| `crates/batlab-core/src/profile.rs` | **nouveau** — le profileur, l'agrégation, les tests |
| `crates/batlab-core/src/gpu_context.rs` | `TIMESTAMP_QUERY` optionnel, `compute_pass`, `wait_idle` |
| `crates/batlab-core/src/model/layer.rs` | les cinq sites de dispatch passent par la porte, + `index` |
| `crates/batlab-core/src/model/model.rs` | `resolve_profiler` dans l'encodeur du pas, index des couches |
| `crates/batlab-core/src/model/training/{diffusion,dataset}.rs` | les deux autres sites de dispatch |
| `crates/batlab-core/src/model/layer_types/mod.rs` | `LayerTypes::variant_name` |
| `crates/batlab-core/src/model/shader/back_upsample_conv.wgsl` | **la correction n°1** |
| `crates/batlab-core/src/model/shader/legacy/back_upsample_conv_naive.wgsl` | **nouveau** — fixture |
| `crates/batlab-core/src/model/upsample_conv_equivalence_tests.rs` | **nouveau** — preuves |
| `crates/batlab-core/src/model/layer_types/convolution.rs` | **la correction n°2** — `TARGET_THREADS` |
| `crates/batlab/src/main.rs` | `--profile-step`, le tableau, le budget, l'aide |
