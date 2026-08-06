# Le flux : un régime sans point de rebroussement — rapport de mission

Branche : `perpetual-flux` · Plateforme : macOS (Darwin 25.4.0) · wgpu 28 ·
Poids : `Greyscale_Diffusion_L/night_run.ckpt`

## Résumé

Retour d'usage après `CLIMB_COHERENCE.md` : la remontée ne frise plus, mais **les
phases elles-mêmes se sentent** — « ça frise entre les étapes ». Le correctif
demandé ne consistait pas à adoucir le rebroussement mais à **le supprimer** :
un régime qui tient un seul niveau `t*` et n'en repart jamais.

| | Commit | |
|---|---|---|
| 1 | `8a5f246` | le régime flux — un churn stationnaire à `t*`, sans phase |
| 2 | `bc38a41` | campagne de mutation M1–M5, deux angles morts des tests refermés |
| 3 | `1284128` | `tools/flux_analysis.py` — mesurer un régime sans phase |
| 4 | `be42c27` | le cadran `t*`/`t_r` était au contrat et n'était pas parsé (§7) |
| 5 | `1a0f516` | `PERPETUAL_FLUX.md` + planches |
| 6 | *(ce commit)* | la première frame du churn s'appelait `descente` (§7.5) |

`cargo test --workspace` : **130 verts** dans `bat_building` (4 ignorés) +
**12 dans `main`** (9 avant la mission). Aucun test affaibli.

Le chiffre qui dit que le rebroussement est mort — distribution de `|Δ|`
image-à-image du pane x̂₀ (celui qui porte l'image), en niveaux de gris 8 bits,
sur 1 400 frames du vrai modèle :

| | errance | respiration | **flux** |
|---|---|---|---|
| médiane | 0,00 (x̂₀ figé 51,6 % du temps) | 0,00 (50,3 %) | **3,59** |
| max | **32,92** | **32,55** | **5,38** |
| max / médiane | ∞ | ∞ | **1,50** |
| pire pixel, max | 165,4 | 170,7 | **31,7** |

Et le fait qui les explique : sur les 10 frames d'errance où x̂₀ bouge de plus de
10 niveaux, **10 sont un rebroussement** ; en respiration, **21 sur 21**. Le flux
n'a aucune frame au-dessus de 10, et **aucun rebroussement** — il n'en existe pas
dans ce régime.

Trois autres résultats, chacun demandé par la mission :

- **Ornstein-Uhlenbeck : NO-GO**, mesuré et non supposé (§4). Le flux garde 75 %
  de son mouvement 25 frames plus tard avec des incréments à corrélation nulle —
  il erre, il ne tremble pas — et son `|Δ|` par frame (3,59) est celui d'une
  descente d'errance (3,10), que personne n'a jamais reproché. Critères de
  réouverture écrits.
- **TUI réel validé** (§6) : entrée en flux depuis chaque régime, `t*` déplacé en
  cours de run par `[↑]`/`[↓]` — marché un cran par frame, jamais sauté — stderr
  vide, sortie code 0.
- **Deux bugs sortis par l'agent de test aveugle** (§7), un par passe. Passe 1 :
  `--t-star` et `--t-r` figuraient au contrat public et n'étaient parsés nulle
  part, ce qu'un flag inconnu ignoré en silence rendait indétectable. Passe 2 :
  la **première frame du churn était étiquetée `descente`** dans le dump et dans
  le TUI — un frame de retard, attrapé de l'extérieur par l'amplitude (40 %
  au-dessus d'un pas inverse au même niveau) parce qu'aucun de mes 129 tests ne
  pouvait le voir : ils lisaient tous déjà la bonne notion de phase, pendant que
  le dump en écrivait une autre (§7.5). Les deux corrigés et testés, et les deux
  chaînes de mesure — la sienne et la mienne — s'accordent à trois décimales.

---

## 1. Protocole

Trois runs headless comparés à réglages identiques — `--depth 64`, `--magnitude
1`, graine 7, **1 600 actions chacun**, mêmes poids :

```bash
cargo run --release -p main -- --headless-perpetual Greyscale_Diffusion_L \
  --checkpoint Models/Greyscale_Diffusion_L/pretrained_weights/night_run.ckpt \
  --regime flux --t-star 64 --actions 1600 --seed 7 --dump flux.bin
python3 tools/flux_analysis.py compare flux.bin wander.bin breathe.bin
```

*(Les dumps de ce rapport ont été produits avec `--depth 64`, l'orthographe qui
était alors la seule parsée ; `--t-star` en est l'alias exact depuis `be42c27` —
voir §7.)*

Trois choix de méthode, chacun pour une raison :

- **Dumps `f32`, jamais des PNG.** En flux le changement image-à-image du pane
  x̂₀ vaut ~3,6 niveaux de gris ; `CLIMB_COHERENCE.md` §3.2 a déjà montré qu'un
  PNG 8 bits plafonne les corrélations de grain vers 0,87 là où le `f32` dit
  1,000. Mesurer sur PNG serait mesurer le quantificateur. `--dump` écrit les
  deux panes de chaque frame en `f32` brut.
- **Le prologue est écarté.** Les trois régimes s'ouvrent sur la même descente
  depuis le bruit pur (191 à 256 frames selon le régime), qui porte les plus
  gros `|Δ|` de tout le run. L'inclure comparerait leur préambule commun.
- **Mesurer le buffer, pas l'écran.** Le chemin headless et la fenêtre live
  composent par la même fonction ; et `CLIMB_COHERENCE.md` §6 documente que
  `screencapture` sur la fenêtre wgpu étrangle le run (20 captures pour 2
  incréments). Les frames mesurées **sont** ce que la fenêtre reçoit.

Contre-épreuve de graine : tout le tableau anti-saut a été refait sous la graine
909 (§2.4). Aucune conclusion ne repose sur un tirage.

---

## 2. Anti-saut

### 2.1 Le pane x̂₀ — l'image

| | flux | errance | respiration |
|---|---|---|---|
| médiane | 3,59 | 0,00 | 0,00 |
| p95 | 4,49 | 3,80 | 4,58 |
| p99 | 4,93 | 4,42 | 26,16 |
| **max** | **5,38** | 32,92 | 32,55 |
| **max / médiane** | **1,50** | ∞ | ∞ |
| secousses (> 3× médiane) | **0 / 1408** | 1343 / 1343 | 1375 / 1375 |
| pire pixel, médiane | 17,00 | 0,00 | 0,00 |
| pire pixel, max | **31,65** | 165,39 | 170,71 |
| x̂₀ figé (`Δ` nul) | **0,0 %** | 51,6 % | 50,3 % |
| hors rebroussement (méd / max) | — | 0,00 / 4,54 | 0,00 / 5,09 |
| **au rebroussement (méd / max)** | — | **8,66 / 32,92** | **13,93 / 32,55** |
| rebroussements | **0** | 20 | 41 |

Le critère demandé — *en flux, max ≈ médiane* — est tenu à **1,50** sur x̂₀ et à
**1,10** sur x_t, contre un rapport infini pour les deux régimes à cycles. Le
« ∞ » n'est pas un artefact de division : la médiane d'errance est un vrai zéro,
parce que **le modèle n'est pas appelé pendant une remontée** et que x̂₀ y reste
l'estimation périmée de la descente précédente. La moitié des frames ne bougent
pas du tout, puis une frame bouge de 33 niveaux — un pixel isolé de 165. C'est la
forme d'un saut, et c'est ce que « ça frise entre les étapes » décrivait.

**Toutes les grosses secousses sont des rebroussements**, sans exception :

| seuil sur `\|Δx̂₀\|` | errance : frames | dont au rebroussement | respiration : frames | dont au rebroussement |
|---|---|---|---|---|
| > 10 | 10 | **10** | 21 | **21** |
| > 15 | 10 | **10** | 20 | **20** |
| > 20 | 9 | **9** | 19 | **19** |

(Un régime à cycles a deux sortes de rebroussement : descente→remontée, où x̂₀
**gèle** — `|Δ| = 0` — et remontée→descente, où il **dégèle d'un coup**. Le
premier ne se voit pas dans un `|Δ|` et se voit à l'écran ; c'est lui qui rend la
médiane nulle.)

### 2.2 Le pane x_t — le latent bruité

| | flux | errance | respiration |
|---|---|---|---|
| médiane | 20,41 | 1,33 | 10,21 |
| max | 22,39 | 14,93 | 15,08 |
| **max / médiane** | **1,10** | **11,19** | 1,48 |
| secousses (> 3× médiane) | **0 / 1408** | 614 / 1343 | 0 / 1375 |
| médiane en descente | — | 10,49 | 12,68 |
| médiane en remontée | — | 1,17 | 1,58 |

Ici le défaut d'un régime à cycles n'est pas l'amplitude d'une frame isolée mais
**le changement de régime de l'amplitude** : en errance, x_t bouge de 10,5
niveaux par frame en descente et de 1,2 en remontée — un facteur **9** qui
s'installe à chaque rebroussement et dure une demi-période. Le flux tient 20,41
avec un p99 à 21,53 : la même cadence de bout en bout, pour toujours.

La contrepartie est visible dans la première colonne : à `t* = 64`, le pane x_t
du flux churne **deux fois plus vite** que la descente d'une errance. Ce n'est
pas un défaut du régime, c'est la loi du niveau tenu — et c'est réglable au
cadran (§5).

### 2.3 Ce que ça donne en images

`flux_samples/flux_t64_consecutives.png` — **12 frames consécutives de flux**,
pane x̂₀. L'image se déforme continûment ; aucune vignette n'est la copie de sa
voisine, aucune ne rompt avec elle.

`flux_samples/errance_rebroussement.png` — les **12 frames d'errance autour d'un
rebroussement** (frames 1355–1366, `t = 59 → 64 → 58`), même pane, même échelle.
Six vignettes **rigoureusement identiques** — la remontée, x̂₀ gelé — puis une
coupe franche sur une autre image, de contraste et de contenu différents. Les
deux planches côte à côte sont l'argument entier.

### 2.4 Contre-épreuve de graine

| | graine 7 | graine 909 |
|---|---|---|
| `\|Δx_t\|` médiane / max / rapport | 20,41 / 22,39 / **1,10** | 20,39 / 22,04 / **1,08** |
| `\|Δx̂₀\|` médiane / max / rapport | 3,59 / 5,38 / **1,50** | 3,58 / 5,65 / **1,58** |
| secousses | 0 / 1408 | 0 / 1408 |

---

## 3. Stationnarité et dérive

Fenêtre de mesure : les frames de flux moins leur premier quart (l'approche vers
`t*`) — 1 056 frames pour le run de référence.

### 3.1 Le couloir tient

| | flux | errance | respiration |
|---|---|---|---|
| std(x_t) moyen | **0,7379 ± 0,0215** | 0,6577 | 0,9438 |
| std(x_t) min→max par frame | **0,674 → 0,802** | 0,429 → 0,850 | 0,825 → 1,014 |
| std(x_t) 1er → 5e cinquième | **0,742 → 0,723** | 0,513 → 0,723 | 0,716 → 0,979 |

Sur 1 056 frames, le niveau de bruit du flux se tient dans une bande de ±3 %
autour de sa moyenne, et le premier cinquième du run ne se distingue pas du
dernier (0,742 contre 0,723 ; graine 909 : 0,744 → 0,712). **Aucune fuite lente**
— ni gonflement ni affaissement du niveau, qui était le risque du montage
(un `reverse_step` et un `forward_step` qui ne se compenseraient pas exactement
dériveraient d'un cran par frame, ce que la mutation M3 a simulé et que les
tests attrapent).

Les colonnes errance/respiration ne sont pas un contre-exemple mais un rappel :
un régime à cycles **balaie** son niveau par construction. Le couloir n'est un
critère que là où la stationnarité est promise.

L'énergie de l'image, std(x̂₀), vaut 0,2805 ± 0,0725 : elle respire au fil du
contenu (une image contrastée, puis une plate) sans tendance — 0,266 au premier
cinquième, 0,230 au dernier.

### 3.2 Le flux erre, il ne vibre pas

Corrélation entre deux frames séparées de `k` frames. Le **plancher** de ces
colonnes n'est pas zéro mais la corrélation entre deux runs de graines
différentes — ils sortent du même modèle et se ressemblent sans être liés :
**−0,029 (x_t) et −0,181 (x̂₀)**.

| décalage | x_t | x̂₀ |
|---|---|---|
| 1 | +0,963 | **+0,991** |
| 25 | +0,407 | +0,838 |
| 100 | +0,083 | +0,661 |
| 300 | **+0,055** | +0,599 *(graine 909 : +0,191)* |

Lecture : d'une frame à la suivante, l'image est à **0,991** d'elle-même — c'est
la continuité. À 300 frames (10 s à 30 pas/s), le latent est **au plancher**
(+0,055 puis +0,001 sous l'autre graine) et l'image a perdu de moitié aux trois
quarts de sa mémoire selon la graine. La vitesse de dérive à long horizon dépend
du contenu traversé, ce qui est attendu d'une marche : c'est la seule colonne du
rapport qui bouge d'une graine à l'autre, et elle est rapportée dans les deux.

À titre de comparaison, l'errance à `t_r = 64` garde **+0,780** à 300 frames sur
x̂₀ : un cycle ré-bruite jusqu'à 64 puis revient sur une image **parente** de la
précédente. Le flux à `t* = 64` ne dérive donc pas moins que l'errance — il
dérive plus, sans jamais s'arrêter pour le faire.

---

## 4. Décision Ornstein-Uhlenbeck : **NON**

La spec exigeait de trancher, mesures à l'appui, dans un sens comme dans l'autre :
un micro-scintillement résiduel justifie-t-il du bruit corrélé en temps ?

### 4.1 Ce qu'est un scintillement, et comment on le mesure

Scintiller, c'est bouger sans aller nulle part : un incrément défait le
précédent. Deux mesures le distinguent d'une dérive :

- **la corrélation entre incréments successifs** — `corr(x_{i+1}−x_i,
  x_{i+2}−x_{i+1})`. Négative = chaque pas revient sur le précédent.
- **le mouvement conservé à 25 frames** — `MSD(25) / (25 · MSD(1))`, c'est-à-dire
  ce qui reste debout d'un pas, 25 frames plus tard, rapporté à ce qu'une marche
  sans mémoire garderait. **1,0 = marche libre ; 0,04 = pur tremblement** (le
  déplacement plafonne à un pas).

| | flux | errance | respiration |
|---|---|---|---|
| corr des incréments x̂₀ | **−0,018** | +0,003 | −0,008 |
| corr des incréments x_t | **−0,021** | +0,483 | +0,474 |
| conservé à 25 frames, x̂₀ | **0,75** | 0,96 | 0,86 |
| conservé à 25 frames, x_t | **0,64** | 1,08 | 1,06 |
| exposant MSD α, x̂₀ | +0,93 | +0,99 | +0,97 |

Le flux **garde les trois quarts de son mouvement** vingt-cinq frames plus tard,
avec des incréments à corrélation nulle (−0,018, à comparer au −0,001 que
`CLIMB_COHERENCE.md` §1.1 relevait sur l'ancienne remontée *crépitante* — mais là
c'était le **changement d'image entier** qui était blanc, ici c'est un pas de
1,3 % de la dynamique). Il n'y a **pas de composante qui se défait** : le léger
`α < 1` est le confinement d'un processus stationnaire — l'image reste une image,
son déplacement doit plafonner — et non un tremblement.

### 4.2 L'échelle, comparée à ce que l'utilisateur accepte déjà

`|Δx̂₀|` médiane en flux : **3,59** niveaux de gris par frame.
`|Δx̂₀|` médiane pendant la **descente d'une errance** : **3,10**.

Le flux bouge l'image, par frame, **de ce que la descente d'errance bouge déjà** —
et la descente n'a jamais fait l'objet du retour d'usage ; les phases, si. Ajouter
du bruit corrélé pour calmer un mouvement de la taille de celui qui plaît serait
une complexité payée pour rien.

### 4.3 Ce que l'OU aurait acheté, et pourquoi le cadran le donne déjà

Un OU sur les deux champs injectés (`z_i = ρ·z_{i−1} + sqrt(1−ρ²)·ξ_i`) laisserait
la loi marginale de `x_t` intacte tout en réduisant `|Δ|` d'un facteur
`sqrt(2(1−ρ))`. C'est un vrai levier — sur le pane **x_t**, qui à `t* = 64` churne
à 20,4 niveaux par frame. Mais **le cadran `t*` couvre déjà cette plage**, sans
état supplémentaire à porter d'une frame à l'autre, sans sémantique à réinventer
pour le re-seed, le changement de régime et la marche de `t*` (§5) :

| | `t*`=8 | `t*`=16 | `t*`=32 | `t*`=64 | `t*`=128 |
|---|---|---|---|---|---|
| `\|Δx_t\|` médiane | 7,97 | **10,69** | 14,67 | 20,41 | 28,53 |

`t* = 16` rend exactement le churn d'une descente d'errance (10,49). Baisser `t*`
d'un cran fait donc, en une touche, ce qu'un OU à `ρ ≈ 0,6` aurait fait en
plusieurs centaines de lignes d'état partagé.

**Décision : NO-GO.** Critères de réouverture, explicites : une corrélation
d'incréments qui passerait sous **−0,2**, ou un mouvement conservé à 25 frames
sous **0,2** (les deux se lisent d'un `flux_analysis.py stats`), ou un retour
d'usage qui décrirait un tremblement **à `t*` bas** — là où le cadran n'a plus de
marge et où l'OU serait le seul levier restant.

---

## 5. Le cadran `t*`, chiffré

Cinq runs de 1 156 actions, mêmes poids, même graine.

| | `t*`=8 | `t*`=16 | `t*`=32 | `t*`=64 | `t*`=128 |
|---|---|---|---|---|---|
| `\|Δx_t\|` médiane | 7,97 | 10,69 | 14,67 | 20,41 | 28,53 |
| `\|Δx_t\|` max / médiane | **1,08** | **1,07** | **1,07** | **1,10** | **1,07** |
| `\|Δx̂₀\|` médiane | 2,64 | 2,61 | 3,22 | 3,59 | 6,63 |
| corr des deux panes x_t↔x̂₀ | **+0,914** | +0,696 | +0,587 | +0,360 | +0,282 |
| corr x̂₀ à 300 frames | +0,587 | +0,292 | +0,120 | +0,599 | +0,010 |

Trois choses :

1. **L'absence de saut ne dépend pas du réglage** : `max / médiane` reste entre
   1,07 et 1,10 sur tout le cadran. Le régime n'a pas un point de fonctionnement
   sage et des extrêmes qui craquent.
2. **Le churn suit la loi du niveau.** Rapporté à `t* = 8`, le `|Δx_t|` mesuré
   suit `sqrt(β_{t*})` à **4 % près sur un facteur 3,7** (1,341 contre 1,365 ;
   1,841 / 1,894 ; 2,562 / 2,652 ; 3,580 / 3,732). Ce n'est pas un paramètre
   libre, c'est le pas du schedule.
3. **Le cadran arbitre la lisibilité du pane gauche.** La fenêtre live montre
   `x_t | x̂₀` ; à `t* = 128` le pane gauche est du bruit (corr +0,28), à `t* = 8`
   c'est l'image elle-même avec du grain (+0,91). Voir
   `flux_samples/flux_t16_consecutives_xt.png` contre
   `flux_samples/flux_t64_consecutives_xt.png`.

**Réglage recommandé pour l'usage : `t*` entre 16 et 32** — churn au niveau d'une
descente d'errance, pane gauche encore lisible, dérive vive.

---

## 6. Le TUI réel

Fenêtre tmux dédiée (`batLab-flux-tui`), modèle chargé **par l'interface**
(`Home` → `Load Saved Model` → `Greyscale_Diffusion_L` → `night_run.ckpt` →
`Perpetual`), graine 7, `t_r = 64`, tempo réglé à 2 pas/s pour lire la marche
frame par frame.

| Vérification | Observé |
|---|---|
| `[m]` en pleine **descente d'errance** → respiration → flux | `respiration descente t=125` → `flux descente t*=64 t=122, 121, 120, …` — l'approche est une **descente ordinaire**, un cran par frame |
| Arrivée à `t*` | `flux flux t*=64 t=64 frames=175, 187, …` — s'arrête **sur** 64, jamais un cran plus bas |
| `[↑]` : `t*` 64 → 72 **en cours de run** | `remontée t=66, 67, 68, 69, 70, 71, 72` puis `flux t*=72 t=72` — 8 crans marchés, aucun saut |
| `[↓][↓]` : `t*` 72 → 56 | `descente t=71, 70, …, 57, 56` puis `flux t*=56 t=56` — 16 crans marchés |
| `[m]` : flux → **errance** | `flux t*=56 t=56` → `errance descente t_r=55` — reprend là où le latent est |
| `[m][m]` en pleine **remontée** d'errance | `remontée t=47, 48, …, 56` puis `flux t*=56` — l'approche par le haut **continue la montée**, elle ne la retourne pas |
| Étiquettes | `t_r` ↔ `t*` et `cycle` ↔ `frames` selon le régime, panneau **et** pied de page (`perpetual · flux · flux · t*=56 · frames 576`) |
| `[espace]` | `frames` figé sur 1022 pendant 2 s, puis reprise |
| `[v]` | accepté sans erreur ni trace (fenêtre non vérifiée visuellement — voir §7) |
| `[q]` | **`EXIT=0`**, terminal rendu propre |

**stderr sur tout le run : 0 octet.**

Le point qui compte : les deux approches — par le bas comme par le haut — sont
des **phases ordinaires** du même moteur, et le compteur `frames` ne bouge pas
pendant qu'elles marchent (423 → 423 sur toute la montée de 64 à 72). Rien n'a
été ajouté pour « adoucir » un déplacement de `t*` : `settle_phase` réutilise la
descente et la montée existantes, et c'est pourquoi il ne peut pas y avoir de
saut à cet endroit.

Une observation que j'ai classée cosmétique et qui ne l'était pas : à la fin
d'une approche, le panneau affichait `descente` pendant **une** frame de trop
avant de basculer sur `flux`. J'avais raison sur la mécanique — le panneau lisait
`drift.phase()` avant que `settle_phase` ne réconcilie, et aucune action de
descente n'était émise — et tort sur la portée. Ce n'était pas un artefact
d'affichage : le **même** décalage écrivait le champ `phase` du dump, c'est-à-dire
la seule chose que lit toute analyse du flux. L'aveugle l'a sorti par la physique
en passe 2 ; c'est le §7.5, et c'est corrigé. Le tableau ci-dessus est ce que le
panneau affiche **après** correctif : la première frame du churn s'annonce `flux`,
dans les deux sens d'approche.

La leçon de process : « sans effet » était une conclusion, pas une mesure. Le
réflexe manquant tient en une question — *qui d'autre lit ce champ ?*

---

## 7. Le test à l'aveugle, et les deux bugs qu'il a sortis

Un agent de test **indépendant** a travaillé en parallèle sur `blind-test-flux`,
sans jamais ouvrir `bat_building/src/` ni `main/src/` : il écrit ses tests depuis
la spec et le contrat public, exécute le binaire, lit ses sorties. Son verdict de
passe 1 : `BLIND_TEST_FLUX.md`, **7 propriétés PASS sur 8**, une FAIL — le cadran
jamais parsé (§7.2).

Rejoué contre le binaire corrigé, il a rapporté un **P0 en passe 2** : l'octet de
phase du dump avait une frame de retard (§7.5). Que la deuxième passe rapporte
davantage que la première n'est pas un accident de méthode : la passe 1 a été
absorbée par le flag manquant, qui masquait tout ce qui se mesure *à `t*` choisi*.
Un dispositif d'aveugle se rejoue après correctif — sinon on ne récolte que le
premier défaut de la pile.

### 7.1 Les deux chaînes de mesure sont d'accord

Avant tout, la contre-épreuve que ce dispositif existe pour produire. Deux
lecteurs de dump écrits séparément, deux jeux de scripts, mêmes poids :

| | l'aveugle (3 000 actions) | ce rapport (1 408 frames) |
|---|---|---|
| `\|Δx_t\|` médiane | 0,16013 *(= **20,42** niveaux)* | **20,41** |
| `\|Δx_t\|` max / médiane | **1,10** | **1,10** |
| `\|Δx̂₀\|` max / médiane | 1,58 | 1,50 |
| std(x_t) sur le palier | 0,7367 | 0,7379 |
| corr x_t décalage 1 / 300 | +0,9627 / +0,0271 | +0,963 / +0,055 |
| corr des incréments | −0,0199 | −0,021 |

Trois décimales d'accord sur des mesures faites sans se lire. Et l'aveugle a
poussé là où je n'étais pas allé : **20 000 frames** (ratios inchangés à 1,098 et
1,662, dérive décile-à-décile −0,68 %), toutes les paires d'incréments du run
(≈ 4×10⁶, corrélation max +0,162 — un maximum d'échantillon gaussien, aucun champ
rejoué), isotropie du champ ajouté **y compris sur l'anti-diagonale** (|ρ| ≤
0,0013 : le piège d'`ANISOTROPY_HUNT.md` est bien absent), et une non-régression
**bit-à-bit** contre le binaire d'avant le flux (`12acad8`) — 8 cycles d'errance
et 8 de respiration strictement identiques.

### 7.2 Le FAIL : un cadran annoncé, jamais parsé

> Spec, ligne 18 : « **Amplitude par frame ~ √β(t\*)** […] Vérifiable en
> comparant deux valeurs de `t*`. »

**Bug réel, confirmé.** `--t-star` et `--t-r` figuraient au contrat public ; seul
`--depth` était parsé. Le §5 de ce rapport a mesuré la loi en `sqrt(β)` — avec
`--depth`, sans voir que le nom du contrat, lui, ne marchait pas. L'aveugle l'a
prouvé de la seule façon possible de l'extérieur, et c'est la bonne : passer
**un flag inventé sur place** et constater que les trois dumps — `--t-star 30`,
`--t-r 30`, `--flag-inexistant-de-controle 30` — ont le **même sha256** que le run
sans aucun flag.

Ce qui l'a rendu invisible pendant toute la campagne est un défaut de plus :
**un flag inconnu était ignoré en silence, code de retour 0**. « Ignoré » et
« pas implémenté » étaient indistinguables, et une faute de frappe passait pour
un réglage. C'est exactement le piège du flag muet que le `CLAUDE.md` du projet
demande de contrer par la bannière — et la bannière affichait `t_r=64` quoi qu'on
passe.

Corrigé en `be42c27`, vérifié **par la bannière** :

```
$ main --headless-perpetual Greyscale_Diffusion_L --regime flux --t-star 30 --actions 400 …
headless perpetual '…': regime=flux t*=30 bound=--actions 400 magnitude=1 seed=5
$ main … --flag-inexistant-de-controle 30
headless perpetual failed: unknown flag `--flag-inexistant-de-controle`. Known flags here: … See --help.   (exit 1)
```

Et P4 devient mesurable par le contrat public — trois runs, graine 5, 400
actions, sha256 désormais **distincts** :

| `--t-star` | palier `t` dans le dump | frames de flux | `\|Δx_t\|` médiane |
|---|---|---|---|
| 30 | 30 | 174 | 14,23 |
| 100 | 100 | 244 | 25,25 |
| 150 | 150 | 294 | 30,87 |

### 7.3 Les six défauts de contrat

| | Défaut signalé | Traitement |
|---|---|---|
| D1 | `--help` ne produit rien ; un flag inconnu est ignoré | `--help` écrit les trois entrées headless **et la disposition du dump** ; tout flag inconnu sort en erreur (code 1) en listant les flags connus |
| D2 | l'invocation de la spec omet le nom du modèle → `failed to load Models/--regime/config_file` | le nom du modèle est refusé s'il commence par `-`, avec un message qui montre la forme correcte. **La spec est à corriger** : le nom est positionnel, comme pour `--headless-train` |
| D3 | `--t-star` inexistant, `--t-r` non parsé (pour **tous** les régimes) | `dial_level` accepte les trois orthographes ; 2 tests, vérifiés par mutation |
| D4 | le dump n'est pas du « f32 brut » : en-tête de 20 o + 5 o par frame | la disposition exacte est dans `--help` et dans `FrameDump`. **La spec est à corriger** ; le format, lui, reste tel quel — l'aveugle a raison de dire qu'il mérite d'être documenté, pas supprimé |
| D5 | flux annonce un dossier de PNG qu'il ne crée jamais | la bannière dit `no PNG in flux (no cycle ever closes) — use --dump`. Et `--regime flux --frames N` **refuse** au lieu de tourner sans fin : `written` ne bougeant jamais en flux, la boucle ne s'arrêtait pas (bug non signalé, trouvé en traitant D5) |
| D6 | la sémantique `--actions`/`--frames` a changé sans être écrite | la bannière annonce la borne en vigueur (`bound=--actions 1600` / `bound=--frames 8 (cycles)`), et `--help` la décrit. Le compte final donne les deux nombres, ce qui lève l'ambiguïté du « N reverse steps » |

### 7.4 Les deux ambiguïtés de spec, arbitrées

- **A1 — « couloir stable sur toute la durée »** : tout run part du bruit pur et
  descend jusqu'à `t*` (191 frames ici). L'aveugle a tranché en faveur du régime
  permanent et l'a signalé plutôt que de l'arbitrer en silence — **c'est le bon
  arbitrage**, et le §1 de ce rapport écarte le même prologue pour la même
  raison. Ce que ça laisse ouvert, en revanche, est une vraie question de
  produit : *rien ne permet de démarrer un run directement à `t*`*. Dans le TUI
  ça n'existe pas — on y entre en flux depuis un run déjà en cours, et l'approche
  est marchée (§6). En headless, c'est une exigence à écrire si on la veut.
- **A2 — « images des cycles successifs différentes »** : elles corrèlent à 0,80–0,97.
  Seuil retenu par l'aveugle : « non identiques », plus la décroissance avec la
  distance de cycle (0,748 à *i+5*). **Arbitrage confirmé** : « décorrélées »
  contredirait le mot *errance*, et §3.2 le mesure de l'autre côté — l'errance
  garde +0,780 à 300 frames là où le flux tombe à +0,19/+0,60.

### 7.5 L'octet de phase : ma correction était fausse à moitié

En passe 1 l'aveugle décrivait l'octet de tête de chaque frame comme « toujours 0
observé ». J'ai corrigé : c'est le numéro de phase, il vaut 0 pendant la descente
d'ouverture et 2 sur le palier — « 192 octets à 0, puis 1 408 à 2 ».

**Ce 192 était le bug.** L'approche descend de `t = 255` à `t* = 64` en 191
actions ; la 192ᵉ frame est déjà un churn, et elle sortait étiquetée `0 descent`.
J'avais le chiffre sous les yeux, je n'ai pas vérifié où la spec place la
frontière — et j'avais même relevé le symptôme jumeau au §6 (le panneau TUI qui
affiche `descente` une frame de trop) sans faire le lien. L'aveugle, lui, ne
pouvait pas lire le compte : il l'a attrapé **par la physique**, en mesurant que
la frame étiquetée `descent` bougeait de 40 % de plus qu'un pas inverse au même
niveau — reproductible sur les 8 valeurs du cadran.

Diagnostic : **pur étiquetage, aucun décalage de séquencement.** Les preuves :

| frame | phase inscrite | `t` | `\|Δx_t\|` |
|---|---|---|---|
| 189 | 0 | 65 | 14,67 |
| 190 | 0 | 64 | 14,63 |
| **191** | **0** ← faux | 64 | **20,78** ← amplitude de churn |
| 192 | 2 | 64 | 19,56 |

La frame 191 *est* un churn (reverse + forward à `t*`), elle porte le nom de la
phase que le run venait de quitter. Cause : `main.rs` lisait `drift.phase()`
**avant** `drift.step()`, alors que `settle_phase` réconcilie la phase *à
l'intérieur* de `step`. Dans les régimes à cycles la lecture d'avant est exacte
(`settle_phase` n'y fait rien), d'où un défaut invisible partout ailleurs.

Corrigé en introduisant `DriftAction::phase()` — le nom de ce qui a été fait, et
non de ce qui était prévu — par où passent désormais le tag du dump **et** le
panneau TUI. Vérification sur le dump de référence de 13 Mo : **un seul octet
change**, celui de la frame 191, `0 → 2`. Errance et respiration sont inchangées
**au bit près** (0 octet de différence sur 1 600 frames chacune), ce qui confirme
que le défaut vivait au seul endroit où `settle_phase` a du travail.

Pourquoi 129 tests verts ne l'ont pas vu : **toute la batterie lisait déjà la
phase depuis l'action** (`walk_tagged`), c'est-à-dire la bonne notion, pendant
que le dump en enregistrait une autre. Rien qui s'accorde avec soi-même ne peut
voir cet écart — c'est très exactement le désaccord que le dispositif d'aveugle
existe pour produire (`CLAUDE.md`). `walk_tagged` route maintenant par
`DriftAction::phase()`, donc une mutation de cet accesseur fait tomber les tests
de flux existants **en plus** des deux nouveaux.

Deux nouveaux, parce que corriger la ligne fautive ne referme rien : n'importe
quel appelant pouvait la réécrire. Ce qui est fait à la place :

1. **La faute est rendue inexprimable.** `FrameDump::record` ne prend plus une
   phase mais l'**action**, et calcule le tag lui-même. Il n'y a plus de
   paramètre où glisser une phase périmée ; l'ancien bug ne se retape pas, il ne
   compile pas.
2. **Un test qui lit les octets** (`the_dump_files_the_first_churn_frame_as_flux`,
   dans `main`) : il déroule une vraie dérive, écrit un vrai dump, le **relit par
   un parseur écrit à part** — jamais par le code d'écriture — compte les phases
   et exige la frontière à `T − 1 − t*` exactement. C'est le test qui manquait :
   le seul de la maison qui juge le dump sur son contenu et non sur l'intention
   de l'appelant.
3. Côté moteur, `a_frame_is_named_after_the_action_it_performed_not_the_phase_it_left`
   épingle les deux notions côte à côte, dans les deux sens d'approche.

Contre-épreuve de mutation : décaler d'un cran la frontière de `settle_phase`
(`t <= ceiling` → `t < ceiling`) fait tomber le test du dump avec le compte exact
en clair — 28 frames de descente au lieu de 23 — et pas seulement un `assert`
opaque. Le message d'échec dit ce qu'il faut regarder.

### 7.6 Ce que ça change pour sa suite

**Le contrat public a bougé** : `blind_tests/run.sh` doit être rejoué contre le
binaire courant. Ce qui peut casser chez lui : un flag inconnu fait maintenant
**sortir en erreur** (ses cas de contrôle par flag inventé attendent un code 0),
`--regime flux --frames N` sans `--actions` est refusé, et la bannière a changé de
format. **Aucune dynamique n'a changé**, vérifié plutôt qu'affirmé : le dump de
`--depth 64` d'avant le correctif et celui de `--t-star 64` d'après ont le même
sha256 (`74dbfe55d8b8…`, 1 600 actions, graine 7).

Le correctif de sa passe 2 déplace en revanche **un octet par approche** dans les
dumps de flux : le sha256 ci-dessus n'est plus celui du binaire courant, et une
mesure qui découpe le palier au champ `phase` gagne une frame. C'est le sens de
sa trouvaille, pas un effet de bord — mais un harnais qui aurait figé ce sha256
le verra bouger, et il doit le voir bouger.

---

## 8. Pièges et limites

- **Ne pas mesurer un régime sans phase sur des PNG.** Le `|Δ|` utile est de
  l'ordre de 3 niveaux de gris ; le pas de quantification en est un. `--dump`
  existe pour ça et `tools/flux_analysis.py` ne lit rien d'autre.
- **Écarter le prologue.** Les `T − 1 − t*` premières frames sont la descente
  d'ouverture — 191 à `t* = 64`, jusqu'à 255 quand `t*` est au plancher —,
  commune aux trois régimes et porteuse des plus gros `|Δ|` du run. L'outil le
  fait ; une mesure faite à la main qui l'oublierait conclurait que tous les
  régimes sautent. Se fier au champ `phase` plutôt qu'à un compte : il est
  maintenant exact à la frame près (§7.5), ce qu'il n'était pas.
- **La corrélation entre frames n'a pas 0 pour plancher.** Deux runs sans aucun
  lien corrèlent à −0,03 (x_t) et −0,18 (x̂₀) parce qu'ils sortent du même
  modèle : `--floor` mesure ce zéro-là.
- **`std(x̂₀)` n'est pas un critère de stationnarité.** Il suit le contenu
  (±25 % d'un cinquième à l'autre, sans tendance). La stationnarité se lit sur
  `std(x_t)`, qui est la quantité que la loi du niveau contraint.
- **Capturer l'écran ne mesure rien** (déjà documenté, `CLIMB_COHERENCE.md` §6) :
  `screencapture` étrangle le run wgpu. C'est aussi pourquoi l'ouverture de la
  fenêtre `[v]` n'est pas cochée visuellement ici — la touche est acceptée, le
  run continue, stderr reste vide, et les frames mesurées sont déjà, à
  l'identique, ce que cette fenêtre reçoit.
- **Un flag qu'on ne parse pas doit être une erreur.** Tant qu'un flag inconnu
  sortait en silence avec le code 0, aucune expérience ne pouvait distinguer
  « ignoré » de « pas implémenté » — c'est ce qui a couvert `--t-star` pendant
  toute la campagne (§7.2). Corollaire de terrain : **vérifier la bannière**,
  qui doit réémettre la valeur *relue depuis l'objet construit*, pas celle qu'on
  croit avoir parsée.
- **Le formulaire refuse `t* < 4`** alors que le régime accepte `t* = 1` :
  `finish_perpetual_params` valide contre `MIN_RENOISE_DEPTH` sans regarder le
  régime. En cours de run, `[↓]` descend bien jusqu'à 1. Écart connu, sans
  conséquence sur la mesure ; il se refermera en faisant lire au formulaire le
  `min_depth()` du régime choisi.
- **Hygiène** : le TUI réécrit `Models/<modèle>/config_file` en fin de run
  (il y persiste le régime, `t*` et le tempo). `git checkout Models/` après toute
  session TUI — comme après `cargo test`.

---

## 9. Planches

| Fichier | Ce qu'il montre |
|---|---|
| `flux_samples/flux_t64_consecutives.png` | 12 frames **consécutives** de flux (x̂₀) — la continuité |
| `flux_samples/errance_rebroussement.png` | 12 frames d'errance **autour d'un rebroussement** (x̂₀) — 6 vignettes identiques puis la coupe |
| `flux_samples/flux_t64_derive_1sur25.png` | 12 frames **espacées de 25** (x̂₀, 275 frames couvertes) — la dérive longue, sans retour ni saut |
| `flux_samples/flux_t64_consecutives_xt.png` | le pane x_t à `t*` = 64 — du bruit (corr des panes +0,36) |
| `flux_samples/flux_t16_consecutives_xt.png` | le pane x_t à `t*` = 16 — l'image avec du grain (+0,69) |
| `flux_samples/flux_t16_consecutives.png` | x̂₀ à `t*` = 16, pour comparer à la planche `t*` = 64 |

---

## 10. Ce qui a changé de la surface observable

Les mesures (§2 à §6) n'ont rien touché : `tools/flux_analysis.py` est un
lecteur. Le traitement du verdict de l'aveugle (§7), lui, **modifie le contrat
public** — voici la liste complète, pour qui teste ce binaire de l'extérieur :

| | Avant | Après (`be42c27`) |
|---|---|---|
| `--t-star` / `--t-r` | ignorés en silence | parsés, alias de `--depth` |
| flag inconnu | ignoré, code 0 | erreur, **code 1**, liste des flags connus |
| `--help` / `-h` | sortie vide, code 0 | usage des trois entrées headless + disposition du dump |
| bannière | `t_r=64` figé | `t*=30` / `t_r=64` selon le régime, relu après clamp, + borne en vigueur |
| `--regime flux --frames N` seul | boucle sans fin | erreur, code 1 |
| ligne finale | `N reverse steps` | `N actions, M of them reverse steps (model calls)` |
| flux : ligne `frames →` | un dossier jamais créé | `no PNG in flux (no cycle ever closes) — use --dump` |

Et le correctif de la passe 2 (§7.5) en change une de plus — la seule qui touche
au **contenu** d'un dump depuis le début de la mission :

| | Avant | Après (ce commit) |
|---|---|---|
| dump, octet de phase à l'arrivée sur `t*` | la 1ʳᵉ frame du churn portait `0` (descente) | elle porte `2` (flux), comme les suivantes |
| TUI, libellé de phase | une frame `descente` de trop en fin d'approche | la 1ʳᵉ frame du churn s'annonce `flux` |

La frontière est désormais celle de la spec : la frame **produite par** le churn
s'appelle flux, la première comprise. Sur un run `t* = 64` de 1 600 actions, la
descente d'ouverture fait 191 frames étiquetées `0` puis 1 409 à `2` — c'était
192 / 1 408. Un lecteur de dump qui coupait son prologue au premier `2` gardait
une frame de churn de moins ; qui filtrait `phase == 2` en jetait une.

**Ce qui n'a pas changé** : la dynamique — pas un f32 de latent ne bouge, seul
l'octet de nom change, et sur une frame par approche — la **disposition** du dump,
et le reste du TUI. Vérifié au sha256 et à l'octet plutôt qu'affirmé (§7.5, §7.6) :
errance et respiration sortent **identiques au bit près**.
