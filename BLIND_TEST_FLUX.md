# Test à l'aveugle du régime « flux » — verdict

Agent de test indépendant. **Aucun fichier de `bat_building/src/` ni `main/src/` n'a été
ouvert**, ni leurs diffs, ni leur historique, ni le banc `tools/flux_analysis.py` livré par
l'implémenteur. Tout ce qui suit provient du binaire, de sa `stdout`, de ses dumps `.f32`
et de ses PNG. La seule chose faite de plus qu'exécuter le binaire courant : **builder et
exécuter le binaire d'un commit antérieur** (`12acad8`, parent de `feat(perpetual): le
regime flux`) pour comparer ses sorties — build et exécution, jamais lecture.

Suite exécutable : `blind_tests/run.sh` (`BLIND_BASELINE=1` pour la comparaison avec le
binaire pré-flux). PASS/FAIL par propriété, code de retour non nul si FAIL.

## Passe 3 — contre-vérification après `419a96e` + `ccb5a15`

**Verdict : 9/9 PASS, 0 FAIL.** Le défaut P0 de la passe 2 est corrigé, et corrigé
exactement — pas approximativement.

| # | Propriété | Passe 1 | Passe 2 | **Passe 3** |
|---|---|---|---|---|
| P0 | format du dump / sanité | PASS | **FAIL** | **PASS** |
| P1 | stationnarité du niveau de bruit (ligne 15) | PASS | PASS | **PASS** |
| P2 | aucun saut image-à-image (ligne 16) | PASS | PASS | **PASS** |
| P3 | continuité locale + dérive longue (ligne 17) | PASS | PASS | **PASS** |
| P4 | amplitude par frame ~ √β(t\*) (ligne 18) | **FAIL** | PASS | **PASS** |
| P5 | graines : reproductibilité, divergence, décorrélation (ligne 19) | PASS | PASS | **PASS** |
| P6 | isotropie (ligne 20) | PASS | PASS | **PASS** |
| P7 | non-régression errance / respiration (ligne 21) | PASS | PASS | **PASS** |
| P8 | bornes et diagnostics du CLI (contrat `--help`) | *OBS* | PASS | **PASS** |

### La spec a bougé, et elle a bougé dans le bon sens

`--help` publie une phrase qui **n'existait pas** en passe 2 :

> « The phase names what the frame DID, so in flux the opening approach is
> **T-1-t\* frames of 0** and every frame from the first churn on is 2 — cut a
> prologue on that byte rather than on a count. »

C'est exactement l'arbitrage que la passe 2 demandait, et il tranche dans le sens de la
convention de l'errance (la frame qui *commence* une phase porte l'étiquette de cette
phase). L'implémenteur n'a pas amendé la spec pour coller au code : il a corrigé le code
**et** publié la règle qui le rend vérifiable. J'ai donc durci P0 en conséquence — c'est
la seule chose que j'ai changée dans la suite, et je la cite plutôt que de la supposer.

**Je n'ai supposé nulle part la valeur de `T`** : le contrat l'emploie sans le définir
(*défaut de spec mineur*, cf. D7 ci-dessous). Je teste la **loi** — longueur du prologue
affine en t\*, de pente exactement −1 — puis je recoupe son ordonnée à l'origine avec le
haut de chaîne lu **dans le dump lui-même** (t de la frame 0, +1). Les deux tombent sur
255, indépendamment :

```
longueur du prologue : t*=8→247  16→239  32→223  64→191  96→159  160→95  224→31  250→5
n0 + t* = {255}  (constante ⇔ pente −1 exacte, sur 9 valeurs de t*)
haut de chaîne lu dans le dump = {255}     → concordant
```

### P0 — la frontière tombe où le contrat l'annonce

Le test que la passe 2 avait écrit **échouait** ; le même test, seuils inchangés, passe
désormais. Comme en passe 2, je ne juge pas sur l'étiquette mais sur la **dynamique** :
au niveau t\*, pas inverse et churn diffèrent de ~40 %, ce qui les sépare sans ambiguïté.

```
frames « 0 descent » ayant l'amplitude d'un churn :
  t*=8→0  16→0  32→0  64→0  96→0  160→0  224→0  250→0   (+ run principal t*=64→0)
  au niveau t*=64 : « descent » ≈0,1187 (pas inverse)  vs  churn = 0,1601
```

**0 frame mal étiquetée sur 9 valeurs du cadran** — la mission en demandait 3. Le bloc de
`0` est en outre un **préfixe contigu** (première frame de phase 2 = nombre de frames de
phase 0) et aucun `1 climb` n'apparaît jamais en flux.

### Ce que j'ai vérifié que l'implémenteur affirmait sans le prouver

L'ordre de mission annonçait « le sha256 de référence a changé d'un octet par approche ».
Ma suite ne **pinne** aucun sha256 de dump (elle n'en calcule que pour comparer deux runs
de même graine entre eux, P5) : rien à mettre à jour de ce côté. Mais l'affirmation elle-
même est un auto-contrôle de l'implémenteur, et elle porte un enjeu réel — **un correctif
d'étiquette qui déplacerait aussi la trajectoire serait une régression silencieuse**, et
P1–P7 ne la verraient pas nécessairement. Je l'ai donc vérifiée en construisant et en
exécutant le binaire de la passe 2 (`be42c27`, build + exécution, **jamais lecture**) et
en comparant les dumps **octet à octet**, à graine et paramètres identiques :

| t\* | octets différents | position | valeur | frame attendue (255−t\*) |
|---|---|---|---|---|
| 64 | **1** / 24 591 020 | frame 191, champ `phase` | `0` → `2` | 191 ✓ |
| 32 | **1** / 12 295 520 | frame 223, champ `phase` | `0` → `2` | 223 ✓ |
| 224 | **1** / 12 295 520 | frame 31, champ `phase` | `0` → `2` | 31 ✓ |

Exactement un octet, toujours le champ `phase`, toujours à la frame que la passe 2 avait
désignée, et toujours au rang que le contrat prédit. **Tous les pixels et tous les `t`
sont bit-à-bit identiques** : le correctif est prouvé *label-only*, il n'a pas effleuré la
dynamique. C'est aussi ce que disent les chiffres de P1–P6, inchangés à l'affichage près
(std plateau 0,7367 ; max/médiane 1,10 ; corr(k,k+1) 0,9627 ; c\* = 1,980 ; rapport
d'isotropie 1,0003).

La suite a également été rejouée avec `BLIND_BASELINE=1`, qui rétablit la non-régression
forte de P7 : contre le binaire **pré-flux** (`12acad8`), à graine identique, **8/8 PNG
strictement identiques en errance et 8/8 en respiration**. Les deux régimes préexistants
n'ont pas bougé d'un pixel depuis avant l'introduction du flux.

### Défauts de spec restant ouverts

- **D7 (neuf, mineur)** — `--help` énonce « T-1-t\* frames » sans définir `T` nulle part
  dans le contrat public. Un lecteur qui veut *prédire* la longueur du prologue (plutôt
  que la constater) doit deviner que T = 256. Mesuré : T−1 = 255. À écrire dans `--help`.
- **A1, A2** (passe 1, inchangés) — la descente initiale n'est couverte par aucune ligne
  de la spec, et « images des cycles successifs différentes » n'a pas de seuil. Ces deux
  points relèvent de l'auteur de la spec, pas de l'implémenteur. *Note* : le contrat
  reconnaît désormais explicitement le prologue et donne le moyen de le découper
  (« cut a prologue on that byte »), ce qui **outille** A1 sans le trancher — la question
  « le flux doit-il pouvoir démarrer *à* t\* ? » reste ouverte.
- **D2 (réserve de la passe 2, toujours ouverte)** — `MISSION_BLIND_TEST.md` ligne 9 porte
  encore la forme d'invocation fautive, sans le modèle positionnel.

---

## Passe 2 — contre-vérification après `be42c27`

État à l'issue de la première passe : 7/8 PASS, **P4 FAIL** (le cadran t\* était annoncé au
contrat et jamais parsé), 6 défauts de contrat consignés.

| # | Propriété | Passe 1 | **Passe 2** |
|---|---|---|---|
| P0 | format du dump / sanité | PASS | **FAIL** (nouveau champ `phase`, off-by-one — §P0) |
| P1 | stationnarité du niveau de bruit (ligne 15) | PASS | **PASS** |
| P2 | aucun saut image-à-image (ligne 16) | PASS | **PASS** |
| P3 | continuité locale + dérive longue (ligne 17) | PASS | **PASS** |
| P4 | amplitude par frame ~ √β(t\*) (ligne 18) | **FAIL** | **PASS** |
| P5 | graines : reproductibilité, divergence, décorrélation (ligne 19) | PASS | **PASS** |
| P6 | isotropie (ligne 20) | PASS | **PASS** |
| P7 | non-régression errance / respiration (ligne 21) | PASS | **PASS** |
| P8 | bornes et diagnostics du CLI (contrat `--help`) | *OBS* | **PASS** |

**8/9 PASS.** Le correctif tient : le cadran est réglable, la loi d'amplitude est vérifiée
quantitativement, les six défauts de contrat sont clos, et errance/respiration restent
bit-à-bit identiques au binaire pré-flux. Le seul FAIL est **neuf** : il porte sur un champ
qui n'existait pas en passe 1.

### Ce que j'ai changé dans les tests, et pourquoi

Deux tests seulement ont bougé, l'un et l'autre parce que **la spec du contrat a été
complétée** — jamais pour faire passer quoi que ce soit :

- **P0** vérifie désormais le champ `phase`, que `--help` documente maintenant :
  « `u8 phase (0 descent, 1 climb, 2 flux)` ». En passe 1 cet octet était un « toujours 0
  observé » sans sémantique publiée : il n'y avait rien à vérifier. C'est ce nouveau test
  qui trouve le seul défaut restant.
- **P4** teste la loi au lieu de constater l'absence de cadran, `--t-star` étant devenu
  réel (`--help` : « `--t-r/--t-star` the level dial, one flag under three spellings […]
  the level held in flux »). Aucun seuil des propriétés P1–P3, P5–P7 n'a été touché : les
  chiffres de la passe 2 sont directement comparables à ceux de la passe 1.

`blind_tests/dumpio.py` est resté **inchangé quant au décodage** : la structure publiée par
`--help` confirme exactement celle que j'avais reconstituée à l'aveugle par dichotomie sur
la taille du fichier. Seuls le nom du premier octet (`tag` → `phase`) et la tolérance au
dernier enregistrement partiel ont suivi le texte du contrat.

---

## P4 — amplitude par frame ~ √β(t\*) : **FAIL → PASS**

> **Spec, ligne 18** : « les changements par frame croissent avec t\* (t\* haut = plus turbulent). Vérifiable en comparant deux valeurs de t\*. »

Le cadran répond, sous ses trois orthographes (`--t-r`, `--t-star`, `--depth` → tous
`t*=137` dans la bannière), et le plateau atteint bien la valeur demandée à chaque fois.
La spec demandait deux valeurs de t\* ; j'en ai pris **huit** (8 → 250) et je suis allé
au-delà de la lettre, en trois marches.

**(1) Le sens de variation** — la lettre de la spec. Amplitude strictement croissante sur
les huit barreaux, rapport bout à bout **4,99×** :

| t\* | 8 | 16 | 32 | 64 | 96 | 160 | 224 | 250 |
|---|---|---|---|---|---|---|---|---|
| \|Δ\| médian par frame | 0,0626 | 0,0838 | 0,1150 | 0,1602 | 0,1947 | 0,2501 | 0,2957 | 0,3125 |

**(2) La forme de la loi.** Pour un schedule DDPM linéaire, β est affine en t, donc une
amplitude en √β doit avoir un **carré affine en t\***. Régression :
`rms(Δ)² = 1,290e-3 + 6,076e-4·t*`, **R² = 0,999994**, écart maximal sur l'amplitude
**0,14 %** sur les huit valeurs. La loi n'est pas seulement croissante : elle a exactement
la forme annoncée.

**(3) Un oracle croisé, indépendant de l'implémentation.** R² élevé sur une seule
observable prouve peu — n'importe quelle courbe lisse s'ajuste. J'ai donc confronté deux
observables **indépendantes** via l'identité DDPM : l'amplitude par frame et le **niveau de
bruit tenu**. En posant β̂(t) = rms(Δ)²/c déduit de la *seule* amplitude, on prédit
ᾱ_t = exp(−Σβ̂) puis var(x_t) = ᾱ·σ₀² + (1−ᾱ) — un unique paramètre libre, σ₀, calé au plus
bas t\*. Résultat : le niveau stationnaire mesuré est reproduit à **2,9 % RMS**.

Le test ne porte pas sur ce résidu (dont l'ampleur est celle de mon approximation : σ₀ y
est traité comme constant alors que std(x̂₀) varie d'un facteur 3 sur la plage). Il porte
sur le **coefficient c**, que les données déterminent seules et dont la valeur théorique
est **2** — un bruitage avant plus un débruitage arrière. Ajusté librement sur une grille :

**c\* = 1,980 pour une théorie à 2,000 — 1 % d'écart.**

Deux mesures qui n'ont aucune raison de s'accorder si la mécanique était fausse s'accordent
au pour-cent près sur un coefficient non ajusté. La loi en √β(t\*) est vérifiée
quantitativement, pas seulement dans son sens de variation.

---

## P0 — format du dump : **PASS → FAIL** (défaut neuf, une frame par run)

> **Contrat publié par `--help`** : « then per frame: `u8 phase (0 descent, 1 climb, 2 flux)` `u32 t (the level this frame landed on)` »

Tout le reste du format est conforme : magic `BATFLUX1`, dimensions 32×32×1, nombre de
frames = `--actions`, x̂₀ borné dans [−1, 1], phase 2 jamais quittée une fois prise, phase 2
absente d'errance et de respiration, et cohérence parfaite étiquette↔sens de t en errance
(1620/1620 frames « descent » à Δt ≤ 0, 1379/1379 « climb » à Δt ≥ 0).

**Le défaut** : à chaque run de flux, **la première frame produite par le churn est
étiquetée `0 descent` au lieu de `2 flux`**.

Je ne l'ai pas déduit de l'étiquette mais de la **dynamique** : au niveau t\*, un pas
inverse et un pas de churn ont des amplitudes qui diffèrent de ~40 %, ce qui les sépare
sans ambiguïté. À t\* = 64 :

```
frame 189  t=65  phase 0   |Δ| = 0,1151   ← pas inverse
frame 190  t=64  phase 0   |Δ| = 0,1147   ← pas inverse, celui qui atterrit sur t*
frame 191  t=64  phase 0   |Δ| = 0,1630   ← CHURN, mais étiquetée « descent »
frame 192  t=64  phase 2   |Δ| = 0,1534   ← churn
```

Référence au même niveau : pas inverse ≈ 0,1187, churn = 0,1601. La frame 191 est à 0,163.
**Reproductible sur tout le cadran** : exactement 1 frame mal étiquetée par run, pour
t\* ∈ {8, 16, 32, 64, 96, 160, 224, 250}.

La convention de l'errance tranche le sens à donner à l'étiquette : au tournant, la frame
qui **commence** la remontée y porte déjà `1 climb` (frame 255 : t=0, phase 0 ; frame 256 :
t=0, phase 1). Par la même convention, la frame qui commence le churn devrait porter `2`.

Impact réel : faible — une frame sur des milliers, et le t dumpé reste juste. Mais un
lecteur qui découpe le run sur le champ `phase` (c'est à cela qu'il sert) attribue une
frame de churn à la descente, et une mesure de « l'amplitude de la descente » qui inclut
cette frame est biaisée. C'est signalé, pas arbitré : ou bien l'implémentation étiquette
`2` dès la première frame de churn, ou bien `--help` précise que la frame de transition
reste comptée dans la phase sortante — et alors errance devrait suivre la même règle.

---

## Les six défauts de contrat de la passe 1 : tous clos

| | Défaut (passe 1) | Statut passe 2 | Preuve |
|---|---|---|---|
| **D1** | `--help` vide (0 octet, code 0) ; flag inconnu **silencieusement ignoré** | **CORRIGÉ** | `--help` publie 2 062 octets : flags perpetual, sémantique `--actions`/`--frames`, format du dump. Un flag inconnu sort en **code 1** avec ``unknown flag `--…`. Known flags here: …`` |
| **D2** | modèle positionnel absent de l'invocation de la spec | **CORRIGÉ au contrat** | `--help` : `--headless-perpetual <model> [--checkpoint <path>] …`. *Réserve* : `MISSION_BLIND_TEST.md` ligne 9 porte toujours la forme fautive, sans le modèle — la spec de mission reste à corriger |
| **D3** | `--t-star` inexistant, `--t-r` non parsé pour aucun régime | **CORRIGÉ** | un cadran, trois orthographes, toutes vérifiées : `--t-r 137`, `--t-star 137`, `--depth 137` → bannière `t*=137` ; plateau atteint = valeur demandée sur 8 valeurs (P4) |
| **D4** | format du dump différent de celui décrit (en-tête + préfixe de frame non documentés) | **CORRIGÉ** | le contrat publié par `--help` décrit exactement la structure reconstituée à l'aveugle en passe 1 : `"BATFLUX1"` + 3 u32, puis par frame `u8 phase` + `u32 t` + 2 images. Le compte de frames est explicitement « inferred from the file size », ce qui autorise un run tué en cours d'écriture — mon lecteur le tolère désormais. *Nouveau défaut sur le seul champ neuf, cf. §P0* |
| **D5** | le flux annonçait un dossier de PNG qu'il ne créait jamais | **CORRIGÉ** | `--regime flux --frames 8` sort en **code 1** : « `--regime flux never closes a cycle, so --frames cannot bound it: pass --actions N` ». Plus aucune annonce mensongère ; 0 PNG en flux, 5 en errance, 11 en respiration |
| **D6** | sémantique de `--actions`/`--frames` changée pour errance et respiration, non documentée | **CORRIGÉ** | `--help` documente les deux bornes et leur priorité ; vérifié : `--actions 700 --frames 2` ensemble → **700 frames** dumpées, `--actions` l'emporte comme annoncé |

Les deux ambiguïtés de spec relevées en passe 1 restent ouvertes — elles relèvent de
l'auteur de la spec, pas de l'implémenteur :

- **A1** — la **descente initiale** (190 frames à t\* = 64, std(x_t) de 1,03 à 0,74) n'est
  couverte ni par « std(x_t) reste dans un couloir stable **sur toute la durée** »
  (ligne 15) ni par aucune autre ligne. J'ai tranché en faveur du régime permanent et
  j'exclus explicitement ces frames de toutes mes mesures de flux. Si le flux doit pouvoir
  **démarrer** à t\* (pour être enchaîné après un autre régime), c'est une exigence à
  écrire ; elle n'est pas satisfaite aujourd'hui. La longueur de la descente est 254 − t\*,
  donc réglable indirectement par le cadran.
- **A2** — « images des cycles successifs **différentes** » (ligne 21) n'a pas de seuil.
  Mesuré : corr 0,803 à 0,968 entre cycles voisins, 0,748 à cinq cycles de distance.
  Différentes mais fortement corrélées ; j'ai retenu « non identiques » (corr ≤ 0,99).

---

## Les propriétés inchangées, revérifiées sur le binaire corrigé

Chiffres de la passe 2, à comparer à ceux de la passe 1 (identiques à l'affichage près :
le correctif n'a rien déplacé).

**P1 — stationnarité.** Plateau de 2 810 frames à t = 64 : std(x_t) moyenne **0,7367**,
couloir **[−8,9 %, +8,9 %]**, dérive premier↔dernier décile **+1,71 %**. Sur 20 000 frames
(passe 1) : dérive **−0,68 %**, aucune fuite vers le net ni vers le bruit.

**P2 — aucun saut.** flux x_t : **max/médiane = 1,10** ; x̂₀ : **1,58**. Contre-épreuve
demandée par la spec : errance **5,14×** et **17,37×** sur la même mesure. Le contraste
attendu est là.

**P3 — continuité + dérive.** x_t : corr(k,k+1) = **0,9627** → corr(k,k+300) = **0,0271**.
x̂₀ : **0,9891** → **0,3230**. Décroissance régulière (lag 5 : 0,829 ; lag 30 : 0,345 ;
lag 100 : 0,064). Ni vibration sur place, ni figement.

**P5 — graines.** Deux runs de même graine : **sha256 identiques**. Graines différentes :
corr moyenne **+0,0012**. Incréments : corr(Δ_k, Δ_{k+1}) = **−0,0199**, et sur ~4×10⁶
paires quelconques la corrélation maximale est **+0,162**, l'ordre de grandeur d'un maximum
d'échantillon gaussien — aucun champ n'est rejoué ni permuté. Direction fixe accumulée :
**0,100×** ce qu'une marche non biaisée produirait. Écart quadratique cumulé : 0,200 (L=1)
→ 1,032 (L=2048), saturation à √2·std(x_t) — diffusif puis borné.

**P6 — isotropie.** Sur le champ **ajouté** : énergie des différences rangée-à-rangée
0,08068 contre colonne-à-colonne 0,08065, **rapport 1,0003**. Autocorrélation spatiale
|ρ| ≤ 0,0013 à tous les décalages testés, **y compris (1,−1)** — le décalage qui aurait
explosé sous le piège anti-diagonale d'`ANISOTROPY_HUNT.md`.

**P7 — non-régression.** Errance : 22 cycles atteignent t = 0, images successives
différentes (corr 0,803–0,968) et dérivant avec la distance (0,748 à i+5). Respiration :
t ∈ [31, 64], jamais 0, min std(x_t) = 0,4817. Et surtout, au niveau fort : le binaire
**pré-flux** (`12acad8`) et le binaire courant produisent des PNG **strictement identiques**
sur les 8 premiers cycles d'errance **et** de respiration (8/8 et 8/8), même graine. Le
correctif du cadran — qui touche pourtant à un flag partagé par les trois régimes — n'a
déplacé aucune des deux dynamiques existantes.

**P8 — bornes et diagnostics du CLI** (promue d'observation à propriété, le contrat étant
désormais publié) : `--help` non vide ; flag inconnu rejeté en code 1 ; `flux --frames`
refusé explicitement ; 0 PNG en flux et > 0 dans les deux autres régimes ; `--actions`
l'emporte sur `--frames` ; le cadran répond à ses trois orthographes.

---

## Reproduire

```bash
./blind_tests/run.sh                     # 9 propriétés, ~60 s
BLIND_BASELINE=1 ./blind_tests/run.sh    # + comparaison bit-à-bit avec le binaire pré-flux
ACTIONS=20000 ./blind_tests/run.sh       # horizon long (stationnarité)
TSTARS="8 24 48 96 192 250" ./blind_tests/run.sh   # autre échantillonnage du cadran
```

Comparaison octet à octet avec le binaire de la passe 2 (§ passe 3), à refaire au besoin :

```bash
git worktree add worktrees/blind-p2-phase be42c27 --detach
cargo build --release -p main --manifest-path worktrees/blind-p2-phase/Cargo.toml
# puis même invocation flux, même --seed, et cmp des deux dumps
git worktree remove worktrees/blind-p2-phase
```

Le runner copie `night_run.ckpt` s'il est absent, écrit ses dumps dans `blind_tests/out/`
(ignoré par git), remet `perpetual_samples/` et `Models/` dans l'état du dépôt, et rend un
code de retour non nul si une propriété est FAIL. `blind_tests/baseline.sh` crée au besoin
le worktree `worktrees/blind-baseline-preflux` sur `12acad8` et le construit ; il est
supprimable par `git worktree remove worktrees/blind-baseline-preflux`.
