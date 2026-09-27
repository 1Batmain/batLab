# Rendre le 32×32 capable — instrument, expressivité, mesure

Mission `archi32` : le modèle de diffusion 32×32 sous-apprend (« ce n'est pas
nettement un éléphant »). Quatre chantiers : (1) l'**instrument** pour juger une
architecture autrement qu'à l'œil ; (2) l'**addition** pour rendre les blocs
résiduels exprimables ; (3) l'**injection du timestep** dans chaque bloc ; (4)
**grossir et mesurer**. Ce rapport donne le tableau `--eval` de chaque étape, ce
qui a rendu et ce qui n'a **rien** rendu — un NO-GO chiffré valant autant qu'un
gain.

**Résumé** : l'instrument (`--eval`), l'addition (`Add`) et l'injection-t
(`TimeBias`) sont livrés, câblés dans le TUI et éprouvés au standard kernel du
dépôt (trois bugs moteur Metal réels corrigés au passage). Passés au banc
elephants (from scratch, 6000 pas, mêmes hyperparamètres), les trois leviers
d'architecture — injection-t, largeur ×1.5, résidu profond — donnent un **NO-GO
complet sur le critère de succès** (aucun ne passe la tranche t-haut sous 0.4366) ;
seule la largeur rend un gain modeste, à 2× le coût. Le défaut central est en
partie **structurel** (x̂₀ mono-pas) et en partie un **manque de pas
d'entraînement**, pas un manque d'expressivité — laquelle est désormais disponible
si une mesure future la réclame.

---

## Item 1 — `--eval`, l'instrument (commit `feat(eval)`)

Le manque que le rapport EMA a payé cash : aucun moyen de dire « ce checkpoint
est meilleur que celui-là » sans regarder l'image. `--eval` le comble.

```
cargo run -p batlab -- --eval <model> --ckpt <a.ckpt> --ckpt <b.ckpt> \
    --dataset <d.batraw> [--samples N] [--buckets N] [--t-per-bucket N] \
    [--seed N] [--raw-weights]
```

Contrat :

- **Tenu à l'écart et déterministe** : mêmes images (la queue du dataset), mêmes
  t, mêmes champs de bruit d'un appel à l'autre ET d'un checkpoint à l'autre. La
  graine des tirages est une fonction pure de `(seed, image, tranche, t)`. Une
  différence dans les chiffres est une différence dans les poids et rien d'autre.
- **ε-MSE par tranche de t**, plus l'erreur portée en espace image : x̂₀ est la
  reconstruction **clipée** que le sampler construit (`x0_estimate`), pas le
  facteur brut `(1-ᾱ)/ᾱ` qui explose à haut t et laisse ε̂=mean gagner pour rien.
  Le clip borne la reconstruction dans [-1,1] et rend les baselines comparables.
- **Trois baselines triviales** (`trivial_baselines.py`) : ε̂=0, ε̂=x_t, ε̂=mean.
  Un modèle qui ne bat pas ε̂=mean en x₀ n'a rien reconstruit que l'image
  moyenne ne donnait déjà.
- Respecte l'EMA du checkpoint ; `--raw-weights` force l'itéré.
- Verdict **non binaire** : rang primaire = ε-MSE pleine échelle ; lecture
  secondaire = combien de tranches battent ε̂=mean en x₀. Un x̂₀ **mono-pas** est
  inapprenable au sommet du schedule pour N'IMPORTE quel modèle (c'est pourquoi
  la génération prend 256 pas), donc une moyenne globale en x₀ n'est
  délibérément pas le verdict.

Moteur GPU-libre : `evaluate` prend le modèle comme closure `predict`, donc les
maths du schedule et la reproductibilité sont testées contre des prédicteurs
oracles sans GPU (7 tests).

**Baseline Elephants_XL à battre** (mesurée par l'architecte avec cet outil, sur
`elephants_all`, queue vraiment tenue à l'écart) :

| tranche | x₀-RMSE modèle | ε̂=mean | verdict |
|---|---|---|---|
| t[0-64) | 0.1688 | 0.437 | vu 0.1652 vs inconnu 0.7703 → **mémorise** |
| t[64-128) | 0.2601 | 0.437 | bat la moyenne |
| t[128-192) | 0.3702 | 0.437 | bat la moyenne |
| **t[192-256)** | **0.6950** | **0.4366** | **BATTU par la moyenne** — paris confiants et faux sur la silhouette |

ε-MSE all = 0.2016. **Critère de succès des items 2-3** : tranche t-haut sous
**0.4366**, et écart vu/inconnu réduit à t bas.

---

## Item 2 — `Add`, la connexion résiduelle courte (commit `feat(add)`)

Le moteur ne savait pas additionner. `Add` réutilise la plomberie de `Concat`
(tenseur sauvé relu, bifurcation du gradient) et n'ajoute que `a+b`. Distinction
tenue : les skips **longue portée** du U-Net concatènent (`Concat`) ; le résidu
**courte portée** `sortie = f(x)+x` additionne (`Add`). L'addition EXIGE des
formes identiques ; une discordance est refusée au build en nommant la couche et
les deux largeurs (`ResidualDimMismatch`).

Validé GPU (oracle f64 forward, différences finies backward à travers un vrai
bloc résiduel, mutation). **Défaut moteur réel corrigé** : un backward en deux
passes partageant un bind group read-write privait le merge aval de sa barrière
(gradient de la conv productrice calculé depuis grad_skip=0, intermittent,
build-dépendant). Backward en une passe → 0/25 série, 0/10 parallèle.

Bout-en-bout : un modèle résiduel s'entraîne (loss 0.019 à 300 pas) et passe à
`--eval`. Utilisable dans les candidats.

---

## Item 3 — `TimeBias`, le timestep dans chaque bloc (commit `feat(time-bias)`)

`output[b,h,w,c] = input[b,h,w,c] + bias[c] + Σ_n emb[b,n]·W[n,c]`. L'embedding
est lu d'une copie sauvée de l'entrée (spatialement constant → lu au pixel 0,
donc marche à toute résolution). Démarre à l'identité (W, biais à zéro).

Validé GPU (oracle f64 forward, per-sample, différences finies sur W ET biais,
mutation). **Trois défauts moteur réels corrigés**, tous à symptôme identique
(grad_weights=0, intermittent, build-dépendant) : (1) uniforme `vec3`+scalaires
mal aligné (encase 28 o vs WGSL 32) ; (2) read-modify-write d'un buffer storage
dans une invocation (indéfini sous WGSL) ; (3) `clear_buffer` puis écrasement =
WAW que wgpu ne barrière pas sur Metal. La forme qui survit à naga→MSL est celle
**exacte** de GroupNorm : deux passes séparées, une workgroup par canal, un seul
écrivain qui **accumule** sur le buffer effacé.

### La mesure — A/B apparié sur CIFAR grey, à paramètres comparables

Deux modèles **identiques** sauf 4 `TimeBias` insérés dans les blocs profonds
(1536 params ajoutés sur ~460k — négligeable). Même dataset, mêmes pas, même
optimiseur, même lr.

Adam, lr 1e-3, `cifar10_grey`. x₀-RMSE par tranche ; ε̂=mean ≈ 0.48 partout.

**À 2500 pas :**

| | ε-MSE all | t[0-64) | t[64-128) | t[128-192) | t[192-256) | ms/pas |
|---|---|---|---|---|---|---|
| Baseline (t à l'entrée seule) | 0.0882 | 0.132 | 0.277 | 0.474 | 0.909 | ~43 |
| TimeBias (t dans chaque bloc) | 0.0879 | 0.132 | 0.277 | 0.477 | 0.916 | ~86 |

**À 6000 pas** (le TimeBias part de l'identité, on lui laisse le temps de monter) :

| | ε-MSE all | t[0-64) | t[64-128) | t[128-192) | t[192-256) |
|---|---|---|---|---|---|
| Baseline | 0.0829 | 0.1296 | 0.2723 | 0.4343 | 0.8674 |
| TimeBias | 0.0827 | 0.1295 | 0.2721 | 0.4337 | **0.8711** |

**NO-GO mesuré sur CIFAR, confirmé à deux échelles.** Le gain est nul à 2500 pas
(ε-MSE 0.0879 vs 0.0882) et toujours nul à 6000 (0.0827 vs 0.0829, dans le
bruit) ; la tranche t-haut n'est PAS améliorée — elle est même très légèrement
pire (0.8711 vs 0.8674), toutes deux battues par la moyenne. Et le pas
d'entraînement **double** (~86 vs ~43 ms). Sur CIFAR, injecter t en profondeur
ne rend rien et coûte 2×.

**Valeur de l'instrument, démontrée ici** : l'item 3 était « l'hypothèse la plus
prometteuse de la mission ». Sans `--eval` elle aurait été livrée comme un gain
supposé ; mesurée, elle est un NO-GO sur ce banc. C'est exactement le rôle de
l'item 1 — empêcher qu'un déplacement inerte passe pour un progrès.

**Pourquoi rien sur CIFAR** — hypothèse : sur un dataset large et divers (50k
images) où le modèle **généralise déjà**, le t injecté à l'entrée (3 canaux
d'embedding) se propage assez ; le chemin profond redondant n'a pas de raison de
s'activer et reste près de l'identité. Le défaut Elephants (t-haut battu par la
moyenne) est mesuré sur **1300 images**. Il fallait donc le mesurer sur elephants
avant de conclure — c'est fait à l'item 4 ci-dessous, et **le NO-GO se confirme
sur la vraie cible** : B ≈ A (ε-MSE 0,0713 = 0,0713 ; t-haut 0,935 vs 0,949).
Injecter t en profondeur ne rend rien, ni sur CIFAR, ni sur elephants.

---

## Item 4 — Grossir et mesurer (banc elephants)

Banc committé (`tools/gen_archi32_bench.py`, `Models/Archi32_{A,B,C,D}_*`), quatre
archi 32×32 RGB entraînées **from scratch, mêmes pas (6000) / lr (1e-3) / batch
(32) / graine**, sur `elephants_all` (1300 images), jugées par `--eval`.

| candidat | params | ckpt | ms/pas | ε-MSE all |
|---|---|---|---|---|
| **A** baseline (archi actuelle, attention 8×8) | 1,18 M | 4,5 Mo | 163 | 0,0713 |
| **B** = A + TimeBias (blocs profonds) | 1,20 M | 4,6 Mo | 220 | 0,0713 |
| **C** = A largeur ×1.5 (72/144/288) | 2,68 M | 10,2 Mo | 342 | 0,0722 |
| **D** = A + 1 bloc résiduel/étage (Add) | 1,63 M | 6,2 Mo | 208 | 0,0712 |

**x₀-RMSE par tranche, sur `elephants_all` (queue tenue à l'écart)** — plus bas
meilleur ; ε̂=mean = **0,4366** partout, la cible à passer :

| | t[0-64) | t[64-128) | t[128-192) | **t[192-256)** | bat la moyenne |
|---|---|---|---|---|---|
| A baseline | 0,1132 | 0,2362 | 0,4496 | **0,9492** | 2/4 |
| B TimeBias | 0,1130 | 0,2359 | 0,4414 | **0,9350** | 2/4 |
| **C wide** | 0,1148 | 0,2401 | **0,4167** | **0,8821** | **3/4** |
| D résiduel | 0,1129 | 0,2377 | 0,4533 | **0,9248** | 2/4 |

### Verdict : NO-GO complet sur le critère de succès

**Aucun candidat ne passe t[192-256) sous 0.4366.** Le meilleur, C, est à 0,882 —
deux fois la cible. Détail :

- **B (TimeBias) : NO-GO, confirmé sur la vraie cible.** Identique à A sur
  l'objectif (ε-MSE 0,0713 = 0,0713), gain infime en x₀ (t-haut 0,935 vs 0,949).
  Exactement le NO-GO CIFAR de l'item 3, reproduit sur elephants. +35 % de coût
  d'entraînement, alourdit l'inférence, ne rend rien : **ne va pas en production.**
- **D (résidu profond) : NO-GO.** À peine mieux que A (t-haut 0,925 vs 0,949),
  ε-MSE identique. Approfondir avec `Add` sans élargir ne bouge pas l'aiguille ici.
- **C (largeur ×1.5) : le seul gain mesurable, et il est modeste.** Bat la moyenne
  à **3/4** tranches (gagne t[128-192) à 0,417), meilleur t-haut (0,882) et
  meilleur x₀-all (0,506 vs 0,541). Mais : ε-MSE all *légèrement pire* (0,0722),
  gain purement dans la repondération x₀, pour **2,3× les params et 2× le coût**.

### Écart vu/inconnu (mémorisation) : plat à 6000 pas

Évalué sur `elephants256` (vu, recouvrement total) ET `elephants_all` (queue
inconnue), t[0-64) :

| | vu (256) | inconnu (all) | écart |
|---|---|---|---|
| A | 0,1122 | 0,1132 | ~0 |
| C | 0,1141 | 0,1148 | ~0 |

**Aucune mémorisation à 6000 pas** (≈150 époques) — les modèles généralisent
encore. C'est un **régime différent** de l'Elephants_XL de l'architecte (écart
0,165 vu / 0,770 inconnu, 4,7×), qui a dû tourner bien plus longtemps. Deux
conséquences : (1) l'indicateur vu/inconnu ne départage pas les candidats à ce
budget ; (2) le t-haut de mon A (0,949) est **pire** que l'Elephants_XL de
l'architecte (0,695) — donc **plus de pas améliore le t-haut davantage que
n'importe quelle de ces trois modifications d'archi.**

### Ce que la mesure dit, franchement

1. **Le t-haut « battu par la moyenne » est d'abord STRUCTUREL, pas un défaut
   d'archi.** Aucun des trois leviers (injection-t, largeur, profondeur) ne
   l'approche de 0,4366 ; le meilleur reste à 2× la cible. Un x̂₀ **mono-pas** au
   sommet du schedule est inapprenable — l'image moyenne, lisse et en gamme, bat
   toujours un pari saturé, et c'est pourquoi la génération prend 256 pas.
   `--eval` mesure le mono-pas ; **il ne faut pas le lire comme le juge unique de
   la silhouette**, qui se décide sur la chaîne complète (l'œil, ou une MSE sur ε
   moyennée sur la trajectoire — non implémentée, cf. le manque signalé par le
   rapport EMA).
2. **Sur l'objectif d'entraînement (ε-MSE), les quatre sont à égalité** (0,071–
   0,072). Les seules différences vivent dans la reconstruction x₀, et seule la
   largeur (C) les déplace de façon lisible.
3. **Le levier dominant à ce stade est le NOMBRE DE PAS, pas l'archi** : mon A à
   6000 pas est plus loin de la cible que l'Elephants_XL bien plus entraîné.

### Recommandation pour le run long

**Ne pas partir sur TimeBias ni sur le résidu profond** — mesurés NO-GO, ils
coûtent sans rendre. Deux options défendables :

- **A (baseline), ~15 000 pas** — le meilleur rapport qualité/coût. L'archi
  actuelle n'est pas le goulot mesurable ici ; le budget de calcul est mieux
  dépensé en pas qu'en paramètres. Coût : 163 ms/pas × 15 000 ≈ **41 min**.
  Au-delà de ~10 000 pas, l'EMA (`--ema 0.999`) reprend un sens (la rampe cesse
  de dominer la valeur nominale, cf. AGENTS.md) et la mémorisation apparaît —
  surveiller alors l'écart vu/inconnu et le `--checkpoint-every` pour garder le
  meilleur avant sur-mémorisation.
- **C (largeur ×1.5), ~15 000 pas** — si l'on veut le petit gain de
  reconstruction mesuré (bat la moyenne à 3/4), au prix de 2× le calcul
  (**~85 min**) et 2,3× les params — ce qui pèse sur la cible web (inférence dans
  le navigateur du visiteur). À ne prendre que si le gain x₀ modeste vaut ce
  doublement ; sur ces chiffres, **je penche pour A + plus de pas.**

**En une phrase** : l'instrument (item 1) a fait son travail — il a transformé
« l'hypothèse la plus prometteuse » (injection-t) et deux paris de capacité en
chiffres, et le verdict est un **NO-GO complet sur le critère t-haut < 0.4366**,
avec un seul gain modeste (largeur) qui ne le justifie pas à 2× le coût. Le
défaut central est en partie structurel (x̂₀ mono-pas) et en partie un manque de
pas d'entraînement, pas un manque d'expressivité de l'archi — qui est désormais,
elle, disponible (addition, injection-t) si une mesure future la réclame.

## Notes pour après les merges

- Les blocs résiduels et t-conditionnés méritent d'être **générés par les
  templates** (mission de l'agent `templates`, en parallèle) : `built_in_templates()`
  n'a pas été touché ici. Le patron d'un bloc t-conditionné est : sauver l'entrée
  (`Activation Linear save "tin"`), et dans chaque bloc `Conv → GN → SiLU →
  TimeBias("tin") → …`, avec un `Add` sur le raccourci quand la largeur ne change
  pas (sinon conv 1×1).
- **Optimisation `TimeBias` backend** : `time_bias_grad_params` tourne en
  `@workgroup_size(1)`, une workgroup par canal — correct et robuste, mais c'est
  ce qui double le pas. Une réduction coopérative (comme le `workgroup_sum` de
  GroupNorm) le ramènerait au coût d'un GroupNorm. À faire si TimeBias entre en
  production.
