# Rendre le 32×32 capable — instrument, expressivité, mesure

Mission `archi32` : le modèle de diffusion 32×32 sous-apprend (« ce n'est pas
nettement un éléphant »). Quatre chantiers : (1) l'**instrument** pour juger une
architecture autrement qu'à l'œil ; (2) l'**addition** pour rendre les blocs
résiduels exprimables ; (3) l'**injection du timestep** dans chaque bloc ; (4)
**grossir et mesurer**. Ce rapport donne le tableau `--eval` de chaque étape, ce
qui a rendu et ce qui n'a **rien** rendu — un NO-GO chiffré valant autant qu'un
gain.

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
moyenne) est mesuré sur **1300 images en régime de mémorisation** — un régime que
CIFAR n'atteint pas. **Le NO-GO CIFAR ne transfère donc pas mécaniquement à
Elephants**, et l'inverse non plus : c'est exactement pourquoi l'item 4 doit se
mesurer **sur `elephants_all`**.

---

## Item 4 — Grossir et recommander

### Blocage

`datasets/elephants_all.batraw` (les images de l'utilisateur) **n'est pas dans ce
worktree** — l'architecte l'a mesuré ailleurs. L'A/B décisif (ancienne archi vs
nouvelle, from scratch sur elephants) doit donc être lancé là où le dataset vit.
Les deux configs sont prêtes et l'outil est en place ; voir la recette ci-dessous.

### Recette recommandée pour le run long (elephants, from scratch)

- **Pas** : ≥ 8000 (le projet note que sous ~10 000 pas la valeur nominale d'EMA
  ne lie pas — c'est la rampe qui compte). Sur 1300 images, surveiller l'écart
  vu/inconnu à t bas comme signal de mémorisation.
- **lr** 1e-3, **optimiseur** adam, **batch** 32.
- **EMA** : à cette échelle (< 10 000 pas), inutile — la rampe domine la valeur
  nominale, et le rapport EMA la classe NO-GO sur run court. Ne pas l'activer
  « pour faire propre ».
- **Coût estimé** sur cette machine : le pas TimeBias mesuré ~86 ms (2× le
  baseline à cause du backend `grad_params` en `@workgroup_size(1)`, optimisable
  — voir plus bas). 8000 pas ≈ 11–12 min.

### A/B à lancer (sur elephants_all)

```
# entraîner les deux, mêmes pas / lr / seed, from scratch
cargo run --release -p batlab -- --headless-train <Baseline>  --steps 8000 \
    --dataset datasets/elephants_all.batraw --lr 1e-3 --optimizer adam --out A.ckpt
cargo run --release -p batlab -- --headless-train <TimeBias> --steps 8000 \
    --dataset datasets/elephants_all.batraw --lr 1e-3 --optimizer adam --out B.ckpt
# juger, sur la queue tenue à l'écart
cargo run --release -p batlab -- --eval <TimeBias> --ckpt A.ckpt --ckpt B.ckpt \
    --dataset datasets/elephants_all.batraw --samples 256
```

Critère : la tranche **t[192-256)** de B passe-t-elle **sous 0.4366** (la
moyenne) ? Et l'écart vu/inconnu à t bas se réduit-il ?

### Recommandation

1. **Lancer d'abord l'A/B elephants ci-dessus — ne pas présumer du gain.** Le
   NO-GO CIFAR n'invalide pas TimeBias sur elephants (régimes différents :
   généralisation vs mémorisation), mais il retire toute raison d'y croire *a
   priori*. Le banc elephants tranche ; il coûte ~25 min (deux runs 8000 pas +
   deux `--eval`). Si t[192-256) de B ne passe pas sous 0.4366, **TimeBias est
   un NO-GO tout court** et ne va pas en production (il coûte 2× à
   l'entraînement et alourdit l'inférence — contrainte web).

2. **La tranche t-haut battue par la moyenne est en partie structurelle, pas
   seulement un défaut de conditionnement.** Un x̂₀ **mono-pas** au sommet du
   schedule est inapprenable : l'image moyenne (lisse, en gamme) bat toujours un
   pari saturé. C'est pourquoi la génération prend 256 pas. `--eval` mesure le
   mono-pas ; la *vraie* silhouette se juge sur la chaîne complète (l'œil, ou une
   MSE sur ε moyennée sur la trajectoire — non implémentée). **Ne pas sur-lire la
   tranche t-haut de `--eval` comme le seul juge de la silhouette.**

3. **Le levier le plus sûr reste la CAPACITÉ, mais le constat le nuance.** Le
   DDPM de référence fait ~35 M params contre 1,19 M ici (30×). Mais sur CIFAR
   grey, ce petit modèle atteint déjà ε-MSE 0.083 et bat la moyenne à 3/4
   tranches — il n'est pas manifestement sous-dimensionné *sur ce banc*. Sur
   elephants (1300 images), le constat note que « dataset plus petit que la
   mémoire n'est pas atteint, mais ce n'est pas le facteur dominant ». Donc :
   grossir est plausible mais **doit être mesuré**, pas supposé — exactement
   comme TimeBias. Deux candidats à passer au banc elephants, à côté du baseline :
   **(a) largeur ×1.5** (32→48 canaux à l'entrée, proportionnel ensuite) ;
   **(b) un bloc résiduel de plus par étage** (avec l'`Add` de l'item 2 :
   `Conv→GN→SiLU→Conv→GN→SiLU→Add(entrée)`), qui approfondit sans exploser la
   largeur ni le champ réceptif. Mesurer les deux avec `--eval` sur
   `elephants_all`, coût ckpt et ms/pas relevés, et ne retenir que ce qui bat le
   baseline **sur elephants**.

**En une phrase** : l'instrument et l'expressivité (addition, injection-t) sont
livrés et éprouvés ; la seule chose qui a été *mesurée* comme rendant quelque
chose reste à établir — sur elephants, pas sur CIFAR — et le banc pour le faire
est en place. Livrer une archi « plus grosse + t profond » sans le banc elephants
serait répéter l'erreur que l'item 1 existe pour empêcher.

---

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
