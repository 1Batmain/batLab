# Adam pour batLab — implémentation, comparatif apparié, recommandation

Branche `optimizer-adam`. Trois questions : Adam vaut-il SGD sur ce pipeline de
diffusion, à quel prix par pas, et quelle initialisation retenir pour entraîner
le modèle L.

**Recommandation, chiffrée sur la baseline `Greyscale_Diffusion` (1500 pas,
CIFAR-10 gris, batch 16) : `--optimizer adam --lr 1e-3 --weight-init uniform`.**

- Adam atteint **dès le pas 75** un niveau que SGD n'atteint **jamais en 1500
  pas**, sur les quatre tranches de t simultanément — un facteur **20 en nombre
  de pas**.
- Le surcoût de calcul d'Adam est **non mesurable** : −0,2 ms sur ~620 ms/pas,
  sous le plancher de bruit de la mesure.
- L'initialisation He est **légèrement en retrait** de l'uniforme historique sur
  cette architecture, partout. Voir §4 pour la réserve importante qui accompagne
  ce résultat.

---

## 1. Ce qui est implémenté

### 1.1 La passe Adam (`bat_building/src/model/shader/adam.wgsl`)

Adam (Kingma & Ba, 2015) tourne comme une passe de mise à jour par couche, en
remplacement de la passe SGD, avec la même structure de bindings plus quatre
buffers d'état persistants (`m`/`v` pour les poids et pour les biais) :

```
m ← β₁·m + (1-β₁)·g          m̂ ← m / (1-β₁ᵗ)
v ← β₂·v + (1-β₂)·g²         v̂ ← v / (1-β₂ᵗ)
w ← w − lr · m̂ / (√v̂ + ε)
```

Hyperparamètres : β₁ = 0,9, β₂ = 0,999, ε = 1e-8 (défauts de l'article).

### 1.2 Les trois choix non évidents

**La correction de biais est calculée sur le CPU, en f64, et passée en uniforme.**
`t` est le compteur de pas global du run et croît sans borne ; `pow(β₂, t)` en f32
dans le shader perdrait de la précision exactement là où `1-β₂ᵗ` s'approche de 1.
Le CPU tient déjà le compteur — autant faire le calcul là où il est exact.
`bias_corrections()` clampe à `f64::MIN_POSITIVE` pour qu'aucune valeur de β ne
puisse produire une division par zéro dans le shader.

**La moyenne du batch va sur le gradient, pas sur le learning rate.** Le gradient
accumulé est une *somme* sur le batch. SGD replie la moyenne dans le lr
(`lr / batch_size`) parce que sa mise à jour est linéaire en `g`. Celle d'Adam ne
l'est pas : elle est (à ε près) **invariante à un rééchelonnage de `g`**, donc
diviser le lr laisserait le pas effectif inchangé et le batch size n'aurait plus
aucun effet. D'où `grad_scale = 1/B` appliqué au gradient avant la mise à jour des
moments. Le test `adam_first_step_is_lr_sized_regardless_of_gradient_scale` fige
cette propriété : quel que soit |g|, le premier pas vaut ±lr.

**Le checkpoint devient versionné.** `BBCKPT2` persiste `m`, `v` et `t` en fin de
fichier ; `BBCKPT1` reste lu tel quel, et un checkpoint SGD se charge dans un run
Adam (état froid). Sans `m`/`v`/`t` dans le fichier, reprendre un run Adam
repartirait avec des moments nuls et une correction de biais à `t=1` : le premier
pas après reprise serait un saut de ±lr par poids, soit un accroc visible dans la
courbe. Le test `adam_state_survives_a_checkpoint_roundtrip` compare un run repris
à un run continu (identiques) et à un run à état froid (divergent).

### 1.3 SGD reste le défaut

`OptimizerKind::default() == Sgd`, et un config sans champ `optimizer` se
désérialise en SGD. Le test `sgd_update_is_unchanged_and_stateless` vérifie que
la passe SGD est restée `w -= lr·g` sans buffer d'état après le refactor : aucun
run antérieur ne devient incomparable.

### 1.4 Initialisation He (`--weight-init he`)

`N(0, 2/fan_in)` par Box–Muller, avec `fan_in` exposé par chaque type de couche
(`kh·kw·c_in` pour les convolutions, `dim_input.length()` pour le fully connected,
`None` pour GroupNorm dont γ/β ne sont pas une projection).

Le tirage uniforme historique — `U(-0,1, 0,1)` — est conservé **bit à bit** comme
défaut. Sur la baseline, les deux schémas ne jouent pas dans le même registre :

| couche | fan_in | poids | écart-type He | écart-type uniforme |
|---|---:|---:|---:|---:|
| conv 3→16 | 27 | 432 | 0,272 | 0,058 |
| conv 16→32 (stride 2) | 144 | 4 608 | 0,118 | 0,058 |
| conv 32→32 | 288 | 9 216 | 0,083 | 0,058 |
| upsample conv 32→16 | 288 | 4 608 | 0,083 | 0,058 |
| conv 32→1 | 288 | 288 | 0,083 | 0,058 |

Total : 19 409 paramètres entraînables. L'état d'Adam (2 f32 par paramètre) pèse
donc **152 Kio** sur ce modèle, et croît linéairement avec le nombre de poids.

### 1.5 Tests

`cargo test -p bat_building --release` : **71 passés, 0 échec**, dont l'oracle f64
d'Adam, l'invariance d'échelle, la non-régression SGD et le roundtrip de
checkpoint.

> ⚠️ `cargo test` **à la racine** affiche « 0 tests » : le paquet racine est
> `batBuilder` et les tests vivent dans le membre `bat_building`. Toujours passer
> `-p bat_building`. Même piège pour la compilation, voir §2.4.

---

## 2. Protocole

### 2.1 L'appariement est structurel, pas un réglage

`main.rs` dérive **toute** quantité aléatoire du seul indice de pas :

- graine de batch = `(step as u64) << 32` → ordre des données (`shuffle`), timestep
  de diffusion tiré par échantillon, et bruit ε ajouté ;
- graine du probe = `0x50B0_1234 ^ step`, sur un jeu de sonde fixe ;
- initialisation des poids : graine fixe.

Deux runs de même longueur et même batch size voient donc des entrées **octet pour
octet identiques**. Il n'y a pas de `--seed` d'entraînement à passer : la
reproductibilité n'est pas optionnelle ici. Vérification empirique
(`analyse.py --check-pairing`) : la loss de batch au pas 0, mesurée avant toute
mise à jour, est bit à bit identique entre les bras de même initialisation.

```
sgd_lr1e-3    1.1219401359558105
adam_lr1e-3   1.1219401359558105
adam_lr3e-4   1.1219401359558105   -> IDENTICAL
```

Toute différence entre deux bras au même pas est donc un effet d'optimiseur ou
d'initialisation, et rien d'autre. Réciproquement, deux bras qui ne diffèrent que
par l'init **doivent** diverger dès le pas 0 — c'est ce qui a permis d'attraper le
bug du §2.4.

### 2.2 La métrique est la loss par tranche de t, pas la loss de batch

La loss de batch porte sur 16 échantillons dont les timesteps sont tirés au
hasard : un batch chargé en t bas donne une loss élevée sans que le modèle ait
régressé. Sur le bras Adam elle oscille entre 0,002 et 0,22 d'un point de mesure à
l'autre — inexploitable pour comparer.

Le diagnostic `train_probe` est écrit tous les 25 pas sur un **jeu de sonde fixe**
avec un bruit déterministe par pas, ventilé en quatre tranches de t (0–64, 64–128,
128–192, 192–256). C'est la seule métrique directement comparable entre runs, et
c'est celle qu'utilise ce rapport.

Un « pas pour atteindre un seuil » exige ici que le seuil **tienne jusqu'à la fin
du run** : un creux isolé sous un seuil n'est pas une convergence.

### 2.3 Le GPU est partagé — ce que ça invalide

Les runs ont tourné pendant qu'un autre agent entraînait `Greyscale_Diffusion_L`
(10 000 pas) sur le même GPU. Conséquence directe, visible dans les temps bruts :

| bras | pas | secondes | s/pas |
|---|---:|---:|---:|
| `sgd_lr1e-3` | 1500 | 1508 | 1,005 |
| `adam_lr1e-3` | 1500 | 1081 | 0,721 |
| `adam_lr3e-4` | 1500 | 1965 | 1,310 |

Adam y apparaît **28 % plus rapide** que SGD, ce qui est impossible : Adam fait
strictement plus de travail par paramètre. Ce chiffre mesure la charge de la
machine, pas l'optimiseur. Les temps bruts des runs longs sont donc **écartés**
pour la question du coût, traitée séparément en §5 par une mesure dédiée.

En revanche la contention **n'affecte pas les courbes de loss** : les runs sont
déterministes, la comparaison numérique reste valide.

### 2.4 Un bras a dû être jeté : le piège du binaire périmé

Le paquet racine du workspace est `batBuilder` ; le binaire entraîné vit dans le
membre `main`. **`cargo build --release` à la racine ne reconstruit donc pas
`target/release/main`** — il affiche « Finished » après n'avoir bâti que la lib
racine.

Une première ablation He a tourné 1500 pas contre un binaire antérieur au commit
d'init He. `--weight-init he` n'était parsé par personne et **silencieusement
ignoré** : le bras était une copie octet pour octet du bras uniforme, et
ressemblait à un résultat parfaitement propre — « He ne change rien ». Détecté
parce que sa loss au pas 0 valait 1.121940, exactement celle du bras uniforme,
alors que changer l'init **doit** changer la passe avant dès le pas 0.

Deux garde-fous sont désormais dans `bench/optimizer/lib.sh` : `ensure_binary`
compile avec le `-p` qui produit réellement le binaire, et `assert_flag` relit la
bannière du run — où le binaire réémet la config qu'il a parsée — et fait échouer
le bras si le drapeau n'y figure pas.

Les résultats du §3 sont conservés : le chemin uniforme est inchangé par le commit
d'init He, ce qui a été vérifié en rejouant les 50 premiers pas avec le binaire à
jour — les enregistrements `train_loss` et `train_probe` (statistiques complètes
de ε̂ et ε comprises) sont bit à bit identiques.

---

## 3. SGD vs Adam

Loss du probe par tranche de t, moyenne des 5 derniers points (pas 1400–1499) :

| bras | t∈[0,64) | t∈[64,128) | t∈[128,192) | t∈[192,256) | moyenne |
|---|---:|---:|---:|---:|---:|
| `sgd_lr1e-3` | 0,6506 | 0,0940 | 0,0474 | 0,0415 | 0,2084 |
| **`adam_lr1e-3`** | **0,5651** | **0,0490** | **0,0102** | **0,0027** | **0,1567** |
| `adam_lr3e-4` | 0,5744 | 0,0525 | 0,0132 | 0,0065 | 0,1617 |

Aucun ε̂ non fini sur aucun bras : les trois runs sont numériquement sains.

### 3.1 Pas nécessaires pour atteindre — et tenir — un seuil

Tranche haut-t (t∈[192,256)), celle que la mission prend pour référence :

| seuil | `sgd_lr1e-3` | `adam_lr1e-3` | `adam_lr3e-4` |
|---|---:|---:|---:|
| ≤ 0,20 | 150 | **25** | 50 |
| ≤ 0,10 | 350 | **25** | 75 |
| ≤ 0,05 | 1075 | **75** | 175 |
| ≤ 0,02 | jamais | **200** | 550 |
| ≤ 0,01 | jamais | **375** | 950 |

**Le seuil de 0,05 en haut-t demandé par la mission : 1075 pas pour SGD, 75 pour
Adam — un facteur 14.**

### 3.2 Le chiffre qui résume : la parité

À quel pas Adam passe-t-il **définitivement** sous le résultat *final* de SGD
après 1500 pas ?

| tranche | SGD @1499 | Adam @1499 | rapport | Adam atteint le final de SGD au pas |
|---|---:|---:|---:|---:|
| t∈[0,64) | 0,6505 | 0,5690 | 1,1× | **75** |
| t∈[64,128) | 0,0931 | 0,0483 | 1,9× | **75** |
| t∈[128,192) | 0,0466 | 0,0102 | 4,5× | **75** |
| t∈[192,256) | 0,0411 | 0,0029 | **14,4×** | **75** |

Sur les quatre tranches à la fois, **Adam fait en 75 pas ce que SGD ne fait pas en
1500** : un facteur 20 en nombre de pas (à la résolution du probe, qui est de
25 pas — la parité est donc atteinte quelque part dans ]50, 75]).

Trajectoires (échelle log, un point tous les 25 pas, pas 0 → 1499) :

```
t∈[192,256)   sgd   ██▇▇▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
              adam  █▅▄▄▄▄▃▃▃▃▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁

t∈[128,192)   sgd   ██▇▆▆▆▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
              adam  █▄▃▃▃▃▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
```

SGD ne stagne pas — il descend, simplement **beaucoup** plus lentement, et son
allure ne suggère pas qu'il rattraperait Adam en prolongeant le run.

### 3.3 Learning rate

`lr = 1e-3` bat `3e-4` sur **toutes** les tranches et **tous** les seuils, sans
aucun signe d'instabilité (zéro ε̂ non fini, aucune divergence tardive). L'optimum
n'étant donc borné que par le bas, un bras `lr = 3e-3` a été ajouté pour
l'encadrer par le haut — voir §6.

### 3.4 Le plancher de la tranche t∈[0,64)

Les deux optimiseurs plafonnent vers 0,55–0,65 à t bas, et l'écart y est le plus
faible (1,1×). Ce n'est pas un échec d'optimiseur : à t petit, l'entrée est
presque propre et ε n'est que très faiblement déterminé par ce que le réseau
observe — la loss y a un plancher irréductible. C'est cohérent avec
`INSIGHTS_TRAINING.md`, qui fait de la loss à t bas un diagnostic de
conditionnement temporel et non une cible à minimiser. **Ne pas piloter un choix
d'hyperparamètre sur cette tranche.**

---

## 4. Ablation He (Adam, lr 1e-3, mêmes pas, même ordre de données)

| bras | t∈[0,64) | t∈[64,128) | t∈[128,192) | t∈[192,256) | moyenne |
|---|---:|---:|---:|---:|---:|
| **`uniform`** | **0,5651** | **0,0490** | **0,0102** | **0,0027** | **0,1567** |
| `he` | 0,5707 | 0,0501 | 0,0118 | 0,0052 | 0,1594 |

Pas pour atteindre et tenir un seuil, tranche haut-t :

| seuil | `uniform` | `he` |
|---|---:|---:|
| ≤ 0,10 | **25** | 50 |
| ≤ 0,05 | **75** | 125 |
| ≤ 0,02 | **200** | 350 |
| ≤ 0,01 | **375** | 850 |

**He est uniformément en retrait** — même ordre de grandeur, jamais devant.
Explication la plus plausible : **une GroupNorm suit la plupart des convolutions**
de cette architecture, et normalise les activations. L'échelle des poids est donc
en grande partie neutralisée en aval, et c'est précisément l'échelle que He fixe.
Le bénéfice de He (préserver la variance des activations à travers la profondeur)
est déjà rendu par la normalisation, tandis que ses poids plus larges (σ = 0,272
contre 0,058 sur conv1, soit 4,7×) perturbent légèrement la dynamique initiale.

### ⚠️ Réserve importante sur ce résultat

Le tirage uniforme historique est **dégénéré** : il n'utilise qu'un seul flux PRNG
à graine fixe pour toutes les couches. Sur cette baseline, la conv 16→32 et
l'upsample conv 32→16 ont toutes deux 4 608 poids et reçoivent donc des matrices
**identiques** ; les autres couches reçoivent des **préfixes** du même flux. He
corrige cela (un flux par couche) *en plus* de changer l'échelle.

L'ablation compare donc `{échelle historique + flux partagé}` à `{échelle He +
flux par couche}` — **deux variables à la fois**. Que l'uniforme l'emporte malgré
cette dégénérescence renforce la conclusion « l'échelle d'init n'est pas le
facteur limitant ici », mais **ne permet pas** de conclure que le flux partagé est
inoffensif. Le test propre reste à faire : *uniforme à flux par couche* vs *He*,
qui isolerait l'échelle. C'est un bras d'une trentaine de minutes, non couvert par
cette mission.

---

## 5. Surcoût d'Adam par pas

Les temps bruts des runs longs étant inexploitables (§2.3), le coût est mesuré à
part : pour chaque optimiseur un run court (50 pas) et un run long (250 pas), la
pente `(t_long − t_court)/200` annulant exactement le coût de démarrage ; bras
entrelacés dans chaque répétition et répétitions multiples.

| rép. | sgd 50 | adam 50 | sgd 250 | adam 250 | sgd ms/pas | adam ms/pas | écart |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 66,6 s | 67,0 s | 335,2 s | 155,2 s | 1343,0 | 441,4 | −901,6 |
| 2 | 30,1 s | 30,9 s | 154,8 s | 154,9 s | 623,6 | 619,6 | −4,0 |
| 3 | 31,6 s | 30,9 s | 155,1 s | 155,1 s | 617,5 | 621,1 | +3,6 |

La répétition 1 est **écartée** : ses temps absolus sont doublés à quintuplés par
rapport aux deux autres (335 s contre 155 s pour un run identique), signature
d'une pointe de charge de la machine. Elle est laissée dans le tableau plutôt que
supprimée, parce qu'elle illustre exactement l'artefact contre lequel ce
protocole est bâti.

**Sur les répétitions 2–3 : SGD 620,5 ms/pas, Adam 620,4 ms/pas, soit −0,2 ms/pas
(−0,03 %).** L'écart entre deux répétitions d'un *même* bras vaut jusqu'à 6,0 ms :
le surcoût d'Adam est donc **sous le plancher de bruit** de la mesure, et tout ce
qu'on peut affirmer est qu'il est inférieur à ~1 %.

Rien de surprenant : le pas est dominé par les passes avant/arrière sur des
activations 32×32, tandis que la passe d'optimisation est un seul kernel sur
19 409 éléments. Côté mémoire, Adam ajoute 2 f32 par paramètre, soit 152 Kio ici.

**Extrapolation au modèle L** : l'état d'Adam croît linéairement avec le nombre de
poids, et le coût de la passe d'optimisation aussi — mais le coût des passes
avant/arrière croît au moins aussi vite. Le surcoût relatif devrait donc rester
du même ordre, sans que la présente mesure le prouve pour L.

---

## 6. Recommandation pour l'entraînement du modèle L

```bash
cargo build --release -p main     # PAS `cargo build --release` seul (§2.4)
./target/release/main --headless-train Greyscale_Diffusion_L \
    --steps N --dataset datasets/cifar10_grey.batraw \
    --optimizer adam --lr 1e-3 --weight-init uniform
```

| paramètre | valeur | fondement |
|---|---|---|
| optimiseur | **adam** | 14× moins de pas pour 0,05 en haut-t ; 14,4× plus bas à 1500 pas ; surcoût non mesurable |
| lr | **1e-3** | bat 3e-4 sur toutes les tranches et tous les seuils, sans instabilité ; voir ci-dessous pour la borne haute |
| init | **uniform** (défaut) | He en retrait partout sur la baseline, sous la réserve du §4 |
| batch | 16 (inchangé) | non testé dans cette mission |

### Réserves à porter au dossier

1. **Tout est mesuré sur la baseline, pas sur L.** `Models/Greyscale_Diffusion_L`
   n'existe pas sur cette branche (il vit sur `unet-scale`). L est plus profond ;
   la profondeur est précisément le régime où l'init compte le plus. Le verdict
   « uniforme » est donc le plus fragile des trois à transposer — d'autant que la
   GroupNorm, à laquelle on impute la neutralisation de l'échelle, est le
   mécanisme qui rendrait aussi He inoffensif à toute profondeur. **Si un run L
   sous-performe, l'init est le premier paramètre à rejouer**, et le bras propre
   du §4 (uniforme à flux par couche) est à faire avant.
2. **Le lr n'est encadré que par le bas** au moment d'écrire ces lignes ; un bras
   `lr = 3e-3` est en cours pour tester la borne haute. Si 3e-3 fait mieux que
   1e-3, la recommandation de lr doit être révisée vers le haut — les autres
   conclusions n'en dépendent pas.
3. **Un seul run par configuration.** L'appariement est exact (mêmes données, même
   bruit), donc les comparaisons sont valides comme comparaisons *de ces graines*.
   Les écarts SGD/Adam sont trop grands pour venir du hasard de l'init ; l'écart
   uniforme/He, lui, est petit — il mériterait plusieurs graines avant d'être
   considéré comme robuste.

---

## Reproduire

```bash
bash bench/optimizer/compare.sh        # sgd 1e-3, adam 1e-3, adam 3e-4
bash bench/optimizer/queue.sh          # ablation He, coût par pas, adam 3e-3
python3 bench/optimizer/analyse.py --check-pairing \
    sgd_lr1e-3=runs/sgd_lr1e-3_metrics.jsonl \
    adam_lr1e-3=runs/adam_lr1e-3_metrics.jsonl
```

`runs/` est gitignoré : les scripts sont versionnés, les métriques brutes non.
