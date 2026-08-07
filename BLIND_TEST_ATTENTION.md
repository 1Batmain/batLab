# Test à l'aveugle — la couche d'attention

Agent de test indépendant, 7 août 2026, branche `blind-attention`.
Ordre de mission : `MISSION_BLIND_ATTENTION.md`.

**Verdict global : les cinq propriétés de la spec sont vérifiées.** Quatre le
sont directement, la cinquième (anti-fuite inter-échantillons) partiellement —
et la partie manquante est *rendue inobservable par la spec elle-même*, pas
esquivée. Quatre défauts d'observabilité et un défaut de contrat CLI sont
consignés en §7.

Ce que j'ai lu : `MISSION_BLIND_ATTENTION.md`, `CLAUDE.md`, `GO_NOGO.md`,
`docs/reports/ATTENTION.md`, la sortie de `batlab --help`, les `config_file` des
modèles, `tools/sample_diversity.py`. Ce que je n'ai pas lu : aucun fichier de
`crates/*/src/`, aucun `.wgsl`, aucun `attention_tests.rs`, aucun diff.

---

## 1. Le verrou, et comment il a sauté

Le binaire n'expose pas la couche isolée : la mission prévoyait explicitement
qu'on puisse conclure « non observable ». Ça n'a pas été nécessaire. Trois
constats, empilés, ouvrent la couche à l'observation complète depuis l'extérieur.

**(a) Le conteneur `.ckpt` se laisse lire — et écrire.** Aucune source n'a été
consultée : un hexdump, plus le comptage des paramètres de configs dont je
connais la géométrie, suffisent à faire tomber le format, et le parseur atterrit
**à l'octet près** sur la fin des deux fichiers de contrôle (le jouet SGD de
2 175 o et les 14 311 503 o de `night_run.ckpt`, moments Adam compris).

```
b"BBCKPT2"  u32 n_records
n_records ×  { u32 index_de_couche, u32 n_w, f32[n_w], u32 n_b, f32[n_b] }
u32 optimiseur_présent   [ u64 pas, u32 n, n × { u32 index, u32 k, k×(u32 len, f32[len]) } ]
```

Les couches sans paramètre (Activation, Concat) sont **absentes** de la liste —
c'est l'`index_de_couche` qui raccroche chaque enregistrement à la config. Le
tenseur d'attention y apparaît exactement comme l'annonce `ATTENTION.md` §2.1 :
un seul bloc de `4·C·C` poids et `4·C` biais (147 456 / 768 sur
`Color_Diffusion_XL`). Codec : `blind_tests_attention/ckpt.py`.

Conséquence : **je choisis les poids**. La couche cesse d'être une boîte noire
au sens des entrées.

**(b) Le pas inverse est affine en la sortie du réseau.** `--headless-perpetual
--dump` publie, par frame, `t`, `x_t` et `x0_hat` en f32. Le DDPM impose

```
x0_hat(t) = a(t) · x_pre − b(t) · ε̂
```

Deux runs de calibration suffisent à fixer `a` et `b` **sans rien supposer du
schedule** : réseau tout à zéro (⇒ `ε̂ ≡ 0`, donc `a = x0_hat / x_pre`), puis
poids nuls et biais constant connu (⇒ `b`). Sur les positions non écrêtées, `a`
et `b` sortent **constants à 3·10⁻⁸ près** sur les 192 valeurs d'une frame :
l'hypothèse affine n'est pas postulée, elle est mesurée. Contrôle croisé
gratuit : `b/a = √(1−ᾱ)` et `a = 1/√ᾱ` sont mutuellement cohérents à 10⁻⁵ près
aux deux niveaux calibrés (t = 94 et t = 54).

`ε̂ = (a·x_pre − x0_hat) / b` : **la sortie brute du réseau est lisible**.

**(c) Un modèle sonde met la couche à nu.** `[Convolution 1×1, Attention]` sur
`[w, h, c+1] → [w, h, c]`. La convolution reçoit l'identité `g·I` sur les `c`
canaux image et **zéro sur le canal d'embedding temporel** : l'entrée de
l'attention vaut donc `g · x_t`, où `x_t` est publié par le dump — sans rien
supposer de l'embedding. Validé de bout en bout avant de servir : sur le modèle
sans attention, `ε̂` lu vaut la convolution attendue à **1,4·10⁻⁷** en relatif.

Le gain vaut `g = a/b`. Ce n'est pas cosmétique : les deux termes en `x_pre`
s'annulent alors dans `x0_hat`, ce qui libère **toute** la fenêtre non écrêtée
du sampler pour la seule contribution de l'attention. C'est ce qui permet de
faire tourner la couche sur une entrée d'amplitude ~1 — donc une softmax
franchement non uniforme, `p_max/p_min` jusqu'à 5·10¹⁰ — au lieu du régime
quasi uniforme où presque n'importe quelle formule passerait.

---

## 2. Verdict par propriété

| # | Propriété de la spec | Verdict |
|---|---|---|
| 1 | Identité au départ (`W_o = 0`) | **PASS** |
| 2 | La couche entraînée fait quelque chose, génération non dégénérée | **PASS** |
| 3 | Round-trip / déterminisme du checkpoint | **PASS** (round-trip via le conteneur publié — voir D4) |
| 4 | Référence f64 indépendante | **PASS** — la propriété la plus fortement établie |
| 5 | Anti-fuite inter-échantillons | **PASS partiel** — forward couvert, backward NON OBSERVABLE (D5) |
| + | Stabilité de la softmax (logits ≫ ln f32_max) | **PASS** (hors mission) |
| + | Grille de dispatch au-delà de 65 535 workgroups | **PASS** (hors mission) |

Suite : `./blind_tests_attention/run.sh` — 7 volets, **32 assertions, 32 vertes**.

---

## 3. P1 — identité au départ

Trois niveaux, du plus structurel au plus observable.

**Structure du checkpoint.** Sur le jouet (C=3) comme sur `Color_Diffusion_XL`
(C=192), le tenseur `4·C·C` fraîchement construit a **exactement un bloc nul, et
c'est le dernier** ; les trois autres ne le sont pas (`|Q|max = |K|max = |V|max
= 0,100` sur le XL). Les quatre biais sont nuls. Les deux moitiés comptent :
une couche entièrement à zéro serait « identité » et morte — la spec le dit
(`ATTENTION.md` §4.4) et c'est vérifié indépendamment ici.

> **Piège** — voir D1 : il faut `--lr 0`. Avec `--steps 0` seul, le binaire
> exécute quand même **un** pas d'optimiseur et `W_o` vaut déjà 1,2·10⁻⁶.

**Sortie de la couche.** Avec `W_o = 0` et Q/K/V arbitraires non nuls, la sortie
brute du réseau lue par la sonde vaut son entrée à **4,6·10⁻⁷** près, sur les
trois formes 8×8, 9×8 et 16×16, pour une entrée d'amplitude 2,9.

**Génération.** `Color_Diffusion_XL` (30 couches) contre le **même modèle privé
de sa seule couche `Attention`** (29 couches, les deux GroupNorm conservées) :
PNG **identiques octet pour octet** (sha256 `1ef7b078…`). Les deux checkpoints
sont écrits par la sonde depuis la même init : **tous les poids partagés sont
bit-à-bit identiques**, il n'y a donc aucun décalage de tirage d'init à
confondre avec l'effet de la couche — le piège naturel de la formulation
« deux configs, poids frais même seed » de l'ordre de mission.

Deux compléments :

- Q/K/V remplacés par du bruit arbitraire, `W_o` toujours nul → **même PNG**.
  L'identité ne tient donc pas par accident d'initialisation.
- La variante littérale de l'ordre de mission (`GroupNorm + Attention` retirées,
  28 couches) donne elle aussi le même PNG : sur cette chaîne, `GN ∘ GN = GN`
  bit à bit. Bonne nouvelle, mais **cette variante-là n'est pas décisive** — son
  résultat dépend de l'idempotence de GroupNorm autant que de l'attention.

---

## 4. P2 — la couche entraînée fait quelque chose

`night_run.ckpt` = `.checkpoint_backup/Color_Diffusion_XL_attn_5700.ckpt`
(copié dans la racine jetable, jamais dans `Models/` du dépôt, jamais commité).

- Le pas enregistré dans le checkpoint est bien **5 700**. Les quatre blocs sont
  sortis de l'init : écarts-types Q/K/V/W_o = 0,0634 / 0,0627 / 0,0677 / 0,0257
  et `|W_o|max = 0,275`. **La couche n'est plus l'identité.**
- Huit graines à magnitude 1,0 : `std ∈ [0,221 ; 0,409]`, `min = −0,995`,
  `max = 0,987`, **zéro valeur non finie**. Bornées, non constantes.
- Diversité par l'outil du dépôt (`tools/sample_diversity.py`) :
  `inter_seed_std = 0,2217` — **au-dessus du seuil de 0,2 de la spec**.

> **Réserve honnête sur le critère.** 0,2217 passe de 11 %. Et `CLAUDE.md`
> avertit que `inter_seed_std` « n'est pas un critère de diversité lisible
> seul » : il est proportionnel à `--magnitude`. Les deux repères qu'il demande
> de lire avec sont bons — `intra_image_std = 0,152`, sous le plafond 0,206
> comme voulu, et `banding_ratio = 1,12`, isotrope (pas de bandes horizontales
> au sens d'`ANISOTROPY_HUNT.md`). Le verdict est donc PASS, mais la marge sur
> le chiffre littéral de l'ordre de mission est mince.

---

## 5. P4 — la référence f64 (le cœur)

`blind_tests_attention/reference.py` : la formule
`y = x + W_o · softmax(qᵀk/√d) · v`, réécrite en f64 depuis la seule spec —
quatre projections avec biais, softmax recentrée par le max.

Comparaison à la sortie brute lue par la sonde :

| forme | positions | C | écart relatif max | non-uniformité de la softmax |
|---|---:|---:|---:|---:|
| 8×8×3 | 64 | 3 | 8,1·10⁻⁷ | `p_max/p_min` = 95 |
| **9×8×3** | **72** | 3 | 5,4·10⁻⁷ | 2,6·10⁷ |
| 16×16×3 | 256 | 3 | 1,4·10⁻⁶ | 1,9·10⁶ |
| 8×8×8 | 64 | **8** | 3,4·10⁻⁶ | 5,0·10¹⁰ |

Le 9×8 = 72 positions dépasse les 64 threads d'un workgroup (la boucle à pas de
`ATTENTION.md` §4.1) ; le 16×16 la dépasse de 4×. Le cas **C = 8** vérifie en
plus que le facteur d'échelle suit bien `d = C` et n'est pas figé à 192.

**Contrôle de sensibilité** — la référence doit *refuser* les variantes fautives
de la même formule. Sur le 8×8, écart relatif de chacune :

| variante | écart |
|---|---:|
| **la formule de la spec** | **8,1·10⁻⁷** |
| Q/K échangés | 2,0 |
| facteur `1/√d` omis | 5,6·10⁻¹ |
| softmax sur le mauvais axe | 2,9·10⁻¹ |
| résiduel absent | 5,3·10¹ |
| biais `b_o` absent | 6,2·10⁻¹ |
| `W_o` transposé | 1,5 |
| `W_v` transposé | 1,5 |
| Q/K échangés **et** axe de softmax inversé | 2,4 |

Cinq à sept ordres de grandeur séparent la bonne formule de la plus proche
fausse. La dernière ligne compte : c'était la seule dégénérescence plausible
(échanger les rôles Q/K *et* l'axe de la softmax se compensent presque). Elle ne
passe pas — donc **l'ordre d'empaquetage Q, K, V, W_o et l'orientation
`[sortie, entrée]` des quatre matrices sont arbitrés de l'extérieur**, pas
supposés. C'est ce qui rend le checkpoint interopérable pour un tiers.

---

## 6. P5 — anti-fuite inter-échantillons

**Ce qui a été cherché avant de conclure.** `--headless-sample --paths N` n'est
**pas** un axe de batch : le temps mural est linéaire en N (0,83 / 3,15 / 12,2 /
55,2 s pour N = 1 / 4 / 16 / 64 sur le XL) — les chemins sont déroulés en
séquence. `--headless-perpetual` travaille sur un seul latent. Le seul axe de
batch atteignable de l'extérieur est `--headless-train --batch B`.

**La prise, et pourquoi elle vise juste.** À l'init `W_o = 0`, donc rien en aval
de la couche ne dépend de son contexte : une fuite y serait invisible. Mais le
gradient de `W_o` vaut `Σ_n (∂L/∂y_n) ⊗ ctx_n`, et **`ctx` EST le tampon où un
échantillon lirait les k/v d'un autre**. Un pas d'optimiseur écrit ce gradient
dans le checkpoint : `ctx` devient observable.

Trois faits mesurés en chemin, tous nécessaires pour que le test soit le bon :

- un pas ne modifie **que** `W_o` — Q, K et V restent bit à bit à leur init.
  C'est la conséquence exacte de la formule (tout le chemin passe par `W_o`), et
  le moteur s'y conforme ;
- le pas est **déterministe** (même dataset ⇒ même `W_o`, bit à bit) ;
- **chaque slot du batch tire son propre `t` et son propre bruit** :
  `f(x,y) ≠ f(y,x)` de 24 % en relatif. Une décomposition naïve
  `f(x,y) = ½(f(x,x)+f(y,y))` serait donc **fausse pour de bonnes raisons** — et
  c'est exactement le piège dans lequel une première version de ce test est
  tombée avant d'être corrigée.

La forme immunisée à ce tirage par slot est l'**additivité croisée** :

```
f(x,y) + f(z,w) == f(x,w) + f(z,y)
```

| configuration | résidu relatif | sensibilité du slot le plus discret | rapport |
|---|---:|---:|---:|
| batch 2 | 5,5·10⁻⁷ | 0,036 | 66 000× |
| batch 4 | 5,3·10⁻⁸ | 0,075 | — |
| batch 1024 | 1,4·10⁻⁸ | 1,8·10⁻⁴ | 12 000× |

Le rapport signal/résidu est la mesure du mordant : le contenu d'un slot pèse
quatre à cinq ordres de grandeur plus lourd que le résidu d'additivité. Une
fuite du type « tout le monde lit les k/v de l'échantillon 0 » produirait un
terme croisé du même ordre que le signal. **Il n'y en a pas.**

> **NON OBSERVABLE — la moitié backward.** Ce chemin ne teste la fuite que dans
> le `ctx` du forward, à `W_o = 0`. Une fuite dans les passes backward de q/k/v
> reste hors de portée : leurs gradients sont exactement nuls tant que
> `W_o = 0`, et **aucune entrée headless ne sait charger des poids entraînés
> dans un entraînement batché** (D5). C'est une limite de la surface d'entrée du
> binaire, pas une lacune du test. La couverture reste assurée par
> `a_batched_pass_equals_the_same_samples_run_separately` et
> `one_samples_data_cannot_reach_another_samples_output` de la suite co-écrite —
> que ce rapport ne remplace pas sur ce point.

**Bonus hors mission : la grille au-delà de 65 535 workgroups.** Au goulot 8×8
(seq = 64), la frontière de `ATTENTION.md` §4.6 tombe à batch 1 024. Franchie
pour de vrai, des deux côtés : batch 1 023 / 1 024 / 1 025, soit 65 472 / 65 536
/ 65 600 workgroups par passe de ligne. Résultat **déterministe bit à bit** aux
trois tailles (une course entre deux workgroups sur la même ligne — le mode
d'échec décrit par la spec — bougerait d'un run à l'autre), et l'additivité
tient encore à 65 536 workgroups, le premier **et** le dernier échantillon
pesant chacun sur `W_o`.

**Bonus hors mission : la softmax ne déborde pas.** Logits `|qᵀk/√d|` poussés à
451 puis à 4,5·10⁴ — bien au-delà de `ln(f32_max) ≈ 88`, là où un `exp` non
recentré rend `inf` puis `NaN`. Sortie finie, et conforme à la référence f64 à
8,9·10⁻⁶ puis 3,4·10⁻⁷.

---

## 7. Défauts consignés

Aucun n'est un bug de calcul de l'attention. Les quatre premiers sont des
défauts d'**observabilité** — la spec affirme des propriétés que sa propre
surface d'entrée rend invérifiables de l'extérieur. Le cinquième est un écart de
contrat CLI.

**D1 — `--headless-train --steps 0` exécute un pas d'optimiseur.**
`--steps 0` et `--steps 1` produisent le même checkpoint (une ligne `step 0`,
`|W_o|max = 1,37·10⁻⁷`) ; `--steps 2` en fait deux. Le compte réel est
`max(1, N)`. Ça compte ici : « 0 pas » est la façon naturelle de figer une
initialisation, et un checkpoint pris ainsi est **déjà un pas SGD après** l'init
— `W_o` n'y est plus exactement nul, ce qui contredirait la phrase de
`ATTENTION.md` §1 « au pas 0 la couche est l'identité exacte » pour quiconque la
vérifie de l'extérieur. Contournement : `--lr 0`. C'est ce que fait cette suite.

**D2 — aucun chemin headless n'échantillonne un modèle neuf.**
`--headless-sample` refuse sans `--checkpoint` (« `--checkpoint <path to .ckpt>`
is required »). La recette littérale de l'ordre de mission — « poids frais même
seed, compare la génération » — n'est donc pas exécutable telle quelle : il faut
matérialiser l'init en passant par `--headless-train`.

**D3 — `--paths N` n'est pas un axe de batch.** Le temps mural est linéaire en
N. L'ordre de mission espérait « un chemin observable » à batch > 1 en
inférence ; il n'y en a pas.

**D4 — il n'existe pas de commande « charger puis resauver ».**
`GO_NOGO.md` §5 range « Round-trip de checkpoint sur les 4 projections » parmi
les preuves, mais elle n'est établissable qu'en processus. De l'extérieur, le
round-trip est fait ici **à travers le conteneur publié**, avec le codec écrit en
boîte noire : relire les 14,3 Mo, les réécrire, puis dépaqueter/repaqueter le
tenseur d'attention par le chemin `(4, C, C)` — la génération ne bouge pas d'un
octet, et le contrôle « Q et K permutés » la fait bien changer. C'est une preuve
du **format**, à laquelle il manque le chemin de chargement du moteur.

**D5 — la spec se prive elle-même de l'anti-fuite backward.**
`--headless-train` force `load_checkpoint = false`. Le seul exercice batché de la
couche tourne donc toujours à `W_o = 0`, c'est-à-dire avec la couche en identité
et les gradients de q/k/v exactement nuls. Voir §6.

**Non-défaut, à noter tout de même** : le seuil `inter_seed_std > 0,2` de
l'ordre de mission est franchi de 11 % seulement, et `CLAUDE.md` déconseille de
lire cet indicateur seul (§4).

---

## 8. Ce que cette suite ne couvre pas

- **Les différences finies** sur Q, K, V, `W_o`, les quatre biais et l'entrée
  (`ATTENTION.md` §4.2) : hors de portée depuis l'extérieur, pour la raison D5.
  Seul le gradient de `W_o` à l'init est observable, et il l'est ici.
- **La limite de 8 storage buffers par bind group** (`ATTENTION.md` §3) : c'est
  une propriété de construction, sans signature observable *tant que la limite
  est respectée*. Si elle était franchie, le mode d'échec (loss et gradients
  exactement nuls) serait immédiatement visible — et il ne l'est pas : la loss
  descend, `W_o` reçoit un gradient non nul, `Q/K/V` restent intacts pour la
  raison mathématique attendue.
- **Le multi-head** : non implémenté, donc rien à tester.
- **Les mesures de performance** de `GO_NOGO.md` §3 (la falaise à batch 64) : la
  mission ne les demandait pas et je ne les ai pas rejouées.

Les deux suites — celle co-écrite avec l'implémentation et celle-ci — sont donc
**complémentaires, pas redondantes** : le backward est couvert par la première,
le format de checkpoint et le comportement bout en bout du binaire par la
seconde. Sur tout ce qu'elles couvrent en commun, **elles sont d'accord**.

---

## 9. Reproduire

```bash
./blind_tests_attention/run.sh              # les 7 volets
./blind_tests_attention/run.sh t4 t6        # une sélection
```

Le runner construit `-p batlab` puis déroule tout dans une racine de stockage
jetable (`BATLAB_ROOT=blind_tests_attention/work/`, gitignorée) : le `Models/`
du dépôt n'est jamais touché, `git status` reste propre. Le checkpoint entraîné
est copié depuis `.checkpoint_backup/` du dépôt principal et n'est pas commité.

Durée : ~1 min sur GPU libre, en partant d'une arborescence vide (les volets
t5 et t7 dominent, jusqu'à batch 1 025).

| fichier | rôle |
|---|---|
| `ckpt.py` | codec du conteneur `.ckpt` (déduit en boîte noire) |
| `dump.py` | lecteur du dump `BATFLUX1` (format publié par `--help`) |
| `batraw.py` | écriture de datasets jouets |
| `reference.py` | l'attention en f64, écrite depuis la seule formule |
| `harness.py` | la sonde : calibration affine + modèle jouet + lecture de `ε̂` |
| `t1_identity.py` | P1 — identité au départ |
| `t2_trained.py` | P2 — checkpoint entraîné, génération non dégénérée |
| `t3_determinism.py` | P3 — déterminisme et round-trip |
| `t4_reference.py` | P4 — référence f64 + contrôle de sensibilité |
| `t5_batch.py` | P5 — anti-fuite inter-échantillons |
| `t6_softmax.py` | stabilité numérique de la softmax |
| `t7_grid.py` | grille de dispatch au-delà de 65 535 workgroups |
