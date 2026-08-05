# Le mode Perpetual, poli — rapport de mission

Branche : `perpetual-smoothness` · Plateforme : macOS (Darwin 25.4.0, Apple M5 Pro) · wgpu 28

## Résumé

Deux retours d'usage, tous deux trouvés en **regardant tourner le mode**, pas en
relisant le code — la même provenance que les deux bugs de `PERPETUAL_INFERENCE.md` :

1. **la fenêtre déformait l'image** — elle s'ouvrait carrée (512×512) sur une frame
   de 67×32, soit des pixels étirés 1:2, et tout redimensionnement rejouait la
   déformation ;
2. **le rebruitage faisait un saut violent** — la remontée `t=0 → t_r` tenait en une
   frame, « une quantité de bruit immense » d'un coup, ce qui casse la contemplation.

| | Commit | |
|---|---|---|
| 1 | `5bc7fa6` | la fenêtre s'ouvre et se redimensionne au ratio de la frame |
| 2 | `139fab9` | `forward_step` : le processus avant, un cran à la fois |
| 3 | `d6c7ce9` | `PerpetualDrift` gagne une phase « montée » |
| 4 | `dcae578` | la montée câblée : une frame par incrément, la phase affichée |
| 5 | `256e805` | les bandes du letterbox sortaient en gris moyen (piège sRGB) |

`cargo test --workspace` : **121 verts** dans `bat_building` (109 avant la mission)
+ 9 dans `main`, 5 ignorés. **14 tests ajoutés, 2 remplacés par des versions plus
fortes, aucun affaibli.**

---

## 1. La fenêtre déformait l'image

### 1.1 Ce que la mesure dit

Taille relevée sur la vraie fenêtre (CoreGraphics `CGWindowListCopyWindowInfo`,
pas une estimation depuis le code) :

| | Taille fenêtre | Zone de rendu | Échelle x / y | Pixel source |
|---|---|---|---|---|
| avant | 512×544 | 512×512 | ×7,64 / ×16,0 | rectangle **1:2,09** |
| après | 536×288 | **536×256** | ×8 / ×8 | **carré** |

`536 × 256 = 67 × 32 × 8` : la fenêtre s'ouvre au plus grand **multiple entier** de
la géométrie de la frame qui tienne dans 560 px de côté. La géométrie vient de la
source, donc le correctif vaut pour **toutes** les vues live, l'entraînement compris
— il n'y a qu'un seul chemin d'ouverture de fenêtre.

`perpetual_samples/fenetre_avant_apres.png` — à gauche l'ancienne fenêtre (les
pixels sont visiblement deux fois plus hauts que larges), au centre la nouvelle à
l'ouverture, à droite la même redimensionnée à 820×620.

### 1.2 Le redimensionnement : letterbox, pas étirement

Le shader dessine un quad plein écran ; rien ne posait de viewport, donc la frame
prenait la forme de la fenêtre. Le rendu passe désormais par
`set_viewport(letterbox_viewport(surface, frame))` : même échelle sur les deux axes,
centré, le reste laissé à la couleur d'effacement.

**Le shader n'a pas bougé.** Letterboxer dans le viewport plutôt que dans les UV le
garde ce qu'il est — un indexeur row-major qui ne sait rien de l'aspect ; on lui
demande simplement moins de pixels. Deux fonctions pures portent toute la
géométrie et sont testées sans GPU :

| Test | Ce qu'il épingle |
|---|---|
| `the_window_opens_at_an_integer_multiple_of_the_frame` | 67×32 → 536×256, et **les deux axes à la même échelle** — c'est la déformation elle-même |
| `the_opening_size_stays_within_the_target_edge_for_any_frame` | frames dégénérées (0×0, 700×3, 4×900) : jamais de fenêtre vide ni géante |
| `a_matching_surface_is_filled_edge_to_edge` | une fenêtre déjà au bon ratio n'a **pas** de bandes (letterboxer là serait une régression de son propre chef) |
| `resizing_letterboxes_instead_of_stretching` | ratio conservé à 1e-4 près, bandes symétriques, sur une fenêtre trop haute puis trop large |
| `the_viewport_never_leaves_the_surface` | wgpu refuse un viewport qui dépasse ; l'arithmétique tient sur les formes que personne ne choisirait |

### 1.3 Un piège trouvé sur capture d'écran, pas dans le code (`256e805`)

La couleur des bandes était réglée à `0.08` — un neutre sombre, sur le papier. À
l'écran elle sortait à **sRGB 80/255**, un gris moyen qui encadrait l'image comme
un passe-partout. La surface est sRGB et la couleur d'effacement **ne passe pas par
la fonction de transfert** : elle est linéaire.

Corrigée à `0.0115` linéaire, **mesurée sur la capture** : `(28, 28, 31)`. C'est
exactement le genre d'écart qu'aucun test unitaire n'aurait vu — la valeur est bien
celle qu'on a écrite, c'est sa signification qui n'était pas celle qu'on croyait.

---

## 2. Le rebruitage sautait

### 2.1 Marcher la chaîne avant, plutôt que de la sauter

`add_noise(x₀, t)` implémente `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε` : une destination, une
frame. `forward_step(x_{k−1}, k)` implémente un **incrément** de la même chaîne,
`x_k = √(1−β_k)·x_{k−1} + √β_k·ε_k`. Chaîner les incréments `0..=t` depuis le même
x₀ mène à la même distribution — c'est l'identité DDPM standard.

Les deux sont donc interchangeables en mathématiques et **pas du tout à l'usage** :
le saut remplace l'image par du bruit entre deux frames, la marche la dissout sur
`t_r` d'entre elles.

### 2.2 Ce que ça a forcé à revoir : d'où part la montée

Le premier réflexe — fractionner l'ancien `add_noise(x̂₀, t_r)` — aurait fait
repartir chaque montée du **x̂₀ propre**, à l'incrément `k = 0`. En errance c'est
sans conséquence (à `t = 0` la sortie du sampler *est* son x̂₀ clippé, cf.
`x0_estimate_is_the_x0_the_reverse_step_uses`). En **respiration**, dont le plancher
est `t_r/2`, ça aurait fait **clignoter l'image résolue** au début de chaque
montée — exactement ce que le régime promet de ne jamais faire.

La montée part donc du **latent**, là où la descente l'a laissé, et court de
`plancher` à `t_r`. C'est précisément ce que l'expression en incréments autorise :
`x_k = √α_k·x_{k−1} + √β_k·ε` est valide depuis **n'importe quel** `x_{k−1}`
légitime, par la propriété de Markov du processus avant, là où `add_noise` suppose
un x₀ propre. L'avertissement de l'en-tête du module (« ne jamais rebruiter le
latent brut ») visait `add_noise` ; il ne s'applique pas à un incrément, et le
lever rend la respiration correcte au lieu de la contraindre.

Effet de bord heureux : les deux régimes deviennent des **triangles** et non des
dents de scie — la montée dure exactement autant de frames que la descente qu'elle
défait (`the_climb_is_exactly_as_long_as_the_descent_it_undoes`).

```
        AVANT (dent de scie)                APRES (triangle)
  t                                    t
t_r│╲    ╲    ╲    ╲                 t_r│╲  ╱╲  ╱╲  ╱╲  ╱
   │ ╲    ╲    ╲    ╲                   │ ╲╱  ╲╱  ╲╱  ╲╱
  0└──╲────╲────╲────╲──▶             0 └──────────────────▶
     le retour tient en 1 frame          le retour dure t_r frames
```

### 2.3 Le test d'équivalence distributionnelle

`climbing_the_forward_chain_matches_add_noise_in_distribution` **mesure**
l'équivalence au lieu de la déduire de l'algèbre — la revendication porte sur les
tirages réels de `gaussian_at`, et une chaîne qui réutiliserait une graine
satisferait l'algèbre et échouerait ici.

Sur 16 384 éléments, x₀ fixe, à t = 8 / 32 / 64 : coefficient de signal par moindres
carrés, moyenne et écart-type du résidu, comparés à la théorie **et** l'un à
l'autre.

| t = 64 | σ du bruit | signal (coef. de x₀) |
|---|---|---|
| montée (65 incréments) | 0,69782 | 0,71611 |
| saut (`add_noise`) | 0,69600 | 0,70313 |
| théorie `√(1−ᾱ₆₄)` / `√ᾱ₆₄` | 0,69654 | — |

Écart montée/saut sur σ : **0,26 %**, sous l'erreur d'échantillonnage (0,55 % à
N = 16 384).

Le test vérifie aussi **chaque niveau intermédiaire** — une montée dont les frames
du milieu ne seraient pas de vrais `x_k` montrerait un bon départ, une bonne
arrivée, et n'importe quoi entre les deux, c'est-à-dire exactement ce que
l'utilisateur regarde.

**Vérifié par mutation, pas seulement vu vert :**

| Mutation injectée | Résultat |
|---|---|
| une seule graine pour toute la montée (le même champ ajouté en boucle) | **ROUGE** |
| `alpha_bar` au lieu d'`alpha` dans l'incrément (le saut déguisé) | **ROUGE**, dès k=9 |
| graine de montée indépendante du cycle (rebruitage à l'identique) | **2 ROUGES** |
| graine de montée indépendante de l'incrément | **ROUGE** |

### 2.4 L'anti-boucle-figée

La crainte exprimée : un rebruitage qui reviendrait identique ferait tourner
l'image en rond. Les graines sont fraîches par **(cycle, incrément)** — par cycle
pour que la pièce n'ait pas de période, par incrément pour que la montée soit une
marche gaussienne et non un champ appliqué `t_r` fois.

Des graines qui diffèrent ne suffisent pas : ce qui compte est que les **champs
qu'elles produisent** diffèrent.
`two_consecutive_cycles_dissolve_the_image_into_different_noise` fait donc tourner
deux montées consécutives sur la **même** image, à travers le vrai schedule, et
corrèle les bruits réellement injectés : **|r| < 0,05** exigé, `r = 1,000` sous la
mutation « graine indépendante du cycle ».

### 2.5 La planche de remontée, et le tempo

`--headless-perpetual … --climb-frames 8` écrit les frames composées de la montée —
seul moyen de la voir sans fenêtre. `perpetual_samples/montee_planche.png`
(t_r = 64, une colonne toutes les 9 marches) :

| frame | écart-type de x_t | corr(x_t, x̂₀) |
|---|---|---|
| t=0 (image résolue) | 0,344 | 1,000 |
| t=9 | 0,364 | 0,927 |
| t=18 | 0,404 | 0,805 |
| t=27 | 0,453 | 0,672 |
| t=36 | 0,492 | 0,558 |
| t=45 | 0,553 | 0,434 |
| t=54 | 0,595 | 0,338 |
| t=63 | 0,626 | 0,299 |

Monotone dans les deux colonnes : l'image **se dissout**, elle n'est pas remplacée.
Avant, ces huit colonnes étaient une seule transition.

**Tempo : `CLIMB_TEMPO_RATIO = 1.0`** — la montée dure exactement aussi longtemps
que la descente. Choix de la symétrie parfaite, et non d'un compromis : un
incrément de montée est de l'**arithmétique pure** (aucun appel au modèle), donc la
symétrie ne coûte rien en calcul, et une remontée plus rapide que la descente se
lit encore comme un saut, seulement plus court. Un cycle `t_r = 64` à 30 pas/s
dure 4,3 s au lieu de 2,2 s. La constante est publique et documentée : la monter
presse la dissolution, la baisser l'étire.

### 2.6 L'interface dit la phase

Une image qui se défait à l'écran est le comportement nominal **la moitié du
temps** ; non étiquetée, elle se lit comme une panne — même leçon que le découpage
en deux panneaux de `PERPETUAL_INFERENCE.md` §2. Le panneau porte une ligne
`phase : descente | remontée`, le pied de page aussi, et une ligne de légende dit
que x̂₀ reste figé pendant la montée (le modèle ne prédit pas — inventer un panneau
droit serait un mensonge).

---

## 3. Validation sur le vrai TUI

TUI réel lancé dans une fenêtre tmux dédiée (`batLab-smoothness-e2e`) et piloté par
`tmux send-keys`, modèle chargé **par l'interface** (`Home` → `Load Saved Model` →
`Greyscale_Diffusion_L` → `night_run.ckpt` → `Perpetual`). Les deux binaires — celui
d'avant la mission et celui d'après — ont fait le même parcours, dans la foulée.

### 3.1 La fenêtre

| Vérification | Résultat |
|---|---|
| Taille à l'ouverture (CoreGraphics) | **512×544 → 536×288** (rendu 512×512 → 536×256) |
| Rapport d'aspect à l'ouverture | 1,00 → **2,094** (= 67/32) |
| Redimensionnement 820×620 | bandes haut/bas, image centrée, **pixels carrés** |
| Redimensionnement 1132×300 | bandes gauche/droite, idem |
| Couleur des bandes (mesurée sur capture) | sRGB **(28, 28, 31)** |
| `[v]` off → on | la fenêtre se masque et revient **à la taille que l'utilisateur lui a donnée** |

### 3.2 La remontée, en direct

Relevé du pied de page toutes les 0,9 s, errance, `t_r = 64`, 30 pas/s :

```
descente t=47 · descente t=19 · remontée t=5  · remontée t=34 · remontée t=61
descente t=39 · descente t=11 · remontée t=14 · remontée t=40 · descente t=61
```

Le triangle est visible et l'étiquette suit le sens de marche. Cycle mesuré ≈ 4,3 s,
soit les 2 × 65 pas attendus à 30 pas/s. **Cadence tenue pendant la montée**
(`29,6–30,1 / 30 pas/s`) : l'arithmétique de la montée ne coûte rien.

Respiration (`[m]`), même `t_r` : `t` relevé sur 10 échantillons dans **[35, 64]** —
jamais sous le plancher 32, **ni en descente ni en montée**. C'est le point de
conception de §2.2 vérifié sur le vrai run.

### 3.3 Le reste des contrôles

| Touche | Observé |
|---|---|
| `espace` | `en pause`, **t figé à 21** sur 4 s, puis reprise |
| `m` | errance ↔ respiration, plancher respecté |
| `s` | `✓ image → perpetual_samples/errance_c024_000.png` |
| `r` | `cycle 24 → 0`, repart du haut du schedule |
| `v` on → off → on | bascule, run continu |
| `q` | **`EXITED=0`** |

**stderr sur tout le run : 0 octet** (ancien binaire comme nouveau).

---

## 4. La dérive n'a pas changé de caractère

L'errance doit toujours *dériver* : ce serait un mauvais échange que de gagner la
douceur en perdant le mouvement. Écart absolu moyen entre images de cycles
consécutifs, 5 graines × 15 cycles, `t_r = 64`, mêmes poids :

| graine | saut (avant) | montée (après) |
|---|---|---|
| 12345 | 0,1360 | 0,1489 |
| 777 | 0,1177 | 0,1308 |
| 4242 | 0,1182 | 0,1238 |
| 99 | 0,1366 | 0,1402 |
| 31337 | 0,0876 | 0,1145 |
| **moyenne** | **0,1192** | **0,1317** |

Même ordre de grandeur, et dans la fourchette 0,121–0,152 relevée en `4082bc0`. La
montée sort systématiquement un peu au-dessus (5 fois sur 5), mais l'écart
(0,012) reste **sous la dispersion inter-graines** (0,020 / 0,014) : les deux bras
partagent la première descente puis divergent dans des images entièrement
différentes, donc l'appariement par graine ne vaut que pour le premier cycle. Il
n'y a pas de quoi conclure à un effet — et il n'y a pas non plus de mécanisme :
l'équivalence à destination est mesurée à 0,26 % près (§2.3).

---

## 5. Hygiène et limites

- Le `config_file` du modèle réécrit par le passage du TUI → restauré par
  `git checkout`. Le PNG écrit par `[s]` pendant le test → supprimé.
- `night_run.ckpt` (5,5 Mo) copié depuis le dépôt principal, laissé **non commité**
  (`Models/` n'est pas gitignoré).
- Fenêtre tmux tuée en fin de mission.
- **`cargo test` salit toujours `Models/Stable_Diffusion/config_file`** — limite
  connue et non corrigée, documentée en `PERPETUAL_INFERENCE.md` §5. Vérifier
  `git status` après un `cargo test`.
- L'échelle d'affichage reste fixe (`[-1, 1] → [0, 1]`, `INFER_VIZ.md` §6) : aux t
  élevés le panneau gauche sature. La montée le rend plus visible qu'avant puisque
  la saturation arrive désormais progressivement — c'est du comportement rendu
  lisible, pas une régression.
- Le letterbox autorise une échelle **non entière** au redimensionnement (le ratio
  est exact, mais certaines lignes/colonnes sources occupent un pixel écran de plus
  que leurs voisines, comme tout zoom au plus proche voisin). Ne verrouiller le
  redimensionnement sur des multiples entiers rendrait la fenêtre récalcitrante à
  la souris ; l'ouverture, elle, est exacte.
