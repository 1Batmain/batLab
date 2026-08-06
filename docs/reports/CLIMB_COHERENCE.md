# La remontée cesse de friser — rapport de mission

Branche : `climb-coherence` · Plateforme : macOS (Darwin 25.4.0, Apple M5 Pro) · wgpu 28

## Résumé

Un retour d'usage sur le mode Perpetual : l'effet plaît, **« ça frise pendant la
phase de remontée »**. Diagnostic de l'architecte **confirmé, et mesuré** : la
montée était la marche de Markov exacte, un `forward_step` par frame, donc un
champ de bruit **indépendant à chaque frame**. À 30 frames/s, trente grains sans
rapport par seconde.

| | Commit | |
|---|---|---|
| 1 | `d4ad824` | `forward_from` : la forme fermée du processus avant depuis un état intermédiaire |
| 2 | `defabe2` | la montée y passe — un seul champ de grain par cycle |

`cargo test --workspace` : **123 verts** dans `bat_building` (121 avant) + 9 dans
`main`, 5 ignorés. **2 tests ajoutés, 2 reformulés, aucun affaibli.**

Le chiffre qui dit que le frisage est mort — corrélation entre le changement
d'une frame à la suivante, sur les frames composées du vrai modèle :

| | avant | après |
|---|---|---|
| **corr(Δ frame k, Δ frame k+1)** | **−0,001** | **+0,842** |

---

## 1. Le diagnostic, confirmé

### 1.1 Pourquoi la descente est fluide et la montée crépitait

Les deux phases se ressemblent — un pas de chaîne par frame — et ne se
comportaient pas pareil, pour une raison qui n'est pas dans le pas mais dans sa
**corrélation avec le précédent** :

- **Descente** : `x_{t−1}` est la moyenne postérieure de `x_t` plus `σ_t·ε`, avec
  `σ_t = sqrt(β_t)` petit. Deux latents successifs sont donc fortement corrélés :
  l'écran montre une image qui bouge.
- **Montée (avant)** : `x_k = sqrt(α_k)·x_{k−1} + sqrt(β_k)·ε_k` avec **`ε_k`
  frais à chaque incrément**. Le terme de signal est presque l'identité
  (`sqrt(α_k) ≈ 0,996`), donc le changement image-à-image *est* `sqrt(β_k)·ε_k` :
  un champ blanc, entièrement nouveau, trente fois par seconde. De la neige
  télévisée posée sur une image qui se dissout.

Mesuré sur les frames du vrai modèle (§4) : la corrélation entre deux
changements consécutifs vaut **−0,001**. C'est la définition d'un scintillement.

### 1.2 Ce qui n'était pas en cause

L'ancienne montée n'avait **aucun défaut distributionnel** — la marche est le
processus avant exact, et `climbing_the_forward_chain_matches_add_noise_in_distribution`
le mesurait déjà. Le profil de dissolution est d'ailleurs identique avant/après
(§4.3). Le défaut ne portait pas sur *ce que chaque frame est*, mais sur *ce que
deux frames voisines ont en commun* — une propriété du **joint**, qu'aucun test
de marginale ne peut voir. C'est exactement ce que la mutation M5 démontre (§3).

---

## 2. Le correctif : la forme fermée, un champ par cycle

### 2.1 L'expression

```text
   x_t = sqrt(ᾱ_t / ᾱ_dep) · x_dep  +  sqrt(1 − ᾱ_t / ᾱ_dep) · ε_cycle
```

`LinearNoiseSchedule::forward_from(x_dep, dep, t, seed)`. C'est `q(x_t | x_s)`
pour **n'importe quel** `s < t` — le ratio d'ᾱ *est* la généralisation. Deux
conséquences, et ce sont les deux dont la montée avait besoin :

1. **Le départ a le droit d'être un latent bruité.** `add_noise` suppose un x₀
   propre ; le ratio corrige exactement le bruit que le départ porte déjà.
   C'est la raison d'être de la forme, et c'est ce qui fait marcher la
   respiration, dont le plancher est `t_r/2` (§4.4).
2. **Chaque niveau est calculé depuis le départ, pas depuis la frame
   précédente.** Un seul `ε_cycle`, tiré une fois par cycle. Le même motif de
   grain monte en amplitude au lieu d'être rebattu.

À `dep = 0` le dénominateur vaut 1 et l'expression se réduit exactement à
`add_noise` — l'errance est donc le cas particulier propre, la respiration le cas
général.

### 2.2 Ce que ça change et ce que ça ne change pas

| | avant (Markov) | après (forme fermée) |
|---|---|---|
| marginale de chaque frame | `q(x_t \| x_dep)` | **identique** (§3.1) |
| loi du point d'arrivée | `q(x_{t_r} \| x_dep)` | **identique** (§3.1) |
| corr(Δ_k, Δ_{k+1}) | 0 | **1** |
| champ par cycle | `t_r + 1` indépendants | **1** |
| unicité du champ d'un cycle à l'autre | oui | **oui** (inchangé) |

Les frames ne forment plus une chaîne de Markov : elles sont **maximalement
corrélées**. C'est précisément la propriété que l'œil réclamait, et elle est
gratuite — le joint n'était contraint par rien.

### 2.3 Pourquoi `forward_step` reste dans le code

Il n'est pas faux, il est *inadapté à une animation*. Il reste parce qu'il est la
définition correcte et testée d'un incrément du processus avant, et surtout
parce qu'il est **la référence contre laquelle `forward_from` est mesuré** : le
test de marginales lance les deux depuis le même départ et exige qu'ils
remettent au sampler la même distribution (§3.1). Supprimer la marche aurait
supprimé l'oracle.

### 2.4 Le câblage

`DriftAction::Climb` porte désormais `departure_step` (le niveau d'où la montée
part, qui fixe le ratio d'ᾱ) et `cycle_seed` (le champ unique du cycle, à la
place d'une graine par incrément). L'appelant photographie le latent sur
l'incrément qui annonce `opens_cycle` et alimente **tous** les niveaux avec cette
même photo. `PerpetualDrift` ne détient toujours ni modèle, ni GPU, ni tenseur.

---

## 3. Les preuves

### 3.1 Marginale exacte à chaque niveau, et point d'arrivée inchangé

`the_closed_form_climb_is_a_valid_x_t_at_every_level` — coefficient de signal par
moindres carrés, moyenne et écart-type du résidu, **à chaque niveau de la
montée**, depuis un x₀ propre *et* depuis un latent (`dep = 24`, le cas de la
respiration). Comparés à la théorie **et** à la marche de Markov exacte lancée du
même départ.

Une montée dont seuls le départ et l'arrivée seraient justes montrerait n'importe
quoi entre les deux — c'est-à-dire exactement la partie que l'utilisateur
regarde.

`N = 65 536` : l'erreur-type de l'estimateur de signal vaut ≈ 0,003, donc la
tolérance de 0,02 est un énoncé à ~6 σ. La version à `N = 16 384` était à ~3 σ —
verte, mais à une réalisation malchanceuse près de ne rien prouver ; c'est la
mutation M5 qui l'a exhibé, en la faisant tomber pour de mauvaises raisons.

L'égalité au point d'arrivée est la propriété n° 2 de la mission : **la descente
suivante reçoit un `x_{t_r}` statistiquement identique à avant, donc l'errance ne
change pas de nature.**

### 3.2 Le chiffre de cohérence, en unité

`consecutive_climb_frames_change_by_the_same_grain` — corrélation entre le
changement d'une frame et celui de la suivante, moyennée sur la montée, calculée
**identiquement sur les deux branches** pour que la comparaison soit honnête.

| | forme fermée | marche de Markov |
|---|---|---|
| corr(Δ_k, Δ_{k+1}), f32 brut | **1,0000** | **−0,0037** |
| la même, après quantification 8 bits | 0,8663 | −0,0119 |

Le test exige `> 0,95` d'un côté et `|r| < 0,15` de l'autre — la seconde
assertion garde la première : si la marche cessait d'être la branche incohérente,
le test ne comparerait plus ce qu'il prétend comparer.

La ligne « 8 bits » n'est pas décorative : elle explique le §4.1. Le changement
par frame de la forme fermée vaut ~1 niveau de gris, comparable au pas de
quantification, ce qui rabote mécaniquement la corrélation mesurée sur des PNG de
1,000 à ~0,87.

### 3.3 Anti-boucle-figée : conservé, reformulé

**Reformulation assumée.** `no_two_climb_increments_anywhere_draw_the_same_seed`
exigeait l'**exact contraire** de ce que la nouvelle montée fait *à l'intérieur*
d'un cycle : un champ frais par incrément. C'était juste quand la montée était
une marche — y réutiliser un champ aurait voulu dire ajouter le même champ `t_r`
fois, ce qui n'est pas une marche gaussienne. La propriété testée **change de
formulation avec la méthode**.

Ce qui **ne bouge pas**, et qui est la crainte réellement exprimée par
l'utilisateur (« un rebruitage qui reviendrait identique ferait tourner l'image
en rond ») : **l'unicité du champ d'un cycle à l'autre**. Elle est assertée aussi
strictement qu'avant.

`one_field_per_climb_and_a_new_one_every_cycle` :

- un seul champ par montée (les graines d'une montée sont toutes égales) ;
- un champ **différent** à chaque cycle, et disjoint des graines de descente ;
- le départ est fixe pendant toute la montée et vaut le **plancher** ;
- la montée visite tous les niveaux depuis son départ.

Lancé dans les **deux régimes** : en errance le départ vaut 0, où un
`departure_step` jamais mis à jour passerait inaperçu ; en respiration il vaut
`t_r/2`, où il ne passe pas (mutation M4).

`two_consecutive_cycles_dissolve_the_image_into_different_noise` garde son
assertion `|r| < 0,05` entre deux cycles consécutifs — reformulée sur la forme
fermée — et gagne sa face complémentaire : **`r > 0,95` le long d'une montée**.
Décorrélé d'un cycle à l'autre, corrélé le long d'un : la pièce dérive sans
crépiter.

### 3.4 Vérifié par mutation, pas seulement vu vert

| Mutation injectée | Résultat |
|---|---|
| M1 — le champ change à chaque niveau (retour au grain indépendant) | **2 ROUGES** |
| M2 — le champ ne dépend plus du cycle (rebruitage à l'identique) | **2 ROUGES** |
| M3 — `forward_from` oublie le dénominateur `ᾱ_dep` | **1 ROUGE** |
| M4 — le départ n'est jamais mis à jour (reste 0) | **1 ROUGE** (branche respiration) |
| M5 — `forward_from` tire un champ **par niveau** | **2 ROUGES — et les marginales restent VERTES** |

M5 est la mutation qui vaut le rapport. Elle produit exactement le bug d'origine
— marginales impeccables, joint détruit — et **seuls les deux tests de cohérence
la voient**. Toute la batterie distributionnelle, la précédente comme la nouvelle,
reste verte devant elle. C'est la preuve mécanique que le frisage était
invisible aux tests qui existaient, et qu'il ne l'est plus.

---

## 4. La mesure sur le vrai modèle

Checkpoint `Greyscale_Diffusion_L/night_run.ckpt`, `t_r = 64`, graine 12345,
errance. Binaire d'**avant** la mission bâti depuis `4a67679` dans un worktree
séparé, binaire d'**après** depuis `HEAD` : les deux ont écrit les 65 frames
composées de la même montée (`--climb-frames 65`).

Les frames mesurées **sont** le contenu de la fenêtre : le chemin headless et le
chemin live composent par la même fonction (`compose_frame`, appelée par
`compose_live_frame` et par `LiveFrame::publish`). Mesurer l'écran n'aurait
ajouté que le pipeline d'affichage — et, vérifié en le tentant, `screencapture`
étrangle le run à 2 incréments pour 20 captures, ce qui détruit la mesure qu'il
prétendait faire.

### 4.1 Le chiffre

| corr(Δ frame k, Δ frame k+1) | moyenne | médiane | min | max |
|---|---|---|---|---|
| **avant** (marche de Markov) | **−0,001** | −0,003 | −0,086 | +0,065 |
| **après** (forme fermée) | **+0,842** | +0,862 | +0,696 | +0,913 |

Changement moyen par frame : **9,02** niveaux de gris avant, **0,96** après. La
montée ne déplace plus l'image d'un dixième de sa dynamique entre deux frames :
elle la déplace d'un niveau de gris, toujours dans le même motif.

Le +0,842 est un **plancher** imposé par le PNG 8 bits, pas la valeur vraie :
sur les mêmes frames non quantifiées la corrélation vaut 1,0000, et la seule
quantification la ramène à 0,866 (§3.2). Le 0,842 mesuré s'en déduit à la texture
de l'image près.

### 4.2 Les planches

`climb_samples/montee_avant_apres.png` — huit points de la même montée
(`t = 0, 9, …, 63`), panneau x_t seul, avant en haut / après en bas. Les deux
rangées se dissolvent de la même façon : c'est ce que « marginale identique »
veut dire, en image.

`climb_samples/grain_avant_apres.png` — **le grain lui-même**, c'est-à-dire ce
que la montée ajoute entre deux frames, amplifié autour du gris moyen (fenêtre de
5 frames, pour passer au-dessus du pas de quantification). En haut : huit champs
sans rapport. En bas : **le même motif, huit fois**, dont l'amplitude monte.
Corrélation entre tuiles voisines de la planche : **−0,013 avant, +0,973 après**.
C'est le frisage, et son absence.

`climb_samples/fenetre_live_remontee.png` — la fenêtre live du vrai TUI en pleine
remontée (`t = 26`), capturée par identifiant de fenêtre : à gauche x_t qui se
dissout, à droite x̂₀ figé.

### 4.3 La dissolution n'a pas changé de profil

| t | écart-type x_t (avant / après) | corr(x_t, x̂₀) (avant / après) |
|---|---|---|
| 0 | 0,344 / 0,343 | +0,998 / +0,998 |
| 9 | 0,364 / 0,357 | +0,927 / +0,934 |
| 18 | 0,404 / 0,391 | +0,805 / +0,810 |
| 27 | 0,453 / 0,440 | +0,672 / +0,674 |
| 36 | 0,492 / 0,492 | +0,558 / +0,553 |
| 45 | 0,553 / 0,539 | +0,434 / +0,454 |
| 54 | 0,595 / 0,578 | +0,338 / +0,374 |
| 63 | 0,626 / 0,608 | +0,299 / +0,310 |

Monotone dans les deux colonnes, et superposable entre les deux branches. La
colonne « avant » reproduit à la troisième décimale la planche de
`PERPETUAL_SMOOTHNESS.md` §2.5 — ce qui valide au passage la chaîne de mesure
contre une mission antérieure.

### 4.4 Le TUI réel

Lancé dans une fenêtre tmux dédiée (`batLab-climb-e2e`) et piloté par
`tmux send-keys`, modèle chargé **par l'interface** (`Home` → `Load Saved Model`
→ `Greyscale_Diffusion_L` → `night_run.ckpt` → `Perpetual`).

| Vérification | Observé |
|---|---|
| Triangle descente/remontée, errance | `descente t=14 → remontée t=13 → remontée t=40 → descente t=60 → …` |
| Cadence tenue **pendant** la montée | 29,7–30,6 / 30 pas/s (la montée ne coûte aucun appel modèle) |
| Respiration, `t_r = 64`, plancher 32 | t relevé dans **[33, 60]** sur 10 échantillons — jamais sous le plancher, **ni en descente ni en montée** |
| `espace` | `t` figé à 63 sur 3 s, puis reprise |
| `r` | `cycle 58 → 0` |
| `s` | `✓ image → perpetual_samples/respiration_c000_000.png` |
| `v` off → on → off → on | la fenêtre se masque et revient, run continu |
| `q` | **`EXITED=0`** |

**stderr sur tout le run : 0 octet.**

Le point de §4.4 qui compte : en respiration, le départ de la montée est un
latent au plancher, jamais un x̂₀ propre — et `t` ne descend jamais sous 32, y
compris pendant les montées. C'est le ratio d'ᾱ qui rend ça correct plutôt que
seulement toléré.

---

## 5. La dérive : un écart mesuré, sans mécanisme

L'errance doit continuer de *dériver*. Écart absolu moyen entre images de cycles
consécutifs, **12 graines × 25 cycles**, `t_r = 64`, mêmes poids, protocole
identique sur les deux binaires :

| | moyenne | écart-type inter-graines |
|---|---|---|
| avant | 0,1231 | 0,0121 |
| après | 0,1169 | 0,0141 |

Différence appariée : **−0,0063** (se 0,0025, t = −2,49), après < avant sur 9
graines sur 12.

**Il n'y a pas de mécanisme, et il ne peut pas y en avoir** : conditionnellement
au départ, les deux branches tirent `x_{t_r}` de la *même* loi
`q(x_{t_r} | x_dep)` — mesuré en §3.1, exact par construction. Le joint entre
frames diffère ; le point remis au sampler, non.

Ce que le t = −2,49 mesure, c'est plutôt la faiblesse du protocole : les deux
bras partagent la première descente puis divergent dans des images entièrement
différentes, et les 24 écarts d'un même run sont **autocorrélés** (un run tombé
dans un bassin peu contrasté y reste), donc l'erreur-type appariée est
sous-estimée — le n effectif est très inférieur à 12 × 24. L'écart (0,006) reste
sous la moitié de la dispersion inter-graines (0,012–0,014), et la mission
précédente avait relevé un écart de signe **opposé**, de taille comparable, par
la même méthode. Rien à conclure.

---

## 6. Hygiène et limites

- Le `config_file` du modèle réécrit par le passage du TUI → restauré par
  `git checkout`. Le PNG écrit par `[s]` pendant le test → supprimé.
- `night_run.ckpt` (5,5 Mo) copié depuis le dépôt principal, laissé **non
  commité** (`Models/` n'est pas gitignoré).
- Worktree temporaire du binaire d'avant (`4a67679`) supprimé, fenêtre tmux tuée.
- **`cargo test` salit toujours `Models/Stable_Diffusion/config_file`** — limite
  connue et non corrigée, documentée en `PERPETUAL_INFERENCE.md` §5. Vérifier
  `git status` après un `cargo test`.
- **La quantification 8 bits est un plafond de mesure, pas de rendu.** Toute
  corrélation de grain lue sur des PNG plafonne vers 0,87 pour cette montée ;
  les chiffres exacts se lisent en f32 (§3.2). Ça ne concerne que la mesure — la
  fenêtre live affiche du f32.
- **Une mesure tentée et abandonnée, à ne pas refaire** : capturer l'écran pour
  mesurer la cohérence de bout en bout. `screencapture` sur la fenêtre wgpu
  bloque le présent du run — 20 captures pour 2 incréments — donc les captures
  successives sont majoritairement identiques et la mesure n'a pas de sens. Le
  buffer composé, lui, est exactement ce que la fenêtre reçoit.
- Le tempo minimum du TUI est 2 pas/s (`MIN_TEMPO`) : on ne peut pas ralentir la
  montée davantage pour l'observer image par image.
