# Correction du pipeline d'entraînement — batLab

Branche : `fix-training-pipeline`, basée sur `audit-training`.
Référence : `AUDIT_TRAINING.md` (diagnostic). Ce document couvre les **fixes**.

Périmètre traité : findings **#1, #2, #3, #4**. Hors périmètre (inchangés) :
#5 (perf `group_norm`), #6, #7.

## Résumé

| # | Sévérité | Fix | Commit |
|---|---|---|---|
| 1 | CRITIQUE | Offset de padding `Same` appliqué en forward **et** backward, avec vrai zero-padding | `d813502` |
| 4 | MAJEUR | Données normalisées en `[-1,1]`, encodage/décodage symétriques | `3064d24` |
| 3 | MAJEUR | Betas rééchelonnés par `1000/T` → `ᾱ_T = 3.3e-5` à T=256 | `940ac93` |
| 2 | MAJEUR | Timestep tiré uniformément, indépendant de l'échantillon ; dataset mélangé par époque | `47da023` |
| — | outil | Chemin d'entraînement headless (validation scriptée) | `cb3efd5` |

**État des tests** : `cargo test --workspace` → **44 tests, 0 échec**.
Les 6 tests d'audit passent (4 échouaient avant cette passe ; voir la réserve
sur `diffusion_timestep_is_decorrelated_from_sample_index` en #2).

---

## Finding #1 — padding `Same` (CRITIQUE)

### Fix

`convolution.wgsl` n'appliquait jamais l'offset de padding : la fenêtre du noyau
démarrait en `(oy*s, ox*s)` au lieu d'être centrée. Deux conséquences cumulées —
décalage de `(KH/2, KW/2)` par couche, et débordement hors bornes que la
robustesse d'accès WGSL clampait sur le dernier élément (ni zero-padding, ni
edge-padding cohérent).

- **`convolution.wgsl`** — `pad_y()`/`pad_x()` alignés sur `upsample_conv.wgsl`,
  offset appliqué et **test de bornes explicite** (`continue` → contribution
  nulle) plutôt que de laisser le clamp opérer.
- **`back_convolution.wgsl`** — même offset répliqué à l'identique :
  - `conv_back_input` inverse la nouvelle carte : `oy = (iy + pad_y - ky)/s`,
    avec tests de signe, de divisibilité par le stride et de bornes ;
  - `conv_back_weights` saute les positions padées.

Le forward et le backward ont été corrigés **ensemble** : corriger le forward
seul aurait rendu le gradient faux.

### Preuve

```
same_padding_conv_is_centered_and_zero_padded          FAIL -> ok
conv_weight_gradients_match_finite_differences_same_..  ok  -> ok
stacked_same_padding_conv_grad_input_is_consistent     FAIL -> ok
```

Le test décisif est le troisième — deux couches `Same` empilées, gradient de la
première couche vérifié par différences finies centrées. Il mesurait **81 %
d'erreur relative** avant le fix (`grad_input` ~5× trop petit, erreur croissante
avec la profondeur) ; il passe désormais sous le seuil de `5e-2`.

Le contrôle négatif sur **une** couche passait déjà avant le fix — le backward
était le gradient exact d'un forward faux. C'est précisément pourquoi le test
empilé était nécessaire : l'opérateur était auto-cohérent mais faux.

### Conséquence

**Les checkpoints antérieurs sont invalidés.** Les poids ont été appris contre
un opérateur convolutif décalé ; ils n'ont pas de sens sous l'opérateur corrigé.

---

## Finding #4 — normalisation en `[-1,1]` (MAJEUR)

### Fix

Le forward process `x_t = √ᾱ·x_0 + √(1-ᾱ)·ε` suppose `x_0` centré. Avec
`x_0 ∈ [0,1]` (moyenne ≈ 0.5), `x_t` gardait un biais de `0.5·√ᾱ_t` à **tous**
les timesteps, alors que l'échantillonneur démarre de `N(0,1)`.

- `from_u8`/`to_u8` forment une paire **symétrique** autour de `[-1,1]`
  (`v/127.5 - 1` et `(v+1)·127.5`), utilisée à l'encodage comme au décodage.
- `image_to_tensor` encode en `[-1,1]` sur les deux chemins (luma et RGB).
- Format `.batraw` : nouveau magic **`BATRAW2`** (payload déjà en `[-1,1]`).
  Les fichiers **`BATRAW1` restent lisibles** et sont rééchelonnés au
  chargement (`v·2 - 1`) — le dataset `cifar10_grey.batraw` du dépôt continue
  donc de fonctionner **sans régénération**.

### Preuve

```
raw_dataset_legacy_unit_payload_is_rescaled_to_signed_range  (nouveau)  ok
u8_encoding_round_trips_through_signed_range                 (nouveau)  ok
raw_dataset_greyscale_round_trip / raw_dataset_rgb_round_trip           ok
```

Le second test vérifie la symétrie encode/decode sur les **256** valeurs
possibles, et que le gris moyen tombe bien près de zéro — l'objet même du fix.

Les deux round-trips `.batraw` existants ont été **mis à jour** pour la nouvelle
convention (valeurs d'échantillon en `[-1,1]`) : la sémantique du format a
délibérément changé, l'assertion d'identité en `[0,1]` n'avait plus de sens.

### Conséquence

**Checkpoints antérieurs invalidés** (entrées et cible apprises dans une autre
plage). Le format `.batraw` reste rétro-compatible en lecture.

---

## Finding #3 — schedule n'atteignant pas le bruit pur (MAJEUR)

### Fix

Les bornes `beta ∈ [1e-4, 0.02]` sont celles du papier DDPM, calibrées pour
**T=1000**, et étaient réutilisées telles quelles à **T=256**.

`new_linear` interprète désormais `(beta_start, beta_end)` comme calibrés pour
T=1000 et les **rééchelonne par `1000/num_steps`**, ce qui conserve le bruit
total injecté à n'importe quel T. Chaque beta est plafonné à `0.999` pour garder
`alpha` strictement positif sur les schedules très courts.

Ajout d'une **assertion de construction** : toute combinaison `(T, betas)`
laissant `ᾱ(T-1) ≥ 1e-3` échoue bruyamment au lieu de dégrader silencieusement
l'entraînement.

### Preuve

| T | betas effectifs | `ᾱ_T` | signal résiduel en `x_T` |
|---|---|---|---|
| 256 — **avant** | 1.0e-4 .. 0.020 | 7.50e-2 | **27.4 %** |
| 256 — **après** | 3.9e-4 .. 0.078 | **3.29e-5** | **0.57 %** |
| 1000 (référence DDPM) | 1.0e-4 .. 0.020 | 4.04e-5 | 0.64 % |

Le schedule à T=256 est maintenant **aligné sur la référence DDPM à T=1000**.

```
audit_tests::schedule_reaches_approximately_pure_noise_at_final_step  FAIL -> ok
production_schedule_reaches_pure_noise                     (nouveau)  ok
rescaling_keeps_terminal_alpha_bar_comparable_across_step_counts      ok
```

Le dernier couvre T = 128, 256, 512, 1000.

---

## Finding #2 — timestep corrélé à l'échantillon (MAJEUR)

### Fix

`sample_index` et `diffusion_step` dérivaient du **même** compteur linéaire : le
timestep n'était pas échantillonné, il était une fonction déterministe de
l'indice d'image. Une image ne voyait que `gcd(sample_count, schedule_len)`
timesteps sur tout le run — **16 sur 256** pour CIFAR-10.

- `diffusion_step_for` tire `t ~ U{0, T-1}` via un **SplitMix64** seedé sur
  `(seed du pas, compteur global)`, sans dépendance externe.
- `SampleShuffle` : permutation **Fisher-Yates** du dataset, régénérée à chaque
  époque, dérivée du numéro d'époque — donc **reproductible** d'un run à l'autre.
- Le plan du batch est résolu avant `ensure_prepare_pass` pour éviter le conflit
  d'emprunt entre `self.shuffle` et `pass`.

### Preuve

```
timestep_is_decorrelated_from_sample_index    (nouveau)  ok
sample_order_is_shuffled_between_epochs       (nouveau)  ok
diffusion_step_progression_does_not_alias_single_batch    ok (renforcé)
```

Le premier vérifie que l'échantillon #0 couvre les **256** timesteps sur 2000
époques (contre 16 avant, quel que soit le nombre d'époques). Le second vérifie
que l'ordre diffère entre époques **et** que chaque époque reste une permutation
(chaque échantillon vu exactement une fois).

Le test anti-aliasing de la PR #6 est **conservé mais renforcé** : la propriété
testée est désormais « les tirages d'un batch se répartissent sur le schedule »
plutôt que « la suite identité `0,1,…,7` », le tirage étant devenu aléatoire.

### ⚠ Réserve — le test d'audit a dû être réécrit

`audit_tests::diffusion_timestep_is_decorrelated_from_sample_index`, tel que
livré par l'audit, **ne pouvait pas passer, quel que soit le fix** :

1. il **répliquait le pairing bogué INLINE** et n'appelait jamais le code de
   production — corriger `diffusion.rs` ne pouvait donc rien y changer ;
2. son assertion était **insatisfiable par construction** : il exigeait 256
   timesteps distincts pour l'échantillon #0 sur **20 époques**, or celui-ci
   n'est tiré qu'**une fois par époque** — au plus 20 valeurs distinctes.

Il a été réécrit pour appeler `diffusion_step_for`/`SampleShuffle` et vérifier
la vraie propriété : la couverture dépasse le réseau `gcd = 16` et approche la
borne d'une valeur par époque. C'est le seul test d'audit modifié dans sa
substance ; les autres ont été laissés intacts et sont passés au vert par les
seuls fixes de production.

---

## Validation en conditions réelles

### Tests

```
cargo test --workspace     44 tests, 0 échec
cargo clippy --workspace   aucun warning nouveau (les warnings restants
                           préexistent et sont hors périmètre)
```

### Entraînement

Modèle `Greyscale_Diffusion` (U-Net 12 couches, 32×32×1, skip `Concat`, padding
`Same` — donc directement exposé au finding #1), dataset
`datasets/cifar10_grey.batraw` (50 000 images CIFAR-10 en niveaux de gris),
**from scratch** (`load_checkpoint = false`, les anciens checkpoints étant
invalidés par #1 et #4).

Le programme étant un TUI interactif, un chemin `--headless-train` a été ajouté
(commit `cb3efd5`). Il **réutilise `run_training` tel quel** — il exerce donc
exactement le chemin de production. Il est marqué DEV/CI, n'est pas atteignable
depuis le TUI, ne réécrit jamais le `config_file` du modèle et écrit son
checkpoint dans un fichier scratch.

```bash
cargo run --release -p main -- --headless-train Greyscale_Diffusion \
    --steps 600 --lr 0.001 --batch 16 --dataset datasets/cifar10_grey.batraw
```

Courbe de perte (600 pas, `lr=1e-3`, batch 16, from scratch) :

```
step    0    loss 1.537129   ████████████████████████████████████████
step   25    loss 0.836887   ██████████████████████
step   50    loss 0.526487   █████████████
step   75    loss 0.364829   █████████
step  100    loss 0.277564   ███████
step  125    loss 0.235042   ██████
step  150    loss 0.259882   ██████
step  175    loss 0.170816   ████
step  200    loss 0.252423   ██████
step  225    loss 0.148697   ███
step  250    loss 0.135947   ███
step  275    loss 0.119770   ███
step  300    loss 0.127171   ███
step  325    loss 0.119777   ███
step  350    loss 0.139152   ███
step  375    loss 0.436359   ███████████
step  400    loss 0.117003   ███
step  425    loss 0.104551   ██
step  450    loss 0.094767   ██
step  475    loss 0.103289   ██
step  500    loss 0.098892   ██
step  525    loss 0.086010   ██
step  550    loss 0.126658   ███
step  575    loss 0.114597   ██
step  599    loss 0.087848   ██
```

**Décroissance saine** : 1.537 → 0.088, soit un facteur **~17×**. La descente
est rapide et monotone en tendance sur les 300 premiers pas, puis se stabilise
autour de 0.09–0.12 avec la variance de mesure décrite ci-dessous. Aucune
divergence, aucun `NaN`, aucun plateau précoce.

Un run plus long (3000 pas) a été lancé et confirme la poursuite de la descente
sur la plage observée ; il n'est pas reproduit ici, le débit d'entraînement
étant limité par le finding #5 (non traité, ~1 pas/2 s sur cette machine).

### Note de lecture sur la courbe de perte

La décroissance de la perte est **une condition nécessaire, pas une preuve de
correction** : l'audit établit lui-même qu'avant les fixes la perte descendait
aussi (« le réseau minimise honnêtement un mauvais opérateur »). Une
comparaison avant/après des courbes ne serait donc pas discriminante — d'autant
que #3 et #4 changent l'échelle même de la perte.

**La preuve de correction est ailleurs** : ce sont les gradient checks par
différences finies du finding #1, et en particulier le test à deux couches
empilées qui passe de 81 % d'erreur relative à moins de 5 %.

La variance résiduelle visible entre points de mesure est **attendue** et
relève du finding #7 (hors périmètre) : la perte rapportée n'échantillonne
qu'un seul couple (image, timestep), et la MSE de diffusion varie de plusieurs
ordres de grandeur selon `t`. Différence notable : cette variance est désormais
**aléatoire** et non plus **cyclique de période 256** — c'était le symptôme
combiné de #2 et #7.

---

## Anomalies constatées, non traitées (hors périmètre)

- **Visualiseur / winit** — en exécution headless, un thread annexe panique
  (`EventLoop must be created on the main thread`, macOS). L'entraînement n'est
  pas affecté et se déroule jusqu'au bout. Le visualiseur est traité sur une
  autre branche : non touché ici, mais signalé.
- **Finding #5 (perf `group_norm`)** — non traité, et **nettement dominant sur
  le temps de pas** lors des runs de validation. C'est le prochain frein à
  l'itération.

## Conséquences à retenir

1. **Tous les checkpoints antérieurs sont invalidés** (#1 : opérateur convolutif
   modifié ; #4 : plage de données modifiée). Réentraîner from scratch.
2. Le format `.batraw` reste **lisible en rétro-compatibilité** (`BATRAW1`
   rééchelonné à la volée) ; les nouveaux fichiers devraient utiliser
   `BATRAW2`. Le script `datasets/cifar_to_raw.py` produit encore du `BATRAW1`
   — fonctionnel, mais à migrer.
3. Toute combinaison `(T, betas)` n'atteignant pas le bruit pur échoue
   désormais à la construction du schedule.
