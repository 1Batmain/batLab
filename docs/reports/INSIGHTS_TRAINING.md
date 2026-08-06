# INSIGHTS_TRAINING — diagnostic du blanc saturé (Greyscale_Diffusion)

Branche `training-insights`. Suite à l'audit/fix du pipeline (`AUDIT_TRAINING.md`,
`FIX_TRAINING.md`) : la loss décroît sainement mais les échantillons générés sont
**saturés de blanc avec quelques bandes sombres**. Ce rapport ajoute
l'instrumentation demandée, **prouve la cause racine**, applique le fix et montre
la guérison avant/après.

---

## TL;DR

- **Cause racine prouvée** : le modèle Greyscale a `input == output == [32,32,1]`,
  donc `timestep_channels = 0` — **le réseau ne reçoit jamais `t`**. Il ne peut
  alors prédire qu'un ε̂ *moyenné sur tous les t* : à `t` bas il **sous-estime**
  massivement la variance du bruit (ε̂ std ≈ 0,6 pour une cible std ≈ 1,0).
- **Mécanisme prouvé** : l'échantillonnage divise par `√α` à chaque étape
  (produit `∏ 1/√α ≈ 174×` sur 256 pas). Un ε̂ trop faible laisse un résidu qui
  est amplifié multiplicativement → le latent **explose** (std 1 → 68) → image
  clampée au **blanc** avec quelques pixels très négatifs (les **bandes sombres**).
- **Le sampler est innocenté** : avec un ε̂ *oracle* (le vrai bruit résiduel), la
  même chaîne inverse reste **bornée** (std 0,98 → 0,57) et reconstruit l'image à
  `rmse = 0`. Avec `ε̂ = 0`, elle explose ×178. Le blanc est donc **100 % un
  problème de qualité de ε̂**, pas de schedule/sampler.
- **Fix** : conditionnement temporel — `input [32,32,3]` (2 canaux d'embedding),
  embedding sinusoïdal **lisse et normalisé** (`τ = t/(T−1)`, `cos(πτ)` monotone),
  identique côté GPU (entraînement) et CPU (inférence). Plus un garde-fou qui
  rejette au build `dim_kernel.z != dim_input.z` (le piège qui avait rendu le
  premier essai inopérant).
- **Guérison** : avec les réglages d'inférence réels du config (magnitude 0,3,
  3 chemins), le modèle corrigé (6000 pas) produit une image **entièrement dans
  [−1,1], sans aucune saturation** (`min −0,76 / max −0,08`), là où le cassé sort
  massivement de la plage (`min −27 / max +7`) et clampe. Voir §4 et
  `insights_samples/comparison.png`.

---

## 1. Instrumentation ajoutée (livrable durable)

Tout est piloté par l'**API publique** (`Model::predict` + `LinearNoiseSchedule`),
donc valable pour un modèle sain comme dégénéré, sans dépendre des internes GPU.

### 1.1 Module `bat_building/src/model/training/metrics.rs`

- `compose_diffusion_input(...)` — **source unique** de la composition
  `[signal | embedding temporel]`. Utilisée par le probe, le sampler et
  l'inférence : garantit que le conditionnement d'entraînement == celui
  d'inférence.
- `probe_diffusion(...)` → `Vec<BucketStat>` — pour chaque **tranche de `t`**
  (4 buckets par défaut), bruite des échantillons, prédit ε̂, et calcule
  `MSE(ε̂, ε)` **par tranche** (corrige le finding #7 : la loss n'était mesurée
  que sur le dernier élément du batch) + stats `ε̂` et `ε` (min/max/mean/std).
- `sample_diffusion(...)` — **le** sampler de diffusion (inférence + diagnostics
  partagent ce code, pas de divergence possible), avec capture optionnelle de la
  **trajectoire de débruitage** : `latent_in`, `eps_hat`, `latent_out`
  (min/max/mean/std) à chaque étape des 256.
- `MetricsLogger` — écrit un **JSONL** à côté du checkpoint.
- `Stats::of(...)` — stats tolérantes aux NaN/Inf, avec compteur `non_finite`.

### 1.2 Chemin d'entraînement (`main/src/run_training`, headless inclus)

À chaque intervalle de report (25 pas) : un enregistrement `train_loss` + un
`train_probe` (loss + stats ε̂/ε par tranche de `t`). Tous les
`SAMPLE_INTERVAL_STEPS` (200) et en fin de run : un `sample` + la trajectoire de
débruitage (`denoise_step`). Fichier : `<checkpoint_stem>_metrics.jsonl`.

### 1.3 Nouveau point d'entrée `--headless-sample` (DEV/CI)

```bash
cargo run -p main --release -- --headless-sample <model> \
    --checkpoint <path.ckpt> [--seed N] [--paths N] [--magnitude F] \
    [--out img.png] [--log metrics.jsonl]
```

Charge un checkpoint, génère l'image **et** le log de la trajectoire de
débruitage, **sans entraîner** et **sans toucher** aux poids/config sauvegardés
(même esprit que `--headless-train`, inatteignable depuis le TUI).

### 1.4 Comment lire le JSONL

Une ligne = un objet JSON avec un champ `kind` :

| `kind`         | champs clés |
|----------------|-------------|
| `train_loss`   | `step`, `loss`, `lr`, `batch_size` |
| `train_probe`  | `step`, `buckets[]` (`t_lo`,`t_hi`,`loss`,`eps_hat`,`eps_target`) |
| `sample`       | `step`, `seed`, `final_image` |
| `denoise_step` | `train_step`, `step_index`, `diffusion_step`, `latent_in`, `eps_hat`, `latent_out` |

Signaux à surveiller : `eps_hat.std` doit approcher **1,0** (surtout à `t` élevé),
la `loss` par tranche doit baisser *à toutes les tranches*, et `latent_out.std` le
long de la trajectoire doit rester **~1** (pas grimper vers des dizaines).

---

## 2. Cause racine — mesures

### 2.1 Le probe démasque un ε̂ moyenné sur `t` (modèle cassé, 600 pas)

La loss batch rapportée descend joliment (1,54 → 0,088) — mais c'est la loss du
*dernier élément*. Le probe par tranche révèle la vérité :

```
t[  0- 64)  loss=0.7130   ε̂ std=0.665   (ε std≈1.00)   ← catastrophique
t[ 64-128)  loss=0.1152   ε̂ std=0.967
t[128-192)  loss=0.0880   ε̂ std=1.011
t[192-256)  loss=0.0863   ε̂ std=1.015
```

À `t` élevé (entrée ≈ bruit pur), prédire ε̂ ≈ x_t suffit → loss faible. À `t`
bas, le réseau **ne sait pas** qu'il est à `t` bas (aucun conditionnement), donc
il produit le même ε̂ « de compromis » : sa variance s'effondre (0,665 ≪ 1,0).
C'est la signature exacte d'un **ε̂ moyenné sur `t`**.

### 2.2 La trajectoire de débruitage montre l'explosion (modèle cassé)

```
idx  diff_t   latent_in[min / max / mean / std]     eps_hat[mean/std]
  0  t=255   [  -2.74 /  +3.22 / +0.012 /  0.985]   [-0.014/1.042]
 64  t=191   [ -14.28 / +11.07 / +0.226 /  4.378]   [+0.058/0.500]
128  t=127   [ -68.62 / +45.21 / +1.092 / 19.764]   [+0.042/0.396]
192  t= 63   [-175.26 /+112.85 / +2.890 / 49.622]   [+0.039/0.382]
255  t=  0   [-238.92 /+151.92 / +3.990 / 67.653]   [+0.038/0.377]
→ image finale : mean=+3.990  → clampée blanc (63.7 % pixels = 255, 35.2 % < 10)
```

Le latent part sain (std 0,98) puis explose (→ 67,7) parce que ε̂ (std ≈ 0,38)
est trop faible pour annuler la composante bruit ; le facteur `1/√α` la
recompose à chaque pas. Image finale `mean = +3.99` → **blanc saturé**, avec une
minorité de pixels très négatifs → **bandes sombres**. Symptôme reproduit.

### 2.3 Le sampler est correct (test oracle)

Réimplémentation fidèle du schedule (bêtas rescalés T=256) + `denoise_step`, en
fournissant à chaque pas le **vrai** bruit résiduel `ε̂ = (x_t − √ᾱ·x_0)/√(1−ᾱ)` :

```
terminal ᾱ = 3.29e-05 (√=0.0057)   ∏(1/√α) = 174.4×
ORACLE  : latent std 0.981 → 0.110 → 0.160 → 0.414 → 0.572   reconstruction rmse=0.0000
ε̂ = 0   : latent std final = 178.2   (l'explosion pure)
```

**Conclusion** : la chaîne inverse est stable et exacte quand ε̂ est bon. Le blanc
saturé est **entièrement** dû à un ε̂ dégénéré → à l'absence de conditionnement.
(script : `scratchpad/oracle_check` reproduit dans §5.)

---

## 3. Fix appliqué

1. **Config `Models/Greyscale_Diffusion/config_file`** :
   - `input_size : [32,32,1] → [32,32,3]` (2 canaux d'embedding temporel).
   - première conv `dim_input : [32,32,3]`, **`dim_kernel : [3,3,1] → [3,3,3]`**.
     La profondeur du noyau **doit** égaler le nombre de canaux d'entrée (le
     shader itère `kz` sur `dim_input.z`). L'oubli de ce point avait rendu le
     premier essai totalement inopérant (buffer de poids sous-dimensionné, canaux
     temporels lus en garbage) → d'où le garde-fou ci-dessous.
   - `timestep_channels = input.z − output.z = 2`.

2. **Embedding temporel lisse** (`schedule.rs` + `diffusion_prepare.wgsl`) :
   `τ = t/(T−1) ∈ [0,1]`, paires `[sin(π·2ⁱ·τ), cos(π·2ⁱ·τ)]`. La paire de plus
   basse fréquence donne `cos(πτ)`, **strictement monotone 1 → −1** sur tout le
   schedule → un code de `t` lisse et injectif. L'ancien `sin(step)/cos(step)`
   (période ≈ 6,28 *pas*) était aliasé : code quasi-aléatoire, inexploitable en
   peu d'époques.

3. **Cohérence GPU/CPU verrouillée** : `total_steps` est passé au shader pour que
   l'embedding d'entraînement (GPU) soit *identique* à celui d'inférence (CPU).
   Test `timestep_embedding_matches_shader_formula` (miroir de la formule WGSL) +
   `timestep_embedding_low_frequency_is_monotone_over_schedule`.

4. **Garde-fou conv** (`convolution.rs`, `error.rs`) : `set_dim_output` rejette
   désormais `dim_kernel.z != dim_input.z` avec `KernelChannelMismatch`, au build.
   Tests unitaires ajoutés.

---

## 4. Guérison — avant / après

Modèle cassé (600 pas, sans conditionnement) vs modèle corrigé (6000 pas,
conditionnement temporel). Même dataset, `lr=1e-3`, `batch=16`.

### 4.1 Convergence du modèle corrigé — probe `MSE(ε̂, ε)` par tranche de `t`

Mesure in-distribution (le vrai signal, moyennée sur 8 échantillons × 2 `t` par
tranche) tout au long des 6000 pas :

```
step | t0-64 | t64-128 | t128-192 | t192-256 | ε̂ std (haut-t)
   0 | 1.132 | 1.163   | 1.1754   | 1.1847   | 0.496
 250 | 0.728 | 0.171   | 0.1214   | 0.1179   | 0.957
 500 | 0.685 | 0.131   | 0.0822   | 0.0791   | 0.989
1000 | 0.665 | 0.102   | 0.0560   | 0.0530   | 0.989
2000 | 0.641 | 0.087   | 0.0406   | 0.0352   | 0.982
3000 | 0.627 | 0.084   | 0.0343   | 0.0287   | 0.986
5999 | 0.596 | 0.074   | 0.0267   | 0.0196   | 0.971
```

Le débruitage haut/moyen-`t` devient quasi-parfait (loss 1,18 → **0,020**,
ε̂ std → **0,97**). Le bas-`t` s'améliore plus lentement (1,13 → 0,60) : c'est
**partiellement intrinsèque** à la prédiction-ε (à `t ≈ 0`, `x_t ≈ x_0 + 0,02·ε`,
le bruit est quasi invisible donc ε est mal identifiable).

> **Note d'échantillonnage** : les previews périodiques d'entraînement
> (`magnitude 1,0`, 1 chemin — le pire cas) restent bruitées car chacune utilise
> un **seed différent** (`seed = step`), et parce que l'amplification `1/√α` est
> si agressive qu'un résidu de ~3 % sur ε̂ se compose. Le config d'inférence
> utilise `magnitude 0,3` + 3 chemins (moins de bruit stochastique injecté +
> moyennage) — c'est ce qui rend le résultat borné (§4.2).

### 4.2 Échantillons avant / après (même seed 390, via `--headless-sample`)

Stats **pixel** de l'image 8 bits générée (`insights_samples/`) :

| réglage | modèle | mean | std | % blanc (255) | % noir (0) | plage f32 |
|---|---|---|---|---|---|---|
| inférence config (mag 0,3 ; 3 ch.) | **cassé**  | 13,5 | 54,9 | 4,1 % | 93,6 % | [−27,2 ; +7,1] |
| inférence config (mag 0,3 ; 3 ch.) | **corrigé** | 82,9 | 14,2 | **0 %** | **0 %** | **[−0,76 ; −0,08]** |
| pire cas (mag 1,0 ; 1 ch.)         | cassé  | 165,8 | 120,8 | 64,2 % | 34,2 % | [−238 ; +232] |
| pire cas (mag 1,0 ; 1 ch.)         | corrigé | 188,7 | 100,3 | 64,2 % | 16,5 % | [−13,3 ; +17,4] |

- **Cassé, inférence config** : image bimodale saturée (93,6 % noir + 4,1 % blanc)
  — le symptôme (ici clampé côté noir ; le signe dépend de l'état dégénéré, le
  run 20 000 pas de l'utilisateur clampait côté blanc). Latent hors [−1,1] d'un
  facteur ~27.
- **Corrigé, inférence config** : **image entièrement dans [−1,1], 0 % de pixels
  saturés** — une vraie image en niveaux de gris (peu contrastée : 6000 pas ≈
  1,9 époque, l'entraînement court manque de détail, mais **la saturation a
  disparu**).
- En pire cas (full-stochastique mono-chemin), le corrigé reste fragile mais son
  latent est ~16× moins explosé que le cassé (std 4,2 vs 67,7).

Comparaison visuelle : `insights_samples/comparison.png` (agrandie ×5). Fixtures :
`insights_samples/{before,after}_{infer,worst}.png`. Le modèle
`Models/Greyscale_Diffusion_broken/` est une **copie de diagnostic** (architecture
d'origine, `input [32,32,1]`) permettant de rééchantillonner l'ancien checkpoint
via `--headless-sample` pour l'avant/après.

### 4.3 Interprétation honnête

Le fix **traite la cause racine prouvée** (absence de conditionnement → ε̂
dégénéré) : le débruitage haut/moyen-`t` passe de cassé à quasi-parfait, et avec
les réglages d'inférence du config le blanc saturé **disparaît**. La qualité
visuelle (contraste, détail) reste limitée par la brièveté de l'entraînement
(1,9 époque) et par la fragilité intrinsèque de la prédiction-ε face à
l'amplification `1/√α` ; des gains supplémentaires viendraient d'un entraînement
plus long et, éventuellement, d'un objectif moins sensible (prédiction de `x_0`
ou `v`) ou d'un schedule cosine — hors périmètre de ce diagnostic.

---

## 5. Reproduire

```bash
# 1. run diagnostique (métriques écrites dans <out sans .ckpt>_metrics.jsonl)
cargo run -p main --release -- --headless-train Greyscale_Diffusion \
    --steps 600 --dataset datasets/cifar10_grey.batraw --lr 1e-3 --batch 16 \
    --out /tmp/run.ckpt

# 2. échantillonner un checkpoint + logguer la trajectoire de débruitage
cargo run -p main --release -- --headless-sample Greyscale_Diffusion \
    --checkpoint /tmp/run.ckpt --seed 390 --paths 3 --magnitude 0.3 \
    --out /tmp/sample.png --log /tmp/sample.jsonl

# 3. test oracle (prouve que le sampler est correct sur ε̂ vrai)
python3 scratchpad/oracle_check.py

# 4. tests de non-régression (embedding CPU==GPU, garde conv, stats)
cargo test -p bat_building schedule:: metrics:: convolution::
```
