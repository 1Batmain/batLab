# L'inférence perpétuelle — rapport de mission

Branche : `perpetual-inference` · Plateforme : macOS (Darwin 25.4.0, Apple M5 Pro) · wgpu 28

## Résumé

Un troisième mode de run à côté de `Train` et `Infer` : **Perpetual**, une chaîne
inverse qui ne finit jamais. Elle descend jusqu'à l'image, la re-bruite d'un cran,
redescend — et se pilote en direct pendant qu'elle tourne.

Huit commits, dans l'ordre où ils ont dû être faits :

| | Commit | |
|---|---|---|
| 1 | `422dd65` | `Screen::LoadPath` n'était jamais assigné — les modèles sauvegardés étaient hors d'atteinte |
| 2 | `75873d9` | `reverse_step` extrait — une seule écriture du pas inverse |
| 3 | `2fbbdc4` | `PerpetualDrift`, machine à états pure (errance / respiration) |
| 4 | `d4fae83` | le mode câblé au TUI (worker, contrôles, panneau) |
| 5 | `4082bc0` | `--headless-perpetual` : time-lapse + planches d'errance |
| 6 | `a6ce182` | le mode « Perpetual » était dessiné mais hors d'atteinte du curseur |
| 7 | `da43f82` | la fenêtre live ne disait pas qu'elle montrait deux images |
| 8 | `a522334` | `--window` : la preuve d'agencement, en image |

Les commits 6–8 répondent à **deux retours d'usage** — l'un et l'autre trouvés en
pilotant le vrai TUI, pas en relisant le code. C'est le sujet des §1 et §2.

---

## 1. Deux bugs d'atteignabilité, même forme

### 1.1 Les modèles sauvegardés (`422dd65`)

`Screen::Home` et `Screen::LoadPath` existaient, étaient dessinés, géraient leurs
touches — et **rien ne les assignait jamais**. L'application démarrait en dur sur
`Screen::TemplateSelector`, donc `Models/Greyscale_Diffusion_L` était inaccessible
depuis l'interface. Gap relevé dans `INFER_VIZ.md` §6 et contourné à l'époque par
un template temporaire.

Corrigé : `App::new()` démarre sur `Home`, qui route vers `LoadPath` ou
`TemplateSelector` ; `Esc` remonte au lieu de quitter. Et surtout, charger un
modèle passe désormais par `apply_loaded_model`, qui **n'appelle jamais**
`write_model_config` — l'effet de bord documenté dans `INFER_VIZ.md` §6 (la route
template écrivait `InferenceConfig::default()` et avait silencieusement réinitialisé
deux `config_file`).

### 1.2 Le mode lui-même (`a6ce182`)

Le mode Perpetual livré en `d4fae83` était **dessiné dans le sélecteur et
définitivement inselectionnable** : `draw_mode_selector` proposait trois modes,
`handle_mode_selector` bornait sa descente à `selected < 1`, héritée de l'époque où
il y en avait deux. Tout le mode n'était joignable que par un `config_file` édité à
la main.

C'est le même défaut que 1.1 sous un autre angle : **du code correct, dessiné,
testé, et inatteignable.** Aucun test ne le voyait parce que les tests
interrogeaient le dessin et le handler *séparément* — les deux avaient raison
chacun de son côté. La borne vient maintenant de la liste elle-même :

```rust
pub const RUN_MODE_CHOICES: [&str; 3] = ["Inference", "Training", "Perpetual"];
```

partagée par le dessin et le handler, avec
`the_mode_cursor_reaches_every_run_mode` qui épingle qu'ajouter un mode suffit à le
rendre atteignable.

**Leçon de méthode** : ces deux bugs ne sont pas sortis d'une relecture. Ils sont
sortis du fait d'avoir *appuyé sur les touches*. Une suite de tests unitaires verte
à 109 ne dit rien sur ce qu'un utilisateur peut atteindre.

---

## 2. « L'image est découpée en deux, la gauche décorrélée de la droite »

Retour utilisateur sur un run perpétuel en direct. C'est le **comportement
nominal** — la frame est x_t à gauche et x̂₀ à droite — mais rien à l'écran ne le
disait, et deux moitiés qui diffèrent sans raison affichée se lisent comme une
désynchronisation. Le retour était juste : l'affichage mentait par omission.

Deux choses à faire, et il fallait les faire dans cet ordre : **d'abord vérifier
qu'il n'y avait pas de vrai bug**, ensuite seulement habiller.

### 2.1 La vérification, parce que le doute était mérité

La validation d'`INFER_VIZ.md` §5 comptait des frames publiées et lisait
`latent[0]` / `x0[0]`. **Aucun de ces contrôles ne distingue deux panneaux côte à
côte d'un buffer entrelacé** : un stride faux aurait passé les deux.

La composition — le seul endroit où l'agencement peut être faux — sort donc de
`publish`, qui exige un GPU, vers `compose_frame`, qui n'exige rien. Quatre tests
l'épinglent :

| Test | Ce qu'il attrape |
|---|---|
| `compose_frame_lays_the_panes_out_row_major_around_the_rule` | agencement pinné élément par élément, chaque valeur source encodant son `(x, y)` : un transposé échange `10y+x` et `10x+y`, un décalage de stride déplace tout le panneau droit |
| `a_checkerboard_and_a_ramp_survive_the_composition_side_by_side` | damier à gauche / rampe à droite — les deux motifs échouent *visiblement différemment* (le damier bave, la rampe descend au lieu de traverser) |
| `the_rule_is_a_full_height_column_on_every_channel` | le filet est une colonne pleine hauteur sur tous les canaux, et n'a mordu sur aucun panneau |
| `identical_sources_produce_two_identical_panes` | sources identiques → panneaux identiques (le contrôle d'errance, cf. 2.3) |

**Les tests ont été vérifiés par mutation, pas seulement vus verts** — un test
d'agencement qui ne tombe jamais ne prouve rien :

| Mutation injectée | Résultat |
|---|---|
| indexation source transposée (`row * channels` au lieu de `row * pane_stride`) | **2 rouges** |
| panneau droit décalé par-dessus le filet | **4 rouges** |

**Verdict : le stride était correct.** Le défaut était bien l'absence de
signalétique — mais on le sait maintenant, au lieu de le supposer.

### 2.2 La signalétique

Trois réponses, dans l'ordre où l'œil les rencontre :

1. **Un filet entre les panneaux** (`SEPARATOR_COLUMNS = [1.0, -1.0, 1.0]`,
   blanc/noir/blanc). La frame passe de `2W` à `2W+3` colonnes. Choisi parce qu'il
   **ne peut pas être pris pour du contenu** : aucun échantillon de ce modèle ne
   contient une colonne d'un pixel noir saturé bordée de deux blancs saturés. Un
   aplat gris « neutre » aurait été plus discret et se serait fondu dans une zone
   plate à t=0 — exactement le moment où la lisibilité compte.
2. **Le titre de la fenêtre** nomme les moitiés :
   `gauche: x_t (bruité) | droite: x̂₀ (estimation)`. C'est la seule légende visible
   quand on regarde l'image et pas le TUI.
3. **Le pied de page d'inférence** porte la même légende que le panneau perpétuel,
   qui l'avait déjà.

Le shader n'a pas bougé : c'est un indexeur row-major
(`buf[(y*width + x)*channels + c]`), on lui annonce `frame_width()` et le filet est
de vrais pixels du buffer.

### 2.3 La preuve, en image

`--window` fait écrire au chemin headless la frame composée elle-même, par la
**même fonction** que le chemin live (`compose_live_frame`, ni GPU ni fenêtre). Une
capture d'écran prouverait ce qu'un écran a fait ; ceci prouve ce que le buffer
contient.

```bash
./target/release/main --headless-perpetual Greyscale_Diffusion_L \
  --checkpoint Models/.../night_run.ckpt --frames 4 --depth 64 --seed 12345 --window \
  --out perpetual_samples/fenetre
```

`perpetual_samples/fenetre_planche.png` — colonne gauche : mi-descente (t=32),
colonne droite : cycle clos (t=0).

| Mesure | Résultat |
|---|---|
| géométrie | **67×32 = 32 + 3 + 32** |
| filet | colonnes `[255, 0, 255]` sur les 32 lignes, `min == max` — jamais rompu |
| t=32, moyenne \|gauche − droite\| | 36/255, corr **+0,70** |
| t=0, moyenne \|gauche − droite\| | **0,00**, corr **+1,000** |
| \|Δ\| entre cycles | 0,138 / 0,132 / 0,151 |

**À t=0 les deux moitiés sont identiques au bit.** C'est le contrôle documenté de
l'errance : à t=0 la sortie du sampler *est* son x̂₀ (σ=0, ᾱ_prev=1, cf.
`x0_estimate_is_the_x0_the_reverse_step_uses`), et la composition le préserve. Les
moitiés diffèrent quand elles doivent différer et coïncident quand elles doivent
coïncider — **le découpage vu par l'utilisateur est le mode nominal.**

Les \|Δ\| tombent dans la fourchette 0,121–0,152 mesurée en `4082bc0` pour t_r=64 :
le mode n'a pas dérivé sous les modifications.

### 2.4 Une note de lecture — l'œil s'est trompé le premier

Agrandi 6× au plus proche voisin, le panneau x_t à t=32 **a l'air de bruit pur**
(5,6 % de pixels saturés), et j'ai failli le signaler comme anomalie. Les chiffres
disent l'inverse : std 0,49 contre 0,33 sur x̂₀, corrélation **+0,70** — une image
bruitée, pas un champ décorrélé, et cohérent avec √(1−ᾱ₃₂) = 0,209.

Mesuré plutôt que supposé. C'est le piège que `SCALE_UNET.md` documente pour les
métriques de diffusion, et il vaut aussi pour ce qu'on croit voir.

---

## 3. Architecture du mode

### 3.1 `reverse_step` — une seule écriture du pas inverse (`75873d9`)

La boucle perpétuelle doit descendre des **portions arbitraires** de la chaîne
(t_r → 0, re-bruitage, en boucle), ce que `sample_diffusion` ne sait pas faire : il
part toujours du bruit pur et descend T→0. Recopier son corps de boucle aurait
dupliqué la composition de l'entrée, la dérivation de la graine et le tirage
postérieur — trois endroits où la dérive silencieuse est **déjà documentée dans ce
dépôt** (`ANISOTROPY_HUNT.md`).

Le corps du pas est donc extrait tel quel, et `sample_diffusion` l'appelle. Aucune
maths touchée, aucun test modifié. `reverse_step_seed` devient la seule dérivation
de graine par pas — le point où l'anisotropie pourrait revenir est désormais
impossible à re-dériver différemment par un appelant
(`reverse_step_seed_folds_the_timestep_in_without_xor`).

### 3.2 `PerpetualDrift` — la boucle infinie rendue testable (`2fbbdc4`)

`PerpetualDrift` dit où en est un run qui ne finit jamais : quel t, quand arrêter de
descendre, jusqu'où re-bruiter. **Il ne détient ni modèle, ni GPU, ni tenseur** — la
boucle infinie, la chose la plus pénible à vérifier à l'œil, devient la moins chère
à tester (8 tests, aucune inférence).

Deux régimes : **errance** (descend jusqu'à t=0, image résolue, puis rebondit à t_r)
et **respiration** (plancher à t_r/2, l'image ne se résout jamais).

Deux points de conception que les tests épinglent :

- **La première descente part toujours du haut du schedule**, quel que soit t_r : le
  run commence sur du bruit pur, et t_r ne gouverne que la remontée des cycles
  suivants. Partir à t_r débruiterait un champ gris.
- **Ce qu'on re-bruite, c'est le x̂₀ clippé, jamais le latent brut.** `add_noise`
  implémente le processus avant, qui suppose un x₀ *propre* dans la plage des
  données ; à un plancher > 0 le latent n'en est pas un, et le ré-injecter
  gonflerait le terme de signal de 1/√ᾱ_plancher **à chaque cycle**.

### 3.3 Le câblage TUI (`d4fae83`)

Le worker `run_perpetual` descend avec le même `reverse_step` que
`sample_diffusion`, re-bruite avec le même `add_noise` que l'entraînement, et publie
dans la `LiveFrame` déjà câblée au visualiseur : **rien de la logique de débruitage
n'est dupliqué**, seul l'itinéraire change.

Le panneau `Perpetual Drift` et le pied de page sont peuplés **exclusivement** par
`PerpetualState`, publié par le worker — l'UI n'anticipe jamais le résultat d'une
touche, donc le t_r affiché est celui que le sampler utilise.

**Throttle : nécessaire, réglé à 30 pas/s.** Le sampler tient ~500 pas/s sur ce
modèle : un cycle t_r=64 s'y boucle huit fois par seconde — un clignotement, pas une
dérive.

---

## 4. Validation end-to-end du TUI

TUI réel (`./target/release/main`) lancé dans une fenêtre tmux dédiée
(`batLab-perpetual-e2e`, 200×50) et piloté par `tmux send-keys`, méthode
`INFER_VIZ.md` §5. Modèle chargé **par l'interface** — `Home` → `Load Saved Model` →
`Greyscale_Diffusion_L` → `night_run.ckpt` → `Perpetual` — ce qui valide `422dd65`
et `a6ce182` sur le chemin réel.

### 4.1 Navigation

| Vérification | Résultat |
|---|---|
| `Screen::Home` au démarrage, deux routes offertes | OK — `> Load Saved Model` / `Select Model Template` |
| `LoadPath` liste les modèles du disque | OK — les 4 modèles de `Models/`, dont `Greyscale_Diffusion_L (32x32x5, 28 layers)` |
| Sélecteur de poids | OK — `night_run.ckpt` et `scale_run.ckpt` proposés |
| Curseur du sélecteur de mode atteint `Perpetual` | **OK — le bug `a6ce182`, reproduit puis corrigé en direct** |
| 3ᵉ `Down` ne dépasse pas le dernier mode | OK — reste sur `Perpetual` |
| Écran de paramètres perpétuels | OK — 6 champs, `Enter` parcourt puis lance |

### 4.2 Chaque contrôle, déroulé

Run réel : `t_r=64`, errance, `night_run.ckpt`, 28 couches.

| Touche | Attendu | Observé |
|---|---|---|
| — (baseline) | run vivant, throttle tenu | `t_r=64 · cycle 6 · 30/30 pas/s` |
| `↑` ×3 | t_r monte, borné | 64 → **88** (pas de 8) |
| `↓` ×2 | t_r descend | 88 → **72** |
| `→` ×3 puis ×6 | tempo monte | consigne 228, mesure **200–211** — le sampler est le plafond, le panneau affiche **les deux** |
| `←` ×4 | tempo descend | 30 → **20/20 pas/s** |
| `espace` | pause | `en pause · 0/20 pas/s`, **t figé à 66 sur 2 s** |
| `espace` | reprise | `en cours · 20/20 pas/s`, t repart |
| `m` | régime bascule | errance → **respiration** → errance ; en respiration t reste ≥ t_r/2 |
| `r` | re-seed | `cycle 181 → cycle 0`, t repart du haut du schedule |
| `s` | PNG écrit | `✓ image → perpetual_samples/errance_c000_000.png` |
| `v` on → off → on | fenêtre bascule, run continue | OK — `cycle 11 → 14`, aucun décrochage |
| `q` | sortie propre | **`EXITED=0`** |

**stderr sur tout le run : 0 octet.**

`cargo test --workspace` : **109 verts** (105 avant la mission), 4 ignorés. Aucun
test affaibli ; 5 ajoutés (1 sur le curseur de mode, 4 sur l'agencement de la
frame).

### 4.3 Hygiène

- Le `config_file` du modèle **réécrit par le passage du TUI** (il persiste le mode
  et ses paramètres, avec des chemins absolus du worktree) → restauré par
  `git checkout`. Le diff produit était **identique octet pour octet** à celui que
  l'agent précédent avait laissé non commité : c'était bien du résidu de TUI, jeté à
  raison.
- `night_run.ckpt` (5,5 Mo) laissé **non commité** — `Models/` n'est pas gitignoré.
- Le PNG écrit par `[s]` pendant le test → supprimé.
- Fenêtre tmux tuée en fin de mission.

---

## 5. Limite connue, non corrigée

**`cargo test` salit des fichiers suivis par git.** Un test qui emprunte la route
template appelle `storage::write_model_config(&template.key, …)`, qui écrit
`Models/<clé>/config_file` dans le dépôt réel — `Models/Stable_Diffusion/config_file`
est ressorti modifié d'un simple `cargo test` pendant cette mission.

C'est le même rayon de souffle que celui documenté en `422dd65` (§1.1), vu depuis
les tests au lieu de l'UI. Inoffensif ici (le champ ajouté est `"checkpoint": null`)
mais c'est un test qui écrit hors de son bac à sable, et il finira par écraser
quelque chose qui compte. **Correctif hors périmètre** : il demande d'injecter la
racine de stockage plutôt que de la déduire d'un global, ce qui touche `storage` et
tous ses appelants. À traiter dans une mission dédiée — et jusque-là, vérifier
`git status` après un `cargo test`.
