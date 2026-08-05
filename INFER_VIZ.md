# Visualisation live du débruitage pendant l'inférence — rapport de mission

Branche : `infer-visualiser` · Plateforme : macOS (Darwin 25.4.0, Apple M5 Pro) · winit 0.30.12, wgpu 28

## Résumé

`[v]` ouvrait le visualiseur pendant l'**entraînement** seulement. Il fonctionne
désormais aussi pendant une **inférence**, et affiche le débruitage en direct :
le latent courant x_t à gauche, l'estimation x̂₀ du modèle à droite.

Trois commits :

1. `9763c0a` — extraction du calcul de x̂₀ du sampler (fondation, aucune maths changée) ;
2. `ad8c462` — la fonctionnalité (observateur de sampler + pont GPU + câblage TUI) ;
3. `ae07087` — un bug de cadence du visualiseur **trouvé par la mesure** demandée
   au point « throttle » de la spec : le budget de 33 ms ne bridait rien, le rendu
   tournait à 120 fps et coûtait à l'inférence les deux tiers de son débit.

## 1. Ce qui est affiché

Une seule fenêtre, une seule image de **largeur double** :

```
      width = 2 × W                              une ligne du buffer
  ┌───────────────┬───────────────┐         ┌──────────┬──────────┐
  │               │               │         │  x_t     │  x̂₀      │
  │      x_t      │      x̂₀       │         │  (W px)  │  (W px)  │
  │  (le latent   │ (ce que le    │         └──────────┴──────────┘
  │   bruité)     │  modèle croit │
  │               │  voir)        │
  └───────────────┴───────────────┘
```

Le shader du visualiseur est un indexeur row-major pur
(`buf[(y * width + x) * channels + c]`, `shader.wgsl`) : lui déclarer une largeur
de `2 × W` suffit pour obtenir deux panneaux côte à côte. **Aucune modification
WGSL, aucune seconde fenêtre, aucun second pipeline** — les deux vues sont
littéralement deux moitiés d'un même tenseur.

x̂₀ est la forme **clippée** que le pas inverse utilise réellement pour construire
sa moyenne postérieure. C'est le point délicat de la mission : afficher « ce que
le modèle croit voir » n'est honnête que si c'est bien le x̂₀ du sampler, pas une
recopie de sa formule qui pourrait diverger en silence. D'où le commit 1 (§3).

Multi-chemins : le chemin courant est affiché (exigence remplie) ; l'image saute
visiblement au passage d'un chemin au suivant, puisque chaque chemin repart du
même bruit de base et redescend la chaîne. La grille de tous les chemins (bonus)
n'est pas implémentée.

## 2. Câblage — flux du latent jusqu'à l'écran

```
   worker thread                                      main thread
   ─────────────                                      ───────────

   run_inference (main.rs)
        │
        │ LiveFrame::new(gpu, W, H, C)      ┌──────────────────────┐
        │ register_visualiser_source(───────►│ visualiser_control   │
        │   live.buffer(), 2W, H, C)        │  (source enregistrée)│
        │                                   └──────────┬───────────┘
        ▼                                              │ spawn
   sample_diffusion (metrics.rs)                       ▼
        │                                   ┌──────────────────────┐
        │  pour chaque pas t :              │ VisualiserManagerApp │
        │   ε̂ = model.predict(x_t)          │  event loop winit    │
        │   x̂₀ = schedule.x0_estimate(…)    │                      │
        │   x_{t-1} = denoise_step(…)       │  ~30 fps :           │
        │        │                          │   render() lit le    │
        │        ▼                          │   buffer en storage  │
        │   observer(&DenoiseFrame{ … })    │   read-only          │
        │        │                          └──────────▲───────────┘
        ▼        ▼                                     │
   LiveFrame::publish(latent, x0_hat)                  │
        │                                              │
        │  entrelace les lignes : [x_t | x̂₀]           │
        │  queue.write_buffer(…) ───────────────────────┘
        │        (même queue → le rendu suivant voit l'écriture)
        ▼
   clear_visualiser_source()  ← en fin d'échantillonnage
```

Points de conception :

- **Pourquoi un pont est nécessaire.** À l'entraînement, le visualiseur lie
  directement `model.last_output_buffer()` : la donnée est déjà sur le GPU. La
  chaîne inverse est de forme opposée — `sample_diffusion` garde son latent en
  `Vec<f32>` côté CPU et ne le remonte jamais (le modèle n'y fait que des
  prédictions). `LiveFrame` lui donne un domicile GPU, au prix d'un upload de
  4 Ko par pas.
- **Le sampler reste unique.** L'observateur est un `Option<&mut dyn FnMut(&DenoiseFrame)>`
  ajouté à `sample_diffusion` — même style que le `trajectory` déjà présent. Aucune
  duplication de la boucle de débruitage, aucune maths touchée. Quand personne ne
  regarde, `x0_estimate` n'est même pas calculé : le chemin non observé alloue
  exactement comme avant.
- **Thread principal préservé.** Rien n'est changé au threading : l'event loop winit
  garde le thread principal (`run_on_main_thread`), l'inférence tourne sur le worker,
  et `wgpu::Queue` étant `Send + Sync` la publication depuis le worker est licite.
  L'écriture et le rendu passent par la **même queue**, donc la frame suivante
  observe l'écriture sans fence ni readback.
- **`[v]` n'est plus gardé par le mode.** L'ancienne garde était
  `app.monitor.current_lr.is_some()` — posée uniquement par l'événement
  `TrainingState`, donc jamais en inférence. Elle devient
  `has_visualiser_source()` : la touche est offerte exactement quand il y a
  quelque chose à montrer, quel que soit le mode.
- **Libération en fin de run.** `clear_visualiser_source()` ferme la fenêtre quand
  l'échantillonnage rend la main : le latent n'existe plus, laisser une image figée
  la ferait passer pour encore vivante.

## 3. La fondation : un seul x̂₀

`denoise_step_with_magnitude` calculait x̂₀ inline. Plutôt que d'en recopier la
formule côté affichage, elle est extraite dans un helper unique `x0_hat_at`, que
le pas inverse **et** le nouveau `x0_estimate` traversent tous deux.

Épinglé par `x0_estimate_is_the_x0_the_reverse_step_uses` : au pas 0, σ = 0 et
ᾱ_prev = 1, donc la sortie du sampler est exactement la moyenne postérieure ;
le test la reconstruit depuis `x0_estimate` et exige l'égalité **bit-à-bit**, avec
des ε̂ volontairement surdimensionnés pour que le clamp soit porteur. Inliner
différemment l'une des deux voies fait tomber le test.

Les gardes existantes (`injected_noise_over_reverse_chain_is_isotropic`,
`reverse_chain_stays_bounded_…`) restent vertes et intactes.

## 4. La mesure, et ce qu'elle a révélé

La spec demandait de mesurer avant de décider d'un throttle. La mesure a trouvé un
bug préexistant du visualiseur.

Méthode : sonde temporaire (retirée depuis, cf. §6) comptant les publications
côté sampler et les frames présentées côté rendu, écrites dans un fichier **et non
sur stderr** pour que le critère « stderr vide » reste lisible. Run L,
`night_run.ckpt`, 60 chemins × 256 pas, bascules `[v]` en cours de route.

**Cause** : `about_to_wait` appelait `request_redraw()` à chaque passage.
`ControlFlow::WaitUntil` ne fixe qu'une échéance de réveil *au plus tard* — il
n'empêche pas la boucle de se réveiller avant, et sur macOS le display link la
cadence à la fréquence de l'écran. Le budget `FRAME_INTERVAL_MS = 33` ne bridait
donc rien du tout.

| `[v]` | avant correctif | après correctif |
|---|---|---|
| masqué | 332 pas/s | 266 pas/s |
| visible | **111 pas/s** | **309 pas/s** |
| cadence de rendu | 120 fps | 28,4 fps |

Ouvrir la fenêtre coûtait **un facteur 3** sur l'inférence ; après correctif le
coût rentre dans le bruit de mesure. Le correctif garde l'instant du dernier
redraw demandé et n'en redemande un qu'une fois l'intervalle écoulé.

**Décision sur le throttle du côté sampler : aucun.** Publier chaque pas coûte un
`write_buffer` de 4 Ko ; à ~300 pas/s c'est 1,2 Mo/s, invisible dans la mesure
ci-dessus. Le vrai coût était côté rendu, et il est corrigé à la source. Pousser
chaque frame est donc gardé — c'est aussi ce qui rend le débruitage lisible.

Note : ce bug affectait aussi l'**entraînement**, qui subissait la même
sur-cadence ; il n'y était juste pas visible faute d'avoir été mesuré.

## 5. Scénario de validation déroulé

TUI réel (`./target/release/main`) lancé dans une fenêtre tmux dédiée
(`batLab-infer-viz-test`) et piloté par `tmux send-keys`, méthode `CRASH_VISUALISER.md` §4.
Checkpoint `night_run.ckpt` copié depuis le dépôt principal vers
`Models/Greyscale_Diffusion_L/pretrained_weights/latest.ckpt` (l'inférence résout
`latest.ckpt`).

### Run A — binaire propre, 200 chemins (51 200 pas)

| Vérification | Résultat |
|---|---|
| `[v]` proposé dans le pied de page en inférence | OK — ` inference: … \| [v] visualise [s] save [q] quit` |
| Progression pas/chemin/t | OK — `Denoising · path 63/200 · step 125/256 (t=131) (15997/51200)` |
| t décroît pendant que le compteur de pas croît | OK (t=215 → t=7 sur la durée du run) |
| `[v]` on → off → on en cours de run | OK — aucun décrochage : 15997 → 17368 → 18736 |
| Débit stable pendant l'affichage | OK — ~682 pas/s constants |
| Fin d'inférence propre + aperçu affiché | OK — `Inference Preview (32x32x1)` |
| **stderr** sur tout le run | **0 octet** |
| Sortie `[q]` | `EXITED=0`, pas de blocage |

### Run B — binaire instrumenté, 60 chemins (15 360 pas)

| Vérification | Résultat |
|---|---|
| Avant `[v]` : publications sans présentation | OK — `published=1250` à 2,8 s, **0 frame présentée** |
| Après `[v]` : frames qui défilent | OK — 480 frames en ~4,5 s, puis **1 080+** sur le run |
| `[v]` off → les présentations s'arrêtent | OK — figées à 3 930 pendant toute la fenêtre off |
| `[v]` on → elles reprennent | OK — 4 590 → 4 650 |
| Les deux panneaux portent des données distinctes et vivantes | OK — `latent[0]=0.0997 x0[0]=-0.2627`, valeurs qui évoluent à chaque pas |
| Publications totales | 15 360 (= 60 × 256, une par pas) |
| Fin de run → fenêtre fermée | OK — présentations figées à 1 080 après `clear_visualiser_source()` |
| **stderr** | **0 octet** |
| Sortie `[q]` | `EXITED=0` |

`cargo test --workspace` : **98 verts**, 5 ignorés. Aucun test affaibli ; 6 ajoutés
(1 sur l'égalité x̂₀, 3 sur l'arithmétique du libellé de progression, 3 sur le
remplissage de ligne de `LiveFrame`, dont un a effectivement attrapé un panic —
`&src[7..7]` panique sur une slice de longueur 2 même en ne demandant rien).

## 6. Limites connues

- **Le TUI ne sait pas atteindre ses modèles sauvegardés.** `Screen::Home` et
  `Screen::LoadPath` existent, sont dessinés et gèrent leurs touches, mais **rien
  ne les assigne jamais** : l'application démarre en dur sur `Screen::TemplateSelector`
  (`app.rs`), et les templates sont codés en dur (`built_in_templates()`).
  `Greyscale_Diffusion_L` est donc inaccessible depuis l'interface. Gap
  **préexistant**, hors périmètre de cette mission, mais il bloquait le scénario
  demandé : la validation est passée par un template temporaire construit depuis
  `Models/Greyscale_Diffusion_L/config_file`, **retiré depuis** (l'arbre ne contient
  que le correctif de cadence, cf. `git diff` vide hors commits). À traiter dans une
  mission dédiée — c'est une touche de plus à câbler, pas une refonte.
- **Effet de bord de ce détour** : passer par le sélecteur de template fait
  réécrire le bloc `inference` du `config_file` du modèle avec les valeurs par
  défaut. Les deux `config_file` touchés ont été restaurés (`git checkout`).
  Le `latest.ckpt` copié a été supprimé (`Models/` n'est pas gitignoré — ne pas
  committer 5,5 Mo de poids par mégarde).
- **Grille multi-chemins non implémentée** (bonus explicite de la spec). Le buffer
  et le shader s'y prêtent : il suffirait d'une grille de `k × 2W`.
- **Fenêtre occultée = pas de rendu.** `WindowEvent::Occluded` coupe les frames
  (comportement préexistant, et souhaitable). Conséquence pratique : une capture
  d'écran est un mauvais moyen de prouver que le visualiseur tourne si sa fenêtre
  est derrière une autre — d'où le comptage par sonde du Run B plutôt qu'une capture.
- **Échelle d'affichage fixe.** Le shader mappe `[-1, 1] → [0, 1]`. Aux t élevés le
  latent sort largement de cette plage et sature ; x̂₀ étant clippé par construction,
  le panneau de droite reste toujours lisible. C'est précisément ce qui rend la vue
  x̂₀ intéressante en début de chaîne.
- La moyenne sur beaucoup de chemins délave l'image finale (propriété du sampler,
  pas de l'affichage) : les runs à 60/200 chemins de la validation ont été choisis
  pour leur durée, pas pour leur rendu.
