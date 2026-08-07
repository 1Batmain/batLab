# Rapports de mission — index

Chaque mission de la campagne a laissé un rapport. **Ces fichiers sont des archives :
leur contenu n'a pas été réécrit lors de la restructuration du dépôt.** Ils citent donc
les chemins qui existaient au moment où ils ont été rédigés (`bat_building/src/…`,
`main/src/main.rs`, `cargo run -p main`, `perpetual_samples/…`). La table de
correspondance ci-dessous suffit à les relire ; le CLAUDE.md racine, lui, est à jour.

Pour une lecture en une passe, commencer par [SYNTHESE_CAMPAGNE.md](SYNTHESE_CAMPAGNE.md).

## Table de correspondance ancienne → nouvelle structure

| Cité dans les rapports | Aujourd'hui |
| --- | --- |
| `bat_building/` | scindé : `crates/batlab-core/` (moteur) et `crates/batlab-ui/` (TUI, visualiseur, storage) |
| `bat_building/src/model/…` | `crates/batlab-core/src/model/…` |
| `bat_building/src/tui/…` | `crates/batlab-ui/src/tui/…` |
| `bat_building/src/tui/storage.rs` | `crates/batlab-ui/src/storage.rs` |
| `bat_building/src/tui/app.rs` | scindé : schéma sérialisable → `crates/batlab-core/src/config.rs`, état d'écran → `crates/batlab-ui/src/tui/app.rs` |
| `bat_building/src/visualiser/` | `crates/batlab-ui/src/visualiser/` |
| `bat_building/src/visualiser/live_frame.rs` | `crates/batlab-core/src/live_frame.rs` (la couture reste côté moteur) |
| `main/`, `main/src/main.rs` | `crates/batlab/`, `crates/batlab/src/main.rs` |
| paquet `bat_building` (`-p bat_building`) | paquets `batlab_core` et `batlab_ui` |
| paquet `main` (`-p main`) | paquet `batlab` (`-p batlab`) |
| binaire `target/release/main` | `target/release/batlab` |
| paquet racine `batBuilder`, `src/lib.rs` | supprimé (la racine est un workspace pur) |
| `<RAPPORT>.md` à la racine | `docs/reports/<RAPPORT>.md` |
| `insights_samples/` | `docs/gallery/insights/` |
| `scale_samples/` | `docs/gallery/scale/` |
| `hunt_samples/` | `docs/gallery/hunt/` |
| `climb_samples/` | `docs/gallery/climb/` |
| `weighting_samples/` | `docs/gallery/weighting/` (le dossier racine reste la **sortie** du banc) |
| `perpetual_samples/` | `docs/gallery/perpetual/` (le dossier racine reste la **sortie** du binaire) |
| `flux_samples/` | `docs/gallery/flux/` |
| `color_samples/` | `docs/gallery/color/` |
| `synthese_assets/` | `docs/gallery/synthese/` |

Inchangés : `Models/`, `datasets/`, `tools/`, `bench/`, `blind_tests/`, tous les flags
CLI (`--headless-train`, `--headless-sample`, `--headless-perpetual`, `--regime`,
`--t-star`…), le format des dumps `.f32`, des `.batraw` et des `*_metrics.jsonl`.

## Les rapports

### Vue d'ensemble

- **[SYNTHESE_CAMPAGNE.md](SYNTHESE_CAMPAGNE.md)** — la campagne du 20 juillet au 5 août
  2026 en un document : les bugs et leurs mécanismes, les chiffres, les planches, ce qui
  reste et ce qui n'est pas prouvé. Le point d'entrée.

### Le pipeline d'entraînement — diagnostic et réparation

- **[AUDIT_TRAINING.md](AUDIT_TRAINING.md)** — audit du pipeline, diagnostic seul, aucune
  modification. Sept findings, dont le padding fantôme et la normalisation `[0,1]`.
- **[FIX_TRAINING.md](FIX_TRAINING.md)** — la réparation : convolution `Same`, données en
  `[-1,1]`, schedule de bruit recalibré, tirage du timestep.
- **[INSIGHTS_TRAINING.md](INSIGHTS_TRAINING.md)** — le blanc saturé expliqué : un modèle
  de diffusion non conditionné sur `t` ne peut pas converger. Planches avant/après.

### Échelle, optimiseur, performance

- **[SCALE_UNET.md](SCALE_UNET.md)** — agrandir le U-Net (`Greyscale_Diffusion_L`) ; le
  « déséquilibre 1e5 » et son unité (x₀ vs ε).
- **[OPTIMIZER_ADAM.md](OPTIMIZER_ADAM.md)** — Adam contre SGD, comparatif apparié :
  ~20× en nombre de pas pour moins de 1 % de surcoût.
- **[LOSS_WEIGHTING.md](LOSS_WEIGHTING.md)** — pondérer la loss par tirage biaisé des
  timesteps. Verdict NO-GO, et pourquoi γ=5 ne redistribue rien sur ce schedule.
- **[PERF_GROUP_NORM.md](PERF_GROUP_NORM.md)** — optimisation de `group_norm`.
- **[PERF_CONVOLUTION.md](PERF_CONVOLUTION.md)** — optimisation de la convolution ; le
  gain local qui ne se transmet pas au pas complet.

### Architecture du modèle

- **[ATTENTION.md](ATTENTION.md)** — self-attention spatiale au goulot 8×8 : les quatre
  projections en un seul tenseur (pour que l'optimiseur la traite comme une
  convolution), le forward multi-passes, et **la limite WebGPU de 8 storage buffers**
  — qui échoue de façon totale et silencieuse, loss et gradients à zéro.

### La génération

- **[ANISOTROPY_HUNT.md](ANISOTROPY_HUNT.md)** — pourquoi les images générées étaient des
  bandes horizontales : deux XOR isotropes qui se composent. Cause racine, correctif,
  guérison prouvée.

### Le mode Perpetual

- **[PERPETUAL_INFERENCE.md](PERPETUAL_INFERENCE.md)** — l'inférence perpétuelle : errance
  et respiration.
- **[PERPETUAL_SMOOTHNESS.md](PERPETUAL_SMOOTHNESS.md)** — le polissage du mode.
- **[CLIMB_COHERENCE.md](CLIMB_COHERENCE.md)** — la remontée cesse de friser ; pourquoi
  les mesures passent par des dumps f32 et non des PNG (§6).
- **[PERPETUAL_FLUX.md](PERPETUAL_FLUX.md)** — le régime « flux » : churn stationnaire
  sans phase, cadran `t*`, contrat CLI durci.
- **[MISSION_BLIND_TEST.md](MISSION_BLIND_TEST.md)** — l'ordre de mission de l'agent
  aveugle (spec du régime flux, sans une ligne d'implémentation).
- **[BLIND_TEST_FLUX.md](BLIND_TEST_FLUX.md)** — son verdict : 9/9 PASS après trois
  passes, dont un correctif prouvé label-only.

### Visualiseur

- **[INFER_VIZ.md](INFER_VIZ.md)** — visualisation live du débruitage pendant l'inférence.
- **[CRASH_VISUALISER.md](CRASH_VISUALISER.md)** — le crash macOS : l'event loop winit
  doit vivre sur le main thread.

### Couleur

- **[COLOR_MODEL.md](COLOR_MODEL.md)** — premier modèle de diffusion couleur (CIFAR-10 RGB
  32×32) : dataset RGB, configs `Color_Diffusion_L`/`XL`, chemin couleur validé
  bout-en-bout.

### Interface

- **[MODEL_MANAGER.md](MODEL_MANAGER.md)** — navigation modèle-centrée (la liste des
  modèles en écran d'accueil, un menu d'actions par modèle), renommage et suppression,
  racine de stockage injectable (`BATLAB_ROOT`) — et la fin de « `cargo test` réécrit
  un fichier suivi ». Contient le contrat observable du manager.
- **[UX_NAV.md](UX_NAV.md)** — quatre retours après usage réel : le vol de focus au
  lancement (cause racine, `activateIgnoringOtherApps` de winit, que la policy
  `Accessory` ne couvre pas), les poids pré-entraînés en défaut avec « from random »
  en opt-out, le fil d'Ariane `Model › Action › Weights › Parameters › Run` et la
  navigation `←`/`→`, l'aide contextuelle sourcée dans ces rapports. Contient le
  contrat observable de la navigation.
- **[IMG2IMG_DRIFT.md](IMG2IMG_DRIFT.md)** — le mode Perpetual devient une dérive
  img2img : départ sur une VRAIE image du dataset, fenêtre x̂₀ seule par défaut
  (`[x]` pour la double vue) — et la cause racine du « pause » (la remontée
  n'appelait pas le modèle, x̂₀ figé 100 % du temps, mesuré 294/294 frames avant
  contre 0/294 après). Aussi : l'aide de la liste des modèles affiche
  l'architecture, champ réceptif et verdict compris. Contient le contrat
  observable de la dérive.

### Restructuration

- **[RESTRUCTURE.md](RESTRUCTURE.md)** — ce qui a bougé, la table ci-dessus, et les pièges
  rencontrés en la faisant.
