# batLab

Framework de deep learning from scratch en Rust + wgpu (compute shaders WGSL), avec TUI ratatui. Cas d'usage principal : modèles de diffusion DDPM sur CIFAR-10. Crates : `bat_building` (cœur : model/layers/shaders/training/tui/visualiser) et `main` (point d'entrée).

## Lancer / valider un entraînement sans le TUI

Le binaire est un TUI interactif plein écran — impossible à scripter directement. Pour toute validation automatisée (agents, CI, tests de convergence), utiliser le chemin headless, qui réutilise `run_training` de production à l'identique :

```bash
cargo run -p main -- --headless-train <model> --steps N --dataset <path> [--lr F] [--batch N]
# ex. : cargo run -p main -- --headless-train Greyscale_Diffusion --steps 2000 --dataset datasets/cifar10_grey.batraw
```

Propriétés : ne réécrit jamais le `config_file` du modèle, écrit son checkpoint dans un fichier scratch (n'écrase pas les poids sauvegardés), force `load_checkpoint=false`. Marqué DEV/CI dans `main/src/main.rs`, inatteignable depuis le TUI.

Pour piloter le vrai TUI malgré tout (test end-to-end) : le lancer dans un pane tmux dédié et le piloter via `tmux send-keys`.

## Points d'attention

- **Piège du workspace** : `cargo build --release` à la racine ne reconstruit PAS le binaire `main` (il ne bâtit que la lib racine). Toujours `cargo build/run --release -p main`. Après un changement de flag CLI, vérifier la bannière du run (le binaire réémet sa config parsée).
- **Optimiseur** : Adam (`--optimizer adam --lr 1e-3`) converge ~20x plus vite que SGD en nombre de pas, surcoût < 1 % (voir `OPTIMIZER_ADAM.md`). L'init He est disponible (`--weight-init he`) mais n'a pas montré de gain (GroupNorm neutralise l'échelle en aval).
- **Lecture des métriques de diffusion** : la MSE sur ε inverse l'importance des tranches — convertir en erreur x₀ (facteur ᾱ/(1−ᾱ)) avant de conclure. Le contenu global d'une image se décide à t haut ; une loss non pondérée y écrase le gradient (~1e5) et produit du mode collapse (voir `SCALE_UNET.md`). Baselines triviales : `tools/trivial_baselines.py`.

- Les checkpoints antérieurs aux fixes du pipeline (padding `Same`, normalisation [-1,1], conditionnement temporel) sont invalidés — toujours réentraîner from scratch, ne pas charger d'anciens `.ckpt`.
- Un modèle de diffusion DOIT être conditionné sur le timestep : `input_size.z > output.z` (les canaux excédentaires reçoivent l'embedding temporel). Sans ça, ε̂ dégénère et l'échantillonnage explose en blanc saturé (voir `INSIGHTS_TRAINING.md`).
- Chaque run d'entraînement écrit un `*_metrics.jsonl` à côté du checkpoint (loss par tranche de t, stats ε̂ vs ε, trajectoires de débruitage). `--headless-sample <model> --ckpt <path>` génère des images + trajectoire depuis un checkpoint sans entraîner. Une loss batch qui décroît ne suffit PAS — vérifier la loss par tranche de t (une loss élevée à t bas = modèle qui n'utilise pas t).
- Format dataset `.batraw` : magic `BATRAW2` = payload en [-1,1] ; les fichiers `BATRAW1` ([0,1]) restent lisibles et sont rééchelonnés au chargement.
- Tests de non-régression du pipeline : `bat_building/src/model/audit_tests.rs` (`cargo test`). Ne pas les affaiblir pour les faire passer.
- macOS : l'event loop winit du visualiseur doit vivre sur le main thread (le TUI et l'entraînement tournent sur un worker) — ne pas réintroduire de `EventLoop::new()` dans un thread secondaire.
