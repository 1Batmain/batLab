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

- Les checkpoints antérieurs aux fixes du pipeline (padding `Same`, normalisation [-1,1]) sont invalidés — toujours réentraîner from scratch, ne pas charger d'anciens `.ckpt`.
- Format dataset `.batraw` : magic `BATRAW2` = payload en [-1,1] ; les fichiers `BATRAW1` ([0,1]) restent lisibles et sont rééchelonnés au chargement.
- Tests de non-régression du pipeline : `bat_building/src/model/audit_tests.rs` (`cargo test`). Ne pas les affaiblir pour les faire passer.
- macOS : l'event loop winit du visualiseur doit vivre sur le main thread (le TUI et l'entraînement tournent sur un worker) — ne pas réintroduire de `EventLoop::new()` dans un thread secondaire.
