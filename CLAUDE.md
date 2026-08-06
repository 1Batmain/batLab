# batLab

Framework de deep learning from scratch en Rust + wgpu (compute shaders WGSL), avec TUI ratatui. Cas d'usage principal : modèles de diffusion DDPM sur CIFAR-10.

## Arborescence

```
Cargo.toml          workspace pur (aucun code à la racine)
crates/
  batlab-core/      paquet `batlab_core` — LE MOTEUR : gpu_context, config, model/layers/shaders,
                    training, inférence (sampler, perpetual), live_frame. wgpu et rien d'autre côté graphique.
  batlab-ui/        paquet `batlab_ui` — TUI ratatui, fenêtre visualiseur winit, disposition sur disque (storage)
  batlab/           paquet `batlab` — le binaire (CLI, modes headless DEV/CI, orchestration)
Models/             un dossier par modèle : config_file + pretrained_weights/
datasets/           .batraw (gitignorés) + cifar_to_raw.py
tools/              analyse et planches (Python)
bench/              bancs optimiseur et pondération de loss (shell + Python)
blind_tests/        suite de tests à l'aveugle du régime flux
docs/reports/       rapports de mission archivés (+ INDEX.md, table ancienne→nouvelle structure)
docs/gallery/       planches et figures des campagnes
```

Le binaire écrit ses sorties de run à la racine : `perpetual_samples/` (`--headless-perpetual` sans `--out`) et `weighting_samples/` (banc de pondération). Ces deux répertoires sont **gitignorés** — les planches retenues sont commitées sous `docs/gallery/`.

### La frontière moteur / interface — à ne pas franchir

wgpu a été choisi pour une raison : le moteur doit tourner **dans le navigateur du visiteur, sur SA carte graphique** (WebGPU côté client, aucun GPU serveur). D'où la règle, vérifiable en une commande :

```bash
# batlab-core ne doit dépendre NI de ratatui, NI de crossterm, NI de winit
awk '/^\[dependencies\]/{f=1;next}/^\[/{f=0}f' crates/batlab-core/Cargo.toml | grep -E '^(ratatui|crossterm|winit)' && echo ÉCHEC || echo OK
cargo check -p batlab_core
```

Concrètement :

- **Tout le chemin d'inférence vit dans `batlab_core`** : `config` (le schéma du `config_file`), le décodage de checkpoint, `compose_diffusion_input`, `reverse_step`/`sample_diffusion`, `PerpetualDrift`. L'entraînement y reste aussi, mais c'est l'inférence qui doit être irréprochablement découplée.
- **Le moteur ne connaît pas le système de fichiers.** Il prend des octets : `ModelConfig::from_json_bytes(&[u8])`, `Model::load_checkpoint_bytes(&[u8])`. Les variantes `load_checkpoint(path)` / `save_checkpoint(path)` ne sont que de minces enveloppes `fs` pour le CLI — ne jamais réintroduire de lecture de fichier plus profond. (`MetricsLogger` écrit un JSONL : c'est de l'instrumentation d'entraînement, hors chemin d'inférence.)
- **`LiveFrame` est la couture visualiseur ↔ sampler** : le moteur compose la frame dans un buffer GPU (`live_frame.rs`, wgpu seul) ; l'affichage — fenêtre winit aujourd'hui, canvas demain — se contente de lire ce buffer, côté `batlab_ui`.
- **La disposition sur disque** (`Models/<name>/config_file`, `datasets/`, `project_root()`) est une décision d'hôte : elle vit dans `batlab_ui::storage`, pas dans le moteur.
- La flèche de dépendance ne pointe que dans un sens : `batlab` → `batlab_ui` → `batlab_core`. Jamais l'inverse.

Un vrai build `wasm32` est une mission future — aujourd'hui on veut seulement que la frontière soit propre.

## Le flow du TUI : le modèle d'abord, l'action ensuite

```
Ouverture → LISTE DES MODÈLES              (l'écran d'accueil)
  chaque entrée : nom · géométrie · nb de couches · checkpoints
  dernière ligne : « New model (from template) » → flux template
  → Entrée sur un modèle → MENU D'ACTIONS
       Train / Infer / Perpetual  → choix des poids → formulaire du mode → Monitor
       Rename / Delete            → le manager
```

`Esc` remonte d'exactement un cran depuis chaque écran, et ne quitte que depuis
les deux racines : la liste (rien au-dessus) et le moniteur (un run en cours, que
quitter termine). `[e]` depuis le menu d'actions ouvre le constructeur de
couches, `[i]` dedans ouvre la géométrie d'entrée.

Le manager, côté contrat observable :

- **Renommer** déplace `Models/<ancien>/` → `Models/<nouveau>/` **et** réécrit le
  `config_file` — `model_name` plus tout chemin de checkpoint qui pointait dans
  le dossier du modèle. Si la réécriture échoue, le renommage est défait.
- **Supprimer** exige le nom retapé à l'identique, puis efface le dossier entier.
  Le chemin visé doit être un **enfant direct** de `Models/` (les deux côtés
  canonicalisés) : un lien symbolique est refusé, pas suivi.
- Les deux **refusent un modèle dont ce processus tient un run**. Ce garde-fou est
  de l'état TUI : **il ne franchit pas la frontière de processus** — un second
  batlab ou un `--headless-train` dans un autre shell reste invisible. Non résolu,
  documenté sur `App::model_run_in_progress`.
- Sur les deux écrans du manager, **toute touche imprimable est du texte**, `q`
  compris (un modèle peut s'appeler `q-experiment`) ; `Esc` est la sortie.
- **Tout modèle créé par un template est conditionnable sur t** (`input_size.z >
  output.z`). Les deux templates ont livré l'inverse (1→1 et 3→3, la géométrie de
  `Models/Greyscale_Diffusion_broken`) jusqu'à ce qu'un test aveugle lise le
  `config_file` écrit. Ils dérivent maintenant leurs dims d'un seul
  `diffusion_unet(signal, temporel)` — ne pas y réintroduire de dims à la main.
  Gardé des deux côtés : `config::tests` sur `built_in_templates()`, et
  `every_model_created_from_a_template_is_conditionable_on_the_timestep` sur le
  fichier réellement écrit.
- **Un checkpoint est un `.ckpt`**, non caché, dans `pretrained_weights/` — le
  `latest_metrics.jsonl` que tout entraînement dépose à côté n'en est pas un.
  Les deux listages (compteur de la liste, sélecteur de poids) passent par
  `is_checkpoint_file` ; le comptaient tous les deux avant. Ne pas relâcher en
  « tout fichier », ni retirer l'exclusion des fichiers cachés par-dessus
  l'extension (les AppleDouble macOS `._latest.ckpt` la passeraient).

Un écran dessiné mais jamais assigné est le bug récurrent de ce dépôt (trois fois :
`LoadPath`, le mode `Perpetual`, `InputSize`). `crates/batlab-ui/src/tui/nav_tests.rs`
parcourt tous les écrans à la touche depuis la porte d'entrée et exige d'avoir vu
`Screen::ALL` — ajouter une variante sans la câbler fait échouer la suite.

Le rapport de mission, avec le contrat observable complet et le parcours e2e
déroulé : `docs/reports/MODEL_MANAGER.md`.

## Lancer / valider un entraînement sans le TUI

Le binaire est un TUI interactif plein écran — impossible à scripter directement. Pour toute validation automatisée (agents, CI, tests de convergence), utiliser le chemin headless, qui réutilise `run_training` de production à l'identique :

```bash
cargo run -p batlab -- --headless-train <model> --steps N --dataset <path> [--lr F] [--batch N]
# ex. : cargo run -p batlab -- --headless-train Greyscale_Diffusion --steps 2000 --dataset datasets/cifar10_grey.batraw
```

Propriétés : ne réécrit jamais le `config_file` du modèle, écrit son checkpoint dans un fichier scratch (n'écrase pas les poids sauvegardés), force `load_checkpoint=false`. Marqué DEV/CI dans `crates/batlab/src/main.rs`, inatteignable depuis le TUI.

Pour piloter le vrai TUI malgré tout (test end-to-end) : le lancer dans un pane tmux dédié et le piloter via `tmux send-keys`.

## Tests à l'aveugle (protocole projet)

Pour tout jalon à invariants comportementaux, les tests de l'agent d'implémentation ne suffisent pas : un agent qui a lu/écrit le code produit des tests-miroirs (ils affirment ce que le code fait, pas ce qu'il devrait faire — le mutation testing ne protège pas de ce biais). L'architecte lance donc, après l'implémentation, un **agent de test aveugle** : spec + contrats publics fournis, **interdiction de lire l'implémentation** (il peut compiler, exécuter, observer les sorties). Un écart se signale en citant la spec, jamais en éditant un test. Les deux suites coexistent ; leur désaccord est le signal.

La suite existante : `./blind_tests/run.sh` (`BLIND_BASELINE=1` ajoute la non-régression bit-à-bit contre un binaire d'un vieux commit ; ce chemin-là reconstruit une arborescence **pré-restructuration**, où le paquet binaire s'appelait encore `main` — le script résout le nom par arborescence, ne pas le figer).

## Points d'attention

- **Toujours cibler le bon paquet** : `cargo build/run --release -p batlab` pour le binaire, `-p batlab_core` pour le moteur, `-p batlab_ui` pour l'interface. (Le piège historique « un build à la racine ne construit que la lib racine » a disparu avec le paquet racine `batBuilder`, mais un `-p` explicite reste plus sûr.) Après un changement de flag CLI, vérifier la bannière du run — le binaire réémet sa config parsée, et `bench/optimizer/lib.sh:assert_flag` en fait un test.
- **Optimiseur** : Adam (`--optimizer adam --lr 1e-3`) converge ~20x plus vite que SGD en nombre de pas, surcoût < 1 % (voir `docs/reports/OPTIMIZER_ADAM.md`). L'init He est disponible (`--weight-init he`) mais n'a pas montré de gain (GroupNorm neutralise l'échelle en aval).
- **Lecture des métriques de diffusion** : la MSE sur ε inverse l'importance des tranches — convertir en erreur x₀ (facteur ᾱ/(1−ᾱ)) avant de conclure. Baselines triviales : `tools/trivial_baselines.py`. Attention, le « déséquilibre 1e5 » de `docs/reports/SCALE_UNET.md` est en unités x₀ ; **en unités ε, où le gradient est réellement calculé, il vaut ≈5,7×** — et pondérer la loss ne rend que ce que 5,7× peut rendre (`docs/reports/LOSS_WEIGHTING.md` : ×1,20 sur le haut-t, NO-GO).
- **Pondération de la loss** : `--loss-weighting uniform|snr [--snr-gamma F]` (défaut `uniform`, bit-à-bit l'ancien tirage). Implémentée par biais du tirage de t, pas dans un shader. Sur ce schedule, **γ=5 (valeur de la littérature) ne redistribue presque rien** — `SNR(t) ≤ 5` dès t=33 ; utiliser γ≈1.
- **`inter_seed_std` n'est pas un critère de diversité lisible seul** : il est proportionnel à `--magnitude` (±6 % sur 0,3/0,6/1,0), donc il mesure surtout le bruit du sampler non débruité. Le lire à magnitude fixée, avec `intra_image_std` (doit approcher 0,206 par le bas, pas le dépasser) et `banding_ratio`.

- **Graines de bruit : jamais `seed ^ index`.** Les images générées ont été des bandes horizontales pendant toute la campagne parce que le sampler XORait le pas dans la graine (`path_seed ^ diffusion_step`) pendant que le champ de bruit XORait l'index pixel : les deux se composent, et les 256 pas retiraient un seul champ permuté par `index ^ d ^ d'` (`docs/reports/ANISOTROPY_HUNT.md`). Chaque champ isolé restait isotrope — seule leur **somme** s'effondrait, d'où un entraînement et une sonde impeccables face à une génération morte. Passer par `gaussian_at` (avalanche de la graine, *puis* flux additif de l'index) ; l'ordre compte, l'inverse commute et donne un champ constant sur les anti-diagonales. Gardé par `injected_noise_over_reverse_chain_is_isotropic`.
- Les checkpoints antérieurs aux fixes du pipeline (padding `Same`, normalisation [-1,1], conditionnement temporel) sont invalidés — toujours réentraîner from scratch, ne pas charger d'anciens `.ckpt`.
- Un modèle de diffusion DOIT être conditionné sur le timestep : `input_size.z > output.z` (les canaux excédentaires reçoivent l'embedding temporel). Sans ça, ε̂ dégénère et l'échantillonnage explose en blanc saturé (voir `docs/reports/INSIGHTS_TRAINING.md`).
- Chaque run d'entraînement écrit un `*_metrics.jsonl` à côté du checkpoint (loss par tranche de t, stats ε̂ vs ε, trajectoires de débruitage). `--headless-sample <model> --ckpt <path>` génère des images + trajectoire depuis un checkpoint sans entraîner. Une loss batch qui décroît ne suffit PAS — vérifier la loss par tranche de t (une loss élevée à t bas = modèle qui n'utilise pas t).
- Format dataset `.batraw` : magic `BATRAW2` = payload en [-1,1] ; les fichiers `BATRAW1` ([0,1]) restent lisibles et sont rééchelonnés au chargement.
- Tests de non-régression du pipeline : `crates/batlab-core/src/model/audit_tests.rs` (`cargo test`). Ne pas les affaiblir pour les faire passer.
- **La racine de stockage s'injecte, elle ne se déduit pas.** `batlab_ui::storage::Storage` porte la racine de tous les chemins de données (`Models/`, `datasets/`, `perpetual_samples/`) ; `Storage::at(chemin)` en construit une ailleurs, et `App::with_storage` la fait descendre dans tout le TUI. **Tout test qui touche au stockage passe par `TempRoot`** — c'est ce qui a fait tomber la limite « `cargo test` réécrit `Models/Stable_Diffusion/config_file` » de `docs/reports/PERPETUAL_INFERENCE.md` §5. Critère de recette permanent : `cargo test --workspace` puis `git status` **propre**.
  Le défaut (`Storage::default()`, et les fonctions libres du module que le CLI utilise) reste le workspace, trouvé en remontant jusqu'au `Cargo.toml` portant `[workspace]` : ne pas le réécrire en un nombre fixe de `parent()` — déplacer un crate ferait alors pointer `Models/` ailleurs, **sans erreur**, juste des listes vides. Gardé par `project_root_is_the_workspace_that_holds_models_and_datasets`.
  **`BATLAB_ROOT=<dir>`** force cette racine par défaut pour tout le processus — c'est la façon de dérouler le vrai TUI end-to-end sans écrire dans le `Models/` du dépôt.
- macOS : l'event loop winit du visualiseur doit vivre sur le main thread (le TUI et l'entraînement tournent sur un worker) — ne pas réintroduire de `EventLoop::new()` dans un thread secondaire.
- Les rapports sous `docs/reports/` sont des **archives** : leur texte cite les anciens chemins (`bat_building/src/…` — qui couvrait alors moteur ET interface —, `main/src/main.rs`, `perpetual_samples/…`) et n'a pas été réécrit. La table de correspondance est dans `docs/reports/INDEX.md`.
