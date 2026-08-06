# Restructuration du dépôt — rapport de mission

Branche `restructure` · 6 août 2026 · Plateforme macOS (Darwin 25.4.0, Apple M5 Pro)

Le dépôt avait grandi par accrétion : un paquet racine vestigial, un crate cœur qui
contenait aussi bien les shaders que l'affichage terminal, 19 rapports de mission à la
racine, et huit dossiers de planches éparpillés autour. Cette mission range — et, en
cours de route, pose la frontière moteur/interface qui conditionne le portage web.

**État final** : `cargo test --workspace` 144 verts, `./blind_tests/run.sh` 9/9 PASS,
outils Python et bancs adaptés, aucun flag CLI ni format d'artefact modifié.

---

## 1. Ce qui a bougé

### 1.1 Les crates

| Avant | Après |
| --- | --- |
| paquet racine `batBuilder` + `src/lib.rs` | supprimé — la racine est un workspace pur |
| `bat_building/` (moteur **et** interface) | `crates/batlab-core/` (moteur) + `crates/batlab-ui/` (interface) |
| `main/` | `crates/batlab/` |
| `-p bat_building`, `-p main` | `-p batlab_core`, `-p batlab_ui`, `-p batlab` |
| `target/release/main` | `target/release/batlab` |

Le paquet racine ne faisait que réexporter `bat_building` et personne n'en dépendait. Sa
disparition supprime au passage le piège documenté depuis `OPTIMIZER_ADAM.md` §2.4 —
« `cargo build --release` à la racine ne construit que la lib racine ». Sans paquet
racine, un build à la racine construit bien les trois crates. Le `-p` explicite reste
recommandé (`bench/optimizer/lib.sh` garde d'ailleurs sa seconde défense, `assert_flag`,
qui était la vraie protection : la bannière du run doit réémettre le flag).

Les `Cargo.lock` des crates membres, inutilisés dans un workspace, sont supprimés.

### 1.2 La frontière moteur / interface

Demandée en cours de mission, et structurante : wgpu a été choisi pour faire tourner
**l'inférence dans le navigateur du visiteur, sur sa propre carte graphique** (WebGPU
côté client). Un moteur qui dépend de ratatui, crossterm et winit ne peut pas y aller.

| Fichier | Destination | Pourquoi |
| --- | --- | --- |
| `tui/app.rs` (moitié sérialisable) | `batlab-core/src/config.rs` | décrire un modèle doit survivre sans terminal |
| `tui/app.rs` (écrans, formulaires, `App`) | `batlab-ui/src/tui/app.rs` | état de terminal, et rien d'autre |
| `tui/{mod,ui,events,visualiser_control}.rs` | `batlab-ui/src/tui/` | ratatui + crossterm |
| `visualiser/mod.rs` | `batlab-ui/src/visualiser/` | winit |
| `visualiser/live_frame.rs` | `batlab-core/src/live_frame.rs` | wgpu pur — c'est la **couture** |
| `tui/storage.rs` | `batlab-ui/src/storage.rs` | le disque est une décision d'hôte |

Le chemin d'inférence complet reste dans le moteur, et cesse de dépendre du système de
fichiers :

```rust
Model::checkpoint_bytes() -> Result<Vec<u8>, ModelError>   // nouveau
Model::load_checkpoint_bytes(&[u8])                        // nouveau
Model::{save,load}_checkpoint(path)                        // enveloppes fs minces
ModelConfig::{from_json_bytes, to_json_bytes}              // nouveau
```

`batlab_ui::storage` passe lui-même par ces API bytes plutôt que par `serde_json` en
direct : la porte que prendra un build wasm doit être la **seule** porte, sinon elle se
referme sans qu'on s'en aperçoive.

Critère vérifiable :

```bash
awk '/^\[dependencies\]/{f=1;next}/^\[/{f=0}f' crates/batlab-core/Cargo.toml \
  | grep -E '^(ratatui|crossterm|winit)'   # → aucune correspondance
cargo check -p batlab_core                 # → vert
cargo tree -p batlab_core                  # → aucune des trois, même transitivement
```

Un build `wasm32` réel reste une mission à part entière : l'état atteint ici n'est pas
« ça tourne dans le navigateur », c'est « la frontière est propre et le restera ».

### 1.3 La documentation et les planches

Les 19 rapports quittent la racine pour `docs/reports/`, les huit dossiers `*_samples/`
et `synthese_assets/` deviennent `docs/gallery/<campagne>/`. Table complète dans
[`INDEX.md`](INDEX.md).

Deux dossiers gardent leur nom **à la racine** parce que ce sont des répertoires de
**sortie**, pas des archives : `perpetual_samples/` (défaut de `--headless-perpetual`) et
`weighting_samples/` (banc de pondération). Ils sont désormais gitignorés ; les planches
retenues de ces campagnes sont commitées sous `docs/gallery/`.

Un `README.md` de vitrine apparaît à la racine.

---

## 2. Les pièges rencontrés

### 2.1 `project_root()` — le déplacement qui ne lève aucune erreur

Le plus dangereux, et le plus silencieux. `tui::storage::project_root()` remontait d'**un
seul** `parent()` depuis `CARGO_MANIFEST_DIR` :

```rust
PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().expect(…)
```

C'était correct tant que le crate était `<racine>/bat_building`. Il est devenu
`<racine>/crates/batlab-core`, puis `<racine>/crates/batlab-ui` : la racine calculée
serait devenue `crates/`. Rien n'aurait planté — `models_dir()` fait `create_dir_all`,
donc un `crates/Models/` vide aurait été créé, la liste des modèles serait revenue vide,
et le TUI aurait affiché « No model configs found in Models/ ». Un symptôme qu'on
attribue à son environnement, pas à un refactor.

Correctif : remonter jusqu'au `Cargo.toml` qui déclare `[workspace]`, ce qui est robuste
à la profondeur, plus un test qui ancre le résultat sur la présence de `Models/` et
`crates/` (`project_root_is_the_workspace_that_holds_models_and_datasets`).

### 2.2 `perpetual_samples/` était deux choses à la fois

Le dossier contenait des planches commitées **et** servait de répertoire de sortie au
binaire. Le déplacer en bloc vers `docs/gallery/` aurait changé le chemin de sortie par
défaut de `--headless-perpetual` — or `blind_tests/baseline.sh` compare octet à octet les
PNG produits par le binaire courant et par un binaire reconstruit d'un **vieux commit**,
qui écrit forcément dans `perpetual_samples/`. La non-régression forte se serait cassée
en silence, en comparant deux répertoires différents.

Choix : le chemin de sortie ne bouge pas, le dossier racine devient gitignoré, et seules
les planches retenues sont commitées sous `docs/gallery/perpetual/`. Corollaire :
l'hygiène de fin de run passe de `git checkout -- perpetual_samples` (qui ne restaurerait
plus rien) à un `rm -rf`.

### 2.3 La baseline reconstruit une arborescence pré-restructuration

`blind_tests/baseline.sh` fait `git worktree add` sur un commit ancien puis
`cargo build --release -p main`. Ce `-p main` doit **rester** `main` de ce côté-là, et
devenir `batlab` du côté courant. Le nom est donc déduit de l'arborescence :

```bash
pkg_of() { if [ -d "$1/crates/batlab" ]; then echo batlab; else echo main; fi; }
```

Figer l'un ou l'autre aurait cassé le script d'un côté ou de l'autre de la bascule.

### 2.4 Une propriété rouge qui ne l'était pas

Premier passage de la suite à l'aveugle avec un horizon raccourci (`ACTIONS=400`,
`TSTARS="16 64 224"`) pour aller vite : **P3 en FAIL**. De quoi croire que la
restructuration avait cassé la continuité de la dérive.

Le détail disait autre chose : `corr(k, k+300) = nan`. À 400 frames, dont une part
consommée par le rodage, il n'en reste pas assez pour mesurer un décalage de 300 —
la corrélation est vide, pas basse. La propriété exige son horizon nominal ; à
`ACTIONS=3000` (le défaut), 9/9 PASS, avant comme après la scission moteur/UI.

La leçon vaut au-delà du cas : raccourcir un run pour « juste vérifier la plomberie »
change le verdict des propriétés qui portent sur le long terme. Lire le détail avant de
conclure à une régression — le FAIL était réel, sa cause était le protocole de mesure.

### 2.5 Visibilités : ce qui était `fn` privé dans un module devient inter-crate

En sortant `app.rs` de son crate, quatre fonctions privées (`built_in_templates`,
`diffusion_template`, `greyscale_diffusion_template`, `update_layer_dim_input`) et deux
`const fn` associées à `PerpetualConfig` sont devenues inaccessibles à leurs appelants,
restés côté UI. Elles passent `pub`. Le compilateur les a toutes signalées ; aucune
n'était un choix d'encapsulation délibéré, juste le résultat d'avoir vécu dans le même
fichier.

### 2.6 Le run en cours dans un autre worktree

Un entraînement couleur tournait dans `worktrees/color-model` sur l'ancienne
arborescence. Rien n'a été touché de ce côté ; `Models/` et `datasets/` gardent leur nom
et leur place, donc son checkpoint se recopiera tel quel dans la nouvelle structure.

---

## 3. Ce qui n'a PAS changé

Volontairement, parce que ce sont des contrats :

- tous les flags CLI : `--headless-train`, `--headless-sample`, `--headless-perpetual`,
  `--regime`, `--t-r`/`--t-star`/`--depth`, `--actions`, `--frames`, `--dump`, `--out`… ;
- les formats : dumps `.f32`, `.batraw` (`BATRAW1`/`BATRAW2`), `*_metrics.jsonl`, les
  checkpoints (magic V1/V2 et trailer d'optimiseur — les octets sont identiques, seul le
  point d'entrée d'encodage a été scindé) ;
- l'emplacement de `Models/`, `datasets/`, `tools/`, `bench/`, `blind_tests/` ;
- le contenu des rapports archivés : ils citent les anciens chemins, et c'est voulu.

---

## 4. Vérifications

| Vérification | Résultat |
| --- | --- |
| `cargo test --workspace` (avant la mission) | 130 + 13 = 143 verts |
| `cargo test --workspace` (après déplacement des crates) | 131 + 13 = 144 (+1 : le test de `project_root`) |
| `cargo test --workspace` (après la scission moteur/UI) | 108 + 23 + 13 = **144 verts** |
| `cargo check --workspace --all-targets` | vert (exemples et tests inclus) |
| `cargo check -p batlab_core` | vert |
| deps interdites dans `batlab-core` | aucune, directe ou transitive |
| `./blind_tests/run.sh` (après déplacements) | **9/9 PASS** |
| `./blind_tests/run.sh` (après scission) | **9/9 PASS** |
| `tools/*.py`, `blind_tests/*.py`, `bench/optimizer/analyse.py` | compilent ; `receptive_field.py` et `gen_unet_config.py` exécutés |
| liens relatifs du README | tous résolus |

Les deux passages de la suite à l'aveugle ont tourné à l'horizon nominal
(`ACTIONS=3000`) ; cf. §2.4 sur ce qu'un horizon raccourci fait dire à P3.
