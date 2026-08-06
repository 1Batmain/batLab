# Gestionnaire de modèles et navigation modèle-centrée — rapport de mission

Branche : `model-manager` · Plateforme : macOS (Darwin 25.4.0) · TUI ratatui, binaire `target/release/batlab`

## Résumé

« On ouvre, on choisit un modèle, et on choisit de l'entraîner ou de faire une
inférence. » Le TUI s'ouvrait sur une bifurcation *Home → Load / Template* ; il
s'ouvre désormais sur la **liste des modèles**, et chaque modèle porte son **menu
d'actions** — Train / Infer / Perpetual, plus **Rename** et **Delete**, qui
n'existaient pas.

Deux corrections de fond accompagnent le flow :

- **La racine de stockage s'injecte** au lieu de se déduire. C'est ce qui a fait
  tomber la limite connue « `cargo test` réécrit `Models/Stable_Diffusion/config_file` »
  (`PERPETUAL_INFERENCE.md` §5), et c'est ce qui rend possible de dérouler le vrai
  TUI de bout en bout, sur une racine jetable, via `BATLAB_ROOT`.
- **`Esc` remonte d'un cran** partout. Six écrans quittaient l'application
  sèchement : un taux d'apprentissage mal tapé coûtait la session.

Sept commits :

| Commit | Objet |
| --- | --- |
| `b7ae401` | `Storage` à racine injectée, `rename_model` / `delete_model` et leurs garde-fous |
| `569110f` | La navigation modèle-centrée, les deux écrans du manager, `nav_tests.rs` |
| `044a0b2` | Relire les checkpoints en revenant au menu d'actions (trouvé au TUI réel) |
| `8044904` | Les barres d'aide disent ce que les touches font vraiment |
| `cc79bf4` | CLAUDE.md : le flow, `BATLAB_ROOT`, la limite levée |
| `7310e94` | Un checkpoint est un `.ckpt` (trouvé au TUI réel) |
| `5435440` | `[Esc] back` annoncé sur le formulaire d'inférence |

Les deux corrections marquées « trouvé au TUI réel » ne sont pas sorties d'une
relecture du code : les deux états fautifs étaient chacun cohérents localement, et
seul le parcours à la main les a exposés. Elles sont détaillées en §4.

## 1. Le flow

```
Ouverture → LISTE DES MODÈLES                      (racine n°1)
  chaque entrée : nom · géométrie · nb de couches · checkpoints
  dernière ligne : « New model (from template) » → sélecteur de templates
  → Entrée sur un modèle → MENU D'ACTIONS
       Train / Infer / Perpetual → choix des poids → formulaire du mode → MONITEUR (racine n°2)
       Rename / Delete           → le manager
```

`[e]` depuis le menu d'actions ouvre le constructeur de couches ; `[i]` dedans
ouvre la géométrie d'entrée. `Screen::InputSize` était dessiné, géré au clavier et
**jamais assigné** — le troisième écran fantôme de ce dépôt après `LoadPath` et le
mode `Perpetual`. `crates/batlab-ui/src/tui/nav_tests.rs` marche tous les écrans à
la touche depuis la porte d'entrée et exige d'avoir vu `Screen::ALL` : ajouter une
variante sans la câbler fait échouer la suite.

## 2. Le contrat observable du manager

Ce que voici est vérifiable **sans lire l'implémentation** — c'est la base sur
laquelle un agent de test aveugle peut travailler.

### 2.1 Les écrans et leurs touches

| Écran | Titre affiché | Touches | `Esc` mène à |
| --- | --- | --- | --- |
| Liste des modèles | `batlab — Models` | `↑↓` `Enter` ouvrir · `r` rafraîchir · `q` quitter | **quitte** (racine) |
| Menu d'actions | `<nom du modèle>` | `↑↓` `Enter` confirmer · `e` couches · `q` quitter | liste des modèles |
| Renommer | `Rename Model` | **toute touche imprimable = texte** · `Backspace` · `Enter` renommer | menu d'actions |
| Supprimer | `Delete Model` | **toute touche imprimable = texte** · `Backspace` · `Enter` supprimer | menu d'actions |
| Sélecteur de poids | `<nom> — Weights` | `↑↓` `Enter` continuer · `q` quitter | menu d'actions |
| Moniteur | — | `r` nouveau run · `s` sauver · `q` quitter | **quitte** (racine, termine le run) |

`Esc` ne quitte que depuis les **deux racines** : la liste (rien au-dessus) et le
moniteur (un run en cours, que quitter termine). Partout ailleurs il remonte
d'exactement un cran, et la chaîne des parents atteint toujours une racine.

Sur les deux écrans du manager, **`q` est du texte, pas une sortie** : un modèle
peut légitimement s'appeler `q-experiment`. C'est `Esc` qui annule.

### 2.2 Renommer — effets disque

- Déplace `Models/<ancien>/` → `Models/<nouveau>/`.
- **Et** réécrit le `config_file` : `model_name`, plus tout chemin de checkpoint
  qui pointait à l'intérieur du dossier du modèle.
- Si la réécriture du config échoue, **le renommage est défait** : jamais de paire
  dossier/config à moitié renommée.
- Le formulaire s'ouvre **prérempli sur le nom courant**.
- Au retour, la liste affiche le nouveau nom et un accusé `✓ Renamed 'x' → 'y'`.

### 2.3 Supprimer — les garde-fous

Trois verrous, dans cet ordre :

1. **Le nom doit être retapé à l'identique.** Tant que ce n'est pas le cas,
   `Enter` refuse et affiche `✗ Type the model name exactly — '<nom>' — to confirm.`
   Un `[y]` sur un menu est à un doigt égaré d'un run d'entraînement perdu.
   La barre d'aide bascule sur `[Enter] DELETE — no further prompt` **seulement**
   quand la frappe correspond : l'écran dit lui-même quand il est armé.
2. **Nom validable** — alphanumérique plus `-_.`, pas de `.` en tête, ≤ 64
   caractères. Une traversée de répertoire n'est même pas *épelable*.
3. **Chemin sous contrôle** — le chemin résolu doit être un **enfant direct** de
   `Models/`, les deux côtés canonicalisés. Un dossier de modèle qui est un **lien
   symbolique est refusé, pas suivi** : un `remove_dir_all` sur la cible effacerait
   les fichiers de quelqu'un d'autre.

La suppression efface ensuite le dossier **entier** — config, checkpoints,
métriques —, ce que l'écran annonce avant de demander la confirmation.

### 2.4 La garde de run, et sa limite

Rename et Delete **refusent tous deux un modèle dont ce processus tient un run**.

Cette garde est de l'état TUI (`running_model`, posé au démarrage d'un run, levé
quand le run se déclare fini). **Elle ne franchit pas la frontière de processus** :
un second `batlab`, ou un `--headless-train` dans un autre shell, reste invisible.
Renommer un modèle sous les pieds d'un de ceux-là enverra son prochain checkpoint
dans un dossier qui n'existe plus. **Non résolu**, documenté sur
`App::model_run_in_progress`.

### 2.5 Ce qui compte comme checkpoint

Un checkpoint est un fichier d'extension **`.ckpt`** dans
`Models/<nom>/pretrained_weights/`, non caché. Le `latest_metrics.jsonl` que tout
run d'entraînement dépose à côté n'en est **pas** un — ni dans le compteur
« N checkpoints » de la liste, ni dans les poids que le sélecteur propose de
charger. Voir §4.2.

## 3. La racine de stockage, et la limite levée

`storage` déduisait sa racine du workspace : un test n'avait **aucun moyen** de
dire « pas le vrai `Models/` ». Toute la disposition passe maintenant par un
`Storage` construit sur une racine explicite (`Storage::at`), que `App::with_storage`
fait descendre dans tout le TUI.

- Le défaut reste le workspace (trouvé en remontant jusqu'au `Cargo.toml` portant
  `[workspace]`, jamais un nombre fixe de `parent()`), mais lit d'abord
  **`BATLAB_ROOT`**.
- Tout test qui touche au stockage passe par `TempRoot`, supprimé au `Drop`.
- `list_models` ne crée plus rien : *regarder* une racine n'y laisse plus un
  `Models/`.

### Preuve que `cargo test` ne pollue plus

Critère de recette permanent, inscrit au CLAUDE.md — `cargo test --workspace`
suivi d'un `git status` **propre** :

```
$ cargo test --workspace
test result: ok. 13 passed; 0 failed
test result: ok. 108 passed; 0 failed; 4 ignored     (batlab_core)
test result: ok. 46 passed; 0 failed                 (batlab_ui)
$ git status --porcelain
                                                     (aucune sortie)
```

Avant cette mission, la même commande laissait `Models/Stable_Diffusion/config_file`
modifié dans le dépôt (`PERPETUAL_INFERENCE.md` §5). **La limite est levée.**

## 4. Les deux bugs que seul le TUI réel a montrés

### 4.1 Le menu d'actions oubliait les poids qu'on venait d'écrire — `044a0b2`

Sur le chemin `[r] new run` du moniteur, l'hôte reconstruit un `App` neuf et le
repose sur le menu d'actions. Rien ne remplissait la liste des poids : le menu
annonçait donc **« no checkpoints » pour un modèle dont le run qui venait de finir
avait justement écrit `latest.ckpt`**. On ne pouvait pas enchaîner une inférence
sur ce qu'on venait d'entraîner.

Chaque `App` pris isolément était cohérent — c'est **la couture entre deux `App`**
qui perdait l'information, et une relecture du code ne l'aurait pas montrée.
`App::enter_model_actions` fait maintenant les deux d'un bloc : relire le disque,
puis poser l'écran.

### 4.2 Un checkpoint, c'est un `.ckpt` — `7310e94`

Les deux listages — le compteur de la liste des modèles et le sélecteur de poids —
prenaient **tout fichier non caché** de `pretrained_weights/`. Or chaque run
d'entraînement y dépose un `latest_metrics.jsonl`. Donc, après le premier
entraînement de n'importe quel modèle :

```
32x32x1 · 12 layers · 2 checkpoints: latest.ckpt, latest_metrics.jsonl
```

— un checkpoint de trop dans le compte, et un JSONL proposé au chargement comme
s'il s'agissait de poids. Un seul prédicat, `is_checkpoint_file`, sert désormais
les deux listages ; les fichiers cachés restent exclus **par-dessus** l'extension,
parce que les AppleDouble de macOS (`._latest.ckpt`) passeraient le test
d'extension sans être des poids. Gardé par
`the_metrics_journal_beside_the_weights_is_not_a_checkpoint`.

## 5. Le parcours e2e déroulé

TUI réel (`target/release/batlab`) dans une fenêtre tmux dédiée
(`batLab-model-manager-e2e`), piloté par `tmux send-keys`, méthode
`INFER_VIZ.md` §5. **Racine jetable** sous le scratchpad via `BATLAB_ROOT` — le
`Models/` du dépôt n'a jamais été ouvert. stderr redirigé vers un fichier, code de
sortie capturé.

| # | Geste | Attendu | Résultat |
| --- | --- | --- | --- |
| 1 | Ouvrir sur une racine vide | liste, message « No models yet », **aucun `Models/` créé** | OK — seul `datasets/` apparaît (voir §6) |
| 2 | `Enter` → template Greyscale → `Enter` | modèle créé sur disque, menu d'actions | OK — `Models/Greyscale_Diffusion/{config_file,pretrained_weights}` |
| 3 | Menu d'actions | géométrie et couches lues du modèle | OK — `32x32x1 · 12 layers · no checkpoints` |
| 4 | `Esc` | retour à la liste, le modèle y figure | OK |
| 5 | Rename, frappe de `q-experiment-e2e` | le `q` initial est **du texte**, l'app ne quitte pas | OK — champ prérempli puis réécrit |
| 6 | `Enter` | dossier déplacé, `model_name` réécrit, accusé | OK — `Models/q-experiment-e2e/`, `model_name: q-experiment-e2e`, aucun résidu de l'ancien |
| 7 | Infer **sans poids** | erreur explicite, pas de crash | OK — `inference checkpoint not found: … (train the model first…)`, chemin **dans la racine jetable** |
| 8 | `Esc` depuis Dataset → 4 crans | Dataset → Params → Weights → Actions → Liste | OK — un cran par pression, app vivante à chaque étape |
| 9 | Train 40 pas sur `cifar10_grey.batraw` | run mené à terme, `latest.ckpt` écrit | OK — stderr toujours vide |
| 10 | `[r] new run` | le menu **voit** le checkpoint fraîchement écrit | OK — c'est le correctif §4.1 (et il a exposé §4.2) |
| 11 | Relance, binaire corrigé | `1 checkpoint: latest.ckpt`, JSONL toujours sur disque | OK — correctif §4.2 confirmé au TUI |
| 12 | Infer depuis `latest.ckpt` | run complet, aperçu affiché | OK — `Inference Preview (32x32x1)` |
| 13 | Delete, nom **incomplet** (`q-experiment-e2`) | refus, modèle intact | OK — `✗ Type the model name exactly…`, dossier toujours là |
| 14 | Compléter le nom | la barre d'aide s'arme | OK — bascule sur `[Enter] DELETE — no further prompt` |
| 15 | `Enter` | dossier entier effacé, retour à la liste | OK — `✓ Deleted 'q-experiment-e2e'`, `Models/` vide |
| 16 | `Esc` depuis le moniteur, **run en cours** | quitte (racine n°2), le run est terminé | OK |

**Confinement** : toutes les écritures — modèles, checkpoints, métriques,
`datasets/generated_samples/step_0039.png` — ont atterri dans la racine jetable.
Le `datasets/generated_samples/` du dépôt réel est resté intact (fichier le plus
récent inchangé), et `git status` du worktree est resté propre pendant tout le
parcours.

**Sorties** : deux sessions, **`EXIT=0`** les deux fois, **stderr 0 octet** les
deux fois.

## 6. Limites connues

- **La garde de run est mono-processus** (§2.4). C'est la limite la plus sérieuse
  du manager, et elle est intacte.
- **Ouvrir une racine y crée `datasets/`** alors que `Models/` n'est créé qu'au
  premier modèle. `list_models` a été rendu sans effet de bord, pas
  `refresh_datasets`. Bénin, mais asymétrique.
- **La liste des datasets ne se rafraîchit pas** en cours de session : un
  `.batraw` déposé pendant que le TUI tourne reste invisible jusqu'à ce qu'on
  ressorte au moins jusqu'à la liste des modèles et qu'on rouvre le modèle. Il n'y
  a pas de `[r]` sur cet écran (préexistant à cette mission).
- **Le journal de métriques écrit sur stdout** pendant que le TUI occupe l'écran,
  ce qui salit l'affichage jusqu'au prochain rendu complet. Cosmétique,
  préexistant, visible au pas 10 du parcours.
- **`Infer` sur un modèle sans poids** est un cul-de-sac assumé : le sélecteur
  propose « Start from random weights », et le run s'arrête sur un message qui dit
  quoi faire (`train the model first or place weights there`). Le message vient du
  binaire et est antérieur à cette branche.

## 7. Recette

```bash
cargo test --workspace     # 167 tests, 0 échec
git status --porcelain     # doit être vide
```

Aucun test n'a été affaibli. Deux ajoutés sur cette mission
(`re_entering_the_action_menu_re_reads_the_checkpoints`,
`the_metrics_journal_beside_the_weights_is_not_a_checkpoint`), tous deux issus
d'un bug observé au TUI réel avant d'être écrit en test.

Pour rejouer le parcours sans toucher au dépôt :

```bash
BATLAB_ROOT=/chemin/vers/racine/jetable ./target/release/batlab
```
