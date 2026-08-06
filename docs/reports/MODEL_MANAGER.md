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

Huit commits :

| Commit | Objet |
| --- | --- |
| `b7ae401` | `Storage` à racine injectée, `rename_model` / `delete_model` et leurs garde-fous |
| `569110f` | La navigation modèle-centrée, les deux écrans du manager, `nav_tests.rs` |
| `044a0b2` | Relire les checkpoints en revenant au menu d'actions (trouvé au TUI réel) |
| `8044904` | Les barres d'aide disent ce que les touches font vraiment |
| `cc79bf4` | CLAUDE.md : le flow, `BATLAB_ROOT`, la limite levée |
| `7310e94` | Un checkpoint est un `.ckpt` (trouvé au TUI réel) |
| `5435440` | `[Esc] back` annoncé sur le formulaire d'inférence |
| `6b91485` | **Les templates créaient des modèles non conditionnables sur t** (trouvé par l'agent aveugle) |

Les deux corrections marquées « trouvé au TUI réel » ne sont pas sorties d'une
relecture du code : les deux états fautifs étaient chacun cohérents localement, et
seul le parcours à la main les a exposés. Elles sont détaillées en §4.

Le défaut bloquant de `6b91485` n'a été trouvé ni par le code ni par mon parcours
e2e — **c'est l'agent de test aveugle qui l'a vu**, en lisant le `config_file` que
le flux de création écrit sur disque. §4.3 dit pourquoi je l'ai manqué.

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

### 4.3 Les templates créaient des modèles mort-nés — `6b91485`

**Trouvé par l'agent de test aveugle**, pas par moi (`9081b22` sur
`blind-test-manager`, `blind_tests_manager/t09_template_geometry.sh`).

Le flux « New model (from template) » écrivait `input_size.z == canaux de sortie` :

| Template | Avant | Après |
| --- | --- | --- |
| Greyscale Diffusion | `[32,32,1]` → 1 ❌ | `[32,32,3]` → 1 (1 signal + 2 temporels) |
| Stable Diffusion | `[32,32,3]` → 3 ❌ | `[32,32,7]` → 3 (3 signal + 4 temporels) |

CLAUDE.md exige l'inverse : « un modèle de diffusion DOIT être conditionné sur le
timestep : `input_size.z > output.z` ». Sans canal excédentaire pour porter
l'embedding temporel, ε̂ dégénère et l'échantillonnage explose en blanc saturé.
La géométrie du template Greyscale était **exactement celle que le dépôt conserve
sous `Models/Greyscale_Diffusion_broken`** — le cas dégénéré historique. Tout
modèle créé depuis le TUI naissait donc mort-né.

Les nouvelles géométries s'alignent sur les configs saines du dépôt :
`Models/Greyscale_Diffusion` (`[32,32,3]` → 1) et `Models/Color_Diffusion_L`
(`[32,32,7]` → 3).

**Ce qui a changé au-delà des chiffres.** Les deux templates n'étaient que le même
U-Net à deux budgets de canaux près, dupliqué douze couches durant avec toutes les
dims écrites à la main — la duplication *était* le terrain du bug. Ils partagent
maintenant `diffusion_unet(signal_channels, time_channels)`, bâti par un
`UNetDraft` qui **dérive** chaque dimension de la couche précédente. C'est la
discipline de `tools/gen_unet_config.py`, portée en Rust pour la raison qui l'a
fait écrire là-bas : `dim_kernel.z` doit suivre le nombre de canaux courant, et si
`Convolution` rejette une profondeur fausse au build, **`UpsampleConv` ne la rejette
pas** — son shader indexe les poids avec `IC = dim_input.z` et corrompt en silence.

**Pourquoi mon parcours e2e ne l'a pas vu.** Au pas 9 j'ai entraîné un modèle issu
du template et j'ai regardé la loss descendre — elle descendait. Une loss qui décroît
ne dit rien du conditionnement temporel : c'est précisément l'avertissement de
CLAUDE.md (« une loss batch qui décroît ne suffit PAS — vérifier la loss par tranche
de t »). Je vérifiais que le manager *manipulait* correctement les modèles, sans
jamais demander si le modèle produit était *valide*. L'aveugle, lui, partait de la
spec — l'invariant y est écrit noir sur blanc — et il est allé lire le fichier.
C'est exactement le désaccord que le protocole cherche à produire.

**Vérification** : au-delà de l'invariant statique, un Greyscale créé au TUI réel
s'entraîne (loss 1,12 → 0,73 sur 30 pas, `EXIT=0`, stderr vide) et le Stable
construit et tourne aussi — aucun `KernelChannelMismatch`, le moteur accepte la
géométrie. Les tests ont été vérifiés **échouants** sur l'ancienne géométrie
(`time_channels = 0` → « emits 1 channels for 1 in »).

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
| 3 | Menu d'actions | géométrie et couches lues du modèle | OK — `32x32x1 · 12 layers · no checkpoints` **(géométrie dégénérée, non vue ici — voir §4.3 ; le template corrigé affiche `32x32x3`)** |
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

- **`Models/Stable_Diffusion` du dépôt porte lui aussi la géométrie dégénérée** —
  `[32,32,3]` → 3, comme le template qui l'a produit. Corriger le template ne l'a
  pas corrigé, lui : c'est un fichier suivi, et le réécrire n'était pas dans cette
  mission. À réentraîner ou régénérer (`tools/gen_unet_config.py`). Les autres
  modèles du dépôt sont sains (`Greyscale_Diffusion` 3→1,
  `Greyscale_Diffusion_L` 5→1, `Color_Diffusion_L` 7→3).
- **« New model (from template) » écrase silencieusement un modèle existant du même
  nom.** Le modèle créé porte la clé du template, et `apply_template` écrit son
  `config_file` sans vérifier que le dossier existe déjà — ni prompt, ni accusé.
  **Vérifié** : un `config_file` marqué à la main a été remplacé sans un mot.
  Dans le dépôt réel, où `Models/Stable_Diffusion/` existe, une frappe sur
  « New model » suffit donc à perdre sa configuration. Non corrigé — hors du
  périmètre de cette mission, mais c'est le prochain défaut à traiter.
- **La garde de run est mono-processus** (§2.4). C'est la limite la plus sérieuse
  du manager côté concurrence, et elle est intacte.
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
cargo test --workspace     # 171 tests, 0 échec
git status --porcelain     # doit être vide
```

Aucun test n'a été affaibli. Six ajoutés sur cette mission, chacun né d'un défaut
observé avant d'être écrit :

| Test | Origine |
| --- | --- |
| `re_entering_the_action_menu_re_reads_the_checkpoints` | TUI réel |
| `the_metrics_journal_beside_the_weights_is_not_a_checkpoint` | TUI réel |
| `every_template_is_conditionable_on_the_timestep` | agent aveugle |
| `every_template_kernel_is_as_deep_as_its_input` | agent aveugle (généralisation) |
| `every_template_stack_starts_on_its_declared_input` | agent aveugle (généralisation) |
| `every_model_created_from_a_template_is_conditionable_on_the_timestep` | agent aveugle |

Une assertion existante a été **mise à jour, pas affaiblie** :
`selecting_template_prefills_architecture_and_advances_to_the_action_menu`
attendait `(32, 32, 3)` en entrée du template RGB — c'est désormais `(32, 32, 7)`,
la géométrie ayant changé volontairement.

Pour rejouer le parcours sans toucher au dépôt :

```bash
BATLAB_ROOT=/chemin/vers/racine/jetable ./target/release/batlab
```
