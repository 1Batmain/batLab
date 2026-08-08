# Dupliquer un modèle — la fondation survit au fine-tune

Branche `duplicate-model`. Une action de plus au menu du modèle, et une
propriété : **après duplication, l'original est exactement ce qu'il était.**

**Résumé** — spécialiser `Color_Diffusion_XL` sur un dataset dédié voulait dire
l'entraîner *sur place* : le run écrivait dans le même `pretrained_weights/`,
et après assez de pas sur un dataset étroit la fondation générale avait disparu.
On duplique désormais d'abord — `Color_Diffusion_XL` → `Elephants_XL`, poids
compris — et on fine-tune la copie. Ce qui voyage est le **dernier run seul**,
sous son nom daté, avec `latest.ckpt` relié dessus **dans la copie**.

---

## 1. Le flux

```
MENU D'ACTIONS  Train / Infer / Perpetual / Rename / Duplicate / Delete
  → Duplicate → FORMULAIRE
        New name        : Color_Diffusion_XL-copy   (prérempli, libre)
        What travels    : > config + weights   (défaut)
                            config only
  → Enter → LISTE DES MODÈLES, curseur SUR LA COPIE
        ✓ Duplicated 'Color_Diffusion_XL' → 'Elephants_XL'
          — config + run-2026-08-08_1323.ckpt, latest.ckpt
```

`Duplicate` s'insère entre `Rename` et `Delete` : **`Delete` reste le dernier**,
y arriver au curseur doit rester une marche délibérée jusqu'au bout.

Le retour se fait sur la **liste**, pas sur le menu de l'original, et le curseur
tombe sur **la copie** — c'est le modèle sur lequel on veut travailler, sinon on
n'aurait pas dupliqué. `active_model_name` reste l'original : rien de lui n'a
bougé, donc rien de la prise que la session a sur lui n'a besoin de bouger.

## 2. Contrat observable

Vérifiable sans lire l'implémentation.

### 2.1 L'écran et ses touches

| Écran | Titre affiché | Touches | `Esc` mène à |
| --- | --- | --- | --- |
| Dupliquer | `Duplicate Model` | **toute touche imprimable = texte** · `Backspace` · `↑↓` ce qui est copié · `Enter` dupliquer | menu d'actions |

Comme sur les deux autres écrans du manager, **`q` est du texte** (une copie de
`q-experiment` doit être nommable), et `e` ne rouvre pas le constructeur de
couches. `↑`/`↓` sont les seules touches dont le champ de nom ne veut pas :
c'est pourquoi le choix de contenu leur est confié. `Esc` est la sortie.

### 2.2 Le nom

- Le formulaire **ouvre sur un nom libre** : `<nom>-copy`, puis `-copy-2`,
  `-copy-3`… Le cas courant — dupliquer, `Enter` — ne se tape pas.
- Le nom est validé **comme celui du renommage** : alphanumérique plus `-_.`,
  pas de `.` en tête, ≤ 64 caractères. `../evil` n'est pas *épelable*.
- Un nom déjà pris est refusé (`a model named 'X' already exists`), le nom du
  modèle lui-même aussi. **Un refus n'écrit rien** — ni dossier, ni config.
- Le nom suggéré reste dans la limite de longueur en rognant la **base**, pas le
  suffixe : d'un nom trop long, ce qui vaut d'être gardé est qu'il dit « copie ».

### 2.3 Ce que la copie est

- `Models/<copie>/config_file` existe, `model_name` **est celui de la copie**, et
  **tout chemin de checkpoint qui pointait dans le dossier source est rebasé sur
  la copie**. C'est la même `retarget_config` que le renommage — un chemin resté
  sur l'original enverrait le premier run de la copie précisément dans le dossier
  que l'opération existe pour protéger.
- `Models/<copie>/pretrained_weights/` existe dans les deux cas. Une copie est un
  modèle, un modèle a ce dossier.
- La liste des modèles montre la copie immédiatement, et **le curseur y est**.

### 2.4 Ce que la copie porte

**config + poids** (défaut) :

- **Le dernier run seul**, sous son propre nom daté, plus `latest.ckpt` reposé
  dessus **dans la copie** — un lien dur, même inode, un seul jeu d'octets
  (`DATED_CHECKPOINTS.md` §1.2).
- **Pas l'historique.** Une nuit sous `--checkpoint-every` laisse des dizaines de
  fichiers de 14 Mo pour le XL ; les copier tous coûterait un demi-gigaoctet pour
  porter des runs auxquels la copie n'a aucune prétention.
- Le fichier daté est retrouvé **par identité d'inode** avec `latest.ckpt`, pas
  par le nom : les deux sont un seul jeu d'octets, et c'est ce qui permet à la
  copie de garder **la date à laquelle les poids ont été faits**. Copier
  `latest.ckpt` seul la perdrait.
- Un `latest.ckpt` d'avant la convention (sans jumeau daté) voyage sous le seul
  nom qu'il a.
- Un modèle **sans poids** duplique en config seule même si les poids ont été
  demandés — et la ligne de statut le **dit** au lieu de le laisser découvrir
  deux écrans plus loin.
- Ce qui n'est **pas** un checkpoint ne voyage pas : le `*_metrics.jsonl` du run
  reste chez l'original (`is_checkpoint_file`, `MODEL_MANAGER.md` §4.2).
- La copie est une **vraie copie**, pas un lien dur vers l'original : son inode
  diffère. Supprimer l'un ne doit pas être une surprise pour l'autre, et le coût
  disque doit être visible.

**config seule** : `pretrained_weights/` vide, la ligne de statut dit
`config only, no weights copied`.

### 2.5 Ce qui ne bouge pas

**L'original est intact** : mêmes octets, mêmes noms, mêmes inodes, mêmes mtimes,
son `config_file` non réécrit. Rien de `duplicate_model` n'ouvre le dossier
source en écriture. C'est la propriété qui justifie la fonctionnalité, et elle
est tenue des deux côtés — test unitaire et parcours e2e.

### 2.6 Les refus

- **Un run en cours dans ce processus** bloque `Duplicate` comme il bloque
  `Rename` et `Delete`. Même garde, **même limite** : elle ne franchit pas la
  frontière de processus (`MODEL_MANAGER.md` §2.4, `App::model_run_in_progress`).
- Un échec en cours de route ne laisse **pas de modèle à moitié fait** : le
  dossier destination est effacé et l'erreur remontée.
- Un dossier source sans `config_file` n'est pas un modèle : `NotFound`.

Tenu par : `storage.rs` (`duplicating_a_model_leaves_the_original_untouched`,
`a_duplicate_is_a_model_of_its_own_and_its_config_says_so`,
`the_newest_run_travels_with_the_copy_and_latest_points_at_it_there`,
`a_duplicate_can_be_taken_without_the_weights`,
`asking_for_weights_a_model_does_not_have_copies_the_config_alone`,
`a_lone_latest_travels_under_its_own_name`,
`duplicating_onto_a_name_that_is_taken_changes_nothing`,
`a_pathological_duplicate_name_is_refused_and_writes_nothing`,
`a_suggested_copy_name_is_free_and_short_enough`),
`tui/app.rs` (`duplicating_lands_on_the_copy_and_leaves_the_original_alone`,
`a_duplicate_can_be_asked_for_without_the_weights`,
`duplicating_under_an_unusable_name_changes_nothing`,
`the_manager_refuses_a_model_with_a_run_in_progress`),
`tui/events.rs` (`the_manager_forms_treat_every_printable_key_as_text`,
`the_action_cursor_reaches_every_action`),
`tui/nav_tests.rs` (`every_screen_is_reachable_by_pressing_keys` —
`Screen::ALL` passe à 15 —, `esc_walks_up_one_step_from_every_screen_and_is_never_inert`,
`right_never_creates_anything`).

## 3. Le parcours e2e, déroulé

TUI réel (`target/release/batlab`) dans une fenêtre tmux dédiée
(`batLab-duplicate-e2e`), piloté par `tmux send-keys`. **Racine jetable** via
`BATLAB_ROOT` sous le scratchpad, dataset en lien symbolique — le `Models/` du
dépôt n'a jamais été ouvert. stderr redirigé, code de sortie capturé.

| # | Geste | Attendu | Résultat |
| --- | --- | --- | --- |
| 1 | Ouvrir sur `Foundation_XL` | le menu porte **Duplicate** entre Rename et Delete | OK |
| 2 | Train, 20 pas | run mené à terme, daté + `latest.ckpt` lié | OK — `run-2026-08-08_1323.ckpt`, inode 15311828, link count 2 |
| 3 | `[r]` → `Duplicate` | formulaire prérempli `Foundation_XL-copy`, **poids cochés** | OK |
| 4 | Frappe d'un `q` | c'est du **texte**, l'app ne quitte pas | OK |
| 5 | `↓` puis `↑` | le choix bascule, la ligne « Only the newest run travels » suit | OK |
| 6 | Nom `Elephants_XL`, `Enter` | liste, **curseur sur la copie**, accusé nommant les fichiers | OK — `✓ Duplicated 'Foundation_XL' → 'Elephants_XL' — config + run-2026-08-08_1323.ckpt, latest.ckpt` |
| 7 | Disque, la copie | config à son nom, chemins rebasés, poids liés entre eux | OK — `model_name: Elephants_XL`, les deux chemins sous `Elephants_XL/`, `latest.ckpt` et le daté sur l'inode **15312741** (link count 2) |
| 8 | Disque, l'original | octet pour octet, inode et mtime compris | OK — sha, inode 15311828, mtime, taille : **identiques à avant** ; le `_metrics.jsonl` n'a pas voyagé |
| 9 | Inférence sur la copie | les poids copiés se chargent et génèrent | OK — run mené à terme sur `Elephants_XL/pretrained_weights/…`, aperçu affiché |
| 10 | Disque, l'original **après** l'inférence | toujours identique | OK — `diff` vide contre l'empreinte du pas 8 |
| 11 | Duplicate → nom `Foundation_XL` | refus, rien créé | OK — `✗ a model named 'Foundation_XL' already exists` |
| 12 | Duplicate → nom `../evil` | refus, rien créé nulle part | OK — `✗ invalid model name: the name must not start with '.'`, aucun `evil` sur le disque |
| 13 | Duplicate `Elephants_XL` → `Elephants_fresh`, **config only** | copie sans poids, dit à l'écran | OK — `no checkpoints` dans la liste, `✓ … — config only, no weights copied` |
| 14 | `q` | sortie propre | OK |

**Sorties** : `EXIT=0`, **stderr 0 octet**. `git status` du worktree resté propre
pendant tout le parcours.

## 4. Recette

```bash
cargo test --workspace     # 291 tests, 0 échec
git status --porcelain     # vide
```

Aucun test n'a été affaibli. Trois tests existants ont été **étendus**, pas
modifiés dans leur intention : la garde « run en cours » et l'inertie de `→`
couvrent maintenant les trois actions du manager, et le test « toute touche
imprimable est du texte » couvre le troisième formulaire.

## 5. Ce qui n'est pas fait

- **Pas de choix « tout l'historique ».** Le formulaire a deux lignes, pas trois.
  Copier des dizaines de checkpoints est faisable à la main (`cp`) et le cas ne
  s'est pas présenté ; l'ajouter demanderait de dire à l'écran ce que ça coûte.
- **La mtime copiée est une faveur du système**, pas un contrat. Sur macOS
  `fs::copy` passe par `fcopyfile` et préserve la date, donc le sélecteur de
  poids de la copie affiche la vraie date des poids. Ailleurs elle pourrait
  paraître neuve — **le nom, lui, garde la vraie date**, ce qui est exactement la
  raison d'être du nom (`DATED_CHECKPOINTS.md` §4).
- **Une copie « config seule » garde les chemins de checkpoint de son config**,
  rebasés sur son propre dossier — donc pointant sur un fichier qui n'existe pas
  encore. Ce n'est pas un défaut : `preselect_pretrained_weights` n'honore un
  chemin enregistré que si le fichier est sur disque, et la copie s'ouvre donc
  sur « random », dit à l'écran. Le point qui compte est tenu : **aucun de ces
  chemins ne pointe plus vers l'original**.
- **La garde de run reste mono-processus.** Un second batlab, ou un
  `--headless-train` dans un autre shell, peut écrire dans la source pendant que
  la copie se fait. La copie serait alors un instantané pris pendant une
  écriture. Même limite que `Rename`/`Delete`, non résolue ici.
- **Le nom suggéré ne consulte pas la casse du système de fichiers.** Sur un
  volume insensible à la casse, `x-copy` et `X-Copy` sont le même dossier ; la
  suggestion s'appuie sur `exists()`, qui l'est aussi, donc elle ne propose pas
  un nom qui serait ensuite refusé. Le refus, lui, est explicite.
