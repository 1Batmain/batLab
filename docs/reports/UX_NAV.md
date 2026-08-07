# UX_NAV — quatre retours après usage réel du TUI

Branche `ux-nav`. Un bug et trois améliorations de navigation, tous côté
`crates/batlab-ui`, tous déroulés end-to-end sur le vrai TUI (tmux `send-keys`,
`BATLAB_ROOT` jetable) et pas seulement en tests.

| # | Item | Commit |
|---|------|--------|
| 1 | Le lancement vole le focus du terminal (macOS) | `97fab7d` |
| 2 | Les poids pré-entraînés deviennent le défaut | `b777c7f` |
| 3 | Fil d'Ariane + navigation `←` / `→` | `4d12f88` |
| 4 | Info-bulle d'aide contextuelle | `535c40a` |
| — | (trouvé en chemin) résidus de l'écran précédent | `2ca69b1` |

---

## 1. Le bug de focus — cause racine confirmée

### Ce qui se passait

Au lancement de `batlab`, le focus quittait le terminal. Aucune fenêtre à
l'écran pour le justifier : l'utilisateur devait recliquer sur le terminal pour
pouvoir naviguer dans un TUI qui, lui, s'était affiché normalement.

### La cause, mesurée

`visualiser::run_on_main_thread` démarre l'event loop winit **au tout début du
processus** — avant le premier draw du TUI, et bien avant qu'une fenêtre
existe. Le backend AppKit de winit termine `applicationDidFinishLaunching` par
un appel inconditionnel :

```rust
// winit 0.30.12, src/platform_impl/macos/app_state.rs:137
app.activateIgnoringOtherApps(self.ivars().activate_ignoring_other_apps);
```

et le drapeau vaut **`true` par défaut** :

```rust
// winit 0.30.12, src/platform_impl/macos/event_loop.rs:212
Self { activation_policy: None, default_menu: true, activate_ignoring_other_apps: true }
```

L'hypothèse de départ — « `ActivationPolicy::Accessory` devrait suffire » —
était fausse, et c'est le point intéressant : la policy et l'activation sont
deux choses distinctes. `Accessory` retire l'icône du Dock et la barre de menus.
Elle n'empêche pas AppKit de faire de batlab l'**application frontale**. Le
diagnostic « créer une NSWindow vole le focus » était lui aussi à côté : la
fenêtre n'existe pas encore à cet instant. C'est l'`NSApplication` elle-même,
pas la fenêtre.

### La preuve

`NSWorkspace.frontmostApplication`, Ghostty au premier plan au moment du
lancement (condition nécessaire : lancé depuis un contexte où le terminal n'est
pas frontal, macOS refuse l'activation et le bug ne se reproduit pas — première
tentative de reproduction, négative, pour cette raison) :

```
                          avant lancement   après lancement
sans le correctif         Ghostty           batlab      ← le vol
avec le correctif         Ghostty           Ghostty     (stable à t+6 s)
```

### Le correctif

Une ligne, dans `build_event_loop()` :

```rust
builder.with_activate_ignoring_other_apps(false);
```

### Ce que `[v]` n'y perd pas

Contrainte dure de la mission, vérifiée sur les deux modes avec un vrai run GPU
(modèle `Greyscale_Diffusion` de template, dataset CIFAR-10 gris) :

| mode | fenêtre après `[v]` (System Events) | frontmost |
|------|--------------------------------------|-----------|
| entraînement | `Model Output  (32×32×1 channels)` | Ghostty |
| inférence | `Denoising — gauche: x_t (bruité) \| droite: x̂₀ (estimation) — 32×32×1` | Ghostty |

et le toggle reste un toggle : `[v]` → fenêtre listée, `[v]` → liste vide,
`[v]` → fenêtre listée. Aucune erreur dans le footer du moniteur.
L'event loop reste sur le main thread ; rien de cette partie n'a bougé.

### Le garde

Il n'existe **aucun observable d'exécution** côté processus : « batlab est-il
frontal » est un fait du window server, pas une valeur que ce processus détient.
Le test porte donc sur la source (`the_event_loop_never_activates_the_app_over_
the_terminal`, `#[cfg(target_os = "macos")]`, via `include_str!`). C'est grossier
et c'est assumé : il échoue si l'appel disparaît, et le faire disparaître
ramènerait un bug que l'utilisateur vit comme « batlab m'a mangé le clavier »
sans rien à l'écran pour l'expliquer.

---

## 2. Les poids pré-entraînés par défaut

Ouvrir un modèle qui a des poids et lancer un entraînement repartait de zéro à
moins d'aller chercher le checkpoint à la main dans le sélecteur — le geste
évident jetait silencieusement tout ce que le modèle avait appris.

### Contrat observable

- Ouvrir un modèle qui possède au moins un checkpoint **pointe le flux dessus** :
  `load_checkpoint_on_start = true`, `selected_checkpoint_path` renseigné, et le
  curseur du sélecteur de poids **sur cette ligne**.
- Le checkpoint préféré est **`latest.ckpt`**, pas le premier par ordre
  alphabétique — c'est le fichier que tout entraînement réécrit.
- Un chemin enregistré dans le `config_file` est honoré **s'il existe encore**
  sur disque ; sinon on retombe sur le préféré.
- Sans aucun checkpoint : repli sur random, et **dit à l'écran** des deux côtés —
  « This model has no weights yet: the run starts from random. » au sélecteur,
  « Yes — no checkpoint found » au formulaire, où la case est alors épinglée
  (rien vers quoi la décocher).
- Le formulaire d'entraînement porte un 4ᵉ champ, **« Start from random »**,
  décoché par défaut, basculé par `espace` / `←` / `→`. Il fait aller-retour :
  cocher puis décocher revient sur le même checkpoint.
- **Éditer l'architecture le coche d'office** et le défaut ne le décoche pas :
  ajouter, supprimer ou éditer une couche, ou changer la géométrie d'entrée,
  invalide les poids.
- `--headless-train` est inchangé (DEV/CI, force `load_checkpoint=false`).

### La décision de conception qui tient tout

Le curseur du sélecteur est **dérivé, jamais mémorisé** :
`load_checkpoint_on_start` + `selected_checkpoint_path` sont la seule source de
vérité, et `refresh_weight_selector()` recalcule la ligne à partir d'eux. C'est
ce qui empêche une ligne périmée de re-sélectionner un checkpoint après une
édition d'architecture. Côté moteur un tel checkpoint serait de toute façon
refusé (`CheckpointLayerMismatch`, `model.rs`) — mais après avoir construit le
modèle sur GPU, c'est-à-dire tard et cher.

### Deux effets de bord traités

- `training_params.fields[3]` est *aussi* le chemin du dataset. Avec la case au
  même index, un `Backspace` non gardé l'aurait mangé caractère par caractère,
  depuis un écran qui n'en montre rien. Gardé + test dédié.
- La colonne des libellés des formulaires est désormais **mesurée** au lieu
  d'être figée à 16 : elle répare au passage l'alignement du formulaire
  perpetual, dont « Renoise Depth (t_r) » (19 caractères) débordait déjà.

---

## 3. Le fil d'Ariane et `←` / `→`

### Le chemin

`PathStep` (5 étapes) est délibérément distinct de `Screen` (14) :

```
Model › Action › Weights › Parameters › Run
```

Les quatre formulaires de paramètres sont **une seule** étape : du point de vue
de l'utilisateur il y en a une, quel que soit le mode de run.
`Screen::path_step()` est un match exhaustif — on ne peut plus ajouter un écran
sans trancher s'il est une étape du chemin ou un détour. Les détours (renommer,
supprimer, l'éditeur d'architecture, la géométrie d'entrée, la popup de contrôle
d'entraînement) n'affichent **rien**, plutôt que de revendiquer une position que
`←` n'honorerait pas.

### Contrat observable de la navigation

**Affichage.** Dernière ligne du terminal, sur les écrans du chemin uniquement.
Étape courante entre crochets et en jaune ; étapes franchies en gris lisible —
ce sont les décisions sur lesquelles l'écran courant s'appuie ; étapes à venir
estompées. Le rappel `← back  → forward` n'apparaît que là où les touches font
quelque chose. Sous 72 colonnes, réduction à `Model (1/5)` ; sous 30,
disparition. La ligne est **réservée** par un split, pas dessinée par-dessus :
le moniteur utilise toute sa surface jusqu'à la bordure basse.

**Où `←` / `→` sont captés.** Uniquement sur les **sélecteurs purs** —
`Screen::walks_the_path_by_arrow()` : liste des modèles, sélecteur de templates,
menu d'actions, sélecteur de poids. Nulle part ailleurs : partout ailleurs un
champ ou un curseur les utilise déjà.

**`←`.** Exactement `Esc`, sans le quit. Les deux passent par le même
`App::path_back()` — deux touches « retour » implémentées deux fois divergent, et
la divergence reste invisible jusqu'à ce que quelqu'un utilise celle qui n'a pas
été maintenue. Seule différence, voulue : à la racine (liste des modèles) `Esc`
quitte, `←` ne fait **rien**. Une flèche ne termine pas une session.

**`→`.** Avance d'une étape, et **seulement là où avancer est de la
navigation** :

| depuis | `→` fait |
|--------|----------|
| liste des modèles | ouvre le modèle sélectionné (ou le flux template sur la dernière ligne) |
| sélecteur de templates | **rien** — avancer y écrirait un `config_file` sur disque |
| menu d'actions | va aux poids, pour les trois modes de run seulement |
| menu d'actions, `Rename` / `Delete` | **rien** — ce ne sont pas des étapes du chemin |
| sélecteur de poids | va au formulaire du mode choisi |

**La promesse de `→`.** Remonter puis redescendre ne coûte aucun choix. Cas
limite traité explicitement : si le modèle sélectionné dans la liste est déjà
celui en main, `→` le **reprend sans relire son `config_file`** — une
réouverture réinitialiserait l'action et les poids d'après le fichier, ce qui
viderait `→` de son sens.

### Ce qui ne doit pas régresser

Quatre bindings flèches préexistants, chacun vérifié par son **effet** et pas par
l'écran atteint (`the_existing_arrow_bindings_still_do_what_they_did`) :

1. type de couche dans le constructeur (`←` / `→` cyclent) ;
2. cycle de dataset dans le sélecteur ;
3. bascules seed et regime des deux formulaires d'inférence ;
4. tempo d'un run perpetual depuis le moniteur (`NudgeTempo`).

---

## 4. L'info-bulle d'aide contextuelle

Chaque formulaire demande des nombres qui engagent des heures de GPU, et chacun
n'offrait qu'une seule affordance : une étiquette. « Magnitude » n'est pas une
explication.

### Contrat observable

- Le panneau s'affiche **à droite** et suit le champ sélectionné, sur les cinq
  formulaires (géométrie d'entrée, entraînement, inférence, perpetual, contrôles
  en cours de run) et sur le sélecteur de poids.
- Il cite le rapport dont son contenu vient (`→ docs/reports/…`).
- Sous **106 colonnes** il disparaît et le formulaire reprend toute la largeur.
- Un champ sans entrée fait simplement taire le panneau : c'est un indice, jamais
  une barrière.

### Le texte est sourcé, pas inventé

C'est le point de l'item. Trois exemples de ce que ça change :

- **lr** — « 1e-3 avec Adam atteint dès le pas 75 un niveau que SGD n'atteint
  jamais en 1500 pas ; plateau utile 1e-3 à 3e-3 » (`OPTIMIZER_ADAM.md`).
- **denoising_paths** — dit l'inconfortable : les N chemins partent du **même**
  `x_T`, donc la moyenne floute au lieu d'enrichir, et `paths=3` masquait une
  partie du banding au lieu de le corriger (`AUDIT_TRAINING.md`,
  `SCALE_UNET.md`).
- **magnitude** — le repère n'est pas `inter_seed_std` (proportionnel à
  magnitude, il mesure surtout le bruit non débruité) mais `intra_image_std`,
  qui doit s'approcher de 0,206 **par le bas** (`LOSS_WEIGHTING.md`).

### Deux gardes, parce qu'une aide fausse est pire qu'une aide absente

- `every_help_entry_sits_on_the_field_it_describes` compare les titres des
  entrées aux `*_FIELD_NAMES`. Les tables sont indexées par position : un champ
  inséré au milieu d'un formulaire re-pointerait toutes les entrées suivantes sur
  la mauvaise étiquette, et le panneau expliquerait alors avec aplomb le mauvais
  paramètre.
- `a_cited_source_is_a_report_that_exists` vérifie que chaque rapport cité est
  bien un fichier de `docs/reports/`.

---

## 5. Trouvé en chemin — les résidus d'écran

En déroulant le chemin de relance (`[r]` depuis un moniteur terminé), la moitié
du panneau Architecture et la table Analytics restaient **derrière** le sélecteur
de poids, entrelacées avec lui. Cause : la plupart des écrans sont une popup
posée sur rien, et les cellules qu'ils ne touchent pas gardent ce que l'écran
précédent y avait laissé. Invisible tant qu'un écran se redessine lui-même —
les frames consécutives sont identiques —, criant dès que le flux passe d'un
écran plein à une popup. Défaut préexistant, pas une régression de cette
mission.

Correctif : un `Clear` sur toute la frame avant de dessiner. Ratatui diffe
toujours avant d'écrire dans le terminal, donc c'est gratuit sur un écran
immobile.

---

## 6. Les parcours e2e déroulés

Tous sur `BATLAB_ROOT` jetable, TUI piloté par `tmux send-keys`, captures par
`tmux capture-pane`.

**Focus (item 1).** Ghostty activé, lancement, `NSWorkspace.frontmostApplication`
relevé avant et après — sans puis avec le correctif. Puis vrai run
d'entraînement (40 pas, puis 4000 pas interrompus) et vraie inférence
(20 chemins de débruitage), `[v]` pressé pendant chacun, fenêtre relevée par
System Events, focus relevé à chaque étape.

**Défaut pretrained (item 2).** Modèle créé par template → deux checkpoints
déposés sur disque pendant que le TUI tourne (`latest.ckpt` et
`epoch-0010.ckpt`) → réouverture → curseur sur `latest.ckpt`, ni sur
`epoch-0010.ckpt` (premier alphabétique) ni sur random → formulaire affichant
« No — continue from latest.ckpt » → `espace` → « Yes » → `espace` → retour au
checkpoint. Puis checkpoints retirés → repli affiché et case épinglée.

**Fil d'Ariane (item 3).** `→ → →` depuis la liste jusqu'au formulaire
d'entraînement (fil d'Ariane suivant à chaque étape, rappel `← back → forward`
disparaissant sur le formulaire) ; `Esc` puis `← ← ←` jusqu'à la racine, le
troisième `←` inerte et l'application toujours vivante ; puis `→ → →` qui rouvre
**Training** Parameters — l'action choisie a survécu, là où une réouverture naïve
l'aurait remise sur `Infer` d'après le `config_file`. Dégradation vérifiée à 80,
50 et 28 colonnes.

**Aide (item 4).** Sélecteur de poids, puis `lr` → `batch` (le panneau suit et
change de source citée), puis le formulaire perpetual jusqu'à `Regime`.
Disparition vérifiée à 100 et 90 colonnes.

---

## 7. Recette

```bash
cargo test --workspace   # 201 tests, dont 65 pour batlab_ui (47 avant la mission)
git status               # propre
```

Aucun test affaibli. `nav_tests.rs` a gagné les six tests du chemin ; `app.rs`
les sept du défaut pretrained ; `help.rs` les trois de l'aide ;
`visualiser/mod.rs` le garde du focus.

## 8. Ce qui n'est pas fait

- Le correctif de focus est **spécifique à macOS**. Windows et Linux n'ont pas
  le chemin fautif (`activateIgnoringOtherApps` est une API AppKit), mais rien
  n'a été vérifié dessus.
- `←` / `→` ne sont pas offerts sur les formulaires de paramètres, par
  construction : les champs y ont déjà les flèches. `Esc` reste la sortie d'un
  formulaire, ce qui rend le chemin asymétrique — descendable à la flèche,
  remontable à la flèche seulement jusqu'aux poids.
- L'aide ne couvre pas le constructeur de couches (nombre de noyaux, stride,
  padding, clés de skip). C'est le prochain endroit évident où le panneau
  gagnerait sa place.
