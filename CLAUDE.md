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
       Train / Infer / Perpetual      → choix des poids → formulaire du mode → Monitor
       Rename / Duplicate / Delete    → le manager
```

`Esc` remonte d'exactement un cran depuis chaque écran, et ne quitte que depuis
les deux racines : la liste (rien au-dessus) et le moniteur (un run en cours, que
quitter termine). `[e]` depuis le menu d'actions ouvre le constructeur de
couches, `[i]` dedans ouvre la géométrie d'entrée.

Ce chemin est **affiché** : `Model › Action › Weights › Parameters › Run`, en bas
du terminal, étape courante mise en évidence (`PathStep`, distinct de `Screen` —
les quatre formulaires de paramètres sont une seule étape). `←`/`→` le remontent
et le redescendent, mais **seulement sur les sélecteurs purs**
(`Screen::walks_the_path_by_arrow`) : partout ailleurs les flèches appartiennent
déjà à un champ ou à un curseur (type de couche, tempo perpetual, cycle de
dataset, bascules seed/regime). `Esc` et `←` passent par le même
`App::path_back()` — deux « retour » implémentés deux fois divergent. Une seule
différence, voulue : à la racine `Esc` quitte, `←` ne fait rien. Et `→` n'avance
que là où avancer est de la navigation : depuis le sélecteur de templates il est
**inerte**, parce qu'avancer y écrirait un `config_file` sur disque.

**Le défaut est de continuer depuis les poids du modèle**, pas de repartir de
zéro : ouvrir un modèle qui a un checkpoint pointe le flux dessus (celui que son
`config_file` a enregistré au dernier run, `latest.ckpt` sinon), et le sélecteur
est **trié du plus récent au plus ancien**, chaque ligne portant sa date et sa
taille (`run-2026-08-08_1120.ckpt   2026-08-08 11:20 · 78 kB`) — choisir « celui
d'hier soir » ne doit pas être une devinette. « Start from random » est une case du
formulaire d'entraînement, décochée. Le curseur du sélecteur de poids est
**dérivé** de `load_checkpoint_on_start` + `selected_checkpoint_path`, jamais
mémorisé : c'est ce qui empêche une ligne périmée de recharger un checkpoint
après une édition d'architecture (tous les chemins qui touchent aux couches
posent le drapeau à faux). Sans checkpoint, repli sur random, dit à l'écran.

Sur la liste des modèles, ce même panneau de droite décrit le modèle sous le
curseur : géométrie, couches, paramètres, **champ réceptif avec son verdict**
(« 13/32 px — NE COUVRE PAS » : les deux templates ne couvrent pas l'image, donc
un pixel de sortie ne peut pas choisir un contenu global) et la pile entière,
élidée en son milieu si elle ne tient pas — jamais tronquée en silence. Le champ
réceptif suit la récurrence de `tools/receptive_field.py`, qui sert d'oracle
tiers ; le compte de paramètres est croisé avec les scalaires d'un vrai
checkpoint (`the_parameter_count_matches_the_scalars_a_checkpoint_holds`).

Chaque champ de formulaire a une **info-bulle sourcée** (`tui/help.rs`) : le
texte cite le rapport dont il vient et ses chiffres en sortent. Une entrée qui
ment est pire que pas d'entrée — `every_help_entry_sits_on_the_field_it_describes`
compare les titres aux `*_FIELD_NAMES` (les tables sont indexées par position) et
`a_cited_source_is_a_report_that_exists` vérifie les citations.

Le manager, côté contrat observable :

- **Renommer** déplace `Models/<ancien>/` → `Models/<nouveau>/` **et** réécrit le
  `config_file` — `model_name` plus tout chemin de checkpoint qui pointait dans
  le dossier du modèle. Si la réécriture échoue, le renommage est défait.
- **Dupliquer** écrit `Models/<copie>/` — un modèle à part entière, dont le
  `config_file` porte son nom et ses chemins rebasés (la **même**
  `retarget_config` que le renommage : ne pas la réécrire) — et **n'ouvre jamais
  la source en écriture**. C'est la propriété qui justifie la fonctionnalité :
  fine-tuner une fondation écrivait dans SON dossier, et après assez de pas sur
  un dataset étroit le modèle général avait disparu. On duplique, puis on
  fine-tune la copie. Le formulaire ouvre sur un nom libre (`<nom>-copy`, puis
  `-copy-2`…) et sur « config + poids », le défaut. Ce qui voyage est **le
  dernier run seul**, sous son nom daté, avec `latest.ckpt` reposé dessus dans la
  copie — pas l'historique (des dizaines de fichiers de 14 Mo). Le daté est
  retrouvé par **identité d'inode** avec `latest.ckpt`, jamais par le nom : c'est
  ce qui garde dans la copie la date à laquelle les poids ont été faits. Un échec
  n'y laisse pas de modèle à moitié fait. Contrat complet et e2e :
  `docs/reports/DUPLICATE_MODEL.md`.
- **Supprimer** exige le nom retapé à l'identique, puis efface le dossier entier.
  Le chemin visé doit être un **enfant direct** de `Models/` (les deux côtés
  canonicalisés) : un lien symbolique est refusé, pas suivi. `Delete` reste **la
  dernière ligne du menu** : y arriver au curseur doit rester une marche
  délibérée jusqu'au bout.
- Les trois **refusent un modèle dont ce processus tient un run**. Ce garde-fou est
  de l'état TUI : **il ne franchit pas la frontière de processus** — un second
  batlab ou un `--headless-train` dans un autre shell reste invisible. Non résolu,
  documenté sur `App::model_run_in_progress`.
- Sur les trois écrans du manager, **toute touche imprimable est du texte**, `q`
  compris (un modèle peut s'appeler `q-experiment`) ; `Esc` est la sortie. Sur le
  formulaire de duplication, `↑`/`↓` choisissent ce qui est copié — ce sont les
  seules touches dont le champ de nom ne veut pas.
- **Tout modèle créé par un template est conditionnable sur t** (`input_size.z >
  output.z`). Les deux templates ont livré l'inverse (1→1 et 3→3, la géométrie de
  `Models/Greyscale_Diffusion_broken`) jusqu'à ce qu'un test aveugle lise le
  `config_file` écrit. Ils dérivent maintenant leurs dims d'un seul
  `diffusion_unet(signal, temporel)` — ne pas y réintroduire de dims à la main.
  Gardé des deux côtés : `config::tests` sur `built_in_templates()`, et
  `every_model_created_from_a_template_is_conditionable_on_the_timestep` sur le
  fichier réellement écrit.
- **Les sauvegardes de poids sont datées.** Un run écrit **son** fichier,
  `run-<AAAA-MM-JJ_HHMM>.ckpt` (heure locale, zéro-padded : trier ces noms par
  nom, c'est trier les runs par date — ne pas remplacer le `_02` de collision
  par un `-02`, `-` trie avant `.` et casse la propriété), puis `latest.ckpt`
  est reposé dessus : un **lien dur**, pas une copie — un checkpoint XL fait
  14 Mo et une nuit avec `--checkpoint-every` en écrit des dizaines. Rien
  n'écrase rien : les poids d'hier sont encore là ce matin. Le `config_file`
  enregistre **où le run écrit**, `RunOptions::load_from` porte ce qu'il **lit**
  (le choix du sélecteur) — les deux étaient un seul champ, d'où l'écrasement.
  `--out` reste roi : nom exact, pas de date, pas de `latest.ckpt` à côté (tous
  les bancs sous `bench/` en dépendent). Sans `--out`, `--headless-train` reste
  du scratch, daté, sous `$TMPDIR/batlab-<modèle>/`. Contrat complet et e2e :
  `docs/reports/DATED_CHECKPOINTS.md`.
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

Les rapports de mission, avec les contrats observables complets et les parcours
e2e déroulés : `docs/reports/MODEL_MANAGER.md` (le manager),
`docs/reports/DUPLICATE_MODEL.md` (la duplication, et l'original intact),
`docs/reports/IMG2IMG_DRIFT.md` (la dérive img2img, la cause racine du « pause »
de la remontée, et le panneau d'architecture),
`docs/reports/DATED_CHECKPOINTS.md` (les checkpoints datés, `latest.ckpt` en
lien dur, la date à l'écran) et
`docs/reports/UX_NAV.md` (le chemin, les poids par défaut, l'aide — et la cause
racine du vol de focus au lancement sur macOS : winit appelle
`activateIgnoringOtherApps(true)` au démarrage de son event loop, ce que
`ActivationPolicy::Accessory` ne couvre pas ; ne pas retirer le
`with_activate_ignoring_other_apps(false)` de `visualiser::build_event_loop`).

## Lancer / valider un entraînement sans le TUI

Le binaire est un TUI interactif plein écran — impossible à scripter directement. Pour toute validation automatisée (agents, CI, tests de convergence), utiliser le chemin headless, qui réutilise `run_training` de production à l'identique :

```bash
cargo run -p batlab -- --headless-train <model> --steps N --dataset <path> [--lr F] [--batch N]
# ex. : cargo run -p batlab -- --headless-train Greyscale_Diffusion --steps 2000 --dataset datasets/cifar10_grey.batraw
```

Propriétés : ne réécrit jamais le `config_file` du modèle, écrit son checkpoint dans un fichier scratch daté (`$TMPDIR/batlab-<modèle>/run-<AAAA-MM-JJ_HHMM>.ckpt`, avec son `latest.ckpt` — n'écrase pas les poids sauvegardés, ni son propre run précédent), et part **from scratch par défaut** — `--resume <ckpt>` est l'opt-in explicite pour continuer. Marqué DEV/CI dans `crates/batlab/src/main.rs`, inatteignable depuis le TUI.

`--resume <ckpt>` reprend poids + moments Adam + compteur de pas `t` + EMA, et **écrit dans `--out`** : un run repris n'écrase jamais le fichier dont il vient. Une géométrie incompatible est refusée en nommant la couche et les longueurs. `--checkpoint-every N` écrit un partiel tous les N pas dans `<out stem>.partial.ckpt` — **un seul fichier, en rotation**, via un temporaire + `rename` atomique (un kill en pleine écriture ne peut pas laisser un `.ckpt` tronqué) ; c'est un checkpoint ordinaire, que `--resume` et le sélecteur de poids voient tous les deux, et sauf `--out` explicite `latest.ckpt` suit chaque rotation (un run tué à la neuvième heure laisse son partiel comme poids les plus récents). Le contrat complet des flags est dans `--help` et dans `docs/reports/EMA.md` §3.

Pour piloter le vrai TUI malgré tout (test end-to-end) : le lancer dans un pane tmux dédié et le piloter via `tmux send-keys`.

## Tests à l'aveugle (protocole projet)

Pour tout jalon à invariants comportementaux, les tests de l'agent d'implémentation ne suffisent pas : un agent qui a lu/écrit le code produit des tests-miroirs (ils affirment ce que le code fait, pas ce qu'il devrait faire — le mutation testing ne protège pas de ce biais). L'architecte lance donc, après l'implémentation, un **agent de test aveugle** : spec + contrats publics fournis, **interdiction de lire l'implémentation** (il peut compiler, exécuter, observer les sorties). Un écart se signale en citant la spec, jamais en éditant un test. Les deux suites coexistent ; leur désaccord est le signal.

La suite existante : `./blind_tests/run.sh` (`BLIND_BASELINE=1` ajoute la non-régression bit-à-bit contre un binaire d'un vieux commit ; ce chemin-là reconstruit une arborescence **pré-restructuration**, où le paquet binaire s'appelait encore `main` — le script résout le nom par arborescence, ne pas le figer).

## Points d'attention

- **Toujours cibler le bon paquet** : `cargo build/run --release -p batlab` pour le binaire, `-p batlab_core` pour le moteur, `-p batlab_ui` pour l'interface. (Le piège historique « un build à la racine ne construit que la lib racine » a disparu avec le paquet racine `batBuilder`, mais un `-p` explicite reste plus sûr.) Après un changement de flag CLI, vérifier la bannière du run — le binaire réémet sa config parsée, et `bench/optimizer/lib.sh:assert_flag` en fait un test.
- **Optimiseur** : Adam (`--optimizer adam --lr 1e-3`) converge ~20x plus vite que SGD en nombre de pas, surcoût < 1 % (voir `docs/reports/OPTIMIZER_ADAM.md`). L'init He est disponible (`--weight-init he`) mais n'a pas montré de gain (GroupNorm neutralise l'échelle en aval).
- **EMA des poids** (`--ema 0.999`, ou la ligne « EMA decay » du formulaire d'entraînement) : `ema ← d·ema + (1−d)·w` après chaque pas, et **c'est l'EMA qu'on utilise pour générer** — inférence, `--headless-sample` et Perpetual la prennent dès que le checkpoint en porte une, `--raw-weights` force l'itéré brut. Absent = aucune moyenne, run bit à bit et checkpoint octet pour octet inchangés. Le warmup est **double** : shadow semé sur les poids (jamais zéro) *et* rampe `d(t) = min(decay, (1+t)/(10+t))` — ne pas remplacer la rampe par une correction de biais, elle suppose un shadow parti de zéro et interdirait le semis. **Sur un run court la valeur nominale ne sert à rien** : à 1200 pas c'est la rampe qui lie (`d(1200)=0,9926`), donc `--ema 0.9999` y est identique à `--ema 0.999` ; elle ne distingue les runs qu'au-delà de ~10 000 pas. Le checkpoint devient `BBCKPT3` = `BBCKPT2` + une remorque EMA (+33 % de taille). **Mesurée NO-GO à 1500 pas** (`docs/reports/EMA.md` §5) : la moyenne est bien à 4,9 % de l'itéré et déplace la sortie sur 16 seeds sur 16, mais les deux tiers de ce déplacement sont un gain de contraste (1,08), un seul indicateur sur trois va dans le bon sens et `pixel_mean` s'éloigne du dataset. Ne pas l'activer sur un run court « pour faire propre » ; l'échelle qu'elle vise est ≥ 10 000 pas, où la valeur nominale reprend la main et où la trajectoire oscille au lieu d'avancer. Ce qui la jugerait vraiment est une MSE sur ε du jeu moyenné contre l'itéré — il n'existe pas de `--eval <ckpt>`.
- **Lecture des métriques de diffusion** : la MSE sur ε inverse l'importance des tranches — convertir en erreur x₀ (facteur ᾱ/(1−ᾱ)) avant de conclure. Baselines triviales : `tools/trivial_baselines.py`. Attention, le « déséquilibre 1e5 » de `docs/reports/SCALE_UNET.md` est en unités x₀ ; **en unités ε, où le gradient est réellement calculé, il vaut ≈5,7×** — et pondérer la loss ne rend que ce que 5,7× peut rendre (`docs/reports/LOSS_WEIGHTING.md` : ×1,20 sur le haut-t, NO-GO).
- **Pondération de la loss** : `--loss-weighting uniform|snr [--snr-gamma F]` (défaut `uniform`, bit-à-bit l'ancien tirage). Implémentée par biais du tirage de t, pas dans un shader. Sur ce schedule, **γ=5 (valeur de la littérature) ne redistribue presque rien** — `SNR(t) ≤ 5` dès t=33 ; utiliser γ≈1.
- **`inter_seed_std` n'est pas un critère de diversité lisible seul** : il est proportionnel à `--magnitude` (±6 % sur 0,3/0,6/1,0), donc il mesure surtout le bruit du sampler non débruité. Le lire à magnitude fixée, avec `intra_image_std` (doit approcher 0,206 par le bas, pas le dépasser) et `banding_ratio`.

- **Perpetual est une dérive img2img** : le run part d'une **vraie image du
  dataset** (`PerpetualOrigin::Image`, dataset dérivé des canaux de sortie —
  1 → `cifar10_grey`, 3 → `cifar10_rgb` — ou nommé par `seed_dataset` /
  `--seed-dataset`), et `[r]` en tire une autre. Introuvable → repli sur le bruit
  pur, annoncé ; **nommé et manquant → erreur**. `--seed-noise` rend l'ancienne
  ouverture. L'index de l'image est **avalanché** depuis la graine, jamais
  `seed % len` (même raison que `gaussian_at`).
- **En remontée, le modèle est appelé aussi.** C'était le « pause » : la montée
  est arithmétique pure (`forward_from`), donc x̂₀ restait figé `t_r + 1` frames —
  mesuré **294/294 frames de remontée gelées avant, 0/294 après**
  (`docs/reports/IMG2IMG_DRIFT.md`). Le fix ne touche pas la trajectoire, c'est un
  *read* : `asking_the_model_during_the_climb_does_not_move_the_latent` le tient
  au bit près. Coût, un appel modèle de plus par frame de montée : 523 → 324
  frames/s non throttlé, contre 30 pas/s de tempo par défaut. Il reste **une**
  frame répétée par cycle — le tournant, où la remontée et la descente lisent le
  même latent au même niveau ; c'est un contrat testé, pas une tolérance.
- **Le pas à pas des tenseurs d'un run perpetual vit dans le moteur**
  (`training/drift.rs`, `DriftWalk`) ; `perpetual.rs` ne porte que l'itinéraire.
  Le couplage au réseau tient en une question (`NoisePredictor`), ce qui rend le
  régime testable sans GPU contre un oracle. Cet oracle est **délibérément
  sous-confiant (×0,9)** : l'oracle exact inverse `x_t` et rend x̂₀ constant, ce
  qui rendrait le test du gel aveugle. Ne pas « corriger » cette constante.
- **La fenêtre perpetual montre x̂₀ SEUL par défaut** (`LiveView::X0Only`, carré,
  au vrai ratio de l'image) ; `[x]` rebascule sur `x_t | x̂₀`. L'inférence
  classique garde la double vue.
- **Graines de bruit : jamais `seed ^ index`.** Les images générées ont été des bandes horizontales pendant toute la campagne parce que le sampler XORait le pas dans la graine (`path_seed ^ diffusion_step`) pendant que le champ de bruit XORait l'index pixel : les deux se composent, et les 256 pas retiraient un seul champ permuté par `index ^ d ^ d'` (`docs/reports/ANISOTROPY_HUNT.md`). Chaque champ isolé restait isotrope — seule leur **somme** s'effondrait, d'où un entraînement et une sonde impeccables face à une génération morte. Passer par `gaussian_at` (avalanche de la graine, *puis* flux additif de l'index) ; l'ordre compte, l'inverse commute et donne un champ constant sur les anti-diagonales. Gardé par `injected_noise_over_reverse_chain_is_isotropic`.
- Les checkpoints antérieurs aux fixes du pipeline (padding `Same`, normalisation [-1,1], conditionnement temporel) sont invalidés — toujours réentraîner from scratch, ne pas charger d'anciens `.ckpt`.
- Un modèle de diffusion DOIT être conditionné sur le timestep : `input_size.z > output.z` (les canaux excédentaires reçoivent l'embedding temporel). Sans ça, ε̂ dégénère et l'échantillonnage explose en blanc saturé (voir `docs/reports/INSIGHTS_TRAINING.md`).
- Chaque run d'entraînement écrit un `*_metrics.jsonl` à côté du checkpoint (loss par tranche de t, stats ε̂ vs ε, trajectoires de débruitage). `--headless-sample <model> --ckpt <path>` génère des images + trajectoire depuis un checkpoint sans entraîner. Une loss batch qui décroît ne suffit PAS — vérifier la loss par tranche de t (une loss élevée à t bas = modèle qui n'utilise pas t).
- **Les limites GPU sont un choix : `--gpu-limits native|web`** (défaut
  `native`). `request_device` ne donne pas ce que l'adaptateur sait faire, il
  donne ce qu'on **demande** — et ne rien demander, c'était demander la base
  WebGPU : 256 Mio par tampon, 128 Mio par binding de stockage. Juste pour
  l'**inférence** (la cible est le navigateur du visiteur, cf. la frontière
  moteur/interface), faux pour l'**entraînement**, qui tourne ici : le même Mac
  autorise 28 Gio par tampon et 4 Gio par binding. Mesuré sur
  `Color_Diffusion_XL` : plafond de batch **341 → 4096** (c'était la limite de
  *binding*, pas la mémoire), et `cifar10_rgb` en u8 (146 Mio) passe de 2 chunks
  streamés à **UN seul chunk résident** — trafic dataset par pas **146,5 Mio →
  1,9 Kio**. Le dimensionnement suit : si le dataset tient dans un tampon que
  l'appareil accepte de lier ET sous le budget de résidence (`gpu_cap/4`, plus
  généreux que le budget de streaming `gpu_cap/8` parce qu'on le paie **une
  fois**), on prend tout. C'est ce qui rend ImageNet 32×32 u8 (3,94 Go < les
  4 Gio de binding) entièrement résident.
  **Honnêteté sur le gain** : sur cette machine à mémoire unifiée le trafic
  n'était **pas** le goulot — 10 pas à batch 32 prennent 46,0 s sous les deux
  profils, à 0,1 % près. Ce que le profil natif achète réellement ici, c'est le
  **plafond de batch** et la possibilité de tenir un corpus entier ; le gain de
  trafic, lui, se paierait sur une carte discrète (PCIe). Ne pas annoncer une
  accélération non mesurée.
  Garde-fous : l'inférence et le perpetual tournent sous les deux profils
  (`--headless-sample` rend des stats identiques au bit près), le profil
  **obtenu** (pas demandé) est imprimé par la bannière de run et par
  `--resources`, un adaptateur qui refuse ses propres limites fait retomber sur
  la base en le disant, et `--resources --no-gpu` / `--device <nom>` donnent
  toujours le verdict **contre les limites WebGPU** (`ProfileSource::Hypothetical`)
  pour savoir si un modèle passerait dans un navigateur.
- **Format dataset `.batraw` : le payload est en u8 (`BATRAW3`)**, élargi en
  [-1,1] **sur le GPU** (`training/shader/dataset_decode.wgsl`). Même en-tête
  qu'avant ; `BATRAW2` (f32 en [-1,1]) et `BATRAW1` (f32 en [0,1], rééchelonné)
  restent lisibles. Toutes les images de ce projet sont 8 bits à la source, donc
  l'ancien encodage stockait quatre octets dont trois se déduisaient du premier :
  CIFAR-10 RGB passe de 586 Mio à 146 Mio, ImageNet 32×32 de 15,7 Go à 3,9 Go.
  Ce qui compte n'est pas le disque mais le **trafic** — un chunk résident tient
  4× plus d'images, donc 4× moins de rechargements. Mesuré, pas déduit :
  `--resources Color_Diffusion_XL --batch 32 --dataset … --measure` donne
  **585,9 Mio/pas → 146,5 Mio/pas** (facteur 4,00) et 7 → 4 soumissions.
  Le décodage est **exact** (256 valeurs, aucun arrondi) : le shader lit une
  **table de 256 f32**, il ne recalcule pas `v/127.5 - 1` — écrite comme
  formule elle divergeait du CPU d'1 ULP sur 111 valeurs sur 256, Metal
  compilant la division en réciproque approchée. Ne pas « simplifier » la table
  en une formule. Gardé au bit près des deux côtés :
  `the_gpu_decode_agrees_with_the_cpu_one`,
  `an_8_bit_file_decodes_to_the_same_bits_as_the_f32_one_it_replaces`, et
  `an_8_bit_dataset_trains_exactly_like_the_f32_one_it_replaces` (mêmes pertes,
  pas pour pas). Un fichier 8 bits dont la géométrie ne colle pas au modèle
  retombe sur le chemin f32 : le rééchantillonnage se fait sur l'hôte de toute
  façon. La définition du format côté Python vit dans `tools/batraw.py`, seule
  et importée par les trois convertisseurs.
- Tests de non-régression du pipeline : `crates/batlab-core/src/model/audit_tests.rs` (`cargo test`). Ne pas les affaiblir pour les faire passer.
- **La racine de stockage s'injecte, elle ne se déduit pas.** `batlab_ui::storage::Storage` porte la racine de tous les chemins de données (`Models/`, `datasets/`, `perpetual_samples/`) ; `Storage::at(chemin)` en construit une ailleurs, et `App::with_storage` la fait descendre dans tout le TUI. **Tout test qui touche au stockage passe par `TempRoot`** — c'est ce qui a fait tomber la limite « `cargo test` réécrit `Models/Stable_Diffusion/config_file` » de `docs/reports/PERPETUAL_INFERENCE.md` §5. Critère de recette permanent : `cargo test --workspace` puis `git status` **propre**.
  Le défaut (`Storage::default()`, et les fonctions libres du module que le CLI utilise) reste le workspace, trouvé en remontant jusqu'au `Cargo.toml` portant `[workspace]` : ne pas le réécrire en un nombre fixe de `parent()` — déplacer un crate ferait alors pointer `Models/` ailleurs, **sans erreur**, juste des listes vides. Gardé par `project_root_is_the_workspace_that_holds_models_and_datasets`.
  **`BATLAB_ROOT=<dir>`** force cette racine par défaut pour tout le processus — c'est la façon de dérouler le vrai TUI end-to-end sans écrire dans le `Models/` du dépôt.
- macOS : l'event loop winit du visualiseur doit vivre sur le main thread (le TUI et l'entraînement tournent sur un worker) — ne pas réintroduire de `EventLoop::new()` dans un thread secondaire.
- Les rapports sous `docs/reports/` sont des **archives** : leur texte cite les anciens chemins (`bat_building/src/…` — qui couvrait alors moteur ET interface —, `main/src/main.rs`, `perpetual_samples/…`) et n'a pas été réécrit. La table de correspondance est dans `docs/reports/INDEX.md`.
