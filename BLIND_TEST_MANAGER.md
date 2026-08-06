# Test à l'aveugle du gestionnaire de modèles et de la navigation — verdict

Branche : `blind-test-manager` · Plateforme : macOS (Darwin 25.4.0) · binaire
`target/release/batlab` construit depuis `feadcae`.

Agent de test **indépendant** : aucun fichier de `crates/*/src/` n'a été lu, ni
leurs diffs, ni leur historique. Sources autorisées et effectivement utilisées :
`MISSION_BLIND_TEST_MANAGER.md`, `CLAUDE.md` (flow, `BATLAB_ROOT`, points
d'attention), `docs/reports/MODEL_MANAGER.md` (la spec), et
`docs/reports/INFER_VIZ.md` §5 pour le protocole de pilotage tmux. Tout le reste
vient de l'écran et du disque.

## Verdict par propriété

| # | Propriété | Verdict | Assertions |
| --- | --- | --- | --- |
| P1 | Flow : liste des modèles à l'ouverture → menu d'actions | **PASS** | 17 ok |
| P2 | Rename : dossier + `model_name` + chemins, validation des noms | **PASS** | 26 ok |
| P3 | Delete : re-frappe exacte, dossier entier, rien d'autre touché | **PASS** | 24 ok |
| P4 | Navigation : `Esc` d'un cran, deux racines, aucun écran orphelin | **PASS** | 35 ok, 1 skip |
| P5 | Checkpoints : seuls les `.ckpt`, checkpoint frais relu | **PASS** | 22 ok |
| P6 | Hygiène : `cargo test --workspace` laisse `git status` propre | **PASS** | 4 ok |
| P7 | Contrat d'inférence : une inférence courte aboutit | **PASS** ¹ | 12 ok |
| — | Sondes hostiles (racine vide, casse, garde de run, `[r]`) | **PASS** ¹ | 12 ok |
| — | Géométrie des templates | **FAIL** | 4 ok, **2 FAIL** |

¹ avec un écart de spec consigné, sans défaut observable — détail plus bas.

**152 assertions, 2 échecs**, tous deux sur le même défaut (§ « Le défaut »).
Aucune des sept propriétés de l'ordre de mission n'est en échec : le
gestionnaire de modèles tient son contrat observable, y compris les garde-fous
que la spec annonce mais ne déroule pas (lien symbolique, AppleDouble, système
de fichiers insensible à la casse).

Rejouer :

```bash
cargo build --release -p batlab
./blind_tests_manager/run.sh              # tout
./blind_tests_manager/run.sh t03 t05      # une sélection
```

Chaque test travaille sur une racine `BATLAB_ROOT` jetable peuplée en shell
depuis `Models/` du dépôt ; **le TUI n'a jamais été ouvert sur le vrai
`Models/`**, et t06/t07 le vérifient par empreinte SHA-256 avant/après.

## Le défaut

### Les deux templates créent un modèle non conditionnable sur le timestep

`blind_tests_manager/t09_template_geometry.sh` — 2 FAIL.

Le flux « New model (from template) », dernière ligne de l'écran d'accueil,
écrit sur disque :

| Template | `input_size.z` | canaux de sortie |
| --- | --- | --- |
| Greyscale Diffusion | 1 | 1 |
| Stable Diffusion | 3 | 3 |

CLAUDE.md, section « Points d'attention » :

> Un modèle de diffusion **DOIT** être conditionné sur le timestep :
> `input_size.z > output.z` (les canaux excédentaires reçoivent l'embedding
> temporel). Sans ça, ε̂ dégénère et l'échantillonnage explose en blanc saturé
> (voir `docs/reports/INSIGHTS_TRAINING.md`).

Les deux templates violent l'inégalité : ils donnent `input_size.z == output.z`,
donc **zéro canal** pour l'embedding temporel. Le repère qui rend le verdict
lisible est dans le dépôt lui-même :

```
Models/Greyscale_Diffusion          input_size.z=3  → sortie 1     (3 canaux de t)
Models/Greyscale_Diffusion_L        input_size.z=5  → sortie 1
Models/Color_Diffusion_L / _XL      input_size.z=7  → sortie 3
Models/Greyscale_Diffusion_broken   input_size.z=1  → sortie 1     ← le template
```

La configuration produite par le template « Greyscale Diffusion » est, à la
section `inference` près (graines), **identique à celle de
`Models/Greyscale_Diffusion_broken`** — le modèle que ce dépôt a lui-même nommé
« broken ». Un utilisateur qui suit le flow d'accueil (« New model (from
template) » → Greyscale → Train) obtient donc la géométrie que le projet a
archivée comme cassée.

Ce n'est pas une régression de la mission `model-manager` : le template lui est
antérieur. Mais c'est le **premier geste** que le nouveau flow propose, et le
parcours e2e de `MODEL_MANAGER.md` §5 est passé à côté — le pas 3 y consigne
`32x32x1 · 12 layers` comme « OK ». La liste affiche `input_size`, pas la sortie,
donc l'écran ne dit rien de l'inégalité : rien, dans le TUI, ne signale que le
modèle qu'on vient de créer ne peut pas apprendre le débruitage.

Non démontré ici : la dégénérescence en aval (« blanc saturé »). Un aveugle
signale l'écart à l'invariant écrit ; il ne rejoue pas la campagne
d'entraînement qui l'a établi.

## Écarts de spec consignés (ni bug, ni PASS gratuit)

### 1. `[i]` n'ouvre pas la géométrie d'entrée depuis l'écran que `[e]` ouvre

CLAUDE.md et `MODEL_MANAGER.md` §1 disent :

> `[e]` depuis le menu d'actions ouvre le constructeur de couches, `[i]` dedans
> ouvre la géométrie d'entrée.

Observé : `[e]` depuis le menu d'actions ouvre la **liste des couches**
(`[up/down] navigate  [Enter] edit selected  [d] delete selected  [e/Esc] back
to add  [q] quit`), où `[i]` est **inerte**. Il faut un `Esc` — qui remonte vers
`Add Layer` — pour trouver l'écran où `[i]` marche ; son pied de page l'annonce
alors explicitement (`[i] input size`).

Les écrans sont donc **honnêtes** : chacun n'annonce que ce qu'il fait. C'est la
phrase de spec qui est lâche (« dedans » recouvre deux états distincts du
constructeur). `Screen::InputSize` est bel et bien atteignable au clavier depuis
la porte d'entrée — le troisième écran fantôme est bien câblé.

### 2. L'inférence n'écrit aucun PNG

L'ordre de mission, P7 : « une inférence courte aboutit **(PNG produit)** ». La
spec du manager, `MODEL_MANAGER.md` §5 pas 12, ne promet que ceci : « run
complet, **aperçu affiché** → OK — `Inference Preview (32x32x1)` ».

Observé (`t07_infer.sh`) : l'inférence depuis un checkpoint frais aboutit en 2 s,
l'aperçu `Inference Preview (32x32x1)` est affiché et porte 16 lignes de pixels ;
**aucun fichier PNG n'apparaît sous la racine**. Les seuls PNG présents viennent
du run d'**entraînement** (`datasets/generated_samples/step_0019.png`) — ce que
§5 documente d'ailleurs comme tel.

Deux textes de spec en désaccord, le comportement conforme au plus précis des
deux : consigné, pas compté en échec. À trancher côté produit — soit l'inférence
doit sauvegarder, soit la formulation de l'ordre doit être corrigée.

### 3. La garde de run §2.4 n'est pas observable au clavier

`MODEL_MANAGER.md` §2.4 :

> Rename et Delete **refusent tous deux un modèle dont ce processus tient un
> run**. Cette garde est de l'état TUI (`running_model`, posé au démarrage d'un
> run, levé quand le run se déclare fini).

Observé : tant qu'un run tourne, le moniteur n'offre **aucun chemin** vers le
menu d'actions. Son pied de page est `[p] pause/resume  [t] tune params
[v] visualise  [s] save snapshot  [q] quit` — `[r] new run` n'apparaît qu'une
fois le run **terminé**, donc la garde déjà levée. `[p]` (pause) ne change rien,
`[r]` reste inerte, et `Esc`/`[q]` quittent l'application.

Le refus annoncé est donc du **code défensif que le TUI ne laisse pas
atteindre** dans un seul processus. Ce n'est pas un défaut — c'est une garde qui
ne peut pas se démentir à l'usage —, mais §2.4 se lit comme un comportement
observable alors qu'il ne l'est pas. La limite sérieuse reste celle que §2.4 et
§6 annoncent : la garde ne franchit pas la frontière de processus, et ce chemin-
là est, lui, parfaitement atteignable (un `--headless-train` dans un autre
shell). **Intacte, confirmée non résolue.**

Le versant utile a été testé et passe : après un run qui **échoue** (inférence
sans poids), la garde se lève correctement — `Rename` fonctionne juste après. Un
`running_model` resté coincé aurait bloqué le manager à vie ; ce n'est pas le
cas.

## Ce qui a été vérifié, propriété par propriété

### P1 — Flow (17 ok)

L'ouverture est bien la liste (`batlab — Models`), chaque entrée porte
`NxNxN · N layers · <checkpoints>`, `New model (from template)` est la
**dernière** ligne (vérifié par position, pas par présence). `Enter` sur un
modèle ouvre le menu titré de son nom, avec les cinq actions Train / Infer /
Perpetual / Rename / Delete, et rappelle la géométrie. Le compteur distingue
`1 checkpoint: one.ckpt` de `no checkpoints`.

### P2 — Rename (26 ok)

**Refus** vérifiés, chacun laissant le disque bit-à-bit inchangé (empreinte
SHA-256 de toute la racine avant/après les six refus) : `../evil`
(« must not start with '.' », et **aucun** `evil` créé hors de `Models/`), nom
vide, collision, 65 caractères, `.hidden`, `a/b` (aucune arborescence
imbriquée créée).

**Renommage valide** vers `q-experiment.v1_2` — un nom qui commence par `q` et
contient un `.` interne :

- dossier déplacé, ancien dossier absent, poids suivis ;
- `model_name` réécrit dans le `config_file` ;
- `inference.checkpoint`, positionné avant le test sur un chemin **interne** au
  dossier du modèle, **réécrit** vers le nouveau dossier — et le chemin réécrit
  désigne un fichier qui existe ;
- accusé `✓ Renamed 'Alpha_Model' → 'q-experiment.v1_2'`, la liste montre le
  nouveau nom, le modèle se rouvre ;
- hors `config_file`, le contenu du modèle est **bit-à-bit identique** et rien
  d'autre sous la racine n'a bougé.

### P3 — Delete (24 ok)

Écran non armé à l'ouverture. `q` tapé est du **texte** (l'application survit et
reste sur l'écran). Trois frappes inexactes refusées — préfixe
(`q-experiment.v1` → `✗ Type the model name exactly`), casse différente
(`Q-EXPERIMENT.V1_2`), nom d'un **autre** modèle (`Bystander`, qui n'est pas
supprimé non plus) — et l'empreinte complète de la racine est inchangée après
les trois. La barre d'aide ne bascule sur `[Enter] DELETE — no further prompt`
qu'à la frappe exacte. `Esc` annule sans quitter.

Sur la frappe exacte : accusé `✓ Deleted`, dossier entier effacé (config,
`.ckpt`, `latest_metrics.jsonl`), et l'empreinte avant/après montre que **rien
d'autre** n'a disparu ni été créé — ni le modèle voisin, ni les témoins déposés
sous `datasets/` et `perpetual_samples/`.

**Verrou 3 (lien symbolique), que la spec annonce sans le dérouler** : un dossier
de modèle qui est un `symlink` vers une arborescence **hors** de `Models/` est
listé, l'écran Delete s'ouvre, la frappe exacte est acceptée — et la cible reste
**bit-à-bit intacte**, avec un `✗` à l'écran. Le lien n'est pas suivi. Conforme.

### P4 — Navigation (35 ok, 1 skip)

Quatorze écrans atteints au clavier depuis la porte d'entrée, chacun quitté par
un `Esc` vérifié **non seulement sur le parent atteint, mais sur le grand-parent
non atteint** (un `Esc` qui saute deux crans échoue le test) :

```
liste ─┬─ templates ────────────────────── Esc → liste
       └─ menu d'actions ─┬─ Rename ────── Esc → menu d'actions
                          ├─ Delete ────── Esc → menu d'actions
                          ├─ poids ─┬─ Training Parameters ─ Dataset
                          │         ├─ Inference Parameters
                          │         └─ Perpetual Inference
                          └─ [e] liste des couches ─ Esc → Add Layer ─ [i] → Model Input Size
```

`Esc` quitte depuis exactement **deux** écrans — la liste et le moniteur —, code
de sortie **0** les deux fois. Le skip est l'écart de spec n°1 ci-dessus.

### P5 — Checkpoints (22 ok)

Six leurres déposés dans `pretrained_weights/` à côté d'un vrai `real_one.ckpt` :
`foo_metrics.jsonl`, `README.txt`, `weights.ckpt.bak`, `._x.ckpt`,
`._latest.ckpt` (AppleDouble), et un **répertoire** nommé `archive.ckpt`. Le
compteur de la liste comme le sélecteur de poids n'en retiennent **aucun** —
`1 checkpoint: real_one.ckpt`. Les deux listages sont d'accord, y compris sur le
répertoire en `.ckpt`, cas que la spec §2.5 ne mentionne pas.

Puis un vrai run de 20 pas : `latest.ckpt` et `latest_metrics.jsonl` écrits côte
à côte, `[r] new run` → le menu d'actions **relit le disque** et annonce
`2 checkpoints`, le JSONL exclu ; le sélecteur de poids propose le checkpoint
frais et pas le journal. Les correctifs §4.1 et §4.2 tiennent.

### P6 — Hygiène des tests (4 ok)

Sur un arbre propre : `cargo test --workspace` → **167 tests verts, 0 échec**
(13 + 108 + 46, 5 ignorés) ; `git status --porcelain` **vide** après ; et
l'empreinte SHA-256 de tout `Models/` du dépôt est **inchangée**. La limite
`PERPETUAL_INFERENCE.md` §5 est bien levée.

### P7 — Inférence (12 ok, 1 écart)

- Infer **sans poids** : message explicite `inference checkpoint not found`,
  chemin cherché **dans la racine jetable**, application vivante.
- Train 20 pas → poids chargeables → `[r]` → menu d'actions → Infer sur
  `latest.ckpt` → `Inference Preview (32x32x1)` en 2 s, 16 lignes de pixels
  dessinées, aucun message d'erreur résiduel.
- Confinement : `Models/` et `datasets/` du dépôt bit-à-bit intacts après tout
  le parcours.
- Sortie : code **0**, **stderr 0 octet** sur toute la session.
- Écart n°2 (aucun PNG) ci-dessus.

### Sondes hostiles (12 ok, 1 écart)

- **Racine vide** : la liste s'ouvre, « No models yet », **aucun `Models/`
  créé** — regarder ne coûte plus rien. L'asymétrie annoncée en §6 est
  confirmée : `datasets/` **est** créé à l'ouverture.
- **`q` sur Rename** aussi est du texte (§2.1 dit « les deux écrans » ; P3 ne
  couvrait que Delete).
- **`[r]` rafraîchit** : un modèle déposé sur disque pendant la session apparaît.
- **Casse du système de fichiers** — le cas que la spec ne prévoit pas : sur
  APFS insensible à la casse, renommer `Alpha` en `beta` alors que `Beta`
  existe. Refusé (`✗ a model named 'beta' already exists`), et le contenu de
  `Models/Beta` est **intact** (SHA-256 d'un fichier témoin). Un `fs::rename`
  aveugle aurait écrasé un autre modèle.
- **Renommer vers son propre nom** : refusé proprement, modèle intact.
- **Garde de run** : écart n°3 ci-dessus, plus la vérification que la garde se
  **lève** après un run en échec.

## Ce que je n'ai pas pu tester

- **La garde de run inter-processus** (§2.4/§6) : non résolue par construction,
  et je n'ai pas cherché à la démentir — c'est une limite déclarée, pas une
  promesse.
- **L'échec de réécriture du `config_file` pendant un renommage** (§2.2, « le
  renommage est défait ») : je n'ai pas trouvé de moyen boîte noire de faire
  échouer l'écriture au bon instant sans toucher au code. Le chemin heureux est
  vérifié ; le chemin de rattrapage ne l'est pas.
- **La dégénérescence en aval de la géométrie des templates** (blanc saturé) :
  signalée comme écart à l'invariant écrit, non rejouée expérimentalement.

## Le harnais

`blind_tests_manager/` — bash + tmux, aucune dépendance au code.

| Fichier | Rôle |
| --- | --- |
| `lib.sh` | racines jetables, lancement tmux détaché, envoi de touches (texte et `Enter` séparés, cf. CLAUDE.md), capture d'écran, `bt_select` qui vise le curseur `>` **par lecture d'écran** et jamais par comptage de flèches, empreinte SHA-256 récursive, `bt_redraw` (le journal de métriques écrit sur stdout par-dessus le TUI : un redimensionnement force ratatui à repeindre) |
| `t01`…`t09` | une propriété par fichier, chaque échec citant la phrase de spec |
| `run.sh` | runner, verdict PASS/FAIL par propriété, journaux conservés |

Trois pièges rencontrés, notés pour la suite :

1. **Les menus gardent leur curseur** d'une visite à l'autre. Compter les
   flèches donne une suite verte qui teste autre chose que ce qu'elle annonce —
   d'où `bt_select`, qui regarde l'écran.
2. **Le panneau d'aperçu s'appelle `Inference Preview` dès le début du run** et
   ne gagne ses dimensions `(32x32x1)` qu'à la fin. Attendre le titre nu, c'est
   conclure « abouti » à 89 % du run.
3. **`[e] edit layers` figure dans le pied de page de deux écrans** (menu
   d'actions et `Add Layer`) : le marqueur qui distingue le menu d'actions est
   `[Enter] confirm  [e] edit layers`. Un marqueur ambigu fait passer un `Esc`
   qui saute un cran pour un `Esc` correct.
