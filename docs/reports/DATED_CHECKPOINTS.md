# Checkpoints datés — l'historique des runs, et la date à l'écran

Branche `dated-ckpt`. Deux volets : le **nom de fichier porte la date**, et la
**date est visible** dans le TUI.

**Résumé** — un run écrivait `latest.ckpt` et écrasait le précédent : entraîner
ce soir effaçait les poids de ce matin, et rien à l'écran ne disait quand un
checkpoint avait été produit. Un run écrit désormais **son** fichier,
`run-<AAAA-MM-JJ_HHMM>.ckpt`, et `latest.ckpt` est reposé dessus — **un lien
dur**, pas une copie. Le sélecteur de poids et la liste des modèles montrent
date et taille, triés du plus récent au plus ancien. `--out` reste roi.

---

## 1. Ce que le run écrit

### 1.1 Le nom

`run-<AAAA-MM-JJ_HHMM>.ckpt`, en **heure locale**, zéro-padded, plus grande
unité d'abord. La seule propriété qui compte : **trier ces noms par nom, c'est
trier les runs par date**. C'est ce que `ls`, un glob de shell et le sélecteur
obtiennent gratuitement, sans avoir à `stat` quoi que ce soit.

Deux détails qui ne sont pas cosmétiques :

- **Heure locale, pas UTC** (`crates/batlab-ui/src/clock.rs`, via `localtime_r`
  — `std` n'a ni calendrier ni fuseau). « Celui d'hier soir » est une notion
  d'heure locale : un checkpoint écrit à 22 h 41 à Paris ne peut pas être classé
  sous `2041` UTC dans son nom et affiché `22:41` sur la ligne d'à côté.
- **Le suffixe de collision est `_02`, pas `-02`.** Deux runs dans la même
  minute — deux fumigènes de 20 pas — se disputent le même nom. Le premier
  suffixe essayé était `-02` : `-` (0x2D) trie **avant** `.` (0x2E), donc
  `run-…_1041-02.ckpt` passait devant `run-…_1041.ckpt` et cassait la seule
  propriété pour laquelle l'horodatage existe. `_` (0x5F) trie après `.`.
  Trouvé par le test, pas à la relecture.

### 1.2 `latest.ckpt` : un lien dur

Le choix demandé était « copier `latest.ckpt` » **ou** « en faire un alias
résolu ». C'est un troisième terme qui a été retenu, qui a les avantages des
deux : **un lien dur posé à côté du fichier daté**.

| | copie | lien symbolique | **lien dur** |
| --- | --- | --- | --- |
| coût disque | ×2 par sauvegarde (14 Mo pour le XL, à chaque rotation de `--checkpoint-every`) | nul | **nul** |
| `latest.ckpt` est un vrai fichier | oui | non — peut pendre si le daté est supprimé | **oui, toujours** |
| survit à la suppression du daté | oui | non | **oui** (les octets vivent tant qu'un nom les référence) |
| lisible par `--resume`, le sélecteur, `fs::read` | oui | oui | **oui** |

Le coût disque est l'argument décisif : une nuit d'entraînement avec
`--checkpoint-every` écrit des dizaines de partiels ; en payer une seconde copie
à chaque fois, pour un fichier qui n'est qu'un **nom**, serait absurde.

Posé via un scratch **caché** (`.latest.ckpt.tmp`, donc jamais listable comme
poids — l'extension seule le laisserait passer) puis un `rename` : `latest.ckpt`
n'est jamais observé absent ni à moitié lié. Repli sur une copie si le système
de fichiers refuse le lien.

`latest.ckpt` suit **aussi** chaque rotation de partiel. Deux raisons : un run
tué à la neuvième heure doit laisser son partiel comme poids les plus récents
(c'est la raison d'être des partiels), et un `rename` remplace le *nom*, pas
l'inode — un lien non refait épinglerait sur disque les octets de la rotation
précédente, soit exactement la copie qu'on voulait éviter.

### 1.3 Lire n'est plus écrire

C'était le nœud. Le sélecteur de poids remplissait `TrainingConfig.checkpoint_path`,
et `run_training` s'en servait pour **charger** *et* **sauvegarder** : continuer
un entraînement réécrivait par-dessus les poids dont il partait.

Les deux bouts sont maintenant distincts :

- le `config_file` du modèle enregistre **où le run écrit** (son fichier daté) —
  c'est une propriété du modèle, et c'est ce qui fait qu'en rouvrant le modèle
  demain le curseur tombe sur les poids que ce run a produits, pas sur ceux dont
  il était parti ;
- `RunOptions.load_from` porte **ce que le run lit** — quel fichier on a choisi
  de continuer ce soir n'est pas une propriété du modèle et n'a rien à faire
  dans son fichier de config (même raison que `resume_from`).

## 2. Contrat observable

1. **Un run du TUI écrit `Models/<modèle>/pretrained_weights/run-<stamp>.ckpt`**
   et ne touche à aucun autre `.ckpt`. Le checkpoint dont il est parti est
   intact, octet pour octet, mtime compris.
2. **`latest.ckpt` existe et désigne le plus récent** — même contenu, même
   inode que le fichier daté. Vrai après le checkpoint final, après un `[s]` du
   moniteur et après chaque rotation de `--checkpoint-every`.
3. **`--out <path>` est écrit exactement là, exactement sous ce nom** : pas de
   date, pas de `latest.ckpt` déposé à côté. Nommer le fichier *est* la façon de
   sortir de la convention ; tous les bancs sous `bench/` en dépendent.
4. **`--headless-train` sans `--out` reste du scratch** : jamais dans les poids
   sauvegardés d'un modèle. Il est daté lui aussi, dans
   `$TMPDIR/batlab-<modèle>/` — un répertoire par modèle, parce que le nom daté
   n'est unique que par répertoire.
5. **Deux runs dans la même minute donnent deux fichiers**, le second suffixé
   `_02`, et le suffixe trie après le nom nu.
6. **Le sélecteur de poids et la liste des modèles sont triés du plus récent au
   plus ancien**, chaque ligne portant sa date (`2026-08-08 11:20`) et sa taille
   (`78 kB`). Les dates sont alignées en colonne : c'est un balayage vertical,
   pas une lecture.
7. **Les checkpoints antérieurs à la convention restent listables et
   chargeables** — `latest.ckpt`, `night_run.ckpt`, `night2.ckpt` du dépôt sont
   des checkpoints ordinaires. Aucune migration, rien de destructif.
8. `--resume`, `--checkpoint-every` et la sélection de poids sont inchangés dans
   leur contrat ; les métriques d'un run suivent le fichier que ce run écrit
   (`run-<stamp>_metrics.jsonl`), donc les diagnostics sont à côté des poids
   qu'ils décrivent.

Tenu par : `storage.rs` (`dated_run_names_sort_by_name_in_the_order_the_runs_happened`,
`two_runs_in_the_same_minute_get_two_files`,
`latest_holds_the_newest_bytes_without_a_second_copy`,
`checkpoints_are_listed_newest_first_and_carry_their_date_and_size`,
`checkpoints_that_predate_the_naming_convention_are_still_offered`),
`clock.rs` (`a_stamp_sorts_as_text_exactly_as_it_sorts_in_time`),
`tui/ui.rs` (`a_weight_row_carries_its_date_and_its_size`,
`the_dates_line_up_whatever_the_names_are`,
`the_model_list_summary_dates_the_newest_checkpoint`),
`main.rs` (`an_explicit_out_is_written_exactly_as_named`,
`a_headless_run_without_out_is_dated_and_stays_in_scratch`).

## 3. Le e2e, déroulé

Racine jetable (`BATLAB_ROOT=$TMPDIR/batlab-e2e`), un `config_file` copié, le
dataset en lien symbolique. Le `Models/` du dépôt n'est jamais touché.

**Étape 0** — un checkpoint « d'avant », écrit par un run headless nommé :

```
--headless-train Greyscale_Diffusion --steps 5 --out …/pretrained_weights/night_run.ckpt
→ night_run.ckpt, night_run_metrics.jsonl. Rien d'autre : pas de latest.ckpt.
```

**Étape 1** — le TUI, piloté par `tmux send-keys`. Liste des modèles :

```
> Greyscale_Diffusion
    32x32x3 · 12 layers · 1 checkpoint (newest 2026-08-08 11:20): night_run.ckpt
```

**Étape 2** — `Train` → sélecteur de poids, curseur sur le seul checkpoint :

```
    Start from random weights (new run)
  Load existing pretrained weights:
    > night_run.ckpt   2026-08-08 11:20 · 78 kB
```

**Étape 3** — 20 pas menés à terme. Sur disque :

```
15185248 -rw-r--r--  2  latest.ckpt                       11:20
15183938 -rw-r--r--  1  night_run.ckpt                    11:20   ← intact
15185248 -rw-r--r--  2  run-2026-08-08_1120.ckpt          11:20
            ^ même inode que latest.ckpt, link count 2 : un seul jeu d'octets
```

et le `config_file` enregistre `checkpoint_path: …/run-2026-08-08_1120.ckpt` —
le fichier que ce run a écrit.

**Étape 4** — relance du TUI sur la même racine. La liste dit « 3 checkpoints
(newest 2026-08-08 11:20) », et le sélecteur ouvre sur les poids que le run
vient de produire :

```
  Load existing pretrained weights:
      latest.ckpt                2026-08-08 11:20 · 78 kB
    > run-2026-08-08_1120.ckpt   2026-08-08 11:20 · 78 kB
      night_run.ckpt             2026-08-08 11:20 · 78 kB
```

**Étape 5 (headless)** — 30 pas avec `--checkpoint-every 10` : le daté, son
`.partial.ckpt`, `latest.ckpt` au même inode. Un second run dans la même minute
→ `run-2026-08-08_1112_02.ckpt`, lien repointé, checkpoint du premier intact. Un
run `--out …/my_named.ckpt` → `my_named.ckpt` et son JSONL seuls dans leur
répertoire, aucun `latest.ckpt`.

## 4. Ce qui n'est pas fait

- **Pas de purge.** Un an de runs nocturnes, c'est un an de fichiers. La
  rotation existante ne concerne que le partiel (un seul fichier, par
  construction) ; un `--keep-last N` serait une autre mission, avec sa propre
  question (« que garde-t-on : les N derniers, ou un tous les mille pas ? »).
- **La date affichée est la mtime du fichier**, pas un horodatage stocké dans le
  checkpoint. Copier un `.ckpt` sans préserver la mtime le fera paraître neuf.
  Le nom, lui, garde la vraie date — c'est la raison d'être du nom.
- **Le format `BBCKPT3` est inchangé** : rien n'a été ajouté dans les octets du
  checkpoint, donc tous les fichiers existants restent lisibles sans conversion.
