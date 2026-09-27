# Mission : test à l'aveugle du gestionnaire de modèles et de la navigation

Tu es un agent de test INDÉPENDANT. Règle absolue : **interdiction de lire l'implémentation** — aucun fichier de `crates/*/src/`, ni leurs diffs, ni leur historique. Tu peux : compiler (`cargo build --release -p batlab`), exécuter le binaire (TUI piloté par `tmux send-keys` dans un pane dédié — méthode éprouvée, cf. `docs/reports/INFER_VIZ.md` §5 pour le protocole d'envoi et de capture), observer l'écran (`tmux capture-pane`) et le DISQUE. Tu peux lire : ce fichier, l'AGENTS.md racine (sections flow/BATLAB_ROOT), et `docs/reports/MODEL_MANAGER.md` (le contrat observable — c'est ta spec). Un écart se signale en citant la spec ; tu n'édites jamais un test pour le faire passer.

## Dispositif

Travaille sur une racine de stockage JETABLE : `BATLAB_ROOT=<tmpdir>` (documenté dans AGENTS.md). Peuple-la en copiant 1-2 modèles depuis le vrai `Models/` (config_file + un petit .ckpt) — en shell, sans lire leur contenu. JAMAIS le TUI sur le vrai `Models/`.

## Propriétés à tester (spec : MODEL_MANAGER.md + AGENTS.md « le modèle d'abord, l'action ensuite »)

- **P1 Flow** : l'ouverture est la liste des modèles (nom, géométrie, checkpoints) + « New model (from template) » ; sélectionner un modèle ouvre le menu d'actions (Train/Infer/Perpetual/Rename/Delete).
- **P2 Rename** : renomme le dossier ET le model_name du config_file, de façon cohérente ; la liste reflète le nouveau nom ; validation des noms (vide/collision/caractères dangereux refusés — essaie `../evil`, un nom existant, un nom vide).
- **P3 Delete** : exige la re-frappe exacte du nom ; une frappe inexacte ne supprime rien ; la suppression retire le répertoire entier ; RIEN d'autre sous la racine n'est touché (état complet du disque avant/après comparé) ; un modèle nommé avec des caractères spéciaux (dont `q`) reste manipulable puisque toute touche imprimable est du texte.
- **P4 Navigation** : Esc remonte d'exactement un cran depuis chaque écran atteignable ; l'application ne quitte que depuis la liste et le moniteur ; aucun écran annoncé n'est inatteignable au clavier.
- **P5 Checkpoints** : seuls les `.ckpt` sont listés/comptés (dépose un `foo_metrics.jsonl` et un `._x.ckpt` dans pretrained_weights : ils ne doivent pas apparaître) ; après un run court, le menu d'actions relu propose le checkpoint frais.
- **P6 Hygiène des tests** : `cargo test --workspace` laisse `git status` propre (la limite historique est annoncée levée — vérifie-le).
- **P7 Contrat d'inférence** : depuis le menu d'actions, une inférence courte aboutit (PNG produit) sur le modèle copié.

## Livrables (sur CETTE branche `blind-test-manager`)
- Suite boîte noire exécutable `blind_tests_manager/` (scripts + runner PASS/FAIL par propriété — le pilotage tmux peut être scripté).
- `BLIND_TEST_MANAGER.md` : verdict par propriété, chaque FAIL citant la spec, défauts de spec consignés.
- Commits atomiques, ne jamais push. GPU parfois occupé (runs) : tes inférences seront lentes, c'est normal.
