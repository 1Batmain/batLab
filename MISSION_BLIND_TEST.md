# Mission : test à l'aveugle du régime « flux » (mode Perpetual)

Tu es un agent de test INDÉPENDANT. Règle absolue : **interdiction de lire l'implémentation** — tu ne dois ouvrir AUCUN fichier de `bat_building/src/` ni `main/src/` (ni leurs diffs/historique git). Tu peux : compiler (`cargo build --release -p main`), exécuter le binaire, lire les sorties (dumps, PNG, stdout/stderr), lire CE fichier et le CLAUDE.md racine (sections d'usage). Si tu as besoin d'une information que seule l'implémentation contient, c'est un défaut de spec : consigne-le, ne triche pas. Tu n'édites JAMAIS un test pour le faire passer : un écart se signale en citant la ligne de spec violée.

## La spec que tu testes (contrat, indépendant du code)

Le binaire (`./target/release/main`, à builder avec `-p main`) expose un mode headless :

    --headless-perpetual --regime flux --actions N --dump <chemin> \
      [--t-star K] [--seed S] --checkpoint <ckpt>

Checkpoint fourni : `Models/Greyscale_Diffusion_L/pretrained_weights/night_run.ckpt` (copie-le depuis `/Users/bat/development/lab/batLab/Models/Greyscale_Diffusion_L/pretrained_weights/` si absent). Le dump est du f32 brut : par frame, deux images 32×32 (x_t puis x̂₀), frames concaténées. Les régimes `errance` et `respiration` existent aussi (mêmes flags, `--t-r` pour la profondeur). Explore `--help` si des flags manquent à cette description — le help fait partie du contrat public.

Comportement spécifié du régime flux (source : demande utilisateur + ordre de mission) :
1. **Stationnarité** : le niveau de bruit reste à t* — std(x_t) reste dans un couloir stable sur toute la durée (pas de dérive d'énergie, pas de fuite vers le net ou vers le bruit).
2. **Pas de saut, jamais** : la distribution des |Δ| image-à-image (x_t et x̂₀) est serrée — max par frame du même ordre de grandeur que la médiane ; aucune frame ne change brutalement. En comparaison, l'errance a des transitions de phase visibles dans cette même mesure.
3. **Continuité locale + dérive longue** : corr(frame k, k+1) élevée ; corr(frame k, k+300) nettement plus basse — l'image évolue réellement (elle erre), elle ne vibre pas sur place ni ne reste figée.
4. **Amplitude par frame ~ √β(t*)** : les changements par frame croissent avec t* (t* haut = plus turbulent). Vérifiable en comparant deux valeurs de t*.
5. **Graines** : deux runs de même seed sont identiques (reproductibilité) ; deux seeds différents divergent ; le bruit ajouté entre frames consécutives est décorrélé (pas de direction fixe qui s'accumule).
6. **Isotropie** (règle du projet) : les images produites ne doivent privilégier aucun axe — les différences rangée-à-rangée et colonne-à-colonne du bruit ajouté sont du même ordre.
7. **Non-régression des autres régimes** : errance et respiration produisent toujours leurs comportements (errance : cycles avec résolution complète, images des cycles successifs différentes ; respiration : jamais résolue, std(x_t) jamais proche de 0).

## Livrables (sur CETTE branche `blind-test-flux`)
- Une suite de tests **boîte noire** exécutable (scripts Python/shell dans `blind_tests/`, un runner `blind_tests/run.sh` qui sort PASS/FAIL par propriété) — fondée uniquement sur le binaire et ses sorties.
- `BLIND_TEST_FLUX.md` : verdict par propriété (PASS/FAIL/AMBIGU), chaque FAIL ou ambiguïté citant la ligne de spec, et les défauts de spec rencontrés.
- Commits atomiques, ne jamais push. Hygiène : `git checkout Models/` si un run a réécrit des configs ; ne committe ni checkpoint ni dumps.
