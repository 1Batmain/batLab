# Mission : test à l'aveugle de la couche d'attention

Tu es un agent de test INDÉPENDANT. Règle absolue : **interdiction de lire l'implémentation** — aucun fichier de `crates/*/src/` touchant l'attention (shaders `.wgsl` d'attention, `layer_types/attention.rs`, `attention_tests.rs`), ni leurs diffs/historique. Tu peux : compiler (`cargo build --release -p batlab`), exécuter le binaire, écrire ta PROPRE référence (NumPy/Python depuis la formule mathématique ci-dessous), lire les sorties. Tu peux lire : ce fichier, le CLAUDE.md racine, `docs/reports/ATTENTION.md` et `GO_NOGO.md` (la spec/contrat). Un écart se signale en citant la spec ; tu n'édites jamais un test pour le faire passer.

## La spec (contrat mathématique et observable)

La couche `Attention` calcule, sur les H·W positions spatiales d'un tenseur C canaux :
`y = x + W_o · softmax(QᵀK/√d) · V`, où Q,K,V,W_o sont 4 projections C×C (1×1). Propriétés à tester :

1. **Identité au départ** : `W_o` est initialisé à zéro → une couche d'attention fraîchement construite est l'IDENTITÉ (sortie == entrée). Observable : un modèle contenant une attention non entraînée doit produire exactement les mêmes échantillons que le même modèle sans la couche d'attention (à poids Q/K/V identiques). Construis deux configs (une avec `GroupNorm+Attention` au goulot, une sans), poids frais même seed, compare la génération.
2. **La couche fait quelque chose une fois entraînée** : le checkpoint `Models/Color_Diffusion_XL/pretrained_weights/night_run.ckpt` (30 couches, avec attention) génère des images NON dégénérées (bornées [-1,1], non constantes, diversité inter-seed > 0,2 à magnitude 1,0 — cf. tools/sample_diversity.py). Si absent, copie-le depuis `.checkpoint_backup/Color_Diffusion_XL_attn_5700.ckpt` du dépôt principal.
3. **Round-trip checkpoint observable** : `--headless-sample` sur ce checkpoint, deux fois même seed → PNG **octet pour octet identiques** (déterminisme) ; et si tu peux charger/resauver via une commande, le second sample reste identique.
4. **Référence f64 indépendante (le cœur)** : écris en NumPy la formule d'attention complète depuis zéro (softmax stabilisé par max, les 4 projections). Le binaire n'expose pas la couche isolée — mais tu peux la sonder indirectement : construis un modèle MINIMAL (une seule attention, ou attention + conv identité) via une config que tu écris, entraîne 0 pas, et compare une génération 1-pas à ta référence si le chemin le permet. Si le harnais ne le permet pas proprement de l'extérieur, DIS-LE (défaut d'observabilité de la spec) plutôt que de tricher — c'est une conclusion valide.
5. **Anti-fuite inter-échantillons** (le risque spécifique de l'attention) : si tu peux exercer un batch > 1 par un chemin observable, vérifie qu'un échantillon n'influence pas la sortie d'un autre. Sinon, consigne que ce n'est pas observable de l'extérieur.

## Livrables (sur CETTE branche `blind-attention`)
- Suite/scripts `blind_tests_attention/` (ce qui est réellement boîte noire) + runner PASS/FAIL.
- `BLIND_TEST_ATTENTION.md` : verdict par propriété (PASS/FAIL/NON-OBSERVABLE), chaque limite d'observabilité consignée honnêtement, défauts de spec cités.
- Commits atomiques, ne jamais push. `git checkout Models/` si pollué, ne committe ni checkpoint ni dumps. Le GPU est libre.
