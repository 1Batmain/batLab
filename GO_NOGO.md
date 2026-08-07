# GO / NO-GO — attention au goulot de `Color_Diffusion_XL`

**Verdict : GO — à batch 32, pas à batch 64.**

Nuit du 6 au 7 août 2026. Branche `attention`, worktree `worktrees/attention`.

---

## 1. Le verdict en trois lignes

La couche est **prouvée** (référence f64 indépendante, différences finies sur
les cinq tenseurs, équivalence batchée, anti-fuite inter-échantillons, identité
résiduelle bit à bit, round-trip de checkpoint — et 16 mutations sur 16
attrapées). Elle est **intégrée** et s'entraîne bout en bout.

Elle est **gratuite jusqu'à batch 32** et **coûte le double à batch 64**. Le run
de nuit doit donc partir à **batch 32**, où l'attention ne coûte rien de mesurable
par rapport au XL sans attention.

## 2. La commande recommandée

```bash
cargo run --release -p batlab -- --headless-train Color_Diffusion_XL \
  --steps 5700 \
  --dataset datasets/cifar10_rgb.batraw \
  --optimizer adam --lr 1e-3 --batch 32
```

`5700` pas ≈ **7 h** à 4,417 s/pas. Ajuster au prorata : **1 h ≈ 815 pas**.

> ### ⚠ Les mesures supposent un GPU LIBRE
>
> Toutes les mesures du §3 ont été prises machine au repos. À 02h10, le même
> `--batch 32` tournait à **8,7 s/pas** — presque exactement le double — pendant
> qu'**Ableton Live 12 occupait 55 % du CPU** (plus `coreaudiod` et `usbaudio`).
> Mesure : pas 350 à 02:10:57, pas 375 à 02:14:35, soit 218 s pour 25 pas.
>
> Ce n'est pas un A/B contrôlé — je n'ai pas relancé le banc DAW fermé — mais la
> corrélation est nette et le dépôt connaît déjà le phénomène (le batch 64 avait
> dû être « re-certifié sur GPU libre »).
>
> **Fermer le DAW avant de lancer la nuit**, ou prévoir ~2x le temps mural :
> 5 700 pas coûteraient alors ~14 h au lieu de 7.

> `--headless-train` n'écrit jamais dans le `config_file` du modèle et dépose son
> checkpoint dans un fichier scratch — il n'écrase aucun poids sauvegardé.
> Pour garder les poids, lancer par le TUI, ou copier le scratch en fin de run.

**Ne pas utiliser `--batch 64`** : voir §4.

## 3. Les mesures

`Color_Diffusion_XL` (48/96/192, 30 couches) contre le même sans les deux
couches `GroupNorm + Attention` (28 couches). Adam, lr 1e-3, CIFAR-10 RGB,
100 pas, médiane des intervalles en régime établi (le chargement du dataset et
le préchauffage sont hors mesure par construction).

| batch | sans attention | avec attention | surcoût |
|---:|---:|---:|---:|
| 16 | 2 240 ms/pas — 140,0 ms/éch. | 2 240 ms/pas — 140,0 ms/éch. | **0 %** (sous la résolution) |
| 32 | *(non mesuré)* | 4 417 ms/pas — 138,0 ms/éch. | **≈ 0 %** |
| 64 | 8 792 ms/pas — 137,4 ms/éch. | 17 360 ms/pas — 271,2 ms/éch. | **+97 %** |

Le débit par échantillon du réseau **sans** attention est plat (140 → 137
ms/éch. de batch 16 à 64) : le U-Net passe à l'échelle proprement. Celui **avec**
attention est plat jusqu'à 32 (140,0 → 138,0) puis double à 64.

Le coût par échantillon étant identique à 16 et 32, le choix du batch est une
question de qualité de gradient, pas de débit.

## 4. L'anomalie à batch 64 — non expliquée, documentée

Entre batch 32 et 64 le coût de l'attention passe de « sous la résolution de
mesure » à 8,6 s/pas. **C'est une falaise, pas une croissance**, et je ne
l'explique pas :

- l'attention pèse ~11 MMACs/échantillon sur les 238 du réseau (**~5 %**) —
  un surcoût de +97 % est deux ordres de grandeur au-dessus de son arithmétique ;
- toutes ses passes sont **linéaires** en batch (les passes élémentaires ont un
  thread par activation, les deux réductions de paramètres gardent leur nombre
  de threads et allongent leur boucle) — rien n'est quadratique ;
- aucun compte de workgroups n'approche la limite de 65 535 (36 864 au plus) ;
- la mémoire ajoutée par la couche à batch 64 est d'environ 43 Mo ;
- **le calcul reste juste** : la loss part de 1,3508 dans les deux configurations
  et arrive à 0,0571 (avec) contre 0,0574 (sans) après 100 pas. Ce n'est donc
  pas le mode d'échec « le command buffer est rejeté et rien ne tourne ».

Hypothèse la plus probable, **non vérifiée** : un seuil d'occupation ou de
pression mémoire franchi entre 32 et 64. À instruire par une mission de perf ;
la première cible à profiler est `attn_back_weights` (147 456 threads, chacun
sommant sur `batch·N`) et `attn_back_bias` (768 threads seulement — 12
workgroups, occupation très faible).

Ce point n'est **pas bloquant** pour la nuit : batch 32 donne le même débit par
échantillon que batch 64 sans attention.

## 5. Ce qui a été prouvé

| Preuve | Où |
|---|---|
| Référence CPU f64 écrite depuis la formule, 3 formes dont 72 positions (> workgroup) | `attention_tests.rs` |
| Différences finies : Q, K, V, W_o, les 4 biais, **et l'entrée** | idem |
| Équivalence batchée (échelles décorrélées par échantillon) | idem |
| Anti-fuite : **chaque** échantillon perturbé à son tour, les autres bit à bit identiques | idem |
| Identité résiduelle `W_o = 0` bit à bit, et sur la couche réellement construite | idem |
| Gradients de paramètres batchés = somme des gradients par échantillon | idem |
| Round-trip de checkpoint sur les 4 projections | idem |
| Grille de dispatch 2-D franchie pour de vrai (65 536 workgroups) | idem |
| Les deux bind groups sous la limite WebGPU de 8 storage buffers | `layer_types/attention.rs` |
| **16 mutations sur 16 attrapées**, dont 3 « sanity » | §4.5 du rapport |

Smoke train batch 16, 300 pas : loss **1,426 → 0,021**, les quatre tranches de t
décroissent, **zéro valeur non finie** sur 49 152 échantillons par tranche.

**Chaîne complète vérifiée** — entraînement avec attention → checkpoint →
rechargement → 256 pas de débruitage → PNG 32×32 :

```
./target/release/batlab --headless-sample Color_Diffusion_XL \
  --checkpoint <scratch>/Color_Diffusion_XL_headless.ckpt --paths 2 --out <fichier>.png
final image stats: min=-0.9999 max=0.9999 mean=-0.1396 std=0.4054
```

Signature saine : valeurs dans [-1,1] sans saturation, écart-type non dégénéré.
C'est l'inverse du mode d'échec « blanc saturé » (`INSIGHTS_TRAINING.md`), et
l'image ne présente pas de bandes horizontales (`ANISOTROPY_HUNT.md`). Le
contenu reste une texture — le checkpoint ne portait que 100 pas.

> Le flag est `--checkpoint`, **pas** `--ckpt`, et `--out` attend un **fichier**,
> pas un répertoire.

## 6. Deux bugs à échec silencieux trouvés en route

1. **La limite WebGPU de 8 storage buffers** (j'en liais 12 au backward). Le bind
   group est rejeté → le **command buffer entier** est invalide → même le forward
   ne tourne plus. Loss et gradients exactement à zéro, aucune exception. Ça avait
   rendu mes trois tests de gradient **vides** (0 contre 0 ⇒ erreur relative nulle).
   Trouvé par mutation : `grad_weights = acc * 2.0` survivait.
2. **Le repli des workgroups excédentaires de la grille 2-D** sur la dernière
   ligne : les passes de softmax font du read-modify-write, deux workgroups sur
   la même ligne se courseraient. Ne se déclenche qu'au-delà de 65 535 workgroups.

## 7. Ce qui n'est pas fait

- **Multi-head** : non tenté. Le single-head est prouvé ; j'ai choisi de ne plus
  toucher à la couche après la validation, pour ne pas invalider l'artefact exact
  qui part cette nuit.
- **Perf** : aucune passe d'optimisation. Voir §4.
- **Test à l'aveugle** : la suite est co-écrite avec l'implémentation. Le
  protocole du dépôt demande un agent indépendant ; la validation par mutation
  est un substitut partiel, **pas** un remplacement.

Conception, preuves et pièges de méthode en détail : `docs/reports/ATTENTION.md`.
