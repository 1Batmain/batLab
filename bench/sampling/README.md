# bench/sampling — le balayage d'échantillonnage

Le harnais de la campagne `docs/reports/SAMPLING_SWEEP.md` : trouver les réglages
de génération les plus nets pour `Elephants_XL`, en ancrant le jugement sur les
statistiques des **vraies** images du dataset (pas sur une préférence).

Le critère de netteté vit dans `tools/sharpness_stats.py` (même vecteur de
mesures sur un lot de PNG ou sur un `.batraw`). Ici, les trois pilotes du
balayage :

- `sweep_night.sh <dir>` — génère magnitude × variance × poids (EMA/brut) sur les
  6 graines de référence (101…606), 1 trajectoire, via `--headless-sample`.
- `analyze_night.py <dir>` — agrège chaque config sur ses 6 graines, classe par
  distance au vecteur cible du dataset, et lance le **contrôle de non-vacuité**
  (bruit blanc et aplat gris doivent mal scorer — sinon la métrique est truquable).
- `planche.py <out.png> <scale> "TEMPLATE::Label"…` — planches comparatives, une
  rangée par config, les 6 graines en colonnes (upscalées, le natif est 32×32).

**Chemins machine-spécifiques** (poids et dataset ne sont pas dans le dépôt) :
les scripts pointent en dur `eleph_night/night.ckpt` (== `web/dist/weights.ckpt`,
le `BBCKPT3` déployé) et `datasets/elephants256.batraw`, sous
`BATLAB_ROOT=/Users/bat/development/lab/batLab`. Les adapter pour rejouer ailleurs.

Sortie retenue : planches et `results.json` sous `docs/gallery/sampling_sweep/`.
