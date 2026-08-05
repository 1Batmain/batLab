# Test à l'aveugle du régime « flux » — verdict

Agent de test indépendant. **Aucun fichier de `bat_building/src/` ni `main/src/` n'a été
ouvert**, ni leurs diffs ou leur historique. Tout ce qui suit provient du binaire
(`cargo build --release -p main`), de sa `stdout`, de ses dumps `.f32` et de ses PNG.
La seule chose que j'aie faite de plus qu'exécuter le binaire courant, c'est **builder et
exécuter le binaire d'un commit antérieur** (`12acad8`, parent de `feat(perpetual): le
regime flux`) pour comparer ses sorties — build et exécution, jamais lecture.

Suite exécutable : `blind_tests/run.sh` (option `BLIND_BASELINE=1` pour la comparaison
avec le binaire pré-flux). Sortie : PASS/FAIL par propriété, code de retour non nul si FAIL.

## Résumé

| # | Propriété (spec) | Verdict |
|---|---|---|
| P0 | format du dump / sanité (ligne 12) | **PASS** |
| P1 | stationnarité du niveau de bruit (ligne 15) | **PASS** *(avec une ambiguïté de spec, cf. §A1)* |
| P2 | aucun saut image-à-image (ligne 16) | **PASS** |
| P3 | continuité locale + dérive longue (ligne 17) | **PASS** |
| P4 | amplitude par frame ~ √β(t\*) (lignes 9-10 + 18) | **FAIL** |
| P5 | graines : reproductibilité, divergence, bruit décorrélé (ligne 19) | **PASS** |
| P6 | isotropie (ligne 20) | **PASS** |
| P7 | non-régression errance / respiration (ligne 21) | **PASS** |
| OBS | PNG annoncés vs PNG écrits (hors liste numérotée) | **défaut signalé** |

7/8 propriétés testables PASS, 1 FAIL. Le régime flux, tel qu'il tourne, **fait ce que la
spec décrit** — sauf qu'il n'expose aucun moyen de choisir son t\*, ce qui rend la
propriété 4 non seulement invérifiable mais inexistante côté utilisateur.

Le point de fonctionnement observé est t\* = 64, atteint après une descente de 190 frames
depuis t = 254, puis tenu indéfiniment (vérifié jusqu'à 20 000 frames).

---

## FAIL — P4 : amplitude par frame ~ √β(t\*)

> **Spec, lignes 9-10** : « `--headless-perpetual --regime flux --actions N --dump <chemin> [--t-star K] [--seed S] --checkpoint <ckpt>` »
> **Spec, ligne 18** : « **Amplitude par frame ~ √β(t\*)** : les changements par frame croissent avec t\* (t\* haut = plus turbulent). **Vérifiable en comparant deux valeurs de t\*.** »

**Aucun des leviers annoncés n'existe.** Preuves indépendantes, trois angles :

1. **Comparaison bit-à-bit des dumps.** À graine et actions égales, `--t-star 30`,
   `--t-r 30` et un flag délibérément inventé (`--flag-inexistant-de-controle 30`)
   produisent tous les trois un dump **strictement identique** (sha256) à celui du run
   sans aucun flag. `--t-star` est donc inerte au même titre qu'un flag inconnu — pas
   « sans effet sur le flux », littéralement non parsé.
2. **La bannière du binaire elle-même.** Le binaire réémet sa config parsée ; il affiche
   `t_r=64` quelle que soit la valeur passée à `--t-r` (`--t-r 150`, `--t_r 150`,
   `--tr 150` → tous `t_r=64`), alors qu'il réémet correctement `magnitude=0.3`,
   `frames=3`, `seed=5` quand on les passe. Le flag documenté ligne 12 (« `--t-r` pour la
   profondeur ») n'est parsé par aucune orthographe.
3. **Le t dumpé.** Le dump porte, par frame, le t courant (cf. §format). Le plateau du
   flux vaut **64 dans tous les runs**, y compris avec `--t-star 30`.

Aucune valeur de t\* autre que 64 n'est atteignable → la propriété « les changements par
frame croissent avec t\* » **ne peut pas être testée via le contrat public**, et
l'utilisateur ne peut de toute façon pas régler la turbulence. Je ne l'ai pas contournée :
mesurer cette loi impliquerait d'aller lire l'implémentation, ce qui m'est interdit — et
c'est justement le symptôme d'un défaut de contrat.

**Observation complémentaire, non concluante** : pendant la descente initiale (t : 254 →
64, mécanisme *différent* du churn — il s'agit de pas inverses, pas d'un maintien à t\*),
l'amplitude par frame décroît bien de façon monotone avec t —
`|Δ|` = 0,221 (t=254) → 0,187 (t=179) → 0,150 (t=119) → 0,115 (t=64). La machinerie de
bruit dépend donc de t dans le bon sens. Cela **ne démontre rien** sur la loi
d'amplitude du flux en fonction de t\*, qui reste non observable. Pour mémoire, le churn
du flux à t\*=64 vaut `|Δ|` médian = 0,160, soit ≈ 1,4× le pas inverse simple au même t
— cohérent avec un aller-retour bruitage/débruitage, mais un seul point ne fait pas une loi.

**Ce que je demande** : que `--t-star` soit réellement implémenté (ou que la spec soit
corrigée pour dire que t\* est figé à 64 et que la propriété 4 est hors périmètre). Je
n'ai pas modifié le test pour le faire passer.

---

## Détail des propriétés PASS

Mesures sur `--actions 3000 --seed 7` (le runner), recoupées sur des runs de 6 000 et
20 000 frames. Toutes les statistiques du flux sont calculées **sur le plateau**, c.-à-d.
après la descente initiale de 190 frames.

### P1 — stationnarité (ligne 15) : PASS
- std(x_t) sur le plateau : moyenne **0,7367**, couloir **[−8,9 %, +8,9 %]**.
- Dérive premier ↔ dernier décile : **+1,71 %**.
- Sur 20 000 frames (19 810 de plateau) : moyenne 0,7318, couloir [−9,8 %, +12,8 %],
  dérive décile-à-décile **−0,68 %**. Aucune fuite vers le net ni vers le bruit sur un
  horizon 100× plus long que la durée de décorrélation.
- Le contenu, lui, bouge : std(x̂₀) oscille entre 0,18 et 0,29 et la moyenne de x̂₀ dérive
  de +0,68 à −0,27 au long des 20 000 frames — sans tendance, ce qui est exactement le
  comportement attendu d'une marche stationnaire.

### P2 — aucun saut (ligne 16) : PASS
- flux, x_t : médiane `|Δ|` = 0,16013, **max/médiane = 1,10**, min/médiane = 0,93.
- flux, x̂₀ : médiane `|Δ|` = 0,02834, **max/médiane = 1,58**.
- Contre-épreuve demandée par la spec (« En comparaison, l'errance a des transitions de
  phase visibles dans cette même mesure ») : errance **5,14×** sur x_t et **17,37×** sur
  x̂₀ à budget de frames égal. Le contraste est net et va dans le sens annoncé.
- Sur 20 000 frames, les ratios du flux ne bougent pas (1,098 et 1,662) : il n'y a pas de
  saut rare qui apparaîtrait à long horizon.

### P3 — continuité locale + dérive longue (ligne 17) : PASS
- x_t : corr(k, k+1) = **0,9627**, corr(k, k+300) = **0,0271**.
- x̂₀ : corr(k, k+1) = **0,9891**, corr(k, k+300) = **0,3230**.
- Décroissance complète (x_t) : lag 1 → 0,963 ; 5 → 0,829 ; 30 → 0,345 ; 100 → 0,064 ;
  300 → 0,027. Ni vibration sur place (la corrélation tombe vraiment), ni figement
  (corr(k, k+1) reste très haute) : l'image erre.

### P5 — graines (ligne 19) : PASS
- **Reproductibilité** : deux runs `--seed 7` → dumps de sha256 **identiques**.
- **Divergence** : `--seed 7` vs `--seed 99`, corrélation moyenne image-à-image
  **+0,0012** (max |corr| 0,118, compatible avec le hasard sur 1 024 pixels).
- **Bruit décorrélé** : corr(Δ_k, Δ_{k+1}) = **−0,0199** ; sur *toutes* les paires
  d'incréments du run (≈ 4×10⁶ paires), la corrélation maximale est **+0,162**, exactement
  l'ordre de grandeur d'un maximum d'échantillon gaussien — aucun champ de bruit n'est
  rejoué ni permuté d'une frame à l'autre.
- **Pas de direction fixe accumulée** : la norme de la moyenne des incréments vaut
  **0,100×** celle qu'une marche non biaisée produirait — encore plus petite qu'un biais nul.
- **Marche bornée** : l'écart quadratique cumulé rms sature — L=1 → 0,200 ; L=8 → 0,529 ;
  L=64 → 0,972 ; L=512 → 1,034 ; L=2048 → 1,032, soit exactement √2·std(x_t) = 1,04.
  Diffusion aux temps courts, saturation aux temps longs : ni dérive balistique ni fuite.

### P6 — isotropie (ligne 20) : PASS
Mesuré sur le **champ ajouté** (Δ = x_t[k+1] − x_t[k]), pas sur l'image :
- énergie des différences rangée-à-rangée = 0,08068 ; colonne-à-colonne = 0,08065 ;
  **rapport 1,0003**.
- autocorrélation spatiale du champ ajouté à ±1 et ±2 px : **|ρ| ≤ 0,0013** pour tous les
  décalages testés, **y compris (1,−1)** — le décalage qui aurait explosé si le champ était
  constant sur les anti-diagonales, c.-à-d. le piège documenté dans `ANISOTROPY_HUNT.md`.
  Ce piège est donc bien absent ici.

### P7 — non-régression errance / respiration (ligne 21) : PASS
Deux niveaux de vérification :
- **Comportemental** : errance atteint t = 0 sur 22 cycles (résolution complète), les
  images des cycles successifs sont différentes (corr 0,803 à 0,968, moyenne 0,935) et
  s'éloignent avec la distance de cycle (corr = 0,748 entre cycles i et i+5) ;
  respiration reste dans t ∈ [31, 64], n'atteint jamais 0, et min std(x_t) = 0,4817 —
  jamais résolue, jamais proche de 0.
- **Bit-à-bit** (`blind_tests/baseline.sh`) : le binaire du commit **précédant** l'ajout du
  flux (`12acad8`) et le binaire courant, mêmes arguments et même graine, produisent des
  PNG **strictement identiques** pour les 8 premiers cycles d'errance **et** de
  respiration (8/8 et 8/8). Les mutations du flux n'ont touché à aucune des deux
  dynamiques existantes.

---

## Défauts et ambiguïtés de spec / de contrat

### D1 — `--help` ne produit rien
> **Spec, ligne 12** : « Explore `--help` si des flags manquent à cette description — **le help fait partie du contrat public**. »

`./target/release/main --help` (et `-h`) n'écrit **rien** et sort avec le code 0. Le
contrat public annoncé par la mission n'existe pas : la seule façon de découvrir les flags
est l'essai-erreur. Aggravant : **un flag inconnu est silencieusement ignoré** (aucun
avertissement, code de retour 0) — une faute de frappe est donc indiscernable d'un flag
non implémenté. C'est précisément ce qui rend D3 invisible à l'usage.

### D2 — l'invocation de la spec est incomplète : il manque le modèle
> **Spec, ligne 9** : « `--headless-perpetual --regime flux --actions N …` »

La forme documentée échoue :
`headless perpetual failed: failed to load .../Models/--regime/config_file`. Le flag prend
un **argument positionnel obligatoire** (le nom du modèle), comme `--headless-train` :
`--headless-perpetual Greyscale_Diffusion_L --regime flux …`. À corriger dans la spec.

### D3 — `--t-star` inexistant, `--t-r` non parsé
Voir le FAIL P4. Détail supplémentaire : `--t-r` n'est parsé pour **aucun** régime
(errance et respiration produisent des sorties identiques avec `--t-r 20`, `64`, `150`),
alors que la ligne 12 le documente comme le réglage de profondeur des deux autres régimes.
Le défaut est donc plus large que le flux ; il est simplement resté invisible parce que
rien n'échoue quand on passe un flag inconnu (D1).

### D4 — le format du dump n'est pas celui décrit
> **Spec, ligne 12** : « Le dump est du f32 brut : par frame, deux images 32×32 (x_t puis x̂₀), frames concaténées. »

Ce n'est pas du f32 brut. Le fichier commence par un en-tête de 20 octets
(magic ASCII `BATFLUX1`, puis trois `u32` : largeur 32, hauteur 32, canaux 1) et **chaque
frame est préfixée de 5 octets** (un octet à 0, jamais observé différent, puis un `u32`
little-endian) avant ses 2 × 1 024 f32. Un lecteur écrit d'après la spec lit donc du
décalé. J'ai dû retrouver la structure en faisant varier `--actions` et en résolvant
`taille = en-tête + N × (préfixe + 2·1024·4)`.

Ce n'est pas seulement un défaut : ce `u32` **est le t courant** (254, 253, … puis palier),
et c'est lui qui m'a permis de constater directement que le plateau du flux vaut toujours
64. Il mérite d'être documenté plutôt que supprimé. Mon lecteur est `blind_tests/dumpio.py`.

### A1 — ambiguïté : la descente initiale n'est pas couverte par la spec
> **Spec, ligne 15** : « std(x_t) reste dans un couloir stable **sur toute la durée** »

Tout run de flux commence par **190 frames de descente** (t = 254 → 64) pendant lesquelles
std(x_t) passe de 1,03 à 0,74, soit −28 %. Pris au pied de la lettre, « sur toute la
durée » est violé sur les 190 premières frames ; pris comme une description du régime
permanent, il ne l'est pas. J'ai tranché en faveur de la seconde lecture — partir du bruit
pur et rejoindre t\* est une mise en route inévitable, et la spec ne dit nulle part que le
flux doit démarrer directement à t\*. **Je le signale plutôt que de l'arbitrer en silence** :
si le flux doit pouvoir démarrer à t\* (par exemple pour être enchaîné après un autre
régime), c'est une exigence à écrire, et elle n'est pas satisfaite aujourd'hui.
Toutes mes mesures de flux excluent explicitement ces 190 frames.

### A2 — ambiguïté : « images des cycles successifs différentes » n'a pas de seuil
> **Spec, ligne 21** : « errance : cycles avec résolution complète, **images des cycles successifs différentes** »

Les images successives de l'errance sont différentes mais **fortement corrélées**
(corr 0,80 à 0,97). J'ai retenu le seuil « corr ≤ 0,99 » (non identiques) et vérifié en
plus que la corrélation décroît avec la distance de cycle (0,748 à i+5). Si « différentes »
voulait dire « décorrélées », la propriété échouerait — mais ce serait contradictoire avec
le mot *errance*. À arbitrer par l'auteur de la spec.

### D5 — le flux annonce un dossier de frames qu'il ne crée jamais
Le binaire affiche `frames → …/perpetual_samples/flux` puis n'écrit **aucun PNG** et ne
crée même pas le dossier, alors qu'errance (10 PNG) et respiration (20 PNG) en écrivent
dans les mêmes conditions. `--frames N` est donc inerte pour le flux — cohérent avec le
fait qu'un churn stationnaire ne boucle jamais de cycle, mais alors la ligne d'annonce
ment. Hors des 7 propriétés numérotées : signalé, sans faire échouer la suite.

### D6 — changement de contrat sur `--actions` / `--frames` pour errance et respiration
Constaté en comparant le binaire pré-flux et le binaire courant :
- **Avant** : `--actions` était **ignoré** ; le run s'arrêtait après `--frames` cycles
  (`--actions 200`, `300`, `1500` → tous 711 pas et 8 cycles).
- **Maintenant** : `--actions` borne le run et `--frames` ne l'arrête plus
  (`--actions 5000 --frames 8` → 37 PNG en errance, 73 en respiration).

Les **dynamiques** sont intactes (les PNG des 8 premiers cycles sont identiques au bit
près, cf. P7) : c'est un changement de sémantique CLI, pas une régression de comportement.
Il est probablement voulu — le flux n'a pas de cycle et avait besoin d'une borne — mais il
casse la signification d'anciennes lignes de commande sur errance et respiration, et il
n'est écrit nulle part. À documenter.

Note connexe : pour errance et respiration, le compteur « N reverse steps » de la bannière
compte **moins** que `--actions` (errance `--actions 300` → « 256 reverse steps ») alors
que le dump contient bien 300 frames. La bannière ne compte apparemment que les pas
descendants. Ce n'est pas faux, c'est juste ambigu à la lecture.

---

## Reproduire

```bash
./blind_tests/run.sh                     # 8 propriétés, ~40 s
BLIND_BASELINE=1 ./blind_tests/run.sh    # + comparaison bit-à-bit avec le binaire pré-flux
ACTIONS=20000 ./blind_tests/run.sh       # horizon long (stationnarité)
```

Le runner copie le checkpoint `night_run.ckpt` s'il est absent, écrit ses dumps dans
`blind_tests/out/` (ignoré par git), et rend un code de retour non nul si une propriété
est FAIL. `blind_tests/baseline.sh` crée au besoin le worktree
`worktrees/blind-baseline-preflux` sur `12acad8` et le construit ; il peut être supprimé
après coup (`git worktree remove worktrees/blind-baseline-preflux`).

### Format du dump, tel que reconstitué
```
offset 0   : magic "BATFLUX1"                (8 octets)
offset 8   : u32 width, u32 height, u32 channels   (32, 32, 1)
offset 20  : par frame, répété N fois :
               u8   (toujours 0 observé)
               u32  little-endian = t courant de la frame
               f32 × width·height·channels  → x_t
               f32 × width·height·channels  → x̂₀
```
N vaut exactement `--actions`. x̂₀ est borné dans [−1, 1].
