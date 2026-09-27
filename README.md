# batLab

Un projet d'apprentissage : **comprendre, en profondeur, comment fonctionne un réseau
U-Net de diffusion** — en l'écrivant from scratch, et en progressant en Rust au passage.

Concrètement : un framework de deep learning écrit **from scratch en Rust**, dont tout le
calcul passe par des **compute shaders WGSL** exécutés sur wgpu. Pas de PyTorch, pas de
LibTorch, pas de crate d'autodiff : les convolutions, la GroupNorm, l'attention,
l'upsample, la loss, SGD et Adam sont des shaders maison, forward *et* backward, et le
graphe est câblé à la main.

Le but n'est pas de rivaliser avec les frameworks existants. C'est que rien ne soit une
boîte noire : chaque rouage a été construit, mesuré, et parfois cassé puis réparé.

---

## Le résultat du moment

Un U-Net de diffusion de **1,19 M de paramètres** (30× plus petit que le DDPM du papier
de Ho et al., 35,7 M), entraîné sur **1300 photos d'éléphants** réduites à 32×32.

![Vraies photos du dataset, et photos générées par le modèle](docs/gallery/elephants/vraies_vs_generees.png)

*En haut, de vraies photos du dataset. En bas, des images générées par le modèle à partir
de pur bruit, sur des graines qu'il n'a jamais vues. La corrélation de chaque image générée
avec sa plus proche vraie photo (0,584) est statistiquement indistinguable de celle de deux
vraies photos entre elles (0,596 ± 0,12) : le modèle ne copie pas, il génère dans la famille.*

## Comment fonctionne le modèle, simplement

Le modèle reçoit une image bruitée et une seule information : « à quel point est-elle
bruitée ? » — quatre canaux d'entrée en plus du rouge-vert-bleu portent ce chiffre. Sa
toute seule tâche : deviner le bruit qui a été versé dedans. On le retire, on recommence
— 256 fois, de plus en plus finement. L'image émerge du bruit, sans qu'aucune ligne du
programme ne sache ce qu'est un éléphant.

Le réseau qui fait ce travail est un U-Net — il descend, puis remonte :

```text
 image bruitée (32×32) + « à quel point c'est bruité »
             │
 ┌───────────▼───────────┐              ┌────────────┐
 │ 32×32 · 48 canaux     │              │ 32×32      │──▶ le bruit prédit
 │ les pixels            │─ passerelle ▶│ remonte    │
 └───────────┬───────────┘              └──────▲─────┘
             │                                 │
 ┌───────────▼───────────┐              ┌──────┴─────┐
 │ 16×16 · 96 canaux     │              │ 16×16      │
 │ les formes            │─ passerelle ▶│ remonte    │
 └───────────┬───────────┘              └──────▲─────┘
             │                                 │
 ┌───────────▼─────────────────────────────────┴─────┐
 │ 8×8 · 192 canaux — ATTENTION                      │
 │ la scène entière, vue d'un seul coup              │
 └───────────────────────────────────────────────────┘
```

En descendant, l'image se comprime : à 32×32 le réseau voit des pixels, à 16×16 des
formes, à 8×8 la scène entière. Au point le plus bas, l'attention lui donne un regard
d'ensemble — c'est là que la composition se décide. En remontant il reconstruit, et les
passerelles ramènent à chaque niveau les détails fins que la compression avait laissés
derrière elle.

Le tout : 30 couches, convolutions 3×3, GroupNorm, SiLU, attention à 8×8 — 1 191 795
paramètres, entraînés en une nuit (160 000 pas, Adam) sur 1300 photos, depuis une
fondation CIFAR-10.
