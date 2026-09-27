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

![Dataset réel, ancien run, run de nuit — à graines égales](docs/gallery/elephants/night_finetune_comparison.png)

*Les trois rangées montrent, aux mêmes graines : de vraies photos du dataset, les échantillons
de l'ancienne lignée (deux fine-tunes trop agressifs, sortis de l'optimum CIFAR), et ceux du
run d'une nuit — 160 000 pas, Adam 1e-4, EMA 0.999, reparti de la fondation CIFAR. Les
échantillons du run de nuit sont à l'échelle in-distribution du dataset : la corrélation avec
la plus proche image réelle (0,584) est statistiquement indistinguable de celle de deux
vraies photos entre elles (0,596 ± 0,12).*
