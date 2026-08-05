#!/usr/bin/env python3
"""Génère les figures de SYNTHESE_CAMPAGNE.md.

Toutes les valeurs tracées viennent soit du JSONL d'un run réel, soit d'un
tableau d'un rapport de mission — la constante porte alors sa source en
commentaire. Aucun chiffre n'est inventé ni arrondi à la main.

Dépendances : matplotlib, numpy, pillow. Exécution recommandée (rien à
installer sur la machine) :

    uv run --with matplotlib --with numpy --with pillow \
        python synthese_assets/make_figures.py

Les fichiers non versionnés (métriques JSONL, dataset .batraw) sont lus dans le
dépôt principal, deux niveaux au-dessus du worktree.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
REPO = WORKTREE.parent.parent  # worktrees/<x>/ -> dépôt principal
OUT = HERE

METRICS = REPO / "Models/Greyscale_Diffusion/pretrained_weights/fable_run_metrics.jsonl"
DATASET = REPO / "datasets/cifar10_grey.batraw"

# ---------------------------------------------------------------- palette ---
# Palette de référence validée (skill dataviz, `references/palette.md`),
# mode clair. Ordre des slots catégoriels respecté.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN = "#eda100", "#e87ba4", "#008300"
VIOLET, RED = "#4a3aa7", "#e34948"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BUCKETS = ["t 0–64", "t 64–128", "t 128–192", "t 192–256"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "figure.dpi": 160,
    })


def fr(value: float, decimals: int = 2) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{value:.{decimals}f}".replace(".", ",")


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"écrit  {path.relative_to(WORKTREE)}")


# ------------------------------------------------------------- chargement ---
def load_metrics():
    """train_loss, train_probe et trajectoires de débruitage du run 10 000 pas."""
    steps, losses = [], []
    probe_steps, probe_loss, probe_std = [], [], []
    traj: dict[int, list] = {}
    with METRICS.open() as fh:
        for line in fh:
            o = json.loads(line)
            kind = o["kind"]
            if kind == "train_loss":
                steps.append(o["step"])
                losses.append(o["loss"])
            elif kind == "train_probe":
                probe_steps.append(o["step"])
                probe_loss.append([b["loss"] for b in o["buckets"]])
                probe_std.append([b["eps_hat"]["std"] for b in o["buckets"]])
            elif kind == "denoise_step":
                traj.setdefault(o["train_step"], []).append(
                    (o["step_index"], o["latent_out"]["std"], o["eps_hat"]["std"])
                )
    return {
        "steps": np.array(steps),
        "loss": np.array(losses),
        "probe_steps": np.array(probe_steps),
        "probe_loss": np.array(probe_loss),
        "probe_std": np.array(probe_std),
        "traj": {k: np.array(sorted(v)) for k, v in traj.items()},
    }


def load_dataset_images(n: int, size: int = 32):
    """n premières images du .batraw, ramenées en [0,1] (cf. tools/sample_diversity.py)."""
    with DATASET.open("rb") as fh:
        header = fh.read(24)
        magic = header[:8].rstrip(b"\x00").decode("ascii", "replace")
        _count, width, height, channels = struct.unpack("<IIII", header[8:24])
        raw = np.frombuffer(fh.read(width * height * channels * n * 4), dtype="<f4")
    imgs = raw.reshape(n, height, width, channels)[..., 0].astype(np.float64)
    if magic == "BATRAW1":
        imgs = imgs * 2.0 - 1.0
    return (imgs + 1.0) / 2.0


def schedule(num_steps: int = 256):
    """Schedule de production : bêtas DDPM rééchelonnés par 1000/T (FIX_TRAINING #3)."""
    scale = 1000.0 / num_steps
    betas = np.linspace(1e-4 * scale, 0.02 * scale, num_steps)
    betas = np.minimum(betas, 0.999)
    return np.cumprod(1.0 - betas)


# =============================================================== figure 1 ===
def fig_timeline():
    """Carte de la campagne : 3 actes, 8 missions, un résultat par mission."""
    acts = [
        ("Acte I — réparer", BLUE, [
            ("Audit du pipeline", "7 findings, dont le padding Same cassé"),
            ("Correction du pipeline", "4 fixes, gradient 81 % faux → < 5 %"),
            ("Crash du visualiseur", "event loop winit sur le thread principal"),
            ("Diagnostic du blanc saturé", "le réseau ne recevait jamais t"),
        ]),
        ("Acte II — accélérer", ORANGE, [
            ("GroupNorm", "10,6× en isolé, 1,63× sur le pas"),
            ("Convolution", "3,7× en isolé, rien sur le pas → goulot trouvé"),
            ("Optimiseur Adam", "facteur ~20 en nombre de pas vs SGD"),
        ]),
        ("Acte III — comprendre", AQUA, [
            ("Agrandir le U-Net", "capacité réfutée ; la cause est la loss"),
        ]),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = 0.0
    rows = []
    for act_name, color, missions in acts:
        rows.append(("act", act_name, color, None))
        for name, result in missions:
            rows.append(("mission", name, color, result))
    total = len(rows)

    for i, (kind, name, color, result) in enumerate(rows):
        y = total - i
        if kind == "act":
            ax.text(0.0, y, name, color=color, fontsize=10.5, fontweight="bold",
                    va="center", ha="left")
        else:
            ax.plot([0.30], [y], "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
            ax.text(0.36, y, name, color=INK, fontsize=9.5, va="center", ha="left")
            ax.text(1.62, y, result, color=INK2, fontsize=9, va="center", ha="left")

    # colonne vertébrale par acte
    y_cursor = total
    for act_name, color, missions in acts:
        top = y_cursor - 1
        bottom = top - len(missions) + 1
        ax.plot([0.30, 0.30], [bottom, top], color=color, linewidth=1.4,
                alpha=0.35, zorder=1)
        y_cursor -= len(missions) + 1

    ax.set_xlim(-0.05, 4.15)
    ax.set_ylim(0.45, total + 0.55)
    ax.axis("off")
    ax.set_title("La campagne en un coup d'œil — 8 missions, 3 actes",
                 color=INK, fontsize=11.5, pad=14)
    save(fig, "01_carte_campagne.png")


# =============================================================== figure 2 ===
def fig_padding():
    """Finding #1 : la convolution Same décalée et clampée (AUDIT_TRAINING §1)."""
    expected = np.arange(1, 17, dtype=float).reshape(4, 4)
    # sortie GPU mesurée avant fix (AUDIT_TRAINING.md, preuve (a))
    obtained = np.array([6, 7, 8, 9, 10, 11, 12, 13,
                         14, 15, 16, 16, 16, 16, 16, 16], dtype=float).reshape(4, 4)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, data, title, sub in (
        (axes[0], expected, "Attendu — l'identité",
         "noyau 3×3 dont seul le tap central vaut 1"),
        (axes[1], obtained, "Obtenu — décalé de +5, puis saturé",
         "fenêtre non recentrée, lectures hors bornes clampées"),
    ):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=22)
        for (r, c), v in np.ndenumerate(data):
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=11,
                    color=INK if v < 12 else SURFACE,
                    fontweight="bold" if data is obtained and v == 16 else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        despine(ax, keep=())
        ax.set_title(title, color=INK)
        ax.text(0.0, -0.10, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5)

    fig.suptitle("Le padding fantôme : une convolution « Same » qui ne l'était pas",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.04)
    save(fig, "02_padding_fantome.png")


# =============================================================== figure 3 ===
def fig_loss(m):
    """Loss batch vs loss par tranche de t, run de référence 10 000 pas."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)

    ax = axes[0]
    ax.plot(m["steps"], m["loss"], color=MUTED, linewidth=0.8, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("pas d'entraînement")
    ax.set_ylabel("MSE sur ε (échelle log)")
    ax.set_title("Ce qu'on voyait : la loss du batch")
    despine(ax)
    ax.annotate(fr(m["loss"][-1], 4), xy=(m["steps"][-1], m["loss"][-1]),
                xytext=(-6, -15), textcoords="offset points",
                color=INK2, fontsize=8.5, ha="right")
    ax.text(0.02, 0.06, "un seul couple (image, t) par point\n→ deux ordres de grandeur de bruit",
            transform=ax.transAxes, color=MUTED, fontsize=8.2)

    ax = axes[1]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for i, (label, color) in enumerate(zip(BUCKETS, colors)):
        ax.plot(m["probe_steps"], m["probe_loss"][:, i], color=color, linewidth=1.6)
        ax.annotate(f"{label} · {fr(m['probe_loss'][-1, i], 3)}",
                    xy=(m["probe_steps"][-1], m["probe_loss"][-1, i]),
                    xytext=(8, -3), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("pas d'entraînement")
    ax.set_title("Ce qu'il faut voir : la loss par tranche de t")
    ax.set_xlim(-400, m["probe_steps"][-1] * 1.62)
    ax.set_xticks([0, 2500, 5000, 7500, 10000])
    despine(ax)
    axes[0].set_xticks([0, 2500, 5000, 7500, 10000])

    fig.suptitle("Une loss qui descend ne prouve rien — il faut la ventiler par niveau de bruit",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.02)
    save(fig, "03_loss_run_reference.png")


# =============================================================== figure 4 ===
def fig_explosion(m):
    """Explosion du latent le long de la chaîne inverse (INSIGHTS §2.2/§2.3 + JSONL)."""
    # INSIGHTS_TRAINING §2.2 — modèle sans conditionnement temporel, 600 pas
    broken_idx = np.array([0, 64, 128, 192, 255])
    broken_std = np.array([0.985, 4.378, 19.764, 49.622, 67.653])
    # INSIGHTS_TRAINING §2.3 — même chaîne alimentée par le vrai bruit résiduel
    oracle_idx = np.array([0, 64, 128, 192, 255])
    oracle_std = np.array([0.981, 0.110, 0.160, 0.414, 0.572])

    healthy = m["traj"][max(m["traj"])]  # dernière trajectoire du run 10 000 pas

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(broken_idx, broken_std, color=CRITICAL, linewidth=2.0, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="sans conditionnement temporel (600 pas)")
    ax.plot(healthy[:, 0], healthy[:, 1], color=BLUE, linewidth=1.8,
            label="après conditionnement temporel (10 000 pas)")
    ax.plot(oracle_idx, oracle_std, color=INK2, linewidth=1.6, linestyle=":",
            marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0,
            label="oracle : la chaîne alimentée par le vrai bruit")

    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.text(368, 1.12, "écart-type attendu ≈ 1", color=MUTED, fontsize=8.2,
            va="bottom", ha="right")

    ax.set_yscale("log")
    ax.set_xlabel("étape de la chaîne de débruitage  (t = 255 → 0)")
    ax.set_ylabel("écart-type du latent (échelle log)")
    ax.set_xlim(-4, 372)
    ax.set_ylim(0.07, 200)
    ax.legend(loc="lower right")
    despine(ax)

    ax.annotate("×67,7 → image clampée,\nblanc saturé et bandes sombres",
                xy=(255, 67.653), xytext=(262, 90),
                color=CRITICAL, fontsize=8.5, va="center",
                arrowprops=dict(arrowstyle="-", color=CRITICAL, linewidth=0.9))
    ax.annotate(f"×{fr(healthy[-1, 1], 1)}", xy=(255, healthy[-1, 1]),
                xytext=(10, -2), textcoords="offset points",
                color=BLUE, fontsize=8.5, fontweight="bold")
    ax.annotate("borné : ×0,6", xy=(255, 0.572), xytext=(10, -2),
                textcoords="offset points", color=INK2, fontsize=8.5)

    ax.set_title("Un bruit prédit trop faible se fait amplifier 174× le long de la chaîne",
                 color=INK, fontsize=11.5, pad=12)
    save(fig, "04_explosion_latent.png")


# =============================================================== figure 5 ===
def fig_perf():
    """Speedups mesurés : couche isolée vs pas d'entraînement complet."""
    # PERF_GROUP_NORM §4.1/§4.2 ; PERF_CONVOLUTION §5.2/§5.3
    labels = ["GroupNorm", "Convolution"]
    isolated = [10.6, 3.73]
    endtoend = [1.63, 0.99]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    ax = axes[0]
    bars = ax.bar(labels, isolated, width=0.32, color=BLUE)
    for b, v in zip(bars, isolated):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{fr(v, 1)}×".replace(",0×", "×"),
                ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.65, 1.6)
    ax.set_ylim(0, 12.6)
    ax.set_ylabel("accélération de la couche seule")
    ax.set_title("Sur le kernel : gains massifs")
    despine(ax)

    ax = axes[1]
    colors = [BLUE, ORANGE]
    bars = ax.bar(labels, endtoend, width=0.32, color=colors)
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.text(-0.62, 1.04, "1× = aucun gain", color=MUTED, fontsize=8, va="bottom")
    for b, v, c in zip(bars, endtoend, colors):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{fr(v, 2)}×",
                ha="center", color=c, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.65, 1.6)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("accélération du pas d'entraînement")
    ax.set_title("Sur le pas complet : le gain s'évapore")
    despine(ax)
    ax.annotate("18 soumissions GPU par pas,\ndes kernels trop petits :\nle gain ne se transmet pas",
                xy=(0.85, 1.02), xytext=(0.24, 1.45),
                color=ORANGE, fontsize=8.4,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.9))

    fig.suptitle("Optimiser un kernel ne suffit pas — il faut que le pas le voie",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.02)
    save(fig, "05_speedups.png")


# =============================================================== figure 6 ===
def fig_optimizer():
    """SGD vs Adam : niveau atteint et pas nécessaires (OPTIMIZER_ADAM §3, §3.1)."""
    sgd = [0.6506, 0.0940, 0.0474, 0.0415]      # §3, moyenne des pas 1400–1499
    adam = [0.5651, 0.0490, 0.0102, 0.0027]
    thresholds = ["≤ 0,20", "≤ 0,10", "≤ 0,05", "≤ 0,02", "≤ 0,01"]
    sgd_steps = [150, 350, 1075, np.nan, np.nan]  # §3.1, tranche t 192–256
    adam_steps = [25, 25, 75, 200, 375]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

    ax = axes[0]
    x = np.arange(4)
    w = 0.30
    ax.bar(x - w / 2 - 0.015, sgd, width=w, color=MUTED, label="SGD (lr 1e-3)")
    ax.bar(x + w / 2 + 0.015, adam, width=w, color=BLUE, label="Adam (lr 1e-3)")
    for xi, (a, b) in enumerate(zip(sgd, adam)):
        ax.text(xi - w / 2, a * 1.12, f"{a:.3f}".replace(".", ","), ha="center",
                color=INK2, fontsize=8)
        ax.text(xi + w / 2, b * 1.12, f"{b:.4f}".replace(".", ","), ha="center",
                color=BLUE, fontsize=8, fontweight="bold")
    ax.set_xticks(x, BUCKETS)
    ax.set_yscale("log")
    ax.set_ylim(1.4e-3, 2.0)
    ax.set_ylabel("MSE sur ε après 1500 pas (log)")
    ax.set_title("Le niveau atteint à budget égal")
    ax.legend(loc="upper right")
    despine(ax)

    ax = axes[1]
    y = np.arange(len(thresholds))[::-1]
    h = 0.30
    ax.barh(y + h / 2 + 0.015, np.nan_to_num(sgd_steps, nan=0.0), height=h,
            color=MUTED, label="SGD")
    ax.barh(y - h / 2 - 0.015, adam_steps, height=h, color=BLUE, label="Adam")
    for yi, (s, a) in zip(y, zip(sgd_steps, adam_steps)):
        if np.isnan(s):
            ax.text(20, yi + h / 2, "jamais en 1500 pas", va="center",
                    color=CRITICAL, fontsize=8.5, fontweight="bold")
        else:
            ax.text(s + 25, yi + h / 2, f"{s:.0f}", va="center", color=INK2, fontsize=8)
        ax.text(a + 25, yi - h / 2, f"{a}", va="center", color=BLUE, fontsize=8,
                fontweight="bold")
    ax.set_yticks(y, thresholds)
    ax.set_xlim(0, 1380)
    ax.set_xlabel("pas nécessaires pour atteindre — et tenir — le seuil")
    ax.set_ylabel("seuil sur la tranche t 192–256")
    ax.set_title("Le nombre de pas pour y arriver")
    ax.legend(loc="lower right")
    despine(ax)

    fig.suptitle("Adam fait en 75 pas ce que SGD ne fait pas en 1500",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.02)
    save(fig, "06_sgd_vs_adam.png")


# =============================================================== figure 7 ===
def fig_diversity():
    """Diversité inter-seeds et banding (SCALE_UNET §4)."""
    labels = ["baseline\n3 chemins", "L\n3 chemins",
              "baseline\n1 chemin", "L\n1 chemin", "CIFAR-10\nréel"]
    div = [0.00329, 0.00299, 0.00587, 0.00589, 0.2316]
    band = [2.414, 1.623, 4.655, 2.875, 1.074]
    colors = [MUTED, BLUE, MUTED, BLUE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

    ax = axes[0]
    bars = ax.bar(labels, div, width=0.46, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.4)
    ax.set_ylabel("écart-type pixel à pixel entre seeds (log)")
    ax.set_title("Diversité : le modèle L n'apporte rien")
    for b, v in zip(bars, div):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, fr(v, 5 if v < 0.01 else 4),
                ha="center", color=INK2, fontsize=8)
    ax.annotate("", xy=(4, 0.2316), xytext=(4, 0.00589),
                arrowprops=dict(arrowstyle="<->", color=CRITICAL, linewidth=1.0))
    ax.text(3.86, 0.035, "×70", color=CRITICAL, fontsize=10, fontweight="bold",
            ha="right", va="center")
    despine(ax)

    ax = axes[1]
    bars = ax.bar(labels, band, width=0.46, color=colors)
    ax.axhline(1.074, color=GREEN, linewidth=1.0, linestyle="--")
    ax.text(-0.42, 1.2, "niveau du dataset", color=GREEN, fontsize=8.2, va="bottom")
    for b, v in zip(bars, band):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.14, fr(v, 3),
                ha="center", color=INK2, fontsize=8)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("rapport bandes horizontales / verticales")
    ax.set_title("Banding : le seul gain réel de l'agrandissement")
    ax.annotate("−33 %", xy=(1, 1.623), xytext=(0.5, 3.6), color=BLUE,
                fontsize=10, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.9))
    ax.annotate("−38 %", xy=(3, 2.875), xytext=(3.55, 4.3), color=BLUE,
                fontsize=10, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.9))
    despine(ax)

    fig.suptitle("11,7× de calcul et 24× de paramètres ne créent aucune diversité",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.02)
    save(fig, "07_diversite.png")


# =============================================================== figure 8 ===
def fig_inversion():
    """L'inversion de lecture : loss ε vs erreur de reconstruction x0 (SCALE_UNET §5.1)."""
    alpha_bar = schedule(256)
    amp = alpha_bar / (1.0 - alpha_bar)
    t = np.arange(256)

    loss_eps = [0.6103, 0.0644, 0.0219, 0.0142]     # SCALE_UNET §5.1, baseline
    rmse_x0 = [0.0218, 0.308, 0.670, 2.947]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))

    ax = axes[0]
    ax.plot(t, amp, color=BLUE, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("facteur ᾱ/(1−ᾱ)  (log)")
    ax.set_title("La loupe du schedule")
    ax.axhline(1.0, color=AXIS, linewidth=1.0)
    ax.annotate("×2559 à t = 0", xy=(0, amp[0]), xytext=(28, amp[0] * 0.55),
                color=INK2, fontsize=8.4)
    ax.annotate("×0,003 à t = 224", xy=(224, amp[224]),
                xytext=(60, amp[224] * 3.0), color=INK2, fontsize=8.4,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))
    despine(ax)

    ax = axes[1]
    bars = ax.bar(BUCKETS, loss_eps, width=0.55, color=MUTED)
    for b, v in zip(bars, loss_eps):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.4f}".replace(".", ","),
                ha="center", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 3.0)
    ax.set_ylabel("MSE sur ε (log)")
    ax.set_title("Ce que la métrique dit")
    ax.tick_params(axis="x", labelrotation=20)
    ax.text(0.02, 0.90, "« catastrophique » à gauche,\n« quasi parfait » à droite",
            transform=ax.transAxes, color=MUTED, fontsize=8.2, va="top")
    despine(ax)

    ax = axes[2]
    bars = ax.bar(BUCKETS, rmse_x0, width=0.55, color=BLUE)
    for b, v in zip(bars, rmse_x0):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.3f}".replace(".", ","),
                ha="center", color=BLUE, fontsize=8, fontweight="bold")
    ax.axhline(2.0, color=CRITICAL, linewidth=1.0, linestyle="--")
    ax.text(-0.42, 2.25, "plage totale de l'image", color=CRITICAL, fontsize=8.2)
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 30)
    ax.set_ylabel("RMSE sur l'image reconstruite (log)")
    ax.set_title("Ce que ça veut dire")
    ax.tick_params(axis="x", labelrotation=20)
    ax.text(0.02, 0.90, "1,1 % d'erreur à gauche,\n147 % à droite",
            transform=ax.transAxes, color=MUTED, fontsize=8.2, va="top")
    despine(ax)

    fig.suptitle("La même colonne de chiffres, lue à l'endroit : la hiérarchie s'inverse",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.045, ha="left", y=1.04)
    save(fig, "08_inversion_metrique.png")


# =============================================================== figure 9 ===
def fig_collapse():
    """Erreur du modèle rapportée au signal de contenu (SCALE_UNET §5.2, §5.3)."""
    ratio_base = [0.05, 0.64, 1.39, 6.10]     # §5.3
    ratio_L = [0.05, 0.66, 1.56, 7.40]
    base = [0.6103, 0.0644, 0.0219, 0.0142]   # §5.2
    modl = [0.6211, 0.0696, 0.0276, 0.0209]
    trivial = [0.8735, 0.1402, 0.0114, 0.0004]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))

    ax = axes[0]
    x = np.arange(4)
    w = 0.30
    ax.bar(x - w / 2 - 0.015, ratio_base, width=w, color=BLUE, label="baseline")
    ax.bar(x + w / 2 + 0.015, ratio_L, width=w, color=ORANGE, label="modèle L (24× plus gros)")
    ax.axhline(1.0, color=CRITICAL, linewidth=1.2)
    ax.text(-0.42, 1.14, "l'erreur dépasse le signal", color=CRITICAL, fontsize=8.2)
    for xi, (a, b) in enumerate(zip(ratio_base, ratio_L)):
        ax.text(xi - w / 2, a * 1.16, fr(a), ha="center", color=BLUE, fontsize=8)
        ax.text(xi + w / 2, b * 1.16, fr(b), ha="center", color=ORANGE, fontsize=8)
    ax.set_xticks(x, BUCKETS)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 30)
    ax.set_ylabel("erreur du modèle ÷ signal de contenu (log)")
    ax.set_title("Rapporté au signal réellement disponible")
    ax.legend(loc="upper left")
    despine(ax)
    ax.annotate("la chaîne de génération\ndémarre ici",
                xy=(3.2, 9.5), xytext=(1.75, 20),
                color=INK2, fontsize=8.4, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.9))

    ax = axes[1]
    w = 0.24
    ax.bar(x - w - 0.015, base, width=w, color=BLUE, label="baseline")
    ax.bar(x, modl, width=w, color=ORANGE, label="modèle L")
    ax.bar(x + w + 0.015, trivial, width=w, color=MUTED,
           label="prédicteur trivial : recopier l'entrée (0 paramètre)")
    ax.set_xticks(x, BUCKETS)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 12.0)
    ax.set_ylabel("MSE sur ε (log)")
    ax.set_title("Comparé à un prédicteur à zéro paramètre")
    ax.legend(loc="upper right")
    despine(ax)
    # dernière tranche seulement : c'est là que se joue le résultat
    for dx, v, c in ((-w - 0.015, base[3], BLUE), (0.0, modl[3], ORANGE),
                     (w + 0.015, trivial[3], MUTED)):
        ax.text(3 + dx, v * 1.2, fr(v, 4), ha="center", color=c, fontsize=8,
                fontweight="bold")
    ax.text(3, 0.13, "36× pire que\nrecopier l'entrée", color=CRITICAL, fontsize=8.4,
            fontweight="bold", ha="center")

    fig.suptitle("Le mode collapse expliqué : le gradient utile est écrasé d'un facteur ~10⁵",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.02)
    save(fig, "09_collapse_explique.png")


# ============================================================== figure 10 ===
def _load_png(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def fig_plate_before_after():
    """Planche avant/après du conditionnement temporel (insights_samples/)."""
    src = WORKTREE / "insights_samples"
    panels = [
        ("before_infer.png", "Sans conditionnement", "réglages d'inférence\n(magnitude 0,3 · 3 chemins)"),
        ("after_infer.png", "Avec conditionnement", "réglages d'inférence\n(magnitude 0,3 · 3 chemins)"),
        ("before_worst.png", "Sans conditionnement", "pire cas\n(magnitude 1,0 · 1 chemin)"),
        ("after_worst.png", "Avec conditionnement", "pire cas\n(magnitude 1,0 · 1 chemin)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.2))
    for ax, (name, title, sub) in zip(axes, panels):
        ax.imshow(_load_png(src / name), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        color = CRITICAL if name.startswith("before") else BLUE
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(color)
            side.set_linewidth(1.4)
        ax.set_title(title, color=color, fontsize=9.5, loc="center")
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, color=MUTED, fontsize=8,
                ha="center", va="top")

    fig.suptitle("Le fix du conditionnement temporel, sur la même graine (390)",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.06, ha="left", y=1.06)
    save(fig, "10_planche_conditionnement.png")


# ============================================================== figure 11 ===
def fig_plate_diversity():
    """Planche baseline / L / CIFAR réel, 8 seeds (scale_samples/ + dataset)."""
    rows = [
        ("baseline · 10 000 pas", sorted((WORKTREE / "scale_samples/baseline").glob("seed_*.png"),
                                         key=lambda p: int(p.stem.split("_")[1]))),
        ("modèle L · 10 000 pas", sorted((WORKTREE / "scale_samples/L").glob("seed_*.png"),
                                         key=lambda p: int(p.stem.split("_")[1]))),
    ]
    real = load_dataset_images(8)

    fig, axes = plt.subplots(3, 8, figsize=(9.6, 4.2))
    for r, (label, paths) in enumerate(rows):
        for c in range(8):
            ax = axes[r][c]
            ax.imshow(_load_png(paths[c]), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_color(AXIS)
            if c == 0:
                ax.set_ylabel(label, color=INK2, fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=10)
            if r == 0:
                ax.set_title(f"seed {c + 1}", color=MUTED, fontsize=8, pad=4)
    for c in range(8):
        ax = axes[2][c]
        ax.imshow(real[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ax.spines.values():
            side.set_color(GREEN)
        if c == 0:
            ax.set_ylabel("CIFAR-10 réel", color=GREEN, fontsize=8.5, rotation=0,
                          ha="right", va="center", labelpad=10)

    fig.suptitle("Huit graines, deux modèles — et ce que le dataset contient vraiment",
                 color=INK, fontsize=11.5, fontweight="bold", x=0.02, ha="left", y=1.02)
    save(fig, "11_planche_diversite.png")


def main() -> None:
    style()
    m = load_metrics()
    fig_timeline()
    fig_padding()
    fig_loss(m)
    fig_explosion(m)
    fig_perf()
    fig_optimizer()
    fig_diversity()
    fig_inversion()
    fig_collapse()
    fig_plate_before_after()
    fig_plate_diversity()


if __name__ == "__main__":
    main()
