#!/usr/bin/env python3
"""P2 — la couche entraînée fait quelque chose, et la génération n'est pas dégénérée.

Spec : « le checkpoint `night_run.ckpt` (30 couches, avec attention) génère des
images NON dégénérées (bornées [-1,1], non constantes, diversité inter-seed
> 0,2 à magnitude 1,0 — cf. tools/sample_diversity.py) ».

Trois volets :
  * les 4 projections de la couche entraînée ne sont plus à l'init — en
    particulier W_o n'est plus nul : la couche n'est plus l'identité ;
  * huit graines à magnitude 1,0 : bornes, non-constance, zéro non-fini ;
  * diversité inter-graines, mesurée par l'outil du dépôt (tools/sample_diversity.py).
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import ckpt as ckptio
import harness as H

XL = "Color_Diffusion_XL"
CKPT = os.path.join(H.WORK, "Models", XL, "pretrained_weights", "night_run.ckpt")
BACKUP = "/Users/bat/development/lab/batLab/.checkpoint_backup/Color_Diffusion_XL_attn_5700.ckpt"
OUT = os.path.join(H.WORK, "out", "night")
SEEDS = list(range(1, 9))
results = []


def ensure_ckpt():
    if not os.path.exists(CKPT):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        open(CKPT, "wb").write(open(BACKUP, "rb").read())


def sample(seed, magnitude=1.0, ckpt=None, tag="n"):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, f"{tag}_seed{seed}.png")
    log = os.path.join(OUT, f"{tag}_seed{seed}.jsonl")
    r = subprocess.run([H.BIN, "--headless-sample", XL, "--checkpoint", ckpt or CKPT,
                        "--seed", str(seed), "--paths", "3",
                        "--magnitude", str(magnitude), "--out", png, "--log", log],
                       env=dict(os.environ, BATLAB_ROOT=H.WORK), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    head = json.loads(open(log).readline())
    return png, head["final_image"]


if __name__ == "__main__":
    ensure_ckpt()

    c = ckptio.read(CKPT)
    rec = [r for r in c["records"] if r["index"] == 14][0]
    W = rec["w"].reshape(4, 192, 192)
    B = rec["b"].reshape(4, 192)
    results.append(("la couche entraînée n'est plus l'identité (W_o ≠ 0)",
                    np.abs(W[3]).max() > 1e-3 and all(np.abs(W[i]).max() > 1e-3 for i in range(3)),
                    "pas %s ; écarts-types Q/K/V/W_o = %s ; |W_o|max=%.3f"
                    % (c["step"], np.round([W[i].std() for i in range(4)], 4).tolist(),
                       np.abs(W[3]).max())))

    stats = [sample(s) for s in SEEDS]
    bad = [(s, st) for (s, (_, st)) in zip(SEEDS, stats)
           if st["non_finite"] or st["min"] < -1.0001 or st["max"] > 1.0001 or st["std"] < 1e-3]
    results.append(("8 graines : bornées [-1,1], non constantes, zéro non-fini", not bad,
                    "std ∈ [%.3f, %.3f] ; min=%.4f max=%.4f ; non-finis=%d"
                    % (min(st["std"] for _, st in stats), max(st["std"] for _, st in stats),
                       min(st["min"] for _, st in stats), max(st["max"] for _, st in stats),
                       sum(st["non_finite"] for _, st in stats))))

    tool = os.path.join(H.ROOT, "tools", "sample_diversity.py")
    r = subprocess.run([sys.executable, tool, "--glob", os.path.join(OUT, "n_seed*.png"),
                        "--label", "night_run"], capture_output=True, text=True)
    txt = r.stdout + r.stderr
    div = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]) if "{" in txt else {}
    inter = div.get("inter_seed_std")
    results.append(("diversité inter-graines > 0,2 à magnitude 1,0 (tools/sample_diversity.py)",
                    inter is not None and inter > 0.2,
                    json.dumps({k: div[k] for k in
                                ("inter_seed_std", "mean_pairwise_rmse", "intra_image_std",
                                 "banding_ratio") if k in div})))

    n = 0
    for label, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + label + "\n        " + detail)
        n += not ok
    sys.exit(1 if n else 0)
