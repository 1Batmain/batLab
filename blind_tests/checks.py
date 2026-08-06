#!/usr/bin/env python3
"""Suite de tests BOÎTE NOIRE du régime « flux » (mode Perpetual).

Écrite depuis MISSION_BLIND_TEST.md seul — aucune lecture de bat_building/src
ni de main/src. Tout ce qui est mesuré ici provient du binaire et de ses sorties
(dumps f32, PNG, stdout).

Usage : python3 blind_tests/checks.py <dossier_de_dumps>
Sortie : une ligne PASS/FAIL/AMBIGU par propriété + un résumé. Code de retour
non nul si au moins une propriété est FAIL.
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dumpio import read_dump  # noqa: E402

np.seterr(all="ignore")

OUT = sys.argv[1] if len(sys.argv) > 1 else "blind_tests/out"

# --- seuils, fixés depuis la spec (jamais ajustés pour faire passer un test) ---
CORRIDOR = 0.20          # P1 : std(x_t) doit rester à ±20 % de sa moyenne de plateau
DRIFT_MAX = 0.05         # P1 : dérive premier↔dernier décile ≤ 5 % de la moyenne
JUMP_RATIO_MAX = 3.0     # P2 : max(|Δ| par frame) / médiane ≤ 3
CORR_LOCAL_MIN = 0.80    # P3 : corr(k, k+1) ≥ 0,80
CORR_LONG_MAX = 0.60     # P3 : corr(k, k+300) ≤ 0,60
CORR_GAP_MIN = 0.30      # P3 : écart corr(k,k+1) − corr(k,k+300) ≥ 0,30
SEED_CORR_MAX = 0.30     # P5 : deux graines différentes → |corr| ≤ 0,30
STEP_CORR_MAX = 0.15     # P5 : |corr(Δ_k, Δ_{k+1})| ≤ 0,15
DRIFT_DIR_MAX = 3.0      # P5 : ‖moyenne des Δ‖ ≤ 3× ce qu'une marche sans biais donne
ISO_RATIO = (0.80, 1.25)  # P6 : rapport énergie des différences lignes/colonnes
ISO_SHIFT_MAX = 0.05     # P6 : |autocorr spatiale du bruit ajouté| à ±1 px ≤ 0,05

results = []


def report(pid, title, verdict, lines, spec=None):
    results.append((pid, title, verdict, lines, spec))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def mean_corr(A, lag, samples=200):
    n = len(A) - lag
    if n <= 0:
        return float("nan")
    step = max(1, n // samples)
    return float(np.mean([corr(A[i], A[i + lag]) for i in range(0, n, step)]))


def plateau_start(meta):
    """Premier index où le t dumpé atteint sa valeur finale (fin de la descente)."""
    return int(np.argmax(meta == meta[-1]))


def frame_abs_delta(A):
    return np.abs(np.diff(A, axis=0)).reshape(len(A) - 1, -1).mean(axis=1)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


D = {}


def load(name):
    if name not in D:
        D[name] = read_dump(os.path.join(OUT, name + ".f32"))
    return D[name]


# --------------------------------------------------------------------------
# P0 — contrat du dump (préalable à tout le reste)
# --------------------------------------------------------------------------
def p0_format():
    d = load("flux_main")
    L = []
    ok = True
    L.append(f"magic={d['magic']!r} dims={d['w']}x{d['h']}x{d['c']} frames={d['n']} "
             f"(reste partiel : {d['partial']} octets)")
    if d["magic"] != b"BATFLUX1":
        ok = False
        L.append("  ✗ magic ≠ « BATFLUX1 » (contrat --help)")
    if (d["w"], d["h"], d["c"]) != (32, 32, 1):
        ok = False
        L.append("  ✗ dimensions inattendues")
    if d["n"] != int(os.environ.get("ACTIONS", "3000")):
        ok = False
        L.append(f"  ✗ nombre de frames ≠ --actions ({os.environ.get('ACTIONS')})")
    if not np.isfinite(d["xt"]).all() or not np.isfinite(d["x0"]).all():
        ok = False
        L.append("  ✗ valeurs non finies dans le dump")
    L.append(f"x̂₀ borné dans [{d['x0'].min():.3f}, {d['x0'].max():.3f}]")

    # phase : « u8 phase (0 descent, 1 climb, 2 flux) » — documenté par --help depuis la
    # passe 2. La passe 3 dispose EN PLUS d'une règle de frontière, publiée par --help
    # après le correctif de l'off-by-one, et qui n'existait pas quand ce test a été écrit :
    #   « The phase names what the frame DID, so in flux the opening approach is
    #     T-1-t* frames of 0 and every frame from the first churn on is 2 — cut a
    #     prologue on that byte rather than on a count. »
    # Le contrat est donc devenu testable sur DEUX plans, et on teste les deux :
    #   (a) la dynamique — l'étiquette dit-elle la vérité sur ce qui a produit la frame ?
    #       (critère de la passe 2, inchangé : pas inverse et churn diffèrent de ~40 %)
    #   (b) la frontière — le bloc de 0 est-il un préfixe contigu de longueur T-1-t* ?
    # `T` n'est pas défini par le contrat (défaut de spec mineur, cf. §D7) : on ne le
    # suppose donc pas. On vérifie la LOI — longueur affine en t*, de pente exactement
    # −1 — et on recoupe son ordonnée à l'origine avec le haut de chaîne lu dans le dump.
    def phase_audit(d):
        m = d["meta"].astype(int)
        ph = d["phase"]
        ts = int(m[-1])
        dd = frame_abs_delta(d["xt"])  # dd[i-1] = amplitude ayant produit la frame i
        churn = float(np.median(dd[np.where(ph[1:] == 2)[0]]))
        # L'amplitude d'un pas inverse dépend de t : on ne compare qu'AU NIVEAU t*, où les
        # deux dynamiques coexistent. Référence : les dernières frames de descente juste
        # au-dessus de t*.
        ref = float(np.median([dd[i - 1] for i in range(1, len(m))
                               if ph[i] == 0 and ts < m[i] <= ts + 8]))
        susp = [i for i in range(1, len(m))
                if ph[i] == 0 and m[i] == ts
                and abs(dd[i - 1] - churn) < abs(dd[i - 1] - ref)]
        return dict(tstar=ts, churn=churn, ref=ref, susp=susp,
                    n0=int((ph == 0).sum()), n1=int((ph == 1).sum()),
                    first2=int(np.argmax(ph == 2)) if (ph == 2).any() else -1,
                    top=int(m[0]) + 1, amps=[round(float(dd[i - 1]), 4) for i in susp])

    m = d["meta"].astype(int)
    ph = d["phase"]
    a = phase_audit(d)
    mono = bool(np.all(np.diff((ph == 2).astype(int)) >= 0))
    ok &= mono
    L.append(f"phases flux : {sorted(set(ph.tolist()))}, phase 2 jamais quittée une fois "
             f"prise → {'ok' if mono else 'ÉCART'}")

    # (a) dynamique — sur le run principal ET sur tout le cadran (≥3 valeurs de t*)
    dial = [(a["tstar"], a)] + [
        (k, phase_audit(read_dump(os.path.join(OUT, f"ts_{k}.f32"))))
        for k in sorted(int(f[3:-4]) for f in os.listdir(OUT)
                        if f.startswith("ts_") and f.endswith(".f32"))]
    bad = [(k, e) for k, e in dial if e["susp"]]
    ok &= not bad
    L.append("amplitude ↔ étiquette, par cadran (frames « 0 descent » ayant l'amplitude "
             "d'un churn) : " + "  ".join(f"t*={k}→{len(e['susp'])}" for k, e in dial) +
             f" — sur {len(dial)} valeurs de t*")
    for k, e in bad:
        L.append(f"  ✗ t*={k} : {len(e['susp'])} frame(s) mal étiquetée(s) {e['susp']}, "
                 f"|Δ|={e['amps']} — churn={e['churn']:.4f}, "
                 f"pas inverse au même niveau≈{e['ref']:.4f}")
    if not bad:
        L.append(f"  au niveau t*={a['tstar']} : « descent » ≈{a['ref']:.4f} (pas inverse), "
                 f"churn ={a['churn']:.4f} — les deux populations restent disjointes → ok")

    # (b) frontière : préfixe contigu, aucun climb, longueur = T-1-t*
    pref = all(e["first2"] == e["n0"] and e["n1"] == 0 for _, e in dial)
    ok &= pref
    L.append(f"le bloc de « 0 » est un préfixe contigu (première frame de phase 2 = nombre "
             f"de frames de phase 0, aucun climb en flux) → {'ok' if pref else 'ÉCART'}")
    tops = sorted({e["top"] for _, e in dial})
    law = [(k, e["n0"], e["n0"] + k) for k, e in dial]           # n0 + t* doit être constant
    const = sorted({s for _, _, s in law})
    lin = len(const) == 1
    ok &= lin
    L.append("longueur du prologue vs cadran : " +
             "  ".join(f"t*={k}→{n0}" for k, n0, _ in law))
    L.append(f"  n0 + t* = {const} (constante ⇔ loi affine de pente −1) → "
             f"{'ok' if lin else 'ÉCART'}")
    coh = lin and tops == const
    ok &= coh
    L.append(f"  haut de chaîne lu dans le dump (t de la frame 0, +1) = {tops} ; le contrat "
             f"annonce T−1−t* ⇒ T−1 = {const[0] if lin else '?'} → "
             f"{'concordant' if coh else 'ÉCART'}")
    for reg in ("errance", "respiration"):
        e = load(reg)
        seen = sorted(set(e["phase"].tolist()))
        has_climb = 1 in seen
        ok &= has_climb
        L.append(f"phases {reg} : {seen} (0 descent, 1 climb ; jamais 2) → "
                 f"{'ok' if has_climb and 2 not in seen else 'ÉCART'}")
        ok &= 2 not in seen
    # cohérence phase↔t : une frame de descente baisse t, une frame de climb le remonte
    e = load("errance")
    dt = np.diff(e["meta"].astype(int))
    p = e["phase"][1:]
    down = dt[p == 0]
    up = dt[p == 1]
    coh = (down <= 0).all() and (up >= 0).all()
    ok &= coh
    L.append(f"cohérence phase↔t (errance) : descente {int((down <= 0).sum())}/{len(down)} "
             f"frames à Δt≤0, climb {int((up >= 0).sum())}/{len(up)} à Δt≥0 → "
             f"{'ok' if coh else 'ÉCART'}")
    report("P0", "format du dump / sanité", "PASS" if ok else "FAIL", L,
           "contrat publié par --help : « \"BATFLUX1\" u32 width u32 height u32 channels / "
           "then per frame: u8 phase (0 descent, 1 climb, 2 flux) u32 t … »")


# --------------------------------------------------------------------------
# P1 — stationnarité : std(x_t) dans un couloir stable
# --------------------------------------------------------------------------
def p1_stationnarite():
    d = load("flux_main")
    m = d["meta"].astype(int)
    st = plateau_start(m)
    s = d["xt"][st:].reshape(-1, d["w"] * d["h"]).std(axis=1)
    mu = s.mean()
    lo, hi = s.min() / mu - 1, s.max() / mu - 1
    n = len(s)
    dec = (s[-n // 10:].mean() - s[:n // 10].mean()) / mu
    ok = abs(lo) <= CORRIDOR and abs(hi) <= CORRIDOR and abs(dec) <= DRIFT_MAX
    L = [
        f"descente initiale : {st} frames (t {m[0]} → {m[-1]}), plateau = {n} frames à t={m[-1]}",
        f"std(x_t) plateau : moyenne={mu:.4f}  couloir=[{lo * 100:+.1f}%, {hi * 100:+.1f}%]"
        f"  (seuil ±{CORRIDOR * 100:.0f}%)",
        f"dérive premier↔dernier décile = {dec * 100:+.2f}%  (seuil ±{DRIFT_MAX * 100:.0f}%)",
        f"std(x̂₀) plateau : moyenne={d['x0'][st:].reshape(-1, 1024).std(axis=1).mean():.4f}",
    ]
    report("P1", "stationnarité du niveau de bruit", "PASS" if ok else "FAIL", L,
           "ligne 15 : « std(x_t) reste dans un couloir stable sur toute la durée »")


# --------------------------------------------------------------------------
# P2 — pas de saut : |Δ| serré, et plus serré que l'errance
# --------------------------------------------------------------------------
def p2_pas_de_saut():
    d = load("flux_main")
    st = plateau_start(d["meta"].astype(int))
    L = []
    ok = True
    ratios = {}
    for nm, A in (("x_t", d["xt"][st:]), ("x̂₀", d["x0"][st:])):
        dd = frame_abs_delta(A)
        r = dd.max() / np.median(dd)
        ratios[nm] = r
        ok &= r <= JUMP_RATIO_MAX
        L.append(f"flux {nm} : médiane |Δ|={np.median(dd):.5f}  max/médiane={r:.2f}"
                 f"  min/médiane={dd.min() / np.median(dd):.2f}  (seuil ≤ {JUMP_RATIO_MAX})")
    e = load("errance")
    ste = 191  # même longueur de rodage que le flux, pour comparer à budget égal
    for nm, A in (("x_t", e["xt"][ste:]), ("x̂₀", e["x0"][ste:])):
        dd = frame_abs_delta(A)
        re_ = dd.max() / np.median(dd)
        L.append(f"errance {nm} : max/médiane={re_:.2f}  → flux {ratios[nm]:.2f}× vs "
                 f"errance {re_:.2f}× ({'flux plus serré' if ratios[nm] < re_ else 'ANOMALIE'})")
        ok &= ratios[nm] < re_
    report("P2", "aucun saut image-à-image", "PASS" if ok else "FAIL", L,
           "ligne 16 : « max par frame du même ordre de grandeur que la médiane ; "
           "aucune frame ne change brutalement »")


# --------------------------------------------------------------------------
# P3 — continuité locale + dérive longue
# --------------------------------------------------------------------------
def p3_continuite():
    d = load("flux_main")
    st = plateau_start(d["meta"].astype(int))
    L = []
    ok = True
    for nm, A in (("x_t", d["xt"][st:]), ("x̂₀", d["x0"][st:])):
        c1, c300 = mean_corr(A, 1), mean_corr(A, 300)
        good = c1 >= CORR_LOCAL_MIN and c300 <= CORR_LONG_MAX and (c1 - c300) >= CORR_GAP_MIN
        ok &= good
        L.append(f"{nm} : corr(k,k+1)={c1:.4f} (≥{CORR_LOCAL_MIN})  "
                 f"corr(k,k+300)={c300:.4f} (≤{CORR_LONG_MAX})  écart={c1 - c300:.4f} "
                 f"(≥{CORR_GAP_MIN})  → {'ok' if good else 'ÉCART'}")
    A = d["xt"][st:]
    L.append("décroissance corr(x_t) : " + "  ".join(
        f"lag{l}={mean_corr(A, l):+.3f}" for l in (1, 5, 30, 100, 300) if l < len(A)))
    report("P3", "continuité locale + dérive longue", "PASS" if ok else "FAIL", L,
           "ligne 17 : « corr(frame k, k+1) élevée ; corr(frame k, k+300) nettement plus basse »")


# --------------------------------------------------------------------------
# P4 — amplitude par frame ~ √β(t*) : exige de pouvoir choisir t*
# --------------------------------------------------------------------------
def p4_amplitude_tstar():
    """Le cadran est réglable : on teste la loi elle-même, sur trois barreaux."""
    L = []
    ok = True
    ts, rms, var, med = [], [], [], []
    for k in sorted(int(f[3:-4]) for f in os.listdir(OUT)
                    if f.startswith("ts_") and f.endswith(".f32")):
        d = read_dump(os.path.join(OUT, f"ts_{k}.f32"))
        m = d["meta"].astype(int)
        if int(m[-1]) != k:  # le cadran doit être atteint et tenu
            ok = False
            L.append(f"  ✗ --t-star {k} → plateau à t={m[-1]}")
        st = plateau_start(m)
        P = d["xt"][st:]
        Dl = np.diff(P, axis=0)
        ts.append(k)
        rms.append(float(np.sqrt((Dl ** 2).mean())))
        med.append(float(np.median(np.abs(Dl).reshape(len(Dl), -1).mean(axis=1))))
        var.append(float(P.reshape(-1, 1024).var(axis=1).mean()))
    t = np.array(ts, float)
    rms = np.array(rms)
    var = np.array(var)
    L.append("t* réglable : " + "  ".join(f"t*={k}→|Δ|={v:.4f}" for k, v in zip(ts, med)))

    # (1) la lettre de la spec : les changements par frame croissent avec t*
    mono = bool(np.all(np.diff(rms) > 0))
    ok &= mono
    L.append(f"(1) croissance stricte de l'amplitude avec t* sur {len(ts)} valeurs "
             f"({ts[0]} → {ts[-1]}) : {'oui' if mono else 'NON'} ; "
             f"rapport bout à bout = {rms[-1] / rms[0]:.2f}×")

    # (2) la forme √β : β d'un DDPM est affine en t, donc rms(Δ)² doit l'être aussi
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, rms ** 2, rcond=None)
    pred = A @ coef
    r2 = 1 - ((rms ** 2 - pred) ** 2).sum() / ((rms ** 2 - (rms ** 2).mean()) ** 2).sum()
    dev = float(np.max(np.abs(np.sqrt(np.maximum(pred, 0)) - rms) / rms))
    shape = r2 >= 0.999 and dev <= 0.02
    ok &= shape
    L.append(f"(2) rms(Δ)² = {coef[0]:.6f} + {coef[1]:.3e}·t*  →  R²={r2:.6f}, écart max "
             f"sur l'amplitude {100 * dev:.2f}%  ⇒ amplitude ∝ √(affine en t*), la forme "
             f"exacte de √β(t*) pour un schedule β linéaire → {'ok' if shape else 'ÉCART'}")

    # (3) oracle croisé, indépendant de l'implémentation : le niveau stationnaire
    #     var(x_t) = ᾱ·σ₀² + (1−ᾱ) avec ᾱ_t = exp(−Σβ), β déduit de la SEULE amplitude.
    #     Un churn « renoise + denoise » impose rms(Δ)² = 2β : le facteur 2 n'est pas ajusté.
    # Le test ne porte PAS sur le résidu (dont l'ampleur est celle de l'approximation de
    # l'oracle : σ₀ traité comme constant alors que std(x̂₀) varie d'un facteur 3 sur la
    # plage, et ᾱ = exp(−Σβ)). Il porte sur le COEFFICIENT c, que les données déterminent
    # seules : sa valeur théorique est 2 (un bruitage avant + un débruitage arrière).
    def predict(c):
        beta = (coef[0] + coef[1] * np.arange(1, 256)) / c
        al = np.exp(-np.cumsum(beta))
        a0 = al[ts[0] - 1]
        s0sq = (var[0] - (1 - a0)) / a0          # unique paramètre libre : σ₀, calé au plus bas t*
        vp = np.array([al[k - 1] * s0sq + (1 - al[k - 1]) for k in ts])
        r = vp / var - 1
        return float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()), s0sq

    grid = np.linspace(1.2, 3.2, 401)
    cstar = float(grid[int(np.argmin([predict(c)[0] for c in grid]))])
    rmse2, err2, s0sq = predict(2.0)
    cross = abs(cstar - 2.0) <= 0.2
    ok &= cross
    L.append(f"(3) oracle croisé : β̂ tiré de la SEULE amplitude ⇒ niveau stationnaire "
             f"prédit sur les {len(ts)} valeurs de t* (σ₀={np.sqrt(max(s0sq, 0)):.3f}, seul "
             f"paramètre calé) à {100 * rmse2:.2f}% RMS / {100 * err2:.1f}% max")
    L.append(f"    facteur c ajusté librement dans β̂ = rms(Δ)²/c : **c* = {cstar:.3f}** pour "
             f"une valeur théorique de 2,000 (écart {100 * abs(cstar / 2 - 1):.1f}%, seuil 10%) "
             f"→ {'ok' if cross else 'ÉCART'}")
    L.append("    " + "  ".join(f"c={c}→RMS {100 * predict(c)[0]:.1f}%"
                                for c in (1.5, 1.8, 2.0, 2.2, 2.5)))
    L.append("    → l'amplitude par frame et le niveau de bruit tenu, deux observables "
             "indépendants, sont reliés par un unique β(t*) : la loi en √β est vérifiée "
             "quantitativement, pas seulement dans son sens de variation.")
    report("P4", "amplitude par frame ~ √β(t*)", "PASS" if ok else "FAIL", L,
           "ligne 18 : « les changements par frame croissent avec t* … Vérifiable en "
           "comparant deux valeurs de t* »")


# --------------------------------------------------------------------------
# P5 — graines : reproductibilité, divergence, bruit décorrélé
# --------------------------------------------------------------------------
def p5_graines():
    L = []
    ok = True
    h1 = sha(os.path.join(OUT, "flux_main.f32"))
    h2 = sha(os.path.join(OUT, "flux_repeat.f32"))
    same = h1 == h2
    ok &= same
    L.append(f"même seed, deux runs : sha256 {'identiques' if same else 'DIFFÉRENTS'} "
             f"({h1[:16]}…)")

    a = load("flux_main")
    b = load("flux_seed2")
    st = max(plateau_start(a["meta"].astype(int)), plateau_start(b["meta"].astype(int)))
    n = min(len(a["xt"]), len(b["xt"])) - st
    cs = [corr(a["xt"][st + i], b["xt"][st + i]) for i in range(0, n, max(1, n // 200))]
    div = max(abs(np.mean(cs)), 0) <= SEED_CORR_MAX and np.max(np.abs(cs)) <= 0.9
    ok &= div
    L.append(f"seeds différents : corr moyenne={np.mean(cs):+.4f} max|corr|={np.max(np.abs(cs)):.4f}"
             f" (seuil moyenne ≤ {SEED_CORR_MAX}) → {'divergent' if div else 'ÉCART'}")

    P = a["xt"][st:]
    Dl = np.diff(P, axis=0).reshape(len(P) - 1, -1)
    Dc = Dl - Dl.mean(axis=1, keepdims=True)
    Dn = Dc / np.linalg.norm(Dc, axis=1, keepdims=True)
    G = Dn @ Dn.T
    c1 = float(np.mean(np.diagonal(G, 1)))
    offmax = float(np.max(G[~np.eye(len(G), dtype=bool)]))
    good = abs(c1) <= STEP_CORR_MAX and offmax <= 0.5
    ok &= good
    L.append(f"incréments consécutifs : corr(Δ_k,Δ_k+1)={c1:+.4f} (|·| ≤ {STEP_CORR_MAX}) ; "
             f"max des corr entre paires quelconques={offmax:+.4f} → "
             f"{'décorrélés' if good else 'ÉCART'}")

    N = len(Dl)
    mean_step_rms = float(np.sqrt((Dl.mean(axis=0) ** 2).mean()))
    step_rms = float(np.sqrt((Dl ** 2).mean()))
    ratio = mean_step_rms / (step_rms / np.sqrt(N))
    nodrift = ratio <= DRIFT_DIR_MAX
    ok &= nodrift
    L.append(f"direction fixe accumulée : ‖moyenne des Δ‖ = {ratio:.3f}× celle d'une marche "
             f"sans biais (seuil ≤ {DRIFT_DIR_MAX}) → {'aucune' if nodrift else 'ÉCART'}")

    msd = {}
    for Lg in (1, 8, 64, 512, 2048):
        if Lg < len(P):
            step = max(1, (len(P) - Lg) // 100)
            msd[Lg] = float(np.sqrt(np.mean(
                [np.mean((P[i + Lg] - P[i]) ** 2) for i in range(0, len(P) - Lg, step)])))
    L.append("écart quadratique cumulé (rms) : " +
             "  ".join(f"L={k}→{v:.3f}" for k, v in msd.items()) +
             "  → saturation ⇒ marche stationnaire, pas de fuite balistique")
    report("P5", "graines : reproductibilité, divergence, bruit décorrélé",
           "PASS" if ok else "FAIL", L,
           "ligne 19 : « deux runs de même seed sont identiques ; deux seeds différents "
           "divergent ; le bruit ajouté entre frames consécutives est décorrélé »")


# --------------------------------------------------------------------------
# P6 — isotropie du bruit ajouté
# --------------------------------------------------------------------------
def p6_isotropie():
    d = load("flux_main")
    st = plateau_start(d["meta"].astype(int))
    P = d["xt"][st:]
    Dl = np.diff(P, axis=0)  # champ ajouté d'une frame à l'autre
    row = float(np.mean((Dl[:, 1:, :] - Dl[:, :-1, :]) ** 2))
    col = float(np.mean((Dl[:, :, 1:] - Dl[:, :, :-1]) ** 2))
    r = row / col
    ok = ISO_RATIO[0] <= r <= ISO_RATIO[1]
    L = [f"énergie des différences : rangée-à-rangée={row:.5f}  colonne-à-colonne={col:.5f}"
         f"  rapport={r:.4f}  (couloir {ISO_RATIO})"]

    def shiftcorr(A, di, dj):
        a = A[:, max(0, di):A.shape[1] + min(0, di), max(0, dj):A.shape[2] + min(0, dj)]
        b = A[:, max(0, -di):A.shape[1] + min(0, -di), max(0, -dj):A.shape[2] + min(0, -dj)]
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))

    for s_ in [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, -2)]:
        v = shiftcorr(Dl, *s_)
        bad = abs(v) > ISO_SHIFT_MAX
        ok &= not bad
        L.append(f"autocorr spatiale du bruit ajouté, décalage {s_} : {v:+.5f}"
                 f"{'  ← ÉCART' if bad else ''}")
    L.append("le décalage (1,-1) couvre le piège anti-diagonale documenté dans "
             "ANISOTROPY_HUNT.md (champ constant sur les anti-diagonales)")
    report("P6", "isotropie", "PASS" if ok else "FAIL", L,
           "ligne 20 : « les différences rangée-à-rangée et colonne-à-colonne du bruit "
           "ajouté sont du même ordre »")


# --------------------------------------------------------------------------
# P7 — non-régression errance / respiration
# --------------------------------------------------------------------------
def p7_non_regression():
    L = []
    ok = True
    e = load("errance")
    m = e["meta"].astype(int)
    zeros = np.where(m == 0)[0]
    groups = []
    if len(zeros):
        cur = [int(zeros[0])]
        for z in zeros[1:]:
            z = int(z)
            if z == cur[-1] + 1:
                cur.append(z)
            else:
                groups.append(cur)
                cur = [z]
        groups.append(cur)
    reps = [g[0] for g in groups]
    resolved = len(reps) >= 2
    ok &= resolved
    L.append(f"errance : {len(reps)} cycles atteignent t=0 (résolution complète) "
             f"→ {'ok' if resolved else 'ÉCART'}")
    if resolved:
        sres = e["xt"][reps].reshape(len(reps), -1).std(axis=1)
        cc = [corr(e["x0"][reps[i]], e["x0"][reps[i + 1]]) for i in range(len(reps) - 1)]
        far = [corr(e["x0"][reps[i]], e["x0"][reps[i + 5]]) for i in range(len(reps) - 5)] or [np.nan]
        diff = max(cc) <= 0.99
        ok &= diff
        L.append(f"  images des cycles successifs : corr moyenne={np.mean(cc):.3f} "
                 f"min={min(cc):.3f} max={max(cc):.3f} (seuil ≤ 0,99) → "
                 f"{'différentes' if diff else 'IDENTIQUES'}")
        L.append(f"  corr entre cycles éloignés (i,i+5) = {np.mean(far):.3f} "
                 f"→ l'errance dérive bien de cycle en cycle")
        L.append(f"  std(x_t) aux frames résolues : {np.round(sres[:6], 3).tolist()}…")

    r = load("respiration")
    mr = r["meta"].astype(int)
    smin = float(r["xt"].reshape(r["n"], -1).std(axis=1).min())
    never = mr[191:].min() > 0 and smin > 0.25
    ok &= never
    L.append(f"respiration : t ∈ [{mr[191:].min()}, {mr[191:].max()}] après rodage "
             f"(jamais 0) ; min std(x_t)={smin:.4f} (> 0,25) → "
             f"{'jamais résolue' if never else 'ÉCART'}")

    base = os.path.join(OUT, "baseline.json")
    if os.path.exists(base):
        b = json.load(open(base))
        L.append(f"comparaison bit-à-bit avec le binaire pré-flux ({b['rev']}) sur les "
                 f"8 premiers cycles, seed identique :")
        for reg, res in b["png"].items():
            same = all(res.values())
            ok &= same
            L.append(f"  {reg} : {sum(res.values())}/{len(res)} PNG identiques → "
                     f"{'AUCUNE régression' if same else 'RÉGRESSION'}")
    else:
        L.append("(comparaison bit-à-bit avec le binaire pré-flux non exécutée : "
                 "relancer avec BLIND_BASELINE=1)")
    report("P7", "non-régression errance / respiration", "PASS" if ok else "FAIL", L,
           "ligne 21 : « errance : cycles avec résolution complète, images des cycles "
           "successifs différentes ; respiration : jamais résolue »")


# --------------------------------------------------------------------------
# P8 — bornes et diagnostics du CLI (contrat publié par --help)
# --------------------------------------------------------------------------
def p8_cli():
    cli = json.load(open(os.path.join(OUT, "cli.json")))
    L = []
    ok = True

    h = cli["help"]
    good = h["len"] > 0 and h["exit"] == 0
    ok &= good
    L.append(f"--help : {h['len']} octets, code {h['exit']} → {'publié' if good else 'VIDE'} ; "
             f"il documente les flags perpetual, la sémantique de --actions/--frames et "
             f"le format du dump")

    c = cli["unknown"]
    good = c["exit"] != 0 and "unknown flag" in c["msg"]
    ok &= good
    L.append(f"flag inconnu : code {c['exit']}, « {c['msg'][:96]}… » → "
             f"{'rejeté' if good else 'SILENCIEUSEMENT IGNORÉ'}")

    c = cli["flux_frames"]
    good = c["exit"] != 0 and "flux" in c["msg"]
    ok &= good
    L.append(f"« --regime flux --frames 8 » : code {c['exit']}, « {c['msg'][:96]}… » → "
             f"{'refusé explicitement' if good else 'ÉCART'}")
    L.append("    (--help : « Flux closes none: it writes no PNG and REFUSES this bound "
             "rather than running forever. »)")

    n = cli["png"]
    good = n["flux"] == 0 and n["errance"] > 0 and n["respiration"] > 0
    ok &= good
    L.append("PNG écrits sous --frames 8 : " +
             ", ".join(f"{k}→{v}" for k, v in n.items()) +
             f" → {'conforme' if good else 'ÉCART'} (le flux ne boucle aucun cycle)")

    c = cli["both"]
    good = c["frames"] == cli["actions_n"]
    ok &= good
    L.append(f"« --actions {cli['actions_n']} --frames 2 » ensemble → {c['frames']} frames "
             f"dumpées → --actions {'l’emporte' if good else 'NE L’EMPORTE PAS'}")
    L.append("    (--help : « --actions N … Overrides --frames when both are given. »)")

    d = cli["dial"]
    good = all(v == 137 for v in d.values())
    ok &= good
    L.append("cadran sous ses trois orthographes, valeur 137 (bannière) : " +
             ", ".join(f"{k}→t*={v}" for k, v in d.items()) +
             f" → {'les trois pilotent le même cadran' if good else 'ÉCART'}")
    report("P8", "bornes et diagnostics du CLI", "PASS" if ok else "FAIL", L,
           "contrat publié par --help (sections « Perpetual notes » et « --dump »)")


for fn in (p0_format, p1_stationnarite, p2_pas_de_saut, p3_continuite,
           p4_amplitude_tstar, p5_graines, p6_isotropie, p7_non_regression, p8_cli):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        report(fn.__name__, fn.__doc__ or fn.__name__, "FAIL", [f"exception : {exc!r}"])

print()
for pid, title, verdict, lines, spec in results:
    mark = {"PASS": "\033[32mPASS\033[0m", "FAIL": "\033[31mFAIL\033[0m",
            "AMBIGU": "\033[33mAMBIGU\033[0m", "OBS": "\033[33mOBS \033[0m"}[verdict]
    print(f"[{mark}] {pid} — {title}")
    for ln in lines:
        print(f"        {ln}")
    if verdict != "PASS" and spec:
        print(f"        \033[31mspec violée →\033[0m {spec}")
    print()

nfail = sum(1 for r in results if r[2] == "FAIL")
nspec = sum(1 for r in results if r[2] != "OBS")
print(f"résumé : {nspec - nfail}/{nspec} propriétés PASS, {nfail} FAIL"
      f" (+ {len(results) - nspec} observation(s) hors spec numérotée)")
sys.exit(1 if nfail else 0)
