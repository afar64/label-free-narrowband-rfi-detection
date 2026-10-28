#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation round 3: promote the champion (Fisher3': MRD+DAE+ST-flags)
onto every validation surface it has not yet been scored on.

1. RSP1 injection ladder + full width grid + negative + multi-line:
   Fisher3' with TWO p-reference variants:
     (a) LOFAR-clean references (cross-domain, the deployment story);
     (b) SELF references fitted on the 249 base RSP1 patches (zero-label
         on-instrument calibration).
2. LOFAR injection grids (25 width x amplitude cells), negative (6), and
   multi-line (3) sets scored with the champion (LOFAR-clean references).
3. Regenerate injection_maps.png with the champion included (8 detectors).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\optimize_detector3.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\injection_maps.png   (updated, 8 panels)
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import sumthreshold_baseline as sb

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "optimize_detector3.json")
OUTF = os.path.join(D, "injection_maps.png")

WIDTHS = [1, 2, 3, 5, 9]
AMPS = [0.05, 0.10, 0.20, 0.50, 1.00]


def load_members():
    dae = tsd.ConvAE(16, 128)
    dae.load_state_dict(torch.load(os.path.join(D, "ae_denoise_weights.pt"),
                                   map_location="cpu"))
    dae.eval()
    stb = json.load(open(os.path.join(D, "sumthreshold_baseline.json"),
                         encoding="utf-8"))
    return dae, stb["calib"]["chi1"]


def member_scores(dae, chi1, X):
    """(MRD, DAE-top16, ST-flags, ST-max) patch scores for a batch X."""
    mrd = tsd.mrd_scores(X)
    with torch.no_grad():
        errs = []
        for a in range(0, len(X), 256):
            b = min(a + 256, len(X))
            xb = torch.from_numpy(tsd.ae_prep(np.asarray(X[a:b], np.float32)))
            e = (xb - dae(xb)) ** 2
            errs.append(F.avg_pool2d(e, 3, 1, 1)[:, 0].numpy())
    emap = np.concatenate(errs)
    dae_s = np.sort(emap.reshape(len(X), -1), axis=1)[:, -16:].mean(1)
    _, um = sb.st_scores(X, want_map=True)
    return mrd, dae_s, (um > chi1).sum(axis=(1, 2)).astype(np.float32), um.max(axis=(1, 2))


def pvals(s, ref_sorted):
    idx = np.searchsorted(np.sort(np.asarray(ref_sorted, np.float32)), s, side="left")
    return np.clip(1.0 - idx / (len(ref_sorted) + 1.0), 1e-12, 1.0)


def fisher(mrd_p, dae_p, stf_p):
    return -2 * (np.log(mrd_p) + np.log(dae_p) + np.log(stf_p))


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    dae, chi1 = load_members()

    # ---------------- clean-LOFAR references (cross-domain) -----------------
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    Xc = np.asarray(xtr[ci[np.random.default_rng(0).permutation(len(ci))[:4000]]],
                    np.float32)
    refs_lofar = member_scores(dae, chi1, Xc)
    print(f"[setup] LOFAR-clean refs {time.time()-t00:.1f}s", flush=True)

    def champ_sc(X, refs):
        m, d, f, _ = member_scores(dae, chi1, X)
        return fisher(pvals(m, refs[0]), pvals(d, refs[1]), pvals(f, refs[2]))

    res = {}

    # ---------------- 1. RSP1: full grids with the champion -----------------
    S = tsd.rsp1_integrated()
    P, _ = tsd.rsp1_patches(S)
    refs_self = member_scores(dae, chi1, P)
    base_lof = champ_sc(P, refs_lofar)
    base_self = champ_sc(P, refs_self)
    y = np.r_[np.ones(len(P)), np.zeros(len(P))]
    grids = {}
    for w in WIDTHS:
        for amp in AMPS:
            Q = P.copy()
            rng = np.random.default_rng(7)
            for k in range(len(Q)):
                c = int(rng.integers(2, PATCH - 1 - w + 1))
                Q[k, :, c:c + w] *= 1.0 + amp
            s_lof = champ_sc(Q, refs_lofar)
            s_self = champ_sc(Q, refs_self)
            grids[f"w{w}_a{amp:.2f}"] = {
                "auroc_lofarref": float(roc_auc_score(y, np.r_[s_lof, base_lof])),
                "auroc_selfref": float(roc_auc_score(y, np.r_[s_self, base_self]))}
            print(f"[rsp1 w={w} +{amp*100:3.0f}%] fisher3' lof-ref={grids[f'w{w}_a{amp:.2f}']['auroc_lofarref']:.2f} "
                  f"self-ref={grids[f'w{w}_a{amp:.2f}']['auroc_selfref']:.2f}", flush=True)
    res["rsp1_champion_grid"] = grids
    res["rsp1_det_ladder"] = {}
    thr_lof = float(np.percentile(base_lof, 95))
    thr_self = float(np.percentile(base_self, 95))
    for amp in (0.05, 0.10, 0.20, 0.50, 1.00):
        Q = P.copy()
        rng = np.random.default_rng(0)
        for k, c in enumerate(np.random.default_rng(0).integers(2, PATCH - 2, size=len(P))):
            Q[k, :, c] *= 1.0 + amp
        s_lof = champ_sc(Q, refs_lofar)
        s_self = champ_sc(Q, refs_self)
        res["rsp1_det_ladder"][f"{amp:.2f}"] = {
            "det_lofarref": float((s_lof > thr_lof).mean()),
            "det_selfref": float((s_self > thr_self).mean()),
            "auroc_lofarref": float(roc_auc_score(y, np.r_[s_lof, base_lof]))}
        v = res["rsp1_det_ladder"][f"{amp:.2f}"]
        print(f"[rsp1 ladder +{amp*100:3.0f}%] det lof={v['det_lofarref']*100:5.1f}% "
              f"self={v['det_selfref']*100:5.1f}% auroc={v['auroc_lofarref']:.3f}", flush=True)

    # ---------------- 2. LOFAR: champion on all grids -----------------------
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    LAB = np.zeros(n, np.uint8)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        LAB[a:b] = tsd.morphology(np.asarray(yte[a:b]).astype(np.uint8))[0]
    clean = np.where(LAB == 0)[0]
    XL = np.asarray(xte[clean[np.random.default_rng(3).permutation(len(clean))[:2000]]],
                    np.float32)
    baseL = champ_sc(XL, refs_lofar)
    yL = np.r_[np.ones(len(XL)), np.zeros(len(XL))]
    gridL = {}
    for w in WIDTHS:
        for amp in AMPS:
            Q = XL.copy()
            rng = np.random.default_rng(7)
            for k in range(len(Q)):
                c = int(rng.integers(2, PATCH - 1 - w + 1))
                Q[k, :, c:c + w] *= 1.0 + amp
            s = champ_sc(Q, refs_lofar)
            gridL[f"w{w}_a{amp:.2f}"] = float(roc_auc_score(yL, np.r_[s, baseL]))
        print(f"[lofar w={w}] " + "  ".join(f"{gridL[f'w{w}_a{a:.2f}']:.2f}"
                                            for a in AMPS), flush=True)
    res["lofar_champion_grid"] = gridL

    negL, mulL = {}, {}
    for w in (1, 3):
        for amp in (-0.10, -0.20, -0.50):
            Q = XL.copy()
            rng = np.random.default_rng(7)
            for k in range(len(Q)):
                c = int(rng.integers(2, PATCH - 1 - w + 1))
                Q[k, :, c:c + w] *= 1.0 + amp
            s = champ_sc(Q, refs_lofar)
            negL[f"w{w}_a{amp:+.2f}"] = float(roc_auc_score(yL, np.r_[s, baseL]))
    for nlines in (2, 4, 8):
        Q = XL.copy()
        rng = np.random.default_rng(9)
        for k in range(len(Q)):
            for c in rng.choice(np.arange(2, PATCH - 2), size=nlines, replace=False):
                Q[k, :, c] *= 1.20
        s = champ_sc(Q, refs_lofar)
        mulL[f"n{nlines}"] = float(roc_auc_score(yL, np.r_[s, baseL]))
    res["lofar_negative"] = negL
    res["lofar_multi"] = mulL
    print("[lofar neg] " + "  ".join(f"{k}={v:.2f}" for k, v in negL.items()), flush=True)
    print("[lofar multi] " + "  ".join(f"{k}={v:.2f}" for k, v in mulL.items()), flush=True)

    # ---------------- 3. regenerate the money figure (8 panels) -------------
    er = json.load(open(os.path.join(D, "experiments_richer.json"), encoding="utf-8"))
    panels = [("MRD (stage-1)", "MRD"), ("DAE (stage-2, ours)", "DAE"),
              ("Fisher3' fusion (ours)", "CHAMP"), ("ConvAE (original)", "AE"),
              ("PatchCore", "PC"), ("PaDiM", "PD"),
              ("SumThreshold (flags)", "ST-flags"), ("SumThreshold (max)", "ST-max")]
    fig, axes = plt.subplots(2, 8, figsize=(25, 7.2), sharey=True)
    for r, ds in enumerate(("map_rsp1", "map_lofar")):
        old = er[ds]
        for c, (title, key) in enumerate(panels):
            M = np.full((len(WIDTHS), len(AMPS)), np.nan)
            for i, w in enumerate(WIDTHS):
                for j, a in enumerate(AMPS):
                    if key == "CHAMP":
                        src = res["rsp1_champion_grid"] if ds == "map_rsp1" else res["lofar_champion_grid"]
                        M[i, j] = src[f"w{w}_a{a:.2f}"]["auroc_lofarref"] if ds == "map_rsp1" else src[f"w{w}_a{a:.2f}"]
                    else:
                        M[i, j] = old[f"w{w}_a{a:.2f}"][key]["auroc"]
            ax = axes[r, c]
            im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
            for i in range(len(WIDTHS)):
                for j in range(len(AMPS)):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold" if M[i, j] < 0.5 else "normal")
                    if M[i, j] < 0.5:
                        ax.text(j, i - 0.28, "v", ha="center", va="center",
                                fontsize=8, color="b")
            ax.set_xticks(range(len(AMPS)), [f"{a*100:.0f}%" for a in AMPS], fontsize=8)
            if c == 0:
                ax.set_yticks(range(len(WIDTHS)), [f"{w}px" for w in WIDTHS], fontsize=8)
                ax.set_ylabel({"map_rsp1": "RSP1", "map_lofar": "LOFAR"}[ds], fontsize=11)
            else:
                ax.set_yticks([])
            ax.set_title(title, fontsize=10)
    fig.suptitle("Injection recovery (AUROC): the champion fusion inherits the "
                 "MRD weak-narrow corner AND the wide/transient coverage", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.7, label="AUROC")
    fig.savefig(OUTF, dpi=160, bbox_inches="tight")
    print(f"figure updated → {OUTF}", flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
