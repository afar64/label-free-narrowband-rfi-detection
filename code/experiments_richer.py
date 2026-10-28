#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Richer experiments for the two-stage detector paper (all on the SAME
LOFAR patches / RSP1 protocol as the existing baselines).

A. Injection-recovery MAPS: width x amplitude grid (1/2/3/5/9 px columns x
   5/10/20/50/100 %), RSP1 AND LOFAR (injected into clean TEST patches),
   scored with ALL detector families (MRD, AE, fusion, SumThreshold
   flags/max, PatchCore in-domain, PaDiM).  The 2-D map is the paper's main
   figure: it shows the exact region where each family breaks.
B. (separate script) image-level bootstrap CIs.
C. Fusion calibration-size ablation: robust-z reference built from 100 /
   250 / 500 / 1000 / 2000 / 4000 clean-train patches -> how much clean data
   the zero-label fusion protocol actually needs.  (The AE/MRD themselves
   stay as trained; this isolates the fusion calibration knob.)
D. Runtime table: ms/patch per detector, same hardware, warm batches.

All thresholds = p95 of each detector's OWN scores on the corresponding
un-injected base set (same convention as run_rsp1).  No test labels are used
for any calibration.

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\experiments_richer.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\test_scores.npz   (for bootstrap_ci.py)
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import sumthreshold_baseline as sb
import patchcore_baseline as pcb

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "experiments_richer.json")
OUTN = os.path.join(D, "test_scores.npz")

WIDTHS = (1, 2, 3, 5, 9)
AMPS = (0.05, 0.10, 0.20, 0.50, 1.00)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def inject_width(P, rng, w, amp):
    """Multiply columns [c, c+w) by (1+amp) per patch; c drawn so the block
    fits.  For w=1 the column distribution matches run_rsp1's
    integers(2, PATCH-2)."""
    Q = P.copy()
    for k in range(len(Q)):
        c = int(rng.integers(2, PATCH - 1 - w + 1))
        Q[k, :, c:c + w] *= 1.0 + amp
    return Q


def main():
    t00 = time.time()
    print(f"[env] device={DEV} threads={os.cpu_count()}", flush=True)
    torch.set_num_threads(os.cpu_count())

    # ---------------------------------------------------------------- setup
    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()

    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Xc = np.asarray(xtr[ci], np.float32)

    s1c = tsd.mrd_scores(Xc)
    s2c = tsd.ae_err(net, Xc)[1]
    stb = json.load(open(os.path.join(D, "sumthreshold_baseline.json"), encoding="utf-8"))
    CHI1 = stb["calib"]["chi1"]
    print(f"[setup] clean train n={len(Xc)}  chi1={CHI1:.3f}  {time.time()-t00:.1f}s",
          flush=True)

    # PatchCore in-domain: features of clean train -> bank / PaDiM stats
    Fc = pcb.features(net, Xc)                                   # (4000,96,16,16)
    Vc = Fc.transpose(0, 2, 3, 1).reshape(-1, 96)
    bank = Vc[np.random.default_rng(2).permutation(len(Vc))[:10_000]]
    bt = torch.from_numpy(np.ascontiguousarray(bank))
    bn = (bt ** 2).sum(1)
    mu = Vc.reshape(4000, 256, 96).mean(0)
    sd = Vc.reshape(4000, 256, 96).std(0) + 1e-6

    def pc_sc(X):
        f = pcb.features(net, X)
        n = f.shape[0]
        v = f.transpose(0, 2, 3, 1).reshape(-1, 96)
        out = np.empty(len(v), np.float32)
        q = torch.from_numpy(v)
        for a in range(0, len(q), 20_000):
            qc = q[a:a + 20_000]
            d2 = (qc ** 2).sum(1, keepdim=True) + bn - 2 * qc @ bt.T
            out[a:a + 20_000] = torch.clamp(d2.min(dim=1).values, min=0).sqrt().numpy()
        return out.reshape(n, 16, 16).max(axis=(1, 2))

    def pd_sc(X):
        f = pcb.features(net, X)
        n = f.shape[0]
        v = f.transpose(0, 2, 3, 1).reshape(n, 256, 96)
        return (((v - mu) / sd) ** 2).sum(-1).max(1)

    def fus_sc(X):
        z1 = tsd.rz(tsd.mrd_scores(X), s1c)
        z2 = tsd.rz(tsd.ae_err(net, X)[1], s2c)
        return np.maximum(z1, z2)

    DETS = {
        "MRD": lambda X: tsd.mrd_scores(X),
        "AE": lambda X: tsd.ae_err(net, X)[1],
        "FUS": fus_sc,
        "PC": pc_sc,
        "PD": pd_sc,
    }

    def st_both(X):
        sm, um = sb.st_scores(X, want_map=True)
        return (um > CHI1).sum(axis=(1, 2)).astype(np.float32), sm

    # ------------------------------------------ Part A1: RSP1 injection map
    S = tsd.rsp1_integrated()
    P, xs = tsd.rsp1_patches(S)
    base = {k: f(P) for k, f in DETS.items()}
    b_flags, b_stmax = st_both(P)
    thr = {k: float(np.percentile(v, 95)) for k, v in base.items()}
    thr["ST-flags"] = float(np.percentile(b_flags, 95))
    thr["ST-max"] = float(np.percentile(b_stmax, 95))
    print(f"[rsp1] base p95 " + "  ".join(f"{k}={v:.3g}" for k, v in thr.items()),
          flush=True)

    map_rsp1 = {}
    y = np.r_[np.ones(len(P)), np.zeros(len(P))]
    for w in WIDTHS:
        for amp in AMPS:
            Q = inject_width(P, np.random.default_rng(7), w, amp)
            row = {}
            for k, f in DETS.items():
                s = f(Q)
                row[k] = {"det": float((s > thr[k]).mean()),
                          "auroc": float(roc_auc_score(y, np.r_[s, base[k]]))}
            fl, sm = st_both(Q)
            row["ST-flags"] = {"det": float((fl > thr["ST-flags"]).mean()),
                               "auroc": float(roc_auc_score(y, np.r_[fl, b_flags]))}
            row["ST-max"] = {"det": float((sm > thr["ST-max"]).mean()),
                             "auroc": float(roc_auc_score(y, np.r_[sm, b_stmax]))}
            map_rsp1[f"w{w}_a{amp:.2f}"] = row
            print(f"[rsp1 w={w} +{amp*100:3.0f}%] " +
                  "  ".join(f"{k}={row[k]['auroc']:.2f}" for k in row), flush=True)

    # --------------------------------- Part A2: LOFAR injection (clean test)
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    LAB = np.zeros(n, np.uint8); NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        Y = np.asarray(yte[a:b]).astype(np.uint8)
        LAB[a:b], _, NP[a:b], NA[a:b] = tsd.morphology(Y)
    K = tsd.classify(NP, NA)
    clean_idx = np.where(LAB == 0)[0]
    li = clean_idx[np.random.default_rng(3).permutation(len(clean_idx))[:2000]]
    XL = np.asarray(xte[li], np.float32)
    baseL = {k: f(XL) for k, f in DETS.items()}
    bl_flags, bl_stmax = st_both(XL)
    thrL = {k: float(np.percentile(v, 95)) for k, v in baseL.items()}
    thrL["ST-flags"] = float(np.percentile(bl_flags, 95))
    thrL["ST-max"] = float(np.percentile(bl_stmax, 95))
    print(f"[lofar] clean base n={len(li)} p95 " +
          "  ".join(f"{k}={v:.3g}" for k, v in thrL.items()), flush=True)

    map_lofar = {}
    yL = np.r_[np.ones(len(XL)), np.zeros(len(XL))]
    for w in WIDTHS:
        for amp in AMPS:
            Q = inject_width(XL, np.random.default_rng(7), w, amp)
            row = {}
            for k, f in DETS.items():
                s = f(Q)
                row[k] = {"det": float((s > thrL[k]).mean()),
                          "auroc": float(roc_auc_score(yL, np.r_[s, baseL[k]]))}
            fl, sm = st_both(Q)
            row["ST-flags"] = {"det": float((fl > thrL["ST-flags"]).mean()),
                               "auroc": float(roc_auc_score(yL, np.r_[fl, bl_flags]))}
            row["ST-max"] = {"det": float((sm > thrL["ST-max"]).mean()),
                             "auroc": float(roc_auc_score(yL, np.r_[sm, bl_stmax]))}
            map_lofar[f"w{w}_a{amp:.2f}"] = row
            print(f"[lofar w={w} +{amp*100:3.0f}%] " +
                  "  ".join(f"{k}={row[k]['auroc']:.2f}" for k in row), flush=True)

    # ------------------------------- Part C: fusion calibration-size ablation
    t0 = time.time()
    S1 = np.zeros(n, np.float32); S2 = np.zeros(n, np.float32)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        X = np.asarray(xte[a:b], np.float32)
        S1[a:b] = tsd.mrd_scores(X)
        S2[a:b] = tsd.ae_err(net, X)[1]
        print(f"  test scores {a}/{n} {time.time()-t0:.0f}s", flush=True)
    fl_all, stmax_all, st_per = [], [], []
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        X = np.asarray(xte[a:b], np.float32)
        fA, sA = st_both(X)
        fl_all.append(fA); stmax_all.append(sA)
        st_per.append(a)
    fl_all = np.concatenate(fl_all); stmax_all = np.concatenate(stmax_all)

    ablation = {}
    for k in (100, 250, 500, 1000, 2000, 4000):
        z1 = tsd.rz(S1, s1c[:k]); z2 = tsd.rz(S2, s2c[:k])
        zf = np.maximum(z1, z2)
        m = tsd.metrics(LAB, zf)
        seln = (K == 2); mn = seln | (LAB == 0)
        ablation[str(k)] = {"overall": m,
                            "narrow_auroc": float(roc_auc_score(LAB[mn], zf[mn]))}
        print(f"[fusion-ablation k={k:4d}] overall AUROC={m['auroc']:.4f} "
              f"narrow={ablation[str(k)]['narrow_auroc']:.4f}", flush=True)

    # ------------------------------------------------ Part D: runtime table
    Xb = np.asarray(xte[:512], np.float32)
    timing = {}
    for name, f in DETS.items():
        f(Xb[:64])                                       # warm
        t0 = time.time()
        for _ in range(2):
            f(Xb)
        timing[name] = (time.time() - t0) / 2 / len(Xb) * 1000
    st_both(Xb[:64])
    t0 = time.time(); st_both(Xb); timing["SumThreshold"] = (time.time()-t0)/len(Xb)*1000
    print("[runtime ms/patch] " + "  ".join(f"{k}={v:.2f}" for k, v in timing.items()),
          flush=True)

    # ------------------------- per-patch test scores for bootstrap_ci.py ---
    z1 = tsd.rz(S1, s1c); z2 = tsd.rz(S2, s2c)
    np.savez_compressed(OUTN, S1=S1, S2=S2, FUS=np.maximum(z1, z2),
                        ST_flags=fl_all, ST_max=stmax_all,
                        LAB=LAB, K=K, n_img=109, per_img=225)

    res = {"config": {"widths": list(WIDTHS), "amps": list(AMPS),
                      "chi1": CHI1, "lofar_clean_subset": int(len(li)),
                      "inject_rng": 7,
                      "notes": "PC-ImageNet arm skipped on the map (CPU budget); "
                               "its w=1 ladder is in ad_imagenet.json"},
           "thr_rsp1": thr, "thr_lofar": thrL,
           "map_rsp1": map_rsp1, "map_lofar": map_lofar,
           "fusion_ablation": ablation,
           "runtime_ms_per_patch": timing}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
