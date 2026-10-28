#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 2 of the richer experiments.

A3. NEGATIVE-line (absorption-like) injection: amp in {-10%,-20%,-50%},
    w in {1,3}, RSP1 + LOFAR clean-test, all 7 detectors.  Motivated by the
    engineering-pack finding that negative notches masquerade as absorption;
    MRD/ST/PC/PD are two-sided by construction, AE is not - this quantifies
    who sees absorption-like RFI.
A4. MULTI-line injection: n in {2,4,8} disjoint 1-px columns at +20%,
    w=1, RSP1 + LOFAR, all 7 detectors.
C2. MRD scale-set ablation: K in {5,9,17,33} progressively reduced, full
    LOFAR test set, overall + narrow-stratum AUROC ("why these scales?").
B2. PAILED permutation test: image-level paired bootstrap of
    dAUROC = FUS - ST-max on the narrow stratum (same resamples), p-value.
F.  Heatmap figure of the injection maps (if matplotlib present).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\experiments_richer2.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\injection_maps.png
"""
import json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import sumthreshold_baseline as sb
import patchcore_baseline as pcb

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "experiments_richer2.json")
OUTF = os.path.join(D, "injection_maps.png")


def build(net, chi1):
    """Return (DETS dict, st_both) exactly as experiments_richer.py."""
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Xc = np.asarray(xtr[ci], np.float32)
    s1c = tsd.mrd_scores(Xc)
    s2c = tsd.ae_err(net, Xc)[1]
    Fc = pcb.features(net, Xc)
    Vc = Fc.transpose(0, 2, 3, 1).reshape(-1, 96)
    bank = Vc[np.random.default_rng(2).permutation(len(Vc))[:10_000]]
    bt = torch.from_numpy(np.ascontiguousarray(bank))
    bn = (bt ** 2).sum(1)
    mu = Vc.reshape(4000, 256, 96).mean(0)
    sd = Vc.reshape(4000, 256, 96).std(0) + 1e-6

    def pc_sc(X):
        f = pcb.features(net, X)
        v = f.transpose(0, 2, 3, 1).reshape(-1, 96)
        out = np.empty(len(v), np.float32)
        q = torch.from_numpy(v)
        for a in range(0, len(q), 20_000):
            qc = q[a:a + 20_000]
            d2 = (qc ** 2).sum(1, keepdim=True) + bn - 2 * qc @ bt.T
            out[a:a + 20_000] = torch.clamp(d2.min(dim=1).values, min=0).sqrt().numpy()
        return out.reshape(f.shape[0], 16, 16).max(axis=(1, 2))

    def pd_sc(X):
        f = pcb.features(net, X)
        v = f.transpose(0, 2, 3, 1).reshape(f.shape[0], 256, 96)
        return (((v - mu) / sd) ** 2).sum(-1).max(1)

    def fus_sc(X):
        return np.maximum(tsd.rz(tsd.mrd_scores(X), s1c),
                          tsd.rz(tsd.ae_err(net, X)[1], s2c))

    def st_both(X):
        sm, um = sb.st_scores(X, want_map=True)
        return (um > chi1).sum(axis=(1, 2)).astype(np.float32), sm

    DETS = {"MRD": tsd.mrd_scores, "AE": lambda X: tsd.ae_err(net, X)[1],
            "FUS": fus_sc, "PC": pc_sc, "PD": pd_sc}
    return DETS, st_both


def evaluate(name, base, b_flags, b_stmax, Q, dets, st_both, y):
    row = {}
    for k, f in dets.items():
        s = f(Q)
        row[k] = float(roc_auc_score(y, np.r_[s, base[k]]))
    fl, sm = st_both(Q)
    row["ST-flags"] = float(roc_auc_score(y, np.r_[fl, b_flags]))
    row["ST-max"] = float(roc_auc_score(y, np.r_[sm, b_stmax]))
    print(f"[{name}] " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()), flush=True)
    return row


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()
    stb = json.load(open(os.path.join(D, "sumthreshold_baseline.json"), encoding="utf-8"))
    chi1 = stb["calib"]["chi1"]
    DETS, st_both = build(net, chi1)
    res = {"config": {"chi1": chi1, "note": "all metrics = AUROC(injected vs base)"}}

    # ---------------- base sets (identical to experiments_richer.py) --------
    S = tsd.rsp1_integrated()
    P, _ = tsd.rsp1_patches(S)
    baseR = {k: f(P) for k, f in DETS.items()}
    bfR, bsR = st_both(P)

    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    LAB = np.zeros(n, np.uint8)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        LAB[a:b] = tsd.morphology(np.asarray(yte[a:b]).astype(np.uint8))[0]
    clean = np.where(LAB == 0)[0]
    li = clean[np.random.default_rng(3).permutation(len(clean))[:2000]]
    XL = np.asarray(xte[li], np.float32)
    baseL = {k: f(XL) for k, f in DETS.items()}
    bfL, bsL = st_both(XL)
    yR = np.r_[np.ones(len(P)), np.zeros(len(P))]
    yL = np.r_[np.ones(len(XL)), np.zeros(len(XL))]
    print(f"[setup] base sets ready {time.time()-t00:.1f}s", flush=True)

    # ------------------------- A3: negative (absorption-like) lines ---------
    neg = {}
    for tag, (BASE, SC, bf, bs, y) in {
        "rsp1": (P, baseR, bfR, bsR, yR), "lofar": (XL, baseL, bfL, bsL, yL),
    }.items():
        thr = {"ST-flags": float(np.percentile(bf, 95)),
               "ST-max": float(np.percentile(bs, 95))}
        for w in (1, 3):
            for amp in (-0.10, -0.20, -0.50):
                Q = BASE.copy()
                rng = np.random.default_rng(7)
                for k in range(len(Q)):
                    c = int(rng.integers(2, PATCH - 1 - w + 1))
                    Q[k, :, c:c + w] *= (1.0 + amp)
                row = evaluate(f"{tag} neg w={w} {amp:+.0%}", SC, bf, bs, Q, DETS, st_both, y)
                neg[f"{tag}_w{w}_a{amp:+.2f}"] = row
    res["negative_lines"] = neg

    # ------------------------- A4: multi-line injection ---------------------
    mul = {}
    for tag, (BASE, SC, bf, bs, y) in {
        "rsp1": (P, baseR, bfR, bsR, yR), "lofar": (XL, baseL, bfL, bsL, yL),
    }.items():
        for nlines in (2, 4, 8):
            Q = BASE.copy()
            rng = np.random.default_rng(9)
            for k in range(len(Q)):
                cols = rng.choice(np.arange(2, PATCH - 2), size=nlines, replace=False)
                for c in cols:
                    Q[k, :, c] *= 1.20
            row = evaluate(f"{tag} n={nlines} +20%", SC, bf, bs, Q, DETS, st_both, y)
            mul[f"{tag}_n{nlines}"] = row
    res["multi_line"] = mul

    # ------------------------- C2: MRD scale-set ablation -------------------
    abl = {}
    sets = [(5, 9, 17, 33), (9,), (17,), (33,), (5, 9), (9, 17, 33), (5,)]
    yte_full = None
    for sc in sets:
        s1 = np.zeros(n, np.float32)
        for a in range(0, n, 4000):
            b = min(a + 4000, n)
            s1[a:b] = tsd.mrd_scores(np.asarray(xte[a:b], np.float32), scales=sc)
        K = np.load(os.path.join(D, "test_scores.npz"))["K"]
        m = tsd.metrics(LAB, s1)
        sel = (K == 2) | (LAB == 0)
        abl[str(sc)] = {"overall": m["auroc"],
                        "narrow": float(roc_auc_score(LAB[sel], s1[sel]))}
        print(f"[mrd-ablation {str(sc):16s}] overall={m['auroc']:.4f} "
              f"narrow={abl[str(sc)]['narrow']:.4f}", flush=True)
    res["mrd_scale_ablation"] = abl

    # ------------------------- B2: paired permutation test ------------------
    z = np.load(os.path.join(D, "test_scores.npz"))
    FUS, STM, K2 = z["FUS"], z["ST_max"], z["K"]
    LAB2 = z["LAB"]
    per_img, n_img = int(z["per_img"]), 109
    img = np.arange(len(FUS)) // per_img
    B = 2000
    rng = np.random.default_rng(11)
    samples = rng.integers(0, n_img, size=(B, n_img))
    sel_narrow = (K2 == 2) | (LAB2 == 0)
    diffs = np.empty(B)
    for bi in range(B):
        idx = np.concatenate([np.arange(im * per_img, (im + 1) * per_img)
                              for im in samples[bi]])
        mm = sel_narrow[idx]
        a1 = roc_auc_score(LAB2[idx][mm], FUS[idx][mm])
        a2 = roc_auc_score(LAB2[idx][mm], STM[idx][mm])
        diffs[bi] = a1 - a2
    p_one = float((diffs <= 0).mean())
    res["paired_narrow"] = {"mean_dAUROC": float(diffs.mean()),
                            "ci95": [float(np.percentile(diffs, 2.5)),
                                     float(np.percentile(diffs, 97.5))],
                            "p_one_sided": p_one,
                            "B": B}
    print(f"[paired narrow] dAUROC={diffs.mean():+.4f} "
          f"CI[{np.percentile(diffs,2.5):+.4f},{np.percentile(diffs,97.5):+.4f}] "
          f"p={p_one:.4f}", flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
