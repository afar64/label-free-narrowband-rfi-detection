#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Industrial anomaly-detection family (PatchCore / PaDiM) on the SAME LOFAR
patches and the SAME RSP1 injection protocol as two_stage_detector.py.

Why this migration (the diagnosis is already measured, see
sumthreshold_baseline_report.md): the 4-level stride-2 ConvAE is structurally
blind to 1-px narrowband RFI (AUROC 0.500 at any contrast), because stride-2
averaging destroys a 1-px line at the first level.  The industrial-AD family
was designed against exactly this failure: PatchCore scores PATCH-level
nearest-neighbour distances in a feature map, and PaDiM fits per-position
statistics - neither has a strided reconstruction bottleneck.

Faithful-in-spirit adaptation, fixed before touching test labels:
  - Backbone = the EXISTING clean-train ConvAE encoder (lofar_ae_weights_final)
    taken at block2 (16x16x32) and block3 (8x8x64, bilinear-upsampled),
    concatenated -> 256 patch vectors of dim 96 per patch.  ImageNet features
    are NOT used: spectrograms are outside their domain; the in-domain encoder
    is the adaptation (the ImageNet arm needs torchvision and is run separately
    if available).
  - PatchCore: memory bank = patch vectors of the 4000 clean-train patches
    (same rng(0) selection as every other calibration here), randomly
    subsampled to BANK=10k (PatchCore's greedy coreset approximated by random
    subsampling at a similar budget; documented).  Score per position = L2
    distance to the NEAREST bank vector (plain 1-NN; no density reweighting).
  - PaDiM: per-position diagonal Gaussian (mean/var over the same 4000
    patches); score = squared Mahalanobis (sum of per-dim z^2).
  - Patch score = max over the 16x16 map (top1, same convention as the other
    detectors); pixel map = bilinear upsample to 64x64.
  - RSP1: two memory variants - (a) the LOFAR bank (cross-domain, mirrors the
    AE domain-shift test), (b) SELF-referential bank built from the 249 base
    RSP1 patches themselves (zero-label deployment-realistic).  Thresholds =
    p95 of base scores, same convention as the other detectors.

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\ad_baseline.json
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import two_stage_detector as tsd

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "ad_baseline.json")

BANK = 10_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Feat(torch.nn.Module):
    """Block2+Block3 of the clean-train ConvAE encoder, concatenated."""

    def __init__(self, net):
        super().__init__()
        self.enc = net.enc
        self.eval()

    def forward(self, x):                       # x: (n,1,64,64) ae_prep space
        f2 = self.enc[0:6](x)                   # (n,32,16,16)
        f3 = self.enc[6:9](f2)                  # (n,64,8,8)
        f3u = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                            align_corners=False)
        return torch.cat([f2, f3u], 1)          # (n,96,16,16)


def features(net, X, bs=256):
    ft = Feat(net).to(DEV)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(tsd.ae_prep(X[i:i + bs])).to(DEV)
            out.append(ft(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float32)   # (n,96,16,16)


def feat_of_patches(net, P):
    return features(net, P)


def map_from_feat(feat, score_fn, size=64):
    """feat (n,96,16,16) -> (map16, map64)."""
    n = feat.shape[0]
    v = feat.transpose(0, 2, 3, 1).reshape(-1, feat.shape[1])   # (n*256, C)
    s = np.empty(n * 256, np.float32)
    CH = 40_000
    for a in range(0, len(v), CH):
        s[a:a + CH] = score_fn(v[a:a + CH])
    m16 = s.reshape(n, 16, 16).astype(np.float32)
    t = torch.from_numpy(m16[:, None])
    m64 = F.interpolate(t, size=size, mode="bilinear",
                        align_corners=False)[:, 0].numpy()
    return m16, m64


def main():
    t00 = time.time()
    print(f"[env] device={DEV}", flush=True)
    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()

    # ---------- clean-train calibration features (same 4000, rng(0)) --------
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Fc = features(net, np.asarray(xtr[ci], np.float32))          # (4000,96,16,16)
    Vc = Fc.transpose(0, 2, 3, 1).reshape(-1, 96)

    # PatchCore memory bank: random subsample (coreset approximation)
    rngb = np.random.default_rng(2)
    bank = Vc[rngb.permutation(len(Vc))[:BANK]]
    bank_t = torch.from_numpy(bank).to(DEV)                       # (BANK,96)
    b_norm = (bank_t ** 2).sum(1)                                 # (BANK,)
    print(f"[calib] clean feats {Vc.shape}  bank {bank.shape}  "
          f"{time.time()-t00:.1f}s", flush=True)

    # PaDiM per-position diagonal Gaussians
    Vp = Vc.reshape(4000, 256, 96)
    mu = Vp.mean(0)                                               # (256,96)
    sd = Vp.std(0) + 1e-6

    def knn_score(Q):
        """(m,96) -> (m,) nearest-bank L2 distance."""
        out = np.empty(len(Q), np.float32)
        q = torch.from_numpy(Q).to(DEV)
        for a in range(0, len(Q), 20_000):
            qc = q[a:a + 20_000]
            d2 = (qc ** 2).sum(1, keepdim=True) + b_norm - 2 * qc @ bank_t.T
            out[a:a + 20_000] = torch.clamp(d2.min(dim=1).values, min=0).sqrt().cpu().numpy()
        return out

    def padim_maps(feat, mu_, sd_):
        n = feat.shape[0]
        v = feat.transpose(0, 2, 3, 1).reshape(n, 256, 96)
        z = (v - mu_) / sd_
        return (z ** 2).sum(-1).astype(np.float32)                # (n,256)

    def both_maps(feat):
        """Returns (patchcore16, patchcore64, padim16, padim64)."""
        m16pc, m64pc = map_from_feat(feat, knn_score)
        n = feat.shape[0]
        s = padim_maps(feat, mu, sd).reshape(-1)
        m16pd = s.reshape(n, 16, 16)
        t = torch.from_numpy(m16pd[:, None])
        m64pd = F.interpolate(t, size=64, mode="bilinear",
                              align_corners=False)[:, 0].numpy()
        return m16pc, m64pc, m16pd, m64pd

    # ---------- LOFAR test, patch level ----------
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    S = {k: np.zeros(n, np.float32) for k in ("PC", "PD")}
    LAB = np.zeros(n, np.uint8); FR = np.zeros(n, np.float32)
    NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
    CH = 2000; t0 = time.time()
    for a in range(0, n, CH):
        b = min(a + CH, n)
        X = np.asarray(xte[a:b], np.float32)
        Y = np.asarray(yte[a:b]).astype(np.uint8)
        ft = features(net, X)
        _, m64pc, _, m64pd = both_maps(ft)
        S["PC"][a:b] = m64pc.max(axis=(1, 2))
        S["PD"][a:b] = m64pd.max(axis=(1, 2))
        LAB[a:b], FR[a:b], NP[a:b], NA[a:b] = tsd.morphology(Y)
        print(f"  patch-level {a}/{n} {time.time()-t0:.0f}s", flush=True)

    K = tsd.classify(NP, NA)
    patch = {}
    for k in ("PC", "PD"):
        patch[k] = tsd.metrics(LAB, S[k])
        print(f"[LOFAR patch] {k} AUROC={patch[k]['auroc']:.4f} "
              f"AUPRC={patch[k]['auprc']:.4f} F1={patch[k]['best_f1']:.4f}", flush=True)

    strata = {}
    for ki in (1, 2, 3):
        sel = (K == ki)
        if sel.sum() < 30:
            continue
        m = sel | (LAB == 0)
        strata[tsd.STRATA[ki]] = {"n": int(sel.sum()),
                                  "auroc": {k: float(roc_auc_score(LAB[m], S[k][m]))
                                            for k in ("PC", "PD")}}
        print(f"[stratum n={sel.sum():5d}] {tsd.STRATA[ki]:42s} " +
              "  ".join(f"{k}={v:.3f}" for k, v in strata[tsd.STRATA[ki]]["auroc"].items()),
              flush=True)

    # ---------- pixel level, same subsample ----------
    rng = np.random.default_rng(1)
    idx_n = np.where(K == 2)[0]
    idx_o = rng.permutation(np.where(K != 2)[0])[:max(0, 8000 - len(idx_n))]
    pix_idx = np.sort(np.r_[idx_n, idx_o])
    print(f"[pixel] {len(pix_idx)} patches (narrow={len(idx_n)})", flush=True)
    ps = {k: [] for k in ("PC", "PD")}; pt = []
    for a in range(0, len(pix_idx), 500):
        ii = pix_idx[a:a + 500]
        X = np.asarray(xte[ii], np.float32)
        Y = np.asarray(yte[ii]).astype(np.uint8)
        _, m64pc, _, m64pd = both_maps(features(net, X))
        ps["PC"].append(m64pc.reshape(-1)); ps["PD"].append(m64pd.reshape(-1))
        pt.append(Y.reshape(-1))
    pt = np.concatenate(pt)
    pix = {}
    for k in ("PC", "PD"):
        p = np.concatenate(ps[k])
        pix[k] = {"auroc": float(roc_auc_score(pt, p)),
                  "auprc": float(average_precision_score(pt, p))}
        print(f"[LOFAR pixel] {k} AUROC={pix[k]['auroc']:.4f} "
              f"AUPRC={pix[k]['auprc']:.4f}", flush=True)

    # ---------- RSP1 injection, same protocol ----------
    # Four zero-label variants: PatchCore/PaDiM x {LOFAR-fitted reference,
    # SELF-fitted reference (base patches only, deployment-realistic)}.
    Sp = tsd.rsp1_integrated()
    P, xs = tsd.rsp1_patches(Sp)
    Fp = feat_of_patches(net, P)                                  # (249,96,16,16)
    rng2 = np.random.default_rng(0)
    cols = rng2.integers(2, PATCH - 2, size=len(P))
    y = np.r_[np.ones(len(P)), np.zeros(len(P))]
    Vs = Fp.transpose(0, 2, 3, 1).reshape(-1, 96)

    def pc_sc(Qft, bt, bn):
        n2 = Qft.shape[0]
        v = Qft.transpose(0, 2, 3, 1).reshape(-1, 96)
        out = np.empty(n2 * 256, np.float32)
        q = torch.from_numpy(v).to(DEV)
        for a2 in range(0, len(q), 20_000):
            qc = q[a2:a2 + 20_000]
            d2 = (qc ** 2).sum(1, keepdim=True) + bn - 2 * qc @ bt.T
            out[a2:a2 + 20_000] = torch.clamp(d2.min(dim=1).values, min=0).sqrt().cpu().numpy()
        return out.reshape(n2, 16, 16).max(axis=(1, 2))

    variants = {
        "PC_lofar": {"sc": lambda Q, bt, bn: pc_sc(Q, bt, bn),
                     "ref": (bank_t, b_norm)},
        "PC_self": {"sc": lambda Q, bt, bn: pc_sc(Q, bt, bn),
                    "ref": None},   # filled below from the base patches
    }
    bt_self = torch.from_numpy(Vs).to(DEV)
    variants["PC_self"]["ref"] = (bt_self, (bt_self ** 2).sum(1))
    mu_self = Vs.reshape(len(P), 256, 96).mean(0)
    sd_self = Vs.reshape(len(P), 256, 96).std(0) + 1e-6

    sweep = {}
    for tag, cfg in variants.items():
        bt, bn = cfg["ref"]
        base = cfg["sc"](Fp, bt, bn)
        thr = float(np.percentile(base, 95))
        sweep[tag] = {"p95": thr}
        for amp in (0.05, 0.1, 0.2, 0.5, 1.0):
            Q = P.copy()
            for k, c in enumerate(cols):
                Q[k, :, c] *= (1.0 + amp)
            FQ = feat_of_patches(net, Q)
            sc_ = cfg["sc"](FQ, bt, bn)
            sweep[tag][f"{amp:.2f}"] = {
                "det": float((sc_ > thr).mean()),
                "auroc": float(roc_auc_score(y, np.r_[sc_, base]))}
            v = sweep[tag][f"{amp:.2f}"]
            print(f"[{tag} inject +{amp*100:5.1f}% 1px] det={v['det']*100:5.1f}% "
                  f"(AUROC {v['auroc']:.3f})", flush=True)

    for tag, mu_, sd_ in (("PD_lofar", mu, sd), ("PD_self", mu_self, sd_self)):
        base = padim_maps(Fp, mu_, sd_).reshape(len(P), -1).max(1)
        thr = float(np.percentile(base, 95))
        sweep[tag] = {"p95": thr}
        for amp in (0.05, 0.1, 0.2, 0.5, 1.0):
            Q = P.copy()
            for k, c in enumerate(cols):
                Q[k, :, c] *= (1.0 + amp)
            FQ = feat_of_patches(net, Q)
            sc_ = padim_maps(FQ, mu_, sd_).reshape(len(P), -1).max(1)
            sweep[tag][f"{amp:.2f}"] = {
                "det": float((sc_ > thr).mean()),
                "auroc": float(roc_auc_score(y, np.r_[sc_, base]))}
            v = sweep[tag][f"{amp:.2f}"]
            print(f"[{tag} inject +{amp*100:5.1f}% 1px] det={v['det']*100:5.1f}% "
                  f"(AUROC {v['auroc']:.3f})", flush=True)

    # ---------- side-by-side ----------
    res = {"config": {"family": "industrial AD (PatchCore/PaDiM), in-domain encoder "
                                 "features (block2+block3), bank=random 10k of clean train",
                      "device": DEV, "bank": BANK},
           "calib": {"n_clean": int(len(ci))},
           "patch": patch, "by_stratum": strata, "pixel": pix,
           "rsp1": {"n_patches": int(len(P)), "sweep": sweep}}
    try:
        two = json.load(open(os.path.join(D, "two_stage_detector.json"), encoding="utf-8"))
        res["comparison_patch"] = two["detectors"]
        res["comparison_strata"] = two["by_stratum"]
        res["comparison_pixel"] = two["pixel"]
        res["comparison_rsp1_sweep"] = two["rsp1"]["sweep_amplitude"]
        stb = json.load(open(os.path.join(D, "sumthreshold_baseline.json"), encoding="utf-8"))
        res["comparison_sumthreshold"] = {"patch": stb["patch"],
                                          "by_stratum": stb["by_stratum"],
                                          "pixel": stb["pixel"],
                                          "rsp1_sweep": stb["rsp1"]["sweep_amplitude"]}
    except Exception as e:
        print(f"[warn] comparison load: {e}", flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
