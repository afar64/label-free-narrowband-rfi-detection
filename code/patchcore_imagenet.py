#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PatchCore with the PUBLISHED backbone (ImageNet WideResNet-50, layer2+3)
on the SAME LOFAR patches / RSP1 injections - the "pure replication" arm.

Purpose: quantify the DOMAIN GAP.  patchcore_baseline.py runs the same
memory-bank kNN with in-domain (clean-train ConvAE) features; this script
swaps in the exact backbone PatchCore was published with (WRN50 layer2+3,
patch-level 1-NN, Coreset-approximated memory bank).  Any difference between
the two arms is the ImageNet-to-spectrogram domain gap, measured.

Adaptation (minimal, fixed before test labels): log-power patch -> 3-channel
replication -> ImageNet mean/std; input kept at native 64x64 (no resize, to
preserve the scale at which a 1-px line lives); layer2 (512ch,16x16) and
layer3 (1024ch,8x8 upsampled) concatenated -> 256 patch vectors of dim 1536;
bank = random 10k of the clean-train patches (same rng(0) set).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\ad_imagenet.json
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score, average_precision_score

import two_stage_detector as tsd

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "ad_imagenet.json")
BANK = 10_000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class WRN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(w.conv1, w.bn1, w.relu)   # maxpool REMOVED:
        # at native 64x64 input it would halve layer2 to 8x8; dropping it keeps
        # 16x16 positions, the same geometry as the in-domain arm
        self.layer1, self.layer2, self.layer3 = w.layer1, w.layer2, w.layer3
        self.eval()

    def forward(self, x):                    # x: (n,1,64,64) in [0,1]-ish
        x = x.repeat(1, 3, 1, 1)
        x = (x - MEAN) / STD
        h = self.stem(x)                     # (n,64,32,32)
        h = self.layer1(h)                   # (n,256,32,32)
        f2 = self.layer2(h)                  # (n,512,16,16)
        f3 = self.layer3(f2)                 # (n,1024,8,8)
        f3u = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                            align_corners=False)
        return torch.cat([f2, f3u], 1)       # (n,1536,16,16)


def features(net, X, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(tsd.ae_prep(X[i:i + bs])).to(DEV)
            out.append(net(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def knn_max(Qft, bt, bn):
    """(n,C,16,16) -> (n,) patch scores (max over positions of 1-NN dist)."""
    n = Qft.shape[0]
    v = Qft.transpose(0, 2, 3, 1).reshape(-1, Qft.shape[1])
    out = np.empty(len(v), np.float32)
    q = torch.from_numpy(v).to(DEV)
    for a in range(0, len(q), 20_000):
        qc = q[a:a + 20_000]
        d2 = (qc ** 2).sum(1, keepdim=True) + bn - 2 * qc @ bt.T
        out[a:a + 20_000] = torch.clamp(d2.min(dim=1).values, min=0).sqrt().cpu().numpy()
    return out.reshape(n, 16, 16).max(axis=(1, 2))


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    print(f"[env] device={DEV} threads={os.cpu_count()}", flush=True)
    net = WRN().to(DEV)

    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Fc = features(net, np.asarray(xtr[ci], np.float32))
    Vc = Fc.transpose(0, 2, 3, 1).reshape(-1, Fc.shape[1])
    bank = Vc[np.random.default_rng(2).permutation(len(Vc))[:BANK]]

    # PCA 1536 -> 256 fitted on CLEAN-TRAIN features only (CPU cost cut for
    # the 1-NN search; the in-domain arm showed the verdict does not hinge on
    # feature dimensionality)
    Xt = torch.from_numpy(bank - Vc.mean(0))
    q = min(256, Xt.shape[1])
    Uc, Sc, Vt = torch.pca_lowrank(Xt, q=q, niter=4)
    P256 = Vt.contiguous()          # torch.pca_lowrank returns V as (N, q) = (1536,256)
    mu0 = torch.from_numpy(Vc.mean(0))
    bank256 = ((torch.from_numpy(bank) - mu0) @ P256).numpy()
    bt = torch.from_numpy(np.ascontiguousarray(bank256)).to(DEV)
    bn = (bt ** 2).sum(1)

    def proj(F):
        v = F.transpose(0, 2, 3, 1).reshape(-1, F.shape[1])
        z = ((torch.from_numpy(v) - mu0) @ P256).numpy()
        return z.reshape(F.shape[0], 16, 16, -1).transpose(0, 3, 1, 2).astype(np.float32)

    print(f"[calib] feats {Vc.shape} bank256 {bank256.shape} {time.time()-t00:.1f}s",
          flush=True)

    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    SI = np.zeros(n, np.float32)
    LAB = np.zeros(n, np.uint8); FR = np.zeros(n, np.float32)
    NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
    CH = 2000; t0 = time.time()
    for a in range(0, n, CH):
        b = min(a + CH, n)
        X = np.asarray(xte[a:b], np.float32)
        Y = np.asarray(yte[a:b]).astype(np.uint8)
        SI[a:b] = knn_max(proj(features(net, X)), bt, bn)
        LAB[a:b], FR[a:b], NP[a:b], NA[a:b] = tsd.morphology(Y)
        if a % 4000 == 0:
            print(f"  patch-level {a}/{n} {time.time()-t0:.0f}s", flush=True)

    K = tsd.classify(NP, NA)
    m_all = tsd.metrics(LAB, SI)
    print(f"[LOFAR patch] PC-ImageNet AUROC={m_all['auroc']:.4f} "
          f"AUPRC={m_all['auprc']:.4f} F1={m_all['best_f1']:.4f}", flush=True)
    strata = {}
    for ki in (1, 2, 3):
        sel = (K == ki)
        if sel.sum() < 30:
            continue
        m = sel | (LAB == 0)
        strata[tsd.STRATA[ki]] = {"n": int(sel.sum()),
                                  "auroc": float(roc_auc_score(LAB[m], SI[m]))}
        print(f"[stratum n={sel.sum():5d}] {tsd.STRATA[ki]:42s} "
              f"PC-ImageNet={strata[tsd.STRATA[ki]]['auroc']:.3f}", flush=True)

    rng = np.random.default_rng(1)
    # (pixel level skipped on the ImageNet arm: the domain-gap question is
    # answered by patch level + strata + injection; CPU budget goes there)

    # RSP1: self-referential bank (deployment-realistic; LOFAR bank is the
    # patchcore_baseline.py PC_lofar row)
    Sp = tsd.rsp1_integrated()
    P, xs = tsd.rsp1_patches(Sp)
    Fp = proj(features(net, P))
    Vs = Fp.transpose(0, 2, 3, 1).reshape(-1, Fp.shape[1])
    bt2 = torch.from_numpy(np.ascontiguousarray(Vs)).to(DEV); bn2 = (bt2 ** 2).sum(1)
    rng2 = np.random.default_rng(0)
    cols = rng2.integers(2, PATCH - 2, size=len(P))
    y = np.r_[np.ones(len(P)), np.zeros(len(P))]
    base = knn_max(Fp, bt2, bn2)
    thr = float(np.percentile(base, 95))
    sweep = {"p95": thr}
    for amp in (0.05, 0.1, 0.2, 0.5, 1.0):
        Q = P.copy()
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        sc_ = knn_max(proj(features(net, Q)), bt2, bn2)
        sweep[f"{amp:.2f}"] = {"det": float((sc_ > thr).mean()),
                               "auroc": float(roc_auc_score(y, np.r_[sc_, base]))}
        v = sweep[f"{amp:.2f}"]
        print(f"[PC-ImageNet self inject +{amp*100:5.1f}% 1px] det={v['det']*100:5.1f}% "
              f"(AUROC {v['auroc']:.3f})", flush=True)

    res = {"config": {"backbone": "wide_resnet50_2 IMAGENET1K_V2, layer2+layer3, "
                                 "native 64x64 input, maxpool removed, 3-channel "
                                 "replicate, ImageNet norm, PCA 1536->256 (clean-fit)",
                      "bank": BANK, "device": DEV},
           "patch": m_all, "by_stratum": strata, "rsp1": sweep,
           "pixel": "skipped on this arm (CPU budget; see ad_baseline.json)"}
    try:
        ind = json.load(open(os.path.join(D, "ad_baseline.json"), encoding="utf-8"))
        res["comparison_in_domain"] = {"patch": ind["patch"],
                                       "by_stratum": ind["by_stratum"],
                                       "pixel": ind["pixel"],
                                       "rsp1": ind["rsp1"]}
    except Exception as e:
        print(f"[warn] {e}", flush=True)
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
