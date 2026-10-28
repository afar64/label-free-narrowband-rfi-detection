#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-level bootstrap CIs for every detector on the LOFAR test set.

Why image level: the 24525 test patches are 15x15 stride-32 crops of only
109 x 512x512 images, so neighbouring patches are strongly correlated; a
patch-level bootstrap would overstate the effective sample size.  We
resample the 109 IMAGES with replacement (B=1000) and recompute AUROC on the
pooled patches of the resampled images, for the overall table and each
morphology stratum (vs clean).

Adds PatchCore/PaDiM full-test scores here (the other detectors' scores come
from experiments_richer.py's test_scores.npz).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\bootstrap_ci.json
"""
import json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import patchcore_baseline as pcb

D = tsd.D
PD = tsd.PD
OUTJ = os.path.join(D, "bootstrap_ci.json")
B = 1000


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    z = np.load(os.path.join(D, "test_scores.npz"))
    S1, S2, FUS = z["S1"], z["S2"], z["FUS"]
    STF, STM, LAB, K = z["ST_flags"], z["ST_max"], z["LAB"], z["K"]
    n, per_img = len(S1), int(z["per_img"])
    n_img = n // per_img
    assert n_img * per_img == n, (n, per_img)
    img = np.arange(n) // per_img
    print(f"[load] n={n} images={n_img} per_img={per_img} {time.time()-t00:.1f}s",
          flush=True)

    # ---- PatchCore / PaDiM scores on the full test set --------------------
    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Xc = np.asarray(xtr[ci], np.float32)
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

    PC = np.empty(n, np.float32); PDs = np.empty(n, np.float32)
    t0 = time.time()
    for a in range(0, n, 2000):
        b = min(a + 2000, n)
        X = np.asarray(xte_slice(a, b), np.float32)
        PC[a:b] = pc_sc(X)
        PDs[a:b] = pd_sc(X)
        print(f"  PC/PD {a}/{n} {time.time()-t0:.0f}s", flush=True)

    DETS = {"MRD": S1, "AE": S2, "FUS": FUS, "ST-flags": STF, "ST-max": STM,
            "PatchCore": PC, "PaDiM": PDs}

    # ---- point estimates + image-level bootstrap --------------------------
    rng = np.random.default_rng(11)
    samples = rng.integers(0, n_img, size=(B, n_img))
    strata = {"overall": np.ones(n, bool),
              "transient": K == 1, "narrow": K == 2, "wide": K == 3}
    out = {}
    for name, s in DETS.items():
        out[name] = {}
        for st, m in strata.items():
            mm = m if st == "overall" else (m | (LAB == 0))
            pt = float(roc_auc_score(LAB[mm], s[mm]))
            reps = np.empty(B)
            for bi in range(B):
                idx = np.concatenate([np.arange(im * per_img, (im + 1) * per_img)
                                      for im in samples[bi]])
                mm2 = (m if st == "overall" else (m | (LAB == 0)))[idx]
                if 0 < LAB[idx][mm2].sum() < mm2.sum():
                    reps[bi] = roc_auc_score(LAB[idx][mm2], s[idx][mm2])
                else:
                    reps[bi] = np.nan
            reps = reps[~np.isnan(reps)]
            out[name][st] = {"auroc": pt,
                             "ci95": [float(np.percentile(reps, 2.5)),
                                      float(np.percentile(reps, 97.5))]}
            print(f"[bs] {name:10s} {st:9s} AUROC={pt:.4f} "
                  f"CI[{out[name][st]['ci95'][0]:.4f},{out[name][st]['ci95'][1]:.4f}]",
                  flush=True)

    res = {"config": {"B": B, "unit": "test image (n=109)", "rng": 11,
                      "strata_clean_pooled": True},
           "n_test": int(n), "n_images": int(n_img), "per_img": int(per_img),
           "bootstrap": out}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


def xte_slice(a, b):
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    return xte[a:b]


if __name__ == "__main__":
    main()
