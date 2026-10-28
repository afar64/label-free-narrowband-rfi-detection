#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimisation round 2.

Lever A. Denoising-AE (DAE) stage-2: train the SAME ConvAE architecture on
    (clean + synthetic RFI) -> clean, so the encoder/decoder learns RFI
    morphology while staying label-free (synthetic injections need no
    annotation).  Injection morphologies during training: 1-3 px narrowband
    lines (both signs, amp 5-50%), 9-17 px wide bands (5-30%), 2-8 row
    transient bursts (20-100%) -- the same families as the evaluation strata.
    Score = top-16 mean of the smoothed (input-output)^2 map.
Lever B. Fisher member search on the selection half (54 images): members from
    {MRD, AE-top16, ST-flags, ST-max, DAE-top16}; the confirmation half (55
    images) is only scored once at the end for the final champion.

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\optimize_detector2.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\ae_denoise_weights.pt
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import sumthreshold_baseline as sb

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "optimize_detector2.json")
OUTW = os.path.join(D, "ae_denoise_weights.pt")
EPOCHS = 30


def synth_rfi(X, rng):
    """Add synthetic RFI to raw power patches (both signs, 3 morphologies)."""
    Q = X.copy()
    n = len(Q)
    # narrowband lines: 1-3 px, both signs, 5-50 %
    for k in range(n):
        if rng.random() < 0.7:
            w = int(rng.integers(1, 4))
            c = int(rng.integers(2, PATCH - 1 - w + 1))
            amp = float(rng.uniform(0.05, 0.5)) * (1 if rng.random() < 0.7 else -1)
            Q[k, :, c:c + w] *= 1.0 + amp
    # wide bands: 9-17 px, positive 5-30 %
    for k in range(n):
        if rng.random() < 0.3:
            w = int(rng.integers(9, 18))
            c = int(rng.integers(2, PATCH - 1 - w + 1))
            amp = float(rng.uniform(0.05, 0.30))
            Q[k, :, c:c + w] *= 1.0 + amp
    # transient bursts: 2-8 rows, 20-100 %
    for k in range(n):
        if rng.random() < 0.3:
            t0 = int(rng.integers(0, PATCH - 2))
            dur = int(rng.integers(2, 9))
            amp = float(rng.uniform(0.2, 1.0))
            Q[k, t0:t0 + dur, :] *= 1.0 + amp
    return Q


def topk_scores(net, X, bs=256):
    maps = np.empty((len(X), 64, 64), np.float32)
    with torch.no_grad():
        for a in range(0, len(X), bs):
            b = min(a + bs, len(X))
            xb = torch.from_numpy(tsd.ae_prep(np.asarray(X[a:b], np.float32)))
            e = (xb - net(xb)) ** 2
            maps[a:b] = F.avg_pool2d(e, 3, 1, 1)[:, 0].numpy()
    part = np.sort(maps.reshape(len(X), -1), axis=1)[:, -16:]
    return part.mean(1)


def pvals(s, ref_sorted):
    idx = np.searchsorted(ref_sorted, s, side="left")
    return np.clip(1.0 - idx / (len(ref_sorted) + 1.0), 1e-12, 1.0)


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    res = {}

    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()

    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    print(f"[dae] clean train patches {len(ci)}", flush=True)

    # ---------------- Lever A: train the denoising AE -----------------------
    dae = tsd.ConvAE(16, 128)
    opt = torch.optim.Adam(dae.parameters(), 1e-3)
    bs = 256
    g = np.random.default_rng(5)
    Xc_all = np.asarray(xtr[np.sort(ci)], np.float32)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = g.permutation(len(Xc_all))
        tot = 0.0
        dae.train()
        for a in range(0, len(Xc_all), bs):
            idx = perm[a:a + bs]
            Xb = Xc_all[idx]
            Qb = synth_rfi(Xb, g)
            xb = torch.from_numpy(tsd.ae_prep(Qb))
            tb = torch.from_numpy(tsd.ae_prep(Xb))
            opt.zero_grad()
            loss = ((dae(xb) - tb) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == EPOCHS - 1:
            print(f"[dae] ep{ep+1} mse={tot/len(Xc_all):.5f} ({time.time()-t0:.0f}s)",
                  flush=True)
    dae.eval()
    torch.save(dae.state_dict(), OUTW)

    # ---------------- scores -------------------------------------------------
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    z = np.load(os.path.join(D, "test_scores.npz"))
    LAB, K = z["LAB"], z["K"]
    n, per_img = len(LAB), int(z["per_img"])
    n_img = n // per_img
    m_sel = np.isin(np.arange(n) // per_img, np.arange(54))
    m_con = ~m_sel

    xall = np.asarray(xtr[ci[np.random.default_rng(0).permutation(len(ci))[:4000]]],
                      np.float32)                      # same 4000 protocol as all baselines
    DAEd = topk_scores(dae, xall)
    S1 = z["S1"]; S2 = z["S2"]; STF = z["ST_flags"]; STM = z["ST_max"]

    DAE = np.empty(n, np.float32)
    for a in range(0, n, 2000):
        b = min(a + 2000, n)
        DAE[a:b] = topk_scores(dae, np.asarray(xte[a:b], np.float32))
    print(f"[dae] test scores done ({time.time()-t0:.0f}s)", flush=True)

    def strata(mask, y, s):
        r = {"overall": float(roc_auc_score(y[mask], s[mask]))}
        for ki, nm in ((1, "transient"), (2, "narrow"), (3, "wide")):
            mm = ((K == ki) | (LAB == 0)) & mask
            r[nm] = float(roc_auc_score(y[mm], s[mm]))
        return r

    # members' clean references (label-free, all on the same 4000 patches)
    s1c = tsd.mrd_scores(xall)
    s2c = topk_scores(net, xall)
    stf_tr, stm_tr = [], []
    chi1 = json.load(open(os.path.join(D, "sumthreshold_baseline.json"),
                          encoding="utf-8"))["calib"]["chi1"]
    for a in range(0, len(xall), 1000):
        _, um = sb.st_scores(xall[a:a + 1000], want_map=True)
        stf_tr.append((um > chi1).sum(axis=(1, 2)))
        stm_tr.append(um.max(axis=(1, 2)))

    def mk_p(s, ref):
        return pvals(s, np.sort(np.asarray(ref, np.float32)))

    pm = {"MRD": mk_p(S1, s1c), "AE": mk_p(S2, s2c), "STF": mk_p(STF, np.concatenate(stf_tr)),
          "STM": mk_p(STM, np.concatenate(stm_tr)),
          "DAE": mk_p(DAE, np.sort(DAEd))}

    def fisher(names):
        return -2 * sum(np.log(pm[k]) for k in names)

    res["dae_alone"] = strata(m_sel, LAB, DAE)
    print("[dae alone] " + "  ".join(f"{a}={b:.4f}" for a, b in res["dae_alone"].items()),
          flush=True)

    members = {
        "fisher2 (MRD+AE)": ("MRD", "AE"),
        "fisher3 (MRD+AE+STF)": ("MRD", "AE", "STF"),
        "fisher3' (MRD+DAE+STF)": ("MRD", "DAE", "STF"),
        "fisher4 (+ST-max)": ("MRD", "AE", "STF", "STM"),
        "fisher5 (all)": ("MRD", "AE", "STF", "STM", "DAE"),
    }
    res["fusion"] = {}
    for k, names in members.items():
        s = fisher(names)
        res["fusion"][k] = strata(m_sel, LAB, s)
        print(f"[{k:24s}] " + "  ".join(f"{a}={b:.4f}" for a, b in res["fusion"][k].items()),
              flush=True)
    champ = max(res["fusion"], key=lambda k: res["fusion"][k]["overall"])
    print(f"[champion on selection] {champ}", flush=True)
    res["champion"] = champ

    # ---------------- confirmation half (scored ONCE, then locked) ----------
    conf = {}
    conf[champ] = strata(m_con, LAB, fisher(members[champ]))
    conf["fisher3 (MRD+AE+STF)"] = strata(m_con, LAB, fisher(members["fisher3 (MRD+AE+STF)"]))
    conf["ST-max(ref)"] = strata(m_con, LAB, STM)
    for k, v in conf.items():
        print(f"[confirm {k:24s}] " + "  ".join(f"{a}={b:.4f}" for a, b in v.items()),
              flush=True)
    res["confirmation"] = conf

    img_con = np.arange(n_img)[54:]
    B = 1000
    rng = np.random.default_rng(17)
    samp = rng.integers(0, len(img_con), size=(B, len(img_con)))
    boot = {}
    for k, names in ((champ, members[champ]), ("fisher3 (MRD+AE+STF)", members["fisher3 (MRD+AE+STF)"]),
                     ("ST_max", None)):
        s = STM if names is None else fisher(names)
        reps = np.empty(B)
        for bi in range(B):
            idx = np.concatenate([np.arange(im * per_img, (im + 1) * per_img)
                                  for im in img_con[samp[bi]]])
            reps[bi] = roc_auc_score(LAB[idx], s[idx])
        boot[k] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        print(f"[boot {k:24s}] CI[{boot[k][0]:.4f},{boot[k][1]:.4f}]", flush=True)
    res["bootstrap_confirm"] = boot

    res["config"] = {"EPOCHS": EPOCHS, "clean_patches": int(len(ci)),
                     "note": "STM p-reference uses its own test distribution "
                             "(documented limitation, label-free)"}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
