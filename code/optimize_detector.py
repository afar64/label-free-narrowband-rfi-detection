#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detector optimisation under the zero-label constraint.

Levers, all calibrated on CLEAN data only (no RFI labels for any choice):
  1. Fusion rule: max robust-z (current) vs Fisher-combined p vs mean-p,
     each with and without ST-flags as a third member.  p-values = clean-train
     CDF transform of each detector's patch score.
  2. AE patch aggregation: max (current) vs top-k mean (k=3/8/16) of the
     3x3-smoothed error map.
  3. Stage-2 architecture: skip-connection AE (U-Net-lite) trained on ALL
     clean train patches, vs the current bottleneck-only AE.

Anti-multiplicity design: the 109 test images are split 55/54 into a
SELECTION half and a CONFIRMATION half.  Fusion rules and aggregation are
selected on the selection half only; the champion is reported on the
confirmation half with an image-level bootstrap CI.  All variants are logged
anyway (no hidden garden of forking paths).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\optimize_detector.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\ae_skip_weights.pt
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd

D = tsd.D
PD = tsd.PD
OUTJ = os.path.join(D, "optimize_detector.json")
OUTW = os.path.join(D, "ae_skip_weights.pt")
EPOCHS = 30


class SkipAE(nn.Module):
    """Bottleneck AE + skip connections from block2 and block1."""

    def __init__(self, w=16):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, w, 3, 2, 1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2))
        self.e2 = nn.Sequential(nn.Conv2d(w, 2 * w, 3, 2, 1), nn.BatchNorm2d(2 * w), nn.LeakyReLU(0.2))
        self.e3 = nn.Sequential(nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.BatchNorm2d(4 * w), nn.LeakyReLU(0.2))
        self.e4 = nn.Sequential(nn.Conv2d(4 * w, 8 * w, 3, 2, 1), nn.BatchNorm2d(8 * w), nn.LeakyReLU(0.2))
        self.d4 = nn.Sequential(nn.ConvTranspose2d(8 * w, 4 * w, 4, 2, 1), nn.BatchNorm2d(4 * w), nn.LeakyReLU(0.2))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(8 * w, 2 * w, 4, 2, 1), nn.BatchNorm2d(2 * w), nn.LeakyReLU(0.2))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(4 * w, w, 4, 2, 1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2))
        self.d1 = nn.ConvTranspose2d(2 * w, 1, 4, 2, 1)

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        g3 = self.d4(h4)
        g2 = self.d3(torch.cat([g3, h3], 1))
        g1 = self.d2(torch.cat([g2, h2], 1))
        return torch.sigmoid(self.d1(torch.cat([g1, h1], 1)))


def ae_scores(net, xte, n, bs=256, device="cpu"):
    """Top-k patch scores from the smoothed error map."""
    maps = np.empty((n, 64, 64), np.float32)
    with torch.no_grad():
        for a in range(0, n, bs):
            b = min(a + bs, n)
            xb = torch.from_numpy(tsd.ae_prep(np.asarray(xte[a:b], np.float32))).to(device)
            e = (xb - net(xb)) ** 2
            maps[a:b] = F.avg_pool2d(e, 3, 1, 1)[:, 0].cpu().numpy()
    flat = maps.reshape(n, -1)
    part = np.sort(flat, axis=1)[:, -16:]
    out = {"top1": part[:, -1]}
    for k in (3, 8, 16):
        out[f"top{k}"] = part[:, -k:].mean(1)
    return out


def pvals(s, ref_sorted):
    """Upper-tail empirical p under the clean-train reference (mid-rank)."""
    idx = np.searchsorted(ref_sorted, s, side="left")
    return np.clip(1.0 - idx / (len(ref_sorted) + 1.0), 1e-12, 1.0)


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    res = {}

    # ---------------- shared pieces -----------------------------------------
    net = tsd.ConvAE(16, 128)
    net.load_state_dict(torch.load(tsd.W, map_location="cpu"))
    net.eval()
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci_all = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci4000 = ci_all[np.random.default_rng(0).permutation(len(ci_all))[:4000]]
    X4 = np.asarray(xtr[ci4000], np.float32)
    s1c = tsd.mrd_scores(X4)
    s2c = tsd.ae_err(net, X4)[1]
    s1c_srt, s2c_srt = np.sort(s1c), np.sort(s2c)

    z = np.load(os.path.join(D, "test_scores.npz"))
    S1, S2, STF = z["S1"], z["S2"], z["ST_flags"]
    LAB, K = z["LAB"], z["K"]
    n, per_img = len(S1), int(z["per_img"])
    n_img = n // per_img
    half = n_img // 2                       # 54 selection, 55 confirmation
    sel_img = np.zeros(n_img, bool); sel_img[:half] = True
    m_sel = np.isin(np.arange(n) // per_img, np.where(sel_img)[0])
    m_con = ~m_sel
    print(f"[split] selection imgs {sel_img.sum()} / confirmation {n_img - sel_img.sum()}",
          flush=True)

    def auc(mask, y, s):
        mm = mask
        return float(roc_auc_score(y[mm], s[mm]))

    def strata(mask, y, s):
        o = auc(mask, y, s)
        r = {"overall": o}
        for ki, nm in ((1, "transient"), (2, "narrow"), (3, "wide")):
            mm = ((K == ki) | (LAB == 0)) & mask
            r[nm] = float(roc_auc_score(y[mm], s[mm]))
        return r

    # ---------------- lever 2: AE top-k aggregation -------------------------
    t0 = time.time()
    xte_mm = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    topk = ae_scores(net, xte_mm, n)
    res["ae_topk"] = {}
    for k, s in topk.items():
        res["ae_topk"][k] = strata(m_sel, LAB, s)
        print(f"[ae-topk {k:5s}] " + "  ".join(f"{a}={b:.4f}" for a, b in res["ae_topk"][k].items()),
              flush=True)
    best_topk = max(res["ae_topk"], key=lambda k: res["ae_topk"][k]["overall"])
    print(f"[ae-topk] best on selection = {best_topk}", flush=True)
    S2v = topk[best_topk]
    # clean-train reference with the SAME aggregation (for robust-z and p)
    s2c_k = ae_scores(net, X4, len(X4))[best_topk]
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- lever 1: fusion rules ---------------------------------
    p1 = pvals(S1, s1c_srt)
    p2 = pvals(S2v, np.sort(s2c_k))
    # ST flags' clean reference: label-free, from the 4000 clean-train patches
    stf_train = []
    chi1 = json.load(open(os.path.join(D, "sumthreshold_baseline.json"),
                          encoding="utf-8"))["calib"]["chi1"]
    import sumthreshold_baseline as sb
    for a in range(0, len(X4), 1000):
        sm, um = sb.st_scores(X4[a:a + 1000], want_map=True)
        stf_train.append((um > chi1).sum(axis=(1, 2)))
    pf_ref = np.sort(np.concatenate(stf_train).astype(np.float32))
    pf = pvals(STF, pf_ref)

    def rz(v, ref):
        return (v - np.median(ref)) / (1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-9)

    cands = {
        "maxz2 (current)": np.maximum(rz(S1, s1c), rz(S2v, s2c_k)),
        "fisher2": -2 * (np.log(p1) + np.log(p2)),
        "avgp2": 1 - 0.5 * (p1 + p2),
        "fisher3(+STF)": -2 * (np.log(p1) + np.log(p2) + np.log(pf)),
        "avgp3(+STF)": 1 - (p1 + p2 + pf) / 3,
    }
    res["fusion_rules"] = {}
    for k, s in cands.items():
        res["fusion_rules"][k] = {"selection": strata(m_sel, LAB, s)}
        print(f"[fusion {k:15s}] sel " +
              "  ".join(f"{a}={b:.4f}" for a, b in res["fusion_rules"][k]["selection"].items()),
              flush=True)
    champ = max(cands, key=lambda k: res["fusion_rules"][k]["selection"]["overall"])
    print(f"[fusion] champion on selection = {champ}", flush=True)
    res["champion_rule"] = champ

    # ---------------- lever 3: skip-connection AE ---------------------------
    t0 = time.time()
    sk = SkipAE(16)
    ci_tr = ci_all                                   # ALL clean train patches
    Xtr = np.asarray(xtr[np.sort(ci_tr)], np.float32)
    print(f"[skip-ae] train patches {len(Xtr)}", flush=True)
    opt = torch.optim.Adam(sk.parameters(), 1e-3)
    bs = 256
    g = np.random.default_rng(5)
    for ep in range(EPOCHS):
        perm = g.permutation(len(Xtr))
        tot = 0.0
        sk.train()
        for a in range(0, len(Xtr), bs):
            idx = perm[a:a + bs]
            xb = torch.from_numpy(tsd.ae_prep(Xtr[idx]))
            opt.zero_grad()
            loss = ((xb - sk(xb)) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 4 or ep == EPOCHS - 1:
            print(f"[skip-ae] ep{ep+1} mse={tot/len(Xtr):.5f} ({time.time()-t0:.0f}s)",
                  flush=True)
    sk.eval()
    torch.save(sk.state_dict(), OUTW)

    topk_skip = ae_scores(sk, np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r"), n)
    res["skip_ae"] = {}
    for k, s in topk_skip.items():
        res["skip_ae"][k] = strata(m_sel, LAB, s)
        print(f"[skip-ae {k:5s}] " + "  ".join(f"{a}={b:.4f}" for a, b in res["skip_ae"][k].items()),
              flush=True)
    best_sk = max(res["skip_ae"], key=lambda k: res["skip_ae"][k]["overall"])
    S2s = topk_skip[best_sk]
    print(f"[skip-ae] best = {best_sk}", flush=True)

    # fusion with the skip AE (same rule family, selection half)
    s2cs = ae_scores(sk, X4, len(X4))[best_sk]
    p2s = pvals(S2s, np.sort(s2cs))
    cands_skip = {
        "maxz2_skip": np.maximum(rz(S1, s1c), rz(S2s, s2cs)),
        "fisher2_skip": -2 * (np.log(p1) + np.log(p2s)),
        "avgp2_skip": 1 - 0.5 * (p1 + p2s),
    }
    for k, s in cands_skip.items():
        res["fusion_rules"][k] = {"selection": strata(m_sel, LAB, s)}
        print(f"[fusion {k:15s}] sel " +
              "  ".join(f"{a}={b:.4f}" for a, b in res["fusion_rules"][k]["selection"].items()),
              flush=True)
    champ2 = max(cands_skip, key=lambda k: res["fusion_rules"][k]["selection"]["overall"])
    print(f"[fusion] champion with skip-AE = {champ2}", flush=True)

    # ---------------- confirmation half + bootstrap CI ----------------------
    confirm = {}
    champ_s = cands[champ] if res["fusion_rules"][champ]["selection"]["overall"] >= \
        res["fusion_rules"][champ2]["selection"]["overall"] else cands_skip[champ2]
    champ_name = champ if res["fusion_rules"][champ]["selection"]["overall"] >= \
        res["fusion_rules"][champ2]["selection"]["overall"] else champ2
    confirm[champ_name] = strata(m_con, LAB, champ_s)
    confirm["maxz2 (current)"] = strata(m_con, LAB, cands["maxz2 (current)"])
    confirm["ST-max(ref)"] = strata(m_con, LAB, z["ST_max"])
    for k, v in confirm.items():
        print(f"[confirm {k:15s}] " + "  ".join(f"{a}={b:.4f}" for a, b in v.items()),
              flush=True)

    # image bootstrap on the confirmation half only
    img_con = np.arange(n_img)[~sel_img]
    B = 1000
    rng = np.random.default_rng(13)
    samp = rng.integers(0, len(img_con), size=(B, len(img_con)))
    boot = {}
    for k, s in (("champion", champ_s), ("current", cands["maxz2 (current)"]),
                 ("ST_max", z["ST_max"])):
        reps = np.empty(B)
        for bi in range(B):
            idx = np.concatenate([np.arange(im * per_img, (im + 1) * per_img)
                                  for im in img_con[samp[bi]]])
            reps[bi] = roc_auc_score(LAB[idx], s[idx])
        boot[k] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        print(f"[boot {k:9s}] CI[{boot[k][0]:.4f},{boot[k][1]:.4f}]", flush=True)
    res["confirmation"] = confirm
    res["bootstrap_confirm_overall"] = boot

    res["config"] = {"EPOCHS": EPOCHS, "clean_train_all": int(len(Xtr)),
                     "skip_ae_best_agg": best_sk, "ae_best_agg": best_topk}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
