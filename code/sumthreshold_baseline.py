#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical SumThreshold baseline (Offringa et al. 2010, MNRAS 405, 155,
arXiv:1002.1957) on the SAME LOFAR test patches and the SAME RSP1 injection
protocol as two_stage_detector.py, so every number is directly comparable.

Pipeline (fixed before touching any test label):
  1. Surface fit in log power  L = log(max(X, 1e-6))  (same space as stage-1
     MRD; no upper clip so bright RFI saturates nothing).  The surface is the
     sliding LOCAL MEDIAN the paper itself recommends over sliding averages
     ("the median is insensitive to peaks and the surface ... remains smooth",
     sec. 2.3), computed with a 2-D median window (K,K), K in {9,17} - the
     max-over-scales convention mirrors stage-1 MRD's K in {5,9,17,33}.
     Two passes: pass 2 re-estimates the surface with pass-1 flagged pixels
     replaced by the pass-1 surface (the paper's iterative, flag-excluded fit).
  2. SumThreshold on the residual R = L - surface, "without surface fitting"
     applied to R itself (the variant analysed in their sec. 2.5.1/Fig. 3):
     one-dimensional in BOTH directions (along time per frequency channel,
     along frequency per time sample), two-sided, i.e. a window is flagged
     when |mean| > chi_M * sigma (their Fig. 3 convention: sum > chi*M*sigma,
     chi measured relative to the distribution mode - we remove the residual
     patch median, one scalar per patch, so a persistent line is NOT absorbed).
     Noise scale: one robust sigma per patch per direction (median over lines
     of 1.4826 * MAD(line)) - the paper thresholds against a sigma from the
     noise distribution and AOFlagger uses a global band scale, so per-line
     sigma would deviate from the published method.
  3. Window sizes M in {1,2,4,8,16,32,64} (exponentially spaced, up to the
     full patch line; Offringa 2013 technical report).  Average-threshold
     chi_M = chi1 / rho^(log2 M) with rho = 1.5 (their Eq. 12 and recommended
     value).  chi1 is calibrated on CLEAN TRAIN patches so the per-pixel
     false-flag rate equals 1e-3 (the paper's stated operating FPR).
  4. Continuous score u = max over passes/scales/directions/windows of
     |mean| * rho^(log2 M)  (chi1 = 1):  AUROC/AUPRC/best-F1 and the
     p95-based RSP1 operating points are invariant to chi1, so the comparison
     does not hinge on the calibration constant.
  5. The iterative "exclude already-flagged samples from larger-M sums"
     refinement of the 2013 technical report is NOT implemented (plain
     variant, as analysed in the 2010 paper's own false-positive section).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\sumthreshold_baseline.json
"""
import json, os, time
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score, average_precision_score

import two_stage_detector as tsd

D = tsd.D
PD = tsd.PD
PATCH = tsd.PATCH
OUTJ = os.path.join(D, "sumthreshold_baseline.json")

MS = (1, 2, 4, 8, 16, 32, 64)
KS = (9, 17)
RHO = 1.5
FPR_TARGET = 1e-3


def _patch_mode(R):
    """Remove the residual mode per patch (their Fig. 3 convention: the
    threshold is relative to the distribution mode).  A single scalar per
    patch, so a persistent line spanning a whole line is NOT absorbed."""
    return R - np.median(R, axis=(1, 2), keepdims=True)


def _line_sigma(R, axis):
    """One robust noise scale per patch per direction (the faithful choice:
    the paper thresholds against a sigma from the noise distribution, and
    AOFlagger uses a global band scale - NOT per-line scales).  mad is per
    line with the line index on axis 1 in both scan directions, so the
    robust cross-line median is always taken over axis 1."""
    mad = 1.4826 * np.median(np.abs(R - np.median(R, axis=axis, keepdims=True)),
                             axis=axis)                     # (n, n_lines)
    return np.median(mad, axis=1)[:, None, None] + 1e-6     # (n,1,1)


def _expand_max(u_win, M, L_axis_len, axis, H, W):
    """Scatter window scores to all M covered pixels (max-accumulate)."""
    n = u_win.shape[0]
    out = np.full((n, H, W), -np.inf, np.float32)
    Lwin = L_axis_len - M + 1
    for st in range(M):
        if axis == 1:
            np.maximum(out[:, st:st + Lwin, :], u_win, out=out[:, st:st + Lwin, :])
        else:
            np.maximum(out[:, :, st:st + Lwin], u_win, out=out[:, :, st:st + Lwin])
    return out


def _st_on_residual(R0):
    """SumThreshold u-map of a mode-removed residual, both directions."""
    n, H, W = R0.shape
    u_map = np.full((n, H, W), -np.inf, np.float32)
    for axis in (1, 2):
        sigma = _line_sigma(R0, axis)
        zshape = list(R0.shape); zshape[axis] = 1
        cs = np.concatenate([np.zeros(zshape, np.float64),
                             np.cumsum(R0, axis=axis, dtype=np.float64)], axis=axis)
        for M in MS:
            sl = cs.ndim * [slice(None)]
            sl[axis] = slice(M, None);  hi = cs[tuple(sl)]
            sl[axis] = slice(None, -M); lo = cs[tuple(sl)]
            mean = (hi - lo) / M
            u_win = np.abs(mean) * (RHO ** float(np.log2(M))) / sigma
            um = _expand_max(u_win.astype(np.float32), M, R0.shape[axis], axis, H, W)
            np.maximum(u_map, um, out=u_map)
    return u_map


def st_scores(X, want_map=False):
    """Full baseline: median-surface removal (2 passes) + SumThreshold."""
    L = np.log(np.maximum(X, 1e-6))
    u = np.full(L.shape, -np.inf, np.float32)
    for K in KS:
        surf = median_filter(L, size=(1, K, K), mode="reflect")
        R = _patch_mode(L - surf)
        u1 = _st_on_residual(R)
        np.maximum(u, u1, out=u)
        # pass 2: re-estimate the surface with pass-1 flagged pixels replaced
        flags = u1 > 1.0
        if flags.any():
            Lm = np.where(flags, surf, L)
            surf2 = median_filter(Lm, size=(1, K, K), mode="reflect")
            np.maximum(u, _st_on_residual(_patch_mode(L - surf2)), out=u)
    return (u.max(axis=(1, 2)), u) if want_map else u.max(axis=(1, 2))


def main():
    t00 = time.time()

    # ---------- clean-train calibration (same 4000 patches as the detector) --
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Xc = np.asarray(xtr[ci], np.float32)
    sc_clean, u_clean = st_scores(Xc, want_map=True)
    chi1 = float(np.quantile(u_clean, 1.0 - FPR_TARGET))
    cnt_clean = (u_clean > chi1).sum(axis=(1, 2))
    print(f"[calib] clean train n={len(Xc)}  STmax p50/p95="
          f"{np.percentile(sc_clean, [50, 95]).round(3)}  chi1(FPR={FPR_TARGET})="
          f"{chi1:.3f}  flags/patch p50/p95={np.percentile(cnt_clean, [50, 95]).round(1)}  "
          f"{time.time()-t00:.1f}s", flush=True)

    # ---------- LOFAR test, patch level ----------
    # Two patch scores: n_flags (classical decision at the calibrated chi1,
    # the primary baseline number) and max-u (deep-detector-style max stat).
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte)
    SCNT = np.zeros(n, np.float32); SMAX = np.zeros(n, np.float32)
    LAB = np.zeros(n, np.uint8); FR = np.zeros(n, np.float32)
    NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
    CH = 2000; t0 = time.time()
    for a in range(0, n, CH):
        b = min(a + CH, n)
        X = np.asarray(xte[a:b], np.float32)
        Y = np.asarray(yte[a:b]).astype(np.uint8)
        sm, um = st_scores(X, want_map=True)
        SMAX[a:b] = sm
        SCNT[a:b] = (um > chi1).sum(axis=(1, 2))
        LAB[a:b], FR[a:b], NP[a:b], NA[a:b] = tsd.morphology(Y)
        print(f"  patch-level {a}/{n} {time.time()-t0:.0f}s", flush=True)

    K = tsd.classify(NP, NA)
    patch = {}
    for name, sc in (("n_flags", SCNT), ("max_u", SMAX)):
        patch[name] = tsd.metrics(LAB, sc)
        print(f"[LOFAR patch] ST-{name:8s} AUROC={patch[name]['auroc']:.4f} "
              f"AUPRC={patch[name]['auprc']:.4f} F1={patch[name]['best_f1']:.4f}", flush=True)

    strata = {}
    for ki in (1, 2, 3):
        sel = (K == ki)
        if sel.sum() < 30:
            continue
        m = sel | (LAB == 0)
        strata[tsd.STRATA[ki]] = {
            "n": int(sel.sum()),
            "auroc": {name: float(roc_auc_score(LAB[m], sc[m]))
                      for name, sc in (("n_flags", SCNT), ("max_u", SMAX))}}
        print(f"[stratum n={sel.sum():5d}] {tsd.STRATA[ki]:42s} " +
              "  ".join(f"{k}={v:.3f}" for k, v in strata[tsd.STRATA[ki]]["auroc"].items()),
              flush=True)

    op = {"chi1": chi1, "fpr_target": FPR_TARGET,
          "clean_flags_per_patch_p50_p95": [float(np.percentile(cnt_clean, 50)),
                                            float(np.percentile(cnt_clean, 95))]}

    # ---------- pixel level, same subsample as the detector ----------
    rng = np.random.default_rng(1)
    idx_n = np.where(K == 2)[0]
    idx_o = rng.permutation(np.where(K != 2)[0])[:max(0, 8000 - len(idx_n))]
    pix_idx = np.sort(np.r_[idx_n, idx_o])
    print(f"[pixel] {len(pix_idx)} patches (narrow={len(idx_n)})", flush=True)
    ps, pt = [], []
    for a in range(0, len(pix_idx), 500):
        ii = pix_idx[a:a + 500]
        X = np.asarray(xte[ii], np.float32)
        Y = np.asarray(yte[ii]).astype(np.uint8)
        _, um = st_scores(X, want_map=True)
        ps.append(um.reshape(-1)); pt.append(Y.reshape(-1))
    ps, pt = np.concatenate(ps), np.concatenate(pt)
    pix = {"auroc": float(roc_auc_score(pt, ps)),
           "auprc": float(average_precision_score(pt, ps)),
           "rfi_pixel_fraction": float(pt.mean())}
    print(f"[LOFAR pixel] SumThreshold AUROC={pix['auroc']:.4f} AUPRC={pix['auprc']:.4f}", flush=True)

    # ---------- RSP1 injection, same protocol as run_rsp1 ----------
    S = tsd.rsp1_integrated()
    P, xs = tsd.rsp1_patches(S)
    bmax, bu = st_scores(P, want_map=True)
    bcnt = (bu > chi1).sum(axis=(1, 2)).astype(np.float32)
    thr_max = float(np.percentile(bmax, 95))
    thr_cnt = float(np.percentile(bcnt, 95))
    print(f"[rsp1] patches {P.shape}  ST max-u p95={thr_max:.2f}  "
          f"flags p95={thr_cnt:.1f}", flush=True)
    rng2 = np.random.default_rng(0)
    cols = rng2.integers(2, PATCH - 2, size=len(P))
    sweep = {}
    for amp in (0.05, 0.1, 0.2, 0.5, 1.0):
        Q = P.copy()
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        amax, au = st_scores(Q, want_map=True)
        acnt = (au > chi1).sum(axis=(1, 2)).astype(np.float32)
        y = np.r_[np.ones(len(P)), np.zeros(len(P))]
        sweep[f"{amp:.2f}"] = {
            "st_det_flags": float((acnt > thr_cnt).mean()),
            "st_auroc_flags": float(roc_auc_score(y, np.r_[acnt, bcnt])),
            "st_det_max": float((amax > thr_max).mean()),
            "st_auroc_max": float(roc_auc_score(y, np.r_[amax, bmax]))}
        v = sweep[f"{amp:.2f}"]
        print(f"[inject +{amp*100:5.1f}% 1px] ST-flags {v['st_det_flags']*100:5.1f}% "
              f"(AUROC {v['st_auroc_flags']:.3f})   ST-max {v['st_det_max']*100:5.1f}% "
              f"(AUROC {v['st_auroc_max']:.3f})", flush=True)

    # ---------- side-by-side with the two-stage detector ----------
    res = {"config": {"windows": list(MS), "surface_windows": list(KS), "rho": RHO,
                      "fpr_target": FPR_TARGET,
                      "variant": "local-median surface (2 passes, K in {9,17}) + plain "
                                 "SumThreshold, both directions, two-sided (Offringa+2010); "
                                 "patch scores = pixel flag count at calibrated chi1 "
                                 "(primary) and max-u (secondary)"},
           "calib": {"chi1": chi1,
                     "clean_maxu_p50": float(np.percentile(sc_clean, 50)),
                     "clean_maxu_p95": float(np.percentile(sc_clean, 95)),
                     "clean_flags_per_patch_p50": float(np.percentile(cnt_clean, 50)),
                     "clean_flags_per_patch_p95": float(np.percentile(cnt_clean, 95))},
           "patch": patch, "by_stratum": strata, "operating_point": op,
           "pixel": pix, "rsp1": {"n_patches": int(len(P)),
                                  "p95_max_u": thr_max, "p95_flags": thr_cnt,
                                  "sweep_amplitude": sweep}}
    try:
        two = json.load(open(os.path.join(D, "two_stage_detector.json"), encoding="utf-8"))
        res["comparison_patch"] = two["detectors"]
        res["comparison_strata"] = two["by_stratum"]
        res["comparison_pixel"] = two["pixel"]
        res["comparison_rsp1_sweep"] = two["rsp1"]["sweep_amplitude"]
    except Exception as e:
        print(f"[warn] could not load two_stage_detector.json: {e}", flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
