# sam_arm_rsp1.py - Foundation-model arm: SAM ViT-B on the RSP1 injection ladder,
# SAME protocol as two_stage_detector.run_rsp1 (64x64 patches, one-column injection,
# AUROC injected-vs-clean, detection at the 5% false-alarm operating point).
#
# Scoring (blind to injection location): AMG masks -> column-coverage profiles
# f_m(x) = fraction of rows covered in column x; a DEDICATED line mask spikes at
# one column.  patch score = max_m [ max_x f_m(x) - median_x f_m(x) ]; 0 if no masks.
# A blob mask covering the image has a flat profile (score ~ 0); only a mask that
# isolates the injected channel spikes.
#
# Oracle control (location-aware, preempts the "prompt grid too coarse" objection):
# SamPredictor with a point prompt placed EXACTLY at the injected column; score =
# best-IoU mask's f_m[c_inj] - median_x f_m(x).  If even the oracle prompt fails to
# isolate the line, the AMG result is not a grid-density artifact.
import json, os, sys, time
import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

DEV = os.environ.get("SAM_DEVICE", "cpu")          # e.g. SAM_DEVICE=cuda for the GPU rerun
OUT_NAME = os.environ.get("SAM_OUT", "sam_arm_rsp1.json")

WS = os.path.dirname(os.path.abspath(__file__))
PATCH = 64
F_IQ = r"F:\RSP1\data\cygnus_return_after_aquila_10min_20260714_023008_1420905800Hz.iq16"
F0, FS, NFFT = 1420.9058e6, 4e6, 65536
CKPT = "F:/21cm/cache/sam_vit_b_01ec64.pth"
AMPS = (0.1, 0.5, 1.0)
N_SUB = 64
N_ORACLE = 32
SEED = 2026

def rsp1_integrated(nfr=8192, rows=64):
    per = nfr // rows
    wn = np.hanning(NFFT).astype(np.float32)
    acc = np.zeros((rows, 8000), np.float64)
    got = 0
    with open(F_IQ, "rb") as f:
        for r in range(rows):
            buf = np.frombuffer(f.read(per * NFFT * 4), dtype="<i2")
            n = buf.size // (2 * NFFT)
            if n == 0:
                break
            z = buf[:n * 2 * NFFT]
            iq = z[0::2].astype(np.float32) + 1j * z[1::2].astype(np.float32)
            S = np.empty((n, 8000), np.float32)
            for i in range(n):
                S[i] = (np.abs(np.fft.fftshift(np.fft.fft(iq[i * NFFT:(i + 1) * NFFT] * wn))) ** 2)[20000:28000]
            acc[r] = S.mean(0)
            got += n
    print(f"[rsp1] integrated {got} frames -> {rows} rows", flush=True)
    return acc.astype(np.float32)

def rsp1_patches(S):
    H, W = S.shape
    xs = list(range(0, W - PATCH + 1, 32))
    P = np.stack([S[:, x:x + PATCH] for x in xs])
    return P, xs

def mrd_profile(X, scales=(5, 9, 17, 33)):
    """Verbatim from two_stage_detector.mrd_profile (patch score only)."""
    L = np.log(np.maximum(X, 1e-6))
    N, H, F = L.shape
    zp = np.zeros(N, np.float32)
    for K in scales:
        R = L - median_filter(L, size=(1, 1, K), mode="reflect")
        d = np.median(R, axis=1)
        e = d - median_filter(d, size=(1, K), mode="reflect")
        sig_row = 1.4826 * np.median(np.abs(R - np.median(R, axis=2, keepdims=True)), axis=2)
        sigma = 1.2533 * np.median(sig_row, axis=1) / np.sqrt(H) + 1e-6
        z = e / sigma[:, None]
        m = np.abs(z).max(1)
        upd = m > zp
        zp[upd] = m[upd]
    return zp

def render(patch):
    lo, hi = np.percentile(patch, (1, 99))
    u = np.clip((patch - lo) / max(hi - lo, 1e-9), 0, 1)
    u = (u * 255).astype(np.uint8)
    return np.stack([u] * 3, axis=2)

def amg_score(masks):
    """Blind per-patch score: sharpest column spike over all masks."""
    best = 0.0
    for m in masks:
        seg = m["segmentation"]                    # (H, W) bool
        f = seg.mean(axis=0)                       # column coverage, (W,)
        if f.max() <= 0:
            continue
        best = max(best, float(f.max() - np.median(f)))
    return best

def run_amg(gen, imgs, tag):
    out = np.zeros(len(imgs), np.float32)
    nm = np.zeros(len(imgs), np.int32)
    thin = np.zeros(len(imgs), np.float32)
    t0 = time.time()
    for k, im in enumerate(imgs):
        ms = gen.generate(im)
        out[k] = amg_score(ms)
        nm[k] = len(ms)
        for m in ms:
            f = m["segmentation"].mean(axis=0)
            if f.max() <= 0:
                continue
            if int((f >= 0.5 * f.max()).sum()) <= 3:
                thin[k] += 1
                break
        if (k + 1) % 16 == 0:
            print(f"  [{tag} {k+1}/{len(imgs)}] {time.time()-t0:.0f}s elapsed", flush=True)
    print(f"[{tag}] done in {time.time()-t0:.0f}s; masks mean {nm.mean():.1f}, "
          f"thin-mask yield {thin.mean():.3f}", flush=True)
    return out, float(nm.mean()), float(thin.mean())

def main():
    t00 = time.time()
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    sam.to(DEV).eval()
    print(f"[device] {DEV} (torch {torch.__version__})", flush=True)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.5,
                                    stability_score_thresh=0.7, min_mask_region_area=2)
    pred = SamPredictor(sam)

    S = rsp1_integrated()
    P, xs = rsp1_patches(S)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P.shape[0], size=min(N_SUB, P.shape[0]), replace=False)
    cols = rng.integers(2, PATCH - 2, size=len(idx))
    print(f"[setup] {P.shape[0]} patches, subsample {len(idx)}, seed {SEED}", flush=True)

    Pc = P[idx]
    imgs_c = [render(p) for p in Pc]
    sc_clean, nm_c, thin_c = run_amg(gen, imgs_c, "clean")
    thr = float(np.percentile(sc_clean, 95))

    res = {"seed": SEED, "n_sub": int(len(idx)), "amps": list(AMPS),
           "sam": {"clean_masks_mean": nm_c, "clean_thin_yield": thin_c,
                   "thr_p95": thr, "per_amp": {}}}
    for amp in AMPS:
        Q = Pc.copy()
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        sc, nmean, thy = run_amg(gen, [render(q) for q in Q], f"amp{amp}")
        y = np.r_[np.ones(len(sc)), np.zeros(len(sc_clean))]
        auroc = float(roc_auc_score(y, np.r_[sc, sc_clean]))
        det = float((sc > thr).mean())
        res["sam"]["per_amp"][f"{amp:.2f}"] = {
            "auroc": auroc, "det_at_p95": det, "masks_mean": nmean, "thin_yield": thy}
        print(f"[sam amp={amp:.2f}] AUROC={auroc:.3f} det@p95={det*100:.1f}% "
              f"(clean thr {thr:.3f})", flush=True)

    # MRD on the SAME subsample, same injections (protocol reference)
    zc = mrd_profile(Pc)
    thr_m = float(np.percentile(zc, 95))
    res["mrd"] = {"thr_p95": thr_m, "per_amp": {}}
    for amp in AMPS:
        Q = Pc.copy()
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        zi = mrd_profile(Q)
        y = np.r_[np.ones(len(zi)), np.zeros(len(zc))]
        auroc = float(roc_auc_score(y, np.r_[zi, zc]))
        det = float((zi > thr_m).mean())
        res["mrd"]["per_amp"][f"{amp:.2f}"] = {"auroc": auroc, "det_at_p95": det}
        print(f"[mrd amp={amp:.2f}] AUROC={auroc:.3f} det@p95={det*100:.1f}%", flush=True)

    # Oracle point-prompt control at both ends (preempts the grid-density objection)
    res["oracle"] = {}
    for amp in (0.1, 1.0):
        cov, hit, n = [], 0, min(N_ORACLE, len(idx))
        t0 = time.time()
        for k in range(n):
            Q = Pc[k].copy()
            Q[:, cols[k]] *= (1.0 + amp)
            im = render(Q)
            pred.set_image(im)
            ms, ious, _ = pred.predict(point_coords=np.array([[int(cols[k]), 32]]),
                                       point_labels=np.array([1]), multimask_output=True)
            f = ms[int(np.argmax(ious))].mean(axis=0)
            o = float(f[cols[k]] - np.median(f))
            cov.append(o)
            hit += 1 if o > 0.3 else 0
            if (k + 1) % 16 == 0:
                print(f"  [oracle amp{amp} {k+1}/{n}] {time.time()-t0:.0f}s", flush=True)
        res["oracle"][f"{amp:.2f}"] = {"median_cov": float(np.median(cov)),
                                       "mean_cov": float(np.mean(cov)),
                                       "dedicated_strip_rate": hit / n, "n": int(n)}
        print(f"[oracle amp={amp:.2f}] median coverage-at-line {np.median(cov):+.3f}, "
              f"dedicated-strip rate {hit}/{n}", flush=True)

    out = os.path.join(WS, "results", OUT_NAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {out}", flush=True)
    print(f"[done] {time.time()-t00:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
