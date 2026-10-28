# sam_fix_readout.py - FIX the foundation-model arm: give SAM the aggregate-first
# representation the two-stage design prescribes, then measure whether it works.
#
# Mechanism recap (sam_why_analysis): the line is 10-100 sigma in column space, but
# in the RAW rendered patch it competes with dominant visual objects and SAM's mask
# space has no scalar path for "statistically elevated channel".
#
# Fix: render the column z-profile (the median-based stage's own output) as a bar
# image -- one separated bar per channel, height prop. to |z| on a FIXED scale
# (z=12 sigma = full height; fixed scale, no per-patch adaptation, no leakage).
# Pre-registered expectations:
#   E1: blind AMG narrow-bar readout AUROC at +10% >= 0.9 (raw-space arm was 0.568)
#   E2: SAM-readout <= max|z| ceiling on every amp (it adds no information)
#   E3: readout score requires NARROW masks (width<=3 cols): clean patches may have
#       zero narrow masks -> the existence test itself is discriminative
import json, os, time
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WS = os.path.dirname(os.path.abspath(__file__))
PATCH = 64
F_IQ = r"F:\RSP1\data\cygnus_return_after_aquila_10min_20260714_023008_1420905800Hz.iq16"
F0, FS, NFFT = 1420.9058e6, 4e6, 65536
CKPT = "F:/21cm/cache/sam_vit_b_01ec64.pth"
AMPS = (0.1, 0.5, 1.0)
N_SUB = 64
SEED = 2026
ZMAX = 12.0                     # fixed render scale: |z| = 12 sigma -> full-height bar
NARROW_W = 3                    # narrow-mask column-width cap

def rsp1_integrated(nfr=8192, rows=64):
    per = nfr // rows
    wn = np.hanning(NFFT).astype(np.float32)
    acc = np.zeros((rows, 8000), np.float64)
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
    return acc.astype(np.float32)

def z_profile(patch):
    """Column z-profile: aggregate over time, robust local normalisation.
    (The median-based stage's own first output; MAD robust to one injected column.)"""
    cm = patch.mean(axis=0)
    base = median_filter(cm, 9, mode="nearest")
    sig = 1.4826 * np.median(np.abs(cm - base)) + 1e-12
    return (cm - base) / sig

def bar_image(z):
    """|z| bar chart, 2 px pitch with 1 px gaps so bars stay separated objects.
    Fixed scale: |z| = ZMAX sigma -> full height."""
    W2 = PATCH * 2
    h = (np.clip(np.abs(z) / ZMAX, 0, 1) * (PATCH - 2)).astype(int)
    img = np.zeros((PATCH, W2), np.uint8)
    for x in range(PATCH):
        img[PATCH - h[x]:, 2 * x] = 255
    return np.stack([img] * 3, axis=2)

def readout_score(masks, z):
    """Blind: among NARROW masks (span <= NARROW_W channels at >=25% coverage),
    the largest |z| they isolate.  0 if no narrow mask.
    Bar image is 2x pitch: map image columns -> z channels via //2."""
    best = 0.0
    nw = 0
    for m in masks:
        seg = m["segmentation"]
        cols_img = np.flatnonzero(seg.mean(axis=0) >= 0.25)
        if len(cols_img) == 0:
            continue
        cols = np.unique(cols_img // 2)
        if 1 <= len(cols) <= NARROW_W:
            nw += 1
            best = max(best, float(np.abs(z[cols]).max()))
    return best, nw

def main():
    t00 = time.time()
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=CKPT)
    sam.to("cuda").eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.5,
                                    stability_score_thresh=0.7, min_mask_region_area=1)
    print("[device] cuda", flush=True)

    S = rsp1_integrated()
    P = np.stack([S[:, x:x + PATCH] for x in range(0, 8000 - PATCH + 1, 32)])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P.shape[0], size=min(N_SUB, P.shape[0]), replace=False)
    cols = rng.integers(2, PATCH - 2, size=len(idx))
    Pc = P[idx]

    def profiles(patches):
        return np.stack([z_profile(p) for p in patches])

    zc = profiles(Pc)
    ceiling_c = np.abs(zc).max(axis=1)
    print(f"[ref] clean max|z|: median {np.median(ceiling_c):.2f} "
          f"(p95 {np.percentile(ceiling_c,95):.2f})", flush=True)

    res = {"seed": SEED, "n_sub": int(len(idx)), "zmax_render": ZMAX,
           "amps": list(AMPS), "sam_readout": {"per_amp": {}},
           "ceiling_maxz": {"per_amp": {}}, "mrd_json": "sam_arm_rsp1.json"}

    imgs_clean = [bar_image(z) for z in zc]
    sc_clean, nw_clean = [], []
    for k, im in enumerate(imgs_clean):
        s, nw = readout_score(gen.generate(im), zc[k])
        sc_clean.append(s); nw_clean.append(nw)
        if (k + 1) % 16 == 0:
            print(f"  [clean {k+1}/{len(imgs_clean)}] {time.time()-t00:.0f}s", flush=True)
    sc_clean = np.array(sc_clean, np.float32)
    thr = float(np.percentile(sc_clean, 95))
    print(f"[clean] narrow-mask yield {np.mean(np.array(nw_clean)>0):.2f}, "
          f"readout median {np.median(sc_clean):.2f} sigma, p95 thr {thr:.2f}", flush=True)

    for amp in AMPS:
        Q = Pc.copy()
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        zi = profiles(Q)
        ceiling_i = np.abs(zi).max(axis=1)
        y_ref = np.r_[np.ones(len(zi)), np.zeros(len(zc))]
        a_ref = float(roc_auc_score(y_ref, np.r_[ceiling_i, ceiling_c]))
        sc_i, nw_i = [], []
        for k, im in enumerate([bar_image(z) for z in zi]):
            s, nw = readout_score(gen.generate(im), zi[k])
            sc_i.append(s); nw_i.append(nw)
            if (k + 1) % 16 == 0:
                print(f"  [amp{amp} {k+1}/{len(zi)}] {time.time()-t00:.0f}s", flush=True)
        sc_i = np.array(sc_i, np.float32)
        y = np.r_[np.ones(len(sc_i)), np.zeros(len(sc_clean))]
        a_sam = float(roc_auc_score(y, np.r_[sc_i, sc_clean]))
        det = float((sc_i > thr).mean())
        res["sam_readout"]["per_amp"][f"{amp:.2f}"] = {
            "auroc": a_sam, "det_at_p95": det,
            "narrow_yield": float((np.array(nw_i) > 0).mean()),
            "median_score": float(np.median(sc_i))}
        res["ceiling_maxz"]["per_amp"][f"{amp:.2f}"] = {"auroc": a_ref}
        print(f"[amp={amp:.2f}] SAM-readout AUROC={a_sam:.3f} det@p95={det*100:.1f}% "
              f"(narrow yield {np.mean(np.array(nw_i)>0):.2f})  |  max|z| ceiling AUROC={a_ref:.3f}",
              flush=True)

    out = os.path.join(WS, "results", "sam_fix_readout.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"[saved] {out}", flush=True)

    # small figure: clean vs injected bar images + masks of one example
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    Q = Pc[0].copy(); Q[:, cols[0]] *= 2.0
    z_inj = z_profile(Q)
    for ax, (z, tt) in zip(axes[:2], [(zc[0], "clean z-profile bars"),
                                      (z_inj, f"+100% (line col {int(cols[0])}, z={z_inj[cols[0]]:.1f})")]):
        ax.imshow(bar_image(z)[..., 0].T, origin="lower", cmap="inferno", aspect="auto")
        ax.set_title(tt, fontsize=9)
    ms = gen.generate(bar_image(z_inj))
    ov = bar_image(z_inj)[..., 0].T.copy()
    for m in ms:
        seg = m["segmentation"].T
        ov[seg] = np.maximum(ov[seg], 128)
    axes[2].imshow(ov, origin="lower", cmap="inferno", aspect="auto")
    axes[2].set_title("AMG masks on the bar image", fontsize=9)
    fig.suptitle("Fix: aggregate first, render the z-profile, let SAM read out the spike")
    fig.tight_layout()
    fig.savefig("F:/data/21cmHI/03_SPIRAL_ARMS/figs/sam_fix_readout.png", dpi=140)
    print(f"[done] {time.time()-t00:.0f}s", flush=True)

if __name__ == "__main__":
    main()
