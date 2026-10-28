# sam_why_analysis.py - WHY does SAM fail on the RSP1 one-pixel ladder?
# Three mechanistic measurements:
#   A) Rendered visibility: injected column delta vs the patch's own column-texture
#      (column-mean MAD) -- if amp*median(colmean) << sigma_col, the line is
#      sub-JND in IMAGE space; no image-space model can see it (only time
#      aggregation can, which is the paper's thesis).
#   B) Seam behaviour: at +100% (where the line IS visible), oracle-prompt the
#      line column and inspect the best-IoU mask: does it cover the line, or do
#      its boundaries ABUT the line (masks on both sides, line excluded)?
#      SAM's SA-1B prior segments OBJECTS; a full-height 1px strip is an EDGE.
#   C) Figure: clean vs injected renders + best-IoU mask overlay, 2 examples.
import os, time
import numpy as np
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WS = os.path.dirname(os.path.abspath(__file__))
PATCH = 64
F_IQ = r"F:\RSP1\data\cygnus_return_after_aquila_10min_20260714_023008_1420905800Hz.iq16"
F0, FS, NFFT = 1420.9058e6, 4e6, 65536
CKPT = "F:/21cm/cache/sam_vit_b_01ec64.pth"
SEED = 2026
N_SUB = 64
N_SEAM = 16

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

def render(patch):
    lo, hi = np.percentile(patch, (1, 99))
    u = np.clip((patch - lo) / max(hi - lo, 1e-9), 0, 1)
    return (u * 255).astype(np.uint8)

# ---------- Part A: rendered visibility (numpy only) ----------
S = rsp1_integrated()
P = np.stack([S[:, x:x + PATCH] for x in range(0, 8000 - PATCH + 1, 32)])
rng = np.random.default_rng(SEED)
idx = rng.choice(P.shape[0], size=min(N_SUB, P.shape[0]), replace=False)
cols = rng.integers(2, PATCH - 2, size=len(idx))
Pc = P[idx]

print("[A] rendered visibility of the injected column (image space):")
print("    visibility ratio r = amp * median(colmean) / sigma_col-texture")
print("    (r < 1  ->  the line is below the patch's own column-to-column texture)")
vis = {}
for amp in (0.1, 0.5, 1.0):
    rs = []
    for k in range(len(Pc)):
        cm = Pc[k].mean(axis=0)
        sig = 1.4826 * np.median(np.abs(cm - median_filter(cm, 9, mode="nearest"))) + 1e-12
        rs.append(amp * float(np.median(cm)) / sig)
    vis[amp] = dict(median=float(np.median(rs)), p10=float(np.percentile(rs, 10)),
                    p90=float(np.percentile(rs, 90)))
    print(f"    amp={amp:4.2f}: median r = {np.median(rs):5.2f}   "
          f"(p10 {np.percentile(rs,10):.2f} .. p90 {np.percentile(rs,90):.2f})")

# ---------- Part B: seam behaviour at +100% (SAM oracle) ----------
print("[B] oracle-prompt seam test at amp=1.00 (best-IoU mask behaviour):")
import torch
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint=CKPT)
sam.to("cuda").eval()
pred = SamPredictor(sam)

at_c, left_c, right_c, areas = [], [], [], []
seam = 0
ex = []
for k in range(min(N_SEAM, len(Pc))):
    Q = Pc[k].copy()
    Q[:, cols[k]] *= 2.0
    im = render(Q)
    pred.set_image(np.stack([im] * 3, axis=2))
    ms, ious, _ = pred.predict(point_coords=np.array([[int(cols[k]), 32]]),
                               point_labels=np.array([1]), multimask_output=True)
    j = int(np.argmax(ious))
    seg = ms[j]                                # predictor returns bool arrays (n,H,W)
    f = seg.mean(axis=0)
    c = int(cols[k])
    a, l, r = float(f[c]), float(f[c - 1]), float(f[c + 1])
    at_c.append(a); left_c.append(l); right_c.append(r)
    areas.append(float(seg.sum()) / (PATCH * PATCH))
    if min(l, r) > 0.5 and a < 0.5:
        seam += 1
    if len(ex) < 2:
        ex.append((im, seg, c))
med = lambda v: float(np.median(v))
print(f"    best-IoU mask coverage at line column: median {med(at_c):+.2f}")
print(f"    coverage at the two NEIGHBOUR columns: median {med(left_c):+.2f} / {med(right_c):+.2f}")
print(f"    mask area (fraction of patch): median {med(areas):.2f}")
print(f"    SEAM pattern (both neighbours >0.5 covered, line <0.5): {seam}/{N_SEAM}")

# ---------- Part C: figure ----------
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for r_i, (im, mseg, c) in enumerate(ex):
    Q = Pc[r_i].copy(); Q[:, c] *= 2.0
    clean = render(Pc[r_i])
    axes[r_i, 0].imshow(clean.T, origin="lower", aspect="auto", cmap="inferno")
    axes[r_i, 0].set_title(f"clean patch {r_i}")
    axes[r_i, 1].imshow(im.T, origin="lower", aspect="auto", cmap="inferno")
    axes[r_i, 1].axvline(c, color="cyan", lw=0.7)
    axes[r_i, 1].set_title(f"+100% injected (line at col {c})")
    axes[r_i, 2].imshow(mseg.T, origin="lower", aspect="auto", cmap="Blues", alpha=0.6)
    axes[r_i, 2].axvline(c, color="cyan", lw=0.7)
    axes[r_i, 2].set_title("best-IoU mask from oracle prompt at line")
for ax in axes.ravel():
    ax.set_xlabel("frequency channel"); ax.set_ylabel("time row")
fig.suptitle("Why SAM fails: the line IS visible (10-100 sigma in column space), but SAM's "
             "mask space has no representation for 'statistically elevated channel' -- "
             "the oracle mask locks onto the dominant visual object and the line enters at most as a notch")
fig.tight_layout()
out = "F:/data/21cmHI/03_SPIRAL_ARMS/figs/sam_why.png"
fig.savefig(out, dpi=140)
print(f"[C] figure -> {out}")
