#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage RFI detector: 1-D median-residual (narrowband) + 2-D AE (extended).

Motivation (measured, not assumed):
    The 4-level strided ConvAE cannot see a 1-pixel-wide line at any contrast
    (AUROC 0.500 -> 0.626 for +10% -> +500%); it responds to feature *area*,
    because 4x stride-2 pooling averages a 1-px line away at the first level.
    L-band RFI and low-cost SDR artifacts are both 1 channel wide.

Stage 1 (MRD, median-residual detector), per patch:
    R[t,c] = logP - median_K(logP along frequency)      remove bandpass + gain
    d[c]   = median_t R[t,c]                            narrowband survives,
                                                        broadband bursts removed
    e[c]   = d[c] - median_K(d)                          remove residual curvature
    sigma  = 1.2533 * median_t(1.4826*MAD_c R) / sqrt(H)
    z[c]   = e[c] / sigma                               in noise units
    score  = max_c |z[c]|, max over K in {5,9,17,33}

Stage 2: clean-only ConvAE trained on LOFAR (lofar_ae_weights_final.pt),
         patch score = max of 3x3-smoothed squared reconstruction error.

Fusion: max of the two detectors' robust z-scores, calibrated on CLEAN TRAIN
patches only (no test labels used).

Outputs
    F:\\data\\21cmHI\\06_RFI_LABELED\\two_stage_detector.json
    F:\\data\\21cmHI\\06_RFI_LABELED\\two_stage_detector.png
"""
import json, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

D = r"F:\data\21cmHI\06_RFI_LABELED"
PD = os.path.join(D, "lofar_patches")
W = os.path.join(D, "lofar_ae_weights_final.pt")
OUTJ = os.path.join(D, "two_stage_detector.json")
OUTP = os.path.join(D, "two_stage_detector.png")

CLIP_LO, CLIP_HI = 0.0, 4217294.69          # from lofar_ae_metrics_final.json
LOGMIN, LOGMAX = -13.815510557964274, 15.2547044117194

PATCH = 64
F_IQ = r"F:\RSP1\data\cygnus_return_after_aquila_10min_20260714_023008_1420905800Hz.iq16"
F0, FS, NFFT = 1420.9058e6, 4e6, 65536
COMB0, COMB_DF = 1420.40702e6, 0.03314e6


class ConvAE(nn.Module):
    def __init__(self, width=16, latent=128):
        super().__init__()
        w = width
        self.c8 = w * 8
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 3, 2, 1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2),
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(w * 2, w * 4, 3, 2, 1), nn.BatchNorm2d(w * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(w * 4, w * 8, 3, 2, 1), nn.BatchNorm2d(w * 8), nn.LeakyReLU(0.2))
        self.fc = nn.Linear(w * 8 * 4 * 4, latent)
        self.fc2 = nn.Linear(latent, w * 8 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w * 8, w * 4, 4, 2, 1), nn.BatchNorm2d(w * 4), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(w * 4, w * 2, 4, 2, 1), nn.BatchNorm2d(w * 2), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(w * 2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(w, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.fc(self.enc(x).flatten(1))
        return self.dec(self.fc2(z).view(-1, self.c8, 4, 4))


# ---------------------------------------------------------------- stage 1
def mrd_profile(X, scales=(5, 9, 17, 33)):
    """Return (patch_score, pixel_z) for a batch.

    patch_score: (N,)
    pixel_z    : (N,H,F) same value across time for a given column (column stat)

    No dataset-specific constants: only a floor at 1e-6 (log) and the scale
    set {5,9,17,33}.  An upper clip would saturate bright RFI and, on data with
    a different power scale, flatten the whole spectrum (that bug hid the RSP1
    comb entirely in the first run).
    """
    L = np.log(np.maximum(X, 1e-6))
    N, H, F = L.shape
    zp = np.zeros(N, np.float32)
    zbest = np.zeros((N, F), np.float32)
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
        zbest[upd] = z[upd]
    pix = np.broadcast_to(np.abs(zbest)[:, None, :], (N, H, F)).astype(np.float32)
    return zp, pix


def mrd_scores(X, scales=(5, 9, 17, 33)):
    return mrd_profile(X, scales)[0]


# ---------------------------------------------------------------- stage 2
def ae_prep(X):
    A = np.log(np.clip(X, 1e-6, CLIP_HI))
    A = np.clip((A - LOGMIN) / (LOGMAX - LOGMIN + 1e-9), 0, 1)
    med = np.median(A, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(A - med), axis=(1, 2), keepdims=True) + 1e-6
    A = np.clip((A - med) / mad, -5, 5) / 10 + 0.5
    return A[:, None].astype(np.float32)


def ae_err(net, X, bs=256):
    """Return pixel error map (N,H,F) and patch score (N,)."""
    P, S = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(ae_prep(X[i:i + bs]))
            e = (xb - net(xb)) ** 2
            e = nn.functional.avg_pool2d(e, 3, 1, 1)
            P.append(e[:, 0].numpy())
            S.append(e.flatten(2).amax(dim=2)[:, 0].numpy())
    return np.concatenate(P), np.concatenate(S)


def best_f1(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    P, N = float(y.sum()), float(len(y) - y.sum())
    f1 = 2 * (tpr * P) / np.maximum(2 * tpr * P + fpr * N + (P - tpr * P), 1e-9)
    k = int(np.argmax(f1))
    return float(f1[k]), float(thr[k])


def metrics(y, s):
    f1, thr = best_f1(y, s)
    return dict(auroc=float(roc_auc_score(y, s)),
                auprc=float(average_precision_score(y, s)),
                best_f1=f1, thr=thr)


def rz(v, ref):
    med = float(np.median(ref)); mad = float(np.median(np.abs(ref - med))) + 1e-9
    return (v - med) / (1.4826 * mad)


def morphology(Y):
    """Per-patch RFI morphology from the expert mask. Y: (N,H,F) uint8."""
    flat = Y.reshape(len(Y), -1)
    lab = flat.any(1)
    frac = flat.mean(1)
    cf = Y.mean(axis=1)                       # (N,F) fraction of time flagged
    pers = cf > 0.6
    npers = pers.sum(1)
    tot = flat.sum(1) + 1e-9
    narrow = (Y * pers[:, None, :]).reshape(len(Y), -1).sum(1) / tot
    return lab.astype(np.uint8), frac, npers.astype(np.int16), narrow.astype(np.float32)


def classify(npers, narrow):
    """0 clean, 1 transient/partial, 2 persistent narrow (1-3 cols), 3 multi-column."""
    k = np.zeros(len(npers), np.int8)
    k[(npers == 0)] = 1
    k[(npers >= 1) & (npers <= 3) & (narrow > 0.6)] = 2
    k[(npers >= 4)] = 3
    return k


STRATA = {1: "transient/partial (no persistent column)",
          2: "persistent narrow (1-3 columns)",
          3: "multi-column / wide (>=4 columns)"}


# ---------------------------------------------------------------- RSP1
def rsp1_data_unused(nfr=400):
    raw = np.fromfile(F_IQ, dtype="<i2", count=NFFT * nfr * 2)
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    nf = iq.size // NFFT
    wn = np.hanning(NFFT).astype(np.float32)
    S = np.empty((nf, 8000), np.float32)
    for i in range(nf):
        S[i] = (np.abs(np.fft.fftshift(np.fft.fft(iq[i * NFFT:(i + 1) * NFFT] * wn))) ** 2)[20000:28000]
    P = np.stack([S[y:y + PATCH, x:x + PATCH]
                  for y in range(0, nf - PATCH + 1, 32)
                  for x in range(0, 8000 - PATCH + 1, 32)])
    freq = F0 + (np.arange(20000, 28000) - NFFT // 2) * FS / NFFT
    return P, freq


def rsp1_integrated(nfr=8192, rows=64):
    """Read the IQ file and return a (rows, 8000) mean-power spectrogram.

    nfr // rows FFT frames are averaged into each row, so a row is `per` times
    deeper than a single 65536-point FFT.  This is the same depth class as the
    475-s integrated spectrum used in the published analysis.
    """
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
    print(f"[rsp1] integrated {got} frames -> {rows} rows of {per} FFTs", flush=True)
    return acc.astype(np.float32)


def rsp1_patches(S):
    """All 64-column blocks of the 64x8000 spectrogram, stride 32."""
    H, W = S.shape
    xs = list(range(0, W - PATCH + 1, 32))
    P = np.stack([S[:, x:x + PATCH] for x in xs])
    return P, xs


def spec_z(spec, K=9):
    """1-D MRD z-profile of a single spectrum (used to locate the comb)."""
    L = np.log(np.maximum(spec, 1e-6))
    R = L - median_filter(L, size=K, mode="reflect")
    e = R - median_filter(R, size=K, mode="reflect")
    sig = 1.4826 * np.median(np.abs(R - median_filter(R, size=3, mode="reflect"))) + 1e-12
    return e / sig, R, sig



def comb_lattice(zsp, spec, lmin=100, lmax=1600, nmc=400, seed=0):
    """Detect a periodic comb in a 1-D detector profile by autocorrelation.

    Autocorrelation uses every bin, so it is far more sensitive than clustering
    the few peaks that pass a threshold.  The statistic is max over lags, so the
    p-value is Monte-Carlo (shuffle zsp, keep its distribution).
    """
    z0 = zsp - zsp.mean()
    var = float(z0.var()) + 1e-12
    lags = np.arange(lmin, lmax + 1)
    ac = np.array([float((z0[:-L] * z0[L:]).mean()) / var for L in lags])
    k = int(np.argmax(ac))
    Lb, acmax = float(lags[k]), float(ac[k])

    rng = np.random.default_rng(seed)
    null = np.empty(nmc)
    for m in range(nmc):
        q = rng.permutation(z0)
        null[m] = max(float((q[:-L] * q[L:]).mean()) / var for L in lags)
    pval = float((null >= acmax).mean())

    # phase of the comb and its contrast in the linear spectrum
    pk = np.array([i for i in range(3, len(zsp) - 3)
                   if zsp[i] > 5 and zsp[i] == zsp[i - 3:i + 4].max()], float)
    base = median_filter(spec, size=31, mode="reflect")
    ratio = spec / np.maximum(base, 1e-12)
    on = np.zeros(len(spec), bool)
    if len(pk) > 3:
        ph = float(np.angle(np.exp(1j * 2 * np.pi * pk / Lb).mean()) % (2 * np.pi))
        phase0 = ph / (2 * np.pi) * Lb
        for k2 in range(-20, 40):
            j = int(round(phase0 + k2 * Lb))
            if 1 <= j < len(spec) - 1:
                on[j - 1:j + 2] = True
        dphase = np.abs(((pk - phase0) / Lb - np.round((pk - phase0) / Lb)))
        non = int((dphase < 0.06).sum())
    else:
        phase0, non = 0.0, 0
    return {"n_peaks_z5": int(len(pk)),
            "spacing_bins": Lb, "spacing_kHz": float(Lb * FS / NFFT / 1e3),
            "spacing_km_s": float(Lb * FS / NFFT / 1420.40575177e6 * 299792.458),
            "spacing_ppm": float(Lb * FS / NFFT / 1420.40575177e6 * 1e6),
            "autocorr": acmax, "p_mc": pval,
            "null_ac_p95": float(np.percentile(null, 95)),
            "contrast_on_lattice": float(np.median(ratio[on]) - 1) if on.any() else 0.0,
            "contrast_off_lattice": float(np.median(ratio[~on]) - 1) if (~on).any() else 0.0,
            "peaks_on_lattice": non}


def run_rsp1(net):
    """Real comb + injection test on the RSP1 IQ file, both detectors on the
    SAME 64x64 patches so the comparison is direct."""
    S = rsp1_integrated()
    P, xs = rsp1_patches(S)
    print(f"[rsp1] patches {P.shape}", flush=True)

    z1 = mrd_scores(P)
    sc2 = ae_err(net, P)[1]
    thr1 = float(np.percentile(z1, 95)); thr2 = float(np.percentile(sc2, 95))
    print(f"[rsp1] p95 stage1={thr1:.2f}  AE={thr2:.4f}", flush=True)

    # ---- locate the comb from the data itself (blind in spacing) ----
    spec = S.mean(0)
    freq = F0 + (np.arange(20000, 28000) - NFFT // 2) * FS / NFFT
    # noise from the TIME axis: immune to the 3-bin correlation of a Hann FFT
    L = np.log(np.maximum(S, 1e-6))
    R = L - median_filter(L, size=(1, 9), mode="reflect")
    sig_row = 1.4826 * np.median(np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1)
    sigma_d = float(1.2533 * np.median(sig_row) / np.sqrt(S.shape[0]))
    d = np.median(R, axis=0)
    e = d - median_filter(d, size=9, mode="reflect")
    zsp = e / sigma_d
    print(f"[rsp1] row noise {100*np.median(sig_row):.1f}%  sigma(column median) "
          f"{100*sigma_d:.2f}%  z p50/p99={np.percentile(zsp,50):.1f}/{np.percentile(zsp,99):.1f}",
          flush=True)
    lat = comb_lattice(zsp, spec)
    print(f"[rsp1] comb: {lat['n_peaks_z5']} peaks z>5, best spacing "
          f"{lat.get('spacing_bins')} bins = {lat.get('spacing_kHz', 0):.2f} kHz "
          f"= {lat.get('spacing_km_s', 0):.2f} km/s ({lat.get('spacing_ppm', 0):.1f} ppm), "
          f"ac={lat.get('autocorr', 0):.3f} vs null p95={lat.get('null_ac_p95', 0):.3f}, "
          f"p_mc={lat.get('p_mc')}", flush=True)
    print(f"[rsp1] lattice contrast {100*(lat.get('contrast_on_lattice') or 0):+.1f}% vs "
          f"off-lattice {100*(lat.get('contrast_off_lattice') or 0):+.1f}%, "
          f"on-lattice peaks {lat.get('peaks_on_lattice')}/{lat['n_peaks_z5']}", flush=True)
    pk = [i for i in range(3, len(zsp) - 3)
          if zsp[i] > 10 and zsp[i] == zsp[i - 3:i + 4].max()]
    teeth = [{"bin": int(i), "freq_MHz": round(float(freq[i] / 1e6), 6),
              "z": round(float(zsp[i]), 1)} for i in pk]

    # ---- injection at the same integration depth ----
    rng2 = np.random.default_rng(0)
    sweep = {}
    for amp in (0.05, 0.1, 0.2, 0.5, 1.0):
        Q = P.copy()
        cols = rng2.integers(2, PATCH - 2, size=len(P))
        for k, c in enumerate(cols):
            Q[k, :, c] *= (1.0 + amp)
        a1 = mrd_scores(Q); a2 = ae_err(net, Q)[1]
        y = np.r_[np.ones(len(a1)), np.zeros(len(z1))]
        sweep[f"{amp:.2f}"] = {
            "stage1_det": float((a1 > thr1).mean()),
            "stage2_det": float((a2 > thr2).mean()),
            "stage1_auroc": float(roc_auc_score(y, np.r_[a1, z1])),
            "stage2_auroc": float(roc_auc_score(y, np.r_[a2, sc2]))}
        v = sweep[f"{amp:.2f}"]
        print(f"[inject +{amp*100:5.1f}% 1px] stage1 {v['stage1_det']*100:5.1f}% "
              f"(AUROC {v['stage1_auroc']:.3f})   AE {v['stage2_det']*100:5.1f}% "
              f"(AUROC {v['stage2_auroc']:.3f})", flush=True)

    np.savez(os.path.join(D, "rsp1_two_stage_arrays.npz"),
             zsp=zsp, spec=spec, freq=freq, z1=z1, sc2=sc2,
             teeth=np.array(pk, dtype=np.int32), thr1=thr1, thr2=thr2,
             sigma_d=sigma_d)
    return {"n_patches": int(len(P)), "patches_x": xs,
            "p95_stage1": thr1, "p95_AE": thr2,
            "comb_lattice": lat, "comb_peaks": teeth,
            "AE_score_p50": float(np.median(sc2)), "AE_score_p95": thr2,
            "AE_score_dynamic_range": float(np.percentile(sc2, 99) / max(np.percentile(sc2, 50), 1e-12)),
            "sweep_amplitude": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--pix-max", type=int, default=8000)
    ap.add_argument("--skip-rsp1", action="store_true")
    ap.add_argument("--skip-lofar", action="store_true",
                    help="reuse the existing json and only recompute the RSP1 part")
    args = ap.parse_args()

    net = ConvAE(16, 128); net.load_state_dict(torch.load(W, map_location="cpu")); net.eval()

    if args.skip_lofar:
        res = json.load(open(OUTJ, encoding="utf-8"))
        res["rsp1"] = run_rsp1(net)
        json.dump(res, open(OUTJ, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("→", OUTJ, flush=True)
        return res

    # ---------- calibration on clean TRAIN patches ----------
    xtr = np.load(os.path.join(PD, "train_x.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(PD, "train_y.npy"), mmap_mode="r")
    ci = np.where(ytr.reshape(len(ytr), -1).sum(1) == 0)[0]
    ci = ci[np.random.default_rng(0).permutation(len(ci))[:4000]]
    Xc = np.asarray(xtr[ci], np.float32)
    t0 = time.time()
    s1c = mrd_scores(Xc); s2c = ae_err(net, Xc)[1]
    print(f"[calib] clean train n={len(Xc)} {time.time()-t0:.1f}s  "
          f"stage1 p50/p95={np.percentile(s1c,[50,95]).round(2)}  "
          f"AE p50/p95={np.percentile(s2c,[50,95]).round(4)}", flush=True)

    # ---------- LOFAR test, patch level (all) ----------
    xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
    yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")
    n = len(xte) if not args.max_test else min(args.max_test, len(xte))
    S1 = np.zeros(n, np.float32); S2 = np.zeros(n, np.float32)
    LAB = np.zeros(n, np.uint8); FR = np.zeros(n, np.float32)
    NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
    CH = 1000; t0 = time.time()
    for a in range(0, n, CH):
        b = min(a + CH, n)
        X = np.asarray(xte[a:b], np.float32)
        Y = np.asarray(yte[a:b]).astype(np.uint8)
        S1[a:b] = mrd_scores(X)
        S2[a:b] = ae_err(net, X)[1]
        LAB[a:b], FR[a:b], NP[a:b], NA[a:b] = morphology(Y)
        if a % 5000 == 0:
            print(f"  patch-level {a}/{n} {time.time()-t0:.0f}s", flush=True)
    print(f"[test] {n} patches in {time.time()-t0:.0f}s", flush=True)

    K = classify(NP, NA)
    z1 = rz(S1, s1c); z2 = rz(S2, s2c); zf = np.maximum(z1, z2)
    det = {"stage1_MRD": S1, "stage2_AE": S2, "fusion": zf}

    res = {"n_test": int(n), "calib": {"stage1_p95": float(np.percentile(s1c, 95)),
                                      "AE_p95": float(np.percentile(s2c, 95)),
                                      "stage1_ref_med": float(np.median(s1c)),
                                      "AE_ref_med": float(np.median(s2c))},
           "composition": {STRATA[k]: int(((K == k) & (LAB == 1)).sum()) for k in (1, 2, 3)},
           "n_rfi_patch": int(LAB.sum()), "rfi_pixel_fraction": float(FR.mean()),
           "detectors": {}, "by_stratum": {}}

    for k, s in det.items():
        res["detectors"][k] = metrics(LAB, s)

    for ki in (1, 2, 3):
        sel = (K == ki)
        if sel.sum() < 30:
            continue
        m = sel | (LAB == 0)
        res["by_stratum"][STRATA[ki]] = {
            "n": int(sel.sum()),
            "detectors": {k: {"auroc": float(roc_auc_score(LAB[m], s[m]))} for k, s in det.items()}}

    for k, v in res["detectors"].items():
        print(f"[LOFAR patch] {k:12s} AUROC={v['auroc']:.4f} AUPRC={v['auprc']:.4f} F1={v['best_f1']:.4f}",
              flush=True)
    print("[composition]", res["composition"], flush=True)
    for s, v in res["by_stratum"].items():
        print(f"[stratum n={v['n']:5d}] {s:42s} " +
              "  ".join(f"{k}={d['auroc']:.3f}" for k, d in v["detectors"].items()), flush=True)

    # ---------- pixel level on a subsample ----------
    rng = np.random.default_rng(1)
    idx_n = np.where(K == 2)[0]
    idx_o = rng.permutation(np.where(K != 2)[0])[:max(0, args.pix_max - len(idx_n))]
    pix_idx = np.sort(np.r_[idx_n, idx_o])
    print(f"[pixel] {len(pix_idx)} patches (narrow={len(idx_n)})", flush=True)

    # clean-train pixel error reference for the AE
    _, ec = ae_err(net, Xc[:600])
    med_e, mad_e = float(np.median(ec)), float(np.median(np.abs(ec - np.median(ec)))) + 1e-9

    ps1, ps2, pt = [], [], []
    for a in range(0, len(pix_idx), 500):
        ii = pix_idx[a:a + 500]
        X = np.asarray(xte[ii], np.float32)
        Y = np.asarray(yte[ii]).astype(np.uint8)
        p1 = mrd_profile(X)[1]
        e2, _ = ae_err(net, X)
        p2 = (e2 - med_e) / (1.4826 * mad_e)
        ps1.append(p1.reshape(-1)); ps2.append(p2.reshape(-1)); pt.append(Y.reshape(-1))
    ps1, ps2, pt = np.concatenate(ps1), np.concatenate(ps2), np.concatenate(pt)
    pf = np.maximum(ps1, ps2)
    res["pixel"] = {}
    for k, s in (("stage1_MRD", ps1), ("stage2_AE", ps2), ("fusion", pf)):
        res["pixel"][k] = {"auroc": float(roc_auc_score(pt, s)),
                           "auprc": float(average_precision_score(pt, s))}
        print(f"[LOFAR pixel] {k:12s} AUROC={res['pixel'][k]['auroc']:.4f} "
              f"AUPRC={res['pixel'][k]['auprc']:.4f}", flush=True)
    res["pixel"]["rfi_pixel_fraction"] = float(pt.mean())

    res["rsp1"] = run_rsp1(net)
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("→", OUTJ, flush=True)
    return res


if __name__ == "__main__":
    main()
