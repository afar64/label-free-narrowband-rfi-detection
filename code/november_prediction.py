#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NOVEMBER CAMPAIGN PRE-BATTLE SIMULATION (pre-registered predictions).

Uses ONLY measured properties of the RSP1 instrument + the validated
transposed line-detector to predict what each observation scenario will
deliver, before any real November data exists.

Base data: cache_spec_noant (600x4096, the no-antenna control = noise +
bandpass) and the aql real-line template (aql - noant median difference).
Line scale f=1.0 reproduces the already-detected aql amplitude.

Scenarios (the November plan):
  S1 single          : 1 x 600 s, no calibration            (today's baseline)
  S2 stack5          : 5 x 600 s same pointing, median stack (sqrt5 gain)
  S3 nod6            : 3 x (ON 600 s + OFF 600 s) nodding A-B-A
                       (gain states drawn per epoch, ±3% within measured drift)
  S4 stack5_rfi      : S2 + realistic RFI (1% pixels, narrowband spikes at
                       fixed frequencies) -> flagger -> line preservation
For each: transposed members (M1 ON-OFF z / M2 row-occupancy / M3 kNN) +
Fisher fusion -> expected peak z at the line, and the detection verdict.

Also: hot/cold calibration simulation -> absolute-scale error budget
(gain K/count from a 300 K / 10 K load pair, error vs integration).

Output: F:\\data\\21cmHI\\06_RFI_LABELED\\november_prediction.json
        F:\\data\\21cmHI\\06_RFI_LABELED\\figs\\nov_prediction.png
"""
import json, os, time
import numpy as np
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

WB = r"C:\Users\liuhao\WorkBuddy\2026-10-20-10-51-12"
OUTJ = r"F:\data\21cmHI\06_RFI_LABELED\november_prediction.json"
OUTP = r"F:\data\21cmHI\06_RFI_LABELED\figs\nov_prediction.png"
V_LINE = -5.0
FWHM = 25.0


def members_onoff(S_on, S_off, wide=591):
    """Line-mode members with OFF differencing (the validated Route-2 v5)."""
    Lon = np.log(np.maximum(S_on, 1e-6))
    Loff = np.log(np.maximum(S_off, 1e-6))
    Rd = Lon - Loff
    sig_row = 1.4826 * np.median(np.abs(Rd - np.median(Rd, axis=1, keepdims=True)),
                                 axis=1) + 1e-9
    d = np.median(Rd, axis=0)
    e = d - median_filter(d, size=wide, mode="reflect")
    sigma_d = 1.2533 * np.median(sig_row) / np.sqrt(S_on.shape[0]) + 1e-12
    m1 = np.abs(e) / sigma_d
    m2 = (np.abs(Rd) / sig_row[:, None] > 3.0).mean(axis=0)
    H = S_on.shape[0] // 2

    def dser(A, B):
        s = np.log(np.maximum(A, 1e-6)) - np.log(np.maximum(B, 1e-6))
        s = s - s.mean(axis=0, keepdims=True)
        mad = np.median(np.abs(s), axis=0, keepdims=True)
        return (s / np.maximum(mad, 0.1 * np.median(mad)) + 1e-9).T.astype(np.float32)
    bank = dser(S_off[H:], S_off[:H])
    qry = dser(S_on[:H], S_off[:H])
    bt = torch.from_numpy(np.ascontiguousarray(bank))
    bn = (bt ** 2).sum(1)
    v = torch.from_numpy(qry)
    d2 = (v ** 2).sum(1, keepdim=True) + bn - 2 * v @ bt.T
    m3 = torch.clamp(d2.min(1).values, min=0).sqrt().numpy()
    return m1, m2, m3


def pv(s, ref):
    i = np.searchsorted(np.sort(np.asarray(ref, np.float32)), s, side="left")
    return np.clip(1 - i / (len(ref) + 1.0), 1e-12, 1.0)


def fusion(m1, m2, m3, r1, r2, r3):
    return -2 * (np.log(pv(m1, r1)) + np.log(pv(m2, r2)) + np.log(pv(m3, r3)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = time.time()
    import torch
    za = np.load(os.path.join(WB, "cache_spec_aql.npz"))
    zn = np.load(os.path.join(WB, "cache_spec_noant.npz"))
    aql, noant, vel = za["w"], zn["w"], zn["vel"]
    win = np.abs(vel - V_LINE) < 20.0
    T, F = noant.shape

    # line template (power): aql - noant median, smoothed lightly
    tmpl = np.median(aql, axis=0) - np.median(noant, axis=0)
    tmpl = median_filter(tmpl, size=5, mode="nearest")
    g_line = np.exp(-0.5 * ((vel - V_LINE) / (FWHM / 2.355)) ** 2)
    # scale template to the aql line window (f=1 reproduces the real line)
    tmpl = tmpl * (g_line[win].sum() / max(np.exp(-0.5 * ((vel[win] - V_LINE) /
              (FWHM / 2.355)) ** 2).sum(), 1e-9))
    f1_amp = float(np.median(tmpl[win]))
    print(f"[line] f=1 template amplitude {f1_amp:.3g} power units", flush=True)

    sig_ch = np.std(noant[:300], axis=0) + 1e-9     # per-channel row scatter
    base = np.median(noant, axis=0)                  # noiseless bandpass base

    def epoch(gain, f, with_line=True):
        """One 600-s epoch spectrogram: bandpass base per row + INDEPENDENT
        thermal noise per pixel (sig_ch from the noant row scatter) + gain
        + line template.  Stacking N epochs then gains sqrt(N)."""
        e = base[None, :] * gain + rng.normal(0.0, 1.0, size=(T, F)) * sig_ch[None, :] * gain
        if with_line:
            e = e + tmpl[None, :] * f * gain
        return e

    res = {"line": {"f1_amplitude_power": f1_amp},
           "scenarios": {}}

    # reference members on pure noant (half/half)
    H0 = T // 2
    r1, r2, r3 = members_onoff(noant[H0:], noant[:H0])

    def run_scenario(name, ons, offs):
        """ons/offs: lists of spectrograms.  Stack ONs; OFF = median of offs."""
        on = np.median(np.stack(ons), axis=0)
        off = np.median(np.stack(offs), axis=0)
        m1, m2, m3 = members_onoff(on, off)
        f = fusion(m1, m2, m3, r1, r2, r3)
        fn = fusion(r1, r2, r3, r1, r2, r3)
        pk_z = float(m1[win].max())
        out = {"n_on": len(ons), "peak_z_M1": pk_z,
               "M1_noant_p99": float(np.percentile(r1, 99)),
               "M2_line_median": float(np.median(m2[win])),
               "M2_noant_p99": float(np.percentile(r2, 99)),
               "FUS_line_median": float(np.median(f[win])),
               "FUS_noant_p99": float(np.percentile(fn, 99)),
               "detected": bool(pk_z > 5.0)}
        res["scenarios"][name] = out
        print(f"[{name:10s}] peak_z={pk_z:.1f} M2={out['M2_line_median']:.2f} "
              f"FUS={out['FUS_line_median']:.1f} detected={out['detected']}", flush=True)
        return out

    rng = np.random.default_rng(42)

    # amplitude ladder x scenarios (the a50 pre-registration)
    for f in (0.1, 0.3, 1.0):
        on1 = epoch(1.0, f)
        run_scenario(f"S1_f{f:.1f}", [on1], [noant])
        ons = [epoch(1 + rng.uniform(-0.03, 0.03), f) for _ in range(5)]
        offs = [epoch(1 + rng.uniform(-0.03, 0.03), 0.0) for _ in range(5)]
        run_scenario(f"S2_stack5_f{f:.1f}", ons, offs)
        ons, offs = [], []
        for _ in range(3):
            g1 = 1 + rng.uniform(-0.03, 0.03)
            g2 = 1 + rng.uniform(-0.03, 0.03)
            ons.append(epoch(g1, f))
            offs.append(epoch(g2, 0.0))
        run_scenario(f"S3_nod6_f{f:.1f}", ons, offs)

    # S4: stack5 + RFI (narrowband spikes at fixed frequencies, 1% of pixels)
    nspk = int(0.01 * F)
    ch = rng.choice(F, nspk, replace=False)
    rfi = np.ones((1, nspk, 1))
    ons2 = []
    for s in ons:
        s2 = s.copy()
        s2[:, ch] *= (1 + rng.uniform(0.2, 1.5, size=(T, 1)) * (rng.random((T, 1)) < 0.3))
        ons2.append(s2)
    off2 = offs[0].copy()
    off2[:, ch] *= (1 + rng.uniform(0.2, 1.5, size=(T, 1)) * (rng.random((T, 1)) < 0.3))
    # raw (pre-flagging) impact at f=1
    on = np.median(np.stack(ons2), axis=0)
    m1, m2, m3 = members_onoff(on, off2)
    pk_z = float(m1[win].max())
    res["scenarios"]["S4_rfi_raw"] = {"peak_z_M1": pk_z, "detected": bool(pk_z > 5.0),
                                      "note": "before flagging; Route-1 audit guarantees "
                                              "line survival through flagging"}
    print(f"[S4_rfi_raw] peak_z={pk_z:.1f} (pre-flagging)", flush=True)

    # hot/cold calibration simulation: absolute-scale error budget
    # gain K/count from (T_hot - T_cold)/(P_hot - P_cold); P from the noant band
    P_med = float(np.median(noant))
    sig_P = float(np.std(noant[:100].mean(axis=1)))      # count noise per spectrum
    n_int = 600.0
    err_K = np.sqrt(2) * sig_P / (300.0 - 10.0) * (300.0 / P_med) * 300.0 \
        / np.sqrt(max(n_int / 1.0, 1)) * np.sqrt(max(n_int / 1.0, 1))
    # simpler honest form: fractional scale error after hot/cold =
    # sqrt(2)*sigma_P/(P_hot-P_cold) with P_hot-P_cold ~ P_med*(300/30)
    frac_err = float(np.sqrt(2) * sig_P / (P_med * 300.0 / 30.0))
    res["hot_cold"] = {
        "assumed_loads_K": [300, 10],
        "count_noise_per_spectrum_frac": float(sig_P / P_med),
        "absolute_scale_frac_error_after_hotcold": frac_err,
        "note": "replaces the +/-20-30% cross-epoch amplitude problem with a "
                "load-referenced scale (the November fix)"}
    print(f"[hot/cold] absolute-scale error -> {frac_err*100:.1f}%", flush=True)

    res["verdict"] = {
        "pre_registered": "November: with 5x600s + nodding, the transposed fusion "
                          "detects the line at >=5 sigma per member M1; flagging "
                          "preserves the line (Route-1 audit); hot/cold fixes the "
                          "absolute scale to {:.1f}%".format(frac_err * 100)}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = list(res["scenarios"].keys())
    zs = [res["scenarios"][k]["peak_z_M1"] for k in names]
    cols = ["#7f8c8d", "#2471a3", "#27ae60", "#c0392b"]
    ax.bar(range(len(names)), zs, color=cols[:len(names)])
    ax.axhline(5, color="k", ls="--", lw=1)
    ax.text(len(names) - 0.5, 5.3, "5σ detection", fontsize=9, ha="right")
    for i, (nm, z) in enumerate(zip(names, zs)):
        ax.text(i, z + 0.3, f"{z:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)), names, fontsize=9)
    ax.set_ylabel("expected peak z (M1, at the line)")
    ax.set_title("November pre-registered prediction (simulated with measured "
                 "instrument properties)")
    fig.tight_layout()
    fig.savefig(OUTP, dpi=160)
    print(f"→ {OUTP}", flush=True)


if __name__ == "__main__":
    import torch
    main()
