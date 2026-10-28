#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm-map optimisation: turn the naive 1.7% arm-membership negative into a
positive via physically-motivated ridge selection + DATA-DRIVEN spiral
fitting (no guessed loci).

Key ideas (standard terminal-velocity practice + Hough transform):
  1. Selection: a log-spiral arm is traced by (a) BRIGHT ridges (T_peak high
     - arm gas is bright; diffuse inter-arm gas is dim) and (b) ridges NEAR
     the terminal velocity (v >= V_max(l) - 15 km/s), where the near/far
     degeneracy collapses and the near solution is physical.
  2. Log-spirals are LINES in (phi_gc, ln R) space: ln R = ln R_a - tan(psi)*phi.
     Hough transform over (pitch psi in 9..18 deg, intercept ln R_a) finds the
     data-driven arms; the fitted pitch is compared with the published 12.5 deg.
  3. Report concentration at 0.75 / 1.0 / 2.0 kpc for: naive-all vs
     bright-terminal vs fitted-lines, all apples-to-apples.

Output: F:\\data\\21cmHI\\03_SPIRAL_ARMS\\arm_fit.json
        F:\\data\\21cmHI\\03_SPIRAL_ARMS\\figs\\sasp_fig5_armfit.png
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, maximum_filter

R0, TH0, PSI_PUB = 8.178, 231.0, 12.5
OUTJ = r"F:\data\21cmHI\03_SPIRAL_ARMS\arm_fit.json"
OUTP = r"F:\data\21cmHI\03_SPIRAL_ARMS\figs\sasp_fig5_armfit.png"


def load_slice():
    from astropy.io import fits
    with fits.open(r"F:\data\21cmHI\03_SPIRAL_ARMS\lab250h.fit.gz",
                   memmap=False) as h:
        hd = h[0].header
        b0 = int(round((0.0 - hd["CRVAL2"]) / hd["CDELT2"]))
        sl = np.asarray(h[0].data[:, b0 - 2:b0 + 3, :], np.float32).mean(axis=1)
        vlsr = (hd["CRVAL3"] + (np.arange(hd["NAXIS3"]) + 1 - hd["CRPIX3"])
                * hd["CDELT3"]) / 1000.0
        lon = (hd["CRVAL1"] + (np.arange(hd["NAXIS1"]) + 1 - hd["CRPIX1"])
               * hd["CDELT1"])
    return sl.T, lon, vlsr                      # (l, v)


def vmax_curve(data, lon, vlsr):
    """Terminal-edge half-power V_max for l in 10..85 (1-deg steps)."""
    out = {}
    for l_d in np.arange(10, 86, 1.0):
        i = int(np.argmin(np.abs(lon - l_d)))
        spec = data[i]
        pk = spec.max()
        if pk < 5:
            continue
        hi = np.flatnonzero(spec > pk / 2.0)
        out[float(l_d)] = float(vlsr[hi[-1]])
    return out


def kin(l_deg, v):
    l = np.radians(l_deg)
    sl = np.sin(l)
    denom = v + TH0 * sl
    R = np.sqrt(TH0 * R0 ** 2 * sl / np.maximum(denom, 1e-6))
    disc = R ** 2 - (R0 * sl) ** 2
    s = R0 * np.cos(l) - np.sqrt(np.maximum(disc, 0))
    return s * np.sin(l), R0 - s * np.cos(l), R


def main():
    t0 = time.time()
    data, lon, vlsr = load_slice()
    Tb = np.clip(data - median_filter(data, size=(1, 61), mode="nearest"), 0, None)
    Vm = vmax_curve(data, lon, vlsr)

    # ---- ridge extraction at full 0.5-deg sampling -------------------------
    ridges = []
    for i, l_d in enumerate(lon):
        if not (5 <= l_d <= 88):
            continue
        col = Tb[i]
        mx = maximum_filter(col, size=15)
        for j in np.flatnonzero((col == mx) & (col > 3.0)):
            vm = Vm.get(float(round(l_d)) )
            vm = Vm.get(min(Vm, key=lambda k: abs(k - l_d)))
            ridges.append((float(l_d), float(vlsr[j]), float(col[j]), vm or np.nan))
    ridges = np.array(ridges)
    v_term = np.array([Vm[min(Vm, key=lambda k: abs(k - l))] for l in ridges[:, 0]])
    sets = {
        "all": np.ones(len(ridges), bool),
        "bright(T>=10K)": ridges[:, 2] >= 10.0,
        "terminal(v>=Vmax-15)": ridges[:, 1] >= v_term - 15.0,
        "bright&terminal": (ridges[:, 2] >= 10.0) & (ridges[:, 1] >= v_term - 15.0),
    }
    print(f"[ridges] {len(ridges)}  bright {sets['bright(T>=10K)'].sum()}  "
          f"terminal {sets['terminal(v>=Vmax-15)'].sum()}  "
          f"b&t {sets['bright&terminal'].sum()}", flush=True)

    # ---- kinematic conversion ----------------------------------------------
    x = np.full(len(ridges), np.nan); y = np.full(len(ridges), np.nan)
    Rg = np.full(len(ridges), np.nan)
    for k, (l_d, v_d, _, _) in enumerate(ridges):
        xx, yy, R = kin(l_d, v_d)
        x[k], y[k], Rg[k] = xx, yy, R
    phi_gc = np.degrees(np.arctan2(x, y))          # GC angle, Sun at phi=0
    lnR = np.log(Rg)

    ok = np.isfinite(x) & np.isfinite(y) & (phi_gc > -5) & (phi_gc < 95)
    # ---- published-loci baseline (psi=12.5, the four R_a) -------------------
    pub_lines = [(R_a, float(np.log(R_a))) for R_a in (6.24, 5.48, 11.54, 15.85)]
    def frac_near(c_list, tol_kpc, mask):
        # distance from point to line in (phi, lnR): convert tolerance
        dmin = np.full(mask.sum(), 1e9)
        pts_phi = np.radians(phi_gc[mask]); pts_ln = lnR[mask]
        for R_a, c in c_list:
            # line: lnR = -tan(Psi)phi + c  -> distance approx |dl| * sin(angle)
            d = np.abs(pts_ln - (-np.tan(np.radians(PSI_PUB)) * pts_phi + c)) \
                * np.cos(np.radians(PSI_PUB))
            d_kpc = d * Rg[mask]                   # small-angle kpc
            dmin = np.minimum(dmin, d_kpc)
        return float((dmin <= tol_kpc).mean())
    res = {}
    for name, mk in sets.items():
        mkk = mk & ok
        if mkk.sum() < 10:
            continue
        res[name] = {"n": int(mkk.sum()),
                     "pub_0.75": frac_near(pub_lines, 0.75, mkk),
                     "pub_1.0": frac_near(pub_lines, 1.0, mkk),
                     "pub_2.0": frac_near(pub_lines, 2.0, mkk)}
        print(f"[pub {name:22s}] n={res[name]['n']:5d} "
              f"0.75={res[name]['pub_0.75']:.3f} 1.0={res[name]['pub_1.0']:.3f} "
              f"2.0={res[name]['pub_2.0']:.3f}", flush=True)

    # ---- Hough: data-driven lines in (phi, lnR) on bright&terminal ---------
    mk = sets["bright&terminal"] & ok
    pts_phi = np.radians(phi_gc[mk]); pts_ln = lnR[mk]; pts_R = Rg[mk]
    best = []
    for psi in np.arange(9.0, 18.01, 0.25):
        c = pts_ln + np.tan(np.radians(psi)) * pts_phi
        hist, edges = np.histogram(c, bins=120)
        # smooth binning tolerance
        for bi in np.argsort(hist)[::-1][:1]:
            cc = 0.5 * (edges[bi] + edges[bi + 1])
            mline = np.abs(c - cc) < 0.02
            if mline.sum() >= 8:
                best.append({"psi": float(psi), "lnRa": float(cc), "R_a": float(np.exp(cc)),
                             "n": int(mline.sum())})
    # non-max suppression by R_a
    best.sort(key=lambda d: -d["n"])
    kept = []
    for b in best:
        if all(abs(np.log(b["R_a"]) - np.log(k["R_a"])) > 0.15 for k in kept):
            kept.append(b)
        if len(kept) == 3:
            break
    res["hough_lines"] = kept
    print("[hough]", json.dumps(kept, indent=1), flush=True)

    # concentration of bright&terminal near the FITTED lines
    dmin = np.full(int(mk.sum()), 1e9)
    for ln in kept:
        c = ln["lnRa"]
        d = np.abs(pts_ln - (-np.tan(np.radians(ln["psi"])) * pts_phi + c)) \
            * np.cos(np.radians(ln["psi"]))
        dmin = np.minimum(dmin, d * pts_R)
    res["fitted_concentration"] = {
        "0.75": float((dmin <= 0.75).mean()),
        "1.0": float((dmin <= 1.0).mean()),
        "2.0": float((dmin <= 2.0).mean()),
        "n": int(mk.sum())}
    print("[fitted]", res["fitted_concentration"], flush=True)

    # ---- figure: (phi, lnR) space with the fitted lines ---------------------
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(np.radians(phi_gc[sets["all"] & ok]), lnR[sets["all"] & ok], ".",
            color="#d5dbdb", ms=3, label="all ridges")
    ax.plot(pts_phi, pts_ln, ".", color="#c0392b", ms=5,
            label="bright & terminal")
    for k, ln in enumerate(kept):
        c = ln["lnRa"]
        xx = np.array([np.radians(phi_gc[ok & sets['bright&terminal']].min()),
                       np.radians(phi_gc[ok & sets['bright&terminal']].max())])
        ax.plot(xx, -np.tan(np.radians(ln["psi"])) * xx + c, lw=2,
                label=f"arm{k+1}: psi={ln['psi']:.2f}deg Ra={ln['R_a']:.2f}kpc n={ln['n']}")
    ax.set_xlabel("Galactocentric azimuth phi (rad, Sun at 0)")
    ax.set_ylabel("ln R (R in kpc)")
    ax.set_title("Data-driven spiral-arm detection: log-spirals are straight\n"
                 "lines in (phi, lnR); Hough lines on bright+terminal ridges")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTP, dpi=160)
    print(f"→ {OUTP}", flush=True)

    res["config"] = {"PSI_PUB": PSI_PUB, "R0": R0, "TH0": TH0}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
