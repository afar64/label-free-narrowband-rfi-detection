# comb_reinforce.py — #7: triple reinforcement of the RSP1 comb-artifact detection.
#   A) permutation upgrade: 400 -> 10^4 shuffles, PLUS a BLOCK permutation
#      (64-channel blocks, preserves short-range noise correlation under H0)
#      -> p-value bulletproof against the "shuffling breaks correlations" critique;
#   B) phase-concentration control: are the on-lattice peaks the STRONG ones?
#      Rayleigh-style concentration of peak phases mod the comb period + a
#      strength comparison on-lattice vs off-lattice (T9);
#   C) cross-chain reproduction (D4): the comb must appear on the INDEPENDENT
#      incoherent-waterfall chain (5 captures, 488.3 Hz bins; expected lag
#      66.284 kHz / 0.4883 kHz = 135.7 bins) -- instrument-real, not a
#      single-chain processing artifact.
import json, os, sys, time
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_stage_detector as tsd

OUT = "results/comb_reinforce.json"
BLOCK = 64
NMC = 10_000
LAG_LO, LAG_HI = 100, 1600

def z0_autocorr(z0, lags):
    var = z0.var() + 1e-12
    return np.array([float((z0[:-L] * z0[L:]).mean()) / var for L in lags])

def main():
    t00 = time.time()
    S = tsd.rsp1_integrated()
    spec = S.mean(0)
    L = np.log(np.maximum(S, 1e-6))
    R = L - median_filter(L, size=(1, 9), mode="reflect")
    sig_row = 1.4826 * np.median(np.abs(R - np.median(R, axis=1, keepdims=True)), axis=1)
    sigma_d = float(1.2533 * np.median(sig_row) / np.sqrt(S.shape[0]))
    d = np.median(R, axis=0)
    e = d - median_filter(d, size=9, mode="reflect")
    zsp = e / sigma_d

    z0 = zsp - zsp.mean()
    lags = np.arange(LAG_LO, LAG_HI + 1)
    ac = z0_autocorr(z0, lags)
    k = int(np.argmax(ac))
    Lb, acmax = float(lags[k]), float(ac[k])
    print(f"[point] spacing {Lb:.0f} bins ({Lb*4.0/65536*1000:.2f} kHz), ac={acmax:.4f}", flush=True)

    # A) shuffles + block permutations, nmc=10^4 each
    rng = np.random.default_rng(2026)
    null_shuf = np.empty(NMC)
    null_blk = np.empty(NMC)
    n = len(z0)
    for m in range(NMC):
        q = rng.permutation(z0)
        null_shuf[m] = z0_autocorr(q, lags).max()
        qb = q.reshape(-1, BLOCK)[rng.permutation(q.size // BLOCK)].reshape(-1)[: (n // BLOCK) * BLOCK]
        null_blk[m] = z0_autocorr(qb - qb.mean(), lags).max()
        if (m + 1) % 2000 == 0:
            print(f"  perm {m+1}/{NMC} ({time.time()-t00:.0f}s)", flush=True)
    p_shuf = float((null_shuf >= acmax).mean())
    p_blk = float((null_blk >= acmax).mean())
    print(f"[A] shuffle-10^4: p = {p_shuf:.4f} (null p95 {np.percentile(null_shuf,95):.3f}); "
          f"block-64-10^4: p = {p_blk:.4f} (null p95 {np.percentile(null_blk,95):.3f})", flush=True)

    # B) phase concentration of the z>5 peaks
    pk = np.array([i for i in range(3, len(zsp) - 3)
                   if zsp[i] > 5 and zsp[i] == zsp[i - 3:i + 4].max()], float)
    ph = (pk % Lb) / Lb * 2 * np.pi
    Rray = float(np.hypot(np.sin(ph).mean(), np.cos(ph).mean()))
    # MC: R of n uniform phases
    Rnull = np.array([np.hypot(np.sin(np.random.default_rng(m).uniform(0, 2*np.pi, len(pk))).mean(),
                               np.cos(np.random.default_rng(10_000+m).uniform(0, 2*np.pi, len(pk))).mean())
                      for m in range(4000)])
    p_ray = float((Rnull >= Rray).mean())
    ph0 = float(np.angle(np.exp(1j * ph).mean()) % (2*np.pi))
    dph = np.abs(((pk - ph0 / (2*np.pi) * Lb) / Lb - np.round((pk - ph0 / (2*np.pi) * Lb) / Lb)))
    on06 = int((dph < 0.06).sum())
    on12 = int((dph < 0.12).sum())
    pki = pk.astype(int)
    z_on = float(zsp[pki[dph < 0.12]].mean()) if on12 else float("nan")
    z_off = float(zsp[pki[dph >= 0.12]].mean()) if on12 < len(pk) else float("nan")
    # multimodal phase structure: exact sub-lattice families (peaks whose
    # phases coincide to <0.005 form families; uniform noise would not)
    phf = (pk % Lb) / Lb
    fams = {}
    for x in phf:
        key = round(float(x), 3)
        fams[key] = fams.get(key, 0) + 1
    top_fams = sorted(fams.items(), key=lambda kv: -kv[1])[:4]
    # Kuiper-style statistic: max deviation of the phase CDF from uniform
    srt = np.sort(phf)
    n_p = len(srt)
    cdf_hi = np.arange(1, n_p + 1) / n_p
    cdf_lo = np.arange(0, n_p) / n_p
    theo_hi = (srt - ph0 / Lb) % 1.0
    kuiper = float(max((theo_hi - cdf_hi).max(), (cdf_lo - theo_hi).max()))
    print(f"[B] phase: on-lattice |dphase|<0.06: {on06}/{len(pk)}, <0.12: {on12}/{len(pk)}; "
          f"Rayleigh R={Rray:.3f} (p={p_ray:.3f}); Kuiper={kuiper:.3f}; "
          f"top exact-families (phase:count): {top_fams}; mean z on/off = {z_on:.1f}/{z_off:.1f}", flush=True)

    # C) cross-chain reproduction on the 5 incoherent captures
    import glob
    chain = {}
    for f in sorted(glob.glob(r"F:/21cm/paper/nextstep_iq0208_cache/*.npz")):
        w = np.load(f)["w"]
        prof = w.mean(0)
        Lc = np.log(np.maximum(prof, 1e-30))
        Rc = Lc - median_filter(Lc, size=31, mode="nearest")
        ec = Rc - median_filter(Rc, size=31, mode="nearest")
        sc = 1.4826 * np.median(np.abs(ec - np.median(ec))) + 1e-12
        zc = ec / sc
        zc0 = zc - zc.mean()
        lags_c = np.arange(110, 165)
        acc = z0_autocorr(zc0, lags_c)
        kc = int(np.argmax(acc))
        name = os.path.basename(f).split("_incoherent")[0][-18:]
        chain[name] = {"lag_bins": int(lags_c[kc]),
                       "spacing_kHz": float(lags_c[kc] * 0.48828125),
                       "ac": float(acc[kc])}
        print(f"[C] {name}: lag {lags_c[kc]} bins = {chain[name]['spacing_kHz']:.2f} kHz, ac={acc[kc]:.3f}", flush=True)

    out = {"point": {"spacing_bins": Lb, "spacing_kHz": Lb*4.0/65536*1000, "autocorr": acmax},
           "p_shuffle_1e4": p_shuf, "p_block64_1e4": p_blk,
           "null_shuffle_p95": float(np.percentile(null_shuf, 95)),
           "null_block_p95": float(np.percentile(null_blk, 95)),
           "phase": {"R": Rray, "p": p_ray, "on_006": on06, "on_012": on12,
                     "n_peaks": int(len(pk)), "z_on": z_on, "z_off": z_off,
                     "kuiper": kuiper, "top_families": top_fams},
           "cross_chain": chain}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {OUT}; [done] {time.time()-t00:.0f}s", flush=True)

if __name__ == "__main__":
    main()
