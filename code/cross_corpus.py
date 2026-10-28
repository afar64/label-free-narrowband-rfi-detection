#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-corpus consistency campaign: run the zero-label detector suite on
every spectral corpus in the repository, with a uniform protocol.

Per patch-capable corpus (train-free, deploy as documented in the paper):
  - build (time, freq) patch arrays, cap 4000 patches (half A = calibration,
    half B = evaluation; index order = chronological where available);
  - calibrate chi1 (pixel FPR 1e-3) on half A u-maps; member references
    (median/MAD + clean-quantile p-transform) also from half A only;
  - on half B report:  [C1] spontaneous flag occupancy (vs the 5% target at
    the p95 patch threshold), [C2] injection ladder AUROC (w=1, amps
    10/20/50%) for MRD / DAE / ST-flags / ST-max / fisher3', and AE
    zero-shot, [C3] for adhyanthac the known 32.82 kHz comb via
    comb_lattice on the time-median profile.
Per 1-D corpus (single spectra): 1-D MRD z profile + ST-1D flags/max counts.

Corpora: belllabs, millennium, bkb3, adhyanthac, pictor, lab (patch);
ttrt, kavinfidel (1-D).  LOFAR/HERA/RSP1 already have dedicated tables.

Output
    F:\\data\\21cmHI\\06_RFI_LABELED\\cross_corpus.json
"""
import gzip, glob, json, os, re, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import two_stage_detector as tsd
import sumthreshold_baseline as sb

D = tsd.D
AM = r"F:\data\21cmHI\05_AMATEUR_COMMUNITY"
OUTJ = os.path.join(D, "cross_corpus.json")
AMPS = (0.10, 0.20, 0.50)
CAP = 4000
ST_MS = (1, 4, 16, 64)          # reduced ladder for speed (documented)


# ---------------------------------------------------------------- loaders
def load_belllabs():
    fn = glob.glob(r"F:\data\21cmHI\07_BELL_LABS\catalog.dat.gz")[0]
    rows = []
    with gzip.open(fn, "rt", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) >= 130:
                try:
                    rows.append([float(x) for x in p[-124:]])
                except ValueError:
                    pass
    return np.asarray(rows, np.float32)


def load_millennium():
    src = glob.glob(r"F:\data\21cmHI\03_ARECIBO_MILLENNIUM\**\table1.dat", recursive=True)
    names, vel, prof = [], [], []
    with open(src[0], encoding="latin-1") as f:
        for line in f:
            if len(line) < 121:
                continue
            names.append(line[0:9].strip())
            vel.append(float(line[10:23]))
            prof.append(float(line[24:37]))
    names = np.array(names); prof = np.array(prof, float)
    uniq = sorted(set(names.tolist()))[:79]
    vel = np.array(vel)
    mats = []
    for u in uniq:
        m = names == u
        o = np.argsort(vel[m])
        mats.append(prof[m][o])
    L = min(len(m) for m in mats)
    return np.stack([m[:L] for m in mats]).astype(np.float32)


def load_bkb3():
    files = sorted(glob.glob(os.path.join(
        AM, "bkb3_hydrogen_observatory", "data", "**", "*.gz"), recursive=True))[:120]
    hdr = re.compile(r"^# Acquisition start: (\S+ \S+) UTC")
    blocks = []
    for fn in files:
        cur = None
        with gzip.open(fn, "rt", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    m = hdr.match(line)
                    if m:
                        if cur is not None and len(cur[2]) > 50:
                            blocks.append(cur)
                        cur = [m.group(1), [], []]
                    continue
                p = line.split()
                if len(p) == 2 and cur is not None:
                    try:
                        cur[1].append(float(p[0])); cur[2].append(float(p[1]))
                    except ValueError:
                        pass
        if cur is not None and len(cur[2]) > 50:
            blocks.append(cur)
    if not blocks:
        return None
    f0 = np.array(blocks[0][1])
    rows = []
    for b in blocks:
        f = np.array(b[1]); p = np.array(b[2])
        if len(f) != len(f0) or np.abs(f - f0).max() > 100.0:
            continue
        rows.append(10 ** (p / 10.0))                 # dB/Hz -> linear
    return np.stack(rows).astype(np.float32)


def load_adhyanthac():
    fns = sorted(glob.glob(os.path.join(AM, "adhyanthac_hline", "**", "*.csv"),
                           recursive=True))
    best = None
    for fn in fns:
        rows, meta = [], []
        for line in open(fn, encoding="utf-8", errors="replace"):
            p = [x.strip() for x in line.split(",") if x.strip() != ""]
            if len(p) < 20:
                continue
            try:
                meta.append((float(p[2]), float(p[4])))
                rows.append(np.array([float(x) for x in p[6:]]))
            except ValueError:
                pass
        if not rows:
            continue
        S = np.array(rows)
        n = S.shape[1]
        if n < 1024 or meta[0][1] > 2100.0:
            continue
        f0, bw = meta[0]
        if abs(f0 - 1419e6) > 1.5e6:
            continue
        if best is None or len(S) > best[0].shape[0]:
            best = (S.astype(np.float32), f0, bw)
    if best is None:
        return None
    return best[0]                      # (S, f0, bw) -> S only


def load_pictor():
    gd = os.path.join(AM, "pictor_users", "gijs")
    off = np.fromfile(os.path.join(gd, "off1024.dat"), dtype="float32")
    off = off.reshape(-1, 1024)
    ons = [np.fromfile(os.path.join(gd, f), dtype="float32").reshape(-1, 1024)
           for f in ("obs_3.dat", "observation_2.dat", "clean_observation.dat")]
    return np.concatenate([off] + ons).astype(np.float32)


def load_lab():
    from astropy.io import fits
    fn = r"F:\data\21cmHI\03_SPIRAL_ARMS\LAB_fullvel.fits"
    with fits.open(fn) as hd:
        a = np.asarray(hd[1].data.field(0), np.float32)   # TEMPERATURE: (3072, 1024)
    assert a.ndim == 2, f"unexpected LAB shape {a.shape}"
    return a[np.isfinite(a).all(1)]


# ---------------------------------------------------------------- protocol
def patchify(S, cap=CAP):
    T, F = S.shape
    if T < 64:
        return None
    ny, nx = (T - 64) // 32 + 1, (F - 64) // 32 + 1
    tot = ny * nx
    idx = np.sort(np.random.default_rng(0).choice(tot, size=min(cap, tot), replace=False))
    X = np.stack([S[(k // nx) * 32:(k // nx) * 32 + 64, (k % nx) * 32:(k % nx) * 32 + 64]
                  for k in idx]).astype(np.float32)
    return X


def run_patch_corpus(name, S, notes="", comb=False, position_axis=False):
    t0 = time.time()
    S = np.clip(S, 1e-6, None)
    X = patchify(S)
    if X is None:
        return {"name": name, "error": f"too few time rows ({S.shape[0]})"}
    half = len(X) // 2
    A, B = X[:half], X[half:]
    # calibration on A
    _, umA = sb.st_scores(A, want_map=True)
    chi1 = float(np.quantile(umA, 1 - 1e-3))
    mrdA = tsd.mrd_scores(A)
    dae = tsd.ConvAE(16, 128)
    dae.load_state_dict(torch.load(os.path.join(D, "ae_denoise_weights.pt"),
                                   map_location="cpu"))
    dae.eval()
    daeA = topk(dae, A)
    refs = {"MRD": mrdA, "DAE": daeA,
            "STF": (umA > chi1).sum(axis=(1, 2)).astype(np.float32)}
    # evaluation on B
    _, umB = sb.st_scores(B, want_map=True)
    stfB = (umB > chi1).sum(axis=(1, 2)).astype(np.float32)
    stmB = umB.max(axis=(1, 2))
    mrdB = tsd.mrd_scores(B)
    daeB = topk(dae, B)
    occ = {"stf_flag_rate": float((stfB > 0).mean()),
           "stf_p95_det": float((stfB > np.percentile(refs["STF"], 95)).mean()),
           "stm_p95_det": float((stmB > np.percentile(
               (umA > chi1).sum(axis=(1, 2)).astype(np.float32) * 0
               + umA.max(axis=(1, 2)), 95)).mean())}
    out = {"name": name, "n_patches": int(len(X)), "chi1": chi1,
           "occupancy": occ, "ladder": {}, "notes": notes}
    y = np.r_[np.ones(len(B)), np.zeros(len(B))]
    if position_axis:
        out["position_axis"] = ("rows are independent sky positions; the persistent-"
            "line injection model is ill-posed (astronomical structure dominates the "
            "null) - corpus retained as an occupancy control only")
        out["runtime_s"] = time.time() - t0
        print(f"[{name}] occupancy-only (position axis) {out['runtime_s']:.0f}s", flush=True)
        return out
    for amp in AMPS:
        Q = B.copy()
        rng = np.random.default_rng(7)
        for k in range(len(Q)):
            c = int(rng.integers(2, 62))
            Q[k, :, c] *= 1.0 + amp
        mq, dq = tsd.mrd_scores(Q), topk(dae, Q)
        _, umq = sb.st_scores(Q, want_map=True)
        fq = (umq > chi1).sum(axis=(1, 2)).astype(np.float32)
        smq = umq.max(axis=(1, 2))
        r = {"MRD": float(roc_auc_score(y, np.r_[mq, mrdB])),
             "DAE": float(roc_auc_score(y, np.r_[dq, daeB])),
             "STF": float(roc_auc_score(y, np.r_[fq, stfB])),
             "STM": float(roc_auc_score(y, np.r_[smq, stmB]))}
        pz = {"MRD": pv(mq, refs["MRD"]), "DAE": pv(dq, refs["DAE"]),
              "STF": pv(fq, refs["STF"])}
        fus = -2 * (np.log(pz["MRD"]) + np.log(pz["DAE"]) + np.log(pz["STF"]))
        pzB = {"MRD": pv(mrdB, refs["MRD"]), "DAE": pv(daeB, refs["DAE"]),
               "STF": pv(stfB, refs["STF"])}
        base_f = -2 * (np.log(pzB["MRD"]) + np.log(pzB["DAE"]) + np.log(pzB["STF"]))
        r["fisher3'"] = float(roc_auc_score(y, np.r_[fus, base_f]))
        out["ladder"][f"{amp:.2f}"] = r
        print(f"[{name} +{amp*100:3.0f}%] " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()),
              flush=True)
    if comb:
        spec = S.mean(0)
        zsp, _, _ = tsd.spec_z(spec)
        lat = tsd.comb_lattice(zsp, spec)
        out["comb"] = lat
        print(f"[{name} comb] {lat['spacing_kHz']:.2f} kHz p={lat['p_mc']}", flush=True)
    out["runtime_s"] = time.time() - t0
    print(f"[{name}] done {out['runtime_s']:.0f}s", flush=True)
    return out


def topk(net, X, bs=256):
    maps = np.empty((len(X), 64, 64), np.float32)
    with torch.no_grad():
        for a in range(0, len(X), bs):
            b = min(a + bs, len(X))
            xb = torch.from_numpy(tsd.ae_prep(np.asarray(X[a:b], np.float32)))
            e = (xb - net(xb)) ** 2
            maps[a:b] = torch.nn.functional.avg_pool2d(e, 3, 1, 1)[:, 0].numpy()
    return np.sort(maps.reshape(len(X), -1), axis=1)[:, -16:].mean(1)


def pv(s, ref):
    idx = np.searchsorted(np.sort(np.asarray(ref, np.float32)), s, side="left")
    return np.clip(1.0 - idx / (len(ref) + 1.0), 1e-12, 1.0)


def run_1d_corpus(name, spectra):
    """spectra: list of 1-D arrays.  MRD 1-D z + ST-1D stats."""
    out = {"name": name, "n_spectra": len(spectra), "max_abs_z": [], "notes": "1-D corpus"}
    zs = []
    for s in spectra:
        s = np.asarray(s, np.float32)
        s = np.maximum(s, 0.05 * np.median(s))      # floor pathological dips
        z, _, _ = tsd.spec_z(s)
        zs.append(z)
        out["max_abs_z"].append(float(np.abs(z).max()))
    out["max_abs_z"] = [float(np.median(out["max_abs_z"])),
                        float(np.max(out["max_abs_z"]))]
    if out["max_abs_z"][0] > 1e6:
        out["degenerate"] = ("quasi-noiseless single profile: residual MAD ~ 0, "
                             "z statistic undefined; corpus excluded from consistency claims")
        out["max_abs_z"] = None
        print(f"[{name}] DEGENERATE for 1-D z (no measurement noise structure)", flush=True)
        return out
    print(f"[{name}] 1-D max|z| median/max = {out['max_abs_z']}", flush=True)
    return out


def main():
    t00 = time.time()
    torch.set_num_threads(os.cpu_count())
    res = {}
    for name, loader, kw in [
        ("belllabs", load_belllabs, {"position_axis": True}),
        ("millennium", load_millennium, {"position_axis": True}),
        ("bkb3", load_bkb3, {}),
        ("adhyanthac", load_adhyanthac, {"comb": True}),
        ("pictor", load_pictor, {}),
        ("lab", load_lab, {"position_axis": True}),
    ]:
        try:
            S = loader()
            if S is None:
                res[name] = {"error": "loader returned None"}
                continue
            print(f"[{name}] spectra {S.shape}", flush=True)
            res[name] = run_patch_corpus(name, S, **kw)
        except Exception as e:
            import traceback
            res[name] = {"error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()[-800:]}
            print(f"[{name}] ERROR {e}", flush=True)
    # 1-D corpora
    try:
        s = np.loadtxt(os.path.join(AM, "ttrt_ghana", "sky_spec.txt"))[:, 1]
        res["ttrt"] = run_1d_corpus("ttrt", [s])
    except Exception as e:
        res["ttrt"] = {"error": str(e)}
    try:
        kd = os.path.join(AM, "kavinfidel_h1")
        specs = [10 ** (np.genfromtxt(os.path.join(kd, "data", f), delimiter=",").reshape(-1)[6:6 + 513] / 10.0)
                 for f in ("20.csv", "44.csv", "60.csv", "80.csv", "100.csv")]
        res["kavinfidel"] = run_1d_corpus("kavinfidel", specs)
    except Exception as e:
        res["kavinfidel"] = {"error": str(e)}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"→ {OUTJ}  ({time.time()-t00:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
