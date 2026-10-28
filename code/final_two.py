# final_two.py — the last two backlog items.
# Task 1: the 427 K==0 positive patches (1-3 persistent columns, narrow power
#         fraction <=0.6 = MIXED morphology) get a dominant-type stratum
#         assignment; stratum AUROCs recomputed and compared to the paper table.
# Task 2: chi1 three-working-point scan — the SumThreshold u-map is chi1-free,
#         the decision threshold scales with chi1; the floor theorem predicts
#         the one-pixel turning CONTRAST scales linearly with chi1_eff.
#         Verify: turning amplitude at chi1_eff in {0.5, 1, 2} x base.
import json, os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_stage_detector as tsd
import sumthreshold_baseline as sb

PD = tsd.PD
OUT = "results/final_two.json"
RNG = np.random.default_rng(2026)

def auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan

# ---------------- Task 1: the 427 mixed-morphology positives ----------------
print("== Task 1: 427 mixed-morphology positives ==", flush=True)
d = np.load("../results/test_scores.npz", allow_pickle=True)
FUS, S1, S2, LAB, K = (d["FUS"], d["S1"], d["S2"],
                       d["LAB"].astype(bool), d["K"])
yte = np.load(os.path.join(PD, "test_y.npy"), mmap_mode="r")

# recompute morphology per patch (chunked) to get NA (narrow power fraction)
n = len(FUS)
NP = np.zeros(n, np.int16); NA = np.zeros(n, np.float32)
CH = 2000
for a in range(0, n, CH):
    b = min(a + CH, n)
    Y = np.asarray(yte[a:b]).astype(np.uint8)
    _, _, np_, na = tsd.morphology(Y)
    NP[a:b] = np_; NA[a:b] = na
mixed = (K == 0) & LAB
print(f"K0 positives: {int(mixed.sum())} (npers range {NP[mixed].min()}..{NP[mixed].max()}, "
      f"narrow frac {NA[mixed].min():.2f}..{NA[mixed].max():.2f})", flush=True)

# dominant-type assignment: narrow power fraction > 0.5 -> narrow(2), else transient(1)
K_fix = K.copy()
dom_narrow = mixed & (NA > 0.5)
K_fix[dom_narrow] = 2
K_fix[mixed & ~dom_narrow] = 1
print(f"assigned: narrow += {int(dom_narrow.sum())}, transient += {int((mixed & ~dom_narrow).sum())}", flush=True)

def stratum_table(keys):
    # ORIGINAL convention (two_stage_detector.py:428): stratum positives vs ALL
    # clean patches (LAB==0) as negatives.
    tab = {}
    for s, name in tsd.STRATA.items():
        m = (keys == s) | (~LAB)
        tab[name] = {"n_pos": int((keys == s).sum()),
                     "MRD": auroc(LAB[m], S1[m]),
                     "AE": auroc(LAB[m], S2[m]),
                     "FUS": auroc(LAB[m], FUS[m])}
    return tab

tab_old = stratum_table(K)
tab_new = stratum_table(K_fix)
for name in tsd.STRATA.values():
    o, nw = tab_old[name], tab_new[name]
    print(f"  {name:42s} n_pos {o['n_pos']:6d}->{nw['n_pos']:6d}  "
          f"MRD {o['MRD']:.4f}->{nw['MRD']:.4f}  FUS {o['FUS']:.4f}->{nw['FUS']:.4f}", flush=True)

# ---------------- Task 2: chi1 three-working-point scan ----------------
print("== Task 2: chi1 scan ==", flush=True)
# u-maps are chi1-free: compute once per (amp) on injected patches, then apply
# three decision thresholds chi1_eff = scale * chi1_base.
xte = np.load(os.path.join(PD, "test_x.npy"), mmap_mode="r")
# clean reference patches for the chi1_base calibration (same protocol: 1500 clean)
ci = np.where(np.asarray(yte[:5000]).reshape(5000, -1).sum(1) == 0)[0]
ci = ci[RNG.permutation(len(ci))[:1500]]
Xc = np.asarray(xte[ci], np.float32)
t0 = time.time()
u_clean = sb.st_scores(Xc, want_map=True)[1]
chi1_base = float(np.quantile(u_clean, 1.0 - sb.FPR_TARGET))
print(f"  chi1_base (FPR {sb.FPR_TARGET}) = {chi1_base:.3f}  ({time.time()-t0:.0f}s)", flush=True)

AMPS = (0.10, 0.20, 0.50, 1.00)
SCALES = (0.5, 1.0, 2.0)
N_INJ = 1200
ii = RNG.choice(np.flatnonzero(~LAB[:]), size=N_INJ, replace=False)  # inject on clean patches
ii = np.concatenate([ci[:500], ii])                                   # + 500 clean ref
ladder = {}
for amp in AMPS:
    Xi = np.asarray(xte[ii], np.float32).copy()
    cc = RNG.integers(4, int(sb.PATCH)-4, size=len(ii))
    for k in range(len(ii)):
        Xi[k, :, cc[k]] *= (1.0 + amp)
    u_inj = sb.st_scores(Xi, want_map=True)[1]
    y = np.r_[np.ones(len(ii)), np.zeros(len(u_clean))]
    row = {}
    for sc in SCALES:
        thr = sc * chi1_base
        s_inj = np.maximum(u_inj.max(axis=(1, 2)) - thr, 0.0)
        s_clr = np.maximum(u_clean.max(axis=(1, 2)) - thr, 0.0)
        row[f"{sc:.1f}"] = auroc(y, np.r_[s_inj, s_clr])
    ladder[f"{amp:.2f}"] = row
    print(f"  amp={amp:.2f}: AUROC at chi1 x0.5/1/2 = "
          f"{row['0.5']:.3f}/{row['1.0']:.3f}/{row['2.0']:.3f}", flush=True)

# turning amplitude per scale: first amp with AUROC >= 0.7
turn = {}
for sc in SCALES:
    tt = next((a for a in AMPS if ladder[f"{a:.2f}"][f"{sc:.1f}"] >= 0.70), None)
    turn[f"{sc:.1f}"] = tt
print(f"  turning amp (AUROC>=0.7): {turn}  -> linear-in-chi1?", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"task1": {"n_mixed": int(mixed.sum()),
                         "assigned_narrow": int(dom_narrow.sum()),
                         "assigned_transient": int((mixed & ~dom_narrow).sum()),
                         "tab_old": tab_old, "tab_new": tab_new},
               "task2": {"chi1_base": chi1_base, "ladder": ladder, "turning": turn}},
              f, indent=1)
print(f"[saved] {OUT}; done", flush=True)
