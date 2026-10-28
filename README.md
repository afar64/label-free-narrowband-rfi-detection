# Reproducibility Package — Label-Free Narrowband RFI Detection (IEEE TIM)

## Layout
- `code/` — every analysis script, in pipeline order (see RUN ORDER below)
- `results/` — every JSON/NPZ behind each table and figure of the manuscript
- `../manuscript_TIM.tex` — the manuscript (this package accompanies it)

## RUN ORDER
1. `two_stage_detector.py`      — stage-1 MRD + stage-2 AE + RSP1 comb/injection
2. `sumthreshold_baseline.py`   — classical reference (Offringa+2010 recipe)
3. `patchcore_baseline.py`      — PatchCore/PaDiM, in-domain features
4. `patchcore_imagenet.py`      — published-backbone arm (domain gap)
5. `experiments_richer.py`      — injection maps, calibration ablation, runtime
6. `experiments_richer2.py`     — negative-line / multi-line injections, scale ablation, paired test
7. `bootstrap_ci.py`            — image-level cluster bootstrap (needs test_scores.npz from 5)
8. `optimize_detector.py`       — fusion rule + aggregation search (selection half)
9. `optimize_detector2.py`      — denoising DAE + Fisher member search
10. `optimize_detector3.py`      — champion on all validation surfaces
11. `cross_corpus.py`           — 8-corpus consistency campaign
12. `sam_arm_rsp1.py`            — foundation-model arm: SAM ViT-B on the RSP1 ladder
                                 (blind AMG scoring + oracle-prompt control; results
                                 `results/sam_arm_rsp1.json` CPU and `sam_arm_rsp1_gpu.json`
                                 CUDA agree on every reported digit)
12b. `sam_fix_readout.py`        — the fix: aggregate-first z-profile rendered as a bar
                                 image, SAM as blind readout (narrow yield 0.88-0.95;
                                 readout 0.63/0.80/0.83 strictly below the max|z| ceiling
                                 0.73/0.96/0.99; `results/sam_fix_readout.json`)
12c. `sarthak_birdie_transfer.py` — cross-instrument transfer on an independent
                                 educational horn corpus (11 single-sweep scans, Mhaske+2022
                                 AJP protocol).  v1 per-scan 1-D z = documented degenerate
                                 class (record in header); final method = cross-scan
                                 common-mode rejection: the known birdie (-168.6 km/s, l=5)
                                 is the largest scan-unique feature (10.7 sigma,
                                 leave-one-out 10.8; common-mode 17-40 sigma collapses <4);
                                 `results/sarthak_birdie_transfer.json`
13. `arm_optimize.py`           — data-driven spiral-arm fitting (spiral paper)
14. `november_prediction.py`    — pre-registered November predictions
15. `november_lsr_disc.py`      — cross-month LSR discriminator (astropy ephemeris)

External-code replications (run separately, see manuscript reproducibility note):
- RFI-NLN original code: `18_CODE_REPOS/RFI-NLN/` (TF 2.21 + tf-keras; patches disclosed)
- anomalib official PatchCore: `anomalib_patchcore_lofar.py` (py3.12 env)
- RFI-HWT original code: `18_CODE_REPOS/RFI-HWT/` (authors' weights)

## Data
- LOFAR + HERA benchmarks: Zenodo record 8275061 (RFI-NLN distribution)
- LAB survey cube, community corpora: each under its own licence
  (paths in the scripts; adjust `PD`/`D` constants for your mount)

## Environment
Python 3.14: numpy, scipy, scikit-learn, torch (CPU sufficient), astropy,
matplotlib, faiss-cpu.  The anomalib replication needs Python 3.12 +
`anomalib==2.7.0`.  The RFI-NLN replication needs `tensorflow>=2.16`,
`tf-keras`, `faiss-cpu`.  The SAM arm needs `segment-anything` + the ViT-B
checkpoint (`F:/21cm/cache/sam_vit_b_01ec64.pth`); CPU works (~2.3 h),
CUDA (~7 min, `SAM_DEVICE=cuda SAM_OUT=sam_arm_rsp1_gpu.json`).

## Provenance
Every table/figure number in the manuscript traces to one of the JSON files
in `results/`.  The mapping is given in the repository README.
