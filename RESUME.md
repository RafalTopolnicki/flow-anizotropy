# Resume point — 2026-10-14/15 (session 20)

Transcript: `conversations/session20_2026-10-14.{md,jsonl}`. Memory: `physics_baselines.md`, `real_rock_2d_feasibility.md`.

## 000000. Physics baselines DONE on both sets + 11 Sandstones pilots

### classical+tda / classical+tda+thr (new groups in train_catboost, phys_table, run_catboost_phys.sh)
* **Berea** (Δ R^2 vs `classical`, folds won): classical+tda K_xx +.001 (2/5), k_xy **+.014 (5/5)**,
  k_xz -.001, k_yz **+.016 (5/5)**; classical+thr +.002/+.002/+.009 (4/5)/+.008 (4/5).
  On Berea the CONE survives the physics baselines, the throat barely.
* **aniso3d DONE 2026-10-15 00:20** (`python scripts/phys_table.py RESULTS/aniso3d_all --ref classical`):
  diagonals saturated (~.99, nothing separates). Off-diagonals k_xy/k_xz/k_yz: classical
  .824/.812/.825; **thr alone beats classical** +.023 (5/5)/+.011 (4/5)/+.026 (5/5);
  **classical+thr +.036/+.033/+.033, all 5/5**; classical+tda only +.007/+.016/+.014.
  KC is weak off-diagonal here (.74), unlike Berea. So: synthetic -> throat, Berea -> cone.

### 11 Sandstones (DRP-317) — downloaded, recipe being chosen
* `scripts/fetch_sandstones11.sh` -> `DATA/rock/sandstones11/` (11 x 1000^3, 2.25 um, 0 = pore,
  anonymous API, recipe in memory). All 11 present.
* `scripts/stretch_rock.py`: NEW `--rock <name>` (registry ROCKS), pilot grids s x g (`--grows`).
  Regression on Imperial Berea s=2 g=4 reproduces por .62 / dt50 2.3.
* Pilots (DATA/rock_pilot_sst11{,_g,_s3}/<rock>/pilot.csv + slices.png):
  1. s 3-6 x g 0,4,8,12 (Parker, CastleGate, Berea): g=0 pores 1 vox; s=5 cannot give por ~.6 with dt50>=2.
  2. s=5, g 5-7: confirms; porosity and dt50 rise together.
  3. **s=3, g 4-6, six rocks**: g=5 gives por .59-.63 / dt50 2.2-2.4 on Kirby, Berea,
     Bentheimer, CastleGate (= rock_berea's .62/2.3); Parker needs g=6 (.60/2.3), Bandera Gray
     g~3-4 (.67 already at g=4). Axis error 1-4 deg, all percolate (Parker g=5: 19/20).
* **DECIDED 2026-10-15 (user): g per rock, porosity target .60; 400 cubes/rock; LBM on ANOTHER machine.**
* **RUNNING since 2026-10-15 08:26** on c3n1: `nohup bash run_sst11_generate.sh > SST11_GEN.log`
  = pilot 4 (s=3; g 3-6 on BanderaBrown BSG BUG BB Leopard, g 2-3 on BanderaGray ->
  DATA/rock_pilot_sst11_s3b) -> `scripts/choose_grow.py` (closest por to .60 among g where all
  pilot cubes percolate -> DATA/rock_sst11_grow.json) -> `stretch_rock.py --generate` per rock
  (sample_id = rock_index*400 + i, rock order BanderaBrown BanderaGray Bentheimer Berea BSG BUG BB
  CastleGate Kirby Leopard Parker; parts in DATA/rock_sst11/parts/<rock>, logs GEN_<rock>.log)
  -> `--merge` into DATA/rock_sst11 (4400 cubes, same layout as rock_berea). ETA ~2-3 h.
  CHECK: Bandera Gray's chosen g (its pores were only 1.9 vox at g=4), "SST11 GEN DONE" in log.
* **DONE 2026-10-15 09:18: DATA/rock_sst11 = 4400 cubes (11 x 400), ids 0..4399 unique, 2.2 GB.**
  BSG died silently in the first pass (likely OOM, 6 SDFs at once) -> `regen_sst11.sh` redid
  BSG, BanderaGray (user override g 3->4) and Leopard (override g 6->7); overrides recorded in
  DATA/rock_sst11_grow.json (`rule_grow`). Mean porosity .57-.68 per rock (BanderaGray .68 the
  outlier), dt50 1.9-3.0. Parker is heterogeneous: porosity sd .11, min .23, 33 cubes < .4,
  correlated with cube position (x -.33, y +.41) -- a low-porosity zone in the rock, kept.
* **LBM scripts written 2026-10-15 (NOT run here)** — user runs them on TWO other 52-CPU
  machines (same type as c3n1, shared disk), from the repo dir:
      A: nohup bash run_lbm_sst11.sh 0 > LBM_SST11_part0.log 2>&1 &
      B: nohup bash run_lbm_sst11.sh 1 > LBM_SST11_part1.log 2>&1 &
      progress: bash lbm_sst11_status.sh ; after both "PART n DONE": bash merge_lbm_sst11.sh
  Part 0 = even sample_id, part 1 = odd (2200 each, 200 per rock). Needs `python` with
  numpy/pandas/tqdm (conda env madrid, or PY=...). Existing LMB3d/lbm3d-perm, no rebuild.
  NEW in solve_permeability3d.py: `--shard I/N` (solve/verify one part, never merges, repair
  only touches its own .tmp files -- that last property NOT tested, the test was interrupted)
  and `--interleave rock` (round-robin solve order; labels unaffected).
  ETA ~35 cubes/h per machine (Berea rate on c3n1) -> ~63 h ≈ 2.6 days per part.
* **Leave-one-rock-out WRITTEN 2026-10-15:** `train_catboost.py --group-cv rock` = one fold per
  rock (11), `held_out` recorded per fold in metrics.json, reports per-fold R^2 (`r2_mean`,
  within-rock variance -> harder) AND `r2_pooled`/`mae_pooled` (all out-of-rock predictions
  together, same scale as k-fold). Works single and --multioutput; aborts if the column is
  missing or is a feature. Tested: k-fold unchanged (kc K_xx Berea folds identical to the
  recorded run); group mode on Berea with --group-cv stratum (3 folds; held-out `strong` worst).
  TODO: phys_table.py hardcodes "x/5" folds -> needs a LORO table (per-rock rows) once labels exist.
* Then: descriptors for rock_sst11 (run_rock_descriptors.sh pattern), CatBoost --group-cv rock.

---
# Resume point — 2026-10-07/08 (session 19)

Transcript: `conversations/session19_2026-10-07.{md,jsonl}`. Memory: `physics_baselines.md`.

## 00000. Physics baselines (Kozeny-Carman + critical path) — the paper-prep step 1
* Paper plan discussed (companion to arXiv:2604.08105): next = physics baselines (DONE below),
  then multi-rock set with leave-one-rock-out (check data access first: 11 Sandstones /
  DRP-317), 3D CNN learning curve with fixed step budget, repeated CV + saved OOF
  predictions, SHAP by direction/homology dim, small 80^3-vs-160^3 label resolution check.
  Venue: AWR or TiPM (PRE if no multi-rock).
* NEW `scripts/physbaselines3d.py` (--self-test 15/15) -> `DESC/{rock_berea,aniso3d}_phys/
  physbaselines.csv` (38 f: kc_ + cp_). NEW groups in train_catboost: kc cp phys phys+thr
  classical (= baselines+kc+cp) classical+thr. Driver `run_catboost_phys.sh` (log
  `CB_PHYS.log`, resumable), table `python scripts/phys_table.py <root> [--ref classical]`.
* **Berea DONE**: KC alone beats thr on K_xx/k_yz; thr over `classical` only +.002..+.009.
* **Synthetic aniso3d (6 targets) RUNNING** since 2026-10-08 00:41, ETA ~04:00, results into
  RESULTS/aniso3d_all/<T>__<group>. This decides the framing: does thr's +.04-.05
  off-diagonal margin survive `classical`? Read with
  `python scripts/phys_table.py RESULTS/aniso3d_all --ref classical`.

---
# Resume point — 2026-10-04 (sessions 17-18)

Transcripts: `conversations/session17_2026-10-04.{md,jsonl}`, `conversations/session18_2026-10-04.{md,jsonl}`. Rock VTKs: VIS/rock_berea_vtk/.

## 0000. Berea LBM DONE 2026-10-04 — permeability.csv merged 17:13, 1000/1000 converged
* 2026-10-04 21:59: rock CatBoost DONE 36/36. Table (with CNN rows): RESULTS/rock_berea/table.md.
  tda/thr beat baselines on all 4 targets (+.002..+.025); best union beats CNN on K_xx (.920 vs .811)
  and k_yz (.888 vs .877 joint), CNN wins k_xy (.873 vs .862) and k_xz (.907 vs .892).
* 2026-10-04 19:09: rock descriptors DONE (cone 5787f, throat 2700f, iso 300f, 0 NaN).
  CNN DONE (RESULTS/rock_berea/cnn3d). 19:34: CatBoost LAUNCHED on c5n1,
  `nohup ./run_catboost_rock.sh > CB_ROCK.log` -> RESULTS/rock_berea/catboost/<T>__<group>,
  single-target K_xx k_xy k_xz k_yz x por tpc fab fabdir baselines tda thr all baselines+thr. Resumable.
* 2026-10-04 18:26: descriptors LAUNCHED on c5n1 (192 cores),
  `SCRATCH=/tmp/rtopolnicki/rockdesc WORKERS=192`, log `ROCK_DESC.log`. Baselines done
  (1000x1819, 0 NaN). Rerun the same command to resume. CNN still not launched (GPU node).
* CNN: NEW `run_cnn3d_rock.sh` (GPU node g6n2, RTX 5090): single K_xx k_xy k_xz k_yz
  (roll) + joint offdiag roll+oct -> RESULTS/rock_berea/cnn3d/. Smoke-tested, NOT run yet.
* Descriptors: `run_rock_descriptors.sh` (below), to run on the 192-core machine.

History of the interruption:
* `DATA/rock_berea/perm/`: **782/1000 solved, all converged** (fx/fy/fz).
  Missing 218: id 781 and 783-999. No `.tmp` leftovers.
* 2026-10-04 16:03: relaunched by the user on a 192-core machine (same disk), log
  `ROCK_LBM_resume2.log`; 810/1000 at 16:04 (its tqdm bar counts cached cubes too).
* Descriptors: NEW `run_rock_descriptors.sh` (SCRATCH=<local disk> required) ->
  `DESC/rock_berea_{cone,throat,throat_iso}`; cone PH imaging re-measured on Berea,
  see its header. Smoke-tested on 4 cubes. User runs it on the 192-core machine.
* (was: NOT relaunched (user's call).) Resume (skips finished cubes), ~6-7 h at ~35/h:
      nohup python -u scripts/solve_permeability3d.py --dataset DATA/rock_berea --workers 52 > ROCK_LBM_resume.log 2>&1 &
  then `--merge-only`, descriptors, CatBoost with por/tpc/fab/fabdir rows.
* k_yz learning curve still not run; do not overlap it with the LBM (both take all cores).

---
# Resume point — 2026-09-30 (end of session 16)

Read `NOTES.md` §12 for the findings; this file is the operational state.
**§000 below supersedes everything older wherever they disagree.**
Transcript: `conversations/session16_2026-09-30.{md,jsonl}`.

---

## 000. State as of 2026-09-30 — READ THIS FIRST

### Sample-size study
* **K_xx DONE** 2026-09-29 20:49 (270/270 both arms, `RESULTS/lc3d_kxx/table.md`).
  thr 30/30; +.26 at n=125 -> +.024 at 4000; reaches baselines@4000 with ~360.
  Cone tda does NOT cross over on K_xx. tpc alone beats the baselines union.
  Plots made: `RESULTS/lc3d_plots/lc3d_kxx_fixed_{r2,mae}.png` (R^2 y-axis
  squashed by fab<0; user: fine for now). k_yz still not run.

### Real-rock 3D dataset — LBM RUNNING
* Berea re-downloaded to `DATA/rock/berea/` (curl needs `-A "Mozilla/5.0"`).
* NEW `scripts/stretch_rock.py`: affine stretch F = R diag(a,b,1) R^T (a>=b>=1),
  downscale s, grain shave g (SDF threshold at -g native voxels), one resampling.
  `--pilot` output in `DATA/rock_pilot/` (s x stretch grid, slices.png).
* User's choices: **s=2 (10.69 um), g=4, 80^3, n=1000**, a log-U[1,3],
  b log-U[1,a]. Porosity ~.62, median pore half-width 2.0-3.3 vox, all
  percolate. Strata weak/mod/strong by a (<1.5, <2.2, else).
* `DATA/rock_berea/` generated; **LBM started 2026-09-30 18:06**, 52 workers,
  log `ROCK_LBM.log`, ETA ~10 h (restartable: rerun the same command).
      nohup python -u scripts/solve_permeability3d.py --dataset DATA/rock_berea --workers 52 > ROCK_LBM.log 2>&1 &
* **Condition 1 PASSED** on the first 52 (19:40): median |k_off|/trace .027 vs
  .018 for synthetic 3D (share > .05: 25% vs 12%); median k1/k3 1.70 vs 1.43;
  reciprocity error ~2e-6; k1 axis within median 8 deg of the stretch axis;
  corr(log k1/k3, log a) .85. All converged. (The .05 bar was 2D-derived and too
  strict: the synthetic set that TDA won on clears it only 12% of the time.)
* Rate: ~35 cubes/h (52 per ~88 min; 52 procs on 26 physical cores with HT).
  156/1000 at 22:58 -> **ETA ~23:00 Thu 2026-10-01**. The tqdm ETA is too
  optimistic (the cubes finish in waves).
  Then merge (`--merge-only`), descriptors, CatBoost with por/tpc/fab/fabdir rows.
* Caveat: at s=2 all cube centres lie in a ~117^3 box of the 400^3 rock -> every
  cube overlaps. Hold-out by rock needs 11 Sandstones (DRP-317), next rock source.


---
## Older resume point — 2026-09-29 (end of session 15)

Read `NOTES.md` §12 for the findings; this file is the operational state.
**§00 below supersedes everything older wherever they disagree.**

---

## 00. State as of 2026-09-29 — READ THIS FIRST

### Sample-size study (3D, `run_lc3d.sh`)
* k_xy DONE 2026-09-23 (NOTES §12.11); **k_xz DONE 2026-09-29 07:28** (540/540,
  NOTES §12.12): replicates k_xy, thr 30/30 per arm. On MAE the cone `tda` does
  NOT cross over (wins 5/5 from n=250 to 4000; it loses on R^2 at 4000), and the
  thr margin in MAE barely shrinks with n. MAE table:
  `RESULTS/lc3d_kxz/table_mae_fixed.md` (`lc_table.py --metric mae`).
* **K_xx RUNNING** since 2026-09-29 11:43 (`TARGET=K_xx OUT=RESULTS/lc3d_kxx`,
  log `LC3D_KXX.log`), ETA ~19:30-20:00. Takes all cores. When done: check
  270/270 in both arms, then `python scripts/plot_lc.py --targets K_xx k_xy k_xz`.
* k_yz not run.
* NEW `scripts/plot_lc.py`: R^2 and MAE vs n PNGs per target ->
  `RESULTS/lc3d_plots/lc3d_<tag>_<arm>_<metric>.png` (por omitted, range in
  footer; grey dashed = tpc/fabdir/fab).
* Still uncommitted: `splits/aniso3d_golden.*` (frozen hold-out), `run_lc3d.sh`,
  `scripts/lc_table.py`, `scripts/plot_lc.py`.

### Generalizability — the next direction (discussion, no code yet)
* The user wants real structures. **2D is dropped**: Berea slices need 8 px pore
  dilation to percolate (porosity -> .73) and are less anisotropic than our iso
  stratum. Details in memory `real_rock_2d_feasibility.md`.
* Rejected by the user: channel networks (artificial), grain packings (too
  easy), foams (no percolation in 2D, no natural anisotropy).
* Periodicity: agreed a non-periodic real image is fine on the LBM torus; the seam
  bias scales ~ correlation length / window. Mirror padding and no-flow walls both
  kill k_xy, so periodic BCs are the right choice for tensors on real rock.
  Descriptors must stay periodic to match the solver.
* **3D real rock, open decisions for the user:**
  1. Anisotropy source: A natural rocks only (headline = diagonal/principal
     values), B affine-stretched real rock (topology preserved, designed A and
     psi), **C both (recommended)**.
  2. Data: 11 Sandstones (DRP-317, DOI 10.17612/F4H1-W124, now on
     digitalporousmedia.org, may need login) + Imperial Berea (downloaded to the
     session scratchpad only; direct zip link in the memory file).
* Berea 80^3 facts: 95% percolate natively, but ~2.9 grains/box at 5.35 um and
  pore dt p50 1.5 vox; 2x down gives 5 grains but dt p50 1.0. -> need ~160^3 at
  ~5 um. Next: resolution/timing LBM pilot (K at 5.35 vs 10.7 um, 160^3 cost),
  RCAP/PH ranges re-derived by rule, split by rock. Zero-shot synthetic->rock is
  uninformative (porosity .2 vs .65-.90; trees can't extrapolate).


---
## Older resume point — 2026-09-22 (end of session 12)

Read `NOTES.md` §12 for the findings; this file is the operational state.
Sections 1-5 below are the session-09/10 record and are kept for context —
**§0 supersedes them wherever they disagree.**

---

## 0. State as of 2026-09-22 — READ THIS FIRST

### The 3D LBM solve FINISHED

`DATA/aniso3d/permeability.csv` merged 2026-09-22T03:26, **5000/5000 rows,
conv_fx/fy/fz/all all 5000**, sample_id 0..4999 contiguous. Everything §3b
describes as "continuing the solve" is done.

### Descriptors extended to 5000 — incrementally, not rebuilt

Only structures [2816, 5000) were computed; the existing 2816-row tables were
kept and joined. 09:05->12:24, **3 h 19 min** against ~7.2 h for a rebuild.

| root | rows | features | seam max\|z\| |
|---|---:|---:|---:|
| `DESC/aniso3d` (cone OLD imaging + ECP) | 5000 | 5787 | never split |
| `DESC/aniso3d_throat` | 5000 | 2700 | 4.08 |
| `DESC/aniso3d_throat_iso` | 5000 | 300 | 2.69 |
| `DESC/aniso3d_ph_fixed` (re-imaged cone) | 5000 | 5787 | 4.00 |

**How the incremental path works** — `FROM=2816 ./run3d_all.sh`. The `.npy`
intermediates are deleted once folded into the CSV, so the drivers' resume
check (which counts `.npy`) cannot see an already-collected head and would
recompute everything. `FROM` starts the chunk loop at the tail, the collect
stage writes `descriptors_part_<FROM>_<LIMIT>.csv` instead of overwriting
`descriptors.csv`, and `scripts/merge_descriptors.py` joins the parts.

**The seam is the risk, and it is checked.** Six fatal checks (identical column
order, disjoint ids, complete [0,N), metadata matched against structures.csv,
no NaN/inf) plus a seam statistic: per-feature standardized mean difference
between the parts, as a z, aborting above `--seam-z 8`. Calibration: a seam
verified BITWISE-identical to a one-pass build measured max|z| 3.11, and the
three real seams came in at 2.69-4.08. `--self-test` is 20/20. Every root
carries `descriptors.csv.parts.json` and both part files.

`ECP is never recomputed` — `DESC/aniso3d/ecp` holds all 5000 `.npy` and is
symlinked into the cone rebuild.

### 3D model results were DELETED, deliberately

`RESULTS/aniso3d{,_conefixed,_throat,_throat_iso}` removed 2026-09-22 at the
user's request: every number in them was at n=2816 and R^2 moves with n.
**NOTES §12 is now the only record of them.** Nothing 2D was touched.

### The 3D CNN is running ON A DIFFERENT MACHINE

It writes into this repo as it goes — `RESULTS/aniso3d/cnn3d/`. Do not delete
it, and do not launch a competing GPU job. Single-target, densenet121,
augment=roll, 30 epochs, batch 8, folds 5, seed 0, n=5000. Done so far:
K_xx .9959, K_yy .9960, K_zz .9953, k_xy .9681.

Diagnostics on the diagonals (from `oof_predictions.npy`): a systematic
over-prediction of +0.14 (~65% of the MAE, identical across all three targets,
debiasing would gain +.0007) and a fit slope of .984 (MSE shrinkage). Absolute
error grows 2.2x from the lowest to the highest porosity quartile while
RELATIVE error falls 5.3% -> 2.8%.

### The 2D arm gained separated baseline rows

Standing instruction 2026-09-22: **every table with `baselines` must also carry
`por`/`tpc`/`fab`(/`fabdir`) rows.** `run_2d_all.sh`, `run_2d_k0.sh`,
`run_2d_porosity.sh` and `train3d_throat.sh` now default to it. All four 2D
result sets were backfilled; `RESULTS/summary.csv` has the rows.

Why it matters: at n=2816 on `k_xy`, `tpc` ALONE scored .7437 against the
1819-feature `baselines` union's .7653, `fabdir` .5387, `fab` .4990, `por`
**-.0014**. Every "TDA vs baselines" claim is really a claim against `tpc`.
In 2D on k_off: `tpc` .6842 of `baselines` .7142, and `por` .4072 — the
2D/3D `por` gap is unexplained (same descriptor definition; the 3D slab
averages 6400 voxels against 2D's ~512 pixels, untested).

### `RESULTS/learning_curve/` was DELETED

The sample-size study will be recomputed. `splits/aniso_golden.csv` + `.json`
survive (1001 held out, seed 12345, sha256 f31fda65…), so a repeat stays
paired. Recorded cost of the deleted arms: es 1.5 h, fixed 2.6 h, cnn 0.4 h.
Before repeating: it read the PRE-bandwidth-fix `DESC/aniso`, has no `thr`
group, and `--pca 64` was never run. NOTES §10 still holds its tables.

### Update 2026-09-23 — ALL 3D jobs complete, nothing running

* Single-target CatBoost pass finished 04:08 (54/54). Table in NOTES §12.10.
* `run_3d_porosity.sh` was run (elsewhere) and copied back: 36/36 in
  `RESULTS/aniso3d_por/`. No porosity dependence of thr's margin (NOTES §12.10);
  the data spans only porosity .65-.90, so percolation is untested.
* NEW: 3D sample-size study on k_xy, `run_lc3d.sh` (11:54->20:16, 540 fits).
  Hold-out `splits/aniso3d_golden.{csv,json}` (999, sha256 2609c9ef…) — frozen,
  commit it. Results `RESULTS/lc3d_kxy/{es,fixed}/runs.csv`, tables
  `RESULTS/lc3d_kxy/table.md` via `python scripts/lc_table.py
  RESULTS/lc3d_kxy/es RESULTS/lc3d_kxy/fixed`. Findings NOTES §12.11: thr beats
  baselines at every size (30/30 per arm), margin largest at small n, ~half the
  data to match baselines@4000; cone tda wins at 250-1000 and loses at
  2000-4000. `scripts/learning_curve.py` now aborts on NaN descriptors.
  Its stdout is block-buffered under nohup — watch runs.csv, not the log.

### Job state at 2026-09-22 23:00

**3D CNN — COMPLETE**, all six at n=5000, on ANOTHER machine writing into
`RESULTS/aniso3d/cnn3d/`: K_xx .9959, K_yy .9960, K_zz .9953, k_xy .9681,
k_xz .9563, k_yz .9684. Single-target, densenet121, augment=roll, 30 epochs,
batch 8, folds 5, seed 0.

**3D CatBoost joint pass — COMPLETE**, 9/9 groups, 14:19->21:27 (~7 h), in
`RESULTS/aniso3d_all/tensor_joint__<group>/`. Headline, n=5000, joint scaled:

    group           feat    K_xx    K_yy    K_zz    k_xy    k_xz    k_yz
    por              720   .7719   .7716   .7727   .0590   .0479   .0488
    tpc              369   .9639   .9644   .9634   .7497   .7463   .7469
    fab               10  -.0508  -.0603  -.0615   .5029   .4931   .5044
    fabdir           720   .4616   .4564   .4537   .5436   .5328   .5407
    baselines       1819   .9544   .9551   .9531   .7725   .7587   .7708
    tda             5787   .9583   .9608   .9587   .7393   .7536   .7689
    thr             2700   .9830   .9839   .9822   .8011   .7788   .8111
    all             7606   .9690   .9706   .9679   .7861   .7867   .8037
    baselines+thr   4519   .9862   .9867   .9856   .8111   .7924   .8268

`thr` alone beats `baselines` on all six; `baselines+thr` wins everything at
4519 features, beating the 7606-feature `all`. `tpc` (369f) beats the whole
1819-feature `baselines` union on the diagonals — the union dilutes it.
NOTE the 2D sign flip: there `thr` LOSES to `baselines` on k_off and
`baselines+tda` is the best union.

**3D CatBoost single pass — RUNNING**, started 21:27 by `/tmp/flow3d/chain.sh`
(which waited on the joint pass, checked for its DONE marker, then launched
`PASSES=single`). 54 fits = 6 components x 9 groups, into
`RESULTS/aniso3d_all/<target>__<group>/`. This is the pass that pairs with the
CNN. The chain is the PARENT of the running pass, so killing the pass means
killing the chain too.

**`run_3d_porosity.sh` — WRITTEN, NOT RUN.** Does the throat margin grow as
porosity falls, in 3D? Refits existing DATA/aniso3d inside porosity quartiles
(cuts .713381/.774594/.838269, n=1250/band), 9 groups x 4 bands on k_xy, ~1.5-2
h, ZERO LBM. The 2D answer was NO (thr margin +.0341 at q4 -> -.0045 at q1,
with `tda` showing the same gradient), which may not transfer: planar duality
forbids both phases percolating in 2D, and connectivity is the proposed
mechanism for `thr` winning in 3D at all. Read `tda` as the control; R^2 is NOT
comparable across bands (k_xy sd .39 -> 1.03).

To run it on ANOTHER machine, copy (~719 MB, preserving paths):
`DATA/aniso3d/permeability.csv`, `DESC/aniso3d_ph_fixed/descriptors.csv`,
`DESC/aniso3d_throat/descriptors.csv`, `DESC/aniso3d/baselines.csv`,
`scripts/{train_catboost,metrics,collect_results}.py`, `run_3d_porosity.sh`.
No structures needed. Copy back `RESULTS/aniso3d_por/` and rerun
`collect_results.py`. Use `rsync -L` if copying whole DESC dirs --
`DESC/aniso3d_throat/baselines.csv` is a symlink.

**The user does NOT want new structures generated for now.**

### What to run next — see the end of this file, §7

---

## 1. The headline, in one paragraph

A **throat-scale persistent-homology descriptor** — the Euclidean distance
transform of the pore space as the filtration function, instead of the cone's
occupancy fraction — **beats every non-topological baseline on all six
components of the 3D permeability tensor**, 5/5 folds on each, p ≤ .016, using
2700 features against the baselines' 1819. It also beats `all` (7606 features)
on all six. This is the first time any topological descriptor has won anything
in this study. In 2D the same descriptor only *drew* with the baselines, which
is consistent with the mechanism: planar duality forbids pore and solid from
both percolating in 2D, so connectivity cannot vary much; in 3D it can.

Two qualifiers that must travel with the claim:

* "TDA" means the **throat** descriptor. The study's original directional
  **cone** descriptors still lose, even after their representation was
  repaired (`tda` .7513/.7577/.7816 vs `baselines` .7653/.7598/.7809).
* The **CNN still dominates** — `cnn3d_densenet121` reaches .958/.941/.964 on
  the off-diagonals against `thr`'s .808/.803/.830.

---

## 2. What survives the machine, and what does not

**Durable (in this repo, under `/home`):**

| path | what |
|---|---|
| `NOTES.md` §12 | the full write-up |
| `conversations/session09_2026-09-18.{md,jsonl}` | this session's transcript |
| `scripts/filtration_throat{2,3}d.py` | the new descriptor, self-tests included |
| `scripts/ph3d.py` | now with `--weight` / `--bandwidth` / per-dim ranges |
| `run_throat3d.sh`, `run_cone_redo3d.sh`, `train3d_throat.sh`, `run3d_all.sh` | the drivers |
| `DESC/*/descriptors.csv` | **the durable descriptor artefacts** |
| `RESULTS/aniso3d_throat/`, `RESULTS/aniso3d_conefixed/` | R² + per-fold metrics |
| `memory-snapshot/` | copy of the agent memory (the original lives in `~/.claude`, which does not survive) |

**Lost with the machine:** every per-structure `.npy` under the scratch
directory (~413 MB — the throat and cone-redo persistence images). They are
*intermediates*: `collect_descriptors.py` already folded them into
`descriptors.csv`, which is what every consumer reads. Losing them only costs
recompute if you want to change imaging parameters or add a direction.

**Not committed.** Everything above is uncommitted working tree as of writing
(the user asked for no commits). `git remote` is
`git@github.com:RafalTopolnicki/flow-anizotropy.git`. **If `/home` goes with
the machine, pushing is the only way any of this survives** — with two catches:

* `DESC/*.csv` are 10–330 MB each and exceed GitHub's 100 MB limit, so the
  descriptor tables cannot be pushed as-is. They are regenerable (§3).
* **`.gitignore` excludes `RESULTS/`**, so `RESULTS/aniso3d_throat/` and
  `RESULTS/aniso3d_conefixed/` — every R² and per-fold metric from the 3D run,
  only ~90 KB — will NOT be pushed unless force-added:
  `git add -f RESULTS/aniso3d_throat RESULTS/aniso3d_conefixed`.

Also staged as of 2026-09-18: the three solver binaries
(`LMB2d/lbm2d-perm`, `LMB2d/lbm2d-perm-gl`, `LMB3d/lbm3d-perm`) were removed
from the index with `git rm --cached` and remain on disk. They leave the repo
on the next commit.

---

## 3. How to regenerate the descriptors from scratch

All three stages resume; rerunning skips completed work.

```bash
./run3d_all.sh          # throat build -> cone re-image -> training
```

Costs, measured on 52 cores, n=2816:

| stage | cost |
|---|---|
| throat build, 9 directions + iso | ~2.0 h (PH ~17.5 s/structure) |
| cone re-image, 9 directions | ~2.1 h (~14 min/direction) |
| training, 6 targets × 2 group sets | ~7.8 h (~78 min/target) |

**Set `TMPROOT` and `NPYROOT` to a scratch path on the new machine** — they
default to this session's scratchpad directory, which will not exist. Both
drivers take them as environment variables:

```bash
TMPROOT=/scratch/filt3d NPYROOT=/scratch/npy ./run3d_all.sh
```

---

## 3b. Continuing the 3D LBM solve on another machine

State as of 2026-09-18 (`--verify` output, all 2816 validated intact):

    DATA/aniso3d: 5000 structures
      complete   2816      <- contiguous sample_id 0..2815
      incomplete 0
      missing    2184      <- 2816..4999
      stray .tmp 0

**What to copy to the solving machine:**

| path | why |
|---|---|
| `DATA/aniso3d/structures/` | the 5000 `.raw` inputs (~2.5 GB) |
| `DATA/aniso3d/structures.csv` | the manifest the driver iterates |
| `DATA/aniso3d/perm/` | the 2816 finished per-sample CSVs — **copy these or you redo them** |
| `LMB3d/lbm3d-perm.cpp`, `LMB3d/make_perm.sh` | rebuild the solver there |
| `scripts/solve_permeability3d.py`, `scripts/metrics.py` | the driver |

The compiled binary `LMB3d/lbm3d-perm` was untracked from git on 2026-09-18
(`git rm --cached`, file kept on disk) — rebuild it on the target machine with
`LMB3d/make_perm.sh` rather than copying an executable across architectures.

**Then:**

```bash
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --verify
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers N
```

It prints `5000 structures: 2816 already solved, 2184 to do` and solves only
the missing ones, merging to `DATA/aniso3d/permeability.csv` at the end.
Interrupt and rerun as often as you like.

**If a run is killed**, clear anything half-written and resume:

```bash
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --verify --repair
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers N
```

Solves now write to `<out>.tmp.<pid>` and `os.replace()` into place only after
validation, so the final path never holds a partial solve (NOTES 12.8 and the
script docstring). Before 2026-09-18 the solver wrote in place and the cache
test was `size > 0`, so a kill mid-write would have poisoned that sample
permanently — your kill happened to land cleanly between samples.

**If the solve reaches 5000**, every 3D number in NOTES §12 must be recomputed:
the descriptors were built with `--limit 2816`, and R² moves with sample size,
so the recorded comparisons are not valid at a different n. Rebuild with
`LIMIT=5000 ./run_throat3d.sh` and `LIMIT=5000 ./run_cone_redo3d.sh` (both
resume), then rerun `train3d_throat.sh`.

## 3c. The 2D re-run (session 10, 2026-09-18) — STATE

The 2D arm was recomputed from scratch, CatBoost only. The CNN was deliberately
not run and `RESULTS/cnn/` was deliberately not deleted, so those numbers still
stand but have no new counterpart.

**Deleted** (backed up only to a session scratchpad, which does not survive):
`RESULTS/{catboost,throat,void,connect,void_connect,ph_bandwidth,bothphases}`.
**Kept:** `RESULTS/cnn/` and `RESULTS/learning_curve/` (the latter holds a
`cnn_densenet121/` arm of the sample-size study and nothing regenerates it).

**Drivers:** `run_2d_all.sh` (main), `run_2d_porosity.sh` (stratified, not yet
run). Env knobs: `ITER`, `OUT`, `FEATGROUPS`, `PASSES`, `SCALE`.

### Recorded, `RESULTS/aniso2d/` — n=4991, 5 folds, seed 0, 1500 iters

| group | feat | k_xx joint | k_yy joint | k_off joint | k_off single |
|---|---|---|---|---|---|
| baselines | 779 | .9205 | .9274 | .6794 | .7142 |
| tda | 3524 | .8988 | .8937 | .6376 | .6607 |
| thr | 3200 | .9248 | .9259 | .6559 | .7060 |
| all (TDA+base) | 4303 | .9579 | .9570 | .7313 | .7549 |
| baselines+thr | 3979 | .9463 | .9486 | .7125 | .7614 |
| all+thr | 7503 | .9497 | .9520 | .7222 | .7632 |

`all+thr` single .7632 is the best descriptor k_off on record but buys nothing
over the smaller unions: +.0018 vs `baselines+thr` (2/5, p=.78), +.0083 vs
`all` (4/5, p=.14). In the JOINT fit it actively hurts (.7222 vs `all` .7313;
k_xx .9497 vs .9579) — feature dilution at 7503 features.

### Scaled joint pass, `RESULTS/aniso2d_scaled/` (--scale-targets)

k_off gains in EVERY group, 5/5 folds, all significant: baselines +.0296
(p=.0001), tda +.0158 (p=.033), thr +.0401 (p=.0017), all +.0280 (p=.0025),
baselines+thr +.0308 (p=.0009), all+thr +.0265 (p=.0035). The diagonals pay
little: `all` -.0109 k_xx / -.0054 k_yy; `baselines+thr` nothing measurable
(-.0003, p=.65). **Report the SCALED joint numbers** — the unscaled joint table
understates k_off by .016-.040 and reorders the groups.
Scaling does not fully close the gap to a single fit (`baselines+thr` still
-.018, p=.016; `all+thr` -.015, p=.022), so variance imbalance was most of the
joint penalty but not all — the rest is genuine capacity sharing. For `all` it
more than closes it (.7593 scaled-joint vs .7549 single, p=.47).

Paired 5-fold tests on k_off single: `baselines` > `tda` +.053 (5/5, p=.041);
`baselines` vs `thr` +.008 (3/5, p=.57, a draw); `all` > `baselines` +.041
(5/5, p=.005); `baselines+thr` > `baselines` +.047 (5/5, p=.010);
`baselines+thr` vs `all` +.0065 (3/5, p=.35, a draw).

**So in 2D neither descriptor family beats the baselines alone, both add on top
of them, and the two unions are indistinguishable.** `tda` here is the
BANDWIDTH-REPAIRED cone (`DESC/aniso_ph_bw125`), so it loses on merit, not by
construction. Two anchors reproduced exactly: `baselines` k_off single .714174
matches the deleted `k12__baselines` digit for digit, and `all` single .7549
reproduces the .754 recorded as the dataset ceiling.

### The joint fit under-weights k_off — measured

MultiRMSE minimises the SUMMED squared error on raw scales and k_xx carries
17.8x k_off's variance, so the diagonals dominate. Cost on k_off, joint vs
single: baselines -.035, tda -.023, thr -.050, all -.024, baselines+thr -.049,
all five significant (p <= .046). It is NOT uniform — it hits the throat groups
about twice as hard — and it REORDERS the top: single says `baselines+thr`
leads, joint says `all` leads (+.0187, 5/5, p=.063). Treat any joint k_off
ranking as suspect.

`--scale-targets` was added to `train_catboost.py` for this: z-scores each
target on the TRAIN fold only, inverts on the predictions so all metrics stay
in raw units. Only affects `--multioutput`; warns and no-ops otherwise.
`train_cnn.py` ALREADY did this correctly and always has (`run_fold`,
`mu, sd = Y[tr].mean(0), Y[tr].std(0)`, inverted at return), so no recorded CNN
number is affected; `--no-scale-targets` was added there only to measure what it
is worth. There is no val split in the CNN — fixed epochs, no early stopping —
so train/test is the only boundary the scaler must respect.

### All 2D work of session 10 is COMPLETE

`run_2d_all.sh` (both passes, six groups), the scaled joint pass, and
`run_2d_porosity.sh` all finished. Nothing is running. No CNN was run.

### A bash trap that cost two false starts

**Never name a shell variable `GROUPS`.** Bash reserves it for the caller's Unix
group IDs and SILENTLY DISCARDS assignments, so `GROUPS="all+thr" bash ...` ran
`--groups 607000063` and CatBoost reported "no columns, skipping" — an empty
result that looks like a finished run. The variable is now `FEATGROUPS`.
Second trap, same hour: a dry-run `sed 's|^python scripts/...|echo|'` failed to
neuter the calls because they are INDENTED two spaces, so the "dry run" trained
for real. Anchor on `[[:space:]]*python`, or add a real `--dry-run`.

## 4. Open threads, highest value first

1. ~~Train the 3D `iso` control.~~ **DONE 2026-09-18** — NOTES 12.9.
   `thr(iso)` scores **−0.1225** on `k_xy`, below the porosity null (−0.1117),
   matching 2D's −.086. The directional walk is load-bearing; a
   reflection-invariant descriptor is *forbidden* from predicting a
   sign-changing target, not merely bad at it. In
   `RESULTS/aniso3d_throat_iso/k_xy`. Only `k_xy` was run — the diagonals are
   sign-invariant, so this says nothing about them.

2. **`all+thr` in 3D — NEVER SUCCESSFULLY RAN.** This item previously claimed
   it produced `RESULTS/aniso3d_allthr/k_xy` and NOTES 12.10. **Both claims are
   false** — verified 2026-09-22, neither exists. The job was killed about a
   minute in, and `train_catboost.py` writes only after every group in a call
   finishes, so nothing survived. (That failure mode is why
   `run_catboost3d_all.sh` calls the script once PER GROUP.) It is not in the
   default grid: at 7606+2700 features it is the most expensive fit in the
   study, and the 2D result says the widest union buys nothing over the
   smaller ones (+.0018 over `baselines+thr`, 2/5 folds, p=.78).

3. ~~**`RESULTS/summary.csv` missed every 3D row.**~~ **FIXED 2026-09-18.**
   `collect_results.py` globbed `RESULTS/*/*/results.csv`, one level too
   shallow for the 3D layout (`RESULTS/aniso3d/catboost/<target>/`). It now
   walks recursively (`find_results`): `kind` is the first component under
   `--results`, `run` is whatever is left, so every 2D path keeps exactly the
   label it had. Summary went 303 → 321 rows / 13 targets, and `thr` was added
   to `GROUP_ORDER`. Note the closing "TDA (1124f) vs fabric (7f)" line is
   still hardcoded 2D text and mislabels the feature count when `--target` is a
   3D component.

4. **The H2 question is still open.** NOTES 10.3 predicted H2 is what makes PH
   worth computing in 3D. It was never testable before because every H2 bar
   landed in one pixel row. The re-imaged cone now resolves H2 (rank 71, top-1
   var .9356) — but nobody has run an H2-only ablation. The *throat* H2 stays
   near rank 1 (top-1 .9991) for a different and legitimate reason: a
   directional min-filter fills cavities in, leaving ~3 H2 bars per structure.

4b. **Porosity-stratified k_off — RUN 2026-09-19, NEGATIVE.** The throat
   hypothesis predicted `thr`'s margin over `baselines` would GROW as porosity
   falls. It shrinks: +.0341 (q4, por .839-.900) -> +.0195 -> -.0027 ->
   -.0045 (q1, por .650-.715). The `tda` control shows the SAME downward
   gradient (+.0214 -> -.0338), as do all three unions (q4->q1 change: tda
   -.055, thr -.039, all -.042, baselines+thr -.029, all+thr -.037), so the
   effect is common to every descriptor family and not throat-specific.
   Independently corroborates NOTES 11.3, whose ECP quartile gradient also
   pointed the wrong way. In `RESULTS/aniso2d_por/q{1..4}/`, log RUN2D_POR.log.
   **This kills the smooth-onset version of the low-porosity case** (throats
   progressively matter more as porosity falls) and is my reason for demoting
   item 5. It does NOT rule out a threshold effect appearing only near phi_c —
   .65-.90 is far from threshold and the generator always percolates. The cheap
   way to test that: `DATA/aniso_lowpor_probe/` already holds 300 generated
   structures at por .45-.55 that were NEVER solved with LBM. Solve those
   before generating anything new.
   Unions stay significant in every band (+.037 to +.074, 5/5 in q2-q4), so
   complementarity is robust across porosity even where the individual families
   are not. Caveat when quoting q4: `baselines` is unusually weak there (.6374
   vs .727-.738 elsewhere), which inflates every q4 margin.

4c. ~~Porosity-stratified k_off — script written.~~ superseded by 4b above:
   `run_2d_porosity.sh`. Fit k_off INSIDE each porosity quartile of the
   existing 4991 labelled 2D structures (cuts .715149 / .778137 / .839478,
   n=1248/1247/1248/1248) and ask whether `thr`'s margin over `baselines` grows
   as porosity falls. Zero LBM cost, ~1 h. **`tda` is the control and must be
   read alongside `thr`**: inside a narrow band porosity explains less of k's
   variance, so every descriptor looks better — if BOTH margins grow it is that
   artefact, and only `thr` growing while `tda` stays flat implicates throats.
   R^2 is NOT comparable across bands (k_off sd runs 37.9 -> 162.1 from q1 to
   q4); compare group differences within a band. It gated item 5, and the
   answer came back negative — see 4b.

5. **Then, the original plan's step 1** — the near-threshold generator
   (drop the resample-until-percolating comfort, hard-core/Matérn placement),
   the ECP screen, and a 60-structure LBM pilot. This is now much better
   motivated: the throat descriptor wins in 3D *at porosity .65–.90 where
   throats are rarely limiting*, so a near-threshold family should widen the
   gap rather than create it.

   **STALE as of 2026-09-19 — read 4b first.** That motivation was tested and
   failed: within .65–.90 the throat margin shrinks toward low porosity, and
   the `tda` control shrinks with it. Demoted. If you still want the regime
   tested, solve the 300 existing `aniso_lowpor_probe` structures rather than
   generating a new family.

   **NOTES 11.3 does not block this, and it is worth being precise about why.**
   It reported two negatives at low porosity, but both measured
   *ECP-from-TPC predictability* — whether the CONE descriptor carries anything
   the two-point correlation does not. `DATA/aniso_lowpor_probe/` (300
   structures, porosity .45–.55) has `structures/` and **no
   `permeability.csv`: it was never solved with LBM**, so no low-porosity
   permeability has ever been predicted by anything. The null is about cone
   redundancy, not about whether THR predicts k better near threshold.
   Two cautions that both sharpen near threshold: `gen_one` resamples until the
   structure percolates in both x and y, so the family is censored exactly
   where acceptance collapses (11.3b); and the k_off label-noise ceiling, which
   is a completely non-binding .99999 on the present data, will start to bite
   once paths get tortuous.

---

## 5. Traps recorded this session — do not rediscover these

* **The binding quota is inodes, not bytes.** `ceph.quota.max_files = 1e6` on
  `/home`; writes failed with `Errno 122` while 23 GiB of bytes were free. The
  descriptor pipeline writes one `.npy` per structure per direction. Keep
  per-structure output on scratch.
* **`python ... | tee log` hides a crash from `set -e`** — pipeline status is
  `tee`'s. Use `set -eo pipefail`.
* **…but `pipefail` then breaks `ls DIR 2>/dev/null | wc -l`**, which exits
  non-zero on a missing directory and kills the script *silently*. Use an
  explicit `[ -d ... ]` guard. Both bugs bit this session, in that order.
* **`--limit 2816` is the labelled prefix** of `DATA/aniso3d` (contiguous,
  sample_id 0..2815, no gaps). `collect_descriptors.py --limit` exists for this.
* **Imaging parameters are per-filtration and per-family.** Never copy them
  between filtrations; measure the bars first and let `ph3d.py`'s `SMEARED`
  flag gate it.

---

## 6. Sanity checks to run first on the new machine

```bash
python scripts/filtration_throat2d.py --self-test   # expect 19/19
python scripts/filtration_throat3d.py --self-test   # expect 23/23
python scripts/filtration3d.py --self-test          # includes the new --channels 1
```

Then confirm the descriptors survived and still reproduce a recorded number —
`ecp` is the strongest control, since it is untouched by PH imaging and must
return exactly `.7326` on `k_xy`:

```bash
python scripts/diagnostics/compare_runs.py \
    --ref RESULTS/aniso3d/catboost/k_xy:baselines \
    RESULTS/aniso3d_throat/k_xy RESULTS/aniso3d_conefixed/k_xy
```

---

## 7. How to start each pass (2026-09-22)

Everything below assumes the descriptors are at 5000 rows (§0) and that the
3D CNN is running on another machine and writes into `RESULTS/aniso3d/cnn3d/`.

### 7.1 3D CatBoost — the joint pass (all six components in one fit)

```bash
bash run_catboost3d_all.sh 2>&1 | tee CB3D_ALL.log
```

Nine groups (`por tpc fab fabdir baselines tda thr baselines+thr all`),
`--targets tensor3d --multioutput --scale-targets`, folds 5, seed 0, 1500
iters, into `RESULTS/aniso3d_all/tensor_joint__<group>/`. ~5 h.

This is the pass for comparing feature GROUPS against each other.

### 7.2 3D CatBoost — the single pass (one fit per component)

```bash
PASSES=single bash run_catboost3d_all.sh 2>&1 | tee CB3D_SINGLE.log
```

54 fits (9 groups x 6 components) into `RESULTS/aniso3d_all/<target>__<group>/`.
~10 h. **This is the pass that is like-for-like with the CNN**, which is
single-target. Both passes in one go: `PASSES="joint single"`.

### 7.3 Why there are two passes

`train_cnn3d.py` and `train_catboost.py` use the SAME split —
`KFold(folds, shuffle=True, random_state=seed)` — both filter `conv_f*`, and
the descriptor merges preserve `permeability.csv`'s sample_id order, all 5000
rows. Verified empirically 2026-09-22: `KFold(5, shuffle=True,
random_state=0)` on the sorted labels reproduced the CNN's per-fold R^2 from
its `oof_predictions.npy` to five decimals. So FOLDS=5 SEED=0 scores the same
structures in the same folds as the CNN.

The one gap no flag can close: the CNN grid is **single-target**, one model per
component. A joint fit is a different estimator — in 2D it cost the
off-diagonal .024-.050 R^2, and `--scale-targets` recovered most but not all.
Use 7.2 to compare against the CNN, 7.1 to compare groups.

### 7.4 Useful knobs

`ITER FOLDS SEED OUT DATASET FEATGROUPS PASSES TARGETS EXPECT`. The driver
skips any group whose `metrics.json` already exists, so it is interruptible and
resumable, and one group can be redone by deleting its directory. It aborts if
any descriptor CSV is not `EXPECT` rows.

Smoke test before committing hours: `FEATGROUPS="fab" ITER=20 OUT=/tmp/smoke
bash run_catboost3d_all.sh` (~1 min).

### 7.5 Extending descriptors again, if the dataset ever grows

```bash
TMPROOT=/scratch/filt3d NPYROOT=/scratch/npy FROM=<current rows> ./run3d_all.sh
```

Set `FROM` to the row count of the existing `descriptors.csv`. Never point
TMPROOT/NPYROOT at a session scratchpad — one already vanished with its
machine, and `/home` has a 1e6 INODE quota that the per-structure `.npy` tree
will hit.

### 7.6 The 2D sample-size study, when you redo it

`scripts/learning_curve.py` against `splits/aniso_golden.csv`. Decide first
whether to point it at the repaired `DESC/aniso_ph_bw125` rather than the
buggy `DESC/aniso`, whether to add `thr`, and whether to finally run
`--pca 64`. All three make it a new experiment rather than a replication.
