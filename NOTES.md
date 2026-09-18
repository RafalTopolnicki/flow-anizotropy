# flow-anizotropy — project notes

Predicting the **2D permeability anisotropy tensor** of porous images from
**direction-aware topological descriptors**.

Status as of 2026-09-15:

| stage | state |
|---|---|
| LBM tensor solver (2D) | built, validated (§5.1) |
| dataset generator (2D) | built, validated (§5.2) |
| **5000-structure dataset** | **generated** — `DATA/aniso/`, 5000/5000 kept |
| **LBM labels (2D)** | **DONE** — 4991/5000 converged both directions (99.82%); §5.2b |
| 2D TDA descriptors | built, validated, **computed for all 5000** (§4.4, §5.5) |
| directional baselines | built, validated, **computed for all 5000** |
| CatBoost + CNN training | **DONE** — `RESULTS/summary.csv`, 160 runs |
| **sample-size study** | **DONE** — `RESULTS/learning_curve/{es,fixed,cnn_densenet121}`; §10 |
| 3D solver `LMB3d/lbm3d-perm` | built, tensor-validated, **rotation-covariant** (§8.6a) |
| 3D pilot | 60 structures generated (`DATA/aniso3d_pilot/`), **not yet solved** |
| **3D structures** | **5000 generated** — `DATA/aniso3d/` (made on another machine) |
| **3D descriptors + baselines** | **DONE** for all 5000 — `DESC/aniso3d/`; §8.5a, `PLAN_descriptors3d.md` |
| **3D LBM labels** | **partial, still arriving** — 2816/5000 at 2026-09-13, all converged; §8.5a |
| 3D CNN | **DONE** — densenet121, six targets, `RESULTS/aniso3d/cnn3d/`; §11.1 |
| **3D CatBoost grid** | **DONE** — 9 groups x 6 targets; §11.1 |
| **descriptor diagnosis** | **ECP is .83 TPC-predictable; PH images were rank 1 — the bandwidth was 8 px**; §11 |
| **PH bandwidth fix (2D)** | **DONE** — `ph` .502 -> .599, but `tda` +.006 and `all` +.000; §11.5c, §11.5d |
| **void phase + tortuosity (2D)** | **DONE** — both null on `all`; three fixes now, headline still .754; §11.8 |
| **3D `all` row** | **DONE** — completed for all six targets; complementarity 6/6, p < .05; §11.9 |
| **new family: channel nets** | **FAILED the screen** — more TPC-predictable than DATA/aniso, not less; §11.10 |
| **junction blocking** | ECP still fails; the PH gain **retracted once** and now confounded with imaging; §11.11 |

### Where to resume (2026-09-16)

Both 3D arms have run and §11 is the result.  The short version: the 3D numbers
reproduced the 2D headline instead of overturning it — TDA alone loses to the
baselines, the CNN dominates everything — and the follow-up diagnostics found
*why*, which is the useful part.  **ECP is ~83% predictable from the two-point
correlation**, and that is mathematics on thresholded Gaussian fields, not a
property of this dataset.

**§11.8 closes the descriptor question.**  Three independent fixes have now been
tried and all three land the same way — a real gain on the descriptor alone, and
nothing on `all`:

| fix | on the descriptor | on `all` |
|---|---|---|
| PH bandwidth (§11.5c) | `ph` +.097, p = .0002 | +.000, p = .92 |
| connectivity / tortuosity (§11.8) | `baselines` +.012, p = .005 | +.006, p = .30 |
| the void phase (§11.8) | nothing, p = .41 | -.003, p = .67 |

`all` has sat at **.754** on k_off through the representation being fixed, a new
non-additive feature family being added, and the descriptor being repointed at
the pore phase instead of the solid — while the 3D CNN reaches .96.  **The
ceiling is not the descriptors, and no further descriptor work is worth doing on
this dataset.**  The screening criterion in §11.7 is therefore the binding
constraint, not one option among several.

**The new-dataset effort has not yet produced a viable family.**  §11.10: a
plain stick network is a Boolean model and scores *worse* on the screen than
DATA/aniso (.779 vs .629, matched n and matched porosity spread).  §11.11:
adding junction blocking leaves ECP still failing (.729 vs .686), and its
apparent PH gain was retracted once as an artefact and then, on the full 2x2,
shrank from .21 to **~.06** once the persistence-image range was separated out
(the range itself moves PH-from-TPC by ~.156).  Both channel candidates are
rejected.  **PH-from-TPC is not a family-invariant
number and must not be compared across datasets without holding the imaging
fixed or reporting the sweep.**

**§11.9 is the one piece of good news and it is now solid.**  The 3D `all` row
had only ever been run for k_xy; the other five targets were run 2026-09-16 and
**complementarity holds on all six**, 5/5 folds each, p < .05.  So TDA does add
something real on top of the baselines — the defensible claim §10 identified —
and the effect size to quote is the off-diagonal mean **+.0264**, not the
+.019 that §11.1 carried, because k_xy turned out to be the weakest of the
three.  A single target is not a sample.

Labels are still arriving on the other machine (2816/5000 at 2026-09-13,
~20/h); re-merge with `--merge-only` and re-run when they land.

Next actions, in order.  **Items 1 and 2 are done; the first live one is item 3.**

1. ~~**Fix the persistence image**~~ — **DONE 2026-09-15, §11.5c and §11.5d.**
   The bandwidth was the cause: `ph` on k_off .502 -> **.599**, 5/5 folds,
   p = .0002.  Nothing downstream moved.  Two negatives worth not repeating:
   the H1 bandwidth correction raised effective rank 4 -> 16 and moved k_off by
   -.002, so **rank is not a proxy for usefulness**; and bandwidth 0.05 is not
   worth running.  Recomputing 3D filtrations (~7.5 h) is **not** recommended —
   it would only matter if 3D leaned on `ph` more than 2D, and §11.1 says not.
2. ~~**Connectivity features, and the void phase**~~ — **DONE 2026-09-16, §11.8.**
   Tortuosity is real (`conn_tort_off` alone is Spearman -.740 on k_off) and
   lifts `baselines` +.012 at 5/5 folds, but +.006 p = .30 on `all`.  The void
   filtration is a clean null: `ecp` +.009 p = .41, and the screen shows both
   phases are equally TPC-predictable (ECP .8571 solid vs .8561 void).  The
   cluster half of the feature family is **dead on DATA/aniso** — the generator
   rejects non-percolating structures, so §11.4's numbers for `beta0_pore` and
   `dead_frac`, measured on the low-porosity probe, do not transfer.
3. **A new structure family** — now the main line of work, not a fallback.
   Two candidates tried and neither passes (§11.10, §11.11).  Untried, and in
   the order I would try them: **arrest rules** (segments grow until they meet
   an existing one, as fracture networks form — placement becomes strongly
   correlated rather than Poisson, which is the property both failures point
   at), then **hard-core exclusion**, then **grain packing with cementation**.
   The screen costs ~15 min end to end per candidate and needs no LBM.
   It must pass the §11.7 screen (**ECP-from-TPC materially below .83**, with a
   sample-size-matched control) *before* any LBM campaign, and it needs a
   60-structure pilot first: a fragmented pore space means lower k and slower
   convergence, and the stopping test is relative to ||q|| (§3.1-3.5).
4. **Re-merge and re-run at 5000** when the LBM finishes.  Every run records its
   label snapshot, so partial-data results stay interpretable.
5. The two joint `roll+oct` CNN runs (§8.5b) — the 48-element octahedral group
   is still unused, and it is aimed squarely at the off-diagonals.
6. Open decisions, not yet taken:
   - **3D `TARGET_GROUPS` aliases** in `train_catboost.py`, so `run_training.sh`
     works unedited.  Raw column names work today, which is why this is not
     blocking.
   - **The 3D golden split.**  Freezing one now draws it from ids 0..2815 only.
   - **Pre-registration.**  Carried over from session 3 and now the most
     urgent item on the list: §11.8 changed the descriptor for the third time
     and the headline did not move, so write down what TDA must beat on a new
     dataset **before** that dataset exists.
   - **Per-fold checkpointing in the trainers.**  Both write `results.csv` only
     after every fold finishes, and `train_cnn3d.py` does not even create its
     output directory until then.

The full working sessions that produced all of this — including the measurements
behind each design decision and the bugs found along the way — are under
`conversations/`, one `.md` (readable) and `.jsonl` (unabridged) per session:

| file | date | covers |
|---|---|---|
| `conversations/session01_2026-09-06.*` | 2026-09-06 | solver, generator, descriptors, CNN, training scripts |
| `conversations/session02_2026-09-07.*` | 2026-09-07 | velocity/VTK tools, the 3D pipeline, 2D labels arriving |
| `conversations/session03_2026-09-08.*` | 2026-09-08/09 | 2D LBM labels complete (4991/5000); full result inspection; sample-size study run and analysed (§10); 3D plan |
| `conversations/session04_2026-09-09.*` | 2026-09-09 | 40^3 rotation-permutation test (§8.6a); ParaView workflow for 2D and 3D (§4.7a, §4.7b); the 2D results table |
| `conversations/session05_2026-09-10.*` | 2026-09-10 | the 3D descriptor pipeline written and run to completion (`PLAN_descriptors3d.md`) |
| `conversations/session06_2026-09-13.*` | 2026-09-13/14 | partial 3D labels merged (§8.5a); `train_cnn3d.py` (§8.5b); the pilot runners (§8.5c); 3D results and the descriptor diagnosis (§11) |
| `conversations/session07_2026-09-15.*` | 2026-09-15 | the PH bandwidth fix: `ph2d.py` gains `--bandwidth` and per-dim ranges; `ph` .502 -> .599 but `tda`/`all` unmoved (§11.5c, §11.5d) |
| `conversations/session08_2026-09-16.*` | 2026-09-16/17 | the void phase and the tortuosity features, both null on `all` (§11.8); the 3D `all` row completed for all six targets (§11.9); the channel-network family and junction blocking, both failing the screen, with one retracted result (§11.10, §11.11) |

```bash
python scripts/export_conversation.py                      # most recent session
python scripts/export_conversation.py --session <id>       # a specific one
ls ~/.claude/projects/-home-rtopolnicki-flow-flow-anizotropy/   # available ids
```

**`export_conversation.py` always writes to `CONVERSATION.md` / `.jsonl` and
overwrites whatever is there.**  It defaults to the *most recent* session, so
running it in a new session silently replaces the previous transcript — which is
how session 1's export was nearly lost (it was untracked, so git could not
restore it).  The session `.jsonl` files under `~/.claude/projects/` are the real
archive; rename each export into `conversations/` straight away.

---

## 1. What this project is

It joins two earlier projects:

| | repo | paper |
|---|---|---|
| **Physics / LBM** | `~/flow/PorousFlow`, `~/flow/PhysInfPorousFluidFLow` | arXiv:2605.20250 |
| **Descriptors** | `~/direction-aware-tda-for-porous-materials` | arXiv:2604.08105 |

From the first we take the 2D D2Q9 LBM solver, the random-trig structure
generators, and the k0/k2/deeppore dataset machinery. From the second we take
the cone+PCA directional filtration → ECP (Julia `Ecp`) + PH (GUDHI persistence
images) → CatBoost pipeline, which was 3D and predicted the elastic stiffness
tensor.

### Why the analogy is exact

The elasticity paper's headline result is that direction-aware TDA is the *only*
method that retains predictive power on the twelve off-diagonal normal–shear
coupling terms — every baseline, non-directional TDA, and a CNN on the raw
voxels all collapse to near-chance there. Those components are small,
sign-carrying, and vanish identically for an orthotropic structure aligned with
the axes.

The flow analogue is one-to-one:

| elastic stiffness | permeability |
|---|---|
| uniaxial C_ii — porosity-dominated, easy | **k_xx, k_yy** |
| off-diagonal coupling — small, sign-carrying, zero when aligned | **k_xy** |

**k_xy is the target of this study.** Everything in the design below exists to
protect it.

### Reference physics

Both in `papers/`:

- **Koza, Matyka & Khalili, PRE 79 066306 (2009)** — 2D finite-size anisotropy.
  Gives the recipe (K from two solves, ĝ = x̂ and ĝ = ŷ), the reciprocity test
  ε = |K_xy − K_yx|/(K_xx + K_yy) < 0.5%, the tensor test (a solve at any other
  force angle must reproduce q = K ĝ), and the scaling σ_α ∝ L^(−d/2) — so in 2D
  σ_α ∝ L⁻¹.
- **arXiv:1305.3426** — the 3D sequel by the same group.

---

## 2. The research plan, as reviewed

The original seven steps were: read everything → plan → modify LBM for the
tensor → generate ~1000 structures (K0/K2-like, no pressure, FF=1) → port the
TDA code to 2D → CatBoost → CNN comparison.

**Verdict: sound, scope about right.** Changes made:

| # | change | reason |
|---|---|---|
| 1 | **Don't reuse the K0/K2 generators** | `gen_k0_structures.py:38` draws continuous \|k\|, so its fields are **not periodic**, and K is only defined on a periodic cell. Measured: agreement between neighbouring columns is 0.979 inside a K0 image but **0.693 across the wrap**, against a chance level of 0.704 for that pore fraction — the two edges are statistically independent. The new dataset gives 0.958 inside and 0.957 across. And an isotropic ensemble puts k_xy in the noise (§3.3). The old surveys also carry only a **scalar** k from single-direction solves, so the 27 h of tensor labelling would have been needed either way; generation itself costs 1 s. |
| 2 | **New generator with designed anisotropy** | Three strata (iso / moderate / strong), mirroring TD → ATTD → RTP in the elasticity paper, which also produces its best figure: the gap grows with anisotropy. |
| 3 | **Rewrite the convergence criterion** | The existing one watches u_x on one column; it is blind to the transverse component and meaningless in a y-forced run. See §3. |
| 4 | **Insert a pilot before the full run** | To check the spread of k_xy exceeds the label noise. Done — §5. |
| 5 | **Add rotation-covariance as a test** | A 90° image rotation is exact and lossless; K must transform as R K Rᵀ. A non-directional descriptor cannot do this even in principle. Free, and it is the 2D form of the paper's central claim. |
| 6 | **Port the baselines too** | The elasticity story only worked because the baselines were strong. Minimum: global porosity, directional porosity profile, directional TPC, fabric tensor, non-directional TDA. |
| 7 | **Right target parametrisation** | Not just (k_xx, k_yy, k_xy). Also log k̄, anisotropy A, and orientation as **(cos 2θ, sin 2θ)** — θ is defined mod 180°, so regressing the angle directly blows up at the wrap. |
| 8 | **CNN → stretch goal** | The elasticity paper already established the CNN result; repeating it in 2D is confirmation, not discovery. |
| 9 | **Sample count** | 1000 is thin for ~1100 features. 5000 preferred, but see the cost in §5. |

Not adopted: physics-informed loss terms from Project 1 — they target velocity-field
regression and buy nothing for tensor regression.

---

## 3. What we measured about the solver

These numbers drove every design decision. All on 256×256, φ = 0.75, ff = 1.

### 3.1 The old convergence criterion is unusable

`volumeflux2d()` sums `U[LX/2][j]` — one column out of 256, and **only u_x**. In
a y-forced run that monitors a quantity unrelated to the driving force. At
eps = 1e-3 it was wrong by up to **39%** on q_y. Structures nearly blocked in one
direction never converge on it and hang indefinitely.

### 3.2 A whole-field L2 residual does not fix it either

The transverse mean is a heavily cancelling sum. Measured rms/|mean| for the
transverse velocity component:

| run | rms(u_t) / \|mean(u_t)\| |
|---|---|
| s0, x-forced | 3.0 |
| s3, x-forced | 14.8 |
| s4, x-forced | **50.1** |
| s4, y-forced | 26.3 |

(The *driven* component: 1.6–1.9 — no cancellation.) In s4 the transverse flow
is a field of ±1.1e-4 whose domain mean is 2.2e-6: 98% cancels, and the
surviving 2% *is* k_yx.

So a field norm, dominated by pointwise magnitudes, bounds the transverse mean
only loosely. Over the last 40 checks of four long runs:

| run | \|q_t\|/\|q_d\| | field residual | rel. change in q_t (med/max) | ratio |
|---|---|---|---|---|
| s0 f=x̂ | 0.237 | 1.1e-3 | 3.4e-4 / 1.0e-3 | 0.3× / 0.9× |
| s3 f=x̂ | 0.140 | 2.0e-3 | 1.8e-3 / 6.2e-3 | 0.9× / 3.1× |
| s4 f=x̂ | 0.021 | 3.1e-4 | 8.3e-4 / 2.9e-3 | 2.7× / 9.4× |
| s4 f=ŷ | 0.017 | 2.8e-4 | 1.1e-3 / 4.0e-3 | 3.9× / **14×** |

The ratio runs 0.3× to 14× and tracks how small the off-diagonal is: the field
residual over-reports when the answer is well determined and under-reports
exactly when it isn't. No single ε means the same thing twice.

**Neither residual actually converges.** At 265k steps the field residual is
still 1.1e-3 (s0) and 2.0e-3 (s3), flat. `U`/`V` are `float` and BGK carries a
persistent fluctuation, so this code floors around 1e-3–1e-4. Koza's ε = 1e-6 was
for a double-precision MRT/Sailfish code and will never trigger here. Past the
transient you are not converging, you are sampling a stationary fluctuation.

### 3.3 The label-noise law — the constraint on the dataset

K = Kᵀ is exact for Stokes flow, so `recip_resid = |k_xy − k_yx|/(k_xx + k_yy)`
is a **free per-sample error bar**. Measured 0.003%–0.17%, i.e. it scales with
the **trace**, not with k_xy. The relative error on k_xy therefore goes as
`recip_resid × trace / |k_off|`. Since

    |k_off| / trace = ½ · (k₁−k₂)/(k₁+k₂) · sin 2ψ

a nearly-isotropic ensemble puts k_xy in the noise. **This is why the generator
must produce real anisotropy ratios with a uniformly random principal-axis
rotation.** Confirmed quantitatively on the pilot — §5.2.

### 3.4 Two traps in the stopping test

- **A single passing check is worthless.** In s4, x-forced, every criterion read
  ~7e-6 at step 30000 and 3.5e-3 at 35000 — a 500× jump. Hence `--hold 3`.
- **The test must compare block means, not instantaneous samples.** This one bit
  us: the first implementation compared instantaneous flux, and on a *converged*
  structure the transverse flux still wobbles ~0.5% between checks, so three
  consecutive passes could never accumulate. ~70% of the first pilot ran to the
  500k step cap on solutions that had settled by step 5000. Averaging within
  each window turns the test back into a drift test; the stuck structure then
  converged in 35k steps with the same q_x to four digits.

### 3.5 Block averaging is where the accuracy comes from

Instantaneous q at the stopping step is a noisy estimate. Averaging trailing
samples gives ~40× reduction:

| run | instantaneous spread of q_t | block-mean std. error |
|---|---|---|
| s0 | 0.23% | 0.008% |
| s3 | 1.78% | 0.043% |
| s4 f=x̂ | 0.39% | 0.014% |
| s4 f=ŷ | 0.64% | 0.019% |

So the *stochastic* error is ~0.05%, and **discretisation truncation, not
convergence, is the binding limit**. Do not over-engineer convergence; get to
~0.1% and spend the effort on making |k_xy| large instead.

---

## 4. Tools

### 4.1 `LMB2d/lbm2d-perm` — permeability tensor solver

Build (headless, **no OpenGL dependency**):

```bash
cd LMB2d && bash make_perm.sh          # -> lbm2d-perm
bash make_wmi.sh                       # -> lbm2d-perm-gl, the full GL build
```

Original sources are preserved in `LMB2d/orig/`.
**`LMB2d/lbm2d-256x256_force` is a stale binary from the old single-direction
code — do not use it.**

```
lbm2d-perm <structure.gif|ppm|dat> <results.csv> [options]

  --ff F            force factor; body force = 2.5e-07 * F   (default 1)
  --eps E           per-component relative tolerance         (default 1e-3)
  --lag N           steps between convergence checks         (default 5000)
  --hold K          consecutive passing checks required      (default 3)
  --min-steps N     no convergence before this step          (default 10000)
  --max-steps N     give up after this many steps            (default 500000)
  --avg-steps N     length of the averaging phase            (default 20000)
  --avg-sample N    flux sampling interval                   (default 50)
  --qfloor R        floor for the relative test, as a fraction of |q| (default 1e-2)
  --ksigma K        also accept a change within K sigma of the block
                    mean's own standard error                (default 3)
  --velocity PREFIX write PREFIX.fx.dat and PREFIX.fy.dat
  --validate DEG    extra solve at DEG degrees; reports the flux predicted
                    by K against the measured one (Koza09 tensor test)
  --quiet           suppress per-check progress on stdout
```

**How it works.** One invocation runs two solves of the same structure,
F = (F, 0) then F = (0, F), each giving one column of K:

    run x:  k_xx = mu q_x / F ,  k_yx = mu q_y / F
    run y:  k_xy = mu q_x / F ,  k_yy = mu q_y / F

Both in one process so the structure is read once and the reciprocity residual
is available per sample.

**Convergence** (§3.4): per-component relative test on the *block mean* of the
domain-averaged flux, each component against its own magnitude, with tolerance
`max(eps·|q|, ksigma·SE)`. The block mean's own standard error makes it
self-calibrating, so eps never needs per-structure tuning. Then a dedicated
20k-step averaging phase produces the recorded q and its standard error.

**Exit codes**: 0 = both directions converged, 3 = at least one did not (the row
is still written, flagged by `conv_fx`/`conv_fy`), 2 = hard error.

**Output columns**:
`filename, ff, force, mu, porosity, steps_fx, conv_fx, steps_fy, conv_fy,
qx_fx, qy_fx, qx_fy, qy_fy, se_*, k_xx, k_yx, k_xy, k_yy, k_off, recip_resid,
k1, k2, k_mean, anisotropy, theta_deg, alpha_x_deg, alpha_y_deg, fieldres_*`

### 4.2 `scripts/gen_aniso_structures.py` — structure generator

```bash
python scripts/gen_aniso_structures.py --self-test
python scripts/gen_aniso_structures.py --output DATA/aniso --n_samples 5000
```

Random trig field thresholded to a target porosity:

    S(r) = sqrt(2/N) Σ cos(q_i · r + φ_i),   solid where S > t

Wavevectors are drawn isotropically in a band around k₀ and squashed by an
area-preserving elliptical map before rounding to **integers** (periodicity is
mandatory):

    q = R(ψ) diag(1/A, A) R(ψ)ᵀ q_iso

A is the axis ratio, ψ the elongation direction. Strata are assigned
**cyclically by sample index**, so any prefix of the dataset is balanced and an
unshuffled KFold sees every stratum in every fold:

| stratum | A |
|---|---|
| `iso` | 1.0 — anisotropy is then purely the Koza09 finite-size effect |
| `mod` | U(1.5, 2.5) |
| `strong` | U(2.5, 4.0) |

Defaults: size 256, k₀ ~ U(4, 10), dk 1, N ~ U(20, 80) modes, porosity ~ U(0.65, 0.90).

**Frame convention** (validated end to end, §5.1): `grid[i,j] = (x, y) =
(j/size, i/size)`, i.e. array axis 1 is the solver's x and axis 0 is its y,
matching how `read_from_gif` maps image (row, col) onto `F[x][y]`. ψ is therefore
directly comparable with the solver's `theta_deg`.

**Percolation**: the pore phase must span a 3×3 tiling in **both** directions,
with **4-connectivity** — a diagonal pinch has zero cross-section, conducts
nothing under D2Q9 bounce-back, and would burn the solver's whole step budget.

**Outputs**: `structures/*.gif`, `structures.csv`, `filelist.txt`, `gen.log`.

**One structure format, deliberately.** The GIF is what the C++ solver reads, so
it is the only copy stored — the same bytes feed the LBM and the TDA descriptors,
with no second file to drift out of sync. On the Python side always load it
through `scripts/structure_io.load_structure`, which reproduces
`LMB2d/lbm.cpp: read_from_gif` rule for rule (`red > 0` is pore, and image
(row, col) -> `F[x][y]`, so the array is indexed `[solver_y, solver_x]` — do not
transpose it). `python scripts/structure_io.py <dataset_dir>` checks the reader
against the porosity the solver itself reports.

> **Do not regex-parse the filenames.** `\w` matches `_`, so `str=(\w+)` on
> `sample_000003_str=iso_A=1.00_...` silently returns `iso_A`. The same class of
> bug produced all-NaN columns in the k0/deeppore datasets. Join on `sample_id`;
> `structures.csv` / `permeability.csv` are the source of truth.

The generator knobs (A, ψ, k₀, N) are recorded for stratification and confound
analysis. **They are not model features** — the point is to predict K from the image.

### 4.3 `scripts/solve_permeability.py` — parallel driver

```bash
python scripts/solve_permeability.py --dataset DATA/aniso --workers 14
python scripts/solve_permeability.py --dataset DATA/aniso --merge-only
```

Writes one CSV per sample under `<dataset>/perm/`, then merges onto
`structures.csv` into `permeability.csv`. Per-sample files are deliberate:
concurrent appends race on the header, and it makes a multi-hour run
**restartable** — rerunning solves only what is missing. Solver exit 3 is kept
and flagged, not treated as a failure.

Derived targets added at merge: `cos2theta`, `sin2theta` (θ is mod 180°, so never
regress the angle itself), `log_k_mean`, `k_ratio`, `log_k_ratio`, plus
`koff_rel_err` and `koff_over_trace` — the per-sample label-noise proxy and the
quantity it depends on.

---

## 4.4 Descriptor pipeline

`run_descriptors.sh` drives it; per direction it computes the filtration, derives
ECP and PH, then deletes the filtration (1 ms/structure, so cheaper to recompute
than to keep — all four directions would be ~10 GB).

```bash
bash run_descriptors.sh                       # DATA/aniso -> DESC/aniso
python scripts/baselines2d.py --dataset DATA/aniso --output DESC/aniso/baselines.csv
```

| script | role | features |
|---|---|---|
| `filtration2d.py` | wedge + PCA filtration, numba | `(H,W,2)` |
| `ecp2d.jl` | 2-parameter ECP, Julia `Ecp` | 81 per direction (grid-res 8) |
| `ph2d.py` | PH on the wedge channel, gudhi | 200 per direction (H0,H1 × 10×10) |
| `collect_descriptors.py` | assemble → `descriptors.csv` | **1124** (4 directions) |
| `baselines2d.py` | porosity profile, TPC, fabric tensor | **779** |
| `train_catboost.py` | CV comparison of feature groups | — |

**Three deliberate departures from the 3D code**, all in `filtration2d.py`:

1. **No grid rotation.** The 3D `rotate_grid_to_z` transposes for axis-aligned
   directions and falls back to a lossy nearest-neighbour `affine_transform` for
   diagonals. In 2D the wedge offsets are built directly at angle θ, so every
   direction is exact and diagonals stop being second-class.
2. **One PCA channel, not two.** In 2D the perpendicular component is redundant
   (|v·d|² + |v·d⊥|² = 1); a second channel would only inflate the ECP grid.
3. **Periodic in both axes** for PH, matching the toroidal cell the LBM solves.
   The 3D code wrapped z only.

Two API traps worth recording: `Ecp.vectorize_ecp` needs a **tuple** of steps
per filtration parameter (a scalar raises `BoundsError`), and there is no H2 in
2D so PH yields 2 × res², not 3 ×.

---

## 4.5 CNN baseline

`scripts/train_cnn.py` — end-to-end from the binary image, same targets, same
5-fold split (KFold, shuffle, seed 0) and same results schema as
`train_catboost.py`, so the rows concatenate into one table.

```bash
python scripts/train_cnn.py --dataset DATA/aniso --backbone convnext_tiny \
    --targets tensor invariant --output RESULTS/cnn_convnext
```

Backbones come from `timm` (`--backbone`), so anything it exposes works; the
three families from the parent projects are verified: **convnext_tiny/small**,
**resnet50/101**, **densenet121/169**. Input is 1-channel (`in_chans=1`, timm
adapts the stem). `--pretrained` needs network access and falls back to random
init with a warning rather than failing.

### The augmentation trap — the part that matters

The cells are periodic, so a **random circular shift is an exact symmetry**: the
structure is genuinely unchanged as a torus and every component of K is
untouched. That is the default (`--augment roll`) — free regularisation, no
label bookkeeping.

The D4 quarter-turns and reflections are also exact *for the image*, but they
are **not label-preserving**:

    quarter-turn:  k_xx <-> k_yy,  k_off -> -k_off,  cos2t -> -cos2t,  sin2t -> -sin2t
    reflection:    k_xx, k_yy fixed,  k_off -> -k_off,  cos2t fixed,   sin2t -> -sin2t

Rotating images while holding labels fixed — the default behaviour of every
standard augmentation pipeline — would teach the network that **k_off is
orientation-independent**, destroying precisely the signal this study is about.
`--augment roll+d4` transforms the labels with the image, and **refuses to run**
on a target whose rule is not known rather than guessing.

This is also why `theta` is never a target: it is defined mod 180 degrees, so it
is learned as (cos 2t, sin 2t).

---

## 4.6 Running the models

`run_training.sh` trains every model on every target set and merges the results
into one table.

```bash
bash run_training.sh                              # everything
TARGETS=k12 bash run_training.sh                  # just the key component
SKIP_CNN=1 bash run_training.sh                   # CatBoost only
LIMIT=500 EPOCHS=5 FOLDS=2 bash run_training.sh   # smoke run
```

Every knob is an environment variable: `DATASET LABELS DESC RESULTS TARGETS
CB_GROUPS BACKBONES FOLDS SEED LIMIT STRATA EPOCHS BATCH AUGMENT SKIP_CNN
SKIP_CB`. It refuses to start if `permeability.csv` is absent.

### The four target sets

| alias | column(s) | |
|---|---|---|
| `k11` | `k_xx` | |
| `k22` | `k_yy` | |
| `k12` | `k_off` | **the headline** — the 2D analogue of the elastic normal–shear coupling terms |
| `full_tensor` | all three | a **joint** model, not three fits |

`k12` is the *symmetrised* off-diagonal, not `k_xy` or `k_yx`: K = Kᵀ holds
exactly for Stokes flow, so averaging the two measurements halves the noise on
the component that needs it most (§3.3).

`full_tensor` being a genuine joint model — CatBoost `MultiRMSE`, the CNN
multi-output natively — is what makes joint-vs-single a real experiment rather
than a relabelling of the same three numbers. The summary keeps `mode`
(`single`/`joint`) in the index so the two are never aggregated together.

Output: `RESULTS/catboost/<target>/`, `RESULTS/cnn/<backbone>_<target>/`, a
merged `RESULTS/summary.csv`, and a printed pivot plus a "best on k_off"
ranking.

**Timing, corrected against the real labels.** The earlier "CatBoost is minutes"
was extrapolated from stand-in labels and is wrong. Measured: the `tda` group
(1124 features) takes **144.6 s at 600 rows x 2 folds**, and CatBoost scales
about linearly in row-folds and in feature count. So at 5000 rows x 5 folds that
is ~50 min for `tda` alone on one target, ~2.8 h for all eight groups on one
target, and **~11 h for the four targets** — plus ~11 min per backbone per
target for the CNN on the GPU. Run `TARGETS=k12` first; it is the headline and
it is a quarter of the grid. `--iterations` (default 1500) is the cheapest dial
if that is too long, but it is only exposed on `train_catboost.py` directly, not
as an env var in `run_training.sh`.

---

### 4.7 Velocity fields and ParaView

Two small tools, for looking at a single structure rather than for the dataset
run.  Output is split by dimension — **`VIZ/2d/` and `VIZ/3d/`** — because the two
carry the same tags (`fx`, `fy`) for different quantities and the same prefix
style, and a flat directory made it impossible to tell at a glance which solver
wrote what.  Both scripts `mkdir -p` their output's parent, so `-o VIZ/3d/v0`
needs no setup.  `VIZ/` is gitignored.

```bash
python scripts/solve_velocity.py <structure.gif> -o VIZ/2d/s0 --vtk
python scripts/to_vtk.py VIZ/2d/s0.fx.npy out.vtk --solid VIZ/2d/s0.solid.npy
python scripts/to_vtk.py --self-test        # 26 checks
```

`solve_velocity.py` runs the same two solves the dataset run does and keeps the
fields under `VIZ/2d/`: `s0.fx.npy` / `s0.fy.npy` as `(ny, nx, 2)` float32, `s0.solid.npy`, the
usual tensor row as `s0.csv`, and with `--vtk` a single ParaView file `s0.vtk`
carrying **both** directions — `velocity_fx`, `velocity_fy`, their magnitudes and
`solid` — in one dataset (see 4.7b).

**Orientation is the whole problem here** and both tools state their convention
in the docstring. `exportvelocity` in `lbm.cpp` writes `U[i][j] V[i][j]` with j
outer, so a plain reshape to `(LY, LX, 2)` is already indexed `[solver_y,
solver_x]` — the same as `structure_io.load_structure`, no transpose. VTK
STRUCTURED_POINTS wants x varying fastest, which is C order for an array indexed
`(z, y, x)`, so `reshape(-1, 3)` is already the right sequence in 2D **and** 3D.

Two checks run on every invocation, because a transposed field still looks like
a plausible flow:

* **`macro()` only ever writes U,V on pore nodes** and the arrays start zeroed,
  so the field must be *exactly* zero on every solid node. A transpose breaks
  this at once. Measured on sample_000002: 0 violations, max \|u\| on solid
  exactly 0.
* **Flux recomputed from the field** must reproduce the `qx_fx`… columns.
  Measured 0.01–0.04%. It is not machine precision on purpose: the CSV holds the
  mean over the 20k-step averaging phase, the exported field is the
  instantaneous last step, and §3.5 measured that spread at 0.2–1.8%.

Verified by reading the file back through **VTK's own reader** (`pyvista`, the
same code path ParaView uses), not just by re-parsing it here: arrays match the
`.npy` exactly, and `find_closest_point((x, y, 0))` returns `npy[y][x]`.

Binary legacy VTK (big-endian) is the default; `--ascii` is readable but ~15×
larger — irrelevant at 256² (1.1 MB), decisive at 256³ (~1 GB), which is why it
was written this way now rather than fixed during the 3D port.

`to_vtk.py` already accepts `(nz, ny, nx, 3)` and is self-tested on it, so it
carries over to the 3D work unchanged — see 4.7a, which is the 3D half.

Note the exported velocity is the raw moment `Σf·e/ρ` without the `+F/(2ρ)`
forcing correction, matching the flux convention used everywhere else (§6).
**The 3D export is not the same quantity** — see 4.7a.

### 4.7a `solve_velocity3d.py` — the 3D half, and the ParaView recipe

```bash
python scripts/solve_velocity3d.py <structure.raw> -o VIZ/3d/v0 --vtk
python scripts/solve_velocity3d.py <structure.raw> -o VIZ/3d/v0 --vtk --validate
```

Same tool one dimension up: three solves (+x, +y, +z), the three columns of K,
and the converged field of each kept as `VIZ/3d/v0.{fx,fy,fz}.npy` `(nz, ny, nx, 3)`
float32 plus `v0.solid.npy`, `v0.csv`, and a single `v0.vtk` carrying all three
directions (4.7b).  `--validate` keeps the (1,1,1) field too, which is the one
worth looking at when the question is whether the tensor really predicts an
off-axis flow.

**Nothing needs to come off the production machine to make a picture.**  The
fields are not in the dataset — the production run writes only CSV rows — so a
visualisation is a fresh local solve of a structure you already have.  One 80^3
structure is ~13 min single-core against the ~7 days the 5000-sample run takes.

The `.vel` format `lbm3d-perm --velocity` writes: a 16-byte header of four int32
`(NX, NY, NZ, 3)`, then NN triples of float32, **x fastest** — the same voxel
order as the `.raw`.  So a plain reshape to `(nz, ny, nx, 3)` is already
`[solver_z][solver_y][solver_x]` and nothing transposes anywhere in the chain.
The header is checked against the structure's shape on every read, which turns a
format or endianness drift into an error on the first file instead of a picture
that looks merely odd.

**The 3D velocity is not the 2D velocity.**  3D `macro_collide()` folds the Guo
correction into U,V,W, so the export is the physical `(Σf·e + F/2)/ρ`; the 2D
export is the raw moment (§6, 8.3 departure 1).  Do not compare their magnitudes
across dimensions.

The two orientation checks run on every invocation, as in 2D, but the second one
is **much sharper here**: measured on a 40^3 strong structure, the flux
recomputed from the exported field reproduces the solver's `qx_fx`... columns to
1.1e-08 - 2.3e-07, i.e. float32 rounding.  In 2D the same check only agrees to
~1%, because the CSV holds a block mean over a fluctuating field.  In double
precision the 3D solver reaches a genuine fixed point with block means
bit-identical from step 4000 (8.4), so there is no fluctuation for the average to
remove and no slack for the check to hide in — a sub-percent disagreement in 3D
would be a real finding, not the expected noise it is in 2D.

Verified by reading back through **VTK's own reader** (`pyvista`, the code path
ParaView uses), not just by re-parsing here: `velocity` equals the `.npy`
exactly, `velocity_magnitude` and `solid` match, and `find_closest_point((x,y,z))`
returns `npy[z][y][x]` — checked at three interior and boundary points.  The same
round-trip passes on the 2D file.

#### ParaView

Open the `.vtk` (File > Open, Apply).  It is a STRUCTURED_POINTS / uniform grid,
so ParaView needs no reader configuration.  In 2D it arrives as a 256x256x1 slab;
in 3D as an nx x ny x nz volume.

The one filter that matters is **Threshold on `solid`**, lower and upper both 0.
Everything else is applied to its output.  Without it every filter also processes
the solid nodes, where the velocity is exactly zero — glyphs collapse to a
forest of zero-length arrows, streamlines terminate the moment they enter rock,
and a volume render is dominated by the zeros.

From that threshold:

* **speed** — colour by `velocity_magnitude`, and switch the colour map to log
  scale.  Pore velocity spans three to four decades, so a linear map shows the
  few fastest channels and nothing else.  That single change is most of the
  difference between a useful picture and a black one.
* **streamlines** — Stream Tracer, seed type *Point Cloud* over the whole
  bounding box.  On a triply periodic cell they will run out of the side of the
  domain; that is the geometry being genuinely periodic, not a bug.  Tube the
  result and colour by `velocity_magnitude`.
* **glyphs** — Glyph > Arrow, Orientation Array `velocity`, Scale Array
  `velocity`, and *Every Nth Point* rather than *All Points* (at 80^3 all points
  is ~500k arrows).
* **the geometry itself** — a second Threshold on `solid`, both bounds 1,
  displayed opaque, or Contour on `solid` at 0.5 for the pore/solid interface,
  which is much lighter to render than the voxels.
* **the anisotropy, which is the actual point** — switch the colouring between
  `velocity_fx`, `velocity_fy` and `velocity_fz` on the *same* pipeline.  The
  transverse response is what `k_xy` measures: colour `velocity_fx` by its *y*
  component rather than by magnitude, on a symmetric diverging range, and a
  structure with a real off-diagonal shows a signed, spatially organised
  transverse flow instead of noise around zero.  `velocity_fy` coloured by its
  x component is the reciprocal picture, and K = K^T says the two means must
  agree.

At 80^3 the combined binary file is ~25 MB.  `--ascii` is ~15x larger (~380 MB)
and is only worth it to read the numbers by eye.

### 4.7b One file, all directions

Both tools write **one** `.vtk` holding every forcing direction as a separately
named array — `velocity_fx`, `velocity_fy`, `velocity_fz`, each with a
`_magnitude` companion, plus `solid` once — rather than one file per direction.
They share a grid, so in ParaView one Threshold and one Glyph/StreamTracer
pipeline serve all of them and the direction is a dropdown change; the split form
meant three datasets whose cameras, colour ranges and filter settings had to be
kept in step by hand, which is precisely the comparison the off-diagonal targets
are about.  It is also slightly *smaller*: 25.1 MB against 3 x 8.7 = 26.1 MB at
80^3, since the geometry and `solid` are stored once.

`--split` restores the old per-direction files, which is still the cheaper way to
open a single direction of a very large volume.

`to_vtk.write_vtk_multi` is the underlying call, and `--field NAME=FILE`
(repeatable) reaches it from the command line — so a combined file can be rebuilt
from saved `.npy` fields without re-solving:

```bash
python scripts/to_vtk.py -o VIZ/3d/v0.vtk --solid VIZ/3d/v0.solid.npy \
       --field velocity_fx=VIZ/3d/v0.fx.npy \
       --field velocity_fy=VIZ/3d/v0.fy.npy \
       --field velocity_fz=VIZ/3d/v0.fz.npy
```

`write_vtk` is now literally the one-field case of `write_vtk_multi` — the
self-test asserts the two produce **byte-identical** files on a single field, so
the refactor cannot have moved the ordering the whole tool exists to get right.
The multi path adds twelve checks of its own (every array present and holding its
own data, `solid` and `POINT_DATA` written exactly once, mismatched grids and
whitespace in an array name rejected): 26 checks total, up from 14.

---

## 5. Tests performed

### 5.1 Solver

| test | result |
|---|---|
| **Tensor test** (`--validate 45`, Koza09 §III.A) — predict a 45° solve from the two axis-aligned ones | **0.05–0.35%** relative error on s0/s1/s3 |
| **Reciprocity** K = Kᵀ across 6 structures | residual 3e-5 … 1.7e-3 of the trace, matching Koza's <0.5% |
| θ against the analytic eigenvector of the measured K (s0) | −23.45° both ways |
| Determinism | bitwise identical across repeat runs |
| Non-percolating structure (s5) | correctly flagged `conv_fx=0`, exit 3, no hang |
| Velocity export | 65536 rows per direction, correct |
| Retune `--qfloor` 1e-3 → 1e-2 | s3's y-run 520k (capped) → 100k steps, k_yy unchanged to 0.2% |

The six probe structures (256², φ=0.75, ff=1; s0–s2 isotropic, s3–s5 anisotropic):

| id | k_xx | k_yy | k_yx | k_xy | recip. resid | rel. err on k_off |
|---|---|---|---|---|---|---|
| s0 | 381.9 | 212.6 | −90.44 | −90.53 | 1.6e-4 | 0.05% |
| s1 | 181.7 | 323.5 | −105.99 | −105.76 | 4.6e-4 | 0.11% |
| s2 | 310.0 | 416.2 | 55.82 | 55.34 | 6.6e-4 | 0.43% |
| s3 | 127.2 | 3218.7 | 17.75 | 17.42 | 1.0e-4 | 0.95% |
| s4 | 1151.5 | 1653.8 | 23.83 | 28.49 | 1.7e-3 | 8.9% |
| s5 | ≈0 (blocked in x) | 4038.8 | — | — | — | flagged |

s4 is the instructive one: 8.9% relative error purely because its k_off is 0.85%
of the trace. That is the whole argument for the generator design.

### 5.2 Generator + 60-sample pilot

`--self-test` (no solver needed) checks: a pure q=(k,0) mode varies along the
solver's x; ψ=0 shrinks the x wavenumbers and ψ=90° the y ones; the field is
continuous across the periodic wrap; and a vertical solid wall gives
`perc_x=False, perc_y=True`. All pass.

`scripts/structure_io.py <dataset_dir>` cross-checks the Python GIF reader
against the porosity the **solver itself** reports (`getporosity_full()`, written
into every row): max difference **0.000e+00** over 25 structures. That is the
guarantee that the C++ and Python sides see the same structure.

Pilot: 60 structures (20 per stratum), generated in **1 s**, solved in 21 min on
14 cores. **60/60 converged in both directions, no failures.** Median 50k steps
per direction, max 205k — well under the 500k cap.

**Frame validated end to end** — the generator's ψ against the solver's θ:

| stratum | median \|θ − ψ\| | within 20° |
|---|---|---|
| strong | **2.4°** | 100% |
| mod | 6.8° | 90% |
| iso | 59.4° | 10% |

(iso is correct behaviour: A = 1 means ψ is unused, so the difference is uniform.)

**The label-noise law confirmed, and monotone:**

| \|k_off\|/trace | n | median rel. err on k_off | max |
|---|---|---|---|
| ≤0.02 | 13 | 1.2% | 5.8% |
| 0.02–0.05 | 7 | 0.34% | 2.1% |
| 0.05–0.10 | 11 | 0.15% | 0.63% |
| 0.10–0.20 | 10 | 0.07% | 0.48% |
| >0.20 | 19 | 0.07% | 0.25% |

Median `|k_off|/trace` by stratum: 0.032 (iso), 0.137 (mod), 0.330 (strong) — so
both anisotropic strata sit at ~0.1% label noise. The iso stratum is noisier but
usable.

**Coverage**: k₁/k₂ from 1.02 to 311, anisotropy 0.01–0.99, k_mean 16–1846.

### 5.2b The full 5000 — labels landed 2026-09-07

**4991/5000 converged in both directions (99.8%)**, no NaNs in k_off. The pilot's
predictions held:

| stratum | median \|k_off\|/trace | pilot predicted | median koff_rel_err |
|---|---|---|---|
| iso | 0.035 | 0.032 | 0.26% |
| mod | 0.121 | 0.137 | 0.10% |
| strong | 0.238 | 0.330 | 0.089% |

and the label-noise law reproduced on the real ensemble:

| \|k_off\|/trace | n | median koff_rel_err | 90th pct |
|---|---|---|---|
| ≤0.02 | 705 | 1.03% | 7.3% |
| 0.02–0.05 | 893 | 0.31% | 1.1% |
| 0.05–0.10 | 892 | 0.14% | 0.50% |
| 0.10–0.20 | 1112 | 0.094% | 0.32% |
| >0.20 | 1389 | 0.048% | 0.19% |

The printed k_off label-noise ceiling is **R² ≤ 1.0000**, i.e. the labels are not
the binding constraint anywhere — the generator design in §2 change 2 did its job.

**First signal on the real target** (smoke run only — 600 rows, 2 folds, so not a
result): on `k_off`, TDA 1124 features **R² 0.511**, fabric 7 features 0.055,
scalar porosity −0.414. That is the *reverse* of §5.6, where fabric beat TDA on
orientation. §5.6 was the warning; this is the first evidence the warning does
not carry over to the actual target. Needs the full 5000 x 5 folds before it
means anything.

### 5.3 Cost

**4.6 core-min per sample.** On 14 cores: ~5.5 h for 1000 samples, ~27 h for
5000. (An earlier estimate of 6 h for 5000 came from three hand-made structures
and understated the tail.) Cheapest levers if that is too long: reduce
`--avg-steps` (20k × 2 runs ≈ 15% of the cost), or raise `--porosity_min` to shed
the slow low-permeability tail.

### 5.4 Reproducibility

Measured, not assumed.

| what | result |
|---|---|
| generator, same `--seed`, **14 vs 3 workers** | byte-identical GIFs and `structures.csv` |
| generator, `--seed 0` vs `--seed 1` | different, as intended |
| solver, same structure run 3× | all columns bitwise identical |

**Generator.** Each structure draws from `np.random.default_rng(seed + sample_id)`,
so a sample's content depends only on its index — never on worker count,
scheduling, or completion order. `structures.csv` is sorted by `sample_id` before
writing. Rejected attempts advance that sample's own stream only, so the retry
loop stays deterministic too.

**Solver.** No RNG at all: the `srand`/`init_genrand` calls were dropped with the
old `main.cpp`, and the remaining `rand()` uses in `lbm.cpp` sit in
`generateRAN`/`generate11`/`generate22`, which build structures inside C++ and are
never called. Output is a pure function of the structure and the CLI flags,
single-threaded per process.

**The one caveat.** NumPy does not formally guarantee that `default_rng` streams
are stable across NumPy feature releases (NEP 19), so *regenerating* on a machine
with a different NumPy may not reproduce byte-identical structures. In practice
PCG64 with `uniform`/`integers` has been stable, and this dataset was built with
NumPy 1.26.4. For exact reproduction **ship the GIFs** — they are only 13 MB for
5000 samples — rather than shipping the seed and regenerating. The seed is for
re-deriving the dataset in the same environment, not a substitute for archiving it.

### 5.5 Descriptors

`filtration2d.py --self-test` (10 checks) and `baselines2d.py --self-test`
(7 checks) both pass. The two that matter most:

* **Exact rotation equivariance** — `rot90(structure)` at θ+90° reproduces
  `rot90(filtration)` to **0.00e+00**. This first failed at 8.3e-2 and exposed a
  real bug: `cos(90°)` is 6.1e-17, not 0, which flipped offsets sitting exactly
  on the wedge boundary, so θ=90° got 48 offsets where θ=0° got 42 and the
  stencil lost central symmetry. Fixed with a 1e-9 tolerance; different
  directions are now measured with identically sized wedges.
* **Fabric tensor equivariance** — `rot90` swaps `fxx`/`fyy` exactly.

Cost on 5000 structures × 4 directions: filtration ~1 ms each, PH ~0.04 s,
whole run ~6 min. Baselines: 17 s. (The 3D pipeline needed 48 s **per
structure**.)

### 5.6 Smoke test — the descriptors carry directional signal

Before the LBM labels existed, predicting the generator's *hidden* ψ and A from
the image alone (mod+strong strata, n=3333, RF 5-fold):

| features | n | cos 2ψ | sin 2ψ | A |
|---|---|---|---|---|
| TDA ECP+PH | 1124 | 0.983 | 0.985 | 0.862 |
| **fabric tensor** | **7** | **0.994** | **0.994** | 0.857 |
| TPC | 260 | 0.981 | 0.983 | 0.559 |
| porosity profile | 512 | 0.593 | 0.538 | 0.113 |
| porosity (scalar) | 1 | −0.14 | −0.16 | −0.14 |

The pipeline works — TDA recovers a hidden generator parameter at R²≈0.98 from
the image. **But the 7-feature fabric tensor beats all 1124 TDA features on
orientation.** That is the risk flagged in §2 change 6, now measured.

Two things keep it in perspective: ψ is a proxy, not the target; and it is the
task the fabric tensor is practically purpose-built for. Whether TDA adds
anything for `k_off` is a different question and needs the real labels. But
plan for fabric to be the baseline to beat, and report it prominently rather
than burying it — the honest framing is the one the elasticity paper used.

### 5.7 CNN

`train_cnn.py --self-test` (6 checks) passes. The important ones verify the D4
label algebra **against the tensor transformation itself** rather than against
my own restatement of it: for a random K, `d4_transform_labels` is compared with
computing R K Rᵀ and re-deriving (cos 2t, sin 2t) from the result.

| check | result |
|---|---|
| quarter-turn matches R K Rᵀ | exact |
| reflection matches F K Fᵀ | exact |
| half-turn is the identity on K | exact |
| four quarter-turns return to the start | exact |
| unknown target under `roll+d4` is rejected | raises |

All three backbone families instantiate, train and score end to end
(convnext_tiny, resnet50, densenet121), on both `roll` and `roll+d4`.

GPU execution verified on hardware, not inferred: model parameters on `cuda:0`,
AMP activations in `float16`, peak **2.60 GB of 24 GB** at batch 32 — so there is
room for a much larger batch. Benchmarked **29.5 steps/s on the RTX 4090 against
0.44 on CPU, i.e. 67x**. A full 5-fold run over 5000 structures at 30 epochs is
roughly **11 min per backbone**; the same on CPU would be ~12 h. `--device cpu`
still works (every AMP path is guarded on `dev.type == "cuda"`), just slowly.

One thing to watch: images are augmented with numpy in the training loop, so for
the lighter backbones the `np.roll` may become the bottleneck before the GPU
does. Worth measuring before optimising — 11 min a run does not justify
restructuring.

---

## 8. The 3D study

Decided: **generate new 3D structures.**  The elasticity repo's 80^3 set was
checked as a possible shortcut and rejected — recorded here because the check
itself is reusable.  Of `~/direction-aware-tda-for-porous-materials/structures/`,
sampling 120 per stratum and comparing wrap-face agreement against
adjacent-plane agreement:

| stratum | n | porosity (median) | wrap - adjacent | |
|---|---|---|---|---|
| td | 2376 | 0.632 | +0.006 | periodic |
| attd | 2375 | 0.607 | +0.009 | periodic |
| rtp | 500 | 0.431 | **-0.140** | **not periodic** |

td/attd are periodic and carry precomputed directional TDA and baselines, but
they are the weakly anisotropic strata and are strut/lattice geometries; rtp,
the anisotropic one, is not periodic and so cannot carry a tensor at all.

Measured wall clock: **2391 s for four solves** of one 80^3 structure at the
current defaults, i.e. ~10 min per solve, ~30 min per structure for the three
production solves.  That is ~7.4 days for 5000 on 14 cores **before** the
avg-steps saving in 8.4.

### 8.1 Sizing

Solver cost scales as L^3 (nodes) x L^2 (diffusive transient) = L^5, so relative
to 80: 96 is ~2.5x, 128 ~10x, 256 ~330x.  Per-process memory (double, D3Q19, two
buffers) is 156 MB at 80^3 and 2.5 GB at 256^3 — at 80 the existing
one-process-per-core driver still works unchanged, with no GPU and no OpenMP.
**80 is the default for that reason.**  Measured throughput and the run estimate
are in 8.4.

### 8.2 `scripts/gen_aniso_structures3d.py` + `scripts/structure_io3d.py`

```bash
python scripts/gen_aniso_structures3d.py --self-test          # 16 checks
python scripts/gen_aniso_structures3d.py --output DATA/aniso3d --n_samples 5000
python scripts/structure_io3d.py                              # 8 checks
```

Same model as 2D — integer wavevectors, cyclic strata, per-sample RNG streams —
with the elliptical squash replaced by a **volume-preserving ellipsoidal** one,
`s = (AB)^(1/3) (1/A, 1/B, 1)` in a uniformly random SO(3) frame.  The rotation
is drawn by Shoemake's method from the sample's own generator rather than from
`scipy.spatial.transform`, so content depends on the seed and not the scipy
version (the 5.4 reproducibility rule).

**B is the knob with no 2D analogue**, and it is what gives K three distinct
eigenvalues rather than two:

    B = 1   rod   (prolate)  elongated along e1 only,   k1 > k2 = k3
    B = A   plate (oblate)   elongated along e1 and e2, k1 = k2 > k3

`shape = (B-1)/(A-1)` records it as 0 (rod) .. 1 (plate).

**Rod/plate degeneracy — measured, and it changes how axes must be validated.**
Recovering the elongation axis from the voxels alone (wavevector covariance of
the power spectrum) on the 60-structure pilot:

| generator shape | long axis recovered | short axis recovered |
|---|---|---|
| rod (B~1) | **7.9 deg** median | 11.1 deg |
| plate (B~A) | 23.7 deg (max 86) | **3.9 deg** median |

That is not a bug: a rod leaves e2/e3 degenerate and a plate leaves e1/e2
degenerate, so in each case one axis is not determined by the structure even in
principle.  Correlation of the long-axis error with `shape` is +0.52, of the
short-axis error -0.49.  **Any axis-recovery check must be conditioned on
`shape`**, and no model should regress eigenvector directions (see 8.5).

Structures are stored as `.raw` — 12-byte header (three int32 nx, ny, nz) then
nx*ny*nz uint8, 1 = solid, x fastest — one format only, as in 2D.  It removes the
image-library dependency entirely, so the 3D solver has no external deps.

Generation cost: **~1 core-s per structure**, 60 in 5.4 s on 14 cores, all kept,
zero rejected attempts (3D percolates far more easily than 2D).  5000 is ~6 min.
Verified on the pilot: `mean(wrap - adjacent agreement) = -0.00006` over 180
axis-checks, i.e. exactly periodic; porosity 0.651-0.899.

### 8.3 `LMB3d/lbm3d-perm` — D3Q19 tensor solver

```bash
cd LMB3d && bash make_perm.sh        # no dependencies at all
lbm3d-perm <structure.raw> <results.csv> [options]
```

Three solves (force along +x, +y, +z), one per column of K, in one invocation.
**K = K^T gives three free error bars instead of one, one per off-diagonal.**

Convergence is a plain per-component drift test — every `--lag` steps, compare
each component of the flux with its value one lag earlier and stop when all
three have moved by less than `eps * ||q||`:

```cpp
const double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
double drift = 0.0;
for(int c = 0; c < 3; c++) drift = max(drift, fabs(q[c] - qprev[c]));
if(have_prev && s >= cfg.min_steps && drift < cfg.eps * norm) hold++;
else                                                          hold = 0;
```

The one part that is **not** stylistic is the scale: the tolerance is relative to
`||q||`, not to the component being tested.  A per-component relative test
divides by something that goes to zero — the transverse components are small by
construction and exactly zero for a blocked direction — so it can never be
satisfied and the run burns its budget on an answer that converged long ago.
Those small components are the off-diagonals.  `--hold 2` is kept because one
passing check is not trustworthy (3.4: a 2D residual read 7e-6 at step 30000 and
3.5e-3 at 35000); it is one extra comparison, not a mechanism.

**The 2D machinery was ported once and then removed on the evidence.**  Block
means, the k-sigma tolerance and the separate relative floor were all forced by
2D failures (3.2-3.5), and every one of those traces to `float` distributions
making BGK carry a persistent roundoff fluctuation.  In double the solver reaches
a genuine fixed point, so none of it has anything to do here.  Removing it took
the parameter count from nine to six (`--avg-sample`, `--qfloor`, `--ksigma` are
gone) and, measured on the analytic slit, **produced a bit-identical K in half
the steps** (8000 against 16000, difference 0.00e+00).  A structure that does
fluctuate will now fail to converge and be flagged by `conv_*` rather than
silently mis-reported — the safe direction to fail in.

Three deliberate departures from the 2D code, none of which has a legacy dataset
to stay compatible with:

1. **Guo (2002) forcing**, replacing the 2D equilibrium-velocity shift and its
   raw moment.  This was not a style choice — see 8.4.
2. **double, not float.**  3.2 measured the 2D residual flooring at 1e-3..1e-4
   because U/V are float; at 80^3 double costs ~156 MB per process.
3. **q_i = mean(u_i)**, the standard Darcy superficial velocity, not the 2D
   `sum(u_i)/(L*L0^2)` with L0 = 4 (which existed only for comparability with the
   k0/k2/deeppore surveys).  Consequently **3D k is not comparable in magnitude
   with 2D k**; every ratio, anisotropy and principal direction is.

### 8.4 Solver validation

**Analytic Poiseuille slit.**  A solid slab normal to z gives an exact discrete
answer: with halfway bounce-back the walls sit at half-spacing, so the aperture
is h = (number of fluid layers) and k = nu * sum_j u_j / (F nz) with
u_j = (F/2nu) z_j (h - z_j), z_j = j + 0.5.  The velocity profile came out a
clean parabola with fitted walls at 11.504 and 31.496 for fluid layers 12..31 —
h = 19.99 against an expected 20.

That test is what forced departure 1.  Under the 2D forcing convention the
measured profile sat a **uniform F/4 below** the exact one — confirmed as
0.249999, 0.250009, 0.250001 of F at apertures 10, 20, 28, with the parabola
shape otherwise exact to ~1e-12.  Harmless in 2D (it is a constant offset on the
driven component only, so it cancels from every off-diagonal), but it also makes
a **blocked direction report a small negative permeability** — measured
k_zz = -0.052 where the answer is exactly 0.  With Guo forcing:

| aperture h | k_xx | exact | rel err | k_zz (blocked) | max off-diagonal |
|---|---|---|---|---|---|
| 10 | 2.63020833 | 2.61718750 | 4.98e-3 | -2.2e-12 | 3.3e-11 |
| 20 | 20.88541667 | 20.85937500 | 1.25e-3 | -4.7e-12 | 2.6e-10 |
| 28 | 57.23958333 | 57.20312500 | 6.37e-4 | -1.5e-10 | 4.8e-10 |

The residual scales as **exactly h^-2** (ratios 3.98 and 1.96 against expected
4.00 and 1.96), which is the textbook second-order bounce-back wall-position
error, not a coding error.

`scripts/validate_solver3d.py` packages this as a portable build check — run it
after compiling anywhere, before committing compute, since a bad flag or a
wrong-endian `.raw` reader does not announce itself on a real structure where
there is nothing to compare against.  On the final simplified build:

    h=10   k_xx  2.63020833   exact  2.61718750   rel err 4.98e-03   k_zz -2.2e-12
    h=20   k_xx 20.88541667   exact 20.85937500   rel err 1.25e-03   k_zz -3.1e-12
    error ratio 3.99, expected 4.00 (second order)          PASSED  A blocked direction now returns 0 to 1e-12 and the
off-diagonals of an uncoupled structure vanish to 1e-10.

**Tensor test and reciprocity**, on `sample_000002` (strong, A=2.89, B=1.56,
phi=0.664), force along (1,1,1) predicted from the three axis-aligned solves:

    predicted q = (1.4264e-06, 1.96351e-06, 1.40289e-06)
    measured  q = (1.4264e-06, 1.96351e-06, 1.40289e-06)
    relative error 9.7e-08

    reciprocity residuals   xy 5.9e-08   xz 1.5e-08   yz 8.3e-08
    K = [ 1.35679  0.219479 0.070803 ; 0.219478 1.92115 0.126641 ;
          0.0708031 0.12664 1.42247 ]     k1 2.030  k2 1.395  k3 1.276  FA 0.253

Against 2D's 0.05-0.35% tensor test and 3e-5..1.7e-3 reciprocity, that is four
to five orders of magnitude better.  **Read the two tests together, not
separately.**  Stokes flow is linear in F and every run here stops at the same
step, so the tensor test alone would be satisfied by superposition even on an
unconverged field; it validates the assembly and the absence of index bugs, not
convergence.  Convergence is established independently, below.

**There is no fluctuation to average away — this is the big difference from 2D.**
The block means are *bit-identical* from step 4000 onward:

    # fx 2000  q=(2.01963e-06, 3.254e-07,   1.05586e-07)
    # fx 4000  q=(2.03518e-06, 3.29218e-07, 1.06205e-07)
    # fx 6000  q=(2.03518e-06, 3.29218e-07, 1.06205e-07)
    # fx 8000  q=(2.03518e-06, 3.29218e-07, 1.06205e-07)

and the standard error over the whole 8000-step averaging phase is **exactly
0.0** for six of the nine components and 2e-16 (double roundoff) for the other
three.  The cause is departure 2: 3.2 measured the 2D code flooring at 1e-3
because U/V are float and BGK then carries a persistent roundoff fluctuation —
"past the transient you are not converging, you are sampling a stationary
fluctuation".  In double there is no such fluctuation and the solver reaches a
genuine fixed point.

Two consequences:

1. **The averaging phase, which 3.5 showed was worth a 40x noise reduction in
   2D, buys nothing here** — and it is 8000 of the 16000 steps per solve.  Cut
   it to a short window kept only as a diagnostic (a nonzero `se_*` then means a
   structure that really does fluctuate) and the run gets ~2x cheaper; tightening
   `--lag` on top of that gives ~3x.  **Do not set it from this one structure**
   — a nearly blocked, low-porosity sample may behave quite differently, and
   underestimating the tail is exactly how the 2D cost estimate went wrong by
   4.5x.  Set it from the pilot.
2. **The label-noise law that drove the entire 2D dataset design may not bind in
   3D.**  3.3 built the whole generator around recip_resid scaling with the
   trace, so that k_off is buried unless the ensemble has real anisotropy.  At
   recip_resid ~1e-8 instead of ~1e-4 that constraint essentially vanishes and
   the off-diagonals are clean targets even for weakly anisotropic structures —
   which would also make the `iso` stratum genuinely usable rather than merely
   tolerable.  **This is n = 1.  The pilot must confirm it before any of it is
   relied on.**

**Performance.**  The first working version ran at 9.3 MLUPS.  Streaming was
rewritten as a *pull*,

    p = i - e_k ;  f1[i][k] = solid[p] ? f0[i][opp[k]] : f0[p][k]

which is exactly equivalent to the push form but writes every entry of f1 once,
so the 78 MB per-step memset disappears and the writes become contiguous; macro
and collision were fused into one pass since both are node-local.  Result:
**12.4 MLUPS, 1.33x**, and **bit-identical output** to the reference build on all
three slit tests (relative difference 0.00e+00) — the rewrite is provably
equivalent, not merely close.  Still below the 20-50 MLUPS a tuned D3Q19 kernel
reaches; the remaining cost is the 19 scattered reads per node, which a
structure-of-arrays layout would fix at the price of a real refactor.

### 8.5 Targets

`k_xy`, `k_xz`, `k_yz` are the **symmetrised** off-diagonals and the headline
targets — three analogues of the 2D k_off, and much closer to the elasticity
paper's twelve coupling terms than 2D could get.  Each carries its own
`koff_rel_err_*` from the K = K^T residual.

**Do not regress eigenvector directions.**  In 2D the fix for theta mod 180 was
(cos 2t, sin 2t).  In 3D the principal frame is worse behaved: eigenvectors are
sign-ambiguous, their order swaps when eigenvalues cross, and by 8.2 two of them
are genuinely degenerate for a rod or a plate.  The unambiguous target is the
tensor — the six components of K, or log k_mean plus the five deviatoric ones.
k1 >= k2 >= k3 and the fractional anisotropy are safe; the axes are not.

### 8.5a Partial labels — `solve_permeability3d.py --merge-only`

The 3D LBM runs on another machine and drops one `DATA/aniso3d/perm/<id>.csv`
per solved structure, so the label set fills in over about a week.  `--merge-only`
skips the solver entirely — it does not even check that the binary exists — so it
is the whole workflow on this machine: re-run it whenever new rows land and it
rebuilds `DATA/aniso3d/permeability.csv` from whatever is present.  A short table
is the normal case, not an error.

```bash
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --merge-only
```

**The naming differs from 2D and is the one thing to keep straight.**  The
solver writes the raw tensor as `K_xx .. K_zz` (K_ij = component i under forcing
along j) and the *symmetrised* off-diagonals as `k_xy, k_xz, k_yz`.  So in 3D a
lowercase-k off-diagonal is already the averaged one — the analogue of 2D's
`k_off`, not of 2D's `k_xy`.  No lowercase diagonal aliases are added: `K_xx` vs
`k_xx` would separate raw from derived by letter case alone.  A corollary is
that `koff_noise_ceiling()` in `train_catboost.py` finds no `k_yx` and stays
silent on 3D rather than computing the wrong thing; the ceilings are printed by
the merge instead.

Derived: `trace`, `log_k_mean`, `k_ratio = k1/k3` and its log (k1/k3, not 2D's
k1/k2 — k2 is the eigenvalue that carries nothing for a rod or a plate),
`koff_rel_err_*` and `koff_over_trace_*` one per off-diagonal — all of which the
script already had — plus, added 2026-09-13, the five scale-free deviator
components `dev_{xx,yy,xy,xz,yz}` (dev_zz is minus the sum of the first two),
`conv_all`, the manifest, and the printed noise ceilings.

**The convergence filter now covers all three directions.**  Every consumer
filtered on `conv_fx == conv_fy == 1` — a literal port of the 2D line, which in
3D silently ignores whether the z solve converged.  `train_catboost.py`,
`train_cnn.py`, `learning_curve.py` and `make_golden_split.py` now filter over
whichever `conv_f*` columns exist, so 2D behaviour is unchanged; the split
manifest's `filter` field is derived the same way rather than hard-coded.

**`permeability_manifest.json`** records `n_rows`, the id range and a sha256 of
the id list.  A partial table changes under you between runs, so a `RESULTS/`
directory trained against it is only interpretable next to that manifest —
without it two runs a day apart are silently incomparable.

Snapshot at 2026-09-13: **2816/5000**, ids 0..2815 contiguous, all three
directions converged on every row, ~20 samples/h arriving, so ~4.5 days to go.
The prefix is not a biased subset — the generator interleaves strata by id, and
the solved and unsolved halves agree on stratum share (1/3 each) and on
porosity, `aniso_A/B`, `k0` and `n_modes` to two decimals.

**The 3D labels are far cleaner than 2D.**  The reciprocity residual gives
var(noise)/var(signal) of 5.7e-10, 6.8e-10 and 5.3e-10 for k_xy, k_xz, k_yz
against 9.2e-6 for the 2D k_off — four orders better, because the 3D solver runs
in double and reaches a genuine fixed point (8.3).  The R^2 ceiling is 1.000000
to six places for all three: **on this dataset the label is not the limit.**

Still open: no golden split for 3D.  Freezing one now would draw it from ids
0..2815 only.  That is unbiased, but it permanently excludes the second half of
the dataset from evaluation — decide before, not after, the first learning curve.

### 8.5b `scripts/train_cnn3d.py` — the 3D CNN arm

The end-to-end counterpart of `train_catboost.py` on the 3D descriptors, and the
3D half of §4.5.  Same KFold(shuffle, seed 0), same `metrics.py`, same results
schema, so its rows concatenate with the CatBoost rows.

```bash
python scripts/train_cnn3d.py --self-test          # no GPU, no dataset needed
python scripts/train_cnn3d.py --dataset DATA/aniso3d --backbone densenet121 \
    --targets k_xy k_xz k_yz --augment roll+oct \
    --output RESULTS/cnn3d/densenet121_offdiag
```

**The backbone is written out rather than imported.**  timm is 2D only; MONAI is
the usual 3D source but is one more package to install on the GPU machine, and
no pretrained 3D weights exist for binary pore space, so the ImageNet
initialisation that the 2D runs could use has no 3D counterpart whatever library
provides the graph.  DenseNet (Huang 2016, `Conv3d` throughout) is ~90 lines and
the depth table is the only difference between 121/169/201.  `densenet_small` is
a quarter-size net for smoke runs, named so it cannot be mistaken for a
published depth in a results table.  Sizes: 11.25 M parameters for densenet121,
18.55 M for densenet169.  **The only dependency is torch.**

**`--augment roll+oct` is the 3D analogue of 2D's `roll+d4`, and the reason the
script has a real self-test.**  The 48 elements of the octahedral group — 6 axis
permutations times 8 sign patterns — each relabel the grid exactly, no voxel
interpolated, and each sends K to P K P^T: the identity `rotation_covariance3d`
measured the solver obeying at machine precision (§8.6a).  Because the sign
factor s_a s_b is +1 when a == b, **diagonals and off-diagonals transform
independently**, so the headline set `k_xy k_xz k_yz` gets the full 48x on its
own, and it teaches the sign structure rather than washing it out.  A target set
that is an incomplete family is refused, not guessed: a lone `k_xy` has nowhere
to come from, since a permutation can send it to +/-k_xz.

The trap the self-test exists for: the volume is indexed `[z, y, x]` and the
tensor `(x, y, z)`, so a convention that is self-consistent but mismatched
between grid and label would train on silently wrong labels and still look
healthy.  It is checked **without the solver**, using the voxel-coordinate
covariance — a rank-2 tensor in the same frame as K, so it must transform the
same way.  All 48 elements agree with P C P^T to 3.6e-15 on a non-cubic 5x7x11
array (non-cubic so a bug that only swaps equal axes cannot hide, as in §8.6a).
The label map is checked against P K P^T directly for the six K components, the
three off-diagonals alone, and the deviator whose `dev_zz` is implied not
stored; plus group closure, grid round-trip, solid fraction, and the four
refusals.

Volumes are held as uint8 and cast per batch — 5000 x 80^3 is 2.6 GB as uint8
against 10 GB as float32.  **Activations, not weights, set the batch size**; the
default 8 is a quarter of the 2D default for that reason.  Every run prints and
records the label snapshot from `permeability_manifest.json` (§8.5a), because
while the LBM is still arriving two results directories are only comparable if
they came from the same one.

Not run yet — there is no GPU on this machine (`torch.cuda.is_available()` is
False and `nvidia-smi` is absent), so the arm runs elsewhere.  What that machine
needs: `DATA/aniso3d/structures/` (2.6 GB), `DATA/aniso3d/permeability.csv` and
its manifest, `scripts/`, and torch + sklearn + pandas.  Not timm, not MONAI.

### 8.5c `run_catboost3d.sh` and `run_cnn3d.sh` — the pilot grid

Two flat scripts, the 3D counterpart of `run_training.sh`, split in two because
the arms run on different machines: CatBoost here, the CNN where the GPU is.

```bash
bash run_catboost3d.sh                       # everything
CB_GROUPS="fab porosity tda" bash run_catboost3d.sh    # the headline three
LIMIT=800 bash run_catboost3d.sh             # a first look
bash run_cnn3d.sh                            # on the GPU machine
```

Both loop over the same six single targets — `K_xx K_yy K_zz k_xy k_xz k_yz` —
so their rows concatenate: each writes `RESULTS/aniso3d/<kind>/<unit>/results.csv`
and both end by calling `collect_results.py`, which merges everything found into
`RESULTS/aniso3d/summary.csv` and ranks on `k_xy`.  Mid-run collection is safe,
which matters because the grid is long.

**`CB_GROUPS`, not `GROUPS`.**  `GROUPS` is a bash built-in array of the
caller's group ids and assignments to it are silently ignored — the pilot's
first run passed `607000063` as a feature group.  `run_training.sh` had already
named it `CB_GROUPS`; this is why.

**The CNN loop uses `--augment roll`, and cannot use `roll+oct`.**  A
single-target run is an incomplete family, and `train_cnn3d.py` refuses it
rather than guessing (§8.5b).  The two joint runs that *can* use the full
48-element group are written out in the script's header as the natural
follow-up; they are also the more interesting experiment.

**Cost — measured at two row counts, because one would have misled.**  `tda`
(5787 features) on 24 cores, 2 folds: **260.6 s at 400 rows, 349.6 s at 800**.
Doubling the rows added only 34%, so **cost here is dominated by the per-feature
work, not by rows** — the opposite of the 2D arm, where scaling linearly in
row-folds was roughly right.  A naive linear extrapolation from the 400-row
point alone would have predicted ~1.3 h per (tda, target); the two-point fit
(171.6 s fixed + 0.2225 s/row, then x2.5 on the fixed part and x4 on the row
part going from 2 folds to 5) gives **~49 min** for 2816 rows x 5 folds.  The
nine groups sum to 15213 features, 2.6x `tda`, so one target across all groups
is ~2.1 h and the six-target grid is **~13 h**.

That is an estimate from two points on one group, and §8.6 records the 2D
estimate being wrong by 4.5x when it was not measured — so run `CB_GROUPS="fab
porosity tda"` first.  It is ~40% of the cost and it carries the argument.

### 8.6 Still to do

- ~~Rotation covariance~~ — **done 2026-09-09, see 8.6a below.**
- Pilot ~60 structures for the real per-sample cost, the distribution of
  convergence steps and `se_*` (which sets `--avg-steps`, see 8.4), and the
  `koff_over_trace` histogram, exactly as 5.2/5.3 did for 2D.  **The 2D estimate was wrong by 4.5x
  (6 h -> 27 h) when based on three hand-made structures — do not skip this.**
- Decide `--size` and the sample count from that pilot.
- Then: 3D TDA descriptors and baselines (port from the elasticity repo, which
  is 3D already — this is the piece that gets *easier* going to 3D), and the CNN.

### 8.6a Rotation covariance — PASSED (2026-09-09)

`scripts/rotation_covariance3d.py`.  Relabelling the grid axes is exact and
lossless — no interpolation, not one voxel changes value — so K must transform as
**K' = P K P^T**.  This was the last structural check on the solver, and it is
the one that mattered: the three headline 3D targets are the off-diagonals, and
**neither existing check covers them**.  Reciprocity (K = K^T) is passed by any
index bug that is symmetric under transpose.  The (1,1,1) tensor test is passed
by superposition — Stokes flow is linear in F and all three solves stop at the
same step — even on a consistently mis-indexed field; it validates the assembly,
not the axis labelling.

On a 40^3 strong-stratum structure (A=3.81, B=1.80, porosity 0.880), all five
non-identity permutations, 18 solves, ~2 min wall on 6 cores:

    reference K (trace 5.147606, recip_resid 2.2e-08, steps 8000/8000/8000)
       [   1.48250427    0.08269636   -0.17312505 ]
       [   0.08269633    1.67260383   -0.00501715 ]
       [  -0.17312516   -0.00501727    1.99249753 ]

      perm          steps x/y/z      resid      scale  resid/scale
       xzy       8000/8000/8000   0.00e+00   6.21e-02     0.00e+00
       yxz       8000/8000/8000   1.94e-13   3.69e-02     5.26e-12
       yzx       8000/8000/8000   0.00e+00   9.91e-02     0.00e+00
       zxy       8000/8000/8000   0.00e+00   9.91e-02     0.00e+00
       zyx       8000/8000/8000   0.00e+00   9.91e-02     0.00e+00

`resid` is max |K' - P K P^T| / trace; `scale` is max |K - P K P^T| / trace, i.e.
how far the permutation moves K at all.  **Both numbers are needed.**  On a
near-isotropic structure P K P^T *is* K and the comparison passes for free, so
the script fails a permutation whose scale is below `--min-scale` rather than
reporting a vacuous pass — and generates a strong-stratum structure for exactly
this reason.  Here the permutation moves K by 4-10% of the trace and the residual
is 1e-13, a margin of eleven orders.

Four of the five are **bit-identical**, which is the expected answer rather than
a lucky one: D3Q19, BGK and Guo forcing are all invariant under axis
permutations, and the convergence test compares a max over components against
`eps * ||q||` — also permutation-invariant, which is why all six orientations
stop at step 8000.  The only thing left to differ is the order the flux mean is
summed in memory, and that is the 1.94e-13.

The convention is also *discriminating*, not merely self-consistent: for the two
3-cycles the sigma and sigma^-1 expectations differ by 9.91e-02 of the trace, so
had the array permutation and the tensor permutation been mismatched, the test
would have failed by eleven orders rather than passed.  (The three transpositions
are self-inverse and carry no information on that particular question; they still
carry the index-bug information.)

`--self-test` checks the permutation helper with no solver: shape, porosity,
inverse round-trip, and an every-voxel check of the index map, on a **non-cubic**
5x7x11 array so a bug that only swaps equal axes cannot hide.

Not covered: the solved domain is cubic, so a bug confusing nx/ny/nz survives
this test — as it survives 8.4's slit, and as it would survive production, whose
structures are cubic too.  The Python side of that convention is what
`structure_io3d.py --self-test` covers.

Two earlier attempts were abandoned when 80^3 batches were killed at ~90 min
before writing anything (the solver only writes its CSV at the end).  Covariance
is a property of the index logic and not of the domain size, so 40^3 was always
enough — ~100 s per orientation instead of ~25 min.

---

## 6. Conventions deliberately left alone

- `q_i = Σu_i / (L · L0²)` with L0 = 4, μ = 0.33(τ−0.5) = 0.165, τ = 1 — matching the
  original `volumeflux2d()`, so k_xx stays comparable with the `k` column of the
  existing k0/k2/deeppore surveys.
- The reported velocity is the raw moment `Σf·e/ρ`, **not** the forced-LBM
  `Σf·e/ρ + F/(2ρ)`. This biases the *diagonal* by ~0.3% at ff=1 as a common
  factor and cancels from all off-diagonals, ratios and the anisotropy. Left as
  is for comparability; say so if you'd rather have it behind a flag.
- The structure reader overwrites the j=0 / j=LY−1 rows that `initlbm` marks as
  walls, so the cell **is** doubly periodic. **That is load-bearing for the
  tensor — do not "fix" it.** `getporosity_full()` was added because the original
  `getporosity()` still skips those two rows.

---

## 7. Not done yet

### Step 5 — port the TDA descriptors to 2D — **DONE** (see 4.4, 5.5, 5.6)

Notes kept from the planning stage, since they explain the choices:

Source: `~/direction-aware-tda-for-porous-materials/scripts/`. Notes from reading it:

- **ECP is nearly free**: `Ecp.compute_contributions_2d(::Array{T,3})` already
  exists in the Julia package, so `directional_ecp.jl` needs a one-line change.
  grid_res g → (g+1)² features (81 at g=8, vs 343 in 3D).
- **Filtration** → `(X, Y, 2)`: the cone becomes a double *wedge*; PCA becomes a
  2×2 covariance → one channel, closed-form eigenvector (drop the power iteration).
- **Skip `rotate_grid_to_z` entirely in 2D** — build the wedge offsets at
  arbitrary θ directly. There is no reason to accept the nearest-neighbour
  interpolation loss that the diagonal 3D directions forced on the original.
  Directions 0°/45°/90°/135° (the cone is double-ended, so [1,0] ≡ [−1,0]).
- **PH**: 2D cubical complex, dims 0 and 1 only, periodic in **both** axes (the
  cells are toroidal, unlike the 3D case where only z was periodic).
- `notes/why_ph_insensitive_to_radius.md` in the source repo applies: PH sees
  only channel 0, so `radius` is a no-op for PH — don't scan it in 2D either.
- Baselines: profiles/TPC resample to 256 / 129 points.
- **Load structures with `scripts/structure_io.load_structure`, not `np.load`** —
  the 3D project stored `.npy`; here there is only the GIF, on purpose.
- `train_catboost_directional.py` is essentially target-agnostic — only
  `TARGET_GROUPS` needs new entries.

### Steps 6–7 — models

`scripts/train_catboost.py` is written and its mechanics validated against
stand-in labels; it needs `DATA/aniso/permeability.csv` from the LBM run. It
filters to converged rows and prints the **k_off label-noise ceiling** derived
from the K = Kᵀ residual, so a mediocre R² on the off-diagonal can be read
against what the labels actually support.

Both are built (§4.5, §4.6) and validated; they need only
`DATA/aniso/permeability.csv`. Nothing in the plan is unimplemented. Report R² on all targets;
the interesting ones are k_off and the orientation. Use `koff_rel_err` to state
the R² ceiling imposed by label noise.

### Open items

- Run `run_training.sh` once labels land (~2-3 h for the full grid).
- **Fabric is the baseline to beat.** §5.6 measured 7 fabric features beating
  1124 TDA features on orientation. Rerun that comparison against the real
  `k_off` the moment labels land — it decides the paper's framing.
- Run the rotation-covariance test (§2, change 5) once a model exists: rotating
  an image 90° is exact, and K must transform as R K Rᵀ. Non-directional
  descriptors cannot satisfy this even in principle.
- Check the `koff_over_trace` histogram on the real 5000 — §5.2 predicts medians
  of 0.03 / 0.14 / 0.33 by stratum from a 60-sample pilot.
- Consider whether the `iso` stratum deserves a larger share — it is the
  finite-size-anisotropy arm and the hardest, most novel target, but also the
  noisiest.
- Descriptor hyperparameters are unscanned: wedge radius/height, ECP grid-res,
  PH resolution. Note `--radius` cannot affect PH at all (it feeds only the PCA
  channel, which PH discards), so never scan it for PH.

---

## 9. Quick start

```bash
cd ~/flow/flow-anizotropy
conda activate madrid

# build the solver
(cd LMB2d && bash make_perm.sh)

# sanity-check the generator with no solver involved
python scripts/gen_aniso_structures.py --self-test

# 1. structures  (~75 s for 5000)
python scripts/gen_aniso_structures.py --output DATA/aniso --n_samples 5000 --seed 0

# 2. labels  (~27 h on 14 cores; restartable, so safe to interrupt)
python scripts/solve_permeability.py --dataset DATA/aniso --workers $(nproc)

# 3. descriptors  (~6 min)  and baselines  (~17 s)
bash run_descriptors.sh
python scripts/baselines2d.py --dataset DATA/aniso --output DESC/aniso/baselines.csv

# 4. models  (needs DATA/aniso/permeability.csv from step 2)
bash run_training.sh

# 5. one flow field for ParaView (4.7), independent of everything above
python scripts/solve_velocity.py DATA/aniso/structures/sample_000002_*.gif \
       -o VIZ/2d/s000002 --vtk
```

Steps 1 and 3 are independent of step 2 — descriptors need only the images, so
they can run locally while the LBM runs elsewhere.

Every stage has a self-test that needs nothing else:

```bash
python scripts/gen_aniso_structures.py --self-test   # generator frame + percolation
python scripts/structure_io.py DATA/aniso            # python reader vs the solver
python scripts/filtration2d.py --self-test           # wedge geometry + equivariance
python scripts/baselines2d.py --self-test            # profiles, TPC, fabric
python scripts/train_cnn.py --self-test              # D4 label algebra vs R K Rᵀ
LMB2d/lbm2d-perm <structure.gif> /dev/null --validate 45   # K is a tensor
python scripts/to_vtk.py --self-test                 # VTK ordering, 2D and 3D
```

### 9.1 The 3D pipeline (section 8)

```bash
(cd LMB3d && bash make_perm.sh)                      # no dependencies

python scripts/gen_aniso_structures3d.py --self-test # 16 checks
python scripts/structure_io3d.py                     # 8 checks
python scripts/validate_solver3d.py                  # analytic slit (8.4)
python scripts/rotation_covariance3d.py --self-test  # index map, no solver
python scripts/rotation_covariance3d.py              # K' = P K P^T (8.6a), ~2 min

# 1. structures  (~6 min for 5000 at 80^3)
python scripts/gen_aniso_structures3d.py --output DATA/aniso3d --n_samples 5000

# 2. labels — restartable, so safe to interrupt.  PILOT FIRST (8.6):
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --limit 60 --workers $(nproc)
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers $(nproc)

# single structure, with the Koza09 tensor test along (1,1,1)
LMB3d/lbm3d-perm <structure.raw> /dev/null --validate

# one flow field for ParaView (4.7a) — ~13 min at 80^3, needs no dataset
python scripts/solve_velocity3d.py \
       DATA/aniso3d_pilot/structures/sample_000002_*.raw -o VIZ/3d/v000002 --vtk
```

---

## 10. The 2D sample-size study — results (2026-09-08/09)

Run with `scripts/learning_curve.py` against the frozen golden split
(`splits/aniso_golden.csv`, 1001 held out, SHA-verified; pool 3990), nested
stratum-balanced subsets, sizes 125/250/500/1000/2000/3990.

| arm | `RESULTS/learning_curve/` | runs |
|---|---|---|
| early-stopped (headline) | `es/` | 480 = 2 targets x 8 groups x 6 sizes x 5 seeds |
| fixed 1500 iterations | `fixed/` | 480 |
| CNN densenet121, k_off | `cnn_densenet121/` | 18 = 6 sizes x 3 seeds |

**The `--pca 64` dimensionality control was never run.**  Given the size of the
gaps below it would not change a conclusion, but it is the missing control.

### 10.1 The strong hypothesis fails on k_off

`tda - baselines` is negative at **every** size, paired by seed, and the gap
*widens* with n: -0.095 at 125 to **-0.107 at 3990** (t = -17.9).  No crossover.
Directional TDA is not more data-efficient than the classical baselines on the
off-diagonal — it is uniformly worse.

Mean R^2 on the golden set, `es` arm, target k_off:

| group | 125 | 250 | 500 | 1000 | 2000 | 3990 |
|---|---|---|---|---|---|---|
| all | 0.373 | 0.455 | 0.587 | 0.641 | 0.685 | **0.714** |
| baselines | 0.355 | 0.419 | 0.516 | 0.580 | 0.645 | 0.691 |
| tpc | 0.147 | 0.316 | 0.462 | 0.565 | 0.628 | 0.669 |
| ecp | 0.228 | 0.382 | 0.489 | 0.543 | 0.578 | 0.593 |
| tda | 0.260 | 0.400 | 0.489 | 0.544 | 0.569 | **0.584** |
| ph | 0.183 | 0.284 | 0.332 | 0.399 | 0.412 | 0.446 |
| fab | 0.282 | 0.325 | 0.338 | 0.377 | 0.384 | 0.387 |
| porosity | -0.042 | -0.024 | 0.000 | -0.008 | -0.003 | -0.004 |

**With 5 seeds nothing at n=125 is significant** — seed sd there is 0.20 on
k_off.  Resolving a 0.09 gap against that needs ~40 seeds.  Do not make a claim
about the n=125 point in either direction without rerunning with more seeds.

### 10.2 What does survive — and the 3D comparison table

**TDA is complementary, and worth most at intermediate n.**  `all - baselines`
on k_off, paired by seed:

| n | 125 | 250 | 500 | 1000 | 2000 | 3990 |
|---|---|---|---|---|---|---|
| delta R^2 | +0.018 | +0.036 | **+0.071** | +0.061 | +0.040 | +0.023 |
| t | 0.7 | 2.4 | **4.5** | 8.2 | 7.5 | 6.9 |
| seeds won | 3/5 | 4/5 | 5/5 | 5/5 | 5/5 | 5/5 |

In sample-size terms: to reach R^2 = 0.55 on k_off, `baselines` needs ~720
structures and `all` needs ~410 — a **1.8x data saving from adding TDA**.  At
R^2 = 0.65: 2160 vs 1150, 1.9x.  This is a defensible headline, and it is a
different claim from the elasticity paper's.

**On k_xx there is a genuine crossover.**  `tda - baselines` = +0.082 (t=2.9) at
n=125, +0.063 at 250, +0.046 at 500, ~0 at 2000, -0.009 at 3990.  TDA needs
0.73-0.80x the samples to reach R^2 = 0.75-0.85, and loses at 0.90.  So the
data-efficiency effect is real, but on the *easy* porosity-dominated component
rather than the one the study was built around.

**PH is dead weight.**  `ecp` (324 features) matches or beats `tda` (= ecp+ph,
1124 features) at essentially every size and target; at n=3990 in the `fixed`
arm ecp - tda = +0.013, t=6.8.  The 800 persistence-image features add nothing
to k_off and dilute ECP.  That is ~70% of the feature budget — diagnose before
porting to 3D.

**The CNN is not close.**  densenet121 on k_off:

| n | 125 | 250 | 500 | 1000 | 2000 | 3990 |
|---|---|---|---|---|---|---|
| CNN | 0.736 | 0.774 | 0.833 | 0.853 | 0.884 | 0.905 |
| best descriptor (`all`) | 0.373 | 0.455 | 0.587 | 0.641 | 0.685 | 0.714 |

**The CNN trained on 125 structures beats every descriptor model trained on
3990.**  And it carries a handicap while doing it: `train_cnn.run_fold` uses
`steps = (len(tr)//batch) * epochs`, so at n=125 it gets 90 gradient steps
against 3720 at n=3990.  0.736 is a lower bound.  Fixing the step budget
(`epochs = max(30, round(30*3990/n))`) widens the gap, it does not close it.

**es vs fixed: the ranking is stable, so the conclusion is robust** — but the
effect went the opposite way from the design note's prediction.  Early stopping
helped the *low*-dimensional groups (porosity +0.80 R^2 at n=125, fab +0.12)
and slightly *hurt* tda/all/baselines (-0.01 to -0.04).  `best_iteration`
explains it: `all` stops at 218-966 and `tda` at 254-521, never near the 1500
cap, so the fixed budget was never overfitting the wide groups — it was
overfitting the 1- and 7-feature ones.  **Never quote fixed-arm numbers for
`fab` or `porosity`; they are overfit artifacts.**

### 10.3 The 3D hypothesis, and how it could be a false positive

The plan is to run the 3D study expecting TDA to do better there.  The
mechanistic reason is sound: in 2D, H0 is connectivity and H1 is essentially the
pore-size distribution, both largely redundant with the directional TPC and
fabric tensor — which is exactly what the numbers show (tda ~ ecp, ph ~ junk).
3D breaks that redundancy in a flow-relevant way: H2 exists, H1 becomes
throat-and-channel linkage rather than pore outlines, and connectivity decouples
from porosity.  Permeability is governed by throat bottlenecks and tortuous
connectivity, which two-point statistics are blind to.

Two ways a 3D "win" could be an artifact:

1. **Baseline asymmetry.**  The 2D result is only credible because the baselines
   were strong enough to beat TDA.  The 3D TDA is the piece that gets *easier*
   (the elasticity repo is already 3D, §8.6); if the 3D baselines get less care,
   a TDA win measures effort, not information.
2. **The CNN is mechanically handicapped in 3D** — no ImageNet pretraining for
   3D convs, and 80^3 costs far more per sample than 256^2.  Some of any TDA
   overtake will be the CNN being under-resourced.  Say so before a reviewer
   does.

**Predictions to register before the 3D labels land** (same protocol: frozen
golden split, nested stratum-balanced subsets, 5 seeds, es + fixed arms):

| quantity | 2D result | 3D prediction if the hypothesis holds |
|---|---|---|
| `tda - baselines` on the off-diagonals, full n | **-0.107** | > 0 |
| `all - baselines`, full n | +0.023 | > +0.023 |
| `ph` vs `ecp` | ph dead weight (0.446 vs 0.593) | ph contributes |
| `cnn - all`, full n | +0.191 | < +0.191 |
| where `all - baselines` peaks | n ~ 500-1000 | — |

Not yet adopted as §8.7 — the decision was open at the end of session 3.

---

## 11. Why the descriptors lose — measured (2026-09-13/14)

The 3D results came in (§11.1) and reproduced the 2D headline rather than
overturning it.  That prompted the question this section answers: is the
*dataset* too easy, or is something wrong with the descriptors?  It is the
descriptors, and the mechanism is now measured rather than guessed.

### 11.1 The 3D results — 2816 labels

CatBoost, 9 groups x 6 targets, 5 folds (`RESULTS/aniso3d/catboost/`):

| group | feat | K_xx | K_yy | K_zz | k_xy | k_xz | k_yz |
|---|---|---|---|---|---|---|---|
| tda | 5787 | .9656 | .9679 | .9632 | .7373 | .7418 | .7677 |
| ecp | 3087 | .9502 | .9539 | .9496 | .7326 | .7434 | .7761 |
| ph | 2700 | .9068 | .9111 | .9081 | .6135 | .6025 | .6250 |
| **baselines** | 1819 | .9587 | .9582 | .9589 | **.7653** | **.7598** | .7809 |
| tpc | 369 | .9632 | .9619 | .9621 | .7437 | .7374 | .7504 |
| fab | 10 | -.0985 | -.1025 | -.1197 | .4990 | .4688 | .4492 |
| porosity | 1 | .4370 | .4369 | .4325 | -.1117 | -.0854 | -.0843 |

Paired per fold: `tda - baselines` = **-.028 / -.018 / -.013** on the three
off-diagonals (t = -2.7, -2.2, -2.1).  **TDA alone loses in 3D as it did in 2D**,
and it loses to `tpc` alone — 369 features against 5787.

`all` (7606 feat) on k_xy = **.7846** vs baselines .7653: paired **+.0193**
(t = +2.90, 5/5 folds).  So the 2D complementarity result carries over but
smaller, the opposite of the study's premise.  **See §11.9**: the other five
targets were run on 2026-09-16 and complementarity holds on all six, 5/5 folds
each, p < .05 — and k_xy is the *weakest* off-diagonal, so the effect size to
quote is the off-diagonal mean **+.0264** (about two thirds of 2D's +.0403),
not the +.019 below.
`RESULTS/aniso3d/catboost/k_xy_all/`; `baselines` reproduced bit-for-bit across
the two runs, all five folds.

**The 3D CNN dominates** (`RESULTS/aniso3d/cnn3d/`, densenet121, `--augment
roll`, ~45 min per target): K_xx/K_yy/K_zz = .9959/.9956/.9957, and k_xy/k_xz/k_yz
= **.9582/.9405/.9635** against .765/.760/.781 for the best descriptor group.
That is **4-6x less unexplained variance** on the off-diagonals.  The 2D gap was
+.21, the 3D gap is +.19 from a higher baseline.  The reciprocity ceilings are
1.000000, so none of this is label-limited.

### 11.2 ECP is a re-encoding of the two-point correlation

Predicting ECP from TPC + porosity, out of fold, 2D, 5000 structures
(`scripts/diagnostics/ecp_vs_tpc.py`): linear ridge gets .46 pooled over all 324
features; **CatBoost on the leading ECP principal components gets .834**
variance-weighted over the components carrying 68% of ECP variance.  The
relationship is strong and **nonlinear** — a linear fit alone would have said
"independent".  Control: porosity alone gives **.030**, so this is TPC's shape
information, not the solid fraction leaking through.

**This is mathematics, not a property of this dataset.**  The structures are
thresholded near-Gaussian fields (a sum of 20-80 random cosines), the Euler
characteristic is an **additive** functional, and for Gaussian excursion sets
additive functionals have closed forms in the covariance and threshold
(Tomita/Adler).  ECP could not have carried independent information here.  The
generator's anisotropy is also second-order — an elliptical squash of the
wavevector distribution — so the label mechanism is one TPC captures by
construction.

### 11.3 Lower porosity does NOT fix it — the finding that changed the plan

Two tests, both negative:

* **Within the existing data**, split by porosity quartile with a fixed PCA
  basis and a sample-size-matched control: R^2 runs **.853 (por .65-.72) ->
  .769 (por .84-.90)**.  The gradient points the *wrong* way — lower porosity
  makes ECP *more* TPC-predictable.  The pooled control (random 1250, all
  porosities) is .767, so pooling is the hard case and the .834 above is if
  anything an underestimate at fixed porosity.
* **300 new structures at porosity .45-.55** (`DATA/aniso_lowpor_probe/`,
  `DESC/aniso_lowpor_probe/`, seed 7000, 25% acceptance, 13 s to generate):
  ECP-from-TPC = **.673**, against .697 and .605 for n=300 controls drawn from
  the existing data.  **No difference.**

The probe structures are genuinely a different regime — `dead_frac` std **.141
vs .0024**, `beta0_pore` mean **7.7 vs 1.18** — and ECP's information content
did not move.  So "generate a harder dataset" alone would not have fixed
anything.

**Two corrections to what was said mid-session.**  (a) A first pass reported the
probe backbone at 57% of pore volume; that used *non-periodic* labelling, which
splits clusters joined through the wrap.  On the torus it is **median 93%, mean
88%**.  (b) Binary directional percolation **cannot** vary in any dataset this
generator produces: `gen_one` rejects anything not percolating in both x and y,
so `n_perc_x = n_perc_y = 1` and `backbone_x - backbone_y = 0` identically, for
every structure in every dataset here.  Graded measures are required.

### 11.4 What does escape TPC: non-additive features

On the same 300 probe structures (`scripts/diagnostics/tort_screen.py`), predicted from
TPC + porosity:

| feature | std | R^2 from TPC+porosity |
|---|---|---|
| `beta0_pore` | 8.70 | .416 |
| `dead_frac` | .141 | **.253** |
| `tort_x` | .496 | .499 |
| `tort_y` | .714 | .385 |
| `tort_aniso` (signed) | .962 | .539 |
| *ECP, same structures, same n* | | *.673* |

Tortuosity is the geodesic pore path across the cell over the straight-line
distance, with the cell tiled **transversely** so a path may use the wrap —
without that, a structure percolating only through the wrap scores NaN instead
of long.

**The split falls exactly on additivity.**  Additive functionals (chi, hence
ECP) are ~two-thirds determined by second-order statistics; non-additive ones
(cluster counts, dead volume, geodesic paths) are a quarter to a half.  ECP
measures texture; permeability is governed by connectivity; they are different
mathematical objects.

### 11.5 PH: the persistence images are rank-3

Persistent homology is **not** additive, so unlike ECP it is not forced to be a
function of the covariance — H0 tracks components merging, which is
connectivity.  It is nonetheless the **worst** performing group in both 2D and
3D.  The reason (`scripts/diagnostics/ph_screen.py`, 2D, 5000 structures):

| block | cols | R^2 from TPC | PC0 alone | dims for 99% of variance |
|---|---|---|---|---|
| PH H0 | 400 | **.914** | 54.7% | **3** |
| PH H1 | 400 | **.929** | 53.7% | **4** |
| PH all | 800 | **.933** | | |
| ECP | 324 | .857 | 27.2% | 60 |

**400 columns carry three effective degrees of freedom.**  Per-column standard
deviations are near-uniform (max/median = 1.3, against 2.6 for ECP), i.e. every
pixel moves together scaled by one number — and that number is essentially the
bar count.

`ph2d.py` (and `ph3d.py`) called `gudhi.representations.PersistenceImage` with
`resolution` and `im_range` but **no `weight`**, so gudhi's default constant
weight applied: a bar of length 1e-3 counted as much as a bar of length 1.  A
porous structure yields thousands of near-diagonal noise bars and a handful of
long structural ones, so the image becomes a smeared bar histogram — a local
texture statistic, which is why it is 93% predictable from TPC.  **The
representation discards the topology before the model sees it.**

`ph2d.py` now takes `--weight {constant,linear,quadratic}`, default **linear**;
`run_descriptors.sh` exposes it as `PH_WEIGHT`.  **gudhi applies
`BirthPersistenceTransform` before calling the weight**, so the function
receives `(birth, persistence)` and the linear weight is `x[1]` — *not*
`x[1] - x[0]`, which would be persistence minus birth.  (That error was in an
earlier draft of this section.)  `DESC/aniso/ph/` and `DESC/aniso3d/ph/` were
built with `constant`; anything compared against them must use the same
setting.  The re-run writes to `DESC/aniso_phw/`, whose `ecp/` is a **symlink**
to the existing one — ECP reads the orientation channel and PH the wedge
channel, so the weight cannot change ECP, and recomputing it would only risk
drift.

### 11.5a The weight was not the cause — the bandwidth was (2026-09-14)

`--weight linear` was added and the 2D images recomputed into `DESC/aniso_phw/`
(ECP symlinked, verified bit-identical).  **The prediction failed**: effective
rank went 3 -> 4, and PC0 got *worse* (H0 54.7% -> 69.0%).  Re-weighting changes
*what* scales the image, not the fact that it is one fixed shape times a scalar.

**The real cause is `bandwidth`.**  gudhi's default is 1.0 — that is the
Gaussian sigma — while `im_range` spans 1.25 over 10 pixels.  So **sigma is 8
pixels on a 10-pixel axis**: every diagram point is smeared across the whole
image.  Measured on 200 structures at direction 0, the images at that setting
are **rank 1, PC0 = 1.00**.  Not "nearly degenerate" — a single number.

Measured diagram scale (150 structures, `scripts/diagnostics/ph_sweep.py`
computes these):

| | bars/structure | median persistence | p99 | fraction > 1 px (.125) |
|---|---|---|---|---|
| H0 | 242 (max 3622) | .042 | 1.25 | 18.7% |
| H1 | 61 (max 877) | .021 | **.083** | **0.4%** |

So there is a second, independent problem: `im_range` gives the persistence axis
0-1.25 for both dimensions, but **99% of H1 bars are shorter than one pixel**.
H1 collapses into the bottom pixel row before any smearing.  The filtration is
quantised in steps of ~1/24 = .042, so one pixel is three steps for H0 and H1's
bars are one or two steps long.

**This is inherited, not a porting error.**
`~/direction-aware-tda-for-porous-materials/scripts/ph_calc.py` uses the same
resolution, the same `im_range`, and also passes neither bandwidth nor weight —
so the elasticity paper's PH features very likely carry the same degeneracy.
Worth checking there independently of this project.

**Sweep**, 60 combinations, 200 structures, direction 0
(`scripts/diagnostics/ph_sweep.py`, results in `ph_sweep_results.csv`).
Bandwidth dominates; resolution barely matters; the weight matters a little.
The two homology dimensions then behave **oppositely** as bandwidth shrinks:

* **H0**: rank 1 -> 13 -> 50, but TPC-predictability *rises* (.27 -> .49 -> .58).
  The recovered dimensions are texture the two-point correlation already knows.
* **H1**: rank 1 -> 4 -> 10, and TPC-predictability *falls* (.21 -> .13 -> .06
  with a per-dimension range; to **-.07** with the shared range, which is a
  warning sign — worse-than-mean usually means noise components, consistent
  with H1 being squeezed into a sub-pixel sliver at the shared range).

**Read the absolute R^2 in that sweep only relatively.**  At n=200 CatBoost is
much weaker than at n=5000, so those numbers are not comparable with the .93 in
11.5 — the same sample-size trap as 11.3.

**What the sweep cannot settle**: whether the recovered dimensions predict
permeability.  Rank up and TPC-predictability down is necessary, not
sufficient; noise dimensions score exactly that way.  Only k_off decides it.

**Proposed next step, not yet run.**  Two configurations to the full 5000, both
`linear`, per-dimension range (H0 0-1.25, H1 0-0.10), resolution 20:
bandwidth **0.125** (1 px; H0 rank 13, H1 rank 4) and **0.05** (0.8 px; H0 rank
50, H1 rank 10).  For each: re-measure PH-from-TPC at proper n against the .93
baseline, then CatBoost `ph` on k_off against its current .50, with tpc's .68 as
the bar.  ~4 min to recompute PH per config, ~35 min per k_off run.
**This needs `--bandwidth` and per-dimension `im_range` in `ph2d.py`** — it
currently has neither, and one `im_range` serves both dimensions.

### 11.5b What is on disk from 2026-09-13/14

New directories, so a fresh session knows what they are and what they are not:

| path | what | keep? |
|---|---|---|
| `DATA/aniso_lowpor_probe/` | 300 structures, porosity .45-.55, seed 7000, **no labels** (§11.3) | yes, 1.2 MB |
| `DESC/aniso_lowpor_probe/` | its ECP/PH/baselines | yes, 13 MB |
| `DESC/aniso_phw/` | 2D PH recomputed with `--weight linear`, **bandwidth still the broken 1.0** | **superseded** — 113 MB, delete once the bandwidth fix lands; its `ecp/` is a symlink to `DESC/aniso/ecp` |
| `RESULTS/aniso3d/` | the 3D CatBoost grid, the CNN runs, `k_xy_all`, `summary.csv` | yes |
| `scripts/diagnostics/` | the eight screens behind §11, plus `ph_sweep_results.csv` and `rerun_ph_2d.sh` | yes |

`rerun_ph_2d.sh` is the recipe for recomputing 2D PH into a *separate*
descroot: it regenerates each direction's filtration, re-images, deletes the
filtration, and symlinks ECP rather than recomputing it.  Point `DESCROOT` and
the `ph2d.py` flags at whatever is being tested — that is the loop the bandwidth
work will run in.

**/home was at 129.6 of 150 GiB on 2026-09-14** (`getfattr -n
ceph.quota.max_bytes ~`; `df` does not show this).  20 GiB free, and each PH
config costs ~110 MB, so the bandwidth sweep is affordable — but delete
`DESC/aniso_phw/` before generating several more.

### 11.5c The bandwidth fix, measured (2026-09-15)

`ph2d.py` now takes `--bandwidth` and per-dimension `--im_range_h0/h1` and
`--bandwidth_h0/h1`, and prints each dimension's bandwidth-to-pixel ratio at
startup with a `<-- SMEARED` warning, since that ratio is the quantity that
went wrong.  The warning fires when the bandwidth exceeds a third of the
shorter axis *extent* — not of a pixel, which flags good configurations too
once the grid is anisotropic.  Verified against all four configurations this
project has used (old H0, new H0, both H1); it catches exactly the two broken
ones.  Two configurations were imaged from one filtration pass
(the filtrations dominate the cost; imaging is nearly free) and scored on k_off
with the *exact* settings that produced the recorded .50 — 5 folds, seed 0,
1500 iterations, depth 6, lr 0.05, the same 4991 rows, so the folds are paired.

| configuration | feat | PH-from-TPC | top5 var | **k_off R^2** |
|---|---|---|---|---|
| bw 1.0, res 10, constant (`DESC/aniso`) | 800 | .9325 | 98% | **.502** |
| bw 0.125, res 20, linear, per-dim range | 3200 | .8874 | 64% | **.599** |
| the same, H1 bandwidth 0.0158 | 3200 | .9121 | 45% | **.597** |

**The bandwidth was real: +.097 on k_off, 5/5 folds, t = 13.4, p = .0002.**  The
rank-1 degeneracy is gone — the top five components fell from 98% of the
variance to 64%.

**It is not enough.**  .599 still loses to `ecp` .641, `tda` .654, `tpc` .684,
`baselines` .714, `all` .754 and the CNN's .922.  It does now clear `por` .407
and `fab` .331.  So the §11.7 framing survives intact: a correctly
parameterised persistence image carries more permeability information than the
broken one, and still less than the two-point correlation.  What changes is the
retraction in §11.5a — PH *has* now been evaluated, and the answer is that it
loses honestly rather than by construction.

**The H1 bandwidth is the instructive negative.**  A shared bandwidth with a
per-dimension range is inconsistent: H1 on p[0,0.10] at resolution 20 has a
0.005 persistence pixel, so bandwidth 0.125 is **25 px** — the original bug,
left in place on half the features.  Correcting it to 0.0158 (the geometric
mean of the H1 birth and persistence pixel sizes; the H1 grid is 10:1
anisotropic and gudhi's kernel is not) moved measured rank from 4 to 16 and the
top-5 variance share from 81% to 31% — and moved k_off by **-.002, p = .73**.
The added H1 dimensions were noise.  This is exactly the caveat `ph_sweep.py`
was written to flag ("rank up + TPC-predictability down is necessary but not
sufficient"), now with a worked example: **do not accept effective rank as a
proxy for usefulness.**

Two costs worth recording, since §8.6 is about extrapolations being wrong:

* The resume block budgeted **30-80 min** for the CatBoost leg.  The recorded
  .50 run's own `metrics.json` said **103 s** at 800 features; the 3200-feature
  runs took **454 s** and **445 s**.  The estimate was an order of magnitude
  high, and the answer was already on disk.
* Filtration is ~1 min per direction for all 5000, not the 6 min a 60-structure
  smoke test extrapolated to.  Whole rebuild, both configs, four directions:
  **13 min**.

On disk (`DESC/aniso_ph_bw125/`, `DESC/aniso_ph_bw125_h1s/`, 405 + 410 MB, each
with a `config.txt` recording its parameters and a symlinked `ecp/`), built by
`scripts/diagnostics/rerun_ph_2d_bw.sh`.  Filtrations went to /tmp, which is
94 GiB and not quota-limited; /home is at 133 of 150 GiB.  `DESC/aniso_phw/`
(the weight-only rebuild) is now doubly superseded and can go.

### 11.5d Does the fixed PH move anything that contains it? (2026-09-15)

`tda` (= ecp + ph) and `all` (= ecp + ph + baselines) were re-run on the fixed
images, same settings again, folds paired against the recorded numbers:

| group | feat old -> new | old R^2 | new R^2 | delta | folds | t | p |
|---|---|---|---|---|---|---|---|
| `ph` | 800 -> 3200 | .5018 | **.5990** | **+.097** | 5/5 | 13.44 | **.0002** |
| `tda` | 1124 -> 3524 | .6544 | .6607 | +.006 | 3/5 | 1.05 | .35 |
| `all` | 1903 -> 4303 | .7545 | .7549 | +.000 | 3/5 | 0.10 | .92 |

**The headline does not move.**  `all` stays at .754 and `baselines` .714 is
untouched, so nothing in the study's ranking changes.

This is the sharper version of §11.7, and it is worth stating precisely: the
bandwidth fix recovered real permeability information — a tenth of R^2 on
`ph` alone, unambiguous — and **every bit of it was already carried by ECP and
the two-point correlation.**  PH was not merely weak; it was *redundant*, and
the broken representation had been hiding the redundancy behind a defect.  The
two findings point the same way and neither is an artefact of the other:

* ECP is .834 TPC-predictable because the Euler characteristic is additive and
  the generator's anisotropy is second-order (§11.2).
* PH is not additive and is not forced to be a function of the covariance — yet
  once correctly imaged it still adds nothing on top of ECP + TPC.

So the redundancy is a property of *this structure family*, not only of the
descriptor.  A dataset whose permeability depends on something the covariance
does not determine — connectivity, tortuosity, a percolation-limited backbone —
is the only thing that could separate them, which is what §11.4 and the
screening criterion in §11.7 are for.  Note that this now has a measured price
tag: the criterion has to be met *before* an LBM campaign, because a descriptor
fix this size did not change the answer.

Runs are under `RESULTS/ph_bandwidth/` (a separate root, so `summary.csv` is
not polluted by two different feature sets sharing the group name `ph`).
The H1-corrected descroot was not re-run for `tda`/`all`: it was null on `ph`
alone (§11.5c), where it had the best chance of showing.

### 11.6 The diagnostic scripts

`scripts/diagnostics/` — written this session, kept because every claim in §11
comes from one of them and they are the tools for re-testing after a change:

| script | what it answers |
|---|---|
| `ecp_vs_tpc.py` | is ECP predictable from TPC? (the .834) |
| `ph_screen.py` | the same for PH, split H0/H1, plus effective rank |
| `screen_porosity.py` | does that change with porosity? (fixed PCA basis, matched-n control) |
| `probe_screen.py` | the same on a new dataset, against matched-n controls |
| `connectivity.py` | how much pore connectivity varies across a dataset |
| `connect_feats.py` | backbone fraction, directional beta0 — periodic labelling |
| `tort_screen.py` | geodesic tortuosity, and whether TPC knows it |
| `ph_sweep.py` | persistence-image parameter sweep: rank + TPC-predictability |
| `rerun_ph_2d.sh` | recompute 2D PH into a separate descroot (ECP symlinked) |
| `rerun_ph_2d_bw.sh` | the same for the bandwidth fix: two configs from one filtration pass (§11.5c) |

Two things they get right that are easy to get wrong, both of which bit this
session: **sample-size-matched controls** (R^2 fell .85 -> .67 on going from
1250 to 300 rows, which would have read as a result), and **periodic
labelling** (non-periodic splits clusters joined through the wrap and
overstated fragmentation by a factor of six).

### 11.7 Where this leaves the study

The defensible claim is no longer "TDA beats baselines" — it is **which
topological quantities carry permeability information, and which are
re-encodings of second-order statistics**.  That is a mechanism, with
measurements behind it, and §11.2-11.5 are the evidence.

Next, cheapest first (and see the resume block at the top):

1. **Persistence weight.**  Re-run `ph2d.py` with `--weight linear` (minutes in
   2D), re-screen.  Falsifiable: effective rank should go from 3 to
   well above 10 and PH-from-TPC should leave the .93 band.  Then re-run
   CatBoost on `ph` against k_off — currently .50, the bar is tpc's .68.  In 3D
   this means recomputing the filtrations too (~7.5 h), since `ph3d.py` discards
   the diagrams after imaging — so validate in 2D first.
2. **Connectivity features on the existing labelled 5000.**  Tortuosity is
   graded, so it varies even where backbone fraction is pinned at 1.0.  Do they
   beat .714 on k_off?  ~20 min, no new LBM, real labels.  This decides whether
   a new dataset is needed at all.
3. **Only then** a low-porosity or channel-network dataset — and it needs a
   60-structure LBM pilot first: a fragmented pore space means lower k and
   slower convergence, and the stopping test is relative to ||q|| (§3.1-3.5).

The screening criterion for any new family, checkable before a single LBM
solve: **ECP-from-TPC must come in materially below .83**, measured with a
sample-size-matched control.

### 11.8 The phase, and the connectivity features — both null (2026-09-16)

Two changes aimed at the same diagnosis, run against the real labels on the
same 4991 rows, 5 folds, seed 0 as every recorded `k12__*` number.

**The observation that started it.**  The filtration was computing both channels
**on solid pixels only**, with void at the 1.25 sentinel — inherited verbatim
from the elasticity repo (`speed.py:173`), where the solid carries the load.
For flow it is the pore space that carries the flux, and at solid fraction
~0.15 the solid convention was computing real values on **15.5%** of each cell
and handing the model the other 84.5% as one undifferentiated blob.  There is
also no distance transform anywhere in this repo: the original's
`distance_transfrom_nondirect.py` has `material`/`void`/`both` options, but it
is non-directional, no run script in that repo uses it, and it was never
ported.

`filtration2d.py` now takes `--phase solid|void`.  The swap is symmetric — `fg`
is the only thing that changes — so in void mode `wedge = 1 - (void in the
wedge)/(wedge area)`, the sublevel filtration grows the pore space from the
most open regions along d outward, and an H0 merge time is the constriction at
which two open pore regions first connect along d.  Self-test is 17/17,
including `void(g) == solid(1-g)` at 0.00e+00, which is what catches a
half-applied swap.

**Relabelling alone would have bought nothing** and it is worth knowing why: on
a torus chi(A) = -chi(A^c), so the Euler characteristics of the complements of
the solid sublevel sets are already determined by the solid ECP.  What makes
this a different descriptor is that the filtration *function* moved to the
void, so its sublevel sets are not those complements.  Measured: median
per-feature |corr| between the two phases' ECP is **0.41** (max 0.83), i.e.
genuinely different numbers.

#### Results

`k_off`, CatBoost, 4991 rows.  Reference: `all` .7545, `baselines` .7141,
`tpc` .6842, `tda` .6544 (.6607 fixed-PH), `ecp` .6410, `ph` .5018 (.5990).

| arm | group | feat | R^2 | vs | delta | folds | p |
|---|---|---|---|---|---|---|---|
| A conn | `conn` | 18 | .4069 | tpc | -.277 | 0/5 | — |
| A conn | `baselines+conn` | 797 | .7259 | baselines | **+.0117** | **5/5** | **.005** |
| A conn | `tda+conn` | 1142 | .6757 | tda | **+.0212** | **5/5** | **.028** |
| A conn | `all+conn` | 1921 | **.7605** | all | +.0060 | 3/5 | .30 |
| B void | `ecp` | 324 | .6499 | solid ecp | +.0088 | 3/5 | .41 |
| B void | `ph` | 3200 | .5243 | solid ph (fixed) | **-.0747** | 0/5 | **.002** |
| B void | `tda` | 3524 | .6591 | solid tda (fixed) | -.0016 | 2/5 | .91 |
| B void | `all` | 4303 | .7515 | all | -.0029 | 3/5 | .67 |
| C both | `all+conn` (void) | 4321 | .7551 | all | +.0006 | 3/5 | .93 |
| D phases | `tda` (both) | 7048 | .6820 | void tda | +.0229 | 4/5 | .051 |
| D phases | `all` (both) | 7827 | .7575 | all | +.0030 | 3/5 | .44 |

**Arm B is a clean null.**  Not one comparison favours the void phase.  The
cleanest test is `ecp`, where nothing was retuned — same grid-res 8, same 324
features, only the phase changed: **+.0088, p = .41**.  The void PH being
*worse* than the fixed solid PH is the one result to hedge, since its imaging
parameters were retuned (below); `ecp` has no such escape hatch.

**Arm A works and does not matter.**  Tortuosity is real information —
`conn_tort_off` alone has Spearman **-0.740** with k_off, and each axis
tortuosity predicts its own diagonal component (-.71 vs -.38 for the other) —
and it lifts `baselines` and `tda` at 5/5 folds.  On top of `all` it is +.006
at p = .30.  Note that the **cluster half of the family is dead on this
dataset**: `backbone_aniso` is identically zero, `dead_frac` averages .0005,
`beta0_pore` is 1 for most structures, because the generator rejects anything
that does not percolate both ways.  §11.4's encouraging numbers for those were
measured on the low-porosity *probe*, not on `DATA/aniso`, and they do not
transfer.  Arm A is a test of directional tortuosity, not of connectivity
broadly.

#### Why arm B was null — the screen

`ph_screen.py` on both descroots, same script, same n, same TPC features:

| block | solid (bw125) | void |
|---|---|---|
| PH H0 (components) | .8901 | .8119 |
| PH H1 (loops) | .8023 | .8858 |
| PH all | .8874 | **.8926** |
| ECP | .8571 | **.8561** |

**The two phases are equally re-encodings of the two-point correlation, to
within noise** — ECP differs by .001.  The H0/H1 split flips (components
dominate in the solid, loops in the void, which is the duality the bar
statistics show) but the total relative to TPC does not move.  Both phases are
complements of one thresholded Gaussian field, so whatever second-order
structure one encodes, the other encodes too.  The "solid wastes 85% of the
cell" argument was about *coverage*; coverage is not the binding constraint,
TPC-redundancy is.

#### Where this leaves §11.7

Three independent instances of one pattern now exist, and they are the whole
finding:

| fix | gain on the descriptor alone | gain on `all` |
|---|---|---|
| PH bandwidth (§11.5c) | `ph` **+.097**, p = .0002 | +.000, p = .92 |
| connectivity (this section) | `baselines` **+.012**, p = .005 | +.006, p = .30 |
| the void phase (this section) | nothing, p = .41 | -.003, p = .67 |

Three different *kinds* of fix — the representation, the feature family, and
the phase the descriptor is even about — and `all` has sat at .754 through all
of them, while the 3D CNN reaches .96.  **The ceiling is not the descriptors.**
On this structure family permeability is determined by second-order statistics,
so anything added is already spanned once TPC is in the set.

So §11.7's screening criterion is not one option among several, it is the
binding constraint, and it now rests on three negatives rather than one: a new
structure family whose permeability turns on something the covariance does not
fix, with **ECP-from-TPC materially below .83** checked before a single LBM
solve.

#### Three things not to repeat

1. **Do not reuse PH imaging parameters across a phase change.**  Measured
   persistence p99 on 40 structures: solid H0 1.25 / H1 0.11; void H0 0.31 /
   H1 **1.25**.  The dimensions swap roles with the phase.  Session 7's
   settings put H1 on p[0,0.10]; on void that crushes the informative dimension
   into one pixel row — exactly the bug §11.5c removed.  Each dimension here
   gets a range covering its own p99 and a bandwidth at the geometric mean of
   the two pixel sizes (0.6-1.7 px, no SMEARED flag).
2. **A slab crossing cannot express a diagonal.**  The first `connect2d.py`
   sheared the lattice so [1,1] became an axis and measured a crossing as
   `tort_screen.py` does.  A shear is an exact lattice automorphism, so it does
   not change *which* paths cross a slab: `d1` and `d2` came out bitwise equal
   and the self-test caught it.  What distinguishes a direction is the net
   displacement, so the quantity is `d_geo(p, p + L*d) / |L*d|` — the shortest
   pore cycle winding once around the torus along d.  Exactly 1 for a straight
   channel, infinite when blocked, antisymmetric under the y -> -y torus mirror
   (not `flipud`, which is that mirror composed with a translation and breaks
   the anchor sublattice).
3. **Wait on the pipeline's DONE, not on the output file existing.**  A waiter
   armed on `[ -f descriptors.csv ]` fired while `collect_descriptors.py` was
   still writing, and the first arm-B run silently trained on **3891 rows
   instead of 5000**.  R^2 moves with sample size (§11.6), so this does not
   crash, it just produces a number that looks fine and is not comparable.
   Those runs were deleted and rerun.

#### What is on disk

    RESULTS/connect/k12__{conn,baselines_conn,all_conn,tda_conn}
    RESULTS/void/k12__{ecp,ph,tda,all}
    RESULTS/void_connect/k12__{tda_conn,all_conn}
    RESULTS/bothphases/k12__{tda,all}
    DESC/aniso/connect.csv            5000 x 18   (20 NaN: blocked directions)
    DESC/aniso_void/descriptors.csv   5000 x 3524 (ecp 324 + ph 3200, 0 NaN)

Separate roots on purpose, as with `RESULTS/ph_bandwidth/`: `summary.csv` must
not end up with two different feature sets sharing the group name `ecp`.
`scripts/diagnostics/compare_runs.py` does the paired per-fold test and refuses
to compare runs whose seed, fold count or row count differ — which is what
caught the 3891-row contamination.

### 11.9 The 3D `all` row, completed (2026-09-16)

The session-6 grid ran 9 groups x 6 targets but **`all` was not one of the 9** —
it was added afterwards as a one-off for `k_xy` alone, because that was the
target the argument turned on.  The other five were never run.  They are now
(`RESULTS/aniso3d/catboost/<target>_all/`, settings copied verbatim from
`k_xy_all/arguments.json`, so every fold pairs with both the one-off and the
9-group grid).

| target | `all` | `baselines` | `tda` | `all` - `baselines` | folds | t | p |
|---|---|---|---|---|---|---|---|
| K_xx | .9805 | .9587 | .9656 | +.0219 | 5/5 | 8.61 | .0010 |
| K_yy | .9826 | .9582 | .9679 | +.0244 | 5/5 | 24.31 | .0000 |
| K_zz | .9792 | .9589 | .9632 | +.0203 | 5/5 | 10.79 | .0004 |
| k_xy | .7846 | .7653 | .7373 | +.0193 | 5/5 | 2.90 | .0442 |
| k_xz | .7928 | .7598 | .7418 | +.0330 | 5/5 | 8.68 | .0010 |
| k_yz | .8078 | .7809 | .7677 | +.0270 | 5/5 | 3.61 | .0225 |

**Complementarity holds on all six targets**, every one at 5/5 folds and
p < .05.  This is the study's one surviving positive claim and it now rests on
six measurements instead of one at p = .044.

**It also corrects the number quoted in §11.1.**  That section reports 3D
complementarity as **half** the 2D size (+.040 -> +.019), but +.019 is `k_xy`,
which turns out to be the **weakest** of the three off-diagonals, not a typical
one.  The off-diagonal mean is **+.0264**, about two thirds of 2D's +.0403 on
k_off.  The qualitative point stands — 3D complementarity is smaller than 2D,
which is still the opposite of the study's premise — but the figure in §11.1 is
the low end of the range rather than its centre, and should be quoted as the
three-target mean.

The general lesson is the one §11.6 records for sample size, in another guise:
**a single target is not a sample.**  Picking the target the argument turns on
and running only that one is how a +.019 gets written down as the effect size
when the effect is +.026.

Cost: 5 x ~19 min.  The descriptors, labels and code were all already on disk;
nothing but the runs was missing.

### 11.10 Channel networks fail the screen — and why (2026-09-17)

First candidate for the new structure family of §11.7, built as
`scripts/gen_channel_structures.py`: a **stick network**.  `n_seg` finite
segments thrown on the torus, orientation from a von Mises on the doubled angle
(mean 2*psi, concentration kappa, so pi-periodic and concentrated about psi),
per-segment length and integer width; pore is the union of the dilated segments.
Periodic by construction — rasterised coordinates modulo `size`, and a
roll-based dilation rather than tile-and-crop, so there is no seam to get wrong.
Self-test 13/13, including the NOTES 2 wrap statistic (inside .905, across .917,
chance .501) and the frame checks against `theta_deg`.

600 structures, 5 s to generate, porosity .34-.72.

**It fails, in the wrong direction.**  ECP-from-TPC, variance-weighted mean R^2
over the top 5 PCs:

| subset | n | porosity | ECP | PH |
|---|---|---|---|---|
| channel network | 600 | .34-.72 | **.7581** | .7623 |
| DATA/aniso control | 600 | .65-.90 | .6860 | .6962 |

and with the porosity spread matched as well, which removes the one artefact
that could have produced this (porosity is an input to the predictor, so a wider
porosity range explains more of ECP's variance for free):

| subset | n | porosity sd | ECP | PH |
|---|---|---|---|---|
| channel, phi .45-.62 | 337 | .0488 | **.7793** | .7759 |
| DATA/aniso, phi .70-.87 | 337 | .0490 | **.6291** | .6700 |

Matched n and matched spread: the candidate is **+.150 more** TPC-predictable
than the family it was meant to replace.

**Why — and it is a design error, not bad luck.**  A union of *independently
placed* sticks is a **Boolean model** (Poisson germs, random grains).  For
Boolean models the Minkowski functionals, Euler characteristic included, have
closed forms in the intensity and the mean grain measures (Miles/Davy), and the
two-point correlation is a function of the same parameters.  So ECP and TPC are
again analytically tied — the same failure mode as Gaussian excursion sets
(§11.2), reached by a different closed form.  Changing the *geometry* of the
grain from a smooth blob to a stick changed nothing that mattered;
**independent placement is what has to go.**

Three ways to break it, cheapest first:

1. **Junction blocking.**  Keep the network, then put small solid plugs at a
   random subset of channel intersections.  A few dozen 3-4 px plugs barely move
   S2, but each splits a loop and can sever a conducting path — so connectivity,
   and permeability, decouple from the second-order statistics almost by
   construction.  Reads physically as cementation at grain contacts.
2. **Arrest rules.**  Grow each segment until it meets an existing one and stop
   (T-junctions), as real fracture networks form.  Placement becomes strongly
   correlated and dead-end branches are generic rather than accidental.
3. **Hard-core exclusion.**  Segments may not overlap — a Matern-type process
   rather than Poisson, which breaks the closed forms outright.

**A correction to §11.7's criterion.**  "ECP-from-TPC materially below .83" is
**n-dependent** and should not be read as an absolute threshold: .834 was
measured on 5000 structures, and the same `DATA/aniso` scores **.686** at n=600
and **.629** at n=337.  The operative test is the matched-n, matched-spread
control, which `scripts/diagnostics/screen_family.py` now runs by default.  Read
the absolute number only against a control measured the same way.

**What the family did improve**, for the record, is graded connectivity —
`dead_frac` mean .0065 and max .521 against DATA/aniso's .0005 and .274,
`beta0_pore` mean 1.67 vs 1.20.  Not enough to matter given the screen, and
`backbone_aniso` is *still* identically zero because the generator rejects
non-percolating structures (§11.3b): binary connectivity cannot vary in any
dataset produced by a generator with that rejection step, whatever the family.

On disk: `DATA/aniso_chan/` (600), `DESC/aniso_chan/` (descriptors, baselines,
connect), `scripts/gen_channel_structures.py`,
`scripts/diagnostics/screen_family.py`.

### 11.11 Junction blocking, and a retracted result (2026-09-17)

Follow-on from §11.10, which diagnosed the plain stick network as a Boolean
model and named independent placement as the thing to remove.  `--block_min` /
`--block_max` in `gen_channel_structures.py` put small solid plugs at a random
subset of channel intersections: plug positions are conditioned on where sticks
crossed, so they are not an independent grain process, and a plug splits a loop
and can sever a path while removing only a handful of pixels.

**Two bugs found by the self-test, both about the periodic seam**, and the
second was not cosmetic: a junction blob straddling the wrap was labelled up to
three times (its copies land near different corners of the 3x3 tiling), and its
centroid, averaged in tile coordinates across the seam, landed in the middle of
the cell — so the plug was placed nowhere near the junction.  Replaced with
union-find over the two wrap edges plus circular-mean centroids
(`_label_periodic`).  The 3x3-tiling trick used elsewhere is right for large
pore clusters, which reach their own translates through the bulk, and wrong for
a small blob on the seam.

#### The retraction

The first screen reported the blocked family's **PH-from-TPC at .4887** against
.7623 unblocked, with a tidy mechanism attached: ECP is additive so each plug
contributes a fixed increment and chi stays tied to counting statistics, while
PH is not additive and escapes.  **That result was an artefact of two
independent faults.**

1. **`run_descriptors.sh` never passed `--bandwidth`**, so every dataset built
   through it got gudhi's default of 1.0 — eight pixels at `--resolution 10`
   over [0,1.25].  This is exactly the §11.5a bug, fixed in `ph2d.py` in session
   7 and still live in the driver.  Both channel datasets measured **numerical
   rank 71, top-5 variance .998** against .792 / rank 543 for the fixed set.
   `ph2d.py` printed `<-- SMEARED` eight times; it was filtered out of the log
   by a `grep -v` on the progress bars.
2. **`ph_screen.py` hardcoded `DESC/aniso/baselines.csv`.**  Correct for its
   original job, comparing imaging parameters on one dataset; silently wrong for
   a different dataset, where it joins the candidate's PH to unrelated
   structures' TPC on `sample_id` and returns R^2 ~ 0 that reads as "PH escapes
   the covariance".

Both are fixed at the cause: the driver now passes explicit per-dimension
bandwidths and **aborts** if `ph2d` reports SMEARED, and `ph_screen` uses the
descroot's own baselines and says so loudly when it falls back.

#### After the fix — and a second confound

Rebuilt with per-family imaging (the same *rule* each time: H1 range covers that
family's own persistence p99, bandwidth at the geometric mean of the two pixel
sizes).  The families differ: blocked H1 p99 is **0.85-0.90** against **0.125-0.146**
unblocked, because blocking creates long-lived loops.  All sets verified
non-degenerate: rank 599 (full), top-5 variance .834 / .893.

| family | porosity | ECP | PH |
|---|---|---|---|
| channel + blocking | .34-.71 | .7290 | .5875 |
| channel, unblocked | .34-.72 | .7581 | .7988 |
| DATA/aniso (fixed PH) | .65-.90 | .6863 | .7549 |

**ECP fails the screen outright** — both channel families are *more*
ECP-predictable than the dataset they were meant to replace, and blocking only
moves it .758 -> .729 against .686.  ECP never touches the bandwidth, so these
three numbers were never in doubt.

**The PH column is confounded with the imaging range.**  The same blocked
structures imaged with the unblocked family's H1 range score **.7623**, not
.5875, while ECP is bit-identical at .7290 in both.  So PH-from-TPC moves by .17
on a parameter choice, and the blocked-vs-unblocked gap cannot be attributed to
blocking on that evidence.  The 2x2:

| | wide H1 p[0,0.90] | narrow H1 p[0,0.15] | gap |
|---|---|---|---|
| blocked | .5875 | .7623 | |
| unblocked | .6634 | .7990 | |
| gap | **.076** | **.037** | |

Decomposed over the full 2x2: the **imaging range moves PH-from-TPC by ~.156**
(wide .625 vs narrow .781, averaged over families) and **blocking by ~.057**
(blocked .675 vs unblocked .731, averaged over ranges).  So blocking does have a
real effect — the sign is consistent at both ranges, which an artefact would not
be — but it is **~.06, not the .21 first reported**.  About three quarters of
that headline was the range.

**The candidate is rejected.**  ECP fails outright either way (.729 blocked,
.758 unblocked, against DATA/aniso's .686), and a .06 movement in a quantity
that is not comparable across datasets without matched imaging is nowhere near
enough to justify an LBM campaign.  Note also that the .5875-vs-.7549
comparison against DATA/aniso remains confounded: that reference was imaged with
its own third set of parameters (H1 p[0,0.10], bandwidth 0.125), and no matched
re-imaging of DATA/aniso was run.

#### The lesson worth keeping

**PH-from-TPC is not a family-invariant quantity.**  It moves by .17 on the H1
range alone, which is the same order as every "family effect" measured here.
Any comparison of PH-predictability across datasets must therefore either hold
the imaging fixed — and accept that one family is mis-imaged — or report the
whole parameter sweep.  A single number per family is not interpretable.  ECP
has no such freedom, which is an argument for keeping the §11.7 criterion on
ECP even though ECP is the additive descriptor least able to escape the
covariance.

On disk: `DATA/aniso_chanblk/` (600), `DESC/aniso_chan{,_v2,_xp}/`,
`DESC/aniso_chanblk{,_v2,_xp}/`.  The `_v2` roots carry the per-family imaging,
`_xp` the swapped-parameter cross-checks; the un-suffixed roots are the
SMEARED ones and are kept only so the retraction can be re-derived.

### 11.12 Throat-scale PH — the descriptor side was underbuilt (2026-09-17)

Step 0 of the plan that followed §11.11: before generating yet another dataset
family, test whether the descriptors were ever given the right filtration
*function*.  No new LBM — this runs on the existing labelled 4991.

**Why a distance transform.**  `filtration2d.py` measures *occupancy*, the
solid fraction of a wedge.  §11.2 is the argument against that: for a Boolean
model or a Gaussian excursion set the Minkowski functionals of occupancy fields
have closed forms in the same parameters that fix the covariance, so ECP
re-encodes TPC and PH inherits most of it.  Critical path analysis says
permeability is set by something occupancy cannot express — the narrowest
constriction on the widest spanning path — and that quantity is a *radius*.
The Euclidean distance transform of the pore space is exactly the inscribed
radius, so in a sublevel filtration of `-dt` **an H0 death time is a throat
radius**.  §11.8 recorded that no distance transform exists anywhere in this
repo: the elasticity original's `distance_transfrom_nondirect.py` was never
ported.  This is that port, plus a directional mode.

`scripts/filtration_throat2d.py`, two modes, both on the void phase:

* `--mode iso` — the original's non-directional `void` option.
  `f = 1 - min(dt, RCAP)/RCAP`, solid at the 1.25 sentinel as everywhere else.
* `--mode dir` — walk ±d at most `--horizon` steps, stopping at solid, and take
  the **minimum** dt encountered: the narrowest constriction met travelling
  along d.

**An anisotropic metric was tried first and is wrong.**  Making displacements
along d cheap and across d expensive measures the channel width perpendicular
to the flow, which reads a channel *blocked* along d as "infinitely wide".  The
min-along-the-walk form gets both limiting cases right and the self-test pins
them: in a channel running along x, θ=0 returns the half-width and θ=90 returns
the wall radius.

**RCAP is a fixed global constant, not a per-structure maximum.**  Normalising
by each structure's own `dt.max()` would destroy the absolute throat scale, and
k ~ r² makes that scale the whole point.  Measured over 60 structures of
`DATA/aniso`: dt p50 **6.40**, p75 11.00, p90 16.12, p95 19.70, p99 **27.66**,
p99.9 37.59 px, per-structure max 13.9-53.7.  RCAP = 28 (the p99).  Descriptors
built with different caps are not comparable; it is recorded in `config.txt`.

**Self-test 19/19**, including the property the descriptor rests on: two
chambers joined by a throat of known half-width w, and the long-lived H0 bar
must die at the level with RCAP(1-t) = w.  Exact to **0.000 px** at
w = 2, 3, 4, 6, 8 — checked over a 4x range, because one width would also pass
under a wrong but monotone calibration.

**Imaging measured from this filtration's own bars**, per §11.5c/§11.11, not
copied from the wedge's values.  Over 40 structures:

| mode | H0 pers p99 | H1 pers p99 | H1 bars/struct |
|---|---|---|---|
| iso | 0.354 | 0.815 | 38 |
| dir a0 | 0.403 | 0.666 | 38 |
| dir a45 | 0.348 | 0.379 | 944 |

so H0 persistence is ranged to 0.45 and H1 to 0.95, each bandwidth the
geometric mean of its two pixel sizes (0.0335, 0.0487 — about 1 px).  No
SMEARED report.  Non-degenerate: rank **1672** of 3200 (dir) and **420** of 800
(iso), top-5 variance .893 / .848, the same regime as the accepted fixed-PH
sets.  Note the contrast with the wedge, where H1 bars are near-diagonal noise:
here iso H1 persistence has **p50 = 0.41**, i.e. the loops are structural.

#### Results — k_off, CatBoost, 4991 rows, 5 folds, seed 0

`dir`, four directions, 3200 features.  `tpc` and `baselines` were re-run in
the same call and reproduce the recorded numbers **exactly**, fold for fold,
which is what makes the pairing below valid.

| group | feat | R^2 | vs tpc | vs baselines | vs all |
|---|---|---|---|---|---|
| `thr` | 3200 | **.7060** | +.0219 (3/5, p=.16) | -.0081 (2/5, p=.57) | -.0484 (0/5, p=.007) |
| `tpc+thr` | 3460 | **.7534** | +.0693 (5/5, p=.0007) | +.0393 (5/5, p=.007) | -.0010 (3/5, p=.84) |
| `baselines+thr` | 3979 | **.7614** | +.0772 (5/5, p=.002) | **+.0472 (5/5, p=.010)** | +.0069 (4/5, p=.33) |

Reference, recorded: `all` .7545, `baselines` .7142, `tpc` .6842, `tda` .6544
(.6607 fixed-PH), `ecp` .6410, `ph` .5018 (.5990), `porosity` -.0599.

**Three things this says.**

1. **Throat PH is by far the best TDA descriptor this study has produced** —
   .7060 against the previous best of .6607, and it is the first one that is
   not *significantly worse* than `tpc` (p=.16) or than the full `baselines`
   (p=.57).  It draws with them; it does not beat them.  Every earlier TDA
   group lost, `ph` by .18.
2. **It is genuinely complementary.**  +.0472 on top of `baselines` at 5/5
   folds, p=.010 — four times the lift the connectivity features gave in §11.8
   (+.0117).  And `tpc+thr`, which is nothing but the two-point correlation
   plus throat PH, **matches `all`** (.7534 vs .7545, p=.84) using neither ECP
   nor the wedge PH nor porosity nor fabric.
3. **It does not break the ceiling.**  `baselines+thr` over `all` is +.0069 at
   p=.33.  That is the fourth independent descriptor improvement to land on
   ~.754, after the three in the descriptor-ceiling entry.  **.754 is the
   dataset.**

#### The iso control, and why it reads as a sanity check

| group | feat | R^2 | vs tpc | vs baselines |
|---|---|---|---|---|
| `thr` (iso) | 800 | **-.0860** | -.7701 (0/5) | -.8001 (0/5) |
| `tpc+thr` (iso) | 1060 | .7320 | +.0478 (5/5, p=.001) | +.0178 (5/5, p=.039) |
| `baselines+thr` (iso) | 1579 | .7453 | +.0612 (5/5, p=.001) | +.0312 (5/5, p=.014) |

The non-directional transform alone is **worse than the porosity null**
(-.086 against -.060), and that is correct rather than alarming: k_off is the
off-diagonal, a purely directional quantity, and an orientation-free descriptor
cannot carry it by symmetry.  Paired against the directional version the gap is
+.7920 at 5/5 folds.  So the directional walk is doing the work on k_off, while
the isotropic transform still contributes the **absolute throat scale** — it
lifts `baselines` by .031 on its own.

#### The rest of the tensor (added later the same day)

§11.12 was first written on `k_off` alone.  `k_xx` and `k_yy` were then run for
the same three throat groups, so the comparison exists on every element.  `tpc`
and `baselines` reproduce their recorded values exactly in all three calls.

| method | feat | k_xx | k_yy | k_off |
|---|---|---|---|---|
| `tpc` | 260 | .9339 | .9375 | .6842 |
| `baselines` | 779 | .9249 | .9290 | .7142 |
| `tda` (wedge) | 1124 | .9150 | .9189 | .6544 |
| `all` | 1903 | **.9579** | **.9604** | .7545 |
| `thr` | 3200 | .9277 | .9255 | .7060 |
| `tpc+thr` | 3460 | .9456 | .9502 | .7534 |
| `baselines+thr` | 3979 | .9467 | .9505 | **.7614** |

**`thr` alone draws with `baselines` on all three elements** — +.0028 on k_xx
(3/5, p=.61), -.0036 on k_yy (2/5, p=.44), -.0081 on k_off (2/5, p=.57).  Three
draws and no losses, where every wedge group loses at least one element badly
(`ph` is .63/.64 on the diagonals, below `por`'s .69).

**The complementarity is not a k_off artefact.**  `baselines+thr` over
`baselines` is +.0218 on k_xx (5/5, p=.003), +.0215 on k_yy (5/5, p=.0006) and
+.0472 on k_off (5/5, p=.010).  It holds on the whole tensor.

**But `all` still wins both diagonals** — +.0112 on k_xx (p=.031) and +.0099 on
k_yy (p=.002) over `baselines+thr`, while losing k_off by .0069 (p=.33).  So
the wedge descriptors carry diagonal information the throat filtration does not
capture, and the two filtrations are **not redundant with each other**.  The
diagonals are nearly saturated for every real method (.91-.96), so they
separate methods far less than k_off does — but the sign here is consistent and
significant, which the k_off comparison against `all` is not.

**`all+thr` — the union — was then run, and it is not the answer.**  Built from
`DESC/aniso/descriptors.csv` + `baselines.csv`, the exact pair the recorded
`all` used per its `arguments.json`, so the delta is the throat block alone and
not the §11.5c bandwidth fix.  `all` was re-run in the same calls as an
internal control and reproduced .9579 / .9604 / .7545 **exactly** on all three
targets.

| target | `all` | `all+thr` (5103f) | delta | folds | p |
|---|---|---|---|---|---|
| k_off | .7545 | **.7624** | +.0080 | 4/5 | .17 |
| k_xx | **.9579** | .9497 | -.0082 | 1/5 | .076 |
| k_yy | **.9604** | .9534 | -.0070 | 0/5 | **.005** |

It gains .008 on the off-diagonal, not significantly, and **loses on both
diagonals — on k_yy at 0/5 folds, p = .005**.  Read as feature dilution: 3200
throat columns stacked on 1903 already-sufficient ones, against targets already
at .96, give the trees more ways to split on noise, and the fold spread widens
from ±.003 to ±.012 accordingly.  The prediction going in was the opposite —
that the union would gain on the diagonals, since that is where `all` beat
`baselines+thr` — and it was wrong.

Note also that `all+thr` is only +.001 over `baselines+thr` on k_off with 1124
more features.  **On the off-diagonal the throat block makes the wedge block
redundant**, and the cheapest model in the band is `tpc+thr` at .7534 from 260
baseline features plus the throat images: no ECP, no wedge PH, no porosity, no
fabric.

So on k_off five models now sit between .7534 and .7624, separated by nothing
significant, and the ~.76 ceiling has been approached from four directions.
On the diagonals `all` is best and nothing involving the throat descriptor
improves on it.

On disk: `RESULTS/throat/k{11,22}__dir/`, `RESULTS/throat/k{11,12,22}__allthr/`,
`THROAT_DIAG.log`, `ALLTHR.log`.

#### The mechanism, measured

`ph_screen.py` (which now takes `--prefix`) at n=5000 against the same X, i.e.
`tpc` + porosity:

| block | throat PH | fixed wedge PH |
|---|---|---|
| H0 (components) | **.7053** | .8901 |
| H1 (loops) | .8510 | .8023 |
| all | .8478 | .8874 |
| ECP, for reference | — | .8571 |

The gain is **concentrated in H0, .705 against .890** — a .185 gap — which is
exactly where the critical-path story puts it: H0 death times are throat radii,
and those escape the covariance in a way the wedge's component structure does
not.  H1 moves the other way (.851 vs .802), so this is not a uniform
improvement in representation quality; it is specifically connectivity.

Two cautions on that table.  Read it alongside a real target, never alone
(§11.5c): the overall column moves much less than H0 does (.848 against .887),
so a screen run only on `all` would have called this descriptor a marginal
change.  The k_off numbers are the ground truth.  And the n-dependence warning
from §11.10 is confirmed here: the fixed
wedge PH screens at **.8874 at n=5000** against the **.7549 at n=600** quoted
in §11.11's table.  Those are the same descriptors on the same structures.
Never compare a screen number across sample sizes.

#### What this does to the plan

Step 0 was there to decide whether steps 1-3 are worth doing, and the answer is
**yes, with a measured reason rather than a hypothesis**.  The descriptors were
underbuilt: a filtration function chosen from the physics recovers ~.05 R^2
over the best previous TDA and adds ~.05 on top of the baselines that TDA had
never improved.  But the ceiling still binds, and the dataset is the reason.

And note *where* this was measured — the family least favourable to it.  Mean
porosity .78, median inscribed radius 6.4 px, and the generator rejects
non-percolating structures, so throats are almost never limiting.  Critical
path analysis predicts the throat descriptor's advantage grows as the pore
space approaches the percolation threshold, which is precisely step 1.  That
prediction now rests on a measured complementarity at the wrong end of the
porosity range, which is a much stronger position than §11.7 item 3 had.

Still far below the CNN: `cnn_densenet121` is .9219 on k_off.  Nothing here
touches that.

#### A loose end closed

`PH_REDO.log` crashed on 2026-09-17 at 11:29 with
`FileNotFoundError: DESC/aniso_ph_bw125/baselines.csv`; the symlink was created
but the screen was never re-run, so §11.11's note that "no matched re-imaging
of DATA/aniso was run" stood.  That screen is the fixed-wedge-PH column above.

On disk: `scripts/filtration_throat2d.py`, `run_throat.sh`,
`DESC/aniso_throat/` (dir, 4 directions), `DESC/aniso_throat_iso/` (iso),
`RESULTS/throat/k12__{dir,iso}/`, `THROAT_DESC.log`, `THROAT_TRAIN.log`.
`ph_screen.py` gained `--prefix`; `compare_runs.py` now reads every experiment
in a `metrics.json` instead of only the first, which had silently compared the
first group of a multi-group sweep and ignored the rest.
