# flow-anizotropy — project notes

Predicting the **2D permeability anisotropy tensor** of porous images from
**direction-aware topological descriptors**.

Status as of 2026-09-06:

| stage | state |
|---|---|
| LBM tensor solver | built, validated (§5.1) |
| dataset generator | built, validated (§5.2) |
| **5000-structure dataset** | **generated** — `DATA/aniso/`, 5000/5000 kept |
| **LBM labels** | **running on a remote machine** (~27 h on 14 cores) |
| 2D TDA descriptors | built, validated, **computed for all 5000** (§4.4, §5.5) |
| directional baselines | built, validated, **computed for all 5000** |
| CatBoost training | script ready, mechanics validated; waits on labels |
| CNN baseline | built, validated on GPU; waits on labels |
| training driver | `run_training.sh` — all models × all targets, one table |

The full working sessions that produced all of this — including the measurements
behind each design decision and the bugs found along the way — are under
`conversations/`, one `.md` (readable) and `.jsonl` (unabridged) per session:

| file | date | covers |
|---|---|---|
| `conversations/session01_2026-09-06.*` | 2026-09-06 | solver, generator, descriptors, CNN, training scripts |
| `conversations/session02_2026-09-07.*` | 2026-09-07 | velocity/VTK tools, the 3D pipeline, 2D labels arriving |

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

Two small tools, for looking at a single structure rather than for the dataset run.

```bash
python scripts/solve_velocity.py <structure.gif> -o VIZ/s0 --vtk
python scripts/to_vtk.py VIZ/s0.fx.npy out.vtk --solid VIZ/s0.solid.npy
python scripts/to_vtk.py --self-test        # 14 checks
```

`solve_velocity.py` runs the same two solves the dataset run does and keeps the
fields: `s0.fx.npy` / `s0.fy.npy` as `(ny, nx, 2)` float32, `s0.solid.npy`, the
usual tensor row as `s0.csv`, and with `--vtk` a ParaView file per direction
carrying `velocity`, `velocity_magnitude` and `solid` in one dataset.

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
carries over to the 3D work unchanged.

Note the exported velocity is the raw moment `Σf·e/ρ` without the `+F/(2ρ)`
forcing correction, matching the flux convention used everywhere else (§6).

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
The convergence machinery of 4.1 is ported unchanged in substance: block means
not instantaneous samples, every component against its own magnitude, tolerance
= max(relative bound, k-sigma of the block mean's own standard error), `--hold`
consecutive passes, then a dedicated averaging phase.  **K = K^T now gives three
free error bars instead of one, one per off-diagonal target.**

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
error, not a coding error.  A blocked direction now returns 0 to 1e-12 and the
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

### 8.6 Still to do

- **Rotation covariance is the one structural check still not run.**  An axis
  permutation is exact and lossless and K must transform as P K P^T.  Reciprocity
  does not cover it: an index bug that is symmetric under transpose survives
  K = K^T but not a permutation.  Until it passes, treat the three-solve assembly
  as unverified.  (The tensor test is done — see 8.4.)
- Pilot ~60 structures for the real per-sample cost, the distribution of
  convergence steps and `se_*` (which sets `--avg-steps`, see 8.4), and the
  `koff_over_trace` histogram, exactly as 5.2/5.3 did for 2D.  **The 2D estimate was wrong by 4.5x
  (6 h -> 27 h) when based on three hand-made structures — do not skip this.**
- Decide `--size` and the sample count from that pilot.
- Then: 3D TDA descriptors and baselines (port from the elasticity repo, which
  is 3D already — this is the piece that gets *easier* going to 3D), and the CNN.

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

# 1. structures  (~6 min for 5000 at 80^3)
python scripts/gen_aniso_structures3d.py --output DATA/aniso3d --n_samples 5000

# 2. labels — restartable, so safe to interrupt.  PILOT FIRST (8.6):
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --limit 60 --workers $(nproc)
python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers $(nproc)

# single structure, with the Koza09 tensor test along (1,1,1)
LMB3d/lbm3d-perm <structure.raw> /dev/null --validate
```
