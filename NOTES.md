# flow-anizotropy — project notes

Predicting the **2D permeability anisotropy tensor** of porous images from
**direction-aware topological descriptors**.

Status as of 2026-09-06: solver and dataset generator are built, tested and
validated on a 60-sample pilot. The TDA port and the models are not started.

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
| 1 | **Don't reuse the K0/K2 generators** | `gen_k0_structures.py` draws continuous \|k\|, so its fields are **not periodic**, and K is only defined on a periodic cell. And an isotropic ensemble puts k_xy in the noise (§3.3). |
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

**Outputs**: `structures/*.gif` (for the solver), `npy/*.npy` (uint8, 1 = solid,
for the TDA pipeline), `structures.csv`, `filelist.txt`, `gen.log`.

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

`.npy` and `.gif` verified to encode byte-identical structures, and the recorded
porosity equals `1 − mean(npy)`.

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

### 5.3 Cost

**4.6 core-min per sample.** On 14 cores: ~5.5 h for 1000 samples, ~27 h for
5000. (An earlier estimate of 6 h for 5000 came from three hand-made structures
and understated the tail.) Cheapest levers if that is too long: reduce
`--avg-steps` (20k × 2 runs ≈ 15% of the cost), or raise `--porosity_min` to shed
the slow low-permeability tail.

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

### Step 5 — port the TDA descriptors to 2D

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
- `train_catboost_directional.py` is essentially target-agnostic — only
  `TARGET_GROUPS` needs new entries.

### Steps 6–7 — models

CatBoost on the descriptors; CNN as a stretch goal. Report R² on all targets;
the interesting ones are k_off and the orientation. Use `koff_rel_err` to state
the R² ceiling imposed by label noise.

### Open items

- Run the rotation-covariance test (§2, change 5) once a model exists.
- Decide the final sample count after looking at the `koff_over_trace` histogram
  on a 1000-sample run.
- Consider whether the `iso` stratum deserves a larger share — it is the
  finite-size-anisotropy arm and the hardest, most novel target, but also the
  noisiest.

---

## 8. Quick start

```bash
cd ~/flow/flow-anizotropy
conda activate madrid

# build the solver
(cd LMB2d && bash make_perm.sh)

# sanity-check the generator with no solver involved
python scripts/gen_aniso_structures.py --self-test

# a pilot
python scripts/gen_aniso_structures.py --output DATA/pilot --n_samples 60
python scripts/solve_permeability.py --dataset DATA/pilot --workers 14

# one structure, verbose, with the tensor test
LMB2d/lbm2d-perm DATA/pilot/structures/<name>.gif /dev/null --validate 45
```
