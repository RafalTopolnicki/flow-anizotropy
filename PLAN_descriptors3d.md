# Plan — baseline and topological descriptors for `DATA/aniso3d`

Status: proposed, 2026-09-10. Fold into `NOTES.md` §8.7 once executed.

Scope: descriptors only. No models, no labels. The LBM run is a separate,
already-running job on another machine.

---

## 0. State of the data

```
DATA/aniso3d/
  structures.csv     5000 rows, sample_id 0..4999
  structures/        5000 .raw files, 80^3, 2.4 GB, triply periodic
  perm/              723 CSVs so far  (ids span 000000..000731 with gaps)
  filelist.txt
```

**Descriptors do not need labels.** Nothing in the filtration, ECP, PH or the
baselines reads `perm/`. So the whole descriptor run can go now, over all 5000
structures, in parallel with the LBM finishing on the other machine. The join
happens once, at training time, on `sample_id`.

The 723 finished samples are **not a contiguous prefix** — there are gaps at
719, 721, 723, 725 where workers are still in flight. Anything that consumes
labels must join on `sample_id`, never slice the first N rows.
`solve_permeability3d.py --merge-only` already does the right thing and can be
rerun any time to refresh `DATA/aniso3d/permeability.csv` from whatever has
landed.

---

## 1. What has to be written

| file | status | work |
|---|---|---|
| `scripts/structure_io3d.py` | **exists** | — |
| `scripts/solve_permeability3d.py` | **exists** | — (`--merge-only` for labels) |
| `scripts/filtration3d.py` | **written**, 18/18 self-test | the real work; port of `filtration2d.py`, *not* of the elasticity `pipeline.py` |
| `scripts/ecp3d.jl` | **written** | ~5 lines changed from `ecp2d.jl`: `compute_contributions_3d`, `Tuple(g for 1:n_params)` |
| `scripts/ph3d.py` | **written** | `ph2d.py` with `DIMS=(0,1,2)` and `periodic_dimensions=[True,True,True]` |
| `scripts/baselines3d.py` | **written**, 13/13 self-test | port of `baselines2d.py` + the fabric *profile* the elasticity repo has and 2D does not |
| `scripts/collect_descriptors.py` | **patched** (`_tag_key`) | its tag sort key is `float(tag.lstrip("a"))`, which throws on `a1_-1_0`; needs a string fallback |
| `run_descriptors3d.sh` | **written** | see §8 |

Everything else — `collect_descriptors.py`'s body, the `DESC/` layout, the
caching-by-existing-output behaviour that makes reruns idempotent — carries over
from 2D unchanged.

Small argparse constraint, verified: `--direction 1 -1 0` parses correctly with
`nargs=3, type=float`, because argparse treats `-1` as a value when no option
string of that parser looks like a negative number. Keep it that way — adding a
short option like `-1` anywhere in `filtration3d.py` would silently turn the
negative directions into unrecognised flags.

### The `collect_descriptors.py` tweak, precisely

```python
def _tag_key(t):
    try:
        return (0, float(t.lstrip("a")), "")   # 2D: a0, a45, a90, a135
    except ValueError:
        return (1, 0.0, t)                     # 3D: a1_0_0, a1_-1_0, ...
```

then `sorted(os.listdir(kind_dir), key=_tag_key)`. Column names become
`ecp_a1_-1_0_0 … ecp_a1_-1_0_342`. Nothing else in that script is 2D-specific:
it keys on `sample_id` parsed from the leading `sample_%06d`, and it already
errors loudly if a direction is missing structures.

---

## 2. Design decisions

Starting point is the elasticity pipeline as documented in
`~/direction-aware-tda-for-porous-materials/METHODS.md`: cone+PCA filtration
(`cone_radius=3.0`, `cone_height=6.0`, PCA sphere `radius=4`), 3-parameter ECP
at grid-res 6 → 343/direction, PH on the cone channel → 300/direction, and the
three directional baselines (porosity profile, two-point correlation, fabric
profile).

Five departures. The first three are forced by our geometry and are not
optional; the last two are choices.

### 2.1 No grid rotation — build the cone along `d` directly

`pipeline.py: rotate_grid_to_z` transposes for the three axes and falls back to
`scipy.ndimage.affine_transform(order=0, cval=0.0)` for every diagonal. Two
problems, and the second is fatal here:

1. It is lossy. Nearest-neighbour resampling of a binary grid, re-thresholded at
   0.5, changes the voxels the descriptor then measures. Diagonals become
   second-class directions — and our diagonals are the ones that carry the
   targets (§3).
2. **`cval=0.0` fills the rotated-in corners with pore.** Our cells are
   *triply* periodic — `gen_aniso_structures3d.py` verified it on the pilot at
   `mean(wrap − adjacent agreement) = −0.00006` (NOTES §8.2) — and that torus is
   what the LBM actually solves. Rotating with a constant fill destroys exactly
   the property the label depends on.

Instead do what `filtration2d.py` does: build the cone offsets directly at the
requested `d`, and index the grid modulo its shape. Every direction is then
exact, all directions are treated identically, and periodicity is preserved by
construction.

Carry over the `EPS = 1e-9` boundary tolerance from 2D. That is not cosmetic:
`cos(90°)` is 6.1e-17, not 0, and without the tolerance offsets sitting exactly
on the cone boundary flip in or out, so θ=90° got 48 stencil offsets where θ=0°
got 42 and the stencil lost central symmetry (NOTES §5.5 — this was a real bug,
caught by the rotation test). The 3D form of the same trap is any `d` with a
zero component, i.e. six of our nine directions.

### 2.2 Frame-free PCA channels

`dir_y = 1 − |v_y|` and `dir_z = 1 − |v_z|` only mean something *after* the grid
has been rotated so `d → z`. With no rotation there is no canonical transverse
axis, so those two channels are not defined. Replacement, both frame-free:

```
ch0  cone      = 1 − (solid inside double cone along ±d) / (cone volume)
ch1  dir_long  = 1 − |v1 · d|        v1 = top PCA eigenvector
ch2  dir_norm  = 1 − |v3 · d|        v3 = smallest PCA eigenvector
```

`v3` is not redundant with `v1`: in 3D, `|v1·d|² + |v2·d|² + |v3·d|² = 1` is one
constraint on three numbers, so two of them are independent. (In 2D the same
identity has only two terms, which is precisely why `filtration2d.py` dropped
the second channel.)

And `v3` is the physically right second channel *for this generator*. The
squash is `s = (AB)^(1/3)·(1/A, 1/B, 1)` in a random SO(3) frame, and `shape =
(B−1)/(A−1)` interpolates rod (B≈1) → plate (B≈A). A **plate is defined by its
normal, not by a long axis** — its `v1` is degenerate, which is exactly what the
pilot measured: long-axis recovery 23.7° median for plates against 3.9° for the
short axis (NOTES §8.2). A descriptor carrying only `v1` is blind to half the
generator's shape parameter.

Three channels keeps ECP 3-parameter, so the elasticity feature counts transfer
unchanged.

### 2.3 PH periodic in all three axes

`ph_calc.py` passes `periodic_dimensions=[False, False, True]` because only `z`
wrapped in that dataset. Ours wrap in all three: `[True, True, True]`. Same
change 2D made for the same reason.

Measured on a real 80³ structure: 4.05 s, 420 MB peak RSS, ~27k finite bars.
Fully-periodic 3D cubical persistence is the single most expensive step in the
pipeline (§5) — but it is affordable, so there is no reason to compromise on it.

### 2.4 ECP grid-res 6 → 343 per direction

`METHODS.md` documents grid-res 6; `run_directional_tda_desc.sh` uses 8. At 3
parameters that is 343 vs 729 features per direction, i.e. 3087 vs 6561 across
nine directions. Start at **6**, matching METHODS. `--grid-res 8` stays a flag;
the elasticity repo has a full scan over {4, 6, 8, 10} to compare against if the
question comes up.

Same for PH: **10×10 over [0, 1.25]²** per METHODS (300/direction), not the
script's 12×12 (432/direction).

Note `METHODS.md` §3.6 miscounts its own total — "1029 + 2100 = 3129 features
for 7 directions", where 1029 is 343×**3**. Seven directions of ECP is 2401.
Do not copy that number into anything.

### 2.5 Keep PH, as its own feature group

In 2D, PH was dead weight: `ecp` (324 features) matched or beat `tda` (=ecp+ph,
1124) at essentially every sample size and target, and the 800 persistence-image
features diluted ECP (NOTES §10.2). NOTES §10.3 registers the prediction that
this *reverses* in 3D — H₂ exists, H₁ becomes throat-and-channel linkage rather
than pore outlines — and that prediction is only testable if PH is computed.

So compute it, and keep `ecp` / `ph` / `tda` as three separate feature groups in
`run_experiments.sh`, exactly as 2D did, so the comparison is a rerun rather
than a new experiment.

---

## 3. Direction set — 9, not 7

The elasticity set is 3 axes + 3 face diagonals + 1 body diagonal, with `[1,1,0]`
but **not** `[1,-1,0]`.

That is fine for stiffness and wrong for us. Our headline targets are `k_xy`,
`k_xz`, `k_yz`, and `k_xy` *is* the contrast between flow along `[1,1,0]` and
along `[1,-1,0]` — a descriptor set containing only one of the pair cannot
express the sign of an off-diagonal at all. The 2D arm already made this choice
for the same reason: it used 0/45/90/135, i.e. both diagonals, not 0/45/90.

The 3D analogue of 0/45/90/135 is **3 axes + 6 signed face diagonals**:

```
[1,0,0]  [0,1,0]  [0,0,1]
[1,1,0]  [1,-1,0]        →  k_xy
[1,0,1]  [1,0,-1]        →  k_xz
[0,1,1]  [0,1,-1]        →  k_yz
```

No antipodal duplicates: the cone is a *double* cone, so `d` and `−d` give an
identical filtration, and the porosity/fabric profiles along `d` and `−d` differ
only by reversal.

The 4 body diagonals `[1,1,1] [1,1,-1] [1,-1,1] [-1,1,1]` are a clean optional
extension — +44% cost, +2572 features. Not in the default set; add them only if
the 9-direction result asks for it.

---

## 4. Feature budget

| group | per direction | × 9 | 2D equivalent |
|---|---|---|---|
| `ecp` | 343 (7³, grid-res 6) | 3087 | 324 |
| `ph` | 300 (3 dims × 10×10) | 2700 | 800 |
| **`tda`** | | **5787** | 1124 |
| `por` | 80 | 720 | 512 |
| `tpc` | 41 | 369 | 260 |
| `fabdir` | 80 | 720 | — (2D has no fabric profile) |
| `fab` | 10, global | 10 | 7 |
| **`baselines`** | | **1819** | 779 |

`fab` in 3D is **6 tensor components + 3 eigenvalues + fractional anisotropy**.
Not eigenvector directions, and not the 2D `(cos 2t, sin 2t)` trick: in 3D
eigenvectors are sign-ambiguous, their order swaps when eigenvalues cross, and
two of them are genuinely degenerate for a rod or a plate (NOTES §8.5). The
tensor components are unambiguous and covariant; the axes are not.

Adding `fabdir` is deliberate. NOTES §10.3 trap 1: the 2D result is only
credible because the baselines were strong enough to *beat* TDA, and if the 3D
baselines get less care than the 3D TDA then a TDA win measures effort rather
than information. The elasticity repo has a directional fabric profile; ours
should too.

---

## 5. Cost — measured today, on this machine

24 cores, 235 GB RAM. All numbers from real 80³ structures out of
`DATA/aniso3d/structures/`, not estimates.

> **Disk is a quota, not free space.** `/home/rtopolnicki` carries a **150 GiB
> ceph quota**, and `df` does not show it — `df` reports the pool's 18 TB, which
> is irrelevant. The first attempt at this run was killed partway through
> direction 1 because this plan sized the run against `df`. Check it with:
>
> ```
> getfattr -n ceph.quota.max_bytes ~     # 161061273600 = 150 GiB
> getfattr -n ceph.dir.rbytes      ~     # current usage
> ```
>
> Home sits at ~114 GiB, so the real headroom is **~36 GiB** — against 31 GB for
> one direction's filtrations. Hence `run_descriptors3d.sh` walks each direction
> in chunks of `CHUNK` (default 1000), holding ~6 GB at a time and costing ~15
> min of extra process startup across the run.

| step | per structure per direction | × 5000 × 9 |
|---|---|---|
| filtration (numba, 3 channels) | 0.084 s wall on 24 threads ≈ **2.0 core-s** | 25 core-h |
| ECP (Julia, 3-param, grid-res 6) | **1.4 core-s** (after JIT warm-up) | 17.5 core-h |
| PH (gudhi, periodic ×3, H₀H₁H₂) | **4.1 core-s**, 420 MB peak RSS | 51 core-h |
| **total** | **≈ 7.5 core-s** | **≈ 94 core-h ≈ 4 h wall on 24 cores** |

Baselines: minutes, not hours — all of it is `np.roll`, `np.bincount` and one
gradient per structure.

- **Disk**: a filtration is 6.1 MB (80³ × 3 × float32) → 31 GB per direction, so
  it is walked in chunks and deleted after ECP and PH consume each one. **Peak
  is CHUNK × 6.1 MB ≈ 6 GB**, not 31 GB — see the quota box above. Persistent
  output is ~710 MB (129 MB ecp + 114 MB ph + 304 MB descriptors.csv + 163 MB
  baselines.csv). At 2 core-s a filtration is cheaper to recompute than to keep;
  keeping all nine would be 275 GB and is impossible under the quota.
- **RAM**: PH is the driver at 420 MB/worker; 24 workers ≈ 10 GB of 235 GB.
- The elasticity repo's "48 s per structure" (NOTES §5.5) is **not** our number —
  that figure includes the `affine_transform` rotation and a serial per-file
  Python loop, both of which §2.1 removes.

Everything is idempotent: `filtration3d.py`, `ph3d.py` and `ecp3d.jl` all skip a
sample whose output already exists, so an interrupted run is resumed by
rerunning the same command.

### 5.1 Pilot — run 2026-09-10, 20 structures x 3 directions

Steady-state, after numba/Julia warm-up, on 24 cores:

| stage | 20 structures | fixed overhead | extrapolated, 5000 x 9 |
|---|---|---|---|
| filtration | 1.7 s | ~1.4 s process start | 62 min |
| ECP | ~1.2 s | **10.7 s** Julia startup + JIT | 46 min |
| PH | ~5 s | ~4 s first-worker warm-up | 128 min |
| | | | **~4 h**, as estimated |

Outputs verified: filtration `(80, 80, 80, 3)` float32 at 5.9 MB (29.5 GB per
direction), ECP `(343,)`, PH `(300,)`, no NaNs anywhere.

Julia's 10.7 s startup is per *invocation*, not per structure, so it is 1.6 min
across the whole run — irrelevant at 5000 but worth knowing if you ever shard a
direction into chunks, where it would be paid once per chunk.

### 5.2 Smoke test on the pilot — the signed pair carries anisotropy

Relative ECP difference between `[1,1,0]` and `[1,-1,0]`, per stratum:

| stratum | n | median rel. diff |
|---|---|---|
| iso | 7 | 0.357 |
| mod | 7 | 0.618 |
| strong | 6 | **0.806** |

Monotone in the generator's anisotropy, before any label exists. The separation
between the two members of a signed pair is the quantity `k_xy` depends on, and
it is present and ordered correctly. (`iso` is not 0 because an isotropic
stratum still has a random realisation with local structure.)

---

## 6. Validation gates — before the production run

This is where the project's time has actually paid off twice (the 2D wedge bug,
§5.5; the 3D solver axis convention, §8.6a), so it is not optional.

**`filtration3d.py --self-test`**

- cone stencil is centrally symmetric (it is a *double* cone) for axis-aligned,
  face-diagonal and negative-component `d`
- stencil size is invariant under axis permutation and sign flip of `d` — the
  `EPS` trap of §2.1, which is what broke 2D
- all-solid → `cone = 0`; void voxel → the `1.25` sentinel in all 3 channels
- direction sensitivity: on solid rods along x, `cone` is low at `d=[1,0,0]` and
  high at `d=[0,1,0]`; on plates, the reverse
- PCA channels: `dir_long ≈ 0` along a rod's axis; `dir_norm ≈ 0` along a
  plate's normal — i.e. the two channels are checked on the two shapes the
  generator actually produces
- **exact axis-permutation equivariance**: permute the grid axes and permute `d`
  the same way ⇒ the filtration permutes, to `0.00e+00`. This is the 3D analogue
  of the 2D `rot90` test and uses the same logic as §8.6a. It is the single test
  most likely to catch a real bug, because a `[z,y,x]` vs `[x,y,z]` slip in the
  offset construction passes every other check on this list.

**`baselines3d.py --self-test`**

- porosity profile flat along a slab's own plane, structured across it
- `S₂(0)` equals the solid fraction
- diagonal profile wraps exactly on the torus: `[1,-1,0]` over an 80³ cell gives
  80 genuine slabs and its mean equals the global solid fraction
- fabric tensor of x-normal plates has normals along x; permuting the grid axes
  permutes the fabric tensor the same way
- fabric profile of an isotropic random structure sits near 1/3

**Pilot before the full run.** `LIMIT=20`, one direction, end to end, and check
the wall clock against §5 before launching all nine. NOTES §8.6 is explicit
about why: *"The 2D estimate was wrong by 4.5x (6 h → 27 h) when based on three
hand-made structures — do not skip this."* The §5 numbers here are per-step
measurements on real structures, which is better than that was, but they still
do not include process startup, npy I/O at 31 GB per direction, or Julia's JIT.

**Smoke test, before any label exists** — the §5.6 analogue. Predict the
generator's hidden parameters from the descriptors alone on all 5000: `A`,
`shape`, and the components of the structure tensor `Σ = R diag(A⁻², B⁻², 1) Rᵀ`
(frame-unambiguous, unlike the eigenvectors). In 2D this is what revealed that 7
fabric features beat all 1124 TDA features at recovering orientation — worth
knowing again here, and it needs no LBM.

---

## 7. Order of work

1. `ecp3d.jl`, `ph3d.py` — trivial ports, ~30 min
2. `filtration3d.py` + self-test — the real work
3. `baselines3d.py` + self-test
4. the `collect_descriptors.py` sort-key tweak
5. pilot: `LIMIT=20 DIRECTIONS_OVERRIDE="1 1 0" bash run_descriptors3d.sh`
6. full run: `bash run_descriptors3d.sh` (~4 h)
7. baselines (minutes) — the same script does it at the end
8. smoke test
9. when labels land: `python scripts/solve_permeability3d.py --dataset DATA/aniso3d --merge-only`

Steps 1–8 are independent of the LBM and can all run while it finishes.

---

## 8. The script

`run_descriptors3d.sh` — flat, one command after another, same shape as
`run_descriptors.sh`. Per direction: filtration → ECP → PH → drop the
filtration. Then collect, then baselines.
