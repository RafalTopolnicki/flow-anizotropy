"""
Affine-stretched, downscaled cubes cut from a real rock image (3D real-rock arm).

One resampling per cube does three things at once:

    source_xyz = centre + s * F^-1 @ (out_xyz - (L-1)/2)
    F = R @ diag(a, b, 1) @ R.T          a >= b >= 1   (stretch only, never compress)

* ``s``  downscale factor: one output voxel = s native voxels (Berea: 5.345 um * s)
* ``F``  stretch: a feature of size d in the rock appears as lambda*d/s voxels
* ``R``  uniform random rotation, so the stretch axes are off-grid and all six
          tensor components are non-zero

The binary image is not resampled directly: we interpolate its signed distance
field (pore > 0) trilinearly and threshold at 0, which avoids staircase edges and
keeps porosity (a linear map scales every volume by the same factor).

All geometry is in the solver frame (x, y, z); arrays are indexed [z, y, x] as in
`structure_io3d.py`, so coordinates are reversed only at the map_coordinates call.

    python scripts/stretch_rock.py --pilot           # s x stretch grid at 80^3
    python scripts/stretch_rock.py --pilot --rock Parker --scales 3 4 5 6 \
        --grows 0 4 8 12 --out DATA/rock_pilot_sst11/Parker     # s x g grid
    python scripts/stretch_rock.py --generate --n 1000 --scale 2 --grow 4 \
        --out DATA/rock_berea                         # dataset for the LBM driver

`--grow g` shaves g native voxels off every grain (threshold the SDF at -g).
Berea at s=2: g=0 gives porosity .20 and median pore half-width 1.0 voxel;
g=4 gives porosity ~.61 and ~2.2 voxels (2026-09-30 pilot).

Dataset sampling: a log-uniform in [1, a_max], b log-uniform in [1, a], R uniform
on SO(3), centre uniform over the positions whose preimage stays inside the rock.
Every sample draws from its own generator seeded (seed, sample_id), so any one
sample regenerates alone. Cubes that do not percolate on all three torus axes are
redrawn (`attempts` records how many).

NOTE: at s=2 and 80^3 the preimage needs a ball of radius ~140 native voxels, so
in 400^3 Berea all centres fall inside a ~117^3 box and every pair of cubes
overlaps. Samples differ in stretch, rotation and window, not in rock.
"""

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_aniso_structures3d import random_rotation  # noqa: E402
from structure_io3d import save_structure  # noqa: E402

BEREA = Path("DATA/rock/berea/Image/Berea.raw")
BEREA_N, BEREA_UM = 400, 5.345

# name -> (path, edge in voxels, voxel size um). Every source is uint8 with
# 0 = pore, 1 = solid. The 11 Sandstones (Neumann et al., DRP-317) come from
# scripts/fetch_sandstones11.sh, 1000^3 at 2.25 um.
SST11 = Path("DATA/rock/sandstones11")
ROCKS = {"berea": (BEREA, BEREA_N, BEREA_UM)}
for _stem in ("BanderaBrown", "BanderaGray", "Bentheimer", "Berea", "BSG", "BUG",
              "BB", "CastleGate", "Kirby", "Leopard", "Parker"):
    ROCKS[_stem] = (SST11 / f"{_stem}_2d25um_binary.raw", 1000, 2.25)
_CONN6 = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                   [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool)

_SDF = None        # worker global, inherited by fork
_SOLID = None


def load_rock(name="berea"):
    """(nz, ny, nx) uint8, 1 = solid, same convention as structure_io3d."""
    path, n, _ = ROCKS[name]
    a = np.fromfile(path, np.uint8)
    assert a.size == n ** 3, f"{path}: {a.size} voxels, expected {n}^3"
    assert a.max() <= 1, f"{path}: values up to {a.max()}, expected binary 0/1"
    return a.reshape(n, n, n)


def signed_distance(solid):
    """Distance to the interface, > 0 in pore, < 0 in solid (voxel units)."""
    pore = solid == 0
    d_in = distance_transform_edt(pore) - 0.5
    d_out = distance_transform_edt(~pore) - 0.5
    return np.where(pore, d_in, -d_out).astype(np.float32)


def stretch_matrix(a, b, R):
    return R @ np.diag([a, b, 1.0]) @ R.T


def source_coords(L, s, F, centre_xyz):
    """(3, L^3) source coordinates in array order (z, y, x) for an L^3 output."""
    g = np.arange(L) - (L - 1) / 2.0
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    out = np.stack([x.ravel(), y.ravel(), z.ravel()])          # xyz
    src = np.asarray(centre_xyz)[:, None] + s * np.linalg.inv(F) @ out
    return src[::-1]                                            # -> zyx


def percolates_torus(solid):
    """(x, y, z) spanning on the periodic torus: what the LBM sees."""
    lab, n = label(np.tile(solid == 0, (3, 3, 3)), structure=_CONN6)
    res = []
    for axis in (2, 1, 0):
        lo = set(np.take(lab, 0, axis=axis).ravel())
        hi = set(np.take(lab, -1, axis=axis).ravel())
        res.append(bool((lo & hi) - {0}))
    return tuple(res)


def mean_chord(phase, axis):
    """Mean run length of True along one array axis (runs cut at the faces)."""
    p = np.moveaxis(phase, axis, -1).reshape(-1, phase.shape[axis]).astype(np.int8)
    d = np.diff(np.pad(p, ((0, 0), (1, 1))), axis=1)
    starts, ends = np.nonzero(d == 1), np.nonzero(d == -1)
    n = len(starts[0])
    return (ends[1] - starts[1]).sum() / n if n else np.nan


def elongation_axis(field):
    """
    Principal axes of the gradient structure tensor of the resampled SDF.
    Interfaces line up with the stretch, so the gradient is weakest along it:
    returns (unit xyz vector of the smallest eigenvalue, eigenvalues ascending).
    """
    gz, gy, gx = np.gradient(field)
    G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])
    w, v = np.linalg.eigh(G @ G.T / G.shape[1])
    return v[:, 0], w


def make_cube(L, s, a, b, R, centre_xyz, sdf=None, solid_src=None, grow=0.0):
    sdf = _SDF if sdf is None else sdf
    solid_src = _SOLID if solid_src is None else solid_src
    F = stretch_matrix(a, b, R)
    coords = source_coords(L, s, F, centre_xyz)
    field = map_coordinates(sdf, coords, order=1, mode="nearest").reshape(L, L, L)
    nn = map_coordinates(solid_src, coords, order=0, mode="nearest").reshape(L, L, L)
    # grow > 0 shaves `grow` NATIVE voxels off every grain surface (SDF units are
    # native voxels), widening every pore and throat by 2*grow native voxels
    solid = (field <= -grow).astype(np.uint8)
    return solid, field + grow, float(1.0 - nn.mean())


def cube_metrics(solid, field, R, a, b):
    pore = solid == 0
    dt = distance_transform_edt(pore)[pore]
    grain = [mean_chord(~pore, ax) for ax in (2, 1, 0)]        # x, y, z
    lab, n = label(pore, structure=_CONN6)
    big = np.bincount(lab.ravel())[1:].max() / pore.sum() if n else 0.0
    e_axis, w = elongation_axis(field)
    m = dict(porosity=1.0 - solid.mean(),
             dt_p10=np.percentile(dt, 10), dt_p50=np.percentile(dt, 50),
             dt_p90=np.percentile(dt, 90),
             grains_x=solid.shape[2] / grain[0], grains_y=solid.shape[1] / grain[1],
             grains_z=solid.shape[0] / grain[2],
             largest_cluster=big, gst_ratio=w[2] / w[0])
    px, py, pz = percolates_torus(solid)
    m.update(perc_x=px, perc_y=py, perc_z=pz)
    if a > b:     # prolate: one designed axis to recover
        m["axis_err_deg"] = np.degrees(np.arccos(min(1.0, abs(e_axis @ R[:, 0]))))
    return m


def _pilot_job(job):
    solid, field, por_nn = make_cube(job["L"], job["s"], job["a"], job["b"],
                                     job["R"], job["centre"], grow=job["g"])
    m = cube_metrics(solid, field, job["R"], job["a"], job["b"])
    tag = (f"s{job['s']:.1f}_g{job['g']:g}_a{job['a']:.1f}_b{job['b']:.1f}"
           f"_r{job['rep']}")
    save_structure(job["out"] / "structures" / f"{job['id']:03d}_{tag}.raw", solid)
    row = dict(id=job["id"], rock=job["rock"], s=job["s"], g=job["g"],
               um=job["um"] * job["s"], a=job["a"],
               b=job["b"], rep=job["rep"], por_nn=por_nn, **m,
               R=" ".join(f"{v:.6f}" for v in job["R"].ravel()),
               centre=" ".join(f"{v:.1f}" for v in job["centre"]))
    return row, solid


def _init(sdf, solid):
    global _SDF, _SOLID
    _SDF, _SOLID = sdf, solid


def pilot(args):
    out = Path(args.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    solid_src = load_rock(args.rock)
    um = ROCKS[args.rock][2]
    print(f"{args.rock}: porosity {1 - solid_src.mean():.4f}; computing SDF ...",
          flush=True)
    sdf = signed_distance(solid_src)
    rng = np.random.default_rng(args.seed)
    L, n = args.size, solid_src.shape[0]
    jobs = []
    for s in args.scales:
        # preimage fits in a ball of radius s*sqrt(3)*L/2 (F^-1 only shrinks)
        r = s * np.sqrt(3) * L / 2 + 2
        assert n - 1 - 2 * r > 0, f"s={s}: preimage does not fit in {n}^3"
        for g in args.grows:
            for a, b in [(1, 1), (1.5, 1), (2, 1), (3, 1), (2, 2)]:
                for rep in range(args.reps):
                    R = random_rotation(rng)
                    centre = rng.uniform(r, n - 1 - r, 3)
                    jobs.append(dict(id=len(jobs), L=L, s=s, g=g, a=a, b=b, rep=rep,
                                     R=R, centre=centre, out=out, rock=args.rock,
                                     um=um))
    with Pool(min(args.jobs, len(jobs)), initializer=_init,
              initargs=(sdf, solid_src)) as pool:
        results = pool.map(_pilot_job, jobs)
    rows = [r for r, _ in results]
    with open(out / "pilot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["axis_err_deg"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    montage(out, jobs, [c for _, c in results], solid_src, L, args.rock)
    print(f"-> {out/'pilot.csv'}")


def montage(out, jobs, cubes, solid_src, L, rock="berea"):
    """Mid xy slice of one a=2, b=1 cube per (s, g): rows s, columns g."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [i for i, j in enumerate(jobs)
            if j["rep"] == 0 and (j["a"], j["b"]) == (2, 1)]
    scales = sorted({jobs[i]["s"] for i in pick})
    grows = sorted({jobs[i]["g"] for i in pick})
    fig, ax = plt.subplots(len(scales), len(grows), squeeze=False,
                           figsize=(2.3 * len(grows), 2.3 * len(scales)))
    for i in pick:
        j = jobs[i]
        a_ = ax[scales.index(j["s"]), grows.index(j["g"])]
        a_.imshow(cubes[i][L // 2], cmap="gray_r", origin="lower",
                  interpolation="nearest")
        a_.set_title(f"s={j['s']:g} g={j['g']:g} por={1 - cubes[i].mean():.2f}",
                     fontsize=7)
    for a_ in ax.ravel():
        a_.set_xticks([]), a_.set_yticks([])
    fig.suptitle(f"{rock} stretched cubes (a=2, b=1), {L}^3 xy, pore = white",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "slices.png", dpi=110)


def _gen_one(job):
    sid, cfg = job
    rng = np.random.default_rng([cfg["seed"], sid])
    L, s, n = cfg["size"], cfg["scale"], _SOLID.shape[0]
    r = s * np.sqrt(3) * L / 2 + 2
    for attempt in range(1, 21):
        a = float(np.exp(rng.uniform(0, np.log(cfg["a_max"]))))
        b = float(np.exp(rng.uniform(0, np.log(a))))
        R = random_rotation(rng)
        centre = rng.uniform(r, n - 1 - r, 3)
        solid, field, _ = make_cube(L, s, a, b, R, centre, grow=cfg["grow"])
        perc = percolates_torus(solid)
        if all(perc):
            break
    else:
        raise RuntimeError(f"sample {sid}: no percolating draw in 20 attempts")
    pore = solid == 0
    dt = distance_transform_edt(pore)[pore]
    por = float(pore.mean())
    stratum = "weak" if a < 1.5 else ("mod" if a < 2.2 else "strong")
    fname = (f"sample_{sid:06d}_rock={cfg['rock']}_str={stratum}_a={a:.2f}_b={b:.2f}"
             f"_por={por:.4f}.raw")
    save_structure(Path(cfg["out"]) / "structures" / fname, solid)
    row = dict(sample_id=sid, filename=fname, stratum=stratum, aniso_A=a, aniso_B=b)
    for k, name in enumerate(("e1", "e2", "e3")):     # principal stretch axes, xyz
        row.update({f"{name}_{c}": R[i, k] for i, c in enumerate("xyz")})
    row.update(porosity=por, rock=cfg["rock"], scale=s, voxel_um=cfg["voxel_um"] * s,
               grow=cfg["grow"], centre_x=centre[0], centre_y=centre[1],
               centre_z=centre[2], dt_p50=float(np.median(dt)), attempts=attempt)
    return row


def generate(args):
    import hashlib
    import json
    out = Path(args.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    solid_src = load_rock(args.rock)
    sdf = signed_distance(solid_src)
    cfg = dict(seed=args.seed, size=args.size, scale=args.scale, grow=args.grow,
               a_max=args.a_max, rock=args.rock, voxel_um=ROCKS[args.rock][2],
               out=str(out))
    with Pool(args.jobs, initializer=_init, initargs=(sdf, solid_src)) as pool:
        rows = pool.map(_gen_one, [(args.id_offset + i, cfg) for i in range(args.n)])
    pd_rows = sorted(rows, key=lambda r: r["sample_id"])
    with open(out / "structures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pd_rows[0].keys()))
        w.writeheader()
        w.writerows(pd_rows)
    (out / "filelist.txt").write_text("".join(r["filename"] + "\n" for r in pd_rows))
    src = ROCKS[args.rock][0]
    cfg["source"] = str(src)
    cfg["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    cfg["n"] = args.n
    cfg["id_offset"] = args.id_offset
    (out / "manifest.json").write_text(json.dumps(cfg, indent=2))
    print(f"-> {out/'structures.csv'} ({len(pd_rows)} rows)")


def merge(args):
    """
    Join per-rock parts (each from --generate --id-offset, in <out>/parts/<rock>)
    into one dataset at <out>: structures/ (files moved), structures.csv,
    filelist.txt, manifest.json listing every part's manifest. Ids must not clash.
    """
    import json
    out = Path(args.out)
    parts = sorted(p for p in (out / "parts").iterdir() if (p / "structures.csv").exists())
    (out / "structures").mkdir(exist_ok=True)
    rows, mans = [], []
    for p in parts:
        with open(p / "structures.csv") as fh:
            rows += list(csv.DictReader(fh))
        mans.append(json.loads((p / "manifest.json").read_text()))
        for f in (p / "structures").iterdir():
            f.rename(out / "structures" / f.name)
    ids = [int(r["sample_id"]) for r in rows]
    assert len(set(ids)) == len(ids), "sample_id clash between parts"
    rows.sort(key=lambda r: int(r["sample_id"]))
    missing = [r["filename"] for r in rows if not (out / "structures" / r["filename"]).exists()]
    assert not missing, f"{len(missing)} structure files missing, e.g. {missing[0]}"
    with open(out / "structures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "filelist.txt").write_text("".join(r["filename"] + "\n" for r in rows))
    (out / "manifest.json").write_text(json.dumps(dict(n=len(rows), parts=mans), indent=2))
    print(f"-> {out/'structures.csv'} ({len(rows)} rows from {len(parts)} parts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--grow", type=float, default=0.0)
    ap.add_argument("--a-max", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--grows", type=float, nargs="+", default=[0.0],
                    help="pilot only: grain shaves g (native voxels) to grid over")
    ap.add_argument("--rock", default="berea", choices=sorted(ROCKS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--id-offset", type=int, default=0,
                    help="generate: first sample_id (keeps ids unique across rocks)")
    ap.add_argument("--merge", action="store_true",
                    help="join <out>/parts/<rock> into one dataset at <out>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=45)
    ap.add_argument("--out", default="DATA/rock_pilot")
    args = ap.parse_args()
    if args.pilot:
        pilot(args)
    elif args.generate:
        generate(args)
    elif args.merge:
        merge(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
