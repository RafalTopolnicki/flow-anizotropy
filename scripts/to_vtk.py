"""
Convert a velocity or scalar field stored as .npy into a legacy VTK file that
ParaView opens directly.

    python scripts/to_vtk.py field.npy field.vtk
    python scripts/to_vtk.py field.npy field.vtk --solid solid.npy
    python scripts/to_vtk.py --self-test

Array convention
----------------
Axes run **slowest to fastest**, i.e. ``(y, x)`` in 2D and ``(z, y, x)`` in 3D,
with an optional trailing component axis:

    (ny, nx)          scalar on a 2D grid
    (ny, nx, 2 or 3)  vector on a 2D grid
    (nz, ny, nx)      scalar on a 3D grid
    (nz, ny, nx, 3)   vector on a 3D grid

That is the same orientation `structure_io.load_structure` returns
(``[solver_y, solver_x]``) and the same one `solve_velocity.py` writes, so a
field and its structure mask line up without transposing either.

VTK STRUCTURED_POINTS stores points with **x varying fastest**, then y, then z —
which is exactly C order for an array indexed ``(z, y, x)``, so `reshape(-1)` is
already the right sequence. This is the only place that fact is relied on; get
it wrong and the field comes out transposed, which for an anisotropic structure
looks plausible and is not.

Binary is the default. Legacy VTK binary is **big-endian** regardless of host.
ASCII (`--ascii`) is readable but ~15x larger: fine for a 256^2 slice, ruinous
for a 256^3 volume (~1 GB), which is why binary is the default now rather than
something to fix later.
"""

import argparse
import sys

import numpy as np

__all__ = ["write_vtk"]


def _grid_shape(arr, kind):
    """Split an array shape into (grid dims, n components)."""
    if kind == "vector":
        if arr.ndim not in (3, 4) or arr.shape[-1] not in (2, 3):
            raise ValueError(
                f"vector field must be (ny,nx,2|3) or (nz,ny,nx,3), got {arr.shape}")
        return arr.shape[:-1], arr.shape[-1]
    if arr.ndim not in (2, 3):
        raise ValueError(f"scalar field must be (ny,nx) or (nz,ny,nx), got {arr.shape}")
    return arr.shape, 1


def _guess_kind(arr):
    """Vector iff there is a trailing axis of length 2 or 3 above a 2D grid."""
    if arr.ndim in (3, 4) and arr.shape[-1] in (2, 3):
        return "vector"
    return "scalar"


def _dims(grid):
    """VTK DIMENSIONS is (nx, ny, nz); our grid is (ny, nx) or (nz, ny, nx)."""
    if len(grid) == 2:
        ny, nx = grid
        return nx, ny, 1
    nz, ny, nx = grid
    return nx, ny, nz


def _as_vtk_vectors(arr, ncomp):
    """Flatten to (npoints, 3), padding a 2-component field with vz = 0."""
    flat = arr.reshape(-1, ncomp)
    if ncomp == 3:
        return flat
    out = np.zeros((flat.shape[0], 3), dtype=flat.dtype)
    out[:, :2] = flat
    return out


def write_vtk(path, field, name="velocity", kind=None, solid=None,
              magnitude=True, ascii_mode=False, title="flow-anizotropy"):
    """
    Write `field` to `path` as a legacy VTK STRUCTURED_POINTS dataset.

    solid      optional mask on the same grid, added as a scalar array so the
               geometry can be shown alongside the flow (a Threshold in ParaView).
    magnitude  for a vector field, also write |v| as a scalar — saves reaching
               for a Calculator filter just to colour by speed.
    """
    field = np.asarray(field)
    kind = kind or _guess_kind(field)
    grid, ncomp = _grid_shape(field, kind)
    nx, ny, nz = _dims(grid)
    npoints = nx * ny * nz

    arrays = []          # (declaration lines, flat float32/uint8 payload)
    if kind == "vector":
        vec = _as_vtk_vectors(field, ncomp)
        arrays.append(([f"VECTORS {name} float"], vec.astype(np.float32)))
        if magnitude:
            mag = np.linalg.norm(field.reshape(-1, ncomp), axis=1)
            arrays.append(([f"SCALARS {name}_magnitude float 1", "LOOKUP_TABLE default"],
                           mag.astype(np.float32)))
    else:
        arrays.append(([f"SCALARS {name} float 1", "LOOKUP_TABLE default"],
                       field.reshape(-1).astype(np.float32)))

    if solid is not None:
        solid = np.asarray(solid)
        if tuple(solid.shape) != tuple(grid):
            raise ValueError(f"solid shape {solid.shape} does not match grid {grid}")
        arrays.append((["SCALARS solid unsigned_char 1", "LOOKUP_TABLE default"],
                       solid.reshape(-1).astype(np.uint8)))

    header = (
        "# vtk DataFile Version 3.0\n"
        f"{title}\n"
        f"{'ASCII' if ascii_mode else 'BINARY'}\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        "ORIGIN 0 0 0\n"
        "SPACING 1 1 1\n"
        f"POINT_DATA {npoints}\n"
    )

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for decl, payload in arrays:
            fh.write(("\n".join(decl) + "\n").encode("ascii"))
            if ascii_mode:
                rows = payload.reshape(payload.shape[0], -1)
                fmt = "%d" if payload.dtype == np.uint8 else "%.7g"
                np.savetxt(fh, rows, fmt=fmt)
            else:
                # legacy VTK binary is big-endian, always
                big = payload.astype(payload.dtype.newbyteorder(">"))
                fh.write(big.tobytes())
                fh.write(b"\n")
    return path


# ---------------------------------------------------------------------------


def _self_test():
    """
    The checks that matter are about ordering, not about VTK syntax: a file that
    parses but is transposed is the failure mode this tool can actually have.
    """
    import struct
    import tempfile
    from pathlib import Path

    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}")

    tmp = Path(tempfile.mkdtemp())

    # 1. a field that is a known function of (x, y) must come back with x fastest
    ny, nx = 3, 5
    yy, xx = np.mgrid[0:ny, 0:nx]
    f = np.stack([xx, yy], axis=-1).astype(np.float64)     # v = (x, y)
    write_vtk(tmp / "v.vtk", f, ascii_mode=True, magnitude=False)
    text = (tmp / "v.vtk").read_text().splitlines()
    i = text.index("VECTORS velocity float")
    body = np.array([[float(t) for t in ln.split()] for ln in text[i + 1: i + 1 + nx * ny]])
    check("DIMENSIONS is (nx, ny, 1)", "DIMENSIONS 5 3 1" in text)
    check("first row is x=0..nx-1 at y=0 (x varies fastest)",
          np.array_equal(body[:nx, 0], np.arange(nx)) and np.all(body[:nx, 1] == 0))
    check("second row is y=1", np.all(body[nx:2 * nx, 1] == 1))
    check("2-component input padded with vz=0", np.all(body[:, 2] == 0))

    # 2. binary round-trip, big-endian, correct byte count
    write_vtk(tmp / "v.bin.vtk", f, magnitude=False)
    raw = (tmp / "v.bin.vtk").read_bytes()
    payload = raw.split(b"VECTORS velocity float\n", 1)[1][:-1]
    check("binary payload is npoints*3 float32", len(payload) == nx * ny * 3 * 4)
    vals = struct.unpack(f">{nx*ny*3}f", payload)
    check("binary matches ascii", np.allclose(np.array(vals).reshape(-1, 3), body))

    # 3. a 3D field keeps (z, y, x) ordering
    nz3, ny3, nx3 = 2, 3, 4
    zz, yy3, xx3 = np.mgrid[0:nz3, 0:ny3, 0:nx3]
    g = np.stack([xx3, yy3, zz], axis=-1).astype(np.float64)
    write_vtk(tmp / "g.vtk", g, ascii_mode=True, magnitude=False)
    t3 = (tmp / "g.vtk").read_text().splitlines()
    j = t3.index("VECTORS velocity float")
    b3 = np.array([[float(t) for t in ln.split()] for ln in t3[j + 1: j + 1 + nx3 * ny3 * nz3]])
    check("3D DIMENSIONS is (nx, ny, nz)", "DIMENSIONS 4 3 2" in t3)
    check("3D order is x fastest then y then z",
          np.array_equal(b3, g.reshape(-1, 3)))

    # 4. scalar + solid companion array
    solid = (xx + yy) % 2
    write_vtk(tmp / "s.vtk", f, solid=solid, ascii_mode=True)
    st = (tmp / "s.vtk").read_text()
    check("magnitude scalar written", "SCALARS velocity_magnitude float 1" in st)
    check("solid scalar written", "SCALARS solid unsigned_char 1" in st)
    check("one POINT_DATA declaration only", st.count("POINT_DATA") == 1)

    # 5. shape guards
    try:
        write_vtk(tmp / "bad.vtk", f, solid=np.zeros((2, 2)), ascii_mode=True)
        check("mismatched solid rejected", False)
    except ValueError:
        check("mismatched solid rejected", True)

    check("scalar (ny,nx) auto-detected", _guess_kind(np.zeros((4, 4))) == "scalar")
    check("vector (ny,nx,2) auto-detected", _guess_kind(np.zeros((4, 4, 2))) == "vector")

    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="input .npy")
    p.add_argument("output", nargs="?", help="output .vtk")
    p.add_argument("--name", default="velocity", help="array name in the VTK file")
    p.add_argument("--kind", choices=["vector", "scalar"],
                   help="override the vector/scalar guess")
    p.add_argument("--solid", help="optional .npy mask on the same grid")
    p.add_argument("--ascii", action="store_true", help="ASCII instead of binary")
    p.add_argument("--no-magnitude", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        return _self_test()
    if not a.input or not a.output:
        p.error("input and output are required (or use --self-test)")

    field = np.load(a.input)
    solid = np.load(a.solid) if a.solid else None
    write_vtk(a.output, field, name=a.name, kind=a.kind, solid=solid,
              magnitude=not a.no_magnitude, ascii_mode=a.ascii)
    print(f"{a.input} {field.shape} -> {a.output} "
          f"({'ascii' if a.ascii else 'binary'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
