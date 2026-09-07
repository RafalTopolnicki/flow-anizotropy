"""
The one place Python reads a structure GIF.

The dataset stores exactly one copy of each structure, as a GIF, because that is
what the C++ solver reads. Anything on the Python side — TDA filtrations,
baselines, CNNs, plots — must go through `load_structure` here so it sees the
identical array the LBM saw.

Mirroring `LMB2d/lbm.cpp: read_from_gif`
----------------------------------------
    PixelPacket *pixels = image.getPixels(0, 0, LY, LX);
    for(int y=0; y<LY; y++)
      for(int x=0; x<LX; x++)
        if(pixels[LX*y + x].red > 0) F[x][y] = 0;   // pore
        else                         F[x][y] = 1;   // solid

Two things to carry over exactly:

* **The rule is `red > 0` is pore**, not a threshold at 128. Reproduced verbatim
  below. For the binary images the generator writes (0 or 255 only) any
  threshold agrees, but copying the solver's rule means they cannot diverge if a
  non-binary image is ever fed in by mistake.
* **The indexing.** Image (row, col) maps to the solver's `F[x][y]` with
  x = col, y = row. The array returned here is indexed `[row, col]`, i.e.
  `[solver_y, solver_x]` — the same orientation the generator builds its `solid`
  array in, so ψ from the generator and `theta_deg` from the solver are directly
  comparable. Do not transpose it.

    >>> from structure_io import load_structure
    >>> s = load_structure("DATA/aniso/structures/sample_000000_....gif")
    >>> s.shape, s.dtype, s.max()
    ((256, 256), dtype('uint8'), 1)
    >>> porosity = 1.0 - s.mean()        # matches structures.csv and the solver
"""

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["load_structure", "load_many", "porosity"]


def load_structure(path) -> np.ndarray:
    """
    Read a structure GIF as a uint8 array: 1 = solid, 0 = pore.

    Indexed [row, col] = [solver_y, solver_x]. See the module docstring for why
    that orientation matters.
    """
    with Image.open(path) as im:
        arr = np.array(im.convert("L"))
    # read_from_gif: red > 0 -> pore (0); otherwise solid (1)
    return (arr == 0).astype(np.uint8)


def load_many(paths) -> np.ndarray:
    """Stack several structures into (N, H, W) uint8."""
    return np.stack([load_structure(p) for p in paths])


def porosity(solid: np.ndarray) -> float:
    """Pore fraction, matching the `porosity` column of structures.csv."""
    return float(1.0 - solid.mean())


def _self_test(dataset_dir) -> int:
    """
    Check this reader against the solver's own reported porosity.

    The solver writes `porosity` (from getporosity_full()) into every row of
    permeability.csv, so agreeing with it to floating-point precision proves the
    C++ and Python readers see the same bytes as the same structure.
    """
    import pandas as pd

    d = Path(dataset_dir)
    csv = d / "permeability.csv"
    if not csv.exists():
        csv = d / "structures.csv"
    df = pd.read_csv(csv)

    worst = 0.0
    for r in df.head(25).itertuples():
        s = load_structure(d / "structures" / r.filename)
        worst = max(worst, abs(porosity(s) - r.porosity))

    ok = worst < 1e-6
    print(f"[{'ok' if ok else 'FAIL'}] {min(25, len(df))} structures: "
          f"max |python porosity - solver porosity| = {worst:.3e}")
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/structure_io.py <dataset_dir>")
    raise SystemExit(_self_test(sys.argv[1]))
