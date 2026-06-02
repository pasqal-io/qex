"""Create an *artificial* H2O dataset: a synthetic geometry set, real CCSD data.

"Artificial" here means the *geometries* are generated programmatically -- a
small scan of O-H bond length and H-O-H angle distortions around equilibrium --
while the reference energies/densities are computed for real with PySCF (CCSD).
The result is one self-describing ``data/h2o.h5`` you can train against with no
further quantum-chemistry at train time::

    python examples/make_h2o_dataset.py                       # -> data/h2o.h5
    qex train --config examples/train_from_dataset.yaml --data.dataset_file data/h2o.h5

This is the programmatic sibling of ``examples/systems_h2.yaml``: instead of
listing molecules in YAML, we build the ``MoleculeConfig`` list in code (handy
when geometries come from a sweep/formula). Both feed the same ``build_dataset``
and produce the same kind of ``.h5``.

Generation is resumable: each system is flushed to disk as it finishes, so a
re-run reuses what is already there and only computes the missing geometries.
"""

from __future__ import annotations

import math
from pathlib import Path

from qex.data_io import DataGenerator, MoleculeConfig, build_dataset

# Output dataset file (consumed by examples/train_from_dataset.yaml).
OUT_PATH = Path("data/h2o.h5")

# Reference settings shared by every geometry.
METHOD = "ccsd"
BASIS = "631g"
UNITS = "Ang"
GRID_DENSITY = 0

# Equilibrium-ish H2O geometry to distort around.
R_EQ = 0.96      # O-H bond length (Angstrom)
THETA_EQ = 104.5  # H-O-H angle (degrees)


def water_coords(r_oh: float, angle_deg: float) -> str:
    """PySCF atom string for a C2v water at bond length ``r_oh`` and angle.

    O at the origin; the two H placed symmetrically in the xz-plane so the
    bisector lies along +z.
    """
    half = math.radians(angle_deg) / 2.0
    x = r_oh * math.sin(half)
    z = r_oh * math.cos(half)
    return f"O 0 0 0; H {x:.6f} 0 {z:.6f}; H {-x:.6f} 0 {z:.6f}"


def _cfg(label: str, r_oh: float, angle_deg: float) -> MoleculeConfig:
    return MoleculeConfig(
        name=label,
        atom_coords=water_coords(r_oh, angle_deg),
        units=UNITS,
        basis=BASIS,
        method=METHOD,
        grid_density=GRID_DENSITY,
    )


def build_split_configs() -> dict[str, list[MoleculeConfig]]:
    """A small, disjoint train/val/test set of distorted water geometries."""
    # Train: a grid of bond-length x angle distortions around equilibrium.
    train = []
    for r in (0.90, 0.96, 1.02, 1.10):
        for a in (100.0, 104.5, 109.0):
            train.append(_cfg(f"H2O_r{r:.2f}_a{a:.0f}", r, a))

    # Val / test: distortions deliberately *off* the training grid.
    val = [
        _cfg("H2O_r0.93_a102", 0.93, 102.0),
        _cfg("H2O_r1.06_a107", 1.06, 107.0),
    ]
    test = [
        _cfg("H2O_r0.88_a98", 0.88, 98.0),
        _cfg("H2O_r0.99_a105", 0.99, 105.0),
        _cfg("H2O_r1.14_a111", 1.14, 111.0),
    ]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    split_configs = build_split_configs()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(
        "Building H2O dataset -> "
        + " | ".join(f"{s}: {len(c)}" for s, c in split_configs.items())
        + f"  (method={METHOD}, basis={BASIS})  [{OUT_PATH}]"
    )
    data_generator = DataGenerator(OUT_PATH.parent)
    dataset = build_dataset(data_generator, split_configs, path=OUT_PATH)

    print(
        f"\nDone -> train: {len(dataset.train)} | val: {len(dataset.val)} | "
        f"test: {len(dataset.test)}  [{OUT_PATH}]"
    )
    print(
        "Train with:\n"
        f"  qex train --config examples/train_from_dataset.yaml "
        f"--data.dataset_file {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
