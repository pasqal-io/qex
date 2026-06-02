"""Single-file dataset storage for QEX (HDF5).

Instead of a folder-per-geometry with scattered ``.npy``/``.chdens`` files, a
whole train/val/test split is stored in **one** ``.h5`` file -- the moral
equivalent of a PyTorch ``.pt``. Each datapoint is *self-describing*: it carries
the full molecule specification (:class:`~qex.data_io.MoleculeConfig`) alongside
every precomputed tensor the SCF loop needs, so a dataset file is reproducible
and portable on its own.

Layout::

    dataset.h5
      attrs: format_version, config_hash (optional)
      /train/0  {energy, density, coords, dm, eri, ao_grid, grid_weights,
                 s1e, h1e, energy_nuc, nelectron, [grid_coords, atom_coords]}
                attrs: name, atom_coords, basis, method, units, grid_density, ...
      /train/1  ...
      /val/...   /test/...

Public API:
    - :class:`Datapoint`              -- one self-describing system.
    - :class:`QexDataset`             -- the {train, val, test} container.
    - :func:`save_dataset`            -- write a :class:`QexDataset` to ``.h5``.
    - :func:`load_dataset`            -- read one back.
    - :func:`build_dataset`           -- generate a dataset from MoleculeConfigs.

Scaling / size limits
---------------------
This storage is designed for **small-to-medium molecule sets** (H2 ... H2O scale,
tens-to-hundreds of geometries). Three things bound the practical size, none of
which is the HDF5 format itself:

1. Dense ERIs are ``O(nao**4)``. Each :class:`Datapoint` stores the full
   ``(nao, nao, nao, nao)`` two-electron integral tensor (``s1`` symmetry, no
   compression). Concretely, at 8 bytes/float::

       H2 / 631g       nao~4     -> ~2 KB    per point
       H2O / cc-pVDZ   nao~24    -> ~21 MB   per point
       benzene/cc-pVDZ nao~114   -> ~1.4 GB  per point
       ~30 heavy atoms nao~300   -> ~65 GB   per point  (infeasible)

   For larger systems this field must change: store 8-fold-symmetric
   (``aosym="s8"``) or, properly, density-fitted 3-index integrals
   ``(naux, nao, nao)`` and teach the SCF loop to contract them -- turning
   ``O(nao**4)`` into ``O(nao**2 * naux)``.
2. :func:`load_dataset` reads a whole split into memory eagerly. Large sets want
   lazy / partial reads (keep the file open, materialise a datapoint on access).
3. The training loop (:func:`qex.train`) ``jnp.stack`` s an entire split into one
   device array, i.e. full-batch on host/VRAM. Minibatching is needed before the
   *file* size is the real ceiling.

So: HDF5 was chosen precisely because it supports compression + partial loading,
but the current code uses it as a plain container. Add gzip+chunking,
density-fitted integrals, lazy loading, and minibatching (in that order) to push
past medium systems.

Data consistency (CRITICAL -- read before mixing codes, e.g. ORCA + PySCF)
-------------------------------------------------------------------------
A :class:`Datapoint` is only correct if **every array refers to the same AO
basis, the same integration grid, and the same conventions**. The arrays are
mutually dependent:

- ``density`` lives on (``coords``, ``grid_weights``); ``ao_grid`` must be the AO
  values on *that same grid*. A grid mismatch silently corrupts the energy
  integral ``sum(exc * rho * w)`` -- it still runs and trains, just on garbage.
- ``s1e`` / ``h1e`` / ``eri`` / ``dm`` must all be in *one* AO convention. The
  "same" nominal basis can differ between codes in AO ordering, Cartesian vs
  spherical (6d/10f vs 5d/7f), and normalisation.

Therefore, when generating with more than one program: **never let two codes
contribute arrays that must be consistent with each other within one datapoint.**
The safe pattern is single-code-per-array-group:

- ORCA (or any high-accuracy code) may supply the **reference total energy only**
  -- one scalar, with no consistency coupling (e.g. DLPNO-CCSD(T), which scales
  to far larger systems than canonical CCSD).
- PySCF supplies *everything that must agree*: basis, grid, ``s1e``/``h1e``/
  ``eri``/``ao_grid``/weights, and the density-on-grid -- consistent by
  construction because one code built them all.

If a multi-code path is added, record provenance explicitly (e.g.
``energy_source`` / ``basis_source`` / ``grid_source`` in ``meta``) and assert
that everything except the energy shares one source, so a mismatch fails loudly
instead of silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from loguru import logger

from qex.data_io.dataset_generation import DataGenerator, MoleculeConfig

FORMAT_VERSION = 1

# Datapoint fields that are numpy arrays stored as HDF5 datasets. Scalars
# (energy, energy_nuc, nelectron) are stored as datasets too for simplicity.
_ARRAY_FIELDS = (
    "energy",
    "density",
    "coords",
    "dm",
    "eri",
    "ao_grid",
    "grid_weights",
    "s1e",
    "h1e",
    "energy_nuc",
    "nelectron",
)

# Optional, extensible *model features* (see qex.functionals.features): named
# arrays that some networks consume on top of the density, produced only when a
# DFT code can compute them. Each is stored as an optional HDF5 dataset and ends
# up as a key in the per-sample feature bag. Adding a feature (e.g. "rho_grad"
# for GGA, or a classical descriptor for a B3LYP-style functional) means adding
# its name here + a field on Datapoint + producing it — no index surgery, no
# change to the SCF loop.
_OPTIONAL_ARRAY_FIELDS = ("grid_coords", "atom_coords")


def datapoint_uid(meta: MoleculeConfig) -> str:
    """Stable identifier for a system, used as its HDF5 group name.

    Two configs that would generate the same data share a uid, so a resumed
    build can tell which systems are already present and skip them.
    """
    key = "|".join(
        str(x)
        for x in (
            meta.name,
            meta.atom_coords,
            meta.units,
            meta.basis,
            meta.method,
            meta.grid_density,
            meta.deriv,
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class Datapoint:
    """One self-describing molecular system: full spec + precomputed tensors.

    ``meta`` is the molecule specification; the array fields are the SCF inputs
    and reference targets. ``grid_coords`` / ``atom_coords`` are only set for
    descriptor-style functionals that need geometric context.

    A datapoint may also be a *failure record*: if the reference calculation did
    not converge (or raised), ``converged`` is False and the array fields are
    ``None``. Such records are persisted (so a resumed build does not blindly
    recompute them) but excluded from training/eval by the loaders.
    """

    meta: MoleculeConfig
    energy: float | None = None
    density: np.ndarray | None = None
    coords: np.ndarray | None = None
    dm: np.ndarray | None = None
    eri: np.ndarray | None = None
    ao_grid: np.ndarray | None = None
    grid_weights: np.ndarray | None = None
    s1e: np.ndarray | None = None
    h1e: np.ndarray | None = None
    energy_nuc: float | None = None
    nelectron: int | None = None
    grid_coords: np.ndarray | None = None
    atom_coords: np.ndarray | None = None
    converged: bool = True
    error: str = ""

    @property
    def uid(self) -> str:
        return datapoint_uid(self.meta)

    @property
    def has_descriptor_ctx(self) -> bool:
        return self.grid_coords is not None and self.atom_coords is not None

    @property
    def features(self) -> dict:
        """The named model-feature bag for this system (see qex.functionals.features).

        Every optional feature field this datapoint actually carries, as a
        ``{name: jnp.ndarray}`` dict. Networks select the subset they declare in
        ``required_features``; the SCF loop forwards the bag without reading it.
        Adding a feature field to :class:`Datapoint` automatically surfaces it
        here — no change to this method.
        """
        bag = {}
        for name in _OPTIONAL_ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                bag[name] = jnp.asarray(value)
        return bag

    @classmethod
    def failed(cls, meta: MoleculeConfig, error: str = "") -> "Datapoint":
        """Construct a failure record for a system that did not converge/raised."""
        return cls(meta=meta, converged=False, error=error)

    def to_training_tuple(self, required_features: tuple[str, ...] = ()) -> tuple:
        """Pack into ``(energy, coords_density, core_inputs, features, dm)``.

        This is exactly the structure consumed by :func:`qex.train` and the SCF
        loops, so loading a dataset is a drop-in for inline generation.

        - ``core_inputs`` is the fixed positional tuple of SCF physics arrays
          ``(eri, ao_grid, grid_weights, s1e, h1e, energy_nuc, nelectron)`` —
          every closed-shell RKS run needs exactly these.
        - ``features`` is the named model-feature bag narrowed to
          ``required_features`` (the consuming network's declared keys; default
          none). A model thus only ever sees the features it asked for, and a
          missing one fails loudly by name — no per-model flag, no extras leaking
          into a model that can't accept them.
        """
        from qex.functionals.features import select

        if not self.converged:
            raise ValueError(
                f"Datapoint {self.meta.name!r} did not converge; it has no data "
                f"to train on. Filter with `converged` before calling this."
            )
        core_inputs = (
            jnp.asarray(self.eri),
            jnp.asarray(self.ao_grid),
            jnp.asarray(self.grid_weights),
            jnp.asarray(self.s1e),
            jnp.asarray(self.h1e),
            float(self.energy_nuc),
            int(self.nelectron),
        )
        coords_density = jnp.c_[jnp.asarray(self.coords), jnp.asarray(self.density)]
        return (
            float(self.energy),
            coords_density,
            core_inputs,
            select(self.features, required_features),
            jnp.asarray(self.dm),
        )


@dataclass
class QexDataset:
    """A train/val/test collection of :class:`Datapoint`."""

    train: list[Datapoint] = field(default_factory=list)
    val: list[Datapoint] = field(default_factory=list)
    test: list[Datapoint] = field(default_factory=list)

    def split(self, name: str, *, converged_only: bool = False) -> list[Datapoint]:
        """Datapoints in a split. With ``converged_only`` skips failure records."""
        points = {"train": self.train, "val": self.val, "test": self.test}[name]
        if converged_only:
            return [dp for dp in points if dp.converged]
        return points

    def converged(self, name: str) -> list[Datapoint]:
        """Only the converged datapoints of a split (training/eval-ready)."""
        return self.split(name, converged_only=True)

    def n_failed(self, name: str) -> int:
        """Count of non-converged (failure) records in a split."""
        return sum(1 for dp in self.split(name) if not dp.converged)

    def training_tuples(
        self, name: str, required_features: tuple[str, ...] = ()
    ) -> list[tuple]:
        """The split's *converged* datapoints as training tuples (see Datapoint).

        Each tuple's feature bag is narrowed to ``required_features``, so one
        dataset serves any model with no per-model flag.
        """
        return [
            dp.to_training_tuple(required_features) for dp in self.converged(name)
        ]

    def uids(self, name: str) -> set[str]:
        """uids already present in a split (for resume/skip)."""
        return {dp.uid for dp in self.split(name)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        def fmt(s):
            tot = len(self.split(s))
            fail = self.n_failed(s)
            return f"{s}={tot - fail}+{fail}f" if fail else f"{s}={tot}"

        return f"QexDataset({fmt('train')}, {fmt('val')}, {fmt('test')})"


# --------------------------------------------------------------------------- #
# HDF5 (de)serialisation
#
# There are three ways to write, by intent:
#   - build_dataset(path=...)  : the canonical generator. Incremental, resumable,
#                                failure-tolerant. Use this for real runs.
#   - append_datapoint(...)    : the single-record primitive build_dataset uses;
#                                exposed for custom generation loops.
#   - save_dataset(...)        : one-shot bulk overwrite of an in-memory dataset
#                                (no resume). Convenience for small/derived sets.
# load_dataset reads any of them back (including failure records).
# --------------------------------------------------------------------------- #
def _write_datapoint(group: h5py.Group, dp: Datapoint) -> None:
    """Write one datapoint into an HDF5 group (arrays as datasets, meta as attrs).

    Failure records (``converged=False``) store only the meta + status attrs,
    no array datasets.
    """
    # Molecule spec -> group attributes (JSON for anything non-scalar like
    # custom_grid; primitives stored directly so they're human-readable in h5).
    meta = asdict(dp.meta)
    for key, value in meta.items():
        if value is None:
            group.attrs[key] = "null"
        elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            group.attrs[key] = value
        else:
            group.attrs[key] = json.dumps(np.asarray(value).tolist())

    # Status attributes (always present).
    group.attrs["converged"] = bool(dp.converged)
    group.attrs["error"] = dp.error
    group.attrs["uid"] = dp.uid

    if not dp.converged:
        return  # failure record: no arrays

    for name in _ARRAY_FIELDS:
        group.create_dataset(name, data=np.asarray(getattr(dp, name)))
    for name in _OPTIONAL_ARRAY_FIELDS:
        value = getattr(dp, name)
        if value is not None:
            group.create_dataset(name, data=np.asarray(value))


def _read_datapoint(group: h5py.Group) -> Datapoint:
    """Reconstruct a datapoint (or failure record) from an HDF5 group."""
    meta_fields = {f.name for f in fields(MoleculeConfig)}
    meta_kwargs: dict[str, Any] = {}
    for key in meta_fields:
        if key not in group.attrs:
            continue
        value = group.attrs[key]
        if isinstance(value, str) and value == "null":
            value = None
        elif isinstance(value, str) and value.startswith(("[", "{")):
            value = np.asarray(json.loads(value))
        meta_kwargs[key] = value
    meta = MoleculeConfig(**meta_kwargs)

    converged = bool(group.attrs.get("converged", True))
    error = str(group.attrs.get("error", ""))
    if not converged:
        return Datapoint.failed(meta, error=error)

    arrays = {name: group[name][()] for name in _ARRAY_FIELDS}
    optional = {
        name: group[name][()] for name in _OPTIONAL_ARRAY_FIELDS if name in group
    }
    return Datapoint(
        meta=meta,
        energy=float(arrays.pop("energy")),
        energy_nuc=float(arrays.pop("energy_nuc")),
        nelectron=int(arrays.pop("nelectron")),
        converged=True,
        error=error,
        **arrays,
        **optional,
    )


def save_dataset(
    dataset: QexDataset,
    path: str | Path,
    *,
    config_hash: str | None = None,
) -> Path:
    """Write a whole :class:`QexDataset` to a single ``.h5`` file (bulk overwrite).

    For long, interruptible generation prefer :func:`build_dataset`, which writes
    incrementally and is resumable. ``save_dataset`` is the one-shot writer.

    Args:
        config_hash: optional fingerprint stored as a file attribute so an
            auto-cache can tell whether a cached file matches the current config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        if config_hash is not None:
            f.attrs["config_hash"] = config_hash
        for split in ("train", "val", "test"):
            grp = f.create_group(split)
            for i, dp in enumerate(dataset.split(split)):
                sub = grp.create_group(dp.uid)
                sub.attrs["order"] = i
                _write_datapoint(sub, dp)
    return path


def init_dataset_file(path: str | Path, *, config_hash: str | None = None) -> Path:
    """Create (or open) an ``.h5`` and ensure the split groups exist.

    Idempotent: an existing file is left intact (so generation can resume), only
    creating any missing top-level structure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if "format_version" not in f.attrs:
            f.attrs["format_version"] = FORMAT_VERSION
        if config_hash is not None:
            f.attrs["config_hash"] = config_hash
        for split in ("train", "val", "test"):
            if split not in f:
                f.create_group(split)
    return path


def append_datapoint(path: str | Path, split: str, dp: Datapoint) -> None:
    """Append one datapoint to a split, flushing immediately (crash-safe resume).

    Each finished system is durably on disk before the next is computed, so an
    interrupted build keeps everything completed so far. Re-appending the same
    uid overwrites the previous record (e.g. retrying a failed system).
    """
    with h5py.File(path, "a") as f:
        grp = f.require_group(split)
        if dp.uid in grp:
            del grp[dp.uid]
        sub = grp.create_group(dp.uid)
        sub.attrs["order"] = len(grp) - 1
        _write_datapoint(sub, dp)
        f.flush()


def existing_uids(path: str | Path) -> dict[str, set[str]]:
    """Map split -> set of uids already present in the file (for resume).

    Returns empty sets if the file does not exist.
    """
    path = Path(path)
    result = {"train": set(), "val": set(), "test": set()}
    if not path.exists():
        return result
    with h5py.File(path, "r") as f:
        for split in result:
            if split in f:
                result[split] = set(f[split].keys())
    return result


def failed_uids(path: str | Path) -> dict[str, set[str]]:
    """Map split -> set of uids present but recorded as non-converged."""
    path = Path(path)
    result = {"train": set(), "val": set(), "test": set()}
    if not path.exists():
        return result
    with h5py.File(path, "r") as f:
        for split in result:
            if split not in f:
                continue
            for uid, grp in f[split].items():
                if not bool(grp.attrs.get("converged", True)):
                    result[split].add(uid)
    return result


def load_dataset(path: str | Path) -> QexDataset:
    """Load a :class:`QexDataset` from a ``.h5`` file (converged + failure records)."""
    path = Path(path)
    dataset = QexDataset()
    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("format_version", FORMAT_VERSION))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported dataset format_version {version} "
                f"(expected {FORMAT_VERSION})."
            )
        for split in ("train", "val", "test"):
            if split not in f:
                continue
            grp = f[split]
            # Stable order via the per-group "order" attr (uid group names are
            # not sortable into insertion order on their own).
            keys = sorted(grp.keys(), key=lambda k: int(grp[k].attrs.get("order", 0)))
            for key in keys:
                dataset.split(split).append(_read_datapoint(grp[key]))
    return dataset


def read_config_hash(path: str | Path) -> str | None:
    """Return the ``config_hash`` attribute of a dataset file, or ``None``."""
    with h5py.File(path, "r") as f:
        value = f.attrs.get("config_hash")
    return None if value is None else str(value)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _datapoint_from_generation(
    data_generator: DataGenerator,
    cfg: MoleculeConfig,
    with_descriptor_ctx: bool,
) -> Datapoint:
    """Run PySCF data generation for one molecule and build a :class:`Datapoint`.

    Returns a *failure record* (``converged=False``) if the reference
    calculation does not converge or raises, instead of aborting the whole
    build -- so one bad geometry never loses the rest of a long run.
    """
    # Local import avoids a hard dependency on the SCF stack at module import.
    from qex.scf import get_ao_value

    try:
        mol, mf, dm, energy, density, coords = data_generator.generate_data(
            cfg, save_data=False
        )
    except Exception as exc:  # noqa: BLE001 - record the failure, keep going
        return Datapoint.failed(cfg, error=f"{type(exc).__name__}: {exc}")

    if not bool(getattr(mf, "qex_converged", True)):
        return Datapoint.failed(cfg, error="reference calculation did not converge")

    grid_coords = np.asarray(mf.grids.coords) if with_descriptor_ctx else None
    atom_coords = np.asarray(mol.atom_coords()) if with_descriptor_ctx else None
    return Datapoint(
        meta=cfg,
        energy=float(energy),
        density=np.asarray(density),
        coords=np.asarray(coords),
        dm=np.asarray(dm),
        eri=np.asarray(mol.intor("int2e", aosym="s1")),
        ao_grid=np.asarray(get_ao_value(mol, mf.grids.coords)),
        grid_weights=np.asarray(mf.grids.weights),
        s1e=np.asarray(mf.get_ovlp(mol)),
        h1e=np.asarray(mf.get_hcore(mol)),
        energy_nuc=float(mol.energy_nuc()),
        nelectron=int(mol.nelectron),
        grid_coords=grid_coords,
        atom_coords=atom_coords,
    )


def build_dataset(
    data_generator: DataGenerator,
    split_configs: dict[str, list[MoleculeConfig]],
    *,
    path: str | Path | None = None,
    with_descriptor_ctx: bool = False,
    config_hash: str | None = None,
    retry_failed: bool = False,
) -> QexDataset:
    """Generate a :class:`QexDataset`, optionally writing incrementally + resuming.

    When ``path`` is given, each system is written to the ``.h5`` *as soon as it
    finishes* (flushed to disk), and a re-run **skips systems already present**.
    So a long generation that is killed (or whose job diverges) can simply be
    restarted: completed systems are reused, only the missing ones are computed.

    Args:
        split_configs: ``{"train": [...], "val": [...], "test": [...]}`` (any
            subset; missing splits are left empty).
        path: if set, the ``.h5`` to write/append to and resume from. If None,
            the dataset is built purely in memory.
        with_descriptor_ctx: also store grid/atom coordinates (descriptor nets).
        config_hash: stored on a freshly created file (cache validation).
        retry_failed: if True, previously recorded *failed* systems are
            recomputed; otherwise they are skipped (not retried blindly).
    """
    # In-memory build: no resume/skip, just generate everything and return.
    if path is None:
        dataset = QexDataset()
        for split in ("train", "val", "test"):
            for cfg in split_configs.get(split, []):
                dp = _datapoint_from_generation(data_generator, cfg, with_descriptor_ctx)
                if dp.converged:
                    logger.debug("[{}] {}: ok", split, cfg.name)
                else:
                    logger.warning("[{}] {}: FAILED ({})", split, cfg.name, dp.error)
                dataset.split(split).append(dp)
        return dataset

    # Incremental, resumable build: skip what's already on disk, write as we go.
    init_dataset_file(path, config_hash=config_hash)
    present = existing_uids(path)
    failed = failed_uids(path)
    for split in ("train", "val", "test"):
        for cfg in split_configs.get(split, []):
            uid = datapoint_uid(cfg)
            already = uid in present[split]
            is_failed = uid in failed[split]
            # Skip if present AND (converged, or failed-but-not-retrying).
            if already and not (is_failed and retry_failed):
                if already:
                    logger.debug("[{}] {}: skip (cached)", split, cfg.name)
                continue
            dp = _datapoint_from_generation(data_generator, cfg, with_descriptor_ctx)
            if dp.converged:
                logger.debug("[{}] {}: ok", split, cfg.name)
            else:
                logger.warning("[{}] {}: FAILED ({})", split, cfg.name, dp.error)
            append_datapoint(path, split, dp)

    return load_dataset(path)
