"""Declarative *systems file*: a YAML list of molecules to compute reference data for.

This is the simple, human-authored input to dataset generation. You write a YAML
naming the molecules (and their basis/method/grid), then build a dataset from it::

    qex gen-data --systems examples/systems_h2.yaml -o data/h2.h5

and afterwards train against ``data/h2.h5`` without ever re-running PySCF.

File format
-----------
A top-level ``defaults`` block (optional) supplies fields shared by every entry;
each per-split entry is a partial :class:`~qex.data_io.MoleculeConfig` that
*overrides* those defaults. ``train`` / ``val`` / ``test`` are lists of entries
(any subset; a missing split is empty)::

    defaults:
      method: ccsd
      basis: 631g
      units: Ang
      grid_density: 0

    train:
      - {name: H2_0.74, atom_coords: "H 0 0 0; H 0 0 0.74"}
      - {name: LiH,     atom_coords: "Li 0 0 0; H 0 0 1.60", basis: 6-31g}  # per-entry override
    val:
      - {name: H2_0.90, atom_coords: "H 0 0 0; H 0 0 0.90"}
    test:
      - {name: H2_0.80, atom_coords: "H 0 0 0; H 0 0 0.80"}

Each entry must resolve to a valid :class:`MoleculeConfig` -- ``name`` and
``atom_coords`` are required (no sensible default exists), everything else falls
back to ``defaults`` and then to the dataclass defaults. Unknown keys are an
error (a typo like ``basis_set:`` fails loudly instead of being silently
ignored).
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from qex.data_io.dataset_generation import MoleculeConfig

_SPLITS = ("train", "val", "test")


def _valid_keys() -> set[str]:
    return {f.name for f in fields(MoleculeConfig)}


def _build_config(entry: dict[str, Any], defaults: dict[str, Any], where: str) -> MoleculeConfig:
    """Merge one entry over ``defaults`` and construct a :class:`MoleculeConfig`."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: each system must be a mapping, got {type(entry).__name__}.")

    merged = {**defaults, **entry}
    valid = _valid_keys()
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"{where}: unknown field(s) {sorted(unknown)} (valid: {sorted(valid)}). "
            "Check for typos in the systems file or its `defaults` block."
        )
    for required in ("name", "atom_coords"):
        if required not in merged:
            raise ValueError(f"{where}: missing required field {required!r}.")
    return MoleculeConfig(**merged)


def load_systems_file(path: str | Path) -> dict[str, list[MoleculeConfig]]:
    """Parse a systems YAML into ``{split: [MoleculeConfig, ...]}``.

    Returns a dict with all three split keys (empty lists for absent splits), so
    callers can iterate ``("train", "val", "test")`` unconditionally.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: on a malformed file, an unknown field, or a missing required
            field -- failing loudly rather than silently producing bad configs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Systems file not found: {path}")

    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level must be a mapping (got {type(doc).__name__}).")

    defaults = doc.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: `defaults` must be a mapping.")

    unexpected = set(doc) - {"defaults", *_SPLITS}
    if unexpected:
        raise ValueError(
            f"{path}: unexpected top-level key(s) {sorted(unexpected)} "
            f"(expected `defaults` and any of {list(_SPLITS)})."
        )

    result: dict[str, list[MoleculeConfig]] = {s: [] for s in _SPLITS}
    for split in _SPLITS:
        entries = doc.get(split, []) or []
        if not isinstance(entries, list):
            raise ValueError(f"{path}: `{split}` must be a list of systems.")
        for i, entry in enumerate(entries):
            result[split].append(
                _build_config(entry, defaults, where=f"{path}:{split}[{i}]")
            )
    return result


def systems_file_hash(path: str | Path) -> str:
    """Content fingerprint of a systems file, for dataset cache-staleness checks.

    Hashing the *resolved* configs (not the raw bytes) means cosmetic edits
    (whitespace, key order, comments) do not invalidate a cached dataset, while
    any change to an actual field does.
    """
    splits = load_systems_file(path)
    payload = {
        split: sorted(repr(cfg) for cfg in configs) for split, configs in splits.items()
    }
    blob = repr(payload)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
