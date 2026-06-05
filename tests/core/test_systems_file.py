"""Tests for the declarative systems-file loader and dataset-file round-trip.

The systems file is the simple, human-authored input to dataset generation:
a YAML of molecules -> ``{split: [MoleculeConfig]}``. These tests cover parsing,
defaults/override merging, validation (loud failures), the content hash, and a
PySCF-free round-trip of a fabricated dataset through HDF5.
"""

import numpy as np
import pytest

from qex.data_io import (
    Datapoint,
    MoleculeConfig,
    QexDataset,
    load_dataset,
    load_systems_file,
    save_dataset,
    systems_file_hash,
)

_SYSTEMS_YAML = """\
defaults:
  method: ccsd
  basis: 631g
  units: Ang
  grid_density: 0

train:
  - {name: H2_0.74, atom_coords: "H 0 0 0; H 0 0 0.74"}
  - {name: LiH, atom_coords: "Li 0 0 0; H 0 0 1.60", basis: 6-31g}
val:
  - {name: H2_0.90, atom_coords: "H 0 0 0; H 0 0 0.90"}
test:
  - {name: H2_0.80, atom_coords: "H 0 0 0; H 0 0 0.80"}
"""


def _write(tmp_path, text, name="systems.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_load_splits_and_counts(tmp_path):
    path = _write(tmp_path, _SYSTEMS_YAML)
    splits = load_systems_file(path)
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert all(isinstance(c, MoleculeConfig) for c in splits["train"])


def test_defaults_and_override(tmp_path):
    path = _write(tmp_path, _SYSTEMS_YAML)
    splits = load_systems_file(path)
    h2, lih = splits["train"]
    # Defaults applied.
    assert h2.basis == "631g"
    assert h2.method == "ccsd"
    assert h2.name == "H2_0.74"
    # Per-entry override wins over defaults.
    assert lih.basis == "6-31g"
    assert lih.method == "ccsd"  # still from defaults


def test_missing_split_is_empty(tmp_path):
    path = _write(
        tmp_path,
        "train:\n  - {name: H2, atom_coords: \"H 0 0 0; H 0 0 0.74\"}\n",
    )
    splits = load_systems_file(path)
    assert len(splits["train"]) == 1
    assert splits["val"] == []
    assert splits["test"] == []


def test_unknown_field_raises(tmp_path):
    path = _write(
        tmp_path,
        "train:\n  - {name: H2, atom_coords: \"H 0 0 0; H 0 0 0.74\", basis_set: 631g}\n",
    )
    with pytest.raises(ValueError, match="unknown field"):
        load_systems_file(path)


def test_missing_required_field_raises(tmp_path):
    path = _write(tmp_path, "train:\n  - {name: H2}\n")  # no atom_coords
    with pytest.raises(ValueError, match="atom_coords"):
        load_systems_file(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_systems_file(tmp_path / "nope.yaml")


def test_hash_stable_and_sensitive(tmp_path):
    a = _write(tmp_path, _SYSTEMS_YAML, "a.yaml")
    # Cosmetic change (comment + reordered keys) -> same hash.
    cosmetic = "# a comment\n" + _SYSTEMS_YAML
    b = _write(tmp_path, cosmetic, "b.yaml")
    assert systems_file_hash(a) == systems_file_hash(b)
    # Real change (different basis) -> different hash.
    changed = _SYSTEMS_YAML.replace("basis: 631g", "basis: cc-pvdz")
    c = _write(tmp_path, changed, "c.yaml")
    assert systems_file_hash(a) != systems_file_hash(c)


def _fake_datapoint(name, nao=2, ngrid=4, with_ctx=False):
    """A fabricated (non-PySCF) datapoint with self-consistent array shapes."""
    cfg = MoleculeConfig(name=name, atom_coords="H 0 0 0; H 0 0 0.74")
    ctx = {}
    if with_ctx:
        ctx = dict(grid_coords=np.zeros((ngrid, 3)), atom_coords=np.zeros((2, 3)))
    return Datapoint(
        meta=cfg,
        energy=-1.0,
        density=np.ones(ngrid),
        coords=np.zeros((ngrid, 3)),
        dm=np.eye(nao),
        eri=np.zeros((nao, nao, nao, nao)),
        ao_grid=np.ones((ngrid, nao)),
        grid_weights=np.ones(ngrid),
        s1e=np.eye(nao),
        h1e=np.eye(nao),
        energy_nuc=0.5,
        nelectron=2,
        **ctx,
    )


def test_features_bag_carries_ctx():
    """A ctx-carrying datapoint exposes grid/atom coords, selectable by name.

    One dataset serves any model: the core inputs are always the same 7 arrays;
    the optional features ride in a named bag the consumer selects from via
    ``required_features``.
    """
    dp = _fake_datapoint("a", with_ctx=True)
    assert dp.has_descriptor_ctx
    # The full bag is available on the datapoint...
    assert set(dp.features) == {"grid_coords", "atom_coords"}
    # ...and to_scf_inputs narrows it to the model's declared features.
    inp = dp.to_scf_inputs(("grid_coords", "atom_coords"))
    assert set(inp.features) == {"grid_coords", "atom_coords"}


def test_features_bag_narrows_to_request():
    """A no-feature model gets an empty bag even when the datapoint carries ctx."""
    dp = _fake_datapoint("a", with_ctx=True)
    inp = dp.to_scf_inputs()  # required_features=()
    assert inp.features == {}


def test_dataset_roundtrip_without_pyscf(tmp_path):
    """A fabricated dataset survives save -> load and yields SCFInputs bundles."""
    ds = QexDataset(
        train=[_fake_datapoint("a"), _fake_datapoint("b")],
        val=[_fake_datapoint("c")],
        test=[_fake_datapoint("d")],
    )
    out = tmp_path / "ds.h5"
    save_dataset(ds, out)
    loaded = load_dataset(out)
    assert len(loaded.train) == 2
    assert len(loaded.val) == 1
    assert len(loaded.test) == 1
    # to_scf_inputs packs the physics arrays + targets into one SCFInputs bundle.
    inp = loaded.train[0].to_scf_inputs()
    assert float(inp.targets["energy"]) == pytest.approx(-1.0)
    assert set(inp.targets) == {"energy", "density", "dm"}  # no vxc on this fake
    assert inp.features == {}  # no optional features stored
    assert inp.dm.shape == (2, 2)


def test_optional_vxc_target_roundtrips(tmp_path):
    """A stored `vxc` reference target survives save -> load and lands in the bag.

    `vxc` is an optional reference target (only a KS reference produces one); a
    datapoint that carries it must round-trip it, and a datapoint without it must
    simply omit the key — never fabricate one.
    """
    dp_with = _fake_datapoint("with_vxc")
    dp_with.vxc = np.full((2, 2), 0.25)
    dp_without = _fake_datapoint("no_vxc")  # vxc stays None

    ds = QexDataset(train=[dp_with, dp_without], val=[], test=[])
    out = tmp_path / "ds.h5"
    save_dataset(ds, out)
    loaded = load_dataset(out)

    by_name = {dp.meta.name: dp for dp in loaded.train}
    assert by_name["with_vxc"].vxc is not None
    np.testing.assert_allclose(by_name["with_vxc"].vxc, 0.25)
    assert by_name["no_vxc"].vxc is None

    # ...and it surfaces in the target bag exactly when present.
    assert "vxc" in by_name["with_vxc"].to_scf_inputs().targets
    assert "vxc" not in by_name["no_vxc"].to_scf_inputs().targets
