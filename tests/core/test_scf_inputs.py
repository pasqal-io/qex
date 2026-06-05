"""Unit tests for the SCFInputs batched-pytree bundle.

`SCFInputs` collapses the per-sample SCF arguments into one registered pytree so
the training loop can `vmap` over it with `in_axes=(None, 0)` instead of carrying
~12 positional args through four sites. These tests pin the pytree contract that
makes that work: stacking, vmap-over-axis-0, the empty-feature-bag case, and the
structural-consistency guard.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qex.scf.inputs import SCFInputs

jax.config.update("jax_enable_x64", True)


def _sample(seed: int, *, with_features: bool = True) -> SCFInputs:
    k = jax.random.PRNGKey(seed)
    feats = {"grid_coords": jnp.ones((5, 3))} if with_features else {}
    dm = jax.random.normal(k, (2, 2))
    return SCFInputs(
        dm=dm,
        eri=jnp.zeros((2, 2, 2, 2)),
        ao_grid=jnp.ones((5, 2)),
        grid_weights=jnp.ones(5),
        s1e=jnp.eye(2),
        h1e=jnp.eye(2),
        energy_nuc=jnp.asarray(1.0),
        nelectron=jnp.asarray(2),
        features=feats,
        targets={"energy": jnp.asarray(0.5), "density": jnp.ones(5), "dm": dm},
    )


def test_is_registered_pytree():
    """SCFInputs flattens to its array leaves (so jit/grad/vmap/scan accept it)."""
    leaves = jax.tree_util.tree_leaves(_sample(0))
    # 8 physics fields + 1 feature-bag entry + 3 target-bag entries = 12 leaves.
    assert len(leaves) == 12
    assert all(hasattr(x, "shape") for x in leaves)


def test_stack_adds_batch_axis():
    """`stack` gives every leaf a leading batch axis of len(samples)."""
    batch = SCFInputs.stack([_sample(0), _sample(1), _sample(2)])
    assert batch.dm.shape == (3, 2, 2)
    assert batch.nelectron.shape == (3,)
    assert batch.energy_nuc.shape == (3,)
    assert batch.features["grid_coords"].shape == (3, 5, 3)


def test_vmap_over_batch_in_axes_0():
    """A batched SCFInputs maps cleanly under in_axes=(None, 0)."""
    batch = SCFInputs.stack([_sample(0), _sample(1)])

    def per_sample(scale, inp):
        return scale * (inp.dm.sum() + inp.energy_nuc)

    out = jax.vmap(per_sample, in_axes=(None, 0))(2.0, batch)
    assert out.shape == (2,)
    # Matches the manual per-element computation.
    expected = jnp.stack([2.0 * (s.dm.sum() + s.energy_nuc)
                          for s in (_sample(0), _sample(1))])
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-12)


def test_empty_feature_bag_stacks_and_vmaps():
    """A plain (no-feature) model carries `{}`; it survives stack + vmap."""
    batch = SCFInputs.stack([_sample(0, with_features=False),
                             _sample(1, with_features=False)])
    assert batch.features == {}
    out = jax.vmap(lambda inp: inp.dm.sum(), in_axes=0)(batch)
    assert out.shape == (2,)


def test_stack_rejects_inconsistent_feature_keys():
    """Samples with different feature keys have different pytree structure, so
    `stack` (a tree_map) must fail loudly rather than silently drop a leaf."""
    with pytest.raises((ValueError, TypeError)):
        SCFInputs.stack([_sample(0, with_features=True),
                         _sample(1, with_features=False)])


def test_stack_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        SCFInputs.stack([])


def test_replace_returns_updated_copy():
    inp = _sample(0)
    inp2 = inp.replace(nelectron=jnp.asarray(4))
    assert int(inp2.nelectron) == 4
    assert int(inp.nelectron) == 2  # original unchanged (frozen)
    # untouched fields are shared
    assert inp2.dm is inp.dm


def test_empty_target_bag_for_eval():
    """Forward-only eval builds a bundle with an empty target bag."""
    inp = _sample(0).replace(targets={})
    leaves = jax.tree_util.tree_leaves(inp)
    # 8 physics + 1 feature entry + 0 targets = 9 leaves.
    assert len(leaves) == 9
    assert inp.targets == {}


def test_optional_target_in_bag():
    """An optional target (e.g. vxc) rides in the bag like any other key."""
    inp = _sample(0).replace(
        targets={**_sample(0).targets, "vxc": jnp.eye(2)}
    )
    batch = SCFInputs.stack([inp, inp])
    assert batch.targets["vxc"].shape == (2, 2, 2)
