"""Tests for the named model-feature bag (qex.functionals.features).

These pin the contract that replaced the positional `precomputed` tail:

- `select` returns exactly the requested keys and fails loudly (by name) on a
  missing one — the whole point of moving off index-by-position.
- A no-feature model gets an empty bag and is immune to extras the dataset
  carries (the case that previously needed an `include_descriptor_ctx` flag).
- The change is jit-safe: a dict feature bag is a pytree, so vmapping the SCF
  loss over it (empty *and* non-empty) traces and runs.
"""

import jax
import jax.numpy as jnp
import pytest

from qex.functionals import DescriptorXC, GlobalMLP, select
from qex.functionals.features import KNOWN_FEATURES


def test_select_returns_exactly_requested_keys():
    bag = {"grid_coords": jnp.zeros((4, 3)), "atom_coords": jnp.zeros((2, 3))}
    out = select(bag, ("atom_coords",))
    assert set(out) == {"atom_coords"}
    assert out["atom_coords"] is bag["atom_coords"]


def test_select_empty_request_is_empty_bag():
    bag = {"grid_coords": jnp.zeros((4, 3))}
    assert select(bag, ()) == {}


def test_select_missing_key_raises_by_name():
    bag = {"grid_coords": jnp.zeros((4, 3))}
    with pytest.raises(KeyError, match="atom_coords"):
        select(bag, ("grid_coords", "atom_coords"))


def test_networks_declare_required_features():
    # The pipeline keys off this attribute instead of model-type booleans.
    assert DescriptorXC.required_features == ("grid_coords", "atom_coords")
    assert GlobalMLP.required_features == ()
    # Declared features are part of the known set (guards typos).
    for key in DescriptorXC.required_features:
        assert key in KNOWN_FEATURES


def test_global_mlp_ignores_a_nonempty_dataset_bag():
    """A no-feature model trains from a dataset that carries extra features.

    Previously a `GlobalMLP` crashed if the data carried grid/atom coords. Now
    it declares `required_features=()`, so the pipeline narrows the bag to `{}`
    and the model never sees them.
    """
    n_grid = 8
    net = GlobalMLP(features=(4,))
    params = net.init(jax.random.PRNGKey(0), jnp.ones(n_grid))
    full_bag = {
        "grid_coords": jnp.zeros((n_grid, 3)),
        "atom_coords": jnp.zeros((2, 3)),
    }
    narrowed = select(full_bag, net.required_features)
    assert narrowed == {}
    # grid_weights rides the global-encoding call signature; the bag is empty.
    out = net.apply(params, jnp.ones(n_grid), jnp.ones(n_grid), **narrowed)
    assert out.shape == ()


def test_vmap_over_feature_bag_is_jit_safe():
    """vmapping a function over a (possibly empty) dict bag traces under jit.

    This is the core jit concern of the refactor: the SCF loss is vmapped with a
    dict argument whose `in_axes` is `{k: 0}`. A dict is a pytree, so both an
    empty bag and a populated one trace and run.
    """
    batch = 3
    core = jnp.arange(batch, dtype=jnp.float64)

    def f(x, features):
        extra = sum(jnp.sum(v) for v in features.values()) if features else 0.0
        return x + extra

    # Empty bag: trivial pytree, vmap is a no-op over it.
    empty = jax.jit(
        lambda c, feats: jax.vmap(f, in_axes=(0, {}))(c, feats)
    )
    out0 = empty(core, {})
    assert out0.shape == (batch,)

    # Non-empty bag: each key mapped over the leading batch axis.
    bag = {"grid_coords": jnp.ones((batch, 4, 3))}
    populated = jax.jit(
        lambda c, feats: jax.vmap(f, in_axes=(0, {"grid_coords": 0}))(c, feats)
    )
    out1 = populated(core, bag)
    assert out1.shape == (batch,)
    # Each row added its own 4*3=12 ones.
    assert jnp.allclose(out1, core + 12.0)
