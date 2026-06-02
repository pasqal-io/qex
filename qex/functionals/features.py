"""Named, extensible model-feature bag (replaces the positional `precomputed` tail).

The SCF loop consumes a fixed set of *core physics inputs* positionally
(`eri, ao_grid, grid_weights, s1e, h1e, energy_nuc, nelectron`). Everything a
*network* needs on top of the density — atom-centered geometry, density
gradients, descriptors imported from another DFT code — is open-ended and varies
per functional. Those are **model features**, carried in a plain
``dict[str, Array]`` (a pytree, so it rides through ``jax.vmap``/``lax.scan``
unchanged) and selected per consumer.

Contract:
    - **Producer** writes whatever feature keys it can compute (a key per
      feature; no index, no arity).
    - **Consumer** (a Flax network) declares ``required_features: tuple[str, ...]``.
    - The pipeline calls :func:`select` to hand each consumer exactly the keys it
      asked for. A missing required key fails loudly *here*, by name, instead of
      as a shape error deep inside ``vmap``.

Adding a feature end-to-end is: produce the key + add it to one network's
``required_features``. No other file changes.
"""

from __future__ import annotations

from collections.abc import Mapping

from chex import Array

# Feature keys currently understood by the pipeline. This tuple is documentation
# + a guard (a typo'd key in `required_features` is caught early), not a hard
# schema: producers may store extra keys and consumers may request any subset.
# Extend it when adding a feature (e.g. "rho_grad", "external_descriptor").
KNOWN_FEATURES = ("grid_coords", "atom_coords")

FeatureBag = dict[str, Array]


def select(bag: Mapping[str, Array], keys: tuple[str, ...]) -> FeatureBag:
    """Return exactly ``keys`` from ``bag`` as a new dict, hard-failing on a miss.

    This is the one place the named-feature contract is enforced: a consumer that
    declares ``required_features`` gets precisely those arrays, and a key it asked
    for but the dataset never produced raises a named, actionable error rather
    than surfacing as a cryptic ``__call__() missing argument`` / shape mismatch
    inside the vmapped SCF loop.
    """
    missing = [k for k in keys if k not in bag]
    if missing:
        raise KeyError(
            f"Required feature(s) {missing} not in the dataset's feature bag "
            f"(available: {sorted(bag)}). Rebuild the dataset so it stores these "
            f"features, or train a model that does not require them."
        )
    return {k: bag[k] for k in keys}
