"""The per-sample input bundle the SCF losses consume.

Closed-shell RKS needs the same handful of physics arrays for every system
(``dm``, ``eri``, the AO/grid tensors, ``s1e``/``h1e``, ``energy_nuc``,
``nelectron``), plus — for training — the reference targets to score against
(``exact_energy``, ``exact_density``) and the named model-feature bag
(``features``; empty for a plain density model). Carrying these as ~12 positional
arguments meant threading them, in lockstep order, through the dataset packer, the
batch stacker, the ``vmap`` ``in_axes`` spec, and the loss signature — four sites
to edit (and keep ordered) to add one input.

:class:`SCFInputs` collapses that into one registered pytree. Because it is a
registered dataclass it flattens to its leaves automatically, so it composes with
``jit`` / ``grad`` / ``vmap`` / ``scan`` with no special handling:

    # batch a list of per-sample SCFInputs into one leading-axis-batched SCFInputs
    batch = SCFInputs.stack(samples)
    # map every array leaf over axis 0; `params` is shared (not mapped)
    losses = jax.vmap(loss_fn, in_axes=(None, 0))(params, batch)

Adding a new physics input becomes a single field here (populated where the data
is produced, read where the SCF math needs it) instead of a four-site edit.

Static hyperparameters (``max_cycle``, ``frac_*``, ``solver``, ...) are the same
for every sample and trace-time constant, so they deliberately do NOT live here —
they stay keyword-only on the loss functions, bound once before ``vmap``.
"""

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from chex import Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SCFInputs:
    """One molecular system's SCF inputs + (optional) training targets.

    All fields are array leaves of the pytree (``features`` is a nested dict of
    arrays). A batched ``SCFInputs`` (from :meth:`stack`) has the same structure
    with a leading batch axis on every leaf — exactly what ``jax.vmap(...,
    in_axes=0)`` expects.

    The supervision targets ``exact_energy`` / ``exact_density`` are only read by
    the loss functions (``rks_loss`` / ``rks_loss_scan``); the forward-only
    ``rks_energy`` ignores them. ``features`` is the named model-feature bag (see
    :mod:`qex.functionals.features`), already narrowed to the network's
    ``required_features``; empty (``{}``) for a model that needs none.
    """

    # --- physics inputs (every closed-shell RKS run needs exactly these) ---
    dm: Array                 # SCF INITIAL GUESS (seed), not a regression target
    eri: Array
    ao_grid: Array
    grid_weights: Array
    s1e: Array
    h1e: Array
    energy_nuc: Array
    nelectron: Array          # discrete; the SCF loop stop_gradient's it
    # --- named model-feature bag (nested {str: Array}; {} for a plain MLP) ---
    features: dict
    # --- named reference-target bag (training only; {} for forward-only eval) ---
    # e.g. {'energy': scalar, 'density': (g,), 'vxc': (n,n), 'dm': (n,n)};
    # only the keys the dataset produced. See qex.scf.targets.
    targets: dict

    @classmethod
    def stack(cls, samples: list["SCFInputs"]) -> "SCFInputs":
        """Stack a list of per-sample ``SCFInputs`` into one batched ``SCFInputs``.

        Every leaf gains a leading batch axis (``jnp.stack`` over the samples).
        All samples must share structure — notably the same ``features`` keys;
        ``tree.map`` raises on a mismatch rather than silently dropping a leaf.
        """
        if not samples:
            raise ValueError("Cannot stack an empty list of SCFInputs.")
        return jax.tree.map(lambda *xs: jnp.stack(xs), *samples)

    def replace(self, **changes) -> "SCFInputs":
        """Return a copy with the named fields replaced (frozen-dataclass update)."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(changes)
        return SCFInputs(**current)
