"""Named reference-target registry for the SCF training loss.

The trained functional is regressed against reference properties from a
high-accuracy method (PySCF). Beyond the total energy, we may match the on-grid
density, the XC potential ``Vxc``, the converged density matrix ``dm``, and more
later (orbital energies, dipole, forces, ...).

These targets share the shape of the *feature bag* (see
:mod:`qex.functionals.features`): an open-ended set of named arrays, only some
present per sample. Each target additionally needs two behaviours — how to read
the **prediction** off the converged SCF state, and how to **reduce**
``(prediction, reference)`` to a scalar. Both live in the :data:`TARGETS`
registry here, keyed by target name.

The loss then iterates exactly the reference keys a sample carries
(``inp.targets``) and sums ``weight[name] * reduce(predict(state), ref)``. Adding
a property is one registry entry plus producing it in data generation — the loss
code never changes, and the loss signature does not grow.

Note: the **energy** term is intentionally NOT scored through this registry by
the unrolled / scan losses, which accumulate it across SCF cycles (with the
``ignore_ks_iter`` mask) — keeping that exact behaviour. The registry owns the
converged-state targets (``density``, ``vxc``, ``dm``); the energy entry is here
for completeness and for paths (e.g. implicit diff) that score energy once at the
fixed point.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from chex import Array


@dataclass(frozen=True)
class SCFState:
    """Converged SCF quantities a target may predict against.

    Filled once at convergence and handed to every target's ``predict`` so the
    loss stays flat. ``vxc`` is the converged XC potential matrix (AO basis) —
    the ``get_veff`` XC part, i.e. ``vhf - J`` — the same object the reference
    ``Vxc`` is built from in data generation.
    """

    e_tot: Array      # scalar total energy
    dm: Array         # converged 1-RDM (AO)
    rho: Array        # density on the grid, (n_grid,)
    vxc: Array        # XC potential matrix (AO), (nao, nao)


@dataclass(frozen=True)
class Target:
    """One reference property: how to predict it, and how to score it."""

    predict: Callable[[SCFState], Array]      # converged state -> prediction
    reduce: Callable[[Array, Array], Array]   # (prediction, reference) -> scalar


def _sq(pred: Array, ref: Array) -> Array:
    """Squared error for a scalar target (e.g. total energy)."""
    return (pred - ref) ** 2


def _mse(pred: Array, ref: Array) -> Array:
    """Mean squared error over all elements (grid density / matrices)."""
    return jnp.mean((pred - ref) ** 2)


# Registry of known reference targets. Add a property here (+ produce it in data
# generation) and it becomes a loss term with no change to the SCF loops.
TARGETS: dict[str, Target] = {
    "energy": Target(predict=lambda st: st.e_tot, reduce=_sq),
    "density": Target(predict=lambda st: st.rho, reduce=_mse),
    "vxc": Target(predict=lambda st: st.vxc, reduce=_mse),
    "dm": Target(predict=lambda st: st.dm, reduce=_mse),
}


def target_loss(
    state: SCFState,
    references: dict,
    weights: dict | None = None,
    *,
    skip: tuple[str, ...] = (),
) -> Array:
    """Sum ``weight[name] * reduce(predict(state), ref)`` over present targets.

    Only keys in ``references`` are considered, and each contributes only if it
    has a **positive weight** in ``weights`` — the default weight is ``0.0`` (off).
    This matters because some references are *always* present (e.g. ``dm`` rides
    in every target bag): a converged-state target must be switched on explicitly,
    never scored just because the data carries it. ``skip`` names targets handled
    elsewhere (e.g. ``"energy"``, accumulated across cycles by the loss bodies).

    An unknown reference key with a positive weight fails loudly — a typo'd target
    name should not silently contribute nothing. (A key absent from ``weights``,
    i.e. weight 0, is simply skipped without a registry lookup.)
    """
    weights = weights or {}
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for name, ref in references.items():
        if name in skip:
            continue
        weight = weights.get(name, 0.0)
        if weight == 0.0:
            continue
        if name not in TARGETS:
            raise KeyError(
                f"Unknown reference target {name!r} (known: {sorted(TARGETS)}). "
                f"Register it in qex.scf.targets.TARGETS."
            )
        spec = TARGETS[name]
        total = total + weight * spec.reduce(spec.predict(state), ref)
    return total
