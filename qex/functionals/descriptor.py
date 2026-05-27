
# =============================================================================
# DescriptorXC: integral-descriptor global encoder
# =============================================================================
#
# Idea: reduce rho(r) (length n_grid) to a small fixed vector of integral
# descriptors (length O(10)), then feed those to a tiny MLP that outputs a
# single scalar E_xc. Cost is O(n_grid) for the integrals (one einsum-like
# reduction) and O(1) for the MLP - independent of n_grid in the dominant
# learning step.
#
# Descriptors used (all smooth functionals of rho, smooth in nuclear coords):
#   d1 = integral rho dV                       (electron count)
#   d2 = integral rho^(4/3) dV                 (LDA exchange-like)
#   d3 = integral rho^(5/3) dV                 (Thomas-Fermi kinetic-like)
#   d4 = integral rho * log(rho + eps) dV      (information / entropy-like)
#   d5_a, d5_b, ... = integral rho(r) * exp(-alpha_k |r - R_a|) dV
#                     summed over atoms (permutation-invariant) for each
#                     alpha_k in CONFIG["descriptor_radial_alphas"].
#
# Gradient descriptors (|grad rho|^2 / rho ...) would require ao_grad on the
# grid - omitted for now to keep the plumbing simple. They can be added by
# precomputing ao_grad and including grad rho in the descriptor function.

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from collections.abc import Sequence, Callable
import jax

def compute_descriptors(
    rho: Array,                # (n_grid,)
    grid_coords: Array,        # (n_grid, 3)
    grid_weights: Array,       # (n_grid,)
    atom_coords: Array,        # (n_atom, 3)
    alphas: Sequence[float],
    rho_floor: float = 1e-10,
) -> Array:
    """Compute fixed-size descriptor vector from rho.

    Returns:
        d: (4 + len(alphas),) array of smooth scalar descriptors.
    """
    rho_safe = jnp.clip(rho, rho_floor, None)

    d1 = jnp.sum(rho * grid_weights)
    d2 = jnp.sum(rho_safe ** (4.0 / 3.0) * grid_weights)
    d3 = jnp.sum(rho_safe ** (5.0 / 3.0) * grid_weights)
    d4 = jnp.sum(rho * jnp.log(rho_safe) * grid_weights)

    # Atom-centered radial shells: sum_a integral rho(r) * exp(-alpha |r - R_a|) dV
    # diff: (n_grid, n_atom, 3) -> dists: (n_grid, n_atom)
    diff = grid_coords[:, None, :] - atom_coords[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)

    rho_w = rho * grid_weights  # (n_grid,)

    def shell(alpha: float) -> Array:
        # exp(-alpha r) on (n_grid, n_atom); sum over atoms then integrate.
        kernel = jnp.exp(-alpha * dists).sum(axis=-1)  # (n_grid,)
        return jnp.sum(rho_w * kernel)

    shells = jnp.stack([shell(a) for a in alphas])  # (n_alpha,)

    return jnp.concatenate([jnp.array([d1, d2, d3, d4]), shells])


class DescriptorXC(nn.Module):
    """Global XC functional from integral descriptors of rho.

    The descriptor vector has fixed size (independent of grid resolution),
    and each descriptor is a smooth functional of rho and atom positions, so
    E_xc is smooth under nuclear motion. The MLP is tiny because its input
    is O(10) numbers, not O(n_grid).

    Architecture:
        d = compute_descriptors(rho, coords, weights, atoms)   # (D,)
        h = MLP(d)                                              # scalar
        E_xc = -scale * softplus(h)                             # <= 0
    """

    hidden: Sequence[int]
    alphas: Sequence[float]
    scale: float = 1.0
    rho_floor: float = 1e-10
    act_fn: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        rho: Array,             # (n_grid,)
        grid_coords: Array,     # (n_grid, 3)
        grid_weights: Array,    # (n_grid,)
        atom_coords: Array,     # (n_atom, 3)
    ) -> Array:                 # scalar
        d = compute_descriptors(
            rho, grid_coords, grid_weights, atom_coords,
            alphas=self.alphas, rho_floor=self.rho_floor,
        )

        # Log-scale the strictly-positive descriptors to compress dynamic range
        # (electron count varies O(1)-O(100), shells span many orders of magnitude).
        d_log = jnp.log(jnp.abs(d) + 1e-12) * jnp.sign(d)
        x = jnp.concatenate([d, d_log])

        h = x
        for feat in self.hidden:
            h = self.act_fn(nn.Dense(feat)(h))
        out = nn.Dense(1)(h).squeeze()

        # Learnable scalar shift so the net can absorb a constant offset
        # without having to fight softplus saturation. Init small (0.001) so
        # it starts as essentially the original architecture.
        shift = self.param(
            "energy_shift",
            nn.initializers.constant(0.001),
            (),
        )

        return -self.scale * nn.softplus(out) + shift


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Dummy system: 2 atoms, 500 grid points, 3 radial shells
    # -------------------------------------------------------------------------

    key = jax.random.PRNGKey(0)

    n_grid = 500
    n_atom = 2
    alphas = (0.5, 1.0, 2.0)

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Electron density: positive, integrates to ~n_electron
    rho = jax.random.uniform(k1, (n_grid,), minval=0.01, maxval=0.5)

    # Grid weights: uniform over a 10-bohr box
    box = 10.0
    grid_weights = jnp.full((n_grid,), box**3 / n_grid)

    # Random grid and atom positions inside the box
    grid_coords = jax.random.uniform(k2, (n_grid, 3), minval=-box / 2, maxval=box / 2)
    atom_coords = jax.random.uniform(k3, (n_atom, 3), minval=-1.0, maxval=1.0)

    # -------------------------------------------------------------------------
    # Build model and initialise parameters
    # -------------------------------------------------------------------------
    model = DescriptorXC(
        hidden=(64, 32),
        alphas=alphas,
        scale=1.0,
        rho_floor=1e-10,
    )

    params = model.init(k4, rho, grid_coords, grid_weights, atom_coords)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {n_params}")

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    exc = model.apply(params, rho, grid_coords, grid_weights, atom_coords)
    print(f"E_xc = {exc:.6f} Ha")

    # -------------------------------------------------------------------------
    # Gradient of E_xc w.r.t. rho  (vrho)
    # -------------------------------------------------------------------------
    vrho = jax.grad(
        lambda r: model.apply(params, r, grid_coords, grid_weights, atom_coords)
    )(rho)
    print(f"vrho  mean={vrho.mean():.4e}  std={vrho.std():.4e}  shape={vrho.shape}")

    # -------------------------------------------------------------------------
    # Descriptor vector
    # -------------------------------------------------------------------------
    d = compute_descriptors(rho, grid_coords, grid_weights, atom_coords, alphas=alphas)
    print(f"Descriptor vector ({d.shape[0]} dims): {d}")

