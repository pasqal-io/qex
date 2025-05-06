"""
Masked functions for SCF for the vectorized 3D version of the code.
"""

from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array
from jax import random
from loguru import logger
from pyscfad import backend

from qedft.train.td import patch_pyscfad
from qedft.train.td.generalized_eigensolver import generalized_eigh
from qedft.train.td.generalized_eigensolver_masked import masked_generalized_eigh
from qedft.train.td.jax_diis import apply_diis, initialize_diis

patch_pyscfad()

logger.info(f"PySCFAD backend: {backend.backend()}")

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Constants
ALL_MODELS = ["density", "density_and_gradient", "kasim"]
NONLOCAL_MODELS = ["density"]
LOCAL_MODELS = ["density_and_gradient", "kasim"]

CONFIG = {
    "experiment_name": "test_vmap",
    "rng": 42,
    "is_global_xc": True,
    "exp_model": "density",
    "hidden_layers": [128, 128, 128, 128, 128],
    "act_fn": "gelu",
    "output_scale": 1e-3,
    "molecule": "H2",
    "basis": "631g",
    "units": "Ang",
    "method": "ccsd",
    "grid_density": 0,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "n_iterations": 1000,
    "checkpoint": 10,
    "batch_size": 5,
    "energy_weight": 1.0,
    "density_weight": 1.0,
    "max_cycle": 15,
    "diis_max_vec": 15,
    "diis_min_vec": 2,
    "diis_start_cycle": 1,
    "diis_damping": 0.0,
    "frac_enabled": 1,
    "frac_theta": 0.04,
    "frac_max_steps": 100,
    "vxc_grad_scale": 1.0,
    "platform": "cpu",  # cpu or cuda
    "output_dir": "results/legacy_vmap",
    "grad_clip": 0.5,
}

# Set platform
jax.config.update("jax_platform_name", CONFIG["platform"])


# -------------------------------------------------------------------------------------------------
# NN is initialized here
# -------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        return -self.scale * nn.swish(nn.Dense(self.features[-1])(x))


network = MLP(features=CONFIG["hidden_layers"], act_fn=nn.gelu)
params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(1192))
print(f"params: {params}")
print(f"network.apply(params, jnp.ones((1192))): {network.apply(params, jnp.ones((1192)))}")


# -------------------------------------------------------------------------------------------------
# Original functions
# -------------------------------------------------------------------------------------------------


def get_exc_and_vrho(
    params: dict,
    rho: Array,
    network: nn.Module,
    **kwargs,
) -> tuple[Array, Array]:
    to_stack = (rho,)
    for key, value in kwargs.items():
        if key == "params_grid_coords":
            x, y, z = value[:, 0], value[:, 1], value[:, 2]
            to_stack += (x, y, z)
        else:
            to_stack += (value,)

    # rho_and_grad = jnp.stack(to_stack, axis=1)
    rho_and_grad = rho
    print(f"rho_and_grad: {rho_and_grad}")
    exc, vjp_fn = jax.vjp(lambda x: jnp.sum(network.apply(params, x).squeeze()), rho_and_grad)
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho[:] * CONFIG["vxc_grad_scale"]


def eval_xc_custom(
    xc_code: str,
    rho: Array,
    spin=0,
    relativity=0,
    deriv=1,
    verbose=None,
    params=None,
    params_grid_coords=None,
    method="density",
):
    if method != "density" or deriv != 1:
        raise ValueError("Only density method with deriv=1 supported")

    if params_grid_coords is not None:
        exc, vrho = get_exc_and_vrho(params, rho, network, params_grid_coords=params_grid_coords)
    else:
        exc, vrho = get_exc_and_vrho(params, rho, network)

    return exc, (vrho, None, None, None), None, None


def get_veff_jax(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
) -> tuple[Array, Array, Array]:
    """JAX implementation of get_veff."""
    # Coulomb matrix
    J = jnp.einsum("ijkl,kl->ij", eri, dm)

    # XC potential
    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid)
    exc, (vrho, _, _, _), _, _ = eval_xc_custom("", rho, params=params)
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid, grid_weights * vrho, ao_grid)

    return J + Vxc, jnp.sum(exc * rho * grid_weights), J


def energy_tot_jax(dm: Array, h1e: Array, J: Array, exc_energy: Array, energy_nuc: float) -> Array:
    """JAX implementation of total energy calculation."""
    e_one = jnp.einsum("ij,ji->", dm, h1e)
    e_hartree = 0.5 * jnp.einsum("ij,ij->", dm, J)
    return e_one + e_hartree + exc_energy + energy_nuc


def get_ao_value(mol, coords: Array) -> Array:
    """Get AO values at given coordinates."""
    deriv = 0
    feval = "GTOval_cart_deriv%d" % deriv if mol.cart else "GTOval_sph_deriv%d" % deriv
    return mol.eval_gto(feval, coords)


def get_occ(nelectron: int, mo_energy: Array | None = None) -> Array:
    """Get occupation numbers for molecular orbitals."""
    e_idx = jnp.argsort(mo_energy)
    nocc = nelectron // 2

    # Create array of indices
    idx = jnp.arange(mo_energy.shape[0])
    # Use jnp.where with rank condition instead of array slicing
    mo_occ = jnp.where(idx < nocc, 2.0, 0.0)
    # Permute back to original energy ordering
    return mo_occ[jnp.argsort(e_idx)]


def make_rdm1(mo_coeff: Array, mo_occ: Array) -> Array:
    return jnp.einsum("ij,j,kj->ik", mo_coeff, mo_occ, mo_coeff)


# -------------------------------------------------------------------------------------------------
# Masked functions
# -------------------------------------------------------------------------------------------------


@jax.jit
def get_veff_jax_masked(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    mask: Array,
    params: dict,
) -> tuple[Array, Array, Array]:
    """JAX implementation of get_veff with mask for padded matrices."""
    n = dm.shape[0]

    # Create 2D mask for matrices
    mask_2d = mask[:, None] & mask[None, :]

    # Create 4D mask for ERI tensor
    mask_4d = (
        mask[:, None, None, None]
        & mask[None, :, None, None]
        & mask[None, None, :, None]
        & mask[None, None, None, :]
    )

    # Apply masks to inputs
    dm_masked = jnp.where(mask_2d, dm, 0.0)
    eri_masked = jnp.where(mask_4d, eri, 0.0)

    # Mask AO grid values for density calculation
    ao_grid_masked = jnp.where(mask[None, :], ao_grid, 0.0)

    # Coulomb matrix
    J = jnp.einsum("ijkl,kl->ij", eri_masked, dm_masked)

    # XC potential
    rho = jnp.einsum("gi,ij,gj->g", ao_grid_masked, dm_masked, ao_grid_masked)
    exc, (vrho, _, _, _), _, _ = eval_xc_custom("", rho, params=params)
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid_masked, grid_weights * vrho, ao_grid_masked)

    # Apply mask to final results
    J = jnp.where(mask_2d, J, 0.0)
    Vxc = jnp.where(mask_2d, Vxc, 0.0)

    return J + Vxc, jnp.sum(exc * rho * grid_weights), J


@jax.jit
def energy_tot_jax_masked(
    dm: Array,
    h1e: Array,
    J: Array,
    exc_energy: Array,
    energy_nuc: float,
    mask: Array,
) -> Array:
    """JAX implementation of total energy calculation with mask for padded matrices."""
    # Create 2D mask
    mask_2d = mask[:, None] & mask[None, :]

    # Apply masks
    dm_masked = jnp.where(mask_2d, dm, 0.0)
    h1e_masked = jnp.where(mask_2d, h1e, 0.0)
    J_masked = jnp.where(mask_2d, J, 0.0)

    # Calculate energies with masked matrices
    e_one = jnp.einsum("ij,ji->", dm_masked, h1e_masked)
    e_hartree = 0.5 * jnp.einsum("ij,ij->", dm_masked, J_masked)

    return e_one + e_hartree + exc_energy + energy_nuc


@jax.jit
def get_occ_masked(nelectron: int, mo_energy: Array, mask: Array) -> Array:
    """Get occupation numbers for molecular orbitals with mask."""
    # Identify padded eigenvalues (set to large value to keep them unoccupied)
    mo_energy_masked = jnp.where(mask, mo_energy, 1e10)

    # Sort the masked eigenvalues
    e_idx = jnp.argsort(mo_energy_masked)

    # Determine occupation based on electron count
    nocc = nelectron // 2

    # Create array of indices
    idx = jnp.arange(mo_energy.shape[0])

    # Use jnp.where with rank condition
    mo_occ = jnp.where((idx < nocc) & mask[e_idx], 2.0, 0.0)

    # Permute back to original energy ordering
    return mo_occ[jnp.argsort(e_idx)]


@jax.jit
def make_rdm1_masked(mo_coeff: Array, mo_occ: Array, mask: Array) -> Array:
    """Create density matrix from MO coefficients and occupations with mask."""
    # Apply mask to inputs
    mo_coeff_masked = jnp.where(mask[:, None], mo_coeff, 0.0)
    mo_occ_masked = jnp.where(mask, mo_occ, 0.0)

    # Create density matrix
    dm = jnp.einsum("ij,j,kj->ik", mo_coeff_masked, mo_occ_masked, mo_coeff_masked)

    # Apply mask to result
    mask_2d = mask[:, None] & mask[None, :]

    return jnp.where(mask_2d, dm, 0.0)


# -------------------------------------------------------------------------------------------------
# Generalized eigenvalue solver
# -------------------------------------------------------------------------------------------------


@jax.jit
def masked_generalized_eigh_stable1(fock: Array, s1e: Array, mask: Array) -> tuple[Array, Array]:
    """Numerically stable masked generalized eigenvalue solver.

    This function improves gradient stability by better handling of the masked regions.
    """
    # Create 2D mask
    mask_2d = mask[:, None] & mask[None, :]

    # Apply mask with high values in diagonal for padded region
    large_diagonal = jnp.eye(fock.shape[0]) * 1e-12
    fock_masked = jnp.where(mask_2d, fock, large_diagonal)

    # For overlap matrix, use identity in padded region
    s1e_masked = jnp.where(mask_2d, s1e, jnp.eye(s1e.shape[0]) * 1e-12)

    # Ensure s1e_masked is positive definite in padded region
    s1e_masked = s1e_masked + jnp.where(mask_2d, 0.0, jnp.eye(s1e.shape[0]) * 1e-12)

    # Solve the generalized eigenvalue problem
    mo_energy, mo_coeff = generalized_eigh(fock_masked, s1e_masked)

    # Apply mask to eigenvalues to ensure padded values are high
    mo_energy = jnp.where(mask, mo_energy, 1e-12)

    # Apply mask to eigenvectors
    mo_coeff = jnp.where(mask_2d, mo_coeff, 1e-12)

    return mo_energy, mo_coeff


@jax.jit
def masked_generalized_eigh_stable2(fock: Array, s1e: Array, mask: Array) -> tuple[Array, Array]:
    """Numerically stable masked generalized eigenvalue solver with better gradient handling."""
    # Create 2D mask
    mask_2d = mask[:, None] & mask[None, :]

    # Apply mask with stable values in padded region
    # Use a small but non-zero value for stability
    eps = 1e-12
    large_diagonal = jnp.eye(fock.shape[0]) * eps

    # Mask the Fock matrix
    fock_masked = jnp.where(mask_2d, fock, large_diagonal)

    # For overlap matrix, use identity in padded region but ensure positive definiteness
    s1e_masked = jnp.where(mask_2d, s1e, jnp.eye(s1e.shape[0]))
    s1e_masked = s1e_masked + eps * jnp.eye(s1e.shape[0])  # Ensure positive definite

    # Solve the generalized eigenvalue problem
    mo_energy, mo_coeff = generalized_eigh(fock_masked, s1e_masked)

    # Apply mask to eigenvalues and eigenvectors in a differentiable way
    mo_energy = jnp.where(mask, mo_energy, eps)
    mo_coeff = jnp.where(mask[:, None], mo_coeff, eps)

    # Normalize the eigenvectors (important for gradient stability)
    norm = jnp.sqrt(
        jnp.einsum("ij,ij->i", mo_coeff, jnp.einsum("ij,jk->ik", s1e_masked, mo_coeff)),
    )
    mo_coeff = mo_coeff / norm[:, None]

    return mo_energy, mo_coeff


@jax.jit
def masked_generalized_eigh_stable3(fock: Array, s1e: Array, mask: Array) -> tuple[Array, Array]:
    """More robust version with gradient stability fixes."""
    EPS = 1e-12
    mask_2d = mask[:, None] & mask[None, :]

    # Regularize matrices
    fock_reg = jnp.where(mask_2d, fock, 0.0)
    s1e_reg = jnp.where(mask_2d, s1e, 0.0)

    # Add small diagonal to ensure positive definiteness
    fock_reg = fock_reg + EPS * jnp.eye(fock.shape[0])
    s1e_reg = s1e_reg + EPS * jnp.eye(s1e.shape[0])

    # Solve with safe normalization
    mo_energy, mo_coeff = generalized_eigh(fock_reg, s1e_reg)

    # Apply mask safely
    mo_energy = jnp.where(mask, mo_energy, 1e10)  # Large value for unoccupied
    mo_coeff = jnp.where(mask[:, None], mo_coeff, 0.0)

    # Normalize eigenvectors (critical for gradients)
    norm = jnp.sqrt(jnp.einsum("ij,ij->i", mo_coeff, jnp.einsum("ij,jk->ik", s1e_reg, mo_coeff)))
    mo_coeff = mo_coeff / jnp.maximum(norm[:, None], EPS)

    return mo_energy, mo_coeff


@jax.jit
def masked_generalized_eigh_stable(fock: Array, s1e: Array, mask: Array) -> tuple[Array, Array]:
    """More robust version with gradient stability fixes."""
    EPS = 1e-12
    mask_2d = mask[:, None] & mask[None, :]

    # Regularize matrices
    fock_reg = jnp.where(mask_2d, fock, 0.0)
    s1e_reg = jnp.where(mask_2d, s1e, 0.0)

    # Add small diagonal to ensure positive definiteness
    fock_reg = fock_reg + EPS * jnp.eye(fock.shape[0])
    s1e_reg = s1e_reg + EPS * jnp.eye(s1e.shape[0])

    # Solve with safe normalization
    mo_energy, mo_coeff = generalized_eigh(fock_reg, s1e_reg)

    # Apply mask safely
    mo_energy = jnp.where(mask, mo_energy, 1e10)  # Large value for unoccupied
    mo_coeff = jnp.where(mask[:, None], mo_coeff, 0.0)

    # Normalize eigenvectors (critical for gradients)
    norm = jnp.sqrt(jnp.einsum("ij,ij->i", mo_coeff, jnp.einsum("ij,jk->ik", s1e_reg, mo_coeff)))
    mo_coeff = mo_coeff / jnp.maximum(norm[:, None], EPS)

    return mo_energy, mo_coeff


@jax.jit
def masked_generalized_eigh_stable(
    fock: Array,
    s1e: Array,
    mask: Array,
    *,
    eps: float = 1e-12,
    big: float = 1e5,
    break_degeneracy: bool = True,
    stop_grad: bool = True,
) -> tuple[Array, Array]:
    """
    Generalised eigh that is robust to the padded (zero) block.

    Parameters
    ----------
    fock, s1e : [n,n] Hermitian
    mask      : [n]   True for *physical* orbitals
    eps       : float tiny positive number
    big       : float base shift used for padded orbitals
    break_degeneracy : bool
        If True add a different shift to every padded orbital so that
        λ_i ≠ λ_j for i≠j.
    stop_grad : bool
        If True cut the backward flow through the padded sub-space.
    """
    n = fock.shape[0]
    pad = (~mask).astype(fock.dtype)

    # ------------------------------------------------------------------
    # 1. make the matrices positive-definite & padded-orbital-huge
    # ------------------------------------------------------------------
    base_shift = big * pad
    if break_degeneracy:
        # tiny, but *different* for every padded orbital
        base_shift = base_shift + pad * eps * jnp.arange(n)

    fock_reg = fock + jnp.diag(base_shift) + eps * jnp.eye(n)
    s1e_reg = s1e + eps * jnp.eye(n)

    # ------------------------------------------------------------------
    # 2. solve the eigenproblem
    # ------------------------------------------------------------------
    # lam, vec = jnp.linalg.eigh(fock_reg, s1e_reg)
    lam, vec = generalized_eigh(fock_reg, s1e_reg)

    # ------------------------------------------------------------------
    # 3. optionally cut the gradients flowing through the padded block
    # ------------------------------------------------------------------
    if stop_grad:
        lam = jnp.where(mask, lam, jax.lax.stop_gradient(lam))
        vec = jnp.where(mask[:, None], vec, jax.lax.stop_gradient(vec))

    return lam, vec


# ----------------------------------------------------------------------
# 1.  A single, globally safe eigensolver
# ----------------------------------------------------------------------


def masked_generalized_eigh_stable(
    fock: Array,
    s1e: Array,
    mask: Array,
    eps: float = 1e-12,  # numerical safety
    big: float = 1e5,
):  # large shift for padded orbs
    """Eigensolver whose backward pass never divides by zero."""
    n = fock.shape[0]

    # --- regularise -----------------------------------------------------------------
    pad = (~mask).astype(fock.dtype)
    uniq_shift = big + eps * jnp.arange(n)  # breaks the degeneracy
    fock_reg = fock + jnp.diag(pad * uniq_shift) + eps * jnp.eye(n)
    s1e_reg = s1e + eps * jnp.eye(n)

    # --- solve ----------------------------------------------------------------------
    lam, vec = generalized_eigh(fock_reg, s1e_reg)

    # --- lock the padded sub-space --------------------------------------------------
    lam = jnp.where(mask, lam, jax.lax.stop_gradient(lam))
    vec = jnp.where(mask[:, None], vec, jax.lax.stop_gradient(vec))
    return lam, vec


# -------------------------------------------------------------------------------------------------
# Maksed density matrix construction
# -------------------------------------------------------------------------------------------------


@jax.jit
def make_rdm1_masked(mo_coeff: Array, mo_occ: Array, mask: Array) -> Array:
    """More numerically stable density matrix construction."""
    EPS = 1e-12

    # Apply masks safely
    mo_coeff_safe = jnp.where(mask[:, None], mo_coeff, 0.0)
    mo_occ_safe = jnp.where(mask, mo_occ, 0.0)

    # Construct density matrix with safe operations
    dm = jnp.einsum(
        "ij,j,kj->ik",
        mo_coeff_safe,
        mo_occ_safe,
        mo_coeff_safe,
    )

    # Symmetrize and mask
    dm_sym = 0.5 * (dm + dm.T)
    mask_2d = mask[:, None] & mask[None, :]
    return jnp.where(mask_2d, dm_sym, 0.0)


@jax.jit
def get_veff_jax_masked(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    mask: Array,
    params: dict,
) -> tuple[Array, Array, Array]:
    """More numerically stable version of get_veff_jax_masked."""
    n = dm.shape[0]

    # Create masks
    mask_2d = mask[:, None] & mask[None, :]
    mask_4d = (
        mask[:, None, None, None]
        & mask[None, :, None, None]
        & mask[None, None, :, None]
        & mask[None, None, None, :]
    )

    # Apply masks with stable values
    eps = 1e-12
    dm_masked = jnp.where(mask_2d, dm, eps)
    eri_masked = jnp.where(mask_4d, eri, eps)
    ao_grid_masked = jnp.where(mask[None, :], ao_grid, eps)

    # Coulomb matrix with safe operations
    J = jnp.einsum("ijkl,kl->ij", eri_masked, dm_masked)

    # XC potential with safe operations
    rho = jnp.einsum("gi,ij,gj->g", ao_grid_masked, dm_masked, ao_grid_masked)
    # Add small constant to rho for numerical stability
    rho = rho + eps

    exc, (vrho, _, _, _), _, _ = eval_xc_custom("", rho, params=params)
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid_masked, grid_weights * vrho, ao_grid_masked)

    # Apply final masks
    J = jnp.where(mask_2d, J, 0.0)
    Vxc = jnp.where(mask_2d, Vxc, 0.0)

    return J + Vxc, jnp.sum(exc * rho * grid_weights), J


# -------------------------------------------------------------------------------------------------
# Test functions
# -------------------------------------------------------------------------------------------------


def test_get_veff_jax_masked():
    """Test the masked get_veff_jax function."""
    print("\nTesting get_veff_jax_masked...")

    # Create test data
    real_size, pad_size = 3, 5
    n_grid = 1192
    rng = random.PRNGKey(43)
    key1, key2, key3 = random.split(rng, 3)

    # Create original matrices
    dm_small = random.normal(key1, (real_size, real_size))
    dm_small = (dm_small + dm_small.T) / 2  # Make symmetric

    # Create ERI tensor (symmetric)
    eri_small = random.normal(key2, (real_size, real_size, real_size, real_size))
    eri_small = (
        eri_small
        + eri_small.transpose((1, 0, 3, 2))
        + eri_small.transpose((2, 3, 0, 1))
        + eri_small.transpose((3, 2, 1, 0))
    ) / 4

    # Create AO grid values
    ao_grid_small = random.normal(key3, (n_grid, real_size))
    grid_weights = jnp.ones(n_grid) / n_grid

    # Create test parameters
    # params = {
    #     'alpha': 0.8,
    #     'beta': 0.2
    # }

    # Get reference results
    ref_veff, ref_exc, ref_J = get_veff_jax(
        dm_small,
        eri_small,
        ao_grid_small,
        grid_weights,
        params,
    )

    # Create padded matrices
    dm_padded = jnp.zeros((pad_size, pad_size))
    dm_padded = dm_padded.at[:real_size, :real_size].set(dm_small)

    eri_padded = jnp.zeros((pad_size, pad_size, pad_size, pad_size))
    eri_padded = eri_padded.at[:real_size, :real_size, :real_size, :real_size].set(eri_small)

    ao_grid_padded = jnp.zeros((n_grid, pad_size))
    ao_grid_padded = ao_grid_padded.at[:, :real_size].set(ao_grid_small)

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Test with padding
    test_veff, test_exc, test_J = get_veff_jax_masked(
        dm_padded,
        eri_padded,
        ao_grid_padded,
        grid_weights,
        mask,
        params,
    )
    # Test old implementation on padded matrices
    test_veff_old, test_exc_old, test_J_old = get_veff_jax(
        dm_padded,
        eri_padded,
        ao_grid_padded,
        grid_weights,
        params,
    )

    # Compare results
    veff_match = jnp.allclose(ref_veff, test_veff[:real_size, :real_size], atol=1e-5)
    print(f"ref_veff: {ref_veff}")
    print(f"test_veff: {test_veff}")
    print(f"test_veff_old: {test_veff_old}")
    exc_match = jnp.allclose(ref_exc, test_exc, atol=1e-5)
    J_match = jnp.allclose(ref_J, test_J[:real_size, :real_size], atol=1e-5)

    print(f"Veff matches reference: {veff_match}")
    print(f"Exc matches reference: {exc_match}")
    print(f"J matches reference: {J_match}")

    return veff_match and exc_match and J_match


def test_energy_tot_jax_masked():
    """Test the masked energy_tot_jax function."""
    print("\nTesting energy_tot_jax_masked...")

    # Create test data
    real_size, pad_size = 3, 5
    rng = random.PRNGKey(44)
    key1, key2, key3 = random.split(rng, 3)

    # Create original matrices
    dm_small = random.normal(key1, (real_size, real_size))
    dm_small = (dm_small + dm_small.T) / 2  # Make symmetric
    h1e_small = random.normal(key2, (real_size, real_size))
    h1e_small = (h1e_small + h1e_small.T) / 2  # Make symmetric
    J_small = random.normal(key3, (real_size, real_size))
    J_small = (J_small + J_small.T) / 2  # Make symmetric

    exc_energy = 1.5
    energy_nuc = 10.0

    # Get reference results
    ref_energy = energy_tot_jax(dm_small, h1e_small, J_small, exc_energy, energy_nuc)

    # Create padded matrices
    dm_padded = jnp.zeros((pad_size, pad_size))
    dm_padded = dm_padded.at[:real_size, :real_size].set(dm_small)

    h1e_padded = jnp.zeros((pad_size, pad_size))
    h1e_padded = h1e_padded.at[:real_size, :real_size].set(h1e_small)

    J_padded = jnp.zeros((pad_size, pad_size))
    J_padded = J_padded.at[:real_size, :real_size].set(J_small)

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Test with padding
    test_energy = energy_tot_jax_masked(
        dm_padded,
        h1e_padded,
        J_padded,
        exc_energy,
        energy_nuc,
        mask,
    )

    # Compare results
    energy_match = jnp.allclose(ref_energy, test_energy, atol=1e-5)

    print(f"Energy matches reference: {energy_match}")
    print(f"Reference energy: {ref_energy}")
    print(f"Masked energy: {test_energy}")

    return energy_match


def test_get_occ_masked():
    """Test the masked get_occ function."""
    print("\nTesting get_occ_masked...")

    # Use global network and params

    # Create test data
    real_size, pad_size = 5, 8
    nelectron = 6  # 3 doubly occupied orbitals

    # Create energy array
    mo_energy_small = jnp.array([-5.0, -2.0, -1.0, 0.5, 2.0])

    # Get reference results
    ref_occ = get_occ(nelectron, mo_energy_small)

    # Create padded array
    mo_energy_padded = jnp.zeros(pad_size)
    mo_energy_padded = mo_energy_padded.at[:real_size].set(mo_energy_small)

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Test with padding
    test_occ = get_occ_masked(nelectron, mo_energy_padded, mask)

    # Compare results
    occ_match = jnp.allclose(ref_occ, test_occ[:real_size], atol=1e-5)
    padding_zeros = jnp.allclose(test_occ[real_size:], jnp.zeros(pad_size - real_size))

    print(f"Occupation matches reference: {occ_match}")
    print(f"Padded values are zero: {padding_zeros}")
    print(f"Reference occ: {ref_occ}")
    print(f"Masked occ: {test_occ[:real_size]}")

    return occ_match and padding_zeros


def test_make_rdm1_masked():
    """Test the masked make_rdm1 function."""
    print("\nTesting make_rdm1_masked...")

    # Create test data
    real_size, pad_size = 3, 5
    rng = random.PRNGKey(45)

    # Create MO coefficients and occupations
    mo_coeff_small = random.normal(rng, (real_size, real_size))
    mo_occ_small = jnp.array([2.0, 2.0, 0.0])  # 2 occupied orbitals

    # Get reference results
    ref_dm = make_rdm1(mo_coeff_small, mo_occ_small)

    # Create padded arrays
    mo_coeff_padded = jnp.zeros((pad_size, pad_size))
    mo_coeff_padded = mo_coeff_padded.at[:real_size, :real_size].set(mo_coeff_small)

    mo_occ_padded = jnp.zeros(pad_size)
    mo_occ_padded = mo_occ_padded.at[:real_size].set(mo_occ_small)

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Test with padding
    test_dm = make_rdm1_masked(mo_coeff_padded, mo_occ_padded, mask)

    # Compare results
    dm_match = jnp.allclose(ref_dm, test_dm[:real_size, :real_size], atol=1e-5)
    padding_zeros = jnp.allclose(
        test_dm[real_size:, :],
        jnp.zeros((pad_size - real_size, pad_size)),
    )

    print(f"Density matrix matches reference: {dm_match}")
    print(f"Padded values are zero: {padding_zeros}")

    return dm_match and padding_zeros


def test_all():
    """Run all tests."""
    print("Running all tests for masked functions...")

    # Call each test function
    veff_test = test_get_veff_jax_masked()
    energy_test = test_energy_tot_jax_masked()
    occ_test = test_get_occ_masked()
    rdm1_test = test_make_rdm1_masked()

    # Check if all tests passed
    all_passed = veff_test and energy_test and occ_test and rdm1_test

    print("\nSummary:")
    print(f"Veff test passed: {veff_test}")
    print(f"Energy test passed: {energy_test}")
    print(f"Occupation test passed: {occ_test}")
    print(f"Density matrix test passed: {rdm1_test}")
    print(f"Overall test result: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


@partial(
    jax.jit,
    static_argnames=(
        "max_cycle",
        "diis_max_vec",
        "diis_min_vec",
        "diis_start_cycle",
        "diis_damping",
    ),
)
def _scf_test_non_padded(
    params: dict,
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    s1e: Array,
    h1e: Array,
    energy_nuc: float,
    nelectron: int,
    max_cycle: int = 15,
    diis_max_vec: int = 15,
    diis_min_vec: int = 2,
    diis_start_cycle: int = 1,
    diis_damping: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Standard SCF loop for testing."""
    vhf, exc_energy, J = get_veff_jax(dm, eri, ao_grid, grid_weights, params)
    e_tot = energy_tot_jax(dm, h1e, J, exc_energy, energy_nuc)

    diis_state = initialize_diis(diis_max_vec)
    energies = []
    dms = []

    for cycle in range(max_cycle):
        fock = h1e + vhf

        if cycle >= diis_start_cycle:
            fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                s1e,
                diis_max_vec,
                diis_min_vec,
                diis_damping,
            )

        mo_energy, mo_coeff = generalized_eigh(fock, s1e)
        mo_occ = get_occ(nelectron, mo_energy)
        dm = make_rdm1(mo_coeff, mo_occ)

        vhf, exc_energy, J = get_veff_jax(dm, eri, ao_grid, grid_weights, params)
        e_tot = energy_tot_jax(dm, h1e, J, exc_energy, energy_nuc)

        energies = energies + [e_tot]
        dms = dms + [dm]

    return e_tot, dm, jnp.array(energies)


@partial(
    jax.jit,
    static_argnames=(
        "max_cycle",
        "diis_max_vec",
        "diis_min_vec",
        "diis_start_cycle",
        "diis_damping",
    ),
)
def _scf_test_padded(
    params: dict,
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    s1e: Array,
    h1e: Array,
    energy_nuc: float,
    nelectron: int,
    mask: Array,
    max_cycle: int = 15,
    diis_max_vec: int = 15,
    diis_min_vec: int = 2,
    diis_start_cycle: int = 1,
    diis_damping: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Padded SCF loop using stabilized generalized eigensolver."""
    vhf, exc_energy, J = get_veff_jax_masked(dm, eri, ao_grid, grid_weights, mask, params)
    e_tot = energy_tot_jax_masked(dm, h1e, J, exc_energy, energy_nuc, mask)

    diis_state = initialize_diis(diis_max_vec)
    energies = []
    dms = []

    for cycle in range(max_cycle):
        fock = h1e + vhf

        if cycle >= diis_start_cycle:
            fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                s1e,
                diis_max_vec,
                diis_min_vec,
                diis_damping,
            )

        # Use the stabilized eigensolver
        mo_energy, mo_coeff = masked_generalized_eigh(fock, s1e, mask)
        mo_occ = get_occ_masked(nelectron, mo_energy, mask)
        dm = make_rdm1_masked(mo_coeff, mo_occ, mask)

        vhf, exc_energy, J = get_veff_jax_masked(dm, eri, ao_grid, grid_weights, mask, params)
        e_tot = energy_tot_jax_masked(dm, h1e, J, exc_energy, energy_nuc, mask)

        energies = energies + [e_tot]
        dms = dms + [dm]

    return e_tot, dm, jnp.array(energies)


def compare_padded_vs_non_padded():
    """Compare padded vs non-padded SCF calculations."""
    print("\nComparing padded vs non-padded SCF calculations...")

    # Create test data
    real_size, pad_size = 10, 16
    n_grid = 1192
    nelectron = 10
    energy_nuc = 5.0

    # Create random keys
    rng = random.PRNGKey(42)
    keys = random.split(rng, 5)

    # Create original matrices
    dm_small = random.normal(keys[0], (real_size, real_size))
    dm_small = (dm_small + dm_small.T) / 2  # Make symmetric

    eri_small = random.normal(keys[1], (real_size, real_size, real_size, real_size))
    # Make ERI tensor symmetric
    eri_small = (
        eri_small
        + eri_small.transpose((1, 0, 3, 2))
        + eri_small.transpose((2, 3, 0, 1))
        + eri_small.transpose((3, 2, 1, 0))
    ) / 4

    ao_grid_small = random.normal(keys[2], (n_grid, real_size))
    grid_weights = jnp.ones(n_grid) / n_grid

    h1e_small = random.normal(keys[3], (real_size, real_size))
    h1e_small = (h1e_small + h1e_small.T) / 2  # Make symmetric

    s1e_small = random.normal(keys[4], (real_size, real_size))
    s1e_small = (s1e_small + s1e_small.T) / 2  # Make symmetric
    s1e_small = s1e_small + jnp.eye(real_size) * 5  # Make positive definite

    # Original sizes print
    print(f"Original sizes:")
    print(f"    dm_small: {dm_small.shape}")
    print(f"    eri_small: {eri_small.shape}")
    print(f"    ao_grid_small: {ao_grid_small.shape}")
    print(f"    h1e_small: {h1e_small.shape}")
    print(f"    s1e_small: {s1e_small.shape}")

    # Create the network parameters (use global from scf_functions_masked.py)
    network = MLP(features=CONFIG["hidden_layers"], act_fn=nn.gelu)
    params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(1192))

    # Create padded matrices
    dm_padded = jnp.zeros((pad_size, pad_size))
    dm_padded = dm_padded.at[:real_size, :real_size].set(dm_small)

    eri_padded = jnp.zeros((pad_size, pad_size, pad_size, pad_size))
    eri_padded = eri_padded.at[:real_size, :real_size, :real_size, :real_size].set(eri_small)

    ao_grid_padded = jnp.zeros((n_grid, pad_size))
    ao_grid_padded = ao_grid_padded.at[:, :real_size].set(ao_grid_small)

    h1e_padded = jnp.zeros((pad_size, pad_size))
    h1e_padded = h1e_padded.at[:real_size, :real_size].set(h1e_small)

    s1e_padded = jnp.zeros((pad_size, pad_size))
    s1e_padded = s1e_padded.at[:real_size, :real_size].set(s1e_small)

    # Padded sizes print
    print(f"Padded sizes:")
    print(f"    dm_padded: {dm_padded.shape}")
    print(f"    eri_padded: {eri_padded.shape}")
    print(f"    ao_grid_padded: {ao_grid_padded.shape}")
    print(f"    h1e_padded: {h1e_padded.shape}")
    print(f"    s1e_padded: {s1e_padded.shape}")

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)
    print(f"Mask: {mask}")

    # Run non-padded SCF
    e_tot_small, dm_final_small, energies_small = _scf_test_non_padded(
        params,
        dm_small,
        eri_small,
        ao_grid_small,
        grid_weights,
        s1e_small,
        h1e_small,
        energy_nuc,
        nelectron,
    )

    # Run padded SCF
    e_tot_padded, dm_final_padded, energies_padded = _scf_test_padded(
        params,
        dm_padded,
        eri_padded,
        ao_grid_padded,
        grid_weights,
        s1e_padded,
        h1e_padded,
        energy_nuc,
        nelectron,
        mask,
    )

    # Run non-padded SCF on padded matrices (should be wrong)
    e_tot_padded_small, dm_final_padded_small, energies_padded_small = _scf_test_non_padded(
        params,
        dm_padded,
        eri_padded,
        ao_grid_padded,
        grid_weights,
        s1e_padded,
        h1e_padded,
        energy_nuc,
        nelectron,
    )

    # Compare results
    print("\nResults comparison:")
    print(f"Final energy (non-padded): {e_tot_small}")
    print(f"Final energy (padded): {e_tot_padded}")
    print(f"Final energy (padded with non-padded SCF): {e_tot_padded_small}")
    energy_match = jnp.allclose(e_tot_small, e_tot_padded, atol=1e-5)
    print(f"Energy matches: {energy_match}")

    # Compare energy convergence
    energy_diff = jnp.abs(energies_small - energies_padded)
    print("\nEnergy convergence difference at each iteration:")
    for i, diff in enumerate(energy_diff):
        print(f"Iteration {i}: {diff:.8e}")

    # Compare density matrices
    dm_match = jnp.allclose(dm_final_small, dm_final_padded[:real_size, :real_size], atol=1e-5)
    print(f"\nFinal density matrix matches: {dm_match}")

    if not dm_match:
        dm_diff = jnp.abs(dm_final_small - dm_final_padded[:real_size, :real_size])
        max_diff = jnp.max(dm_diff)
        avg_diff = jnp.mean(dm_diff)
        print(f"Maximum difference in density matrix: {max_diff:.8e}")
        print(f"Average difference in density matrix: {avg_diff:.8e}")

    return energy_match and dm_match


def compare_gradients():
    """Compare gradients between regular and padded SCF implementations."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from jax import random

    # Create test data (similar to compare_padded_vs_non_padded function)
    real_size, pad_size = 10, 16
    n_grid = 1192
    nelectron = 10
    energy_nuc = 5.0

    # Create random keys
    rng = random.PRNGKey(42)
    keys = random.split(rng, 5)

    # Create original matrices
    dm_small = random.normal(keys[0], (real_size, real_size))
    dm_small = (dm_small + dm_small.T) / 2  # Make symmetric

    eri_small = random.normal(keys[1], (real_size, real_size, real_size, real_size))
    # Make ERI tensor symmetric
    eri_small = (
        eri_small
        + eri_small.transpose((1, 0, 3, 2))
        + eri_small.transpose((2, 3, 0, 1))
        + eri_small.transpose((3, 2, 1, 0))
    ) / 4

    ao_grid_small = random.normal(keys[2], (n_grid, real_size))
    grid_weights = jnp.ones(n_grid) / n_grid

    h1e_small = random.normal(keys[3], (real_size, real_size))
    h1e_small = (h1e_small + h1e_small.T) / 2  # Make symmetric

    s1e_small = random.normal(keys[4], (real_size, real_size))
    s1e_small = (s1e_small + s1e_small.T) / 2  # Make symmetric
    s1e_small = s1e_small + jnp.eye(real_size) * 5  # Make positive definite

    # Create the network and parameters
    network = MLP(features=[128, 128, 128, 128, 128], act_fn=nn.gelu)
    params = network.init(random.PRNGKey(42), jnp.ones(1192))

    # Create padded matrices
    dm_padded = jnp.ones((pad_size, pad_size)) * 1e-12
    dm_padded = dm_padded.at[:real_size, :real_size].set(dm_small)

    eri_padded = jnp.ones((pad_size, pad_size, pad_size, pad_size)) * 1e-12
    eri_padded = eri_padded.at[:real_size, :real_size, :real_size, :real_size].set(eri_small)

    ao_grid_padded = jnp.ones((n_grid, pad_size)) * 1e-12
    ao_grid_padded = ao_grid_padded.at[:, :real_size].set(ao_grid_small)

    h1e_padded = jnp.ones((pad_size, pad_size)) * 1e-12
    h1e_padded = h1e_padded.at[:real_size, :real_size].set(h1e_small)

    s1e_padded = jnp.ones((pad_size, pad_size)) * 1e-12
    s1e_padded = s1e_padded.at[:real_size, :real_size].set(s1e_small)

    # Create mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Set up mock reference values
    exact_energy = 0.0
    exact_density = jnp.ones(n_grid)
    exact_dm_small = dm_small.copy()
    exact_dm_padded = dm_padded.copy()

    # Define loss functions for both implementations
    def loss_fn_regular(params):
        return _scf_test_non_padded(
            params,
            dm_small,
            eri_small,
            ao_grid_small,
            grid_weights,
            s1e_small,
            h1e_small,
            energy_nuc,
            nelectron,
        )[0]

    def loss_fn_padded(params):
        return _scf_test_padded(
            params,
            dm_padded,
            eri_padded,
            ao_grid_padded,
            grid_weights,
            s1e_padded,
            h1e_padded,
            energy_nuc,
            nelectron,
            mask,
        )[0]

    # Calculate gradients
    print("\nComputing gradients...")
    grad_regular = jax.grad(loss_fn_regular)(params)
    grad_padded = jax.grad(loss_fn_padded)(params)

    # Flatten gradients for comparison
    flat_grad_regular = jax.tree_util.tree_flatten(grad_regular)[0]
    flat_grad_padded = jax.tree_util.tree_flatten(grad_padded)[0]

    # Compare gradients
    print("\nGradient comparison:")
    all_close = True
    for i, (gr, gp) in enumerate(zip(flat_grad_regular, flat_grad_padded)):
        # Check if shapes match
        if gr.shape != gp.shape:
            print(f"Param {i}: Shape mismatch - Regular: {gr.shape}, Padded: {gp.shape}")
            all_close = False
            continue

        # Check for numerical similarity (accounting for zeros)
        if jnp.allclose(gr, gp, atol=1e-5, rtol=1e-5):
            print(f"Param {i}: Gradients match with shape {gr.shape}")
        else:
            rel_diff = jnp.max(jnp.abs(gr - gp)) / (jnp.max(jnp.abs(gr)) + 1e-10)
            print(
                f"Param {i}: Gradients differ - Max abs diff: {jnp.max(jnp.abs(gr - gp)):.6e}, Rel diff: {rel_diff:.6e}, Regular: {gr}, Padded: {gp}",
            )
            all_close = False

    print(f"\nOverall gradient matching: {'SUCCESS' if all_close else 'FAILED'}")
    return all_close


def test_gradients_for_masked_functions():
    """Test gradients for all masked functions with artificial padded data to identify NaN sources."""
    import jax
    import jax.numpy as jnp
    from jax import grad, random

    # Setup test data with padding
    key = random.PRNGKey(42)
    keys = random.split(key, 10)

    # Dimensions
    real_size = 5
    pad_size = 8
    n_grid = 10

    # Create mask for padded region
    mask = jnp.zeros(pad_size, dtype=bool).at[:real_size].set(True)
    mask_2d = mask[:, None] & mask[None, :]

    # Create test arrays
    dm = jnp.ones((pad_size, pad_size)) * 1e-5
    dm = dm.at[:real_size, :real_size].set(random.normal(keys[0], (real_size, real_size)))
    dm = (dm + dm.T) / 2  # Make symmetric

    eri = jnp.ones((pad_size, pad_size, pad_size, pad_size)) * 1e-5
    eri_real = random.normal(keys[1], (real_size, real_size, real_size, real_size))
    eri_real = (
        eri_real
        + eri_real.transpose((1, 0, 3, 2))
        + eri_real.transpose((2, 3, 0, 1))
        + eri_real.transpose((3, 2, 1, 0))
    ) / 4
    eri = eri.at[:real_size, :real_size, :real_size, :real_size].set(eri_real)

    ao_grid = jnp.ones((n_grid, pad_size)) * 1e-5
    ao_grid = ao_grid.at[:, :real_size].set(random.normal(keys[2], (n_grid, real_size)))

    grid_weights = jnp.ones(n_grid) / n_grid

    h1e = jnp.ones((pad_size, pad_size)) * 1e-5
    h1e = h1e.at[:real_size, :real_size].set(random.normal(keys[3], (real_size, real_size)))
    h1e = (h1e + h1e.T) / 2  # Make symmetric

    s1e = jnp.eye(pad_size) * 1e-3
    s1e = s1e.at[:real_size, :real_size].set(
        jnp.eye(real_size) + random.normal(keys[4], (real_size, real_size)) * 0.1,
    )
    s1e = (s1e + s1e.T) / 2  # Make symmetric

    mo_coeff = jnp.ones((pad_size, pad_size)) * 1e-5
    mo_coeff = mo_coeff.at[:real_size, :real_size].set(
        random.normal(keys[5], (real_size, real_size)),
    )

    mo_energy = jnp.ones(pad_size) * 1e-5
    mo_energy = mo_energy.at[:real_size].set(jnp.sort(random.normal(keys[6], (real_size,))))

    mo_occ = jnp.zeros(pad_size)
    mo_occ = mo_occ.at[:3].set(2.0)  # Fill lowest 3 orbitals with 2 electrons each

    J = jnp.ones((pad_size, pad_size)) * 1e-5
    J = J.at[:real_size, :real_size].set(random.normal(keys[7], (real_size, real_size)))
    J = (J + J.T) / 2  # Make symmetric

    exc_energy = jnp.array(0.5)
    energy_nuc = 1.0
    nelectron = 6

    fock = h1e + J

    # Create parameters for network
    params = {"kernel": jnp.ones((10, 1)) * 0.1}

    print("\n=== Testing gradients for masked functions ===")

    # 1. Test get_veff_jax_masked
    print("\nTesting get_veff_jax_masked gradients:")

    def loss_veff(p):
        veff, _, _ = get_veff_jax_masked(dm, eri, ao_grid, grid_weights, mask, p)
        return jnp.sum(veff)

    try:
        grad_veff = grad(loss_veff)(params)
        has_nan = any(jnp.isnan(v).any() for v in jax.tree_util.tree_leaves(grad_veff))
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in get_veff_jax_masked gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 2. Test energy_tot_jax_masked
    print("\nTesting energy_tot_jax_masked gradients:")

    def loss_energy(d):
        return energy_tot_jax_masked(d, h1e, J, exc_energy, energy_nuc, mask)

    try:
        grad_energy = grad(loss_energy)(dm)
        has_nan = jnp.isnan(grad_energy).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in energy_tot_jax_masked gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 3. Test get_occ_masked
    print("\nTesting get_occ_masked gradients:")

    def loss_occ(e):
        return jnp.sum(get_occ_masked(nelectron, e, mask))

    try:
        grad_occ = grad(loss_occ)(mo_energy)
        has_nan = jnp.isnan(grad_occ).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in get_occ_masked gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 4. Test make_rdm1_masked
    print("\nTesting make_rdm1_masked gradients:")

    def loss_rdm1(c):
        return jnp.sum(make_rdm1_masked(c, mo_occ, mask))

    try:
        grad_rdm1 = grad(loss_rdm1)(mo_coeff)
        has_nan = jnp.isnan(grad_rdm1).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in make_rdm1_masked gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 5. Test masked_generalized_eigh_stable
    print("\nTesting masked_generalized_eigh_stable gradients:")

    def loss_eigh(f):
        e, c = masked_generalized_eigh_stable(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh = grad(loss_eigh)(fock)
        has_nan = jnp.isnan(grad_eigh).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh))
            print(f"  NaN positions: {where_nan}")
            print("  NaN found in masked_generalized_eigh_stable gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # Test each variant of the eigensolver
    print("\nTesting masked_generalized_eigh_stable1 gradients:")

    def loss_eigh1(f):
        e, c = masked_generalized_eigh_stable1(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh1 = grad(loss_eigh1)(fock)
        has_nan = jnp.isnan(grad_eigh1).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh1))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    print("\nTesting masked_generalized_eigh_stable2 gradients:")

    def loss_eigh2(f):
        e, c = masked_generalized_eigh_stable2(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh2 = grad(loss_eigh2)(fock)
        has_nan = jnp.isnan(grad_eigh2).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh2))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    print("\nTesting masked_generalized_eigh_stable3 gradients:")

    def loss_eigh3(f):
        e, c = masked_generalized_eigh_stable3(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh3 = grad(loss_eigh3)(fock)
        has_nan = jnp.isnan(grad_eigh3).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh3))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")


def test_gradients_with_zero_padding():
    """Test gradients for all masked functions with zero-padded data to identify NaN sources."""
    import jax
    import jax.numpy as jnp
    from jax import grad, random

    # Setup test data with zero padding
    key = random.PRNGKey(42)
    keys = random.split(key, 10)

    # Dimensions
    real_size = 5
    pad_size = 8
    n_grid = 10

    # Create mask for padded region
    mask = jnp.zeros(pad_size, dtype=bool).at[:real_size].set(True)

    # Create test arrays with exact zeros in padded region
    dm = jnp.zeros((pad_size, pad_size))
    dm = dm.at[:real_size, :real_size].set(random.normal(keys[0], (real_size, real_size)))
    dm = (dm + dm.T) / 2  # Make symmetric

    eri = jnp.zeros((pad_size, pad_size, pad_size, pad_size))
    eri_real = random.normal(keys[1], (real_size, real_size, real_size, real_size))
    eri_real = (
        eri_real
        + eri_real.transpose((1, 0, 3, 2))
        + eri_real.transpose((2, 3, 0, 1))
        + eri_real.transpose((3, 2, 1, 0))
    ) / 4
    eri = eri.at[:real_size, :real_size, :real_size, :real_size].set(eri_real)

    ao_grid = jnp.zeros((n_grid, pad_size))
    ao_grid = ao_grid.at[:, :real_size].set(random.normal(keys[2], (n_grid, real_size)))

    grid_weights = jnp.ones(n_grid) / n_grid

    h1e = jnp.zeros((pad_size, pad_size))
    h1e = h1e.at[:real_size, :real_size].set(random.normal(keys[3], (real_size, real_size)))
    h1e = (h1e + h1e.T) / 2  # Make symmetric

    # For s1e, use identity only in real region to avoid singularity issues
    s1e = jnp.zeros((pad_size, pad_size))
    s1e = s1e.at[:real_size, :real_size].set(
        jnp.eye(real_size) + random.normal(keys[4], (real_size, real_size)) * 0.1,
    )
    s1e = (s1e + s1e.T) / 2  # Make symmetric

    mo_coeff = jnp.zeros((pad_size, pad_size))
    mo_coeff = mo_coeff.at[:real_size, :real_size].set(
        random.normal(keys[5], (real_size, real_size)),
    )

    mo_energy = jnp.zeros(pad_size)
    mo_energy = mo_energy.at[:real_size].set(jnp.sort(random.normal(keys[6], (real_size,))))

    mo_occ = jnp.zeros(pad_size)
    mo_occ = mo_occ.at[:3].set(2.0)  # Fill lowest 3 orbitals with 2 electrons each

    J = jnp.zeros((pad_size, pad_size))
    J = J.at[:real_size, :real_size].set(random.normal(keys[7], (real_size, real_size)))
    J = (J + J.T) / 2  # Make symmetric

    exc_energy = jnp.array(0.5)
    energy_nuc = 1.0
    nelectron = 6

    fock = h1e + J

    # Create parameters for network
    params = {"kernel": jnp.ones((10, 1)) * 0.1}

    print("\n=== Testing gradients with zero padding ===")

    # 1. Test get_veff_jax_masked
    print("\nTesting get_veff_jax_masked gradients:")

    def loss_veff(p):
        veff, _, _ = get_veff_jax_masked(dm, eri, ao_grid, grid_weights, mask, p)
        return jnp.sum(veff)

    try:
        grad_veff = grad(loss_veff)(params)
        has_nan = any(jnp.isnan(v).any() for v in jax.tree_util.tree_leaves(grad_veff))
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in get_veff_jax_masked gradients!")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 2. Test energy_tot_jax_masked
    print("\nTesting energy_tot_jax_masked gradients:")

    def loss_energy(d):
        return energy_tot_jax_masked(d, h1e, J, exc_energy, energy_nuc, mask)

    try:
        grad_energy = grad(loss_energy)(dm)
        has_nan = jnp.isnan(grad_energy).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in energy_tot_jax_masked gradients!")
            print(f"  NaN positions: {jnp.where(jnp.isnan(grad_energy))}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 3. Test get_occ_masked
    print("\nTesting get_occ_masked gradients:")

    def loss_occ(e):
        return jnp.sum(get_occ_masked(nelectron, e, mask))

    try:
        grad_occ = grad(loss_occ)(mo_energy)
        has_nan = jnp.isnan(grad_occ).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in get_occ_masked gradients!")
            print(f"  NaN positions: {jnp.where(jnp.isnan(grad_occ))}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 4. Test make_rdm1_masked
    print("\nTesting make_rdm1_masked gradients:")

    def loss_rdm1(c):
        return jnp.sum(make_rdm1_masked(c, mo_occ, mask))

    try:
        grad_rdm1 = grad(loss_rdm1)(mo_coeff)
        has_nan = jnp.isnan(grad_rdm1).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            print("  NaN found in make_rdm1_masked gradients!")
            print(f"  NaN positions: {jnp.where(jnp.isnan(grad_rdm1))}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # 5. Test masked_generalized_eigh_stable
    print("\nTesting masked_generalized_eigh_stable gradients:")

    def loss_eigh(f):
        e, c = masked_generalized_eigh_stable(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh = grad(loss_eigh)(fock)
        has_nan = jnp.isnan(grad_eigh).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    # Test each variant of the eigensolver
    print("\nTesting masked_generalized_eigh_stable1 gradients:")

    def loss_eigh1(f):
        e, c = masked_generalized_eigh_stable1(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh1 = grad(loss_eigh1)(fock)
        has_nan = jnp.isnan(grad_eigh1).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh1))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    print("\nTesting masked_generalized_eigh_stable2 gradients:")

    def loss_eigh2(f):
        e, c = masked_generalized_eigh_stable2(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh2 = grad(loss_eigh2)(fock)
        has_nan = jnp.isnan(grad_eigh2).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh2))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")

    print("\nTesting masked_generalized_eigh_stable3 gradients:")

    def loss_eigh3(f):
        e, c = masked_generalized_eigh_stable3(f, s1e, mask)
        return jnp.sum(e)

    try:
        grad_eigh3 = grad(loss_eigh3)(fock)
        has_nan = jnp.isnan(grad_eigh3).any()
        print(f"  Has NaN: {has_nan}")
        if has_nan:
            where_nan = jnp.where(jnp.isnan(grad_eigh3))
            print(f"  NaN positions: {where_nan}")
    except Exception as e:
        print(f"  Error computing gradient: {e}")


if __name__ == "__main__":
    # Test functions to see if we get Nans when differentiating and if padded matches
    # the non-padded SCF.
    test_gradients_with_zero_padding()
    test_gradients_for_masked_functions()
    compare_gradients()
    # Run all tests
    success = test_all()
    # Exit with appropriate code
    results_match = compare_padded_vs_non_padded()
    print(f"\nOverall test result: {'PASSED' if (results_match and success) else 'FAILED'}")
