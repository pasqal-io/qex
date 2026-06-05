"""Reference-target registry: regressing Vxc and dm alongside energy/density.

Covers `qex.scf.targets` (the registry + `target_loss`) and the converged-state
target terms the SCF losses add for the optional `vxc` / `dm` targets:

- `target_loss` sums only the present keys, honours per-target weights, skips the
  inline-scored ones, and fails loudly on an unknown key.
- `rks_loss_scan` (both unroll and implicit) produces finite gradients when the
  vxc / dm targets are switched on, and reproduces the energy+density-only loss
  when their weights are zero (so turning them off is a true no-op).

Fixture mirrors `test_rks_scan.py`: H2/STO-3G, tiny grid, small GlobalMLP.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

from qex.functionals.mlp import GlobalMLP
from qex.functionals.xc import make_eval_xc_global
from qex.scf.inputs import SCFInputs
from qex.scf.operators import get_ao_value
from qex.scf.rks import rks_loss_scan
from qex.scf.targets import SCFState, TARGETS, target_loss

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = "lda"
    mf.grids.level = 0
    mf.grids.build()
    mf.kernel()

    net = GlobalMLP(features=[16], act_fn=nn.gelu)
    ng = mf.grids.coords.shape[0]
    params = net.init(jax.random.PRNGKey(0), jnp.ones(ng))
    xc = make_eval_xc_global(net, vxc_grad_scale=1.0)

    ao = jnp.asarray(get_ao_value(mol, mf.grids.coords))
    dm = jnp.asarray(mf.make_rdm1())
    rho = jnp.einsum("gi,ij,gj->g", ao, dm, ao)
    # Reference Vxc (KS) = veff - J, matching qex's `vhf - J`.
    dm_np = mf.make_rdm1()
    vxc_ref = jnp.asarray(np.asarray(mf.get_veff(mol, dm_np)) - np.asarray(mf.get_j(mol, dm_np)))

    targets = {"energy": jnp.asarray(float(mf.e_tot)), "density": rho,
               "dm": dm, "vxc": vxc_ref}
    inp = SCFInputs(
        dm=dm, eri=jnp.asarray(mol.intor("int2e", aosym="s1")), ao_grid=ao,
        grid_weights=jnp.asarray(mf.grids.weights), s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=jnp.asarray(float(mol.energy_nuc())),
        nelectron=jnp.asarray(int(mol.nelectron)),
        features={}, targets=targets,
    )
    return dict(params=params, xc=xc, inp=inp)


_FRAC = dict(frac_enabled=0, frac_theta=0.04, frac_mu=None,
             frac_mu_shift=0.001, frac_step_grad=0.6, frac_max_steps=100)


# --- registry unit tests --------------------------------------------------- #

def test_target_loss_sums_present_weighted():
    state = SCFState(e_tot=jnp.asarray(1.0), dm=jnp.ones((2, 2)),
                     rho=jnp.ones(3), vxc=jnp.zeros((2, 2)))
    refs = {"dm": jnp.zeros((2, 2)), "vxc": jnp.ones((2, 2))}
    # dm term: mean((1-0)^2)=1 ; vxc term: mean((0-1)^2)=1
    out = target_loss(state, refs, {"dm": 2.0, "vxc": 0.5})
    np.testing.assert_allclose(float(out), 2.0 * 1.0 + 0.5 * 1.0)


def test_target_loss_skips_named_and_unweighted():
    state = SCFState(e_tot=jnp.asarray(1.0), dm=jnp.ones((2, 2)),
                     rho=jnp.ones(3), vxc=jnp.zeros((2, 2)))
    refs = {"energy": jnp.asarray(0.0), "dm": jnp.zeros((2, 2))}
    # energy is skipped; dm has weight 1.0 -> mean((1-0)^2)=1.0.
    out = target_loss(state, refs, {"dm": 1.0}, skip=("energy",))
    np.testing.assert_allclose(float(out), 1.0)


def test_target_loss_unweighted_is_zero():
    """A present target with no (or zero) weight contributes nothing."""
    state = SCFState(e_tot=jnp.asarray(1.0), dm=jnp.ones((2, 2)),
                     rho=jnp.ones(3), vxc=jnp.zeros((2, 2)))
    refs = {"dm": jnp.zeros((2, 2))}  # present but no weight given
    np.testing.assert_allclose(float(target_loss(state, refs, {})), 0.0)


def test_target_loss_unknown_key_raises_when_weighted():
    state = SCFState(e_tot=jnp.asarray(1.0), dm=jnp.ones((2, 2)),
                     rho=jnp.ones(3), vxc=jnp.zeros((2, 2)))
    with pytest.raises(KeyError, match="Unknown reference target"):
        target_loss(state, {"bogus": jnp.ones((2, 2))}, {"bogus": 1.0})


def test_registry_has_expected_targets():
    assert {"energy", "density", "vxc", "dm"} <= set(TARGETS)


# --- end-to-end loss tests ------------------------------------------------- #

def _loss(h2, params, tw, mode):
    return rks_loss_scan(
        params, h2["inp"], xc_eval_fn=h2["xc"], encoding="global",
        max_cycle=12, differentiation=mode, target_weights=tw, **_FRAC,
    )


@pytest.mark.parametrize("mode", ["unroll", "implicit"])
def test_vxc_dm_terms_give_finite_grads(h2, mode):
    tw = (("vxc", 0.5), ("dm", 0.3))
    g = jax.grad(lambda p: _loss(h2, p, tw, mode))(h2["params"])
    leaves = jax.tree_util.tree_leaves(g)
    assert leaves and all(jnp.all(jnp.isfinite(x)) for x in leaves)


@pytest.mark.parametrize("mode", ["unroll", "implicit"])
def test_zero_weight_is_noop(h2, mode):
    """Switching vxc/dm on with weight 0 must equal the energy+density-only loss."""
    base = float(_loss(h2, h2["params"], (("density", 1.0),), mode))
    zeroed = float(_loss(
        h2, h2["params"], (("density", 1.0), ("vxc", 0.0), ("dm", 0.0)), mode))
    np.testing.assert_allclose(base, zeroed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["unroll", "implicit"])
def test_density_can_be_switched_off(h2, mode):
    """Density is a normal weighted loss now: weight 0 drops its contribution
    (energy-only), which differs from the energy+density loss — no code edit."""
    energy_and_density = float(_loss(h2, h2["params"], (("density", 1.0),), mode))
    energy_only = float(_loss(h2, h2["params"], (("density", 0.0),), mode))
    # Turning density off changes the loss (its term was non-zero here)...
    assert not np.isclose(energy_and_density, energy_only)
    # ...and energy-only equals the loss with no non-energy terms at all.
    all_off = (("density", 0.0), ("vxc", 0.0), ("dm", 0.0))
    none_on = float(_loss(h2, h2["params"], all_off, mode))
    np.testing.assert_allclose(energy_only, none_on, rtol=1e-12, atol=1e-12)


def test_vxc_weight_increases_loss(h2):
    """A positive vxc weight against a mismatched reference raises the loss."""
    # Use a deliberately wrong vxc reference (the converged prediction is not all
    # zeros) so the term is strictly positive and must lift the total loss.
    inp = h2["inp"]
    bad = inp.replace(targets={**inp.targets, "vxc": jnp.zeros_like(inp.targets["vxc"]) + 5.0})
    base = float(rks_loss_scan(
        h2["params"], inp, xc_eval_fn=h2["xc"], encoding="global",
        max_cycle=12, differentiation="unroll", target_weights=(), **_FRAC,
    ))
    with_vxc = float(rks_loss_scan(
        h2["params"], bad, xc_eval_fn=h2["xc"], encoding="global",
        max_cycle=12, differentiation="unroll",
        target_weights=(("vxc", 1.0),), **_FRAC,
    ))
    assert with_vxc > base


# --- availability validation / recompute ----------------------------------- #

def _fake_dp(name, method="rks", with_vxc=False):
    from qex.data_io.dataset import Datapoint, MoleculeConfig

    cfg = MoleculeConfig(name=name, atom_coords="H 0 0 0; H 0 0 0.74", method=method)
    dp = Datapoint(
        meta=cfg, energy=-1.0, density=np.ones(4), coords=np.zeros((4, 3)),
        dm=np.eye(2), eri=np.zeros((2, 2, 2, 2)), ao_grid=np.ones((4, 2)),
        grid_weights=np.ones(4), s1e=np.eye(2), h1e=np.eye(2),
        energy_nuc=0.5, nelectron=2,
    )
    if with_vxc:
        dp.vxc = np.zeros((2, 2))
    return dp


def test_vxc_requested_but_absent_raises():
    """vxc weight > 0 with no stored vxc and no recompute -> a clear error."""
    from qex.config.config import Config
    from qex.data_io.dataset import QexDataset
    from qex.training.experiment import _ensure_optional_targets_available

    ds = QexDataset(train=[_fake_dp("a", with_vxc=False)], val=[], test=[])
    config = Config()
    config.set("training.target_weights", {"vxc": 1.0})
    with pytest.raises(ValueError, match="no 'vxc' target"):
        _ensure_optional_targets_available(
            ds, config, data_generator=None,
            splits=("train",), dataset_file=None,
        )


def test_vxc_absent_but_off_is_ok():
    """vxc weight 0 (off) never trips the availability check, even with no vxc."""
    from qex.config.config import Config
    from qex.data_io.dataset import QexDataset
    from qex.training.experiment import _ensure_optional_targets_available

    ds = QexDataset(train=[_fake_dp("a", with_vxc=False)], val=[], test=[])
    config = Config()  # no target_weights -> vxc weight defaults to 0
    # Must not raise (and must not touch data_generator).
    _ensure_optional_targets_available(
        ds, config, data_generator=None, splits=("train",), dataset_file=None,
    )


def test_vxc_present_passes_check():
    """A dataset that already carries vxc passes the check without recompute."""
    from qex.config.config import Config
    from qex.data_io.dataset import QexDataset
    from qex.training.experiment import _ensure_optional_targets_available

    ds = QexDataset(train=[_fake_dp("a", with_vxc=True)], val=[], test=[])
    config = Config()
    config.set("training.target_weights", {"vxc": 0.5})
    _ensure_optional_targets_available(
        ds, config, data_generator=None, splits=("train",), dataset_file=None,
    )
