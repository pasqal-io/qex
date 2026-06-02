"""Validate qex's JAX SCF pipeline against PySCF using STANDARD functionals.

The point is *not* to re-test libxc -- it is to drive the exact
:func:`qex.scf.rks_energy` loop the trained networks use, but with a known
PySCF/libxc functional (LDA / GGA / meta-GGA / global hybrid), and confirm the
converged H2 energy matches a plain ``pyscf.dft.RKS`` run. If qex's SCF
iteration (Fock build, generalized eigensolve, occupations, DIIS, energy
assembly) is correct, the numbers must agree to near machine precision.

The standard functional reaches the loop via ``encoding="libxc"``; its
ingredients (AO value/gradient arrays on the grid, the libxc code, the hybrid
coefficient) are packed into ``params`` by
:func:`qex.functionals.libxc_veff.make_libxc_ingredients`, so the SCF loop
signature is unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

from qex.functionals.libxc_veff import (
    make_eval_xc_libxc_ref,
    make_libxc_ingredients,
)
from qex.scf.rks import rks_energy

jax.config.update("jax_enable_x64", True)


# (label, libxc code, xc_type it exercises)
_FUNCTIONALS = [
    ("LDA", "lda,vwn", "LDA"),
    ("GGA-PBE", "pbe,pbe", "GGA"),
    ("GGA-BLYP", "blyp", "GGA"),
    ("MGGA-SCAN", "scan,scan", "MGGA"),
    ("hybrid-B3LYP", "b3lyp", "GGA+HFX"),
    ("hybrid-PBE0", "pbe0", "GGA+HFX"),
]


def _pyscf_reference(xc_code: str):
    """Converged PySCF RKS energy for H2 plus the SCF inputs qex needs."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = xc_code
    mf.grids.level = 3
    mf.grids.build()
    e_pyscf = float(mf.kernel())
    assert mf.converged, f"PySCF reference for {xc_code!r} did not converge"

    inputs = dict(
        dm=jnp.asarray(mf.get_init_guess()),
        eri=jnp.asarray(mol.intor("int2e", aosym="s1")),
        # ao_grid is unused by the libxc path (it uses params['ao0']), but the
        # loop still passes it through; hand it the value-array so shapes are sane.
        ao_grid=jnp.asarray(mol.eval_gto("GTOval_sph", mf.grids.coords)),
        grid_weights=jnp.asarray(mf.grids.weights),
        s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=float(mol.energy_nuc()),
        nelectron=int(mol.nelectron),
    )
    return e_pyscf, mol, mf.grids.coords, inputs


def _qex_libxc_energy(xc_code, mol, coords, inp, **kw):
    """Run qex `rks_energy` with encoding='libxc' for a standard functional."""
    ingredients = make_libxc_ingredients(mol, coords, xc_code)
    return float(
        rks_energy(
            ingredients,  # `params` carries the libxc ingredients
            inp["dm"], inp["eri"], inp["ao_grid"], inp["grid_weights"],
            inp["s1e"], inp["h1e"], inp["energy_nuc"], inp["nelectron"],
            xc_eval_fn=make_eval_xc_libxc_ref(xc_code),  # never called
            encoding="libxc",
            frac_enabled=0,  # integer aufbau, matches vanilla closed-shell RKS
            **kw,
        )
    )


@pytest.mark.parametrize(
    "label,xc_code,xctype",
    _FUNCTIONALS,
    ids=[f[0] for f in _FUNCTIONALS],
)
def test_qex_pipeline_matches_pyscf(label, xc_code, xctype):
    """qex SCF loop + standard libxc functional == pyscf.dft.RKS, for H2."""
    e_pyscf, mol, coords, inp = _pyscf_reference(xc_code)
    e_qex = _qex_libxc_energy(
        xc_code, mol, coords, inp, max_cycle=50, use_diis=True
    )
    np.testing.assert_allclose(
        e_qex, e_pyscf, atol=1e-7, rtol=0.0,
        err_msg=f"{label} ({xc_code}): qex {e_qex} vs pyscf {e_pyscf}",
    )


def test_lda_slater_analytic():
    """Sanity-check the libxc backend itself on pure Slater exchange ('lda,'),
    where the energy density is analytic: exc = -Cx rho^(1/3)."""
    from pyscf.dft import libxc

    rho = np.array([0.1, 0.5, 1.0, 2.0])
    exc, vxc, _, _ = libxc.eval_xc("lda,", rho, spin=0, deriv=1)
    cx = (3 / 4) * (3 / np.pi) ** (1 / 3)
    exc_ref = -cx * rho ** (1 / 3)
    np.testing.assert_allclose(exc, exc_ref, rtol=1e-10)
    # vrho = d(exc*rho)/drho = (4/3) exc.
    np.testing.assert_allclose(vxc[0], (4 / 3) * exc_ref, rtol=1e-10)


def test_ingredients_shapes():
    """make_libxc_ingredients evaluates AO derivatives at the order each
    xc_type needs: LDA=value only, GGA=+gradient, MGGA=+second derivatives."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    coords = dft.RKS(mol).grids.build().coords

    lda = make_libxc_ingredients(mol, coords, "lda,vwn")
    assert "ao0" in lda and "ao1" not in lda and lda["hyb"] == 0.0

    gga = make_libxc_ingredients(mol, coords, "pbe,pbe")
    assert gga["ao1"].shape[0] == 3 and "ao2_diag" not in gga

    mgga = make_libxc_ingredients(mol, coords, "scan,scan")
    assert mgga["ao2_diag"].shape[0] == 3

    hyb = make_libxc_ingredients(mol, coords, "b3lyp")
    assert hyb["hyb"] == pytest.approx(0.2)
