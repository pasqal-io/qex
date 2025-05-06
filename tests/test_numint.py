"""
Tests for numerical integration routines in DFT calculations.

This module contains tests for various exchange-correlation functionals and
numerical integration methods used in density functional theory (DFT) calculations.
Tests include standard LDA functionals, custom functionals, and neural network
based exchange-correlation functionals.
"""

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft.xcfun import define_xc_
from pyscfad import dft as addft
from pyscfad import gto as adgto

from qedft.train.td.numint_legacy import NumInt


def test_standard_xc():
    """Test standard LDA exchange-correlation functional.

    Creates a simple H2 molecule and tests the numerical integration
    of the LDA exchange-correlation functional. Verifies that the
    electron number, energy and potential matrix have expected properties.
    """
    # Create a simple H2 molecule
    mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g")

    # Set up DFT calculation with LDA functional
    mf = dft.RKS(mol)
    mf.xc = "LDA"
    mf.grids.level = 1  # Use coarse grid for testing
    mf.kernel()
    print("Energy:", mf.energy_tot())
    # Get initial density matrix
    dm = mf.get_init_guess()

    # Perform numerical integration
    ni = NumInt()
    nelec, exc, vxc = ni.nr_rks(mol, mf.grids, "LDA", dm)

    # Verify results
    assert isinstance(nelec, (int, float))
    assert vxc.ndim == 2
    assert vxc.shape[0] > 0
    assert vxc.shape[1] > 0


def test_define_xc():
    """Test custom exchange-correlation functional definition.

    Tests the define_xc_ functionality for creating custom functionals:
    1. Using string notation to combine standard functionals
    2. Using a custom function to define a new functional
    """
    mol = gto.M(atom="O 0 0 0; H 0 0 1; H 0 1 0", basis="sto-3g")

    # Test hybrid functional using string notation
    mf = dft.RKS(mol)
    define_xc_(mf._numint, ".2*HF + .08*LDA  + .19*VWN")
    e1 = mf.kernel()
    assert isinstance(e1, float)

    # Test custom functional using function definition
    def eval_xc(xc_code, rho, *args, **kwargs):
        """Simple quadratic functional for testing."""
        exc = 0.01 * rho**2
        vrho = 0.01 * 2 * rho
        vxc = (vrho, None, None, None)
        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative
        return exc, vxc, fxc, kxc

    mf = dft.RKS(mol)
    mf = mf.define_xc_(description=eval_xc, xctype="LDA")
    e3 = mf.kernel()
    assert isinstance(e3, float)


def test_nn_amplitude_encoding():
    """Test neural network exchange-correlation with amplitude encoding.

    Tests a neural network based functional that uses amplitude encoding
    for the density representation. Verifies that the integration produces
    valid electron numbers, energies and potential matrices.
    """
    # Create H2 molecule with automatic differentiation support
    mol = adgto.Mole(atom="H 0 0 0; H 0 0 1", basis="sto-3g")

    # Initialize DFT calculation
    mf = addft.RKS(mol)
    mf.grids.level = 1  # Use coarse grid for testing

    # Define test exchange-correlation functional
    def eval_xc(xc_code, rho, *args, **kwargs):
        """Simple quadratic functional for testing."""
        exc = 0.01 * rho**2
        vrho = 0.01 * 2 * rho
        vxc = (vrho, None, None, None)
        fxc = None
        kxc = None
        return exc, vxc, fxc, kxc

    mf = mf.define_xc_(
        description=eval_xc,
        xctype="LDA",
    )
    mf.kernel()

    # Get initial density matrix
    dm = mf.get_init_guess()

    # Initialize numerical integrator
    ni = NumInt()

    # Create mock neural network parameters
    nn_params = {
        "weights": np.random.random((10, 10)),
        "biases": np.random.random(10),
        "encoding_params": np.random.random(5),
    }

    # Define amplitude encoding functional
    def eval_xc_amplitude(xc_code, rho, *args, **kwargs):
        """Neural network functional with amplitude encoding."""
        exc = np.sum(0.01 * rho**2)
        vrho = 0.01 * 2 * rho
        vxc = (vrho, None, None, None)
        fxc = None
        kxc = None
        return exc, vxc, fxc, kxc

    # Perform integration with amplitude encoding
    ni.eval_xc = eval_xc_amplitude
    nelec, exc, vxc = ni.nr_rks(mol, mf.grids, "NN-AmplitudeEncoding", dm, params=nn_params)
    exc = float(exc)
    nelec = float(nelec)

    # Verify results
    assert isinstance(nelec, (int, float))
    assert isinstance(exc, (int, float))
    assert vxc.ndim == 2
    assert vxc.shape[0] > 0
    assert vxc.shape[1] > 0


def test_nn_xc():
    """Test basic neural network exchange-correlation functional.

    Tests a neural network based functional without amplitude encoding.
    Verifies integration results for electron number, energy and potential.
    """
    # Create H2 molecule with automatic differentiation support
    mol = adgto.Mole(atom="H 0 0 0; H 0 0 1", basis="sto-3g")

    # Initialize DFT calculation
    mf = addft.RKS(mol)
    mf.xc = "NN"
    mf.grids.level = 1  # Use coarse grid for testing

    # Define neural network functional
    def eval_xc(xc_code, rho, *args, **kwargs):
        """Simple neural network functional for testing."""
        exc = 0.01 * rho**2
        vrho = 0.01 * 2 * rho
        vxc = (vrho, None, None, None)
        fxc = None
        kxc = None
        return exc, vxc, fxc, kxc

    mf = mf.define_xc_(
        description=eval_xc,
        xctype="LDA",
    )

    # Get initial density matrix
    dm = mf.get_init_guess()

    # Initialize numerical integrator
    ni = NumInt()

    # Create mock neural network parameters
    nn_params = {
        "weights": np.random.random((10, 10)),
        "biases": np.random.random(10),
    }

    # Perform integration
    ni.eval_xc = eval_xc
    nelec, exc, vxc = ni.nr_rks(
        mol,
        mf.grids,
        "NN",
        dm,
        params=nn_params,
    )
    exc = float(exc)
    nelec = float(nelec)
    # Verify results
    assert isinstance(nelec, (int, float))
    assert isinstance(exc, (int, float))
    assert vxc.ndim == 2
    assert vxc.shape[0] > 0
    assert vxc.shape[1] > 0


def test_pyscf_pyscfad_consistency():
    """Test consistency between PySCF and PySCFAD RKS calculations.

    Verifies that PySCFAD produces the same results as PySCF when using
    identical settings for molecule, functional, grid, and initial guess.
    """
    # Create identical molecules in both frameworks
    mol_pyscf = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g")
    mol_ad = adgto.Mole(atom="H 0 0 0; H 0 0 1", basis="sto-3g")

    # Set up DFT calculations with identical settings
    mf_pyscf = dft.RKS(mol_pyscf)
    mf_ad = addft.RKS(mol_ad)

    # Use same functional and grid level
    mf_pyscf.xc = "LDA"
    mf_ad.xc = "LDA"
    mf_pyscf.grids.level = 1
    mf_ad.grids.level = 1

    # Run both calculations
    e_pyscf = mf_pyscf.kernel()
    e_ad = mf_ad.kernel()

    # Get density matrices
    dm_pyscf = mf_pyscf.get_init_guess()
    dm_ad = mf_ad.get_init_guess()

    # Initialize numerical integrators
    ni_pyscf = dft.numint.NumInt()
    ni_ad = NumInt()

    # Perform numerical integration for both
    nelec_pyscf, exc_pyscf, vxc_pyscf = ni_pyscf.nr_rks(
        mol_pyscf,
        mf_pyscf.grids,
        "LDA",
        dm_pyscf,
    )
    nelec_ad, exc_ad, vxc_ad = ni_ad.nr_rks(
        mol_ad,
        mf_ad.grids,
        "LDA",
        dm_ad,
    )

    # Verify consistency between frameworks
    np.testing.assert_allclose(e_pyscf, float(e_ad), rtol=1e-6)
    np.testing.assert_allclose(nelec_pyscf, float(nelec_ad), rtol=1e-6)
    np.testing.assert_allclose(exc_pyscf, float(exc_ad), rtol=1e-6)
    np.testing.assert_allclose(vxc_pyscf, vxc_ad, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
