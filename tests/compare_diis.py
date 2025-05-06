"""
Comparison of PySCF DIIS and JAX DIIS implementations.

This module compares the convergence behavior of SCF calculations using:
1. PySCF's built-in DIIS implementation
2. A custom JAX-based DIIS implementation
3. No DIIS acceleration

The comparison uses a challenging molecular system (O-O-Be) to demonstrate
the effectiveness of DIIS for improving SCF convergence.
"""

import jax.numpy as jnp
import numpy as np
from pyscf import gto, lib, scf

from qedft.train.td.jax_diis import apply_diis, initialize_diis


def run_pyscf_diis() -> list[float]:
    """Run SCF with PySCF's DIIS implementation.

    Returns:
        List[float]: Energy values from each SCF iteration
    """
    print("=== PySCF DIIS Implementation ===")

    # Set up a very challenging molecule for SCF convergence
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.2; Be 0 0 2.4",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)

    # Get initial matrices
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)

    # Initialize DIIS
    adiis = lib.diis.DIIS()

    # Run SCF iterations
    energies = []
    for i in range(20):  # Increase iterations for harder convergence
        dm = mf.make_rdm1(c, occ)
        f = h + mf.get_veff(mol, dm)

        # Apply DIIS after the first two iterations
        if i > 1:
            f = adiis.update(f)

        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"E_{i} = {energy:.12f}")

    return energies


def run_jax_diis() -> list[float]:
    """Run SCF with JAX DIIS implementation.

    Returns:
        List[float]: Energy values from each SCF iteration
    """
    print("\n=== JAX DIIS Implementation ===")

    # Set up a very challenging molecule for SCF convergence
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.2; Be 0 0 2.4",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)

    # Get initial matrices
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)

    # Initialize JAX DIIS
    diis_state = initialize_diis(max_vec=6)

    # Convert numpy arrays to JAX arrays
    s_jax = jnp.array(s)

    # Run SCF iterations
    energies = []
    for i in range(20):  # Increase iterations for harder convergence
        dm = mf.make_rdm1(c, occ)
        dm_jax = jnp.array(dm)
        veff = mf.get_veff(mol, dm)
        f = h + veff
        f_jax = jnp.array(f)

        # Apply JAX DIIS after the first two iterations
        if i > 1:
            # Calculate error vector and apply DIIS
            f_jax, diis_state = apply_diis(
                diis_state,
                f_jax,
                dm_jax,
                s_jax,
                max_vec=20,
                min_vecs=1,
            )
            # Convert back to numpy for PySCF
            f = np.array(f_jax)

        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"E_{i} = {energy:.12f}")

    return energies


def run_no_diis() -> list[float]:
    """Run SCF with NO DIIS implementation.

    Returns:
        List[float]: Energy values from each SCF iteration
    """
    print("\n=== NO DIIS Implementation ===")

    # Set up a very challenging molecule for SCF convergence
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.2; Be 0 0 2.4",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)

    # Get initial matrices
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)

    # Run SCF iterations
    energies = []
    for i in range(20):  # Increase iterations for harder convergence
        dm = mf.make_rdm1(c, occ)
        veff = mf.get_veff(mol, dm)
        f = h + veff
        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"E_{i} = {energy:.12f}")

    return energies


def run_pyscf_diis_h2() -> list[float]:
    """Run SCF with PySCF's DIIS implementation for H2."""
    print("\n=== PySCF DIIS Implementation (H2 @ 5 A) ===")
    mol = gto.M(
        atom="H 0 0 0; H 0 0 5",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)
    adiis = lib.diis.DIIS()
    energies = []
    for i in range(20):
        dm = mf.make_rdm1(c, occ)
        f = h + mf.get_veff(mol, dm)
        if i > 1:
            f = adiis.update(f)
        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"H2_E_{i} = {energy:.12f}")
    return energies


def run_jax_diis_h2() -> list[float]:
    """Run SCF with JAX DIIS implementation for H2."""
    print("\n=== JAX DIIS Implementation (H2 @ 5 A) ===")
    mol = gto.M(
        atom="H 0 0 0; H 0 0 5",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)
    diis_state = initialize_diis(max_vec=6)
    s_jax = jnp.array(s)
    energies = []
    for i in range(20):
        dm = mf.make_rdm1(c, occ)
        dm_jax = jnp.array(dm)
        veff = mf.get_veff(mol, dm)
        f = h + veff
        f_jax = jnp.array(f)
        if i > 1:
            f_jax, diis_state = apply_diis(
                diis_state,
                f_jax,
                dm_jax,
                s_jax,
                max_vec=20,
                min_vecs=1,
            )
            f = np.array(f_jax)
        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"H2_E_{i} = {energy:.12f}")
    return energies


def run_no_diis_h2() -> list[float]:
    """Run SCF with NO DIIS implementation for H2."""
    print("\n=== NO DIIS Implementation (H2 @ 5 A) ===")
    mol = gto.M(
        atom="H 0 0 0; H 0 0 5",
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    h = mf.get_hcore()
    s = mf.get_ovlp()
    e, c = mf.eig(h, s)
    occ = mf.get_occ(e, c)
    energies = []
    for i in range(20):
        dm = mf.make_rdm1(c, occ)
        veff = mf.get_veff(mol, dm)
        f = h + veff
        e, c = mf.eig(f, s)
        energy = mf.energy_tot(dm, h, mf.get_veff(mol, dm))
        energies.append(energy)
        print(f"H2_E_{i} = {energy:.12f}")
    return energies


def compare_implementations():
    """Compare all three DIIS implementations for both systems."""

    print("\n" + "=" * 30 + " O-O-Be System " + "=" * 30)
    pyscf_energies_oobe = run_pyscf_diis()
    jax_energies_oobe = run_jax_diis()
    no_diis_energies_oobe = run_no_diis()

    print("\n=== Comparison of Energy Differences (O-O-Be) ===")
    print(
        "Iteration | PySCF Energy    | JAX Energy      | No DIIS Energy | "
        "Difference PySCF-JAX | Difference PySCF-No DIIS",
    )
    print("-" * 80)
    for i, (e1, e2, e3) in enumerate(
        zip(
            pyscf_energies_oobe,
            jax_energies_oobe,
            no_diis_energies_oobe,
        ),
    ):
        diff = abs(e1 - e2)
        diff_no_diis = abs(e1 - e3)
        print(
            f"{i:9d} | {e1:.12f} | {e2:.12f} | {e3:.12f} | " f"{diff:.12e} | {diff_no_diis:.12e}",
        )

    print("\n" + "=" * 30 + " H2 System (5 A) " + "=" * 30)
    pyscf_energies_h2 = run_pyscf_diis_h2()
    jax_energies_h2 = run_jax_diis_h2()
    no_diis_energies_h2 = run_no_diis_h2()

    print("\n=== Comparison of Energy Differences (H2 @ 5 A) ===")
    print(
        "Iteration | PySCF Energy    | JAX Energy      | No DIIS Energy | "
        "Difference PySCF-JAX | Difference PySCF-No DIIS",
    )
    print("-" * 80)
    for i, (e1, e2, e3) in enumerate(
        zip(
            pyscf_energies_h2,
            jax_energies_h2,
            no_diis_energies_h2,
        ),
    ):
        diff = abs(e1 - e2)
        diff_no_diis = abs(e1 - e3)
        print(
            f"{i:9d} | {e1:.12f} | {e2:.12f} | {e3:.12f} | " f"{diff:.12e} | {diff_no_diis:.12e}",
        )

    # Return values are not strictly needed if just printing, but kept for consistency
    # You might want to return a dictionary or structure if you need these results later
    return (
        pyscf_energies_oobe,
        jax_energies_oobe,
        no_diis_energies_oobe,
        pyscf_energies_h2,
        jax_energies_h2,
        no_diis_energies_h2,
    )


if __name__ == "__main__":
    compare_implementations()
