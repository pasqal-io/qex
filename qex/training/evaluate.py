"""Evaluation helpers: dissociation profile + plotting."""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qex.scf.operators import get_ao_value


def calculate_dissociation_profile(
    params: dict[str, Any],
    data_generator,
    molecule_config_factory: Callable,
    *,
    scf_energy_fn: Callable,
    xc_eval_fn: Callable,
    bond_lengths: np.ndarray | None = None,
    method: str = "ccsd",
    basis: str = "631g",
    units: str = "Ang",
    grid_density: int = 0,
    path_results: str | None = None,
    **scf_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate trained model + reference on a sweep of bond lengths.

    `scf_energy_fn` is the SCF loop that returns final total energy (e.g.
    `qex.scf.rks.rks_energy`); `scf_kwargs` are forwarded as-is.
    """
    if bond_lengths is None:
        bond_lengths = np.linspace(0.5, 3.0, 20)

    scf_eval_bound = partial(scf_energy_fn, xc_eval_fn=xc_eval_fn, **scf_kwargs)
    scf_eval_jit = jax.jit(scf_eval_bound)

    ml_energies = []
    ref_energies = []

    print("\nCalculating dissociation profile...")
    for bond_length in tqdm(bond_lengths):
        molecule_config = molecule_config_factory(
            bond_length,
            method=method,
            basis=basis,
            units=units,
            grid_density=grid_density,
        )

        mol, mf, dm, ref_energy, _density, _coords = data_generator.generate_data(
            molecule_config, save_data=False
        )

        eri = mol.intor("int2e", aosym="s1")
        ao_grid = get_ao_value(mol, mf.grids.coords)
        s1e = mf.get_ovlp(mol)
        h1e = mf.get_hcore(mol)
        energy_nuc = mol.energy_nuc()

        ml_energy = scf_eval_jit(
            params,
            dm,
            eri,
            ao_grid,
            mf.grids.weights,
            s1e,
            h1e,
            energy_nuc,
            mol.nelectron,
        )

        ml_energies.append(ml_energy)
        ref_energies.append(ref_energy)
        print(
            f"Bond length: {bond_length:.2f} Å | ML energy: {ml_energy:.6f} | "
            f"Ref energy: {ref_energy:.6f}"
        )

    ml_energies = np.array(ml_energies)
    ref_energies = np.array(ref_energies)

    if path_results is not None:
        np.save(f"{path_results}/dissociation_bond_lengths.npy", bond_lengths)
        np.save(f"{path_results}/dissociation_ml_energies.npy", ml_energies)
        np.save(f"{path_results}/dissociation_ref_energies.npy", ref_energies)

    return bond_lengths, ml_energies, ref_energies


def plot_dissociation_profile(
    bond_lengths,
    ml_energies,
    ref_energies,
    *,
    method: str,
    exp_model: str,
    path_results: str | None = None,
):
    """Two-panel plot: energy curves + absolute error on a log axis."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(bond_lengths, ref_energies, "o-", color="blue", label=f"Reference ({method})")
    ax1.plot(bond_lengths, ml_energies, "s--", color="red", label="ML model")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_title(f"H₂ Dissociation Profile - {exp_model} Model")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    errors = ml_energies - ref_energies
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    npe = float(np.max(errors) - np.min(errors))

    ax2.semilogy(
        bond_lengths,
        abs_errors,
        "o-",
        color="green",
        label=f"Absolute Error (MAE: {mae:.2e}, NPE: {npe:.2e})",
    )
    ax2.axhspan(0, 1e-3, alpha=0.2, color="gray", label="Chemical Accuracy (10⁻³)")
    ax2.set_xlabel("Bond Length (Å)")
    ax2.set_ylabel("Error (Hartree)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    if path_results is not None:
        plt.savefig(f"{path_results}/dissociation_profile.png", dpi=300, bbox_inches="tight")
    plt.show()
