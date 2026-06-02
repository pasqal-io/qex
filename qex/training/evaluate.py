"""Evaluation helpers.

Evaluation is split into a geometry-agnostic core and curve-specific helpers:

- :func:`evaluate_samples` runs the trained functional over an arbitrary set of
  molecules and returns predicted vs. reference energies plus error metrics.
  This is the general evaluation -- it makes no assumption that the geometries
  form a dissociation curve.
- :func:`parity_plot` is the general plot: reference on x, prediction on y, the
  ``y=x`` diagonal, and a chemical-accuracy band (default ±1.6 mHa).
- :func:`calculate_dissociation_profile` / :func:`plot_dissociation_profile`
  are the curve-specific case (energy vs. bond length), kept for H2-style runs.

Metrics:
- MAE  = mean(|pred - ref|)
- NPE  = max(pred - ref) - min(pred - ref)  (non-parallelity error)
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qex.scf.operators import get_ao_value

# Chemical accuracy: 1 kcal/mol ~= 1.6 mHa. Used as the parity-plot tolerance band.
CHEMICAL_ACCURACY_HA = 1.6e-3


def _metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """MAE and NPE (Hartree) from predicted/reference energy arrays."""
    errors = predicted - reference
    return {
        "mae": float(np.mean(np.abs(errors))),
        "npe": float(np.max(errors) - np.min(errors)) if errors.size else 0.0,
        "max_abs": float(np.max(np.abs(errors))) if errors.size else 0.0,
    }


def evaluate_samples(
    params: dict[str, Any],
    data_generator,
    molecule_configs: list,
    *,
    scf_energy_fn: Callable,
    xc_eval_fn: Callable,
    pass_descriptor_ctx: bool = False,
    label: str = "evaluation",
    verbose: bool = True,
    **scf_kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run the trained functional over a set of molecules (geometry-agnostic).

    Args:
        molecule_configs: list of :class:`qex.data_io.MoleculeConfig` to evaluate.
        scf_energy_fn: SCF loop returning the final total energy (e.g.
            :func:`qex.scf.rks_energy`); ``scf_kwargs`` are forwarded as-is.
        pass_descriptor_ctx: forward grid/atom coordinates (needed by descriptor
            networks).
        label: name used in progress output.

    Returns:
        ``(predicted, reference, metrics)`` where ``predicted``/``reference`` are
        energy arrays aligned with ``molecule_configs`` and ``metrics`` holds
        ``mae`` / ``npe`` / ``max_abs`` in Hartree.
    """
    scf_eval_jit = jax.jit(partial(scf_energy_fn, xc_eval_fn=xc_eval_fn, **scf_kwargs))

    predicted: list[float] = []
    reference: list[float] = []

    if verbose:
        print(f"\nRunning {label} over {len(molecule_configs)} molecule(s)...")
    iterator = tqdm(molecule_configs) if verbose else molecule_configs
    for cfg in iterator:
        mol, mf, dm, ref_energy, _density, _coords = data_generator.generate_data(
            cfg, save_data=False
        )

        scf_args = [
            params,
            dm,
            mol.intor("int2e", aosym="s1"),
            get_ao_value(mol, mf.grids.coords),
            mf.grids.weights,
            mf.get_ovlp(mol),
            mf.get_hcore(mol),
            mol.energy_nuc(),
            mol.nelectron,
        ]
        if pass_descriptor_ctx:
            scf_args.append(jnp.asarray(mf.grids.coords))
            scf_args.append(jnp.asarray(mol.atom_coords()))

        predicted.append(float(scf_eval_jit(*scf_args)))
        reference.append(float(ref_energy))

    predicted = np.array(predicted)
    reference = np.array(reference)
    return predicted, reference, _metrics(predicted, reference)


def parity_plot(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    title: str = "Predicted vs reference energy",
    tolerance: float = CHEMICAL_ACCURACY_HA,
    path_results: str | None = None,
    filename: str = "parity_plot.png",
    show: bool = True,
):
    """General evaluation plot: reference (x) vs predicted (y) energies.

    Draws the ideal ``y=x`` line and a shaded ``±tolerance`` band (default
    chemical accuracy, 1.6 mHa) so points outside the band are immediately
    visible. MAE and NPE are annotated in the legend.

    Args:
        reference / predicted: energy arrays (Hartree), same length.
        tolerance: half-width of the accuracy band in Hartree.
    """
    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    metrics = _metrics(predicted, reference)

    lo = float(min(reference.min(), predicted.min()))
    hi = float(max(reference.max(), predicted.max()))
    pad = 0.05 * (hi - lo or 1.0)
    line = np.array([lo - pad, hi + pad])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(
        line,
        line - tolerance,
        line + tolerance,
        color="gray",
        alpha=0.2,
        label=f"±{tolerance * 1e3:.1f} mHa (chemical accuracy)",
    )
    ax.plot(line, line, "--", color="black", linewidth=1, label="y = x")
    ax.scatter(
        reference,
        predicted,
        color="red",
        s=40,
        zorder=3,
        label=f"MAE {metrics['mae']:.2e} Ha · NPE {metrics['npe']:.2e} Ha",
    )

    ax.set_xlabel("Reference energy (Hartree)")
    ax.set_ylabel("Predicted energy (Hartree)")
    ax.set_title(title)
    ax.set_xlim(line)
    ax.set_ylim(line)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if path_results is not None:
        plt.savefig(f"{path_results}/{filename}", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return metrics


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
    pass_descriptor_ctx: bool = False,
    **scf_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate trained model + reference on a sweep of bond lengths.

    A curve-specific convenience wrapper around :func:`evaluate_samples`:
    it builds the per-geometry molecule configs from ``molecule_config_factory``
    and returns the energies aligned with ``bond_lengths``.
    """
    if bond_lengths is None:
        bond_lengths = np.linspace(0.5, 3.0, 20)

    molecule_configs = [
        molecule_config_factory(
            d, method=method, basis=basis, units=units, grid_density=grid_density
        )
        for d in bond_lengths
    ]

    ml_energies, ref_energies, _ = evaluate_samples(
        params,
        data_generator,
        molecule_configs,
        scf_energy_fn=scf_energy_fn,
        xc_eval_fn=xc_eval_fn,
        pass_descriptor_ctx=pass_descriptor_ctx,
        label="dissociation profile",
        **scf_kwargs,
    )

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
    ax2.axhspan(0, CHEMICAL_ACCURACY_HA, alpha=0.2, color="gray",
                label="Chemical Accuracy (1.6 mHa)")
    ax2.set_xlabel("Bond Length (Å)")
    ax2.set_ylabel("Error (Hartree)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    if path_results is not None:
        plt.savefig(f"{path_results}/dissociation_profile.png", dpi=300, bbox_inches="tight")
    plt.show()
