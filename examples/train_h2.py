"""Train an MLP XC functional on H₂ (CCSD reference) and plot the dissociation curve.

Self-contained entry point: imports the qex library and runs end-to-end.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from qex.data_io.dataset_generation import DataGenerator, MoleculeConfig
from qex.functionals.mlp import MLP
from qex.functionals.xc import make_eval_xc
from qex.legacy.td.train import patch_pyscfad
from qex.scf.operators import get_ao_value
from qex.scf.rks import rks_energy, rks_loss
from qex.training.evaluate import (
    calculate_dissociation_profile,
    plot_dissociation_profile,
)
from qex.training.train import train

patch_pyscfad()
jax.config.update("jax_enable_x64", True)


CONFIG = {
    "rng": 42,
    "hidden_layers": [128, 128, 128, 128, 128],
    "method": "ccsd",
    "basis": "631g",
    "units": "Ang",
    "grid_density": 0,
    "learning_rate": 1e-4,
    "n_iterations": 300,
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
    "frac_mu": None,
    "frac_mu_shift": 0.001,
    "frac_step_grad": 0.6,
    "vxc_grad_scale": 1.0,
    "platform": "cpu",
    "output_dir": "results/h2_dissociation",
    "grad_clip": 0.5,
    "exp_model": "density",
}

jax.config.update("jax_platform_name", CONFIG["platform"])


def _h2_molecule_config(bond_length, *, method, basis, units, grid_density):
    return MoleculeConfig(
        name=f"H2_{bond_length:.2f}",
        atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
        units=units,
        basis=basis,
        method=method,
        grid_density=grid_density,
    )


def _scf_kwargs():
    return dict(
        max_cycle=CONFIG["max_cycle"],
        diis_max_vec=CONFIG["diis_max_vec"],
        diis_min_vec=CONFIG["diis_min_vec"],
        diis_start_cycle=CONFIG["diis_start_cycle"],
        diis_damping=CONFIG["diis_damping"],
        frac_enabled=CONFIG["frac_enabled"],
        frac_theta=CONFIG["frac_theta"],
        frac_mu=CONFIG["frac_mu"],
        frac_mu_shift=CONFIG["frac_mu_shift"],
        frac_step_grad=CONFIG["frac_step_grad"],
        frac_max_steps=CONFIG["frac_max_steps"],
    )


if __name__ == "__main__":

    network = MLP(features=CONFIG["hidden_layers"], act_fn=nn.gelu)
    params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(1240))
    xc_eval_fn = make_eval_xc(network, vxc_grad_scale=CONFIG["vxc_grad_scale"])

    ccsd_bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"Generating {len(ccsd_bond_lengths)} CCSD configurations...")
    molecule_configs = [
        _h2_molecule_config(
            d,
            method="ccsd",
            basis=CONFIG["basis"],
            units=CONFIG["units"],
            grid_density=CONFIG["grid_density"],
        )
        for d in ccsd_bond_lengths
    ]

    data_generator = DataGenerator(CONFIG["output_dir"])

    training_data = []
    for cfg in molecule_configs:
        mol, mf, dm, energy, density, coords = data_generator.generate_data(cfg)
        eri = mol.intor("int2e", aosym="s1")
        ao_grid = get_ao_value(mol, mf.grids.coords)
        precomputed = (
            eri,
            ao_grid,
            mf.grids.weights,
            mf.get_ovlp(mol),
            mf.get_hcore(mol),
            mol.energy_nuc(),
            mol.nelectron,
        )
        training_data.append((energy, jnp.c_[coords, density], precomputed, dm))

    optimizer = optax.adam(CONFIG["learning_rate"])
    if CONFIG["grad_clip"] is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(CONFIG["grad_clip"]),
            optimizer,
        )

    trained_params = train(
        params,
        training_data,
        optimizer,
        scf_loss_fn=rks_loss,
        xc_eval_fn=xc_eval_fn,
        n_iterations=CONFIG["n_iterations"],
        energy_weight=CONFIG["energy_weight"],
        density_weight=CONFIG["density_weight"],
        **_scf_kwargs(),
    )

    print("\n" + "-" * 70)
    print("Calculating H2 dissociation profile")
    print("-" * 70)

    bond_lengths = np.linspace(0.5, 3.0, 30)
    bond_lengths, ml_energies, ref_energies = calculate_dissociation_profile(
        params=trained_params,
        data_generator=data_generator,
        molecule_config_factory=_h2_molecule_config,
        scf_energy_fn=rks_energy,
        xc_eval_fn=xc_eval_fn,
        bond_lengths=bond_lengths,
        method=CONFIG["method"],
        basis=CONFIG["basis"],
        units=CONFIG["units"],
        grid_density=CONFIG["grid_density"],
        path_results=CONFIG["output_dir"],
        **_scf_kwargs(),
    )

    plot_dissociation_profile(
        bond_lengths=bond_lengths,
        ml_energies=ml_energies,
        ref_energies=ref_energies,
        method=CONFIG["method"],
        exp_model=CONFIG["exp_model"],
        path_results=CONFIG["output_dir"],
    )

    print("\n" + "-" * 70)
    print("Training and evaluation complete!")
    print("-" * 70)
