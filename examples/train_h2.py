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
from qex.functionals.descriptor import DescriptorXC
from qex.functionals.mlp import GlobalMLP, LocalMLP
from qex.functionals.qcnn import QCNN
from qex.functionals.xc import make_eval_xc_global, make_eval_xc_local
# from qex.legacy.td.train import patch_pyscfad
from qex.scf.operators import get_ao_value
from qex.scf.rks import rks_energy, rks_loss, rks_loss_scan
from qex.training.evaluate import (
    calculate_dissociation_profile,
    plot_dissociation_profile,
)
from qex.training.train import train

# patch_pyscfad()
jax.config.update("jax_enable_x64", True)


CONFIG = {
    "rng": 42,
    "method": "ccsd",
    "basis": "631g",
    "units": "Ang",
    "grid_density": 0,
    "learning_rate": 1e-4,
    "n_iterations": 500,
    "energy_weight": 1.0,
    "density_weight": 1.0,
    "max_cycle": 15,
    "use_diis": True,
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
    # "local": network outputs exc(r) per electron at each grid point
    #          (LDA-style; E_xc assembled outside as Σ exc·rho·w).
    # "global": network outputs the scalar E_xc[ρ] directly (already integrated).
    "encoding": "global",
    "n_grid": 1240,  # only used to size GlobalMLP's first Dense layer
    # "mlp" (LocalMLP for local / GlobalMLP for global),
    # "descriptor"
    # (DescriptorXC — only valid with encoding="global"; grid-size invariant),
    # or "qcnn" (quantum convolutional network; encoding="global" only).
    "model": "descriptor",
    "n_atom": 2,
    # Global: [128, 128, 128, 128, 128]
    # Local: [32, 32]
    "hidden_layers":  [128, 128, 128, 128, 128],
    # Descriptor parameters
    "descriptor_alphas": (0.5, 1.0, 2.0, 4.0),
    "descriptor_scale": 1.0,
    "descriptor_rho_floor": 1e-10,
    # QCNN parameters. `n_grid` must be divisible by qcnn_n_features ** qcnn_n_layers.
    "qcnn_n_qubits": 2,
    "qcnn_n_features": 2,
    "qcnn_n_layers": 2,
    "qcnn_n_var_layers": 1,
    "qcnn_feature_map": "direct",
    "qcnn_head_features": (32,),
    # QCNN noise. `qcnn_gate_noise` is a list of (type, probability) pairs applied
    # to every gate in the feature map and ansatz; types: "bitflip",
    # "amplitude_damping", "depolarizing", "phaseflip" (any horqrux DigitalNoiseType).
    # `qcnn_gaussian_noise_std` adds Gaussian noise to each layer's output to
    # emulate sampling/readout noise (0 disables it).
    "qcnn_gate_noise": [],  # e.g. [("bitflip", 0.01), ("amplitude_damping", 0.01)]
    "qcnn_gaussian_noise_std": 0.0,
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
        encoding=CONFIG["encoding"],
        max_cycle=CONFIG["max_cycle"],
        use_diis=CONFIG["use_diis"],
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

    is_descriptor = CONFIG["model"] == "descriptor"
    is_qcnn = CONFIG["model"] == "qcnn"
    if is_descriptor and CONFIG["encoding"] != "global":
        raise ValueError("DescriptorXC requires encoding='global'.")
    if is_qcnn and CONFIG["encoding"] != "global":
        raise ValueError("QCNN requires encoding='global'.")

    if is_qcnn:
        gate_noise = None
        if CONFIG["qcnn_gate_noise"]:
            from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
            gate_noise = tuple(
                DigitalNoiseInstance(DigitalNoiseType[kind.upper()], prob)
                for kind, prob in CONFIG["qcnn_gate_noise"]
            )
        network = QCNN(
            n_qubits=CONFIG["qcnn_n_qubits"],
            n_features=CONFIG["qcnn_n_features"],
            n_layers=CONFIG["qcnn_n_layers"],
            n_var_layers=CONFIG["qcnn_n_var_layers"],
            feature_map=CONFIG["qcnn_feature_map"],
            head_features=CONFIG["qcnn_head_features"],
            noise=gate_noise,
            gaussian_noise_std=CONFIG["qcnn_gaussian_noise_std"],
        )
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=CONFIG["vxc_grad_scale"])
        params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(CONFIG["n_grid"]))
    elif is_descriptor:
        network = DescriptorXC(
            hidden=CONFIG["hidden_layers"],
            alphas=CONFIG["descriptor_alphas"],
            scale=CONFIG["descriptor_scale"],
            rho_floor=CONFIG["descriptor_rho_floor"],
            act_fn=nn.gelu,
        )
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=CONFIG["vxc_grad_scale"])
        params = network.init(
            random.PRNGKey(CONFIG["rng"]),
            jnp.ones(CONFIG["n_grid"]),
            jnp.zeros((CONFIG["n_grid"], 3)),
            jnp.ones(CONFIG["n_grid"]) / CONFIG["n_grid"],
            jnp.zeros((CONFIG["n_atom"], 3)),
        )
    elif CONFIG["encoding"] == "local":
        network = LocalMLP(features=CONFIG["hidden_layers"], act_fn=nn.gelu)
        xc_eval_fn = make_eval_xc_local(network, vxc_grad_scale=CONFIG["vxc_grad_scale"])
        params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(CONFIG["n_grid"]))
    elif CONFIG["encoding"] == "global":
        network = GlobalMLP(features=CONFIG["hidden_layers"], act_fn=nn.gelu)
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=CONFIG["vxc_grad_scale"])
        params = network.init(random.PRNGKey(CONFIG["rng"]), jnp.ones(CONFIG["n_grid"]))
    else:
        raise ValueError(f"Unknown encoding: {CONFIG['encoding']!r}")

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
        precomputed = [
            eri,
            ao_grid,
            mf.grids.weights,
            mf.get_ovlp(mol),
            mf.get_hcore(mol),
            mol.energy_nuc(),
            mol.nelectron,
        ]
        if is_descriptor:
            precomputed.append(jnp.asarray(mf.grids.coords))
            precomputed.append(jnp.asarray(mol.atom_coords()))
        training_data.append((energy, jnp.c_[coords, density], tuple(precomputed), dm))

    optimizer = optax.adam(CONFIG["learning_rate"])
    if CONFIG["grad_clip"] is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(CONFIG["grad_clip"]),
            optimizer,
        )

    # `rks_loss_scan` uses lax.scan over SCF cycles (no DIIS) so compile time
    # is roughly independent of `max_cycle` — for QCNN training that's the
    # difference between ~3s and several minutes per jit. We strip DIIS-specific
    # kwargs since the scan variant doesn't take them.
    scf_kwargs = _scf_kwargs()
    for k in ("diis_max_vec", "diis_min_vec", "diis_start_cycle", "diis_damping"):
        scf_kwargs.pop(k, None)

    trained_params = train(
        params,
        training_data,
        optimizer,
        scf_loss_fn=rks_loss_scan,
        xc_eval_fn=xc_eval_fn,
        n_iterations=CONFIG["n_iterations"],
        energy_weight=CONFIG["energy_weight"],
        density_weight=CONFIG["density_weight"],
        **scf_kwargs,
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
        pass_descriptor_ctx=is_descriptor,
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
