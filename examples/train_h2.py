"""Train an XC functional on H2 (CCSD reference) and plot the dissociation curve.

This is the *explicit* example: it spells out every step of the pipeline so you
can read the wiring top to bottom and adapt it. It calls the same public library
functions that the ``qex train`` CLI uses under the hood, so there is no
duplicated plumbing -- only the orchestration is laid bare here.

For a one-liner equivalent, see ``qex.run_experiment`` (or just run
``qex train --config examples/h2.yaml``). Reach for this script when you want to
change *how* the pieces fit together (custom data, a different loss, extra
logging) rather than just *which* settings are used.

Run it directly::

    python examples/train_h2.py

Settings live in ``examples/h2.yaml`` and are loaded into a :class:`qex.Config`.
"""

from pathlib import Path

import jax
import numpy as np
import optax

from qex import Config
from qex.data_io import DataGenerator
from qex.scf import rks_energy, rks_loss_scan
from qex.training import (
    build_network,
    calculate_dissociation_profile,
    dataset_for_config,
    evaluate_samples,
    h2_molecule_config_factory,
    parity_plot,
    plot_dissociation_profile,
    train,
)

CONFIG_PATH = Path(__file__).with_name("h2.yaml")

# SCF kwargs the scan loop (`rks_loss_scan`) does not accept; stripped at train
# time and kept for the DIIS-based evaluation loop (`rks_energy`).
_DIIS_ONLY_KEYS = ("diis_max_vec", "diis_min_vec", "diis_start_cycle", "diis_damping")


def _scf_kwargs(config: Config) -> dict:
    """Collect the SCF / fractional-occupation kwargs from the config."""
    return dict(
        encoding=config.get("model.encoding", "global"),
        max_cycle=config.get("scf.max_cycle", 15),
        use_diis=config.get("scf.use_diis", True),
        diis_max_vec=config.get("scf.diis_max_vec", 15),
        diis_min_vec=config.get("scf.diis_min_vec", 2),
        diis_start_cycle=config.get("scf.diis_start_cycle", 1),
        diis_damping=config.get("scf.diis_damping", 0.0),
        frac_enabled=config.get("scf.frac_enabled", 1),
        frac_theta=config.get("scf.frac_theta", 0.04),
        frac_mu=config.get("scf.frac_mu", None),
        frac_mu_shift=config.get("scf.frac_mu_shift", 0.001),
        frac_step_grad=config.get("scf.frac_step_grad", 0.6),
        frac_max_steps=config.get("scf.frac_max_steps", 100),
    )


def main() -> None:
    config = Config(config_path=str(CONFIG_PATH))

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", config.get("platform", "cpu"))

    output_dir = config.get("output_dir", "results/h2_dissociation")

    # 1. Build the XC network from the config (descriptor / mlp / qcnn). This is
    #    the same library helper the CLI uses, so behaviour stays identical.
    #    (`_network` itself isn't needed downstream -- training uses xc_eval_fn
    #    and the init params; `is_descriptor` flags whether the SCF loop needs
    #    grid/atom-coordinate context.)
    _network, xc_eval_fn, params, is_descriptor = build_network(config)

    # 2. Reference data for the whole train/val/test split, as ONE self-
    #    describing HDF5 file (auto-cached: generated once, reloaded on reruns).
    #    Geometries come from data.{train,val,test}_bond_lengths. No per-geometry
    #    folders or scattered .npy -- everything lives in output_dir/dataset.h5.
    data_generator = DataGenerator(output_dir)
    molecule_factory = h2_molecule_config_factory(config)
    dataset = dataset_for_config(
        config,
        molecule_config_factory=molecule_factory,
        is_descriptor=is_descriptor,
    )
    training_data = dataset.training_tuples("train")
    val_data = dataset.training_tuples("val")
    test_configs = [dp.meta for dp in dataset.test]
    print(
        f"Data split -> train: {len(training_data)} | "
        f"val: {len(val_data)} | test: {len(test_configs)}"
    )

    # 3. Optimizer: Adam with optional global-norm gradient clipping.
    optimizer = optax.adam(config.get("training.learning_rate", 1e-4))
    grad_clip = config.get("training.grad_clip", 0.5)
    if grad_clip is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optimizer)

    # 4. Train. `rks_loss_scan` uses lax.scan over SCF cycles, so compile time is
    #    ~independent of max_cycle; it does not take the DIIS-only kwargs.
    #    Validation runs every n_val_iter steps (separate jitted closure -> no
    #    per-step slowdown); the lowest-val-loss params are returned as `best`.
    scf_kwargs = _scf_kwargs(config)
    train_scf_kwargs = {k: v for k, v in scf_kwargs.items() if k not in _DIIS_ONLY_KEYS}
    trained_params, history = train(
        params,
        training_data,
        optimizer,
        scf_loss_fn=rks_loss_scan,
        xc_eval_fn=xc_eval_fn,
        n_iterations=config.get("training.n_iterations", 1000),
        log_every=config.get("training.log_every", 5),
        val_data=val_data or None,
        n_val_iter=config.get("training.n_val_iter", 50),
        patience=config.get("training.patience", None),
        return_history=True,
        energy_weight=config.get("training.energy_weight", 1.0),
        density_weight=config.get("training.density_weight", 1.0),
        **train_scf_kwargs,
    )

    # 5a. General evaluation on the held-out TEST set: parity plot (reference vs
    #     predicted, y=x, ±1.6 mHa band) + MAE/NPE. Geometry-agnostic.
    test_predicted, test_reference, test_metrics = evaluate_samples(
        trained_params,
        data_generator,
        test_configs,
        scf_energy_fn=rks_energy,
        xc_eval_fn=xc_eval_fn,
        pass_descriptor_ctx=is_descriptor,
        label="test set",
        **scf_kwargs,
    )
    parity_plot(
        test_reference,
        test_predicted,
        title=f"Test set: predicted vs {config.get('data.method', 'ccsd')} reference",
        path_results=output_dir,
    )

    # 5b. Dissociation profile (curve-specific): energy vs bond length.
    eval_bond_lengths = np.linspace(
        config.get("data.eval_min", 0.5),
        config.get("data.eval_max", 3.0),
        config.get("data.eval_points", 30),
    )
    bond_lengths, ml_energies, ref_energies = calculate_dissociation_profile(
        params=trained_params,
        data_generator=data_generator,
        molecule_config_factory=molecule_factory,
        scf_energy_fn=rks_energy,
        xc_eval_fn=xc_eval_fn,
        bond_lengths=eval_bond_lengths,
        method=config.get("data.method", "ccsd"),
        basis=config.get("data.basis", "631g"),
        units=config.get("data.units", "Ang"),
        grid_density=config.get("data.grid_density", 0),
        path_results=output_dir,
        pass_descriptor_ctx=is_descriptor,
        **scf_kwargs,
    )
    plot_dissociation_profile(
        bond_lengths=bond_lengths,
        ml_energies=ml_energies,
        ref_energies=ref_energies,
        method=config.get("data.method", "ccsd"),
        exp_model=config.get("model.exp_model", "density"),
        path_results=output_dir,
    )

    print("\n" + "-" * 70)
    print("Training and evaluation complete!")
    if history.best_iter is not None:
        print(f"  Best val loss: {history.best_val_loss:.6f} @ iter {history.best_iter}")
    print(f"  Test MAE: {test_metrics['mae']:.3e} Ha   NPE: {test_metrics['npe']:.3e} Ha")
    print(f"  Results written to: {output_dir}")
    print("-" * 70)


if __name__ == "__main__":
    main()
