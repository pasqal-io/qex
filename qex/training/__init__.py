"""Training loop, evaluation helpers, and the high-level experiment driver.

- :func:`train` is the physics-agnostic, vmap-batched SCF training loop.
- :func:`calculate_dissociation_profile` / :func:`plot_dissociation_profile`
  evaluate and visualise a trained functional.
- :func:`run_experiment` is the high-level driver used by the ``qex train`` CLI
  and the example scripts: it builds the network, generates data, trains, and
  evaluates from a single :class:`qex.config.Config`.
"""

from qex.training.evaluate import (
    calculate_dissociation_profile,
    evaluate_dataset_split,
    evaluate_samples,
    parity_plot,
    plot_dissociation_profile,
)
from qex.training.experiment import (
    ExperimentResult,
    assemble_training_data,
    build_network,
    dataset_for_config,
    dataset_from_file,
    h2_molecule_config_factory,
    molecule_configs_for_split,
    run_experiment,
)
from qex.training.train import TrainHistory, train

__all__ = [
    "train",
    "TrainHistory",
    "run_experiment",
    "ExperimentResult",
    # Building blocks shared by run_experiment and the example scripts.
    "build_network",
    "assemble_training_data",
    "molecule_configs_for_split",
    "dataset_for_config",
    "dataset_from_file",
    "h2_molecule_config_factory",
    # Evaluation: general (parity) + curve-specific (dissociation).
    "evaluate_samples",
    "evaluate_dataset_split",
    "parity_plot",
    "calculate_dissociation_profile",
    "plot_dissociation_profile",
]
