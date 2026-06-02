"""QEX: Quantum-Enhanced Density Functional Theory in JAX.

QEX trains (quantum-)neural exchange-correlation functionals for Kohn-Sham DFT
using differentiable SCF loops written in JAX.

Public API
----------
The most common entry points are re-exported here so they can be imported
directly from the top-level package::

    from qex import Config, run_experiment, train
    from qex import GlobalMLP, DescriptorMLP  # XC networks
    from qex import make_eval_xc_global, rks_loss_scan

For the full per-subpackage API see :mod:`qex.scf`, :mod:`qex.functionals`,
:mod:`qex.training`, :mod:`qex.data_io`, and :mod:`qex.config`.

Imports are lazy (PEP 562): touching ``qex.run_experiment`` only imports the
heavy JAX/PySCF machinery on first access, so ``import qex`` stays cheap.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# name -> "module:attribute" it is re-exported from. Resolved lazily.
_LAZY_EXPORTS: dict[str, str] = {
    # config
    "Config": "qex.config:Config",
    "setup_config": "qex.config:setup_config",
    # functionals
    "LocalMLP": "qex.functionals:LocalMLP",
    "GlobalMLP": "qex.functionals:GlobalMLP",
    "DescriptorXC": "qex.functionals:DescriptorXC",
    "QCNN": "qex.functionals:QCNN",
    "make_eval_xc_local": "qex.functionals:make_eval_xc_local",
    "make_eval_xc_global": "qex.functionals:make_eval_xc_global",
    # scf
    "rks_loss": "qex.scf:rks_loss",
    "rks_loss_scan": "qex.scf:rks_loss_scan",
    "rks_energy": "qex.scf:rks_energy",
    # data_io
    "DataGenerator": "qex.data_io:DataGenerator",
    "MoleculeConfig": "qex.data_io:MoleculeConfig",
    "Datapoint": "qex.data_io:Datapoint",
    "QexDataset": "qex.data_io:QexDataset",
    "build_dataset": "qex.data_io:build_dataset",
    "save_dataset": "qex.data_io:save_dataset",
    "load_dataset": "qex.data_io:load_dataset",
    # training
    "train": "qex.training:train",
    "TrainHistory": "qex.training:TrainHistory",
    "run_experiment": "qex.training:run_experiment",
    "ExperimentResult": "qex.training:ExperimentResult",
    "evaluate_samples": "qex.training:evaluate_samples",
    "parity_plot": "qex.training:parity_plot",
    "calculate_dissociation_profile": "qex.training:calculate_dissociation_profile",
    "plot_dissociation_profile": "qex.training:plot_dissociation_profile",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str):
    """Lazily resolve a re-exported symbol on first access (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'qex' has no attribute {name!r}")
    module_name, _, attr = target.partition(":")
    value = getattr(import_module(module_name), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:
    # Give type checkers / IDEs the real symbols without paying the import cost
    # at runtime.
    from qex.config import Config, setup_config
    from qex.data_io import (
        DataGenerator,
        Datapoint,
        MoleculeConfig,
        QexDataset,
        build_dataset,
        load_dataset,
        save_dataset,
    )
    from qex.functionals import (
        DescriptorXC,
        GlobalMLP,
        LocalMLP,
        QCNN,
        make_eval_xc_global,
        make_eval_xc_local,
    )
    from qex.scf import rks_energy, rks_loss, rks_loss_scan
    from qex.training import (
        ExperimentResult,
        TrainHistory,
        calculate_dissociation_profile,
        evaluate_samples,
        parity_plot,
        plot_dissociation_profile,
        run_experiment,
        train,
    )
