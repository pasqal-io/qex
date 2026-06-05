"""High-level experiment driver: config in, trained functional + metrics out.

:func:`run_experiment` is the single code path shared by the ``qex train`` CLI
and the example scripts. It takes a :class:`qex.config.Config` (loaded from
YAML) and runs the full pipeline end to end:

    build network -> generate reference data -> train -> evaluate

Everything that used to live in the module-level ``CONFIG`` dict of
``examples/train_h2.py`` is now read from the config object, so a run is fully
described by a YAML file (and any CLI overrides on top of it).

The molecule sweep is currently specialised to H2 (a bond-length scan), matching
the published example. Swapping in another molecule means providing a different
``molecule_config_factory``; the rest of the pipeline is geometry-agnostic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import random
from loguru import logger

from qex.config import Config
from qex.data_io import (
    DataGenerator,
    MoleculeConfig,
    QexDataset,
    build_dataset,
    load_dataset,
    read_config_hash,
    reference_vxc,
)
from qex.functionals import (
    DescriptorXC,
    GlobalMLP,
    LocalMLP,
    QCNN,
    make_eval_xc_global,
    make_eval_xc_local,
    select,
)
from qex.scf import get_ao_value, rks_energy, rks_loss_scan
from qex.scf.inputs import SCFInputs
from qex.training.evaluate import (
    calculate_dissociation_profile,
    evaluate_dataset_split,
    evaluate_samples,
    parity_plot,
    plot_dissociation_profile,
)
from qex.training.train import TrainHistory, train
from qex.utils.log import configure_logging

# SCF kwargs that the DIIS loop understands but the scan loop does not. They are
# stripped before training (which uses `rks_loss_scan`) and kept for evaluation.
_DIIS_ONLY_KEYS = ("diis_max_vec", "diis_min_vec", "diis_start_cycle", "diis_damping")

# SCF kwargs the training loss understands but the eval energy fn does not.
# `differentiation` only selects how gradients flow through the SCF fixed point,
# which is meaningless for the forward-only `rks_energy`; strip it for eval.
_TRAIN_ONLY_KEYS = ("differentiation", "target_weights")


@dataclass
class ExperimentResult:
    """Outcome of :func:`run_experiment`.

    Attributes:
        params: Trained network parameters (lowest-val-loss when validation ran).
        test_predicted: Predicted energies on the held-out test set (Hartree).
        test_reference: Reference energies on the test set (Hartree).
        test_metrics: ``{"mae", "npe", "max_abs"}`` on the test set (Hartree).
        bond_lengths: Bond lengths the dissociation profile was evaluated on
            (``None`` if the dissociation profile was skipped).
        ml_energies / ref_energies: Predicted / reference energies along the
            dissociation profile (``None`` if skipped).
        history: Per-iteration training/validation history.
        output_dir: Directory results/plots were written to.
    """

    params: dict
    test_predicted: np.ndarray
    test_reference: np.ndarray
    test_metrics: dict[str, float]
    bond_lengths: np.ndarray | None
    ml_energies: np.ndarray | None
    ref_energies: np.ndarray | None
    history: TrainHistory
    output_dir: str

    @property
    def mae(self) -> float:
        """Test-set mean absolute error (Hartree)."""
        return self.test_metrics["mae"]

    @property
    def npe(self) -> float:
        """Test-set non-parallelity error (Hartree)."""
        return self.test_metrics["npe"]


def _scf_kwargs(config: Config) -> dict[str, Any]:
    """Collect SCF/fractional-occupation kwargs from the config."""
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
        # KS "solve" step: "dense" (default generalized_eigh), "lobpcg"
        # (lowest-N_occ iterative), or "purify" (McWeeny purification; integer
        # aufbau only, needs scf.frac_enabled=0). Threaded through both the scan
        # training loop and the DIIS eval loop.
        solver=config.get("scf.solver", "dense"),
        # SCF gradient mode for training: "unroll" (backprop through every cycle)
        # or "implicit" (implicit-function-theorem on the fixed point; cost
        # independent of max_cycle). Eval (`rks_energy`) ignores it.
        differentiation=config.get("scf.differentiation", "unroll"),
        # Per-target loss weights for the extra reference targets scored at the
        # converged state (vxc, dm, ...). `energy`/`density` keep their own
        # dedicated weights (training.energy_weight / density_weight). A weight of
        # 0 (the default for vxc/dm) switches a target off. Passed as a hashable
        # tuple of (name, weight) pairs so it can be a static jit argument.
        target_weights=_target_weights(config),
    )


# Non-energy loss terms, all scored at the converged state and switched on by a
# non-zero weight (see qex.scf.targets). `energy` is the one always-on term and
# keeps its own `training.energy_weight`. `density` defaults ON (weight 1.0, or
# the legacy `training.density_weight`); `vxc`/`dm` default OFF (0.0).
_CONVERGED_TARGETS = ("density", "vxc", "dm")


def _target_weights(config: Config) -> tuple[tuple[str, float], ...]:
    """Read per-target loss weights from config as a hashable (name, weight) tuple.

    One map drives every non-energy loss: ``training.target_weights`` keyed by
    target name. A target is *on* iff its weight is non-zero. Defaults: ``density``
    is on (weight 1.0, or the legacy ``training.density_weight`` if set), while
    ``vxc``/``dm`` are off (0.0) until you give them a weight. ``energy`` is not
    here — it is always on via ``training.energy_weight``.
    """
    weights = config.get("training.target_weights", {}) or {}
    defaults = {
        "density": config.get("training.density_weight", 1.0),
        "vxc": 0.0,
        "dm": 0.0,
    }
    return tuple(
        (name, float(weights.get(name, defaults[name]))) for name in _CONVERGED_TARGETS
    )


# Converged-state targets that are NOT always stored (need the reference method to
# produce them, e.g. a KS Vxc). `dm` is a core field and always present, so it is
# not listed here. Mirrors qex.data_io.dataset._OPTIONAL_TARGET_FIELDS.
_OPTIONAL_TARGET_NAMES = ("vxc",)


def _ensure_optional_targets_available(
    dataset: QexDataset,
    config: Config,
    data_generator: DataGenerator,
    *,
    splits: tuple[str, ...],
    dataset_file: str | None,
) -> None:
    """Guarantee every requested optional target is present on the training data.

    A target is *requested* when its ``training.target_weights`` entry is > 0.
    For an optional target (``vxc``), a dataset built before the target existed
    will not carry it. This is caught up front:

    - With ``data.recompute_missing_targets: true`` (default false), the missing
      target is recomputed from PySCF per affected sample (``reference_vxc``) and
      written onto the in-memory datapoints so training proceeds. (This recompute
      is per-run and not persisted back to the ``.h5``; regenerate the dataset to
      store it permanently.)
    - Otherwise it raises a clear, actionable error rather than silently training
      against a target that is absent for some samples.
    """
    on = {name for name, w in _target_weights(config) if w > 0.0}
    requested_optional = [n for n in _OPTIONAL_TARGET_NAMES if n in on]
    if not requested_optional:
        return

    recompute = config.get("data.recompute_missing_targets", False)
    for name in requested_optional:
        missing = [
            dp for split in splits for dp in dataset.converged(split)
            if getattr(dp, name) is None
        ]
        if not missing:
            continue
        if not recompute:
            where = f" at {dataset_file}" if dataset_file else ""
            raise ValueError(
                f"target_weights[{name!r}] > 0 but {len(missing)} training/val "
                f"sample(s) in the dataset{where} have no {name!r} target "
                f"(it was generated before {name!r} was added, or the reference "
                f"method does not produce it). Regenerate the dataset, set "
                f"data.recompute_missing_targets: true to recompute {name!r} via "
                f"PySCF, or set target_weights.{name} to 0 to disable it."
            )
        logger.info(
            "Recomputing missing {!r} target for {} sample(s) via PySCF...",
            name, len(missing),
        )
        for dp in missing:
            _recompute_target(dp, name, data_generator)


def _recompute_target(dp, name: str, data_generator: DataGenerator) -> None:
    """Recompute one optional target on a datapoint in place (PySCF).

    Re-runs the reference for ``dp.meta`` and fills ``dp.<name>``. Only ``vxc`` is
    supported; ``vxc`` is undefined for a non-KS reference, which is reported as a
    clear error (such a dataset cannot supply a Vxc target at all).
    """
    if name != "vxc":
        raise ValueError(f"No recompute rule for target {name!r}.")
    mol, mf, dm, _energy, _density, _coords = data_generator.generate_data(dp.meta)
    vxc = reference_vxc(mol, mf, dm, dp.meta.method)
    if vxc is None:
        raise ValueError(
            f"Cannot recompute a 'vxc' target for {dp.meta.name!r}: its reference "
            f"method {dp.meta.method!r} has no Kohn-Sham Vxc (only method='rks' "
            f"defines one). Use an RKS reference or disable the vxc target."
        )
    dp.vxc = np.asarray(vxc)


def build_network(config: Config) -> tuple[nn.Module, Callable, dict, bool]:
    """Instantiate the XC network, its eval wrapper, and initial params.

    This is the single place that maps ``model.*`` config keys onto a concrete
    network (``descriptor`` / ``mlp`` / ``qcnn``), so the CLI driver and the
    example scripts build networks identically. Returns
    ``(network, xc_eval_fn, params, is_descriptor)``; ``is_descriptor`` flags
    whether the SCF loop must be passed grid/atom-coordinate context.
    """
    model = config.get("model.type", "descriptor")
    encoding = config.get("model.encoding", "global")
    rng = config.get("rng", 42)
    n_grid = config.get("model.n_grid", 1240)
    n_atom = config.get("model.n_atom", 2)
    hidden = list(config.get("model.hidden_layers", [128, 128, 128, 128, 128]))
    vxc_grad_scale = config.get("model.vxc_grad_scale", 1.0)

    is_descriptor = model == "descriptor"
    is_qcnn = model == "qcnn"
    if is_descriptor and encoding != "global":
        raise ValueError("DescriptorXC requires model.encoding='global'.")
    if is_qcnn and encoding != "global":
        raise ValueError("QCNN requires model.encoding='global'.")

    if is_qcnn:
        gate_noise = _build_qcnn_noise(config)
        network = QCNN(
            n_qubits=config.get("model.qcnn.n_qubits", 2),
            n_features=config.get("model.qcnn.n_features", 2),
            n_layers=config.get("model.qcnn.n_layers", 2),
            n_var_layers=config.get("model.qcnn.n_var_layers", 1),
            feature_map=config.get("model.qcnn.feature_map", "direct"),
            head_features=tuple(config.get("model.qcnn.head_features", [32])),
            noise=gate_noise,
            gaussian_noise_std=config.get("model.qcnn.gaussian_noise_std", 0.0),
        )
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=vxc_grad_scale)
        params = network.init(random.PRNGKey(rng), jnp.ones(n_grid))
    elif is_descriptor:
        network = DescriptorXC(
            hidden=hidden,
            alphas=tuple(config.get("model.descriptor.alphas", [0.5, 1.0, 2.0, 4.0])),
            scale=config.get("model.descriptor.scale", 1.0),
            rho_floor=config.get("model.descriptor.rho_floor", 1e-10),
            act_fn=nn.gelu,
        )
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=vxc_grad_scale)
        params = network.init(
            random.PRNGKey(rng),
            jnp.ones(n_grid),
            jnp.ones(n_grid) / n_grid,
            grid_coords=jnp.zeros((n_grid, 3)),
            atom_coords=jnp.zeros((n_atom, 3)),
        )
    elif encoding == "local":
        network = LocalMLP(features=hidden, act_fn=nn.gelu)
        xc_eval_fn = make_eval_xc_local(network, vxc_grad_scale=vxc_grad_scale)
        params = network.init(random.PRNGKey(rng), jnp.ones(n_grid))
    elif encoding == "global":
        network = GlobalMLP(features=hidden, act_fn=nn.gelu)
        xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=vxc_grad_scale)
        params = network.init(random.PRNGKey(rng), jnp.ones(n_grid))
    else:
        raise ValueError(f"Unknown model.encoding: {encoding!r}")

    return network, xc_eval_fn, params, is_descriptor


def _build_qcnn_noise(config: Config):
    """Translate the config's gate-noise spec into horqrux noise instances."""
    spec = config.get("model.qcnn.gate_noise", []) or []
    if not spec:
        return None
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    return tuple(
        DigitalNoiseInstance(DigitalNoiseType[kind.upper()], prob) for kind, prob in spec
    )


def h2_molecule_config_factory(config: Config) -> Callable[..., MoleculeConfig]:
    """Build the H2 ``MoleculeConfig`` factory used by data gen and evaluation.

    The returned ``factory(bond_length, *, method, basis, units, grid_density)``
    is what :func:`assemble_training_data` and
    :func:`qex.training.calculate_dissociation_profile` call per geometry.
    """

    def factory(bond_length, *, method, basis, units, grid_density, verbose=0):
        return MoleculeConfig(
            name=f"H2_{bond_length:.2f}",
            atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
            units=units,
            basis=basis,
            method=method,
            grid_density=grid_density,
            verbose=verbose,
        )

    return factory


# Default bond lengths per split (Angstrom). Overridable via
# ``data.{train,val,test}_bond_lengths`` in the config.
_DEFAULT_SPLIT_BOND_LENGTHS = {
    "train": [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0],
    "val": [0.6, 0.9, 1.2],
    "test": [0.8, 1.3, 1.75, 2.25, 2.75],
}


def molecule_configs_for_split(
    config: Config,
    molecule_config_factory: Callable,
    split: str,
) -> list[MoleculeConfig]:
    """Build the list of :class:`MoleculeConfig` for a train/val/test split.

    Geometries come from ``data.{split}_bond_lengths`` (explicit lists in the
    config), falling back to sensible defaults per split.
    """
    bond_lengths = list(
        config.get(
            f"data.{split}_bond_lengths",
            _DEFAULT_SPLIT_BOND_LENGTHS.get(split, []),
        )
    )
    method = config.get("data.method", "ccsd")
    basis = config.get("data.basis", "631g")
    units = config.get("data.units", "Ang")
    grid_density = config.get("data.grid_density", 0)
    verbose = config.get("data.verbose", 0)
    return [
        molecule_config_factory(
            d,
            method=method,
            basis=basis,
            units=units,
            grid_density=grid_density,
            verbose=verbose,
        )
        for d in bond_lengths
    ]


def _pack_sample(
    data_generator: DataGenerator,
    cfg: MoleculeConfig,
    required_features: tuple[str, ...],
) -> SCFInputs:
    """Generate one molecule's data and pack it into an :class:`SCFInputs`.

    The named feature bag is narrowed to the network's ``required_features``
    (see :mod:`qex.functionals.features`); the reference-target bag carries the
    energy/density/dm (and ``vxc`` for a KS reference) the loss regresses against
    (see :mod:`qex.scf.targets`).
    """
    mol, mf, dm, energy, density, coords = data_generator.generate_data(cfg)
    available = {
        "grid_coords": jnp.asarray(mf.grids.coords),
        "atom_coords": jnp.asarray(mol.atom_coords()),
    }
    targets = {
        "energy": jnp.asarray(float(energy)),
        "density": jnp.asarray(density),
        "dm": jnp.asarray(dm),
    }
    vxc = reference_vxc(mol, mf, dm, cfg.method)
    if vxc is not None:
        targets["vxc"] = jnp.asarray(vxc)
    return SCFInputs(
        dm=jnp.asarray(dm),
        eri=jnp.asarray(mol.intor("int2e", aosym="s1")),
        ao_grid=jnp.asarray(get_ao_value(mol, mf.grids.coords)),
        grid_weights=jnp.asarray(mf.grids.weights),
        s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=jnp.asarray(float(mol.energy_nuc())),
        nelectron=jnp.asarray(int(mol.nelectron)),
        features=select(available, required_features),
        targets=targets,
    )


def assemble_training_data(
    config: Config,
    data_generator: DataGenerator,
    molecule_config_factory: Callable,
    required_features: tuple[str, ...],
    split: str = "train",
) -> list:
    """Generate reference data for ``split`` as a list of :class:`SCFInputs` (inline).

    This is the **no-file** route: it generates and packs straight to
    :class:`SCFInputs` in memory, nothing is written to disk. Use it when you
    want the bundles without the HDF5 dataset layer.

    For the normal pipeline prefer :func:`dataset_for_config`, which is
    file-backed, **resumable**, and records non-converged systems -- then call
    :meth:`QexDataset.training_inputs`. The two produce equivalent bundles; this
    one trades persistence/resumability for simplicity. The geometries for
    ``split`` are taken from ``data.{split}_bond_lengths``.
    """
    molecule_configs = molecule_configs_for_split(
        config, molecule_config_factory, split
    )
    return [
        _pack_sample(data_generator, cfg, required_features)
        for cfg in molecule_configs
    ]


# --------------------------------------------------------------------------- #
# Single-file dataset: build / cache
# --------------------------------------------------------------------------- #
# Config keys that change the *contents* of the generated data. The cache is
# keyed on these so a stale file (different basis, geometries, ...) is rebuilt.
_DATA_KEYS = (
    "data.method",
    "data.basis",
    "data.units",
    "data.grid_density",
    "data.train_bond_lengths",
    "data.val_bond_lengths",
    "data.test_bond_lengths",
)


def dataset_config_hash(config: Config, is_descriptor: bool) -> str:
    """Stable fingerprint of the data-affecting config (for auto-caching)."""
    payload = {k: config.get(k) for k in _DATA_KEYS}
    payload["descriptor_ctx"] = bool(is_descriptor)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def dataset_for_config(
    config: Config,
    *,
    molecule_config_factory: Callable | None = None,
    is_descriptor: bool = False,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> QexDataset:
    """Build (incrementally, resumably) the train/val/test :class:`QexDataset`.

    The ``.h5`` file *is* the cache and the checkpoint: generation writes each
    system as it finishes, so a long run that is killed or whose SCF diverges can
    simply be restarted -- completed systems are reused and only the missing ones
    are computed. Non-converged systems are recorded (not silently lost) and
    skipped by training; set ``data.retry_failed: true`` to recompute them.

    The data-affecting config is fingerprinted (:func:`dataset_config_hash`); if
    a cached file was built for a *different* config, it is rebuilt from scratch.
    With ``use_cache=False`` the dataset is built in memory and not persisted.
    """
    if molecule_config_factory is None:
        molecule_config_factory = h2_molecule_config_factory(config)

    output_dir = config.get("output_dir", "results/h2_dissociation")
    if cache_path is None:
        cache_path = Path(output_dir) / "dataset.h5"
    cache_path = Path(cache_path)

    current_hash = dataset_config_hash(config, is_descriptor)
    retry_failed = bool(config.get("data.retry_failed", False))

    data_generator = DataGenerator(output_dir)
    split_configs = {
        split: molecule_configs_for_split(config, molecule_config_factory, split)
        for split in ("train", "val", "test")
    }

    if not use_cache:
        logger.info(
            "Generating dataset in memory -> {}",
            " | ".join(f"{s}: {len(c)}" for s, c in split_configs.items()),
        )
        return build_dataset(
            data_generator, split_configs, with_descriptor_ctx=is_descriptor
        )

    # A cached file built for a different config is not reusable (uids differ by
    # geometry but basis/method changes would otherwise pile up stale entries).
    if cache_path.exists() and read_config_hash(cache_path) != current_hash:
        logger.info(
            "Cached dataset at {} is stale (config changed); rebuilding.", cache_path
        )
        cache_path.unlink()

    logger.info(
        "Preparing dataset (resumable) -> {}  [{}]",
        " | ".join(f"{s}: {len(c)}" for s, c in split_configs.items()),
        cache_path,
    )
    dataset = build_dataset(
        data_generator,
        split_configs,
        path=cache_path,
        with_descriptor_ctx=is_descriptor,
        config_hash=current_hash,
        retry_failed=retry_failed,
    )
    # Report convergence outcome so failures are visible, not silent.
    for split in ("train", "val", "test"):
        nfail = dataset.n_failed(split)
        if nfail:
            logger.warning(
                "{}: {} system(s) failed to converge (excluded).", split, nfail
            )
    return dataset


def dataset_from_file(path: str | Path) -> QexDataset:
    """Load a pre-built dataset ``.h5`` for training (no generation, no PySCF).

    This is the decoupled path: a dataset built earlier (``qex gen-data
    --systems ...`` or :func:`qex.data_io.build_dataset`) is read straight back
    in. Training, validation, and test-set evaluation all run from the stored
    tensors, so no reference solver is invoked at train time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data.dataset_file points to a missing file: {path}. Build it first, "
            "e.g. `qex gen-data --systems <systems.yaml> -o <path>`."
        )
    dataset = load_dataset(path)
    for split in ("train", "val", "test"):
        nfail = dataset.n_failed(split)
        if nfail:
            logger.warning(
                "{}: {} non-converged record(s) in dataset (excluded).", split, nfail
            )
    return dataset


def run_experiment(
    config: Config,
    *,
    molecule_config_factory: Callable | None = None,
    make_plot: bool = True,
) -> ExperimentResult:
    """Run the full train + evaluate pipeline described by ``config``.

    Pipeline: build network -> get train/val/test data -> train (with periodic
    validation + optional early stopping) -> evaluate on the held-out test set
    (parity plot) and, when configured, the dissociation profile.

    Data comes from one of two routes, chosen automatically:

    - **Pre-built dataset** (decoupled): if ``data.dataset_file`` names an
      existing ``.h5`` (e.g. from ``qex gen-data --systems ...``), it is loaded
      and train/val/test all run from the stored tensors -- no PySCF, no geometry
      factory, no dissociation curve. This is the simple, reproducible path.
    - **Config/factory-driven** (the H2 example): geometries per split come from
      ``data.{train,val,test}_bond_lengths`` via ``molecule_config_factory``;
      the dataset is generated (and cached at ``data.dataset_file``) on the fly,
      and the dissociation curve is available.

    Args:
        config: A :class:`qex.config.Config`. See ``examples/h2.yaml`` (factory
            path) or ``examples/train_from_dataset.yaml`` (pre-built path).
        molecule_config_factory: Optional override for the molecule geometry
            factory (factory path only). Defaults to an H2 bond-length scan.
        make_plot: Whether to render/save the evaluation plots.

    Returns:
        An :class:`ExperimentResult` with the trained params, test-set parity
        arrays + metrics, optional dissociation profile, and training history.
    """
    # Quiet by default. `logging.level` (e.g. TRACE/DEBUG/INFO) is the global
    # verbosity knob across the whole pipeline (data gen, SCF, eval); `debug:
    # true` is a shortcut for level=DEBUG. An explicit level wins.
    configure_logging(
        debug=bool(config.get("debug", False)),
        level=config.get("logging.level", None),
    )

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", config.get("platform", "cpu"))

    output_dir = config.get("output_dir", "results/h2_dissociation")
    method = config.get("data.method", "ccsd")
    # Ensure the results directory exists up front. The factory path creates it
    # implicitly (DataGenerator writes dataset.h5 there); the dataset-only path
    # loads the .h5 from elsewhere and would otherwise reach savefig with no dir.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting experiment -> output_dir={}", output_dir)
    logger.debug(
        "jax x64 enabled; platform={}", config.get("platform", "cpu")
    )

    network, xc_eval_fn, params, is_descriptor = build_network(config)
    # The network declares exactly which named features it consumes; the whole
    # pipeline (generation, selection, eval) keys off this instead of model-type
    # booleans. A plain density model declares `()` (see qex.functionals.features).
    required_features = tuple(getattr(network, "required_features", ()))
    needs_features = bool(required_features)
    logger.info(
        "Network built: type={} encoding={} required_features={}",
        config.get("model.type", "descriptor"),
        config.get("model.encoding", "global"),
        required_features,
    )

    if molecule_config_factory is None:
        molecule_config_factory = h2_molecule_config_factory(config)

    data_generator = DataGenerator(output_dir)

    # --- Train/val/test split ---------------------------------------------------
    # Decoupled path: if `data.dataset_file` already exists on disk, train from
    # it directly -- no geometry factory, no PySCF, no dissociation curve. This
    # is the `qex gen-data --systems ... -o data.h5` then `qex train` workflow.
    # Otherwise fall back to the config/factory-driven (auto-caching) generation,
    # which the H2 example uses.
    dataset_file = config.get("data.dataset_file", None)
    from_file = dataset_file is not None and Path(dataset_file).exists()
    if from_file:
        logger.info("Loading pre-built dataset -> {} (no generation)", dataset_file)
        dataset = dataset_from_file(dataset_file)
    else:
        dataset = dataset_for_config(
            config,
            molecule_config_factory=molecule_config_factory,
            is_descriptor=needs_features,
            cache_path=dataset_file,
            use_cache=config.get("data.cache_dataset", True),
        )
    # If a converged-state target (e.g. vxc) is switched on (weight > 0), make
    # sure every training/val sample actually carries it — recomputing via PySCF
    # when allowed, or failing loudly with an actionable message otherwise.
    _ensure_optional_targets_available(
        dataset, config, data_generator, splits=("train", "val"),
        dataset_file=dataset_file if from_file else None,
    )

    # Each sample's feature bag is narrowed to exactly the keys this network
    # declared (`required_features`), so the vmapped SCF loop never hands a model
    # a feature it can't accept and one .h5 serves any model. A missing key fails
    # loudly by name inside `select`; surface a rebuild hint when that happens.
    try:
        training_data = dataset.training_inputs("train", required_features)
        val_data = dataset.training_inputs("val", required_features)
    except KeyError as exc:
        raise ValueError(
            f"This model requires features {list(required_features)}, but the "
            f"dataset" + (f" at {dataset_file}" if from_file else "") + " does "
            "not carry them all. Rebuild it so it stores these features (e.g. "
            "`qex gen-data --systems <systems.yaml> -o <path>` stores grid/atom "
            "coords), or train a model that does not require them."
        ) from exc
    test_points = dataset.converged("test")
    logger.info(
        "Data split -> train: {} | val: {} | test: {}",
        len(training_data),
        len(val_data),
        len(test_points),
    )

    optimizer = optax.adam(config.get("training.learning_rate", 1e-4))
    grad_clip = config.get("training.grad_clip", 0.5)
    if grad_clip is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optimizer)

    # `rks_loss_scan` uses lax.scan over SCF cycles (no DIIS) so compile time is
    # roughly independent of `max_cycle`. Strip DIIS-only kwargs it can't take.
    scf_kwargs = _scf_kwargs(config)
    train_scf_kwargs = {k: v for k, v in scf_kwargs.items() if k not in _DIIS_ONLY_KEYS}
    # Eval uses the forward-only `rks_energy`, which has no `differentiation` arg.
    eval_scf_kwargs = {k: v for k, v in scf_kwargs.items() if k not in _TRAIN_ONLY_KEYS}

    # --- Train with periodic validation + optional early stopping -------------
    logger.info(
        "Training -> {} iterations (val every {})",
        config.get("training.n_iterations", 1000),
        config.get("training.n_val_iter", 50),
    )
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

    # --- General evaluation on the held-out test set (parity) -----------------
    # From a pre-built dataset, evaluate from the stored tensors (no PySCF);
    # otherwise re-run the reference solver per test geometry.
    if from_file:
        test_predicted, test_reference, test_metrics = evaluate_dataset_split(
            trained_params,
            test_points,
            scf_energy_fn=rks_energy,
            xc_eval_fn=xc_eval_fn,
            required_features=required_features,
            label="test set",
            **eval_scf_kwargs,
        )
    else:
        test_predicted, test_reference, test_metrics = evaluate_samples(
            trained_params,
            data_generator,
            [dp.meta for dp in test_points],
            scf_energy_fn=rks_energy,
            xc_eval_fn=xc_eval_fn,
            required_features=required_features,
            label="test set",
            **eval_scf_kwargs,
        )
    logger.info(
        "Test set -> MAE {:.3e} Ha | NPE {:.3e} Ha",
        test_metrics["mae"],
        test_metrics["npe"],
    )
    if make_plot and len(test_points):
        logger.info("Making parity plot -> {}", output_dir)
        parity_plot(
            test_reference,
            test_predicted,
            title=f"Test set: predicted vs {method} reference",
            path_results=output_dir,
            show=True,
        )

    # --- Dissociation profile (curve-specific, optional) ----------------------
    # Geometry-parametric: it sweeps bond lengths via the molecule factory, so it
    # is meaningful only on the config/factory-driven path (e.g. the H2 example).
    # When training from a pre-built dataset there is no factory, so it is skipped
    # and the dataset-driven parity plot above is the evaluation.
    bond_lengths = ml_energies = ref_energies = None
    if from_file and config.get("data.eval_dissociation", False):
        logger.info(
            "Skipping dissociation profile: training from a pre-built dataset has "
            "no geometry factory (use the H2 example for the dissociation curve)."
        )
    if not from_file and config.get("data.eval_dissociation", True):
        eval_bond_lengths = np.linspace(
            config.get("data.eval_min", 0.5),
            config.get("data.eval_max", 3.0),
            config.get("data.eval_points", 30),
        )
        logger.info(
            "Dissociation profile -> {} bond lengths in [{:.2f}, {:.2f}]",
            len(eval_bond_lengths),
            float(eval_bond_lengths[0]),
            float(eval_bond_lengths[-1]),
        )
        bond_lengths, ml_energies, ref_energies = calculate_dissociation_profile(
            params=trained_params,
            data_generator=data_generator,
            molecule_config_factory=molecule_config_factory,
            scf_energy_fn=rks_energy,
            xc_eval_fn=xc_eval_fn,
            bond_lengths=eval_bond_lengths,
            method=method,
            basis=config.get("data.basis", "631g"),
            units=config.get("data.units", "Ang"),
            grid_density=config.get("data.grid_density", 0),
            verbose=config.get("data.verbose", 0),
            path_results=output_dir,
            required_features=required_features,
            **eval_scf_kwargs,
        )
        if make_plot:
            logger.info("Making dissociation profile plot -> {}", output_dir)
            plot_dissociation_profile(
                bond_lengths=bond_lengths,
                ml_energies=ml_energies,
                ref_energies=ref_energies,
                method=method,
                exp_model=config.get("model.exp_model", "density"),
                path_results=output_dir,
            )

    return ExperimentResult(
        params=trained_params,
        test_predicted=test_predicted,
        test_reference=test_reference,
        test_metrics=test_metrics,
        bond_lengths=None if bond_lengths is None else np.asarray(bond_lengths),
        ml_energies=None if ml_energies is None else np.asarray(ml_energies),
        ref_energies=None if ref_energies is None else np.asarray(ref_energies),
        history=history,
        output_dir=output_dir,
    )
