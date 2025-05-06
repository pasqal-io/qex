"""Utilities for loading molecular datasets."""

from pathlib import Path
from typing import Any

import numpy as np
from jax_dft import datasets, utils
from loguru import logger


def load_molecular_datasets(
    molecule_names: list[str],
    datasets_params: dict[str, list[float]],
    num_grids: int | dict[str, int],
    base_path: str | Path | None = None,
    check_grid_centering: bool = True,
) -> list[tuple[datasets.Dataset, Any]]:
    """Load multiple molecular datasets with validation.

    This function loads datasets for a list of molecules, validating the
    input parameters and ensuring that the datasets are correctly configured.

    For every molecule, the dataset is loaded with specified distances
    in x100 distance units. For example, a distance of 1.74 is represented
    as 174.

    Example:
        molecule_names = ['h2']
        datasets_params = {'dataset1': [174, 348]}
        num_grids = {'h2': 512}
        base_path = Path('data/od')

    Args:
        molecule_names: A list of molecule names corresponding to dataset paths.
        datasets_params: A dictionary containing dataset parameters keyed by
            'dataset1', 'dataset2', etc. Each value should be a list of distances.
        num_grids: Either a single integer for all datasets, or a dictionary mapping
            molecule names to their specific grid sizes.
        base_path: An optional base path where molecule directories are located.
            If None, uses relative paths directly with molecule names.
        check_grid_centering: A boolean indicating whether to verify that molecules
            are centered on the grid.

    Returns:
        A list of tuples, each containing a Dataset object and a training set
        for each molecule.

    Raises:
        ValueError: If the number of molecules doesn't match the dataset parameters,
            or if molecules aren't centered
            on the grid.
    """

    if len(molecule_names) != len([k for k in datasets_params.keys() if k.startswith("dataset")]):
        raise ValueError(
            "Number of molecule names must match number of datasets in params. "
            f"Got {len(molecule_names)} names and {len(datasets_params)} datasets.",
        )

    loaded_datasets = []

    for idx, mol_name in enumerate(molecule_names, 1):
        logger.info(f"Loading dataset for {mol_name}")

        # Determine num_grids for this molecule
        current_num_grids = num_grids[mol_name] if isinstance(num_grids, dict) else num_grids

        # Construct path
        if base_path is not None:
            mol_path = Path(base_path) / mol_name
        else:
            mol_path = mol_name

        # ╔════════════════════════════════════════════════════════════════╗
        # ║               Dataset Loading & Preparation                    ║
        # ║               -------------------------                        ║
        # ║ This section handles loading the molecular dataset from        ║
        # ║ disk and preparing the training set with atomic distances      ║
        # ╚════════════════════════════════════════════════════════════════╝

        try:
            # Convert to Path object to str
            logger.info(f"Loading dataset from {mol_path}")
            dataset = datasets.Dataset(
                path=f"{mol_path}/",
                num_grids=current_num_grids,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset for {mol_name} from path {mol_path}",
            ) from e

        # Get training distances for this molecule
        distances = datasets_params[f"dataset{idx}"]
        logger.info(f"Training distances: {distances}")

        # Get molecule set
        train_set = dataset.get_molecules(distances)

        # Print dataset info
        logger.info(f"Number of electrons: {dataset.num_electrons}")
        logger.info(f"Grid shape: {dataset.grids.shape}")

        # Verify grid centering if requested
        if check_grid_centering:
            if not np.all(
                utils.location_center_at_grids_center_point(
                    train_set.locations,
                    dataset.grids,
                ),
            ):
                raise ValueError(
                    f"Training set for {mol_name} contains examples not "
                    "centered at the center of the grids.",
                )

        loaded_datasets.append((dataset, train_set))

    return loaded_datasets


def load_molecular_datasets_from_config(
    config: dict[str, Any],
    base_path: str | Path,
    check_grid_centering: bool = True,
) -> list[tuple[datasets.Dataset, Any]]:
    """
    Load molecular datasets directly from a configuration object.

    This function extracts dataset-related parameters from a configuration
    dictionary and loads the datasets using these parameters.

    Args:
        config: A configuration dictionary containing dataset parameters.
        base_path: The path to the data directory.

    Returns:
        A list of loaded datasets, each represented as a tuple containing
        a Dataset object and a training set.
    """
    # Extract all dataset-related parameters
    datasets_params = {
        key: value
        for key, value in config.items()
        if "dataset" in key.lower() and key != "dataset"
    }

    # Create hparams dictionary
    hparams = {
        "molecule_names": config["molecule_names"],
        "datasets_params": datasets_params,
        "num_grids": config["num_grids"],
    }

    # Load datasets using existing functionality
    return load_molecular_datasets(
        molecule_names=hparams["molecule_names"],
        datasets_params=hparams["datasets_params"],
        num_grids=hparams["num_grids"],
        base_path=base_path,
        check_grid_centering=check_grid_centering,
    )


if __name__ == "__main__":

    import os

    import qedft

    config = {
        "molecule_names": ["h2"],
        "num_grids": {"h2": 513},
        "dataset1": [128, 384],
    }
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
    load_molecular_datasets_from_config(
        config,
        project_path / "data/od",
        check_grid_centering=True,
    )
