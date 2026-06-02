"""Data generation and I/O: reference electronic-structure data via PySCF(AD).

Reference data can be generated on the fly (:class:`DataGenerator`) or stored as
a single self-describing HDF5 file (:class:`QexDataset` + :func:`save_dataset` /
:func:`load_dataset`) -- one file for the whole train/val/test split, the moral
equivalent of a PyTorch ``.pt``.
"""

from qex.data_io.dataset import (
    Datapoint,
    QexDataset,
    append_datapoint,
    build_dataset,
    datapoint_uid,
    existing_uids,
    failed_uids,
    init_dataset_file,
    load_dataset,
    read_config_hash,
    save_dataset,
)
from qex.data_io.dataset_generation import DataGenerator, MoleculeConfig
from qex.data_io.systems_file import load_systems_file, systems_file_hash

__all__ = [
    "DataGenerator",
    "MoleculeConfig",
    "load_systems_file",
    "systems_file_hash",
    "Datapoint",
    "QexDataset",
    "datapoint_uid",
    "build_dataset",
    "save_dataset",
    "load_dataset",
    "read_config_hash",
    "init_dataset_file",
    "append_datapoint",
    "existing_uids",
    "failed_uids",
]
