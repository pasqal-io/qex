"""Self-consistent-field (SCF) loops and supporting operators.

The public entry points are the differentiable RKS loss/energy functions used
for training and evaluation, plus the operator helpers needed to assemble SCF
inputs (effective potentials, AO values on the grid, density matrices).
"""

from qex.scf.operators import (
    energy_tot,
    get_ao_value,
    get_occ,
    get_veff,
    get_veff_global,
    get_veff_local,
    make_rdm1,
)
from qex.scf.rks import rks_energy, rks_loss, rks_loss_scan

__all__ = [
    "rks_loss",
    "rks_loss_scan",
    "rks_energy",
    "get_veff",
    "get_veff_local",
    "get_veff_global",
    "get_ao_value",
    "get_occ",
    "make_rdm1",
    "energy_tot",
]
