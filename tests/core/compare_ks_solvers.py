"""End-to-end validation of the KS solvers through the full qex SCF loop.

Standalone script (NOT a `test_*` file, so pytest does not collect it — same
convention as `compare_eigensolvers.py`). It is SLOW: each case builds the full
O(N^4) ERI tensor and JIT-compiles the SCF loop, so the whole matrix takes
minutes. Run it on demand when you change a solver or the loop, not in CI.

For each (system, functional, solver) it drives `qex.scf.rks_energy` with a
standard libxc functional via `encoding="libxc"` and checks the converged
energy against:
  1. PySCF's own `dft.RKS` energy, and
  2. the qex dense-solver energy through the identical loop.

Both LDA and B3LYP are exercised; integer aufbau (`frac_enabled=0`), matching
vanilla closed-shell RKS (purification builds the idempotent density directly).

Usage
-----
    python tests/core/compare_ks_solvers.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import dft, gto

from qex.functionals.libxc_veff import (
    make_eval_xc_libxc_ref,
    make_libxc_ingredients,
)
from qex.scf.rks import rks_energy

jax.config.update("jax_enable_x64", True)


# cc-pVTZ so LOBPCG's 5*n_occ < nao constraint holds for all three systems.
_SYSTEMS = [
    ("H2", "H 0 0 0; H 0 0 0.74"),
    ("LiH", "Li 0 0 0; H 0 0 1.60"),
    ("H2O", "O 0 0 0; H 0 0.96 0; H 0 -0.24 0.93"),
]
_FUNCTIONALS = [("LDA", "lda,vwn"), ("B3LYP", "b3lyp")]
_SOLVERS = ["dense", "lobpcg", "purify"]


def _pyscf_reference(atom, xc_code):
    mol = gto.M(atom=atom, basis="cc-pVTZ")
    mf = dft.RKS(mol)
    mf.xc = xc_code
    mf.grids.level = 3
    mf.grids.build()
    e_pyscf = float(mf.kernel())
    inputs = dict(
        dm=jnp.asarray(mf.get_init_guess()),
        eri=jnp.asarray(mol.intor("int2e", aosym="s1")),
        ao_grid=jnp.asarray(mol.eval_gto("GTOval_sph", mf.grids.coords)),
        grid_weights=jnp.asarray(mf.grids.weights),
        s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=float(mol.energy_nuc()),
        nelectron=int(mol.nelectron),
    )
    return e_pyscf, mol, mf.grids.coords, inputs


def _qex_energy(xc_code, mol, coords, inp, solver):
    ingredients = make_libxc_ingredients(mol, coords, xc_code)
    return float(
        rks_energy(
            ingredients,
            inp["dm"], inp["eri"], inp["ao_grid"], inp["grid_weights"],
            inp["s1e"], inp["h1e"], inp["energy_nuc"], inp["nelectron"],
            xc_eval_fn=make_eval_xc_libxc_ref(xc_code),
            encoding="libxc",
            frac_enabled=0,
            solver=solver,
            max_cycle=50,
            use_diis=True,
        )
    )


def main():
    print(f"{'system':6s} {'xc':6s} {'solver':7s} "
          f"{'E_qex':>16s} {'dE(pyscf)':>12s} {'dE(dense)':>12s}")
    n_fail = 0
    for sys_label, atom in _SYSTEMS:
        for fxc_label, xc_code in _FUNCTIONALS:
            e_pyscf, mol, coords, inp = _pyscf_reference(atom, xc_code)
            e_dense = _qex_energy(xc_code, mol, coords, inp, "dense")
            for solver in _SOLVERS:
                e = (e_dense if solver == "dense"
                     else _qex_energy(xc_code, mol, coords, inp, solver))
                d_pyscf = e - e_pyscf
                d_dense = e - e_dense
                ok = abs(d_pyscf) < 1e-6 and abs(d_dense) < 1e-7
                n_fail += not ok
                flag = "" if ok else "  <-- FAIL"
                print(f"{sys_label:6s} {fxc_label:6s} {solver:7s} "
                      f"{e:16.8f} {d_pyscf:12.2e} {d_dense:12.2e}{flag}")
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
