"""Standard (PySCF/libxc) functionals as a drop-in for qex's SCF ``get_veff``.

Purpose
-------
Validate the qex JAX SCF pipeline (the exact :func:`qex.scf.rks_energy` loop the
networks drive) against PySCF by running a *standard* functional -- LDA, GGA,
meta-GGA, or a global hybrid such as B3LYP -- through that same loop. If qex's
SCF iteration is correct, the converged energy must match a plain
``pyscf.dft.RKS`` run to (near) machine precision.

Why this is not the network path
--------------------------------
The network XC encodings (``local`` / ``global`` in
:mod:`qex.functionals.xc`) get ``rho`` as the density only and recover ``vrho``
by autodiff. A real GGA/meta-GGA/hybrid needs more *ingredients* that the
pure-functional JAX loop cannot pull off a PySCF ``mf`` object: density
gradients ``grad rho`` (for GGA), kinetic-energy density ``tau`` (for mGGA), and
the exact-exchange matrix ``K`` (for hybrids). PySCF carries all of this on
``mf``/``mol``; here every ingredient must be passed in explicitly.

This module assembles ``Vxc`` (and the hybrid ``K``) on NumPy via
``pyscf.dft.libxc``, reproducing PySCF's ``numint`` math:

* ``rho   = sum_g ao0 . dm . ao0``                                   (density)
* ``grad  = 2 * sum_g ao1 . dm . ao0``                               (GGA)
* ``tau   = 0.5 * sum_g ao1 . dm . ao1``                             (mGGA)
* ``Vxc``  from ``vrho`` / ``vsigma`` / ``vtau`` via the ``_rks_gga_wv0``
  weighting (verified to ~1e-15 against ``numint.nr_rks``).
* hybrids: ``K_ij = sum (ik|jl) dm_kl`` from the same ``eri`` qex already
  stores; subtract ``0.5 * hyb * K`` from the Fock and ``0.25 * hyb * <dm,K>``
  from the energy (closed shell).

It is **not** differentiable w.r.t. ``rho`` (libxc is NumPy) -- which is fine,
the goal is a fixed reference. A future JAX-native (``jax_xc``) backend can slot
in behind the same :func:`make_libxc_ingredients` / ``encoding="libxc"`` seam to
recover autodiff.

Ingredients carried in ``params``
----------------------------------
qex's loop threads an opaque ``params`` through ``get_veff`` untouched. For a
standard functional ``params`` is otherwise unused, so we pack the static
ingredients there (built once by :func:`make_libxc_ingredients`): the AO value
and gradient arrays on the grid, the libxc code, and the hybrid coefficient.
Only ``dm`` changes per SCF cycle, so ``rho``/``grad``/``tau`` are recomputed
each cycle from the cycle's ``dm`` -- exactly as PySCF does.
"""

from collections.abc import Callable

import numpy as np
from pyscf.dft import libxc


def make_libxc_ingredients(mol, coords, xc_code: str) -> dict:
    """Precompute the static (dm-independent) ingredients for a libxc functional.

    Returns a ``params``-style dict to hand to :func:`qex.scf.rks_energy` with
    ``encoding="libxc"``. The AO arrays are evaluated at the derivative order the
    functional needs (LDA: value only; GGA: +gradient; mGGA: +second
    derivatives for ``tau``).

    Args:
        mol: PySCF ``Mole`` (or pyscfad ``Mole``) the basis comes from.
        coords: Grid coordinates, shape ``(ngrid, 3)`` -- typically
            ``mf.grids.coords``.
        xc_code: A PySCF/libxc XC string, e.g. ``"lda,vwn"``, ``"pbe,pbe"``,
            ``"scan,scan"``, ``"b3lyp"``, ``"pbe0"``.

    Returns:
        dict with keys ``xc_code``, ``xctype``, ``hyb``, ``ao0`` and (for
        GGA/mGGA) ``ao1`` / (for mGGA) ``ao2_diag``.
    """
    xctype = libxc.xc_type(xc_code)
    # libxc hybrid_coeff lives on a NumInt instance.
    from pyscf.dft.numint import NumInt

    hyb = float(NumInt().hybrid_coeff(xc_code, spin=0))

    if xctype == "LDA":
        ao_deriv = 0
    elif xctype in ("GGA", "NLC"):
        ao_deriv = 1
    elif xctype == "MGGA":
        ao_deriv = 2
    else:
        raise NotImplementedError(
            f"xc_type {xctype!r} (from {xc_code!r}) is not supported. "
            f"Supported: LDA, GGA, MGGA, and global hybrids of those."
        )

    feval = (
        "GTOval_cart_deriv%d" % ao_deriv if mol.cart else "GTOval_sph_deriv%d" % ao_deriv
    )
    ao = np.asarray(mol.eval_gto(feval, np.asarray(coords)))

    ingredients = {"xc_code": xc_code, "xctype": xctype, "hyb": hyb}
    if ao_deriv == 0:
        ingredients["ao0"] = ao  # (ngrid, nao)
    else:
        ingredients["ao0"] = ao[0]  # (ngrid, nao)
        ingredients["ao1"] = ao[1:4]  # (3, ngrid, nao)
    if ao_deriv == 2:
        # Diagonal second derivatives (xx, yy, zz) needed for the Laplacian if a
        # functional uses it; tau itself only needs ao1. PySCF's deriv=2 layout
        # is [val, x, y, z, xx, xy, xz, yy, yz, zz]; xx,yy,zz are indices 4,7,9.
        ingredients["ao2_diag"] = ao[[4, 7, 9]]  # (3, ngrid, nao)
    return ingredients


def _build_rho(dm: np.ndarray, ing: dict) -> np.ndarray:
    """Density (and, by xctype, its gradient / tau) from ``dm`` and AO arrays.

    Layout matches ``pyscf.dft.numint.eval_rho``:
    LDA -> ``(ngrid,)``; GGA -> ``(4, ngrid)``; MGGA -> ``(6, ngrid)``
    (``[rho, gx, gy, gz, lapl, tau]``).
    """
    ao0 = ing["ao0"]
    rho0 = np.einsum("gi,ij,gj->g", ao0, dm, ao0)
    if ing["xctype"] == "LDA":
        return rho0

    ao1 = ing["ao1"]
    # grad rho = 2 * sum_x ao_x . dm . ao0  (dm hermitian)
    grad = 2.0 * np.einsum("xgi,ij,gj->xg", ao1, dm, ao0)  # (3, ngrid)
    if ing["xctype"] in ("GGA", "NLC"):
        return np.vstack([rho0[None, :], grad])  # (4, ngrid)

    # MGGA: also tau = 0.5 * sum_x (ao_x . dm . ao_x)
    tau = 0.5 * np.einsum("xgi,ij,xgj->g", ao1, dm, ao1)  # (ngrid,)
    # Laplacian row: 2*(ao0 . dm . lap_ao) + 2*sum_x ao_x.dm.ao_x ; libxc rarely
    # uses it (vlapl is usually 0). Provide it for completeness.
    ao2 = ing.get("ao2_diag")
    lap_ao = ao2.sum(axis=0)  # (ngrid, nao) = (d2x+d2y+d2z) ao
    lapl = 2.0 * np.einsum("gi,ij,gj->g", ao0, dm, lap_ao)
    lapl += 2.0 * np.einsum("xgi,ij,xgj->g", ao1, dm, ao1)
    return np.vstack([rho0[None, :], grad, lapl[None, :], tau[None, :]])  # (6, ngrid)


def _assemble_vxc(
    rho: np.ndarray, vxc: tuple, weight: np.ndarray, ing: dict
) -> np.ndarray:
    """Build the XC potential matrix ``Vxc`` from libxc derivatives.

    Reproduces ``numint`` / ``_rks_gga_wv0``: the density term ``w*vrho*ao0ao0``
    plus, for GGA/mGGA, the gradient term ``w*2*vsigma*grad . (ao1 ao0 + ao0
    ao1)`` and, for mGGA, the kinetic term ``0.5*w*vtau * ao1.ao1``. Verified to
    ~1e-15 vs ``numint.nr_rks``.
    """
    ao0 = ing["ao0"]
    vrho = vxc[0]
    xctype = ing["xctype"]

    if xctype == "LDA":
        wv0 = weight * vrho
        # *0.5 + symmetrize is equivalent to the full einsum below for LDA.
        return np.einsum("g,gi,gj->ij", wv0, ao0, ao0)

    ao1 = ing["ao1"]
    vsigma = vxc[1]
    grad = rho[1:4]  # (3, ngrid)

    wv0 = weight * vrho
    wvg = weight * 2.0 * vsigma * grad  # (3, ngrid)

    Vxc = np.einsum("g,gi,gj->ij", wv0, ao0, ao0)
    grad_term = np.einsum("xg,xgi,gj->ij", wvg, ao1, ao0)
    Vxc = Vxc + grad_term + grad_term.T

    if xctype == "MGGA":
        vtau = vxc[3]
        # tau = 0.5 sum |grad phi|^2 -> dVxc/dtau term is 0.5 * vtau on ao1.ao1.
        wvt = 0.5 * weight * vtau
        for x in range(3):
            Vxc = Vxc + np.einsum("g,gi,gj->ij", wvt, ao1[x], ao1[x])
    return Vxc


def get_veff_libxc(
    dm,
    eri,
    ao_grid,  # unused: we use ing["ao0"]; kept for get_veff signature parity
    grid_weights,
    params: dict,
    xc_eval_fn=None,  # unused for libxc; kept for signature parity
):
    """Effective potential for a standard libxc functional (``encoding='libxc'``).

    ``params`` must be the dict from :func:`make_libxc_ingredients`. Returns the
    same ``(vhf = J + Vxc [- 0.5*hyb*K], exc_energy, J)`` triple as the network
    ``get_veff_*`` paths, so the SCF loop body is unchanged.
    """
    dm = np.asarray(dm)
    weight = np.asarray(grid_weights)
    ing = params

    J = np.einsum("ijkl,kl->ij", np.asarray(eri), dm)

    rho = _build_rho(dm, ing)
    exc, vxc, _fxc, _kxc = libxc.eval_xc(
        ing["xc_code"], rho, spin=0, deriv=1
    )
    rho0 = rho if ing["xctype"] == "LDA" else rho[0]
    exc_energy = float(np.einsum("g,g,g->", exc, rho0, weight))

    Vxc = _assemble_vxc(rho, vxc, weight, ing)
    vhf = J + Vxc

    hyb = ing["hyb"]
    if hyb != 0.0:
        # Closed-shell hybrid: subtract hyb * 0.5 * K from Fock, hyb*0.25*<dm,K>
        # from the energy. K_ij = sum (ik|jl) dm_kl.
        K = np.einsum("ikjl,kl->ij", np.asarray(eri), dm)
        vhf = vhf - 0.5 * hyb * K
        exc_energy = exc_energy - 0.25 * hyb * float(np.einsum("ij,ij->", dm, K))

    return vhf, exc_energy, J


def make_eval_xc_libxc_ref(xc_code: str = "lda,vwn") -> Callable:
    """Placeholder ``xc_eval_fn`` for ``encoding='libxc'``.

    The libxc path assembles Vxc directly in :func:`get_veff_libxc` and does not
    call an ``xc_eval_fn``; this exists only so callers can pass *something*
    where the network paths expect a callable. It is never invoked.
    """

    def _never_called(*_a, **_k):  # pragma: no cover
        raise RuntimeError(
            "encoding='libxc' does not use xc_eval_fn; veff is built in "
            "get_veff_libxc from the ingredients in `params`."
        )

    return _never_called
