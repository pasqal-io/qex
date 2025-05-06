"""
Modified version of pyscf.dft.numint to include amplitude encoding.
"""

import numpy
from pyscf.dft import numint
from pyscf.lib import load_library
from pyscfad import numpy as np
from pyscfad.dft import libxc
from pyscfad.dft.numint import _dot_ao_ao, _rks_gga_wv0, _scale_ao, _vv10nlc, eval_rho
from pyscfad.ops import stop_grad

libdft = load_library("libdft")


def nr_rks(
    ni,
    mol,
    grids,
    xc_code,
    dms,
    relativity=0,
    hermi=0,
    max_memory=2000,
    verbose=None,
    params=None,
):
    if "NN" not in xc_code:
        xctype = ni._xc_type(xc_code)
    else:
        xctype = xc_code
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    nelec = [0] * nset
    excsum = [0] * nset
    vmat = [0] * nset
    aow = None

    if xctype == "LDA":
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "LDA")
                exc, vxc = ni.eval_xc(
                    xc_code,
                    rho,
                    spin=0,
                    relativity=relativity,
                    deriv=1,
                    verbose=verbose,
                )[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                # *.5 because vmat + vmat.T
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho, out=aow)
                aow = _scale_ao(ao, 0.5 * weight * vrho, out=None)
                vmat[idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = vrho = None
    elif xctype == "GGA":
        ao_deriv = 1
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "GGA")
                exc, vxc = ni.eval_xc(
                    xc_code,
                    rho,
                    spin=0,
                    relativity=relativity,
                    deriv=1,
                    verbose=verbose,
                )[:2]
                den = rho[0] * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                # ref eval_mat function
                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=None)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = wv = None
    elif xctype == "NLC":
        nlc_pars = ni.nlc_coeff(xc_code[:-6])
        if nlc_pars == [0, 0]:
            raise NotImplementedError(
                f"VV10 cannot be used with {xc_code[:-6]}. "
                f"The supported functionals are {ni.libxc.VV10_XC}",
            )
        ao_deriv = 1
        vvrho = numpy.empty([nset, 4, 0])
        vvweight = numpy.empty([nset, 0])
        vvcoords = numpy.empty([nset, 0, 3])
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ao = stop_grad(ao)
            rhotmp = numpy.empty([0, 4, weight.size])
            weighttmp = numpy.empty([0, weight.size])
            coordstmp = numpy.empty([0, weight.size, 3])
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "GGA")
                rho = numpy.asarray(stop_grad(rho))
                rho = numpy.expand_dims(rho, axis=0)
                rhotmp = numpy.concatenate((rhotmp, rho), axis=0)
                weighttmp = numpy.concatenate(
                    (weighttmp, numpy.expand_dims(weight, axis=0)),
                    axis=0,
                )
                coordstmp = numpy.concatenate(
                    (coordstmp, numpy.expand_dims(coords, axis=0)),
                    axis=0,
                )
                rho = None
            vvrho = numpy.concatenate((vvrho, rhotmp), axis=2)
            vvweight = numpy.concatenate((vvweight, weighttmp), axis=1)
            vvcoords = numpy.concatenate((vvcoords, coordstmp), axis=1)
            rhotmp = weighttmp = coordstmp = None
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "GGA")
                exc, vxc = _vv10nlc(
                    rho,
                    coords,
                    vvrho[idm],
                    vvweight[idm],
                    vvcoords[idm],
                    nlc_pars,
                )
                den = rho[0] * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                # ref eval_mat function
                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wv, out=aow)
                aow = _scale_ao(ao, wv, out=None)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho = exc = vxc = wv = None
        vvrho = vvweight = vvcoords = None
    elif xctype == "MGGA":
        if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
            raise NotImplementedError("laplacian in meta-GGA method")
        ao_deriv = 2
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "MGGA")
                exc, vxc = ni.eval_xc(
                    xc_code,
                    rho,
                    spin=0,
                    relativity=relativity,
                    deriv=1,
                    verbose=verbose,
                )[:2]
                # pylint: disable=W0612
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)

                wv = _rks_gga_wv0(rho, vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao[:4], wv, out=aow)
                aow = _scale_ao(ao[:4], wv, out=None)
                vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                # pylint: disable=W0511
                # FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
                # Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = (0.5 * 0.5 * weight * vtau).reshape(-1, 1)
                vmat[idm] += _dot_ao_ao(mol, ao[1], wv * ao[1], mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(mol, ao[2], wv * ao[2], mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(mol, ao[3], wv * ao[3], mask, shls_slice, ao_loc)
                rho = exc = vxc = vrho = wv = None

    elif xctype == "NN":
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "LDA")
                exc, vxc = ni.eval_xc(
                    xc_code,
                    rho,
                    spin=0,
                    relativity=relativity,
                    deriv=1,
                    verbose=verbose,
                    params=params,
                )[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                # *.5 because vmat + vmat.T
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho, out=aow)
                # aow = _scale_ao(ao, 0.5 * weight * vrho, out=None)
                aow = _scale_ao(ao, 0.5 * weight * vrho)
                # vmat[idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(ao, aow)
                rho = exc = vxc = vrho = None

    elif xctype == "NN-AmplitudeEncoding":
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, "LDA")
                excsum_i, vxc = ni.eval_xc(
                    xc_code,
                    rho,
                    spin=0,
                    relativity=relativity,
                    deriv=1,
                    verbose=verbose,
                    params=params,
                )[:2]
                vrho = vxc[0]
                den = rho * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += excsum_i
                # *.5 because vmat + vmat.T
                #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho, out=aow)
                # aow = _scale_ao(ao, 0.5 * weight * vrho, out=None)
                aow = _scale_ao(ao, 0.5 * weight * vrho)
                # vmat[idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                vmat[idm] += _dot_ao_ao(ao, aow)
                rho = exc = vxc = vrho = None

        # ao_deriv = 1
        # for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        #     for idm in range(nset):
        #         # Evaluates the gradients using GGA keyword
        #         rho = make_rho(idm, ao, mask, 'GGA')

        #         # Get excsum directly from eval_xc instead of computing from exc
        #         excsum_i, vxc = ni.eval_xc(
        #             xc_code,
        #             rho,
        #             spin=0,
        #             relativity=relativity,
        #             deriv=1,
        #             verbose=verbose,
        #             params=params,
        #             # params_grid_coords=params_grid_coords,
        #         )[:2]

        #         # LDA part
        #         vrho = vxc[0]
        #         den = rho[0] * weight
        #         nelec[idm] += stop_grad(den).sum()

        #         # Key difference: Use excsum_i directly instead of np.dot(den, exc)
        #         excsum[idm] += excsum_i

        #         # Compute potential contribution
        #         aow = _scale_ao(ao[0], 0.5 * weight * vrho, out=None)
        #         vmat[idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
        #         rho = exc = vxc = vrho = None

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].conj().T
    # nelec = numpy.asarray(nelec)
    # excsum = np.asarray(excsum)
    # vmat = np.asarray(vmat)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat


class NumInt(numint.NumInt):
    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if getattr(dms, "mo_coeff", None) is not None:
            # pylint: disable=W0511
            # TODO: test whether dm.mo_coeff matching dm
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
            nao = mo_coeff[0].shape[0]
            ndms = len(mo_occ)

            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho2(
                    mol,
                    ao,
                    mo_coeff[idm],
                    mo_occ[idm],
                    non0tab,
                    xctype,
                )

        else:
            if getattr(dms, "ndim", None) == 2:
                dms = [dms]
            if not hermi:
                # For eval_rho when xctype==GGA, which requires hermitian DMs
                dms = [(dm + dm.conj().T) * 0.5 for dm in dms]
            nao = dms[0].shape[0]
            ndms = len(dms)

            def make_rho(idm, ao, non0tab, xctype):
                return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi=1)

        return make_rho, ndms, nao

    def eval_xc(
        self,
        xc_code,
        rho,
        spin=0,
        relativity=0,
        deriv=1,
        omega=None,
        verbose=None,
        params=None,
    ):
        if omega is None:
            omega = self.omega
        # No params because libxc has tabulated functionals
        # This function is overloaded by a custom eval_xc
        return libxc.eval_xc(
            xc_code,
            rho,
            spin,
            relativity,
            deriv,
            omega,
            verbose,
        )

    def eval_rho(self, mol, ao, dm, non0tab=None, xctype="LDA", hermi=0, verbose=None):
        return eval_rho(mol, ao, dm, non0tab, xctype, hermi, verbose)

    nr_rks = nr_rks


if __name__ == "__main__":
    # Import required packages, using numpy from pyscfad for automatic differentiation
    import numpy as np
    from pyscfad import dft as addft
    from pyscfad import gto as adgto

    # Create H2 molecule using automatic differentiation-enabled Mole class
    mol = adgto.Mole(atom="H 0 0 0; H 0 0 1", basis="sto-3g")

    # Initialize DFT calculation with automatic differentiation support
    mf = addft.RKS(mol)
    mf.grids.level = 1  # Use coarse integration grid for faster testing

    # Define a simple test exchange-correlation functional
    # This quadratic functional is used to verify basic functionality
    def eval_xc(xc_code, rho, *args, **kwargs):
        """Simple quadratic functional for testing.

        Returns:
            exc: Exchange-correlation energy density
            vxc: Tuple of (vrho, None, None, None) where vrho is the first derivative
            fxc: Second derivative (not implemented)
            kxc: Third derivative (not implemented)
        """
        exc = 0.01 * rho**2  # Simple quadratic energy expression
        vrho = 0.01 * 2 * rho  # First derivative of energy wrt density
        vxc = (vrho, None, None, None)
        fxc = None
        kxc = None
        return exc, vxc, fxc, kxc

    # Set the custom XC functional and run SCF
    mf = mf.define_xc_(description=eval_xc, xctype="LDA")
    print("Testing Custom XC:")
    mf.kernel()

    print("Testing NumInt:")
    # Get initial density matrix for testing
    dm = mf.get_init_guess()

    # Create numerical integrator instance
    ni = NumInt()

    # Create mock neural network parameters for testing
    # In practice, these would come from a trained neural network
    nn_params = {
        "weights": np.random.random((10, 10)),
        "biases": np.random.random(10),
        "encoding_params": np.random.random(5),
    }

    # Define a test functional that mimics neural network with amplitude encoding
    def eval_xc_amplitude(xc_code, rho, *args, **kwargs):
        """Neural network functional with amplitude encoding.

        This is a placeholder that uses the same quadratic form as the test functional.
        In practice, this would implement actual neural network evaluation.
        """
        exc = np.sum(0.01 * rho**2)
        vrho = 0.01 * 2 * rho
        vxc = (vrho, None, None, None)
        fxc = None
        kxc = None
        return exc, vxc, fxc, kxc

    # Perform numerical integration with the amplitude encoding functional
    ni.eval_xc = eval_xc_amplitude
    nelec, exc, vxc = ni.nr_rks(
        mol,
        mf.grids,
        "NN-AmplitudeEncoding",
        dm,
        params=nn_params,
    )

    # Print results for verification
    print("Number of electrons:", nelec)
    print("Exchange-correlation energy:", exc)
    print("Exchange-correlation potential:", vxc)
