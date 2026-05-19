from .eval_gto import _eval_gto, eval_gto


def patch_pyscfad():
    """Monkey patch PySCFAD with our custom implementations"""
    import pyscfad.gto.eval_gto

    pyscfad.gto.eval_gto.eval_gto = eval_gto
    pyscfad.gto.eval_gto._eval_gto = _eval_gto
