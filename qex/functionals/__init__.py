"""Exchange-correlation (XC) functional models and their evaluation wrappers.

Networks come in two encodings:

- ``local``: the network outputs ``exc(r)`` per electron at each grid point
  (LDA-style); the energy is assembled outside as ``Sum exc * rho * w``.
- ``global``: the network outputs the scalar ``E_xc[rho]`` directly.

Use :func:`make_eval_xc_local` / :func:`make_eval_xc_global` to turn a network
into the ``xc_eval_fn`` consumed by the SCF loops in :mod:`qex.scf`.
"""

from qex.functionals.descriptor import DescriptorXC
from qex.functionals.features import KNOWN_FEATURES, FeatureBag, select
from qex.functionals.mlp import GlobalMLP, LocalMLP
from qex.functionals.qcnn import QCNN
from qex.functionals.xc import make_eval_xc_global, make_eval_xc_local

__all__ = [
    "LocalMLP",
    "GlobalMLP",
    "DescriptorXC",
    "QCNN",
    "make_eval_xc_local",
    "make_eval_xc_global",
    "FeatureBag",
    "KNOWN_FEATURES",
    "select",
]
