"""Quantum convolutional network XC functional.

Reformulation of the legacy `convolutional_models.py` construction:

- Split the density vector ρ into blocks of `n_features` values.
- Apply a small QNN (feature-map + HEA ansatz, measured by total magnetization)
  pointwise to every block (vmap over blocks). Each block contributes one scalar,
  so the output vector shrinks by a factor of `n_features` per layer.
- Repeat for `n_layers` QCNN layers (with `n_features` allowed to vary by layer).
- A final small MLP collapses the surviving vector to the scalar E_xc[ρ].

The whole thing is a single Flax `nn.Module` with the same interface as
`GlobalMLP`, so it slots into `make_eval_xc_global` and the existing training
pipeline without any glue code.

Per-block QNN primitives live in `qnn_utils.py` and wrap the slim vendored
horqrux (no string-keyed param dicts, no per-block `QuantumCircuit` build) so
that the XLA graph stays small when this module is vmapped under the SCF loop.
"""

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array

# Kept for backwards compatibility on the public API. The vendored horqrux
# does not implement noise channels, so a non-None `noise` field is ignored.
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import DiffMode

from qex.functionals.qnn_utils import (
    FEATURE_MAPS,
    block_qnn_apply,
    make_zero_state,
    n_ansatz_params,
)


class QCNNLayer(nn.Module):
    """One QCNN layer: ρ (length L) -> ρ' (length L // n_features).

    Reshapes the input into `(L // n_features, n_features)` blocks and applies
    a shared per-block QNN (vmap over blocks).
    """

    n_features: int
    n_qubits: int
    n_var_layers: int = 2
    feature_map: str = "chebyshev"
    noise: NoiseProtocol | None = None
    diff_mode: DiffMode = DiffMode.AD
    n_shots: int = 0
    gaussian_noise_std: float = 0.0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        if self.n_qubits < self.n_features:
            raise ValueError(
                f"n_qubits ({self.n_qubits}) must be >= n_features ({self.n_features}).",
            )
        if self.feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"Unknown feature_map {self.feature_map!r}; choose from {list(FEATURE_MAPS)}.",
            )

        n_params = n_ansatz_params(self.n_qubits, self.n_var_layers)
        var_params = self.param(
            "vparams",
            lambda rng: jax.random.uniform(rng, (n_params,), minval=-0.1, maxval=0.1),
        )
        state = make_zero_state(self.n_qubits)

        # Pad so length is divisible by n_features, then chunk.
        pad = (-x.shape[0]) % self.n_features
        x_padded = jnp.pad(x, (0, pad))
        blocks = x_padded.reshape(-1, self.n_features)

        def block_fn(block: Array) -> Array:
            return block_qnn_apply(
                block,
                var_params,
                n_qubits=self.n_qubits,
                n_var_layers=self.n_var_layers,
                feature_map=self.feature_map,
                state=state,
            )

        out = jax.vmap(block_fn)(blocks)

        # Gaussian readout noise, pulled from Flax's "noise" stream so the
        # function stays pure under jit. Skipped when no stream is supplied.
        if self.gaussian_noise_std > 0.0 and self.has_rng("noise"):
            rng = self.make_rng("noise")
            out = out + jax.random.normal(rng, out.shape) * self.gaussian_noise_std
        return out


class QCNN(nn.Module):
    """Quantum convolutional XC functional: ρ vector -> scalar E_xc[ρ].

    Architecture
    ------------
    `n_layers` stacked `QCNNLayer`s (each shrinks the vector by `n_features`),
    followed by a small classical MLP head that produces the integrated XC
    energy. The non-positivity prior `-scale * softplus(·)` matches `GlobalMLP`.
    """

    n_qubits: int = 4
    n_features: int = 4
    n_layers: int = 2
    n_var_layers: int = 2
    feature_map: str = "chebyshev"
    head_features: Sequence[int] = (32,)
    act_fn: Callable = nn.gelu
    scale: float = 1.0
    noise: NoiseProtocol | None = None
    diff_mode: DiffMode = DiffMode.AD
    n_shots: int = 0
    gaussian_noise_std: float = 0.0

    @nn.compact
    def __call__(self, rho: Array) -> Array:
        h = rho
        for _ in range(self.n_layers):
            h = QCNNLayer(
                n_features=self.n_features,
                n_qubits=self.n_qubits,
                n_var_layers=self.n_var_layers,
                feature_map=self.feature_map,
                noise=self.noise,
                diff_mode=self.diff_mode,
                n_shots=self.n_shots,
                gaussian_noise_std=self.gaussian_noise_std,
            )(h)
        for feat in self.head_features:
            h = self.act_fn(nn.Dense(feat)(h))
        out = nn.Dense(1)(h).squeeze()
        return -self.scale * nn.softplus(out)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    n_grid = 32
    rho = jnp.linspace(0.01, 1.0, n_grid)

    model = QCNN(
        n_qubits=4,
        n_features=4,
        n_layers=2,
        n_var_layers=2,
        feature_map="chebyshev",
        head_features=(16,),
    )

    key = jax.random.PRNGKey(0)
    params = model.init(key, rho)
    print("param tree shapes:")
    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    exc = model.apply(params, rho)
    print("E_xc:", exc, "shape:", exc.shape)
    assert exc.shape == (), f"expected scalar, got {exc.shape}"
    assert exc <= 0, f"non-positivity prior violated: {exc}"

    grad_rho = jax.grad(lambda r: model.apply(params, r))(rho)
    print("grad rho shape:", grad_rho.shape, "||g||:", float(jnp.linalg.norm(grad_rho)))
    assert grad_rho.shape == rho.shape

    jit_apply = jax.jit(model.apply)
    exc_jit = jit_apply(params, rho)
    print("JIT E_xc:", exc_jit)
    assert jnp.allclose(exc, exc_jit)

    print("OK")
