"""Quantum convolutional network XC functional.

Elegant reformulation of the legacy `convolutional_models.py` construction:

- Split the density vector ρ into blocks of `n_features` values.
- Apply a small QNN (feature-map + HEA ansatz, measured by total magnetization)
  pointwise to every block (vmap over blocks). Each block contributes one scalar,
  so the output vector shrinks by a factor of `n_features` per layer.
- Repeat for `n_layers` QCNN layers (with `n_features` allowed to vary by layer).
- A final small MLP collapses the surviving vector to the scalar E_xc[ρ].

The whole thing is a single Flax `nn.Module` with the same interface as
`GlobalMLP`, so it slots into `make_eval_xc_global` and the existing training
pipeline without any glue code.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array
from horqrux import QuantumCircuit, expectation
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import DiffMode, TargetQubits

from qex.qnn_backend.horqrux.feature_maps import chebyshev_gates, direct_gates
from qex.qnn_backend.horqrux.hardware_ansatz import hea
from qex.qnn_backend.horqrux.measurement import total_magnetization_ops


_FEATURE_MAPS = {"direct": direct_gates, "chebyshev": chebyshev_gates}


def _build_block_circuit(
    n_qubits: int,
    n_var_layers: int,
    noise: NoiseProtocol | None,
    param_prefix: str = "v_",
):
    """Build the static parts of a per-block QNN.

    Returns (ansatz, param_names, observable, target_idx). The ansatz is a list
    of horqrux gates whose parameter names are deterministic — they form the
    flat vector of variational parameters the Flax module owns.

    `hea` names parameters with UUIDs by default, which breaks JIT cache reuse
    across re-traces (every call yields fresh keys). We override the prefix
    here, but UUID suffixes are still appended inside `hea`. Callers should
    rely on positional alignment with `param_names`, not on stable string keys
    across instantiations.
    """
    ansatz = hea(n_qubits, n_var_layers, noise=noise, variational_param_prefix=param_prefix)
    param_names = [op.param for op in ansatz if hasattr(op, "param")]
    observable = total_magnetization_ops(n_qubits)
    target_idx = TargetQubits(tuple((i,) for i in range(n_qubits)))
    return ansatz, param_names, observable, target_idx


def _block_apply(
    x_block: Array,
    param_dict: dict,
    *,
    n_qubits: int,
    feature_map_fn: Callable,
    ansatz: list,
    observable: list,
    target_idx: TargetQubits,
    state: Array,
    diff_mode: DiffMode,
    n_shots: int,
    key,
) -> Array:
    """Apply the QNN to a single `n_features`-wide block; return a scalar.

    `param_dict` (ansatz variational params) is built once outside the vmap so
    its keys are static — only the values are traced. Feature-map gates carry
    the per-block angle as a traced value directly on `op.param`.
    """
    fm = feature_map_fn(x_block, target_idx)
    circuit = QuantumCircuit(n_qubits=n_qubits, operations=fm + ansatz)
    return jnp.sum(
        expectation(
            state,
            circuit,
            observable,
            param_dict,
            n_shots=n_shots,
            diff_mode=diff_mode,
            key=key,
        ),
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
        if self.feature_map not in _FEATURE_MAPS:
            raise ValueError(
                f"Unknown feature_map {self.feature_map!r}; choose from {list(_FEATURE_MAPS)}.",
            )
        feature_map_fn = _FEATURE_MAPS[self.feature_map]

        ansatz, param_names, observable, target_idx = _build_block_circuit(
            self.n_qubits, self.n_var_layers, self.noise,
        )
        state = jnp.zeros((2,) * self.n_qubits, dtype=jnp.complex128).at[(0,) * self.n_qubits].set(1.0)

        var_params = self.param(
            "vparams",
            lambda rng: jax.random.uniform(rng, (len(param_names),), minval=-0.1, maxval=0.1),
        )

        # Pad so length is divisible by n_features, then chunk.
        pad = (-x.shape[0]) % self.n_features
        x_padded = jnp.pad(x, (0, pad))
        blocks = x_padded.reshape(-1, self.n_features)

        # Build the ansatz param dict once outside the vmap. Keys are static
        # (they come from `hea`'s deterministic naming); only the values are
        # traced JAX arrays.
        param_dict = dict(zip(param_names, var_params))
        key = jax.random.PRNGKey(0)

        def block_fn(block: Array) -> Array:
            return _block_apply(
                block,
                param_dict,
                n_qubits=self.n_qubits,
                feature_map_fn=feature_map_fn,
                ansatz=ansatz,
                observable=observable,
                target_idx=target_idx,
                state=state,
                diff_mode=self.diff_mode,
                n_shots=self.n_shots,
                key=key,
            )

        out = jax.vmap(block_fn)(blocks)

        # Emulates sampling/readout noise on the layer output. Mirrors the
        # `add_gaussian_noise_to_qnn_output` path in the legacy module. The RNG
        # is pulled from Flax's "noise" stream so the function stays pure under
        # jit; if no stream is supplied, this path is skipped.
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

    # ρ vector in, scalar out: no geometric/extra features (see GlobalMLP).
    required_features: ClassVar[tuple[str, ...]] = ()

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
    def __call__(self, rho: Array, grid_weights: Array | None = None) -> Array:
        # `grid_weights` is part of the global-encoding signature; QCNN ignores it.
        del grid_weights
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

    # -----------------------------------------------------------------
    # Global QCNN from the paper, QNN layers before final MLP projection
    # -----------------------------------------------------------------

    model = QCNN(
        n_qubits=4,
        n_features=4,
        n_layers=2,  # 32 -> 8 -> 2, then MLP head -> scalar
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

    # Gradient w.r.t. rho (vrho path used by make_eval_xc_global).
    grad_rho = jax.grad(lambda r: model.apply(params, r))(rho)
    print("grad rho shape:", grad_rho.shape)
    assert grad_rho.shape == rho.shape

    # JIT smoke test.
    jit_apply = jax.jit(model.apply)
    exc_jit = jit_apply(params, rho)
    print("JIT E_xc:", exc_jit)
    assert jnp.allclose(exc, exc_jit)

    # -----------------------------------------------------------------
    # Noisy variant: digital gate noise + Gaussian readout noise.
    # Mirrors the build_conv_qnn(..., noise=..., add_gaussian_noise_to_qnn_output=True)
    # entry in the legacy convolutional_models.py.
    # -----------------------------------------------------------------
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    gate_noise = (
        DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),
        DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, 0.01),
    )
    noisy_model = QCNN(
        n_qubits=4,
        n_features=4,
        n_layers=2,
        n_var_layers=2,
        feature_map="chebyshev",
        head_features=(16,),
        noise=gate_noise,
        gaussian_noise_std=0.05,
    )

    init_key, noise_key1, noise_key2 = jax.random.split(jax.random.PRNGKey(1), 3)
    noisy_params = noisy_model.init({"params": init_key, "noise": noise_key1}, rho)
    exc_noisy_a = noisy_model.apply(noisy_params, rho, rngs={"noise": noise_key1})
    exc_noisy_b = noisy_model.apply(noisy_params, rho, rngs={"noise": noise_key2})
    print("Noisy E_xc (sample A):", exc_noisy_a)
    print("Noisy E_xc (sample B):", exc_noisy_b)
    assert exc_noisy_a.shape == ()
    # Different RNGs must give different outputs when Gaussian noise is on.
    assert not jnp.allclose(exc_noisy_a, exc_noisy_b)

    print("OK")
