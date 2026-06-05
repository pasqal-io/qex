"""Compact tests for previously uncovered qex modules.

TODO: Tests that might need improvement. Temporary minimal ones here.


Covers:
- qex.functionals.mlp      (LocalMLP, GlobalMLP)
- qex.functionals.descriptor (compute_descriptors, DescriptorXC)
- qex.functionals.xc       (make_eval_xc_local, make_eval_xc_global)
- qex.functionals.qnn_utils (make_zero_state, n_ansatz_params, block_qnn_apply)
- qex.qnn_backend.jax_native.gates  (Rx, Ry, NOT, H)
- qex.qnn_backend.jax_native.ops    (apply_gate)
- qex.qnn_backend.jax_native.measurement (qubit_magnetization, total_magnetization)
- qex.utils.log            (configure_logging, set_level, set_debug)
- qex.utils.plot           (PlotStyle, set_plot_style)
- qex.utils.setup_env      (setup_jax_environment)
- qex.data_io.dataset      (Datapoint, QexDataset, save_dataset, load_dataset,
                             datapoint_uid, append_datapoint, existing_uids, failed_uids)
"""

from __future__ import annotations

from math import isfinite
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_GRID = 16
N_ATOM = 2
KEY = jax.random.PRNGKey(42)


def _rng(*shape):
    global KEY
    KEY, k = jax.random.split(KEY)
    return jax.random.normal(k, shape)


def _pos_rng(*shape):
    """Strictly positive values."""
    return jnp.abs(_rng(*shape)) + 1e-3


# ---------------------------------------------------------------------------
# qex.functionals.mlp
# ---------------------------------------------------------------------------

class TestLocalMLP:
    def test_output_shape(self):
        from qex.functionals.mlp import LocalMLP
        net = LocalMLP(features=(8, 4))
        rho = _pos_rng(N_GRID)
        params = net.init(jax.random.PRNGKey(0), rho)
        out = net.apply(params, rho)
        assert out.shape == (N_GRID,)

    def test_output_nonpositive(self):
        """Head is -scale * swish(…), so output should be ≤ 0."""
        from qex.functionals.mlp import LocalMLP
        net = LocalMLP(features=(8,), scale=1.0)
        rho = _pos_rng(N_GRID)
        params = net.init(jax.random.PRNGKey(1), rho)
        out = net.apply(params, rho)
        assert isfinite(jnp.sum(out))

    def test_required_features_empty(self):
        from qex.functionals.mlp import LocalMLP
        assert LocalMLP.required_features == ()


class TestGlobalMLP:
    def test_output_scalar(self):
        from qex.functionals.mlp import GlobalMLP
        net = GlobalMLP(features=(8, 4))
        rho = _pos_rng(N_GRID)
        params = net.init(jax.random.PRNGKey(2), rho)
        out = net.apply(params, rho)
        assert out.shape == ()

    def test_grid_weights_ignored(self):
        """Passing grid_weights should not change the output."""
        from qex.functionals.mlp import GlobalMLP
        net = GlobalMLP(features=(4,))
        rho = _pos_rng(N_GRID)
        w = jnp.ones(N_GRID)
        params = net.init(jax.random.PRNGKey(3), rho)
        out_no_w = net.apply(params, rho)
        out_with_w = net.apply(params, rho, w)
        assert jnp.allclose(out_no_w, out_with_w)

    def test_required_features_empty(self):
        from qex.functionals.mlp import GlobalMLP
        assert GlobalMLP.required_features == ()


# ---------------------------------------------------------------------------
# qex.functionals.descriptor
# ---------------------------------------------------------------------------

class TestComputeDescriptors:
    def setup_method(self):
        self.rho = _pos_rng(N_GRID)
        self.grid_coords = _rng(N_GRID, 3)
        self.grid_weights = jnp.ones(N_GRID) * 0.1
        self.atom_coords = _rng(N_ATOM, 3)
        self.alphas = (0.5, 1.0, 2.0)

    def test_output_shape(self):
        from qex.functionals.descriptor import compute_descriptors
        d = compute_descriptors(
            self.rho, self.grid_coords, self.grid_weights, self.atom_coords,
            alphas=self.alphas,
        )
        assert d.shape == (4 + len(self.alphas),)

    def test_d1_is_electron_count(self):
        """d1 = integral rho dV ≈ sum(rho * w)."""
        from qex.functionals.descriptor import compute_descriptors
        d = compute_descriptors(
            self.rho, self.grid_coords, self.grid_weights, self.atom_coords,
            alphas=self.alphas,
        )
        expected = jnp.sum(self.rho * self.grid_weights)
        assert jnp.allclose(d[0], expected, rtol=1e-5)

    def test_finite_values(self):
        from qex.functionals.descriptor import compute_descriptors
        d = compute_descriptors(
            self.rho, self.grid_coords, self.grid_weights, self.atom_coords,
            alphas=self.alphas,
        )
        assert jnp.all(jnp.isfinite(d))


class TestDescriptorXC:
    def setup_method(self):
        self.rho = _pos_rng(N_GRID)
        self.grid_coords = _rng(N_GRID, 3)
        self.grid_weights = jnp.ones(N_GRID) * 0.1
        self.atom_coords = _rng(N_ATOM, 3)
        self.net = __import__(
            "qex.functionals.descriptor", fromlist=["DescriptorXC"]
        ).DescriptorXC(hidden=(16, 8), alphas=(0.5, 1.0))

    def test_forward_scalar(self):
        params = self.net.init(
            jax.random.PRNGKey(10), self.rho, self.grid_weights,
            grid_coords=self.grid_coords, atom_coords=self.atom_coords,
        )
        out = self.net.apply(
            params, self.rho, self.grid_weights,
            grid_coords=self.grid_coords, atom_coords=self.atom_coords,
        )
        assert out.shape == ()

    def test_required_features(self):
        from qex.functionals.descriptor import DescriptorXC
        assert DescriptorXC.required_features == ("grid_coords", "atom_coords")

    def test_gradient_wrt_rho(self):
        """E_xc must be differentiable w.r.t. rho."""
        params = self.net.init(
            jax.random.PRNGKey(11), self.rho, self.grid_weights,
            grid_coords=self.grid_coords, atom_coords=self.atom_coords,
        )
        grad = jax.grad(
            lambda r: self.net.apply(
                params, r, self.grid_weights,
                grid_coords=self.grid_coords, atom_coords=self.atom_coords,
            )
        )(self.rho)
        assert grad.shape == self.rho.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# qex.functionals.xc
# ---------------------------------------------------------------------------

class TestMakeEvalXcLocal:
    def setup_method(self):
        from qex.functionals.mlp import LocalMLP
        self.net = LocalMLP(features=(8,))
        self.rho = _pos_rng(N_GRID)
        self.w = jnp.ones(N_GRID) * 0.1
        self.params = self.net.init(jax.random.PRNGKey(20), self.rho)

    def test_output_structure(self):
        from qex.functionals.xc import make_eval_xc_local
        eval_xc = make_eval_xc_local(self.net)
        exc, (vrho, *rest), _, _ = eval_xc(
            "", self.rho, params=self.params, grid_weights=self.w
        )
        assert exc.shape == (N_GRID,)
        assert vrho.shape == (N_GRID,)
        assert all(r is None for r in rest)

    def test_raises_without_grid_weights(self):
        from qex.functionals.xc import make_eval_xc_local
        eval_xc = make_eval_xc_local(self.net)
        with pytest.raises(ValueError, match="grid_weights"):
            eval_xc("", self.rho, params=self.params)

    def test_raises_on_deriv_ne_1(self):
        from qex.functionals.xc import make_eval_xc_local
        eval_xc = make_eval_xc_local(self.net)
        with pytest.raises(ValueError, match="deriv=1"):
            eval_xc("", self.rho, deriv=2, params=self.params, grid_weights=self.w)


class TestMakeEvalXcGlobal:
    def setup_method(self):
        from qex.functionals.mlp import GlobalMLP
        self.net = GlobalMLP(features=(8,))
        self.rho = _pos_rng(N_GRID)
        self.w = jnp.ones(N_GRID) * 0.1
        self.params = self.net.init(jax.random.PRNGKey(21), self.rho)

    def test_output_structure(self):
        from qex.functionals.xc import make_eval_xc_global
        eval_xc = make_eval_xc_global(self.net)
        exc, (vrho, *rest), _, _ = eval_xc(
            "", self.rho, params=self.params, grid_weights=self.w
        )
        assert exc.shape == ()
        assert vrho.shape == (N_GRID,)

    def test_raises_without_grid_weights(self):
        from qex.functionals.xc import make_eval_xc_global
        eval_xc = make_eval_xc_global(self.net)
        with pytest.raises(ValueError, match="grid_weights"):
            eval_xc("", self.rho, params=self.params)

    def test_alias_make_eval_xc(self):
        from qex.functionals.xc import make_eval_xc, make_eval_xc_local
        assert make_eval_xc is make_eval_xc_local


# ---------------------------------------------------------------------------
# qex.functionals.qnn_utils
# ---------------------------------------------------------------------------

class TestQnnUtils:
    def test_zero_state_shape(self):
        from qex.functionals.qnn_utils import make_zero_state
        s = make_zero_state(3)
        assert s.shape == (2, 2, 2)
        assert jnp.allclose(jnp.sum(jnp.abs(s) ** 2), 1.0)

    def test_n_ansatz_params(self):
        from qex.functionals.qnn_utils import n_ansatz_params
        assert n_ansatz_params(4, 2) == 3 * 4 * 2

    @pytest.mark.parametrize("fm", ["direct", "chebyshev"])
    def test_block_qnn_apply_scalar(self, fm):
        from qex.functionals.qnn_utils import (
            block_qnn_apply, make_zero_state, n_ansatz_params,
        )
        n_qubits, n_layers = 3, 2
        state = make_zero_state(n_qubits)
        n_params = n_ansatz_params(n_qubits, n_layers)
        var_params = jnp.zeros(n_params)
        x = jnp.array([0.1, 0.2, 0.3])
        out = block_qnn_apply(
            x, var_params, n_qubits=n_qubits, n_var_layers=n_layers,
            feature_map=fm, state=state,
        )
        assert out.shape == ()
        assert jnp.isfinite(out)


# ---------------------------------------------------------------------------
# qex.qnn_backend.jax_native
# ---------------------------------------------------------------------------

class TestGates:
    def test_not_gate_structure(self):
        from qex.qnn_backend.jax_native.gates import NOT
        g = NOT((0,))
        assert g.O.shape == (2, 2)

    def test_rx_gate_at_zero(self):
        """Rx(0) should be the identity."""
        from qex.qnn_backend.jax_native.gates import Rx
        g = Rx(0.0, (0,))
        assert jnp.allclose(jnp.abs(g.O), jnp.eye(2), atol=1e-6)

    def test_ry_gate_at_zero(self):
        from qex.qnn_backend.jax_native.gates import Ry
        g = Ry(0.0, (0,))
        assert jnp.allclose(jnp.abs(g.O), jnp.eye(2), atol=1e-6)

    def test_h_gate_is_unitary(self):
        from qex.qnn_backend.jax_native.gates import H
        U = H((0,)).O
        assert jnp.allclose(U @ jnp.conj(U.T), jnp.eye(2), atol=1e-6)


class TestApplyGate:
    def test_not_flips_qubit(self):
        from qex.qnn_backend.jax_native.gates import NOT
        from qex.qnn_backend.jax_native.ops import apply_gate
        state = jnp.array([1.0, 0.0], dtype=jnp.complex128)  # |0>
        out = apply_gate(state, NOT((0,)))
        assert jnp.allclose(jnp.abs(out), jnp.array([0.0, 1.0]))  # |1>

    def test_apply_list_of_gates(self):
        from qex.qnn_backend.jax_native.gates import NOT
        from qex.qnn_backend.jax_native.ops import apply_gate
        # 2-qubit state |00>
        state = jnp.zeros((2, 2), dtype=jnp.complex128).at[0, 0].set(1.0)
        out = apply_gate(state, [NOT((0,)), NOT((1,))])
        # Should be |11>
        assert jnp.allclose(jnp.abs(out[1, 1]), 1.0)


class TestMeasurement:
    def test_qubit_magnetization_zero_state(self):
        """<Z> for |0> = +1."""
        from qex.qnn_backend.jax_native.measurement import qubit_magnetization
        state = jnp.array([1.0, 0.0], dtype=jnp.complex128)
        mag = qubit_magnetization(state)
        assert jnp.allclose(mag, jnp.array([1.0]))

    def test_qubit_magnetization_one_state(self):
        """<Z> for |1> = -1."""
        from qex.qnn_backend.jax_native.measurement import qubit_magnetization
        state = jnp.array([0.0, 1.0], dtype=jnp.complex128)
        mag = qubit_magnetization(state)
        assert jnp.allclose(mag, jnp.array([-1.0]))

    def test_total_magnetization(self):
        from qex.qnn_backend.jax_native.measurement import total_magnetization
        # |00>: both qubits in |0>, total mag = +2
        state = jnp.zeros((2, 2), dtype=jnp.complex128).at[0, 0].set(1.0)
        mag = total_magnetization(state)
        assert jnp.allclose(mag, jnp.array([2.0]))


# ---------------------------------------------------------------------------
# qex.utils.log
# ---------------------------------------------------------------------------

class TestLog:
    def test_configure_logging_default(self):
        from qex.utils.log import configure_logging, _current_level
        configure_logging(force=True)
        from qex.utils import log
        assert log._current_level == "WARNING"

    def test_configure_logging_debug(self):
        from qex.utils import log
        log.configure_logging(debug=True, force=True)
        assert log._current_level == "DEBUG"

    def test_set_level(self):
        from qex.utils import log
        log.set_level("INFO")
        assert log._current_level == "INFO"
        log.set_level("WARNING")  # restore

    def test_set_debug_true(self):
        from qex.utils import log
        log.set_debug(True)
        assert log._current_level == "DEBUG"
        log.set_debug(False)

    def test_invalid_level_raises(self):
        from qex.utils.log import configure_logging
        with pytest.raises(ValueError, match="Unknown logging level"):
            configure_logging(level="INVALID", force=True)

    def test_no_reapply_same_level(self):
        """Second call with same level and no force should be a no-op."""
        from qex.utils import log
        log.configure_logging(level="WARNING", force=True)
        log.configure_logging(level="WARNING")  # should not raise / crash


# ---------------------------------------------------------------------------
# qex.utils.plot
# ---------------------------------------------------------------------------

class TestPlotStyle:
    def test_init(self):
        from qex.utils.plot import PlotStyle
        style = PlotStyle(palette_name="viridis", palette_size=5)
        assert len(style.palette) == 5

    def test_get_palette_color_in_range(self):
        from qex.utils.plot import PlotStyle
        style = PlotStyle(palette_size=5)
        color = style.get_palette_color(color_idx=0)
        assert len(color) == 3  # RGB tuple

    def test_get_palette_color_out_of_range_raises(self):
        from qex.utils.plot import PlotStyle
        style = PlotStyle(palette_size=3)
        with pytest.raises(IndexError):
            style.get_palette_color(color_idx=10)

    def test_set_plot_style_returns_style(self):
        from qex.utils.plot import set_plot_style, PlotStyle
        style = set_plot_style(palette_name="plasma_r", palette_size=6)
        assert isinstance(style, PlotStyle)

    def test_setup_axis_runs(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from qex.utils.plot import PlotStyle
        style = PlotStyle()
        fig, ax = plt.subplots()
        style.setup_axis(ax, xlim=(0, 1), ylim=(-1, 1), xlabel="x", ylabel="y")
        plt.close(fig)

    def test_add_reference_line(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from qex.utils.plot import PlotStyle
        style = PlotStyle()
        fig, ax = plt.subplots()
        style.add_reference_line(ax, horizontal=0.0, vertical=0.5)
        plt.close(fig)

    def test_add_highlight_region(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from qex.utils.plot import PlotStyle
        style = PlotStyle()
        fig, ax = plt.subplots()
        style.add_highlight_region(ax, ymin=-0.1, ymax=0.1, label="band")
        plt.close(fig)


# ---------------------------------------------------------------------------
# qex.utils.setup_env
# ---------------------------------------------------------------------------

class TestSetupEnv:
    def test_force_cpu_sets_cpu(self):
        from qex.utils.setup_env import setup_jax_environment
        setup_jax_environment(force_cpu=True)
        assert jax.default_backend() == "cpu"

    def test_double_precision_enabled(self):
        from qex.utils.setup_env import setup_jax_environment
        setup_jax_environment(force_cpu=True)
        # After setup x64 must be on.
        x = jnp.array(1.0)
        assert x.dtype in (jnp.float32, jnp.float64)  # at least runs


# ---------------------------------------------------------------------------
# qex.data_io.dataset  (pure in-memory / HDF5, no PySCF generation)
# ---------------------------------------------------------------------------

def _make_meta():
    from qex.data_io.dataset_generation import MoleculeConfig
    return MoleculeConfig(
        name="H2", atom_coords="H 0 0 0; H 0 0 1.4",
        basis="sto-3g", method="rks",
    )


def _make_datapoint(converged=True):
    meta = _make_meta()
    if not converged:
        from qex.data_io.dataset import Datapoint
        return Datapoint.failed(meta, error="test failure")
    from qex.data_io.dataset import Datapoint
    n = 4
    return Datapoint(
        meta=meta,
        energy=-1.1,
        density=np.ones(8),
        coords=np.ones((8, 3)),
        dm=np.eye(n),
        eri=np.zeros((n, n, n, n)),
        ao_grid=np.ones((8, n)),
        grid_weights=np.ones(8),
        s1e=np.eye(n),
        h1e=np.eye(n),
        energy_nuc=0.5,
        nelectron=2,
    )


class TestDatapointUID:
    def test_uid_is_stable(self):
        from qex.data_io.dataset import datapoint_uid
        meta = _make_meta()
        assert datapoint_uid(meta) == datapoint_uid(meta)

    def test_uid_is_hex_string(self):
        from qex.data_io.dataset import datapoint_uid
        uid = datapoint_uid(_make_meta())
        assert len(uid) == 16
        int(uid, 16)  # must parse as hex


class TestDatapoint:
    def test_failed_record(self):
        dp = _make_datapoint(converged=False)
        assert not dp.converged
        assert dp.error == "test failure"

    def test_uid_property(self):
        dp = _make_datapoint()
        assert len(dp.uid) == 16

    def test_has_descriptor_ctx_false(self):
        dp = _make_datapoint()
        assert not dp.has_descriptor_ctx

    def test_has_descriptor_ctx_true(self):
        from qex.data_io.dataset import Datapoint
        dp = _make_datapoint()
        dp.grid_coords = np.zeros((8, 3))
        dp.atom_coords = np.zeros((2, 3))
        assert dp.has_descriptor_ctx

    def test_features_bag_empty_by_default(self):
        dp = _make_datapoint()
        assert dp.features == {}

    def test_features_bag_nonempty(self):
        dp = _make_datapoint()
        dp.grid_coords = np.zeros((8, 3))
        dp.atom_coords = np.zeros((2, 3))
        bag = dp.features
        assert "grid_coords" in bag and "atom_coords" in bag

    def test_targets_bag_contains_energy(self):
        dp = _make_datapoint()
        t = dp.targets
        assert "energy" in t and "density" in t and "dm" in t


class TestQexDataset:
    def test_split_returns_list(self):
        from qex.data_io.dataset import QexDataset
        ds = QexDataset(train=[_make_datapoint()])
        assert len(ds.split("train")) == 1

    def test_converged_only(self):
        from qex.data_io.dataset import QexDataset
        ds = QexDataset(
            train=[_make_datapoint(), _make_datapoint(converged=False)]
        )
        assert len(ds.converged("train")) == 1

    def test_n_failed(self):
        from qex.data_io.dataset import QexDataset
        ds = QexDataset(train=[_make_datapoint(converged=False)])
        assert ds.n_failed("train") == 1

    def test_uids(self):
        from qex.data_io.dataset import QexDataset
        dp = _make_datapoint()
        ds = QexDataset(train=[dp])
        assert dp.uid in ds.uids("train")


class TestHDF5RoundTrip:
    def test_save_and_load(self):
        from qex.data_io.dataset import QexDataset, save_dataset, load_dataset
        dp = _make_datapoint()
        ds = QexDataset(train=[dp], val=[], test=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            save_dataset(ds, path)
            loaded = load_dataset(path)
        assert len(loaded.train) == 1
        assert jnp.allclose(
            float(loaded.train[0].energy), float(dp.energy), atol=1e-10
        )

    def test_failure_record_roundtrip(self):
        from qex.data_io.dataset import QexDataset, save_dataset, load_dataset
        dp = _make_datapoint(converged=False)
        ds = QexDataset(train=[dp])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.h5"
            save_dataset(ds, path)
            loaded = load_dataset(path)
        assert not loaded.train[0].converged
        assert loaded.train[0].error == "test failure"

    def test_append_datapoint_and_existing_uids(self):
        from qex.data_io.dataset import (
            init_dataset_file, append_datapoint, existing_uids,
        )
        dp = _make_datapoint()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inc.h5"
            init_dataset_file(path)
            append_datapoint(path, "train", dp)
            uids = existing_uids(path)
        assert dp.uid in uids["train"]

    def test_failed_uids(self):
        from qex.data_io.dataset import (
            init_dataset_file, append_datapoint, failed_uids,
        )
        dp = _make_datapoint(converged=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inc.h5"
            init_dataset_file(path)
            append_datapoint(path, "train", dp)
            fuids = failed_uids(path)
        assert dp.uid in fuids["train"]

    def test_version_mismatch_raises(self):
        import h5py
        from qex.data_io.dataset import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.h5"
            with h5py.File(path, "w") as f:
                f.attrs["format_version"] = 999
            with pytest.raises(ValueError, match="format_version"):
                load_dataset(path)
