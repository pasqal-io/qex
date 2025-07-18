"""
Collection of all the models that are used directly to make the XC functional via wrapper for
1D or 3D cases.

Quantum and classical neural networks for Kohn-Sham DFT.
Here we define the full model, boiling down to:
 - init_fn and apply_fn functions.
 - config_dict for the model.
 - build_network function that builds init_fn and apply_fn.

The idea is only those functions should be defined and then can be trained in 1D or 3D cases.

So models with LOCAL embedding should have input:
    - input: jnp.ndarray of shape (n_grids,)
    - output: jnp.ndarray of shape (n_grids,)
    because the output is the XC potential on the grid points,
    to be integrated over the grid points.

So models with GLOBAL embedding should have input:
    - input: jnp.ndarray of shape (n_grids,)
    - output: jnp.ndarray of shape (1,)
    because the output is the total integrated XC energy.

# NOTE: The QNNs also use the parameter use_amplitude_encoding to determine
# if they use amplitude encoding or angle encoding.
# use_amplitude_encoding is also used for DFT when using global encoding.
# will be made more clean in the future.
"""

from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from horqrux import zero_state
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import DiffMode
from loguru import logger

from qedft.models.quantum.hardware_ansatz import hea


class KohnShamNetwork(Protocol):
    """Protocol for Kohn-Sham neural networks.

    This protocol defines the interface that all Kohn-Sham neural network
    implementations must follow.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
        """

    def build_network(self) -> tuple[Callable, Callable]:
        """Build and return the network architecture.

        Args:
            Any additional arguments are passed to the network constructor.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.

            init_fn takes a PRNGKey
            (e.g., jax.random.PRNGKey(0)), grid shape (e.g., (num_grid_points,))
            and returns initial parameters.

            apply_fn takes parameters and inputs and returns network outputs.

        Raises:
            NotImplementedError: This is a protocol method that must be implemented.
        """
        raise NotImplementedError


# ================================================================
# Classical models
# ================================================================


class LocalMLP(KohnShamNetwork):
    """Local MLP processing each grid point independently."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": False,
        }
        if config_dict is not None:
            self.config.update(config_dict)
        if self.config.get("use_amplitude_encoding") is True:
            raise ValueError("Set use_amplitude_encoding to False.")

    def build_network(self, grids: jnp.ndarray) -> tuple[Callable, Callable]:
        from qedft.models.classical.classical_models import build_local_mlp

        return build_local_mlp(
            n_neurons=self.config.get("n_neurons", 64),
            n_layers=self.config.get("n_layers", 3),
            activation=self.config.get("activation", "tanh"),
            density_normalization_factor=self.config.get("density_normalization_factor", 2.0),
            grids=grids,
        )


class GlobalMLP(KohnShamNetwork):
    """Global MLP with amplitude encoding."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp_ksr",
            "wrap_self_interaction": True,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,  # This model only uses global encoding
        }
        if config_dict is not None:
            self.config.update(config_dict)
        if self.config.get("use_amplitude_encoding") is False:
            raise ValueError("Set use_amplitude_encoding to True.")

    def build_network(self, grids: jnp.ndarray) -> tuple[Callable, Callable]:
        from qedft.models.classical.classical_models import build_global_mlp

        return build_global_mlp(
            n_neurons=self.config.get("n_neurons", 64),
            n_layers=self.config.get("n_layers", 3),
            activation=self.config.get("activation", "tanh"),
            density_normalization_factor=self.config.get("density_normalization_factor", 2.0),
            grids=grids,
        )


class BatchedGlobalMLP(KohnShamNetwork):
    """Batched Global MLP with amplitude encoding."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp_ksr",
            "wrap_self_interaction": True,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,
            "n_neurons": 64,
            "n_layers": 3,
            "activation": "tanh",
            "n_outputs": 1,
            "density_normalization_factor": 2.0,
            "n_features": 1,
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(self, grids: jnp.ndarray) -> tuple[Callable, Callable]:
        from qedft.models.classical.classical_models import build_batched_global_mlp

        return build_batched_global_mlp(
            n_neurons=self.config.get("n_neurons", 64),
            n_layers=self.config.get("n_layers", 3),
            activation=self.config.get("activation", "tanh"),
            n_outputs=self.config.get("n_outputs", 1),
            density_normalization_factor=self.config.get("density_normalization_factor", 2.0),
            n_features=self.config.get("n_features", 1),
        )


# ================================================================
# Quantum models
# ================================================================


class LocalQNN(KohnShamNetwork):
    """Basic QNN with angle encoding."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": False,
            "qnn_type": "LocalQNN",  # "LocalQNN", "GlobalQNN"
            "layer_type": "DirectQNN",  # "DirectQNN", "ChebyshevQNN", "ProductQNN"
            "map_fn": None,
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        from qedft.models.quantum.quantum_models import build_qnn

        return build_qnn(
            n_qubits=self.config.get("n_qubits", 2),
            ansatz=self.config.get(
                "ansatz",
                hea(self.config.get("n_qubits", 2), self.config.get("n_layers", 2)),
            ),
            state=self.config.get("state", zero_state(self.config.get("n_qubits", 2))),
            qnn_type=self.config.get("qnn_type", "LocalQNN"),
            layer_type=self.config.get("layer_type", "DirectQNN"),
            map_fn=self.config.get("map_fn", None),
            grids=grids,
            n_features=self.config.get("n_features", 1),
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )


class GlobalQNNLayer(KohnShamNetwork):
    """Basic Global QNN layer with angle encoding.
    Needs the convolutional stacking to be able to work on large inputs.
    See GlobalQNN for the full model."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,
            "qnn_type": "GlobalQNN",  # "LocalQNN", "GlobalQNN"
            "layer_type": "DirectQNN",  # "DirectQNN", "ChebyshevQNN", "ProductQNN"
            "map_fn": None,
            "n_features": 1,
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        from qedft.models.quantum.quantum_models import build_qnn

        return build_qnn(
            n_qubits=self.config.get("n_qubits", 2),
            n_features=self.config.get("n_features", 1),
            ansatz=self.config.get(
                "ansatz",
                hea(self.config.get("n_qubits", 2), self.config.get("n_layers", 2)),
            ),
            state=self.config.get("state", zero_state(self.config.get("n_qubits", 2))),
            qnn_type=self.config.get("qnn_type", "GlobalQNN"),
            layer_type=self.config.get("layer_type", "DirectQNN"),
            map_fn=self.config.get("map_fn", None),
            grids=grids,
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )


class AmplitudeEncodingGlobalQNNLayer(KohnShamNetwork):
    """Basic Global QNN layer with amplitude encoding.
    Needs the convolutional stacking to be able to work on large inputs.
    See GlobalQNN for the full model."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp_ksr",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,
            "qnn_type": "GlobalAmplitudeEncodingQNN",  # "LocalQNN", "GlobalQNN"
            "layer_type": "DirectQNN",  # "DirectQNN", "ChebyshevQNN", "ProductQNN"
            "map_fn": None,
            "n_features": 1,
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        from qedft.models.quantum.quantum_models import build_qnn

        return build_qnn(
            n_qubits=self.config.get("n_qubits", 2),
            n_features=self.config.get("n_features", 1),
            ansatz=self.config.get(
                "ansatz",
                hea(self.config.get("n_qubits", 2), self.config.get("n_layers", 2)),
            ),
            state=self.config.get("state", zero_state(self.config.get("n_qubits", 2))),
            qnn_type=self.config.get("qnn_type", "GlobalAmplitudeEncodingQNN"),
            layer_type=self.config.get("layer_type", "DirectQNN"),
            map_fn=self.config.get("map_fn", None),
            grids=grids,
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )


class GlobalQNN(KohnShamNetwork):
    """Convolutional Quantum Neural Network.

    This model applies a series of quantum convolutional layers to process
    the input density, reducing the dimension at each step according to
    kernel widths determined by the input dimension.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "conv_dqc",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,  # This model uses global encoding
            "n_qubits": 4,
            "n_var_layers": 2,
            "largest_kernel_width": 4,
            "max_number_conv_layers": 100,
            "list_qubits_per_layer": [],
            "force_qubits_per_layer_is_kernel_width": False,
            "normalization": 1.0,
            "last_layer_type": "dense",  # "dense" or "mlp"
            "use_bias_mlp": False,
            "last_layer_features": [1],
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        """Build the convolutional quantum neural network.

        Args:
            grids: Grid points for the density functional calculations.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.
        """
        from qedft.models.quantum.convolutional_models import build_conv_qnn

        # Get the input dimension from the grid size
        input_dimension = grids.shape[0]

        # Construct the full convolutional model
        network = build_conv_qnn(
            n_qubits=self.config.get("n_qubits", 4),
            n_var_layers=self.config.get("n_var_layers", 2),
            n_out=1,  # Always output a single value per point
            input_dimension=input_dimension,
            largest_kernel_width=self.config.get("largest_kernel_width", 4),
            max_number_conv_layers=self.config.get("max_number_conv_layers", 100),
            list_qubits_per_layer=self.config.get("list_qubits_per_layer", []),
            force_qubits_per_layer_is_kernel_width=self.config.get(
                "force_qubits_per_layer_is_kernel_width",
                False,
            ),
            normalization=self.config.get("normalization", 1.0),
            last_layer_type=self.config.get("last_layer_type", "dense"),
            use_bias_mlp=self.config.get("use_bias_mlp", False),
            last_layer_features=self.config.get("last_layer_features", [1]),
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )

        # Build the full network with the convolutional layers
        return network


# ================================================================
# Quantum models for paper proofs
# ================================================================


class GlobalQNNReverse(KohnShamNetwork):
    """Convolutional Quantum Neural Network.

    This model applies a series of quantum convolutional layers to process
    the input density, reducing the dimension at each step according to
    kernel widths determined by the input dimension.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "conv_dqc",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,  # This model uses global encoding
            "n_qubits": 4,
            "n_var_layers": 2,
            "largest_kernel_width": 4,
            "max_number_conv_layers": 100,
            "list_qubits_per_layer": [],
            "force_qubits_per_layer_is_kernel_width": False,
            "normalization": 1.0,
            "last_layer_type": "dense",  # "dense" or "mlp"
            "use_bias_mlp": False,
            "last_layer_features": [1],
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        """Build the convolutional quantum neural network.

        Args:
            grids: Grid points for the density functional calculations.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.
        """
        from qedft.models.quantum.convolutional_models import build_conv_qnn_reverse

        # Get the input dimension from the grid size
        input_dimension = grids.shape[0]

        # Construct the full convolutional model
        network = build_conv_qnn_reverse(
            n_qubits=self.config.get("n_qubits", 4),
            n_var_layers=self.config.get("n_var_layers", 2),
            n_out=1,  # Always output a single value per point
            input_dimension=input_dimension,
            largest_kernel_width=self.config.get("largest_kernel_width", 4),
            max_number_conv_layers=self.config.get("max_number_conv_layers", 100),
            list_qubits_per_layer=self.config.get("list_qubits_per_layer", []),
            force_qubits_per_layer_is_kernel_width=self.config.get(
                "force_qubits_per_layer_is_kernel_width",
                False,
            ),
            normalization=self.config.get("normalization", 1.0),
            last_layer_type=self.config.get("last_layer_type", "dense"),
            use_bias_mlp=self.config.get("use_bias_mlp", False),
            last_layer_features=self.config.get("last_layer_features", [1]),
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )

        # Build the full network with the convolutional layers
        return network


class GlobalQNNClassicalToQuantum(KohnShamNetwork):
    """Convolutional Quantum Neural Network.

    This model applies a series of quantum convolutional layers to process
    the input density, reducing the dimension at each step according to
    kernel widths determined by the input dimension.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "conv_dqc",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": False,
            "use_amplitude_encoding": False,  # This model uses global encoding
            "n_qubits": 4,
            "n_var_layers": 2,
            "largest_kernel_width": 4,
            "max_number_conv_layers": 100,
            "list_qubits_per_layer": [],
            "force_qubits_per_layer_is_kernel_width": False,
            "normalization": 1.0,
            "last_layer_type": "dense",  # "dense" or "mlp"
            "use_bias_mlp": False,
            "last_layer_features": [1],
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        """Build the convolutional quantum neural network.

        Args:
            grids: Grid points for the density functional calculations.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.
        """
        from qedft.models.quantum.convolutional_models import build_conv_qnn_classical_to_quantum

        # Get the input dimension from the grid size
        input_dimension = grids.shape[0]

        # Construct the full convolutional model
        network = build_conv_qnn_classical_to_quantum(
            n_qubits=self.config.get("n_qubits", 4),
            n_var_layers=self.config.get("n_var_layers", 2),
            n_out=1,  # NOTE: not used in this model
            input_dimension=input_dimension,
            normalization=self.config.get("normalization", 1.0),
            use_bias_mlp=self.config.get("use_bias_mlp", False),
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )

        # Build the full network with the convolutional layers
        return network


# ================================================================
# Quantum Inspired models
# ================================================================


class GlobalQiCQNN(KohnShamNetwork):
    """Convolutional Quantum Neural Network where QNNs use amplitude encoding for data.

    This model applies a series of quantum convolutional layers to process
    the input density, reducing the dimension at each step according to
    kernel widths determined by the input dimension.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "conv_dqc",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,  # This model uses global encoding
            "n_qubits": 4,
            "n_var_layers": 2,
            "largest_kernel_width": 4,
            "max_number_conv_layers": 100,
            "list_qubits_per_layer": [],
            "force_qubits_per_layer_is_kernel_width": False,
            "normalization": 1.0,
            "last_layer_type": "dense",  # "dense" or "mlp"
            "use_bias_mlp": False,
            "last_layer_features": [1],
            "diff_mode": DiffMode.AD,
            "n_shots": 0,
            "key": jax.random.PRNGKey(0),
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        """Build the convolutional quantum neural network.

        Args:
            grids: Grid points for the density functional calculations.
            noise: Noise protocol for the QNN.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.
        """
        from qedft.models.quantum.convolutional_models import build_conv_amplitude_encoding_qnn

        input_dimension = grids.shape[0]

        network = build_conv_amplitude_encoding_qnn(
            n_qubits=self.config.get("n_qubits", 4),
            n_var_layers=self.config.get("n_var_layers", 2),
            n_out=1,  # Always output a single value per point
            input_dimension=input_dimension,
            largest_kernel_width=self.config.get("largest_kernel_width", 4),
            max_number_conv_layers=self.config.get("max_number_conv_layers", 100),
            list_qubits_per_layer=self.config.get("list_qubits_per_layer", []),
            force_qubits_per_layer_is_kernel_width=self.config.get(
                "force_qubits_per_layer_is_kernel_width",
                False,
            ),
            normalization=self.config.get("normalization", 1.0),
            last_layer_type=self.config.get("last_layer_type", "dense"),
            use_bias_mlp=self.config.get("use_bias_mlp", False),
            last_layer_features=self.config.get("last_layer_features", [1]),
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )

        return network


class GlobalQiQNN(KohnShamNetwork):
    """Basic Global QNN layer with amplitude encoding.
    Needs the convolutional stacking to be able to work on large inputs.
    See GlobalQNN for the full model."""

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "mlp_ksr",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,
            "qnn_type": "GlobalAmplitudeEncodingQNN",  # "LocalQNN", "GlobalQNN"
            "layer_type": "DirectQNN",  # "DirectQNN", "ChebyshevQNN", "ProductQNN"
            "n_qubits": 4,
            "n_layers": 2,
            "map_fn": None,
            "n_features": 9,
            "diff_mode": DiffMode.AD,
            "n_shots": 0,
            "key": jax.random.PRNGKey(0),
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(
        self,
        grids: jnp.ndarray,
        noise: NoiseProtocol | None = None,
    ) -> tuple[Callable, Callable]:
        from qedft.models.quantum.quantum_models import build_qnn

        # Check if the number of features is correct
        n_features = self.config.get("n_features", grids.shape[0])

        logger.info(
            f"The number of features is {n_features}, and the statevector is {2**self.config.get('n_qubits', 4)} dim.",
        )
        if n_features != grids.shape[0]:
            logger.info(f"The number of features is {n_features}, not {grids.shape[0]}.")
            n_features = grids.shape[0]
            logger.info(
                f"Forcing to {n_features} so all data is encoded in the amplitude, provided enough qubits are used, otherwise some data gets lost.",
            )

        # All features have to be encoded in the amplitude
        return build_qnn(
            n_qubits=self.config.get("n_qubits", 4),
            n_features=n_features,
            ansatz=self.config.get(
                "ansatz",
                hea(self.config.get("n_qubits", 4), self.config.get("n_layers", 2)),
            ),
            state=self.config.get("state", zero_state(self.config.get("n_qubits", 2))),
            qnn_type=self.config.get("qnn_type", "GlobalAmplitudeEncodingQNN"),
            layer_type=self.config.get("layer_type", "DirectQNN"),
            map_fn=self.config.get("map_fn", None),
            grids=grids,
            noise=noise,
            diff_mode=self.config.get("diff_mode", DiffMode.AD),
            n_shots=self.config.get("n_shots", 0),
            key=self.config.get("key", jax.random.PRNGKey(0)),
        )


# ================================================================
# Additional models for paper proofs
# ================================================================


class GlobalQNNwithMLP(KohnShamNetwork):
    """Global QNN structure where all QNNs were replaced by MLPs (referee request).
    For comparison with the GlobalQNN where we have QNNs instead of MLPs.

    This model applies a series of quantum convolutional layers to process
    the input density, reducing the dimension at each step according to
    kernel widths determined by the input dimension.
    """

    def __init__(self, config_dict: dict | None = None):
        """Initialize from config dictionary."""
        super().__init__(config_dict)
        self.config = {
            "network_type": "conv_dqc",
            "wrap_self_interaction": False,
            "wrap_with_negative_transform": True,
            "use_amplitude_encoding": True,  # This model uses global encoding
            "n_qubits": 4,
            "n_var_layers": 2,
            "largest_kernel_width": 4,
            "max_number_conv_layers": 100,
            "list_qubits_per_layer": [],
            "force_qubits_per_layer_is_kernel_width": False,
            "normalization": 1.0,
            "last_layer_type": "dense",  # "dense" or "mlp"
            "use_bias_mlp": False,
            "last_layer_features": [1],
        }
        if config_dict is not None:
            self.config.update(config_dict)

    def build_network(self, grids: jnp.ndarray) -> tuple[Callable, Callable]:
        """Build the convolutional MLP.

        Args:
            grids: Grid points for the density functional calculations.

        Returns:
            tuple: (init_fn, apply_fn) pair of network initialization and application functions.
        """
        from qedft.models.quantum.convolutional_models import build_conv_mlp

        # Get the input dimension from the grid size
        input_dimension = grids.shape[0]

        # Construct the convolutional model
        network = build_conv_mlp(
            n_qubits=self.config.get("n_qubits", 4),
            n_var_layers=self.config.get("n_var_layers", 2),
            n_out=1,  # Always output a single value per point
            input_dimension=input_dimension,
            largest_kernel_width=self.config.get("largest_kernel_width", 4),
            max_number_conv_layers=self.config.get("max_number_conv_layers", 100),
            list_qubits_per_layer=self.config.get("list_qubits_per_layer", []),
            force_qubits_per_layer_is_kernel_width=self.config.get(
                "force_qubits_per_layer_is_kernel_width",
                False,
            ),
            normalization=self.config.get("normalization", 1.0),
            last_layer_type=self.config.get("last_layer_type", "dense"),
            use_bias_mlp=self.config.get("use_bias_mlp", False),
            last_layer_features=self.config.get("last_layer_features", [1]),
        )

        # Build the full network with the convolutional layers
        return network


if __name__ == "__main__":
    import jax

    # Create different QNN models
    models = {
        "GlobalQNNClassicalToQuantum": GlobalQNNClassicalToQuantum(config_dict={"n_qubits": 6}),
        "GlobalQNNReverse": GlobalQNNReverse(
            config_dict={"n_qubits": 6},
        ),
        "DirectQNN": LocalQNN(config_dict={"qnn_type": "LocalQNN", "layer_type": "DirectQNN"}),
        "ChebyshevQNN": LocalQNN(
            config_dict={"qnn_type": "LocalQNN", "layer_type": "ChebyshevQNN"},
        ),
        "ProductQNN": LocalQNN(config_dict={"qnn_type": "LocalQNN", "layer_type": "ProductQNN"}),
        "GlobalQNN": GlobalQNN(config_dict={"qnn_type": "GlobalQNN", "layer_type": "DirectQNN"}),
        "GlobalQNNwithMLPInstead": GlobalQNNwithMLP(
            config_dict={"qnn_type": "GlobalQNN", "layer_type": "DirectQNN"},
        ),
        "GlobalQiCQNN": GlobalQiCQNN(),  # Convolutional stacking
        "GlobalQiQNN": GlobalQiQNN(),  # One QNN, density input as state -> XC energy
    }

    # Setup test data
    num_points = 14
    density = jnp.linspace(0, 1, num_points)
    print(f"Density shape: {density.shape}")
    grids = jnp.ones(density.shape)
    print(f"Grids shape: {grids.shape}")
    rng_key = jax.random.PRNGKey(0)

    # Test each model
    results = {}
    for name, model in models.items():
        print(f"Building {name}...")
        # Build network
        init_fn, apply_fn = model.build_network(grids)

        # Initialize parameters and run inference
        _, params = init_fn(rng_key, input_shape=(-1, num_points, 1))
        output = apply_fn(params, density)

        # Store results
        results[name] = output

        # Print results
        print(f"\n=== {name} ===")
        print(f"Output shape: {output.shape}")
        print(f"Output values: {output}")
