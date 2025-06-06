"""
Quantum models.

This module provides quantum neural network models for quantum machine learning tasks.
The main class is LocalQNN which implements a quantum neural network with a feature
map encoding, variational ansatz, and measurement.

The noise is applied to the gates in the ansatz and feature map outside the QNN class.

Example:
    # Initialize a 2-qubit QNN with 2 variational layers
    n_qubits = 2
    n_layers = 2

    # Create feature map and ansatz
    feature_map = chebyshev_gates
    ansatz = hea(n_qubits, n_layers)

    # Add noise to the gates
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.5),)
    feature_map = partial(feature_map, noise=noise)
    ansatz = partial(hea, noise=noise)

    # Initialize QNN
    qnn = QNN(
        n_qubits=n_qubits,
        feature_map_fn=feature_map,
        ansatz=ansatz
    )

    # Run on batched input data
    key = jax.random.PRNGKey(0)
    params = jax.random.uniform(key, (qnn.n_vparams,))
    x = jnp.array([[0.1, 0.2], [0.3, 0.4]])

    # Get expectation values
    y = qnn(params, x)  # Shape: (batch_size, 1)
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array
from horqrux import QuantumCircuit, expectation
from horqrux.noise import NoiseProtocol
from horqrux.primitives.primitive import Primitive
from horqrux.utils import random_state, zero_state
from horqrux.utils.operator_utils import DiffMode, TargetQubits
from jax import vmap
from loguru import logger

from qedft.models.quantum.feature_maps import chebyshev_gates, direct_gates, product_gates
from qedft.models.quantum.hardware_ansatz import hea
from qedft.models.quantum.measurement import total_magnetization_ops


class QNN(QuantumCircuit):
    def __init__(
        self,
        n_qubits: int,
        feature_map_fn: list[Primitive],
        ansatz: list[Primitive],
        state: Array | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        """Initialize a quantum neural network.

        The QNN consists of:
        1. A feature map to encode classical data into quantum states
        2. A variational ansatz with trainable parameters
        3. Measurement operators to extract quantum expectation values

        Args:
            n_qubits: Number of qubits in the circuit
            feature_map_fn: Function that generates feature map gates given input data
            ansatz: List of parametrized quantum gates for the variational part
            state: Optional initial quantum state, defaults to |0...0>

        Example:
            >>> qnn = QNN(
            ...     n_qubits=2,
            ...     feature_map_fn=chebyshev_gates,
            ...     ansatz=hea(2, 2)
            ... )
            >>> params = jnp.zeros(qnn.n_vparams)
            >>> x = jnp.array([0.1, 0.2])
            >>> y = qnn(params, x)
            This will apply the qnn on each element of the batch.
            For example, if x has 2 elements, then y will have 2 elements.
            If you do:
            >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
            >>> y = qnn(params, x)

        Note: Then y will have shape (2, 1) because there are 2 elements in the batch
            and each element is one feature that goes into a gate in the feature map.
            if n_features > n_qubits, then error will be raised.



        """
        super().__init__(n_qubits=n_qubits, operations=ansatz)
        self.feature_map_fn = feature_map_fn
        self.observable: list[Primitive] = total_magnetization_ops(n_qubits)
        self.state = state if state is not None else random_state(n_qubits)
        self.ansatz = ansatz
        self.target_idx = TargetQubits(tuple((i,) for i in range(n_qubits)))
        self.diff_mode = diff_mode
        self.n_shots = n_shots
        self.key = key

    @partial(vmap, in_axes=(None, None, 0))
    def __call__(self, param_values: Array, x: Array) -> Array:
        """Run quantum circuit on input data.

        The method is vmapped over the input data dimension, allowing for batch
        processing.

        Args:
            param_values: Variational parameters for the ansatz
            x: Input data array of shape (batch_size, input_dim)

        Returns:
            Array of expectation values with shape (batch_size,)

        Example:
            >>> qnn = LocalQNN(2, chebyshev_gates, hea(2, 2))
            >>> params = jnp.zeros(qnn.n_vparams)
            >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
            >>> y = qnn(params, x)  # Shape: (2,)
        """
        feature_map_assigned = self.feature_map_fn(x, self.target_idx)
        param_dict = {name: val for name, val in zip(self.param_names, param_values)}

        # Create a combined circuit instead of just concatenating lists
        combined_operations = feature_map_assigned + self.ansatz
        combined_circuit = QuantumCircuit(
            n_qubits=self.n_qubits,
            operations=combined_operations,
        )

        # Sum, old version of horqrux no sum needed here
        # Summing the expectation values of Zi for i in range(n_qubits)
        return jnp.sum(
            expectation(
                self.state,
                combined_circuit,
                self.observable,
                {**param_dict},
                n_shots=self.n_shots,
                diff_mode=self.diff_mode,
                key=self.key,
            ),
        )


def amplitude_encode(x: Array, n_qubits: int, hilbert_dim: int, data_dim: int) -> Array:
    """Encode classical data into quantum amplitudes of a state vector.

    This function takes a vector of data and encodes it into the amplitudes
    of a quantum state. The function handles cases where the data dimension
    doesn't match the Hilbert space dimension by padding or truncating.

    Note that if shape changes it will trigger recompilation of the function.

    Args:
        x (Array): Input data of shape (N,)
        n_qubits (int): Number of qubits in the target state

    Returns:
        State: Normalized quantum state with data encoded in amplitudes
    """

    # New implementation
    # x_flat = jnp.ravel(x)
    # # Handle case where data doesn't fit in the Hilbert space
    # if data_dim >= hilbert_dim:
    #     # Truncate the data to fit
    #     x_flat = x_flat[:hilbert_dim]
    # elif data_dim < hilbert_dim:
    #     # Pad with zeros to fill the Hilbert space
    #     padding = jnp.zeros(hilbert_dim - data_dim)
    #     x_flat = jnp.concatenate([x_flat, padding])
    # # Normalize the vector
    # norm = jnp.linalg.norm(x_flat)
    # # Avoid division by zero
    # x_normalized = jnp.where(norm > 0, x_flat / norm, x_flat)
    # # Reshape to quantum state format [2, 2, ..., 2] (n_qubits times)
    # state = jnp.reshape(x_normalized, tuple([2 for _ in range(n_qubits)]))

    # Legacy implementation
    x_flat = jnp.ravel(x)
    new_state = jnp.zeros(shape=(hilbert_dim,), dtype=jnp.complex128)
    # # Take the part of input to fit in the Hilbert space...
    x = x_flat[:hilbert_dim]
    # # New shape of x, if smaller than Hilbert space then it is embedded in the first part as below...
    shape_x = jnp.shape(x)[0]
    new_state = new_state.at[0:shape_x].set(x)
    new_state = new_state / jnp.linalg.norm(new_state)
    # # Reshape x to be compatible with the number of qubits, to be a state-vector.
    new_state = jnp.reshape(new_state, tuple([2 for _ in range(n_qubits)]))

    return new_state


class AmplitudeEncodingQNN(QuantumCircuit):
    def __init__(
        self,
        n_qubits: int,
        ansatz: list[Primitive],
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        """Initialize a quantum neural network.

        The QNN consists of:
        1. A feature map to encode classical data into quantum states
        2. A variational ansatz with trainable parameters
        3. Measurement operators to extract quantum expectation values

        Args:
            n_qubits: Number of qubits in the circuit
            feature_map_fn: Function that generates feature map gates given input data
            ansatz: List of parametrized quantum gates for the variational part
            state: Optional initial quantum state, defaults to |0...0>
            diff_mode: Differential mode for the QNN
            n_shots: Number of shots for the QNN
            key: Random key for the QNN

        Example:
            >>> qnn = QNN(
            ...     n_qubits=2,
            ...     feature_map_fn=chebyshev_gates,
            ...     ansatz=hea(2, 2)
            ... )
            >>> params = jnp.zeros(qnn.n_vparams)
            >>> x = jnp.array([0.1, 0.2])
            >>> y = qnn(params, x)
            This will apply the qnn on each element of the batch.
            For example, if x has 2 elements, then y will have 2 elements.
            If you do:
            >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
            >>> y = qnn(params, x)

        Note: Then y will have shape (2, 1) because there are 2 elements in the batch
            and each element is one feature that goes into a gate in the feature map.
            if n_features > n_qubits, then error will be raised.



        """
        super().__init__(n_qubits=n_qubits, operations=ansatz)
        self.observable: list[Primitive] = total_magnetization_ops(n_qubits)
        self.ansatz = ansatz
        self.target_idx = TargetQubits(tuple((i,) for i in range(n_qubits)))
        self.diff_mode = diff_mode
        self.n_shots = n_shots
        self.key = key

    @partial(vmap, in_axes=(None, None, 0))
    def __call__(self, param_values: Array, x: Array) -> Array:
        """Run quantum circuit on input data.

        The method is vmapped over the input data dimension, allowing for batch
        processing.

        Args:
            param_values: Variational parameters for the ansatz
            x: Input data array of shape (batch_size, input_dim)

        Returns:
            Array of expectation values with shape (batch_size,)

        Example:
            >>> qnn = LocalQNN(2, chebyshev_gates, hea(2, 2))
            >>> params = jnp.zeros(qnn.n_vparams)
            >>> x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
            >>> y = qnn(params, x)  # Shape: (2,)
        """
        param_dict = {name: val for name, val in zip(self.param_names, param_values)}

        # Create a combined circuit instead of just concatenating lists
        combined_operations = self.ansatz
        combined_circuit = QuantumCircuit(
            n_qubits=self.n_qubits,
            operations=combined_operations,
        )

        # Amplitude encode the input data, no feature map here
        hilbert_dim = 2**self.n_qubits
        x_flat = jnp.ravel(x)
        data_dim = x_flat.shape[0]
        state = amplitude_encode(x, self.n_qubits, hilbert_dim, data_dim)

        return jnp.sum(
            expectation(
                state,
                combined_circuit,
                self.observable,
                {**param_dict},
                n_shots=self.n_shots,
                diff_mode=self.diff_mode,
                key=self.key,
            ),
        )


class DirectQNN(QNN):
    """Direct QNN (Direct QNN).

    This class implements a local quantum neural network with a direct feature map.
    The direct feature map is a product of single-qubit gates applied to the input data.
    """

    def __init__(
        self,
        n_qubits: int,
        ansatz: list[Primitive],
        state: Array | None = None,
        noise: NoiseProtocol | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        direct_gates_noisy = partial(direct_gates, noise=noise)
        super().__init__(n_qubits, direct_gates_noisy, ansatz, state, diff_mode, n_shots, key)


class ChebyshevQNN(QNN):
    """Chebyshev QNN (Chebyshev QNN).

    This class implements a local quantum neural network with a product feature map.
    The product feature map is a product of sine and cosine gates applied to the input data.
    """

    def __init__(
        self,
        n_qubits: int,
        ansatz: list[Primitive],
        state: Array | None = None,
        noise: NoiseProtocol | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        chebyshev_gates_noisy = partial(chebyshev_gates, noise=noise)
        super().__init__(n_qubits, chebyshev_gates_noisy, ansatz, state, diff_mode, n_shots, key)


class ProductQNN(QNN):
    """Product QNN (Product QNN).

    This class implements a local quantum neural network with a product feature map.
    The product feature map is a product of sine and cosine gates applied to the input data.
    """

    def __init__(
        self,
        n_qubits: int,
        ansatz: list[Primitive],
        state: Array | None = None,
        map_fn: Callable[[float], float] = jnp.arcsin,
        noise: NoiseProtocol | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        product_gates_with_map_fn = partial(product_gates, map_fn=map_fn, noise=noise)
        super().__init__(
            n_qubits,
            product_gates_with_map_fn,
            ansatz,
            state,
            diff_mode,
            n_shots,
            key,
        )


class LocalQNN(QNN):
    """Local QNN (Local Quantum Neural Network).

    This class implements a quantum neural network that processes each input element
    independently. It expects inputs of shape (N,) where N is the number of data points.
    If inputs have shape (N, P), it will raise an error as this indicates global data.

    This implementation uses a specified QNN type (DirectQNN, ChebyshevQNN, or ProductQNN)
    as its underlying implementation.
    """

    def __init__(
        self,
        n_qubits: int,
        qnn_type: str = "DirectQNN",
        ansatz: list[Primitive] | None = None,
        state: Array | None = None,
        map_fn: Callable[[float], float] | None = None,
        noise: NoiseProtocol | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        """Initialize a local quantum neural network.

        Args:
            n_qubits: Number of qubits in the circuit
            qnn_type: Type of QNN to use - "DirectQNN", "ChebyshevQNN", or "ProductQNN" (default)
            ansatz: List of parametrized quantum gates for the variational part
            state: Optional initial quantum state, defaults to random state
            map_fn: Optional mapping function for product feature map (default: jnp.arcsin)

        Raises:
            ValueError: If qnn_type is not one of the supported types
        """
        # Create default ansatz if not provided
        if ansatz is None:
            ansatz = hea(n_qubits, 2, noise=noise)  # Default to 2-layer HEA

        # Create the appropriate QNN based on type
        if qnn_type == "DirectQNN":
            self.qnn = DirectQNN(
                n_qubits=n_qubits,
                ansatz=ansatz,
                state=state,
                noise=noise,
                diff_mode=diff_mode,
                n_shots=n_shots,
                key=key,
            )
        elif qnn_type == "ChebyshevQNN":
            self.qnn = ChebyshevQNN(
                n_qubits=n_qubits,
                ansatz=ansatz,
                state=state,
                noise=noise,
                diff_mode=diff_mode,
                n_shots=n_shots,
                key=key,
            )
        elif qnn_type == "ProductQNN":
            map_function = map_fn if map_fn is not None else jnp.arcsin
            self.qnn = ProductQNN(
                n_qubits=n_qubits,
                ansatz=ansatz,
                state=state,
                map_fn=map_function,
                noise=noise,
                diff_mode=diff_mode,
                n_shots=n_shots,
                key=key,
            )
        else:
            raise ValueError(
                f"Unsupported QNN type: {qnn_type}. Choose from 'DirectQNN', 'ChebyshevQNN', or 'ProductQNN'.",
            )

        # Initialize parent class with the same parameters as the underlying QNN
        super().__init__(
            n_qubits=n_qubits,
            feature_map_fn=self.qnn.feature_map_fn,
            ansatz=ansatz,
            state=state,
        )

    def __call__(self, param_values: Array, x: Array) -> Array:
        """Run quantum circuit on input data.

        The method is vmapped over the input data dimension, allowing for batch
        processing of local data points.

        Args:
            param_values: Variational parameters for the ansatz
            x: Input data array of shape (batch_size,) for local processing

        Returns:
            Array of expectation values with shape (batch_size,)

        Raises:
            ValueError: If input data has shape (batch_size, features) indicating global data
        """
        # Check if input has more than 1 dimension, indicating global data
        # Use jnp.ndim for JAX jit compatibility instead of checking shape length
        if jnp.ndim(x) > 1:
            raise ValueError(
                f"LocalQNN expects inputs of shape (N,) for local processing, "
                f"but got shape {x.shape}. For inputs with shape (N, P), "
                f"use a global QNN variant instead.",
            )

        return self.qnn(param_values, x)


class GlobalQNN(LocalQNN):
    """Global QNN (Global Quantum Neural Network).

    This class implements a quantum neural network that processes all input elements
    simultaneously. It expects inputs of shape (N, P) where N is the number of data points
    and P is the number of features.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        qnn_type: str = "DirectQNN",
        ansatz: list[Primitive] | None = None,
        state: Array | None = None,
        map_fn: Callable[[float], float] | None = None,
        noise: NoiseProtocol | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(n_qubits, qnn_type, ansatz, state, map_fn, noise, diff_mode, n_shots, key)
        self.n_features = n_features

    def __call__(self, param_values: Array, x: Array) -> Array:
        """Run quantum circuit on input data.

        Args:
            param_values: Variational parameters for the ansatz
            x: Input data array of shape (batch_size, features) for global processing

        Returns:
            Array of expectation values with shape (batch_size,)

        Raises:
            ValueError: If input data doesn't have shape (batch_size, features) or if features > n_qubits
        """
        # Check if input has the correct shape for global processing
        if self.n_qubits < self.n_features:
            raise ValueError(
                f"Number of features ({self.n_features}) exceeds number of qubits ({self.n_qubits}). "
                f"Increase n_qubits to at least {self.n_features} to embed this data in angles.",
            )

        # Reshape input to (batch_size, features)
        x = jnp.reshape(x, (-1, self.n_features))

        # Check if input has the correct shape for global processing
        if jnp.ndim(x) < 2:
            raise ValueError(
                f"GlobalQNN expects inputs of shape (N, P) for global processing, "
                f"but got shape {x.shape}. For inputs with shape (N,), "
                f"use a LocalQNN instead.",
            )

        if x.shape[1] > self.n_qubits:
            raise ValueError(
                f"Number of features ({x.shape[1]}) exceeds number of qubits ({self.n_qubits}). "
                f"Increase n_qubits to at least {x.shape[1]} to process this data.",
            )

        # Use vmap to process each batch element
        return self.qnn(param_values, x)


class GlobalAmplitudeEncodingQNN(AmplitudeEncodingQNN):
    """Global Amplitude Encoding QNN (Global Amplitude Encoding Quantum Neural Network).

    This class implements a quantum neural network that processes all input elements
    simultaneously. It expects inputs of shape (N, P) where N is the number of data points
    and P is the number of features.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        ansatz: list[Primitive] | None = None,
        diff_mode: DiffMode = DiffMode.AD,
        n_shots: int = 0,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        super().__init__(n_qubits, ansatz=ansatz, key=key, diff_mode=diff_mode, n_shots=n_shots)
        self.n_features = n_features

    def __call__(self, param_values: Array, x: Array) -> Array:
        """Run quantum circuit on input data.

        Args:
            param_values: Variational parameters for the ansatz
            x: Input data array of shape (batch_size, features) for global processing

        Returns:
            Array of expectation values with shape (batch_size,)

        Raises:
            ValueError: If input data doesn't have shape (batch_size, features) or if features > n_qubits
        """

        # This model will embed anything in a given statevector size

        # Check if input has the correct shape for global processing
        # if 2**self.n_qubits < self.n_features:
        #     raise ValueError(
        #         f"Number of features ({self.n_features}) exceeds number of amplitudes ({2**self.n_qubits}). "
        #         f"Increase n_qubits to at least {self.n_features} to embed this data in amplitudes.",
        #     )

        # Reshape input to (batch_size, features)
        x = jnp.reshape(x, (-1, self.n_features))

        # if x.shape[1] > 2**self.n_qubits:
        #     raise ValueError(
        #         f"Number of features ({x.shape[1]}) exceeds number of amplitudes ({2**self.n_qubits}). "
        #         f"Increase n_qubits to at least {x.shape[1]} to process this data.",
        #     )

        # Use vmap to process each batch element
        return super().__call__(param_values, x)


# ================================================================
# Factory function
# ================================================================


def build_qnn(
    n_qubits: int,
    n_features: int,
    ansatz: list[Primitive],
    qnn_type: str = "GlobalQNN",  # "LocalQNN", "GlobalQNN", "GlobalAmplitudeEncodingQNN"
    layer_type: str = "DirectQNN",  # DirectQNN has no domain restriction
    state: Array | None = None,
    map_fn: Callable[[float], float] | None = None,
    grids: jnp.ndarray | None = None,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    **kwargs,
) -> tuple[Callable, Callable]:
    """Builds a quantum neural network with init and apply functions.

    This function creates initialization and application functions for a quantum neural network
    with the specified architecture. It supports both local and global QNN types with different
    feature map implementations.

    Args:
        n_qubits: Number of qubits in the circuit
        ansatz: List of parametrized quantum gates for the variational part
        qnn_type: Type of QNN architecture - "LocalQNN" or "GlobalQNN" (default: "GlobalQNN")
        layer_type: Type of feature map to use - "DirectQNN", "ChebyshevQNN", or "ProductQNN" (default: "ProductQNN")
        state: Optional initial quantum state (default: None, will use random state)
        map_fn: Optional mapping function for product feature map (default: jnp.arcsin)
        **kwargs: Additional arguments (unused)

    Returns:
        (init_fn, apply_fn) tuple of network functions:
        - init_fn: Function that initializes parameters given a random key
        - apply_fn: Function that applies the QNN to input data using parameters

    Raises:
        ValueError: If qnn_type is not one of the supported types

    Example:
        >>> n_qubits = 2
        >>> ansatz = hea(n_qubits, 2)
        >>> init_fn, apply_fn = build_qnn(
        ...     n_qubits=n_qubits,
        ...     ansatz=ansatz,
        ...     qnn_type="GlobalQNN",
        ...     layer_type="ProductQNN",
        ...     map_fn=jnp.arcsin
        ... )
        >>> key = jax.random.PRNGKey(0)
        >>> _, params = init_fn(key)
        >>> x = jnp.array([[0.5, 0.6], [0.9, 1.0]])
        >>> result = apply_fn(params, x)
    """
    # Warning about the layer_type
    if layer_type == "ProductQNN":
        logger.info(
            "ProductQNN can generate NaNs due to domain restriction of the arcsin function. "
            "Modify the map_fn (default: jnp.arcsin) to avoid this issue or normalize the input data.",
        )

    logger.info(f"Building {qnn_type} QNN with {layer_type} layer_type")

    # Create the appropriate QNN based on type
    def create_local_qnn():
        return LocalQNN(
            n_qubits=n_qubits,
            qnn_type=layer_type,
            ansatz=ansatz,
            state=state,
            map_fn=map_fn,
            noise=noise,
            diff_mode=diff_mode,
            n_shots=n_shots,
            key=key,
        )

    def create_global_qnn():
        return GlobalQNN(
            n_qubits=n_qubits,
            n_features=n_features,
            qnn_type=layer_type,
            ansatz=ansatz,
            state=state,
            map_fn=map_fn,
            noise=noise,
            diff_mode=diff_mode,
            n_shots=n_shots,
            key=key,
        )

    def create_global_amplitude_encoding_qnn():
        # Noise comes through ansatz and feature map, but here no feature map via gates
        return GlobalAmplitudeEncodingQNN(
            n_qubits=n_qubits,
            n_features=n_features,
            ansatz=ansatz,
            diff_mode=diff_mode,
            n_shots=n_shots,
            key=key,
        )

    if qnn_type == "LocalQNN":
        qnn_class = create_local_qnn
    elif qnn_type == "GlobalQNN":
        qnn_class = create_global_qnn
    elif qnn_type == "GlobalAmplitudeEncodingQNN":
        qnn_class = create_global_amplitude_encoding_qnn
    else:
        raise ValueError(
            f"Unsupported QNN type: {qnn_type}. Choose from 'LocalQNN', 'GlobalQNN', 'GlobalAmplitudeEncodingQNN'.",
        )

    # Create a temporary QNN instance to get the parameter count
    temp_qnn = qnn_class()
    n_params = temp_qnn.n_vparams
    num_grids = grids.shape[0]

    def init_fn(rng, input_shape):
        """Initialize the QNN parameters.

        Args:
            rng: JAX random key
            input_shape: Ignored, included for API compatibility

        Returns:
            Randomly initialized parameters for the QNN
        """
        del input_shape  # Not used but included for API compatibility
        # Output shape is (batch_size, num_grids, 1) and parameters shape is (n_params,)
        return (-1, num_grids, 1), jax.random.uniform(
            rng,
            shape=(n_params,),
            minval=-0.1,
            maxval=0.1,
        )

    def apply_fn(params, inputs, **kwargs):
        """Apply the QNN to the inputs.

        Args:
            params: QNN parameters
            inputs: Input data
            **kwargs: Additional arguments (unused)

        Returns:
            QNN output (expectation values)
        """
        del kwargs  # Not used but included for API compatibility
        qnn = qnn_class()
        # might cause a bug with wrap_self_interaction=True because of the squeeze() for
        # global QNN; for local QNN it's fine, tested
        return qnn(params, inputs).squeeze()

    return init_fn, apply_fn


if __name__ == "__main__":

    # Demonstration of quantum neural network models.
    # This example shows how to:
    # 1. Create different types of QNNs
    # 2. Apply them to input data
    # 3. Verify JIT compilation works correctly
    # 4. Use the build_local_qnn factory function

    # Basic configuration
    n_qubits = 2
    n_layers = 2
    num_grids = 4
    n_features = 2

    state = random_state(n_qubits)
    state = zero_state(n_qubits)

    # Create an ansatz
    ansatz = hea(n_qubits, n_layers)
    grids = jnp.linspace(0, 1, num_grids)

    # Sample input data
    # Local QNN has 1 dimension for the input data.
    # Global QNN has 2 dimensions for the input data, batch size and features.
    x = jnp.array([0.5, 0.6, 0.8, 0.9, 1.0])  # For single-input QNNs
    x_global = jnp.array([[0.5, 0.6], [0.9, 1.0]])  # For GlobalQNN
    # The value 20 causes NaNs with ProductQNN layer_type
    # x_global = jnp.array([[0.5, 0.6], [0.9, 20.0]])  # For GlobalQNN
    # Generates NaNs
    # x_global = jnp.array([[0.80993999, -3.7785351], [-3.67977304, 11.3710262]])  # For GlobalQNN

    print("ndim(x)", jnp.ndim(x))
    print("ndim(x_global)", jnp.ndim(x_global))

    # Generate random parameters
    key = jax.random.PRNGKey(0)

    print("=== Direct QNN instantiation ===")

    # Initialize different QNN types
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),)
    noisy_chebyshev_gates = partial(chebyshev_gates, noise=noise)
    circuits = {
        "Standard": QNN(
            n_qubits=n_qubits,
            feature_map_fn=chebyshev_gates,
            ansatz=ansatz,
            state=state,
        ),
        "Standard Noisy": QNN(
            n_qubits=n_qubits,
            feature_map_fn=noisy_chebyshev_gates,
            ansatz=ansatz,
            state=state,
        ),
        "Direct": DirectQNN(n_qubits=n_qubits, ansatz=ansatz, state=state),
        "Chebyshev": ChebyshevQNN(n_qubits=n_qubits, ansatz=ansatz, state=state),
        "Product": ProductQNN(
            n_qubits=n_qubits,
            ansatz=ansatz,
            state=state,
            map_fn=jnp.arcsin,
        ),
        "Global": GlobalQNN(n_qubits=n_qubits, n_features=n_features, ansatz=ansatz, state=state),
        "Global Noisy": GlobalQNN(
            n_qubits=n_qubits,
            n_features=n_features,
            ansatz=ansatz,
            state=state,
            noise=noise,
        ),
        "Amplitude Encoding": AmplitudeEncodingQNN(n_qubits=n_qubits, ansatz=ansatz),
    }

    # Get parameters for the first circuit (all have same number of parameters)
    param_values = jax.random.uniform(key, shape=(circuits["Standard"].n_vparams,))

    # Test each circuit with and without JIT
    for name, circuit in circuits.items():
        # Skip GlobalQNN for the standard input
        if name == "Global" or name == "Global Noisy" or name == "Amplitude Encoding":
            # GlobalQNN uses different input format
            jitted_fn = jax.jit(circuit.__call__)
            result = circuit(param_values, x_global)
            result_jit = jitted_fn(param_values, x_global)
        else:
            jitted_fn = jax.jit(circuit.__call__)
            result = circuit(param_values, x)
            result_jit = jitted_fn(param_values, x)

        print(f"\n{name} QNN results:")
        print(f"  Standard: \n{result}")
        print(f"  JIT:      \n{result_jit}")
        assert jnp.allclose(result, result_jit), f"JIT mismatch for {name} QNN"

    print("\n=== Factory function usage ===")

    print("=== GlobalQNN with ProductQNN layer_type (can give NaNs) ===")
    init_fn, apply_fn = build_qnn(
        n_qubits=n_qubits,
        n_features=n_features,
        ansatz=ansatz,
        qnn_type="GlobalQNN",
        layer_type="ProductQNN",
        map_fn=jnp.arcsin,
        grids=grids,
    )

    # Initialize parameters
    _, params = init_fn(key, input_shape=(num_grids,))

    # Apply QNN with and without JIT
    jitted_apply_fn = jax.jit(apply_fn)
    result = apply_fn(params, x_global)
    result_jit = jitted_apply_fn(params, x_global)
    print(f"  Standard: \n{result}")
    print(f"  JIT:      \n{result_jit}")

    # Demonstrate the build_local_qnn factory function
    qnn_types = ["LocalQNN", "GlobalQNN", "GlobalAmplitudeEncodingQNN"]

    for qnn_type in qnn_types:
        # Create initialization and application functions
        init_fn, apply_fn = build_qnn(
            n_qubits=n_qubits,
            n_features=n_features,
            ansatz=ansatz,
            qnn_type=qnn_type,
            layer_type="DirectQNN",
            grids=grids,
            diff_mode=DiffMode.AD,
            n_shots=0,
            key=key,
        )

        # Initialize parameters
        _, params = init_fn(key, input_shape=(num_grids,))

        # Apply QNN with and without JIT
        jitted_apply_fn = jax.jit(apply_fn)
        if qnn_type == "LocalQNN":
            result = apply_fn(params, x)
            result_jit = jitted_apply_fn(params, x)
        else:
            result = apply_fn(params, x_global)
            result_jit = jitted_apply_fn(params, x_global)

        print(f"\n{qnn_type} via factory function:")
        print(f"  Standard: \n{result}")
        print(f"  JIT:      \n{result_jit}")
        print(f"  Standard shape: {result.shape}")
        print(f"  JIT shape:      {result_jit.shape}")
        assert jnp.allclose(result, result_jit), f"JIT mismatch for {qnn_type}"

        # Speed test of the apply_fn vs jitted_apply_fn
        import time

        n_iterations = 100
        # Timing for standard apply_fn
        start_time = time.time()
        for _ in range(n_iterations):
            if qnn_type == "LocalQNN":
                _ = apply_fn(params, x)
            else:
                _ = apply_fn(params, x_global)
        standard_time = time.time() - start_time
        # Timing for jitted apply_fn
        start_time = time.time()
        for _ in range(n_iterations):
            if qnn_type == "LocalQNN":
                _ = jitted_apply_fn(params, x)
            else:
                _ = jitted_apply_fn(params, x_global)
        jitted_time = time.time() - start_time
        # Calculate speedup
        speedup = standard_time / jitted_time if jitted_time > 0 else float("inf")
        print(f"  Speed comparison ({n_iterations} iterations):")
        print(f"    Standard: {standard_time:.4f} seconds")
        print(f"    JIT:      {jitted_time:.4f} seconds")
        print(f"    Speedup:  {speedup:.2f}x")
