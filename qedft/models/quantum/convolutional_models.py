"""
Convolutional models for Kohn-Sham DFT.
"""

import copy
from collections.abc import Callable

import jax
import jax.numpy as jnp
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import DiffMode
from jax.example_libraries import stax
from loguru import logger

from qedft.models.networks import (
    AmplitudeEncodingGlobalQNNLayer,
    BatchedGlobalMLP,
    GlobalMLP,
    GlobalQNNLayer,
)
from qedft.models.utils import count_parameters


def compute_kernel_width_per_layer(
    input_dimension: int,
    largest_kernel_dimension: int,
    max_number_conv_layers: int = 100,  # by default do max number of convolutions
):
    """
    Given a dimension of the input, compute the number of layers of the convolutional DQC
    that can be applied

    Args:
        input_dimension (int): number of grid points for density
        largest_kernel_dimension (int): max number of features (if 4, then 4 datapoints are placed in the feature map in 4 rotation gates, so you will need 4 qubits, and each has Ry with a feature value as angle)
    Returns:
        list_kernel_dimensions, list_outputs_per_conv_layer: list with kernel dimensions and list with output dimensions
    """

    def largest_divisor_of_input_dimension(
        input_dimension,
        largest_kernel_dimension,
    ):
        for i in range(largest_kernel_dimension, 0, -1):
            if input_dimension % i == 0:
                return i
        return 1

    # Dimensions of output of each layer
    list_kernel_dimensions = []
    list_outputs_per_conv_layer = []

    # Some starting values
    kernel_dimension = 2.0
    input_dimension_old = 2.0

    conv_layer_number = 0

    while (
        input_dimension > 0
        and input_dimension_old % kernel_dimension == 0
        and kernel_dimension > 1
        and input_dimension > 1
        and conv_layer_number < max_number_conv_layers
    ):
        kernel_dimension = largest_divisor_of_input_dimension(
            input_dimension,
            largest_kernel_dimension,
        )
        input_dimension_old = copy.deepcopy(input_dimension)
        input_dimension = input_dimension // kernel_dimension

        if kernel_dimension == input_dimension and input_dimension == 1:
            pass
        else:
            list_kernel_dimensions.append(kernel_dimension)
            list_outputs_per_conv_layer.append(input_dimension)

        conv_layer_number += 1

    return list_kernel_dimensions, list_outputs_per_conv_layer


def construct_convolutional_model(
    n_qubits,
    n_var_layers,
    n_out,
    input_dimension,
    largest_kernel_width,
    max_number_conv_layers: int = 100,
    list_qubits_per_layer: list = [],
    force_qubits_per_layer_is_kernel_width: bool = False,
    normalization: float = 1.0,
    last_layer_type: str = "dense",
    use_bias_mlp: bool = False,
    last_layer_features: list = [1],
    use_qnn_layers: bool = True,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    use_amplitude_encoding: bool = False,
):
    """
    Construct a convolutional model with a given input dimension and largest kernel width

    Args:
            n_qubits ([type]): Number of qubits
            n_var_layers ([type]): [description]
            n_out ([type]): [description]
            input_dimension ([type]): [description]
            largest_kernel_width ([type]): [description]
            max_number_conv_layers ([type]): [description]
            list_qubits_per_layer (list, optional): [description]. Defaults to [].
            force_qubits_per_layer_is_kernel_width (bool, optional): [description]. Defaults to False.
            use_qnn_layers (bool, optional): Whether to use QNN layers. Defaults to True.
            diff_mode (DiffMode, optional): [description]. Defaults to DiffMode.AD.
            n_shots (int, optional): [description]. Defaults to 0.
            key (jax.random.PRNGKey, optional): [description]. Defaults to jax.random.PRNGKey(0).
            use_amplitude_encoding (bool, optional): Whether to use amplitude encoding. Defaults to False.
    Returns:
            [type]: [description]
    """
    if len(list_qubits_per_layer) != 0 and force_qubits_per_layer_is_kernel_width is False:
        list_qubits_per_layer = list_qubits_per_layer[:max_number_conv_layers]

    (
        list_kernel_dimensions,
        list_outputs_per_conv_layer,
    ) = compute_kernel_width_per_layer(
        input_dimension,
        largest_kernel_width,
        max_number_conv_layers,
    )

    list_kernel_dimensions, list_outputs_per_conv_layer = (
        list_kernel_dimensions[:max_number_conv_layers],
        list_outputs_per_conv_layer[:max_number_conv_layers],
    )

    print(
        "list_kernel_dimensions , list_outputs_per_conv_layer: ",
        list_kernel_dimensions,
        list_outputs_per_conv_layer,
    )

    list_conv_layers = []

    # Select the QNN layer type
    QNNLayer = AmplitudeEncodingGlobalQNNLayer if use_amplitude_encoding else GlobalQNNLayer
    logger.info(f"Using {QNNLayer.__name__} layer")

    # Create mock grids for GlobalQNN initialization
    grids = jnp.ones(input_dimension)

    for i, (kernel_width, n_outputs) in enumerate(
        zip(list_kernel_dimensions, list_outputs_per_conv_layer),
    ):

        if not force_qubits_per_layer_is_kernel_width:
            if len(list_qubits_per_layer) != 0:
                if list_qubits_per_layer[i] < kernel_width:
                    raise ValueError(
                        "The number of qubits per layer cannot be smaller than"
                        " the kernel width. Cannot embed the data since there"
                        " are not enough qubits (one rotation - per data point"
                        " - per qubit).",
                    )
                n_qubits_layer = list_qubits_per_layer[i]
            else:
                # by default all quantum layers have the same number of qubits
                n_qubits_layer = n_qubits
        else:
            # never have 1 qubits for the variational circuits
            if kernel_width > 1:
                n_qubits_layer = kernel_width
            else:
                n_qubits_layer = 2

        if use_qnn_layers:
            logger.info(f"GlobalQNNLayer Layer {i}: n_qubits_layer {n_qubits_layer}")

            # Create GlobalQNN configuration
            qnn_config = {
                "n_qubits": n_qubits_layer,
                "n_var_layers": n_var_layers,
                "normalization": normalization,
                "use_amplitude_encoding": True,
                "n_features": kernel_width,
                "diff_mode": diff_mode,
                "n_shots": n_shots,
                "key": key,
            }

            # Create GlobalQNN instance and get its build_network function
            qnn = QNNLayer(config_dict=qnn_config)

            # Use GlobalQNNLayer
            if i <= len(list_kernel_dimensions) - 2:
                # Get init_fn and apply_fn from build_network
                init_fn, apply_fn = qnn.build_network(grids, noise=noise)
                list_conv_layers.append((init_fn, apply_fn))
            else:
                # For the last layer, we might want different configuration
                qnn_config["wrap_with_negative_transform"] = True  # Equivalent to "swish_and_norm"
                qnn_config["n_features"] = kernel_width
                qnn = QNNLayer(config_dict=qnn_config)
                init_fn, apply_fn = qnn.build_network(grids, noise=noise)
                list_conv_layers.append((init_fn, apply_fn))
        else:
            # BatchedGlobalMLP if you want to replace by a simple MLP
            logger.info(
                f"BatchedGlobalMLP Layer {i}: n_qubits_layer {n_qubits_layer}, kernel_width {kernel_width}",
            )
            bmlp_config = {
                "n_neurons": n_qubits_layer,
                "n_layers": n_var_layers,
                "use_bias": use_bias_mlp,
                "density_normalization_factor": normalization,
                "n_features": kernel_width,
                "activation": "tanh",
                "n_outputs": 1,
            }
            bmlp = BatchedGlobalMLP(bmlp_config)
            init_fn, apply_fn = bmlp.build_network(grids)
            list_conv_layers.append((init_fn, apply_fn))

    # Append an MLP to reduce to 1 scalar output
    # Or even just a linear transformation.
    if list_outputs_per_conv_layer[-1] > 1:
        # Dense by default worked well in training 2 bond separations to chem.
        # acc. and low density error.
        if last_layer_type == "dense":
            logger.info(
                "Adding a single dense layer at the end (outputs last layer"
                f" {list_outputs_per_conv_layer[-1]})",
            )
            # Just a linear combination without bias
            # so the outputs of DQC are just a linear combination

            # Create a simple dense layer with stax-style API
            def init_dense(rng, input_shape):
                k1, k2 = jax.random.split(rng)
                output_size = 1
                input_size = list_outputs_per_conv_layer[
                    -1
                ]  # Use the known output size from previous layer
                w = jax.random.normal(k1, (input_size, output_size)) * 0.1
                b = None
                if use_bias_mlp:
                    b = jax.random.normal(k2, (output_size,)) * 0.1
                return (output_size,), (w, b)

            def apply_dense(params, inputs, **kwargs):
                w, b = params
                # Ensure inputs have the right shape for matrix multiplication
                inputs_reshaped = inputs.reshape(-1, w.shape[0])
                out = jnp.dot(inputs_reshaped, w)
                if b is not None:
                    out = out + b
                # Make sure it outputs a scalar in shape (N,)
                return out.reshape(-1)  # Flatten output

            list_conv_layers.append((init_dense, apply_dense))

        elif last_layer_type == "mlp":
            # MLP in case we want to get rid of all the DQC chaining.
            # Chaining was hard to train.
            logger.info(
                "Adding a single MLP layer at the end (outputs last layer"
                f" {list_outputs_per_conv_layer[-1]})",
            )

            # Use GlobalMLP instead of MLP
            mlp_config = {
                "n_neurons": last_layer_features[0] if len(last_layer_features) > 0 else 64,
                "n_layers": len(last_layer_features),
                "use_bias": use_bias_mlp,
                "density_normalization_factor": normalization,
            }

            mlp = GlobalMLP(config_dict=mlp_config)
            init_fn, apply_fn = mlp.build_network(grids)
            list_conv_layers.append((init_fn, apply_fn))
        else:
            raise ValueError(
                f"Last layer type {last_layer_type} not implemented.",
            )

    return list_conv_layers


# ================================================================
# Factory function
# ================================================================


def build_conv_qnn(
    n_qubits: int,
    n_var_layers: int,
    n_out: int,
    input_dimension: int,
    largest_kernel_width: int,
    max_number_conv_layers: int = 100,
    list_qubits_per_layer: list[int] = [],
    force_qubits_per_layer_is_kernel_width: bool = False,
    normalization: float = 1.0,
    last_layer_type: str = "dense",
    use_bias_mlp: bool = False,
    last_layer_features: list[int] = [1],
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> tuple[Callable, Callable]:
    """Build a convolutional quantum neural network.

    Args:
        conv_layers: List of convolutional layers (ConvDQCLayer objects)
        grids: Grid points for the density functional calculations
        density_normalization_factor: Scale factor for input normalization

    Returns:
        (init_fn, apply_fn) tuple of network functions
    """

    # Construct model
    list_conv_layers = construct_convolutional_model(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_width,
        max_number_conv_layers=max_number_conv_layers,
        list_qubits_per_layer=list_qubits_per_layer,
        force_qubits_per_layer_is_kernel_width=force_qubits_per_layer_is_kernel_width,
        normalization=normalization,
        last_layer_type=last_layer_type,
        use_bias_mlp=use_bias_mlp,
        last_layer_features=last_layer_features,
        noise=noise,
        diff_mode=diff_mode,
        n_shots=n_shots,
        key=key,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


def build_conv_mlp(
    n_qubits: int,
    n_var_layers: int,
    n_out: int,
    input_dimension: int,
    largest_kernel_width: int,
    max_number_conv_layers: int = 100,
    list_qubits_per_layer: list[int] = [],
    force_qubits_per_layer_is_kernel_width: bool = False,
    normalization: float = 1.0,
    last_layer_type: str = "dense",
    use_bias_mlp: bool = False,
    last_layer_features: list[int] = [1],
) -> tuple[Callable, Callable]:
    """
    For comparison with the GlobalQNN but instead replace QNNs by MLPs.
    Build a convolutional quantum neural network.

    No noise, shots, key. Only Gaussian noise on outputs can be added.

    Args:
        conv_layers: List of convolutional layers (ConvDQCLayer objects)
        grids: Grid points for the density functional calculations
        density_normalization_factor: Scale factor for input normalization

    Returns:
        (init_fn, apply_fn) tuple of network functions
    """

    # Construct model
    list_conv_layers = construct_convolutional_model(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_width,
        max_number_conv_layers=max_number_conv_layers,
        list_qubits_per_layer=list_qubits_per_layer,
        force_qubits_per_layer_is_kernel_width=force_qubits_per_layer_is_kernel_width,
        normalization=normalization,
        last_layer_type=last_layer_type,
        use_bias_mlp=use_bias_mlp,
        last_layer_features=last_layer_features,
        use_qnn_layers=False,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


def build_conv_amplitude_encoding_qnn(
    n_qubits: int,
    n_var_layers: int,
    n_out: int,
    input_dimension: int,
    largest_kernel_width: int,
    max_number_conv_layers: int = 100,
    list_qubits_per_layer: list[int] = [],
    force_qubits_per_layer_is_kernel_width: bool = False,
    normalization: float = 1.0,
    last_layer_type: str = "dense",
    use_bias_mlp: bool = False,
    last_layer_features: list[int] = [1],
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> tuple[Callable, Callable]:
    """
    For comparison with the GlobalQNN but instead replace QNNs by MLPs.
    Build a convolutional quantum neural network.

    No noise, shots, key. Only Gaussian noise on outputs can be added.

    Args:
        conv_layers: List of convolutional layers (ConvDQCLayer objects)
        grids: Grid points for the density functional calculations
        density_normalization_factor: Scale factor for input normalization

    Returns:
        (init_fn, apply_fn) tuple of network functions
    """

    # Construct model
    list_conv_layers = construct_convolutional_model(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_width,
        max_number_conv_layers=max_number_conv_layers,
        list_qubits_per_layer=list_qubits_per_layer,
        force_qubits_per_layer_is_kernel_width=force_qubits_per_layer_is_kernel_width,
        normalization=normalization,
        last_layer_type=last_layer_type,
        use_bias_mlp=use_bias_mlp,
        last_layer_features=last_layer_features,
        use_qnn_layers=True,
        use_amplitude_encoding=True,
        noise=noise,
        diff_mode=diff_mode,
        n_shots=n_shots,
        key=key,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


if __name__ == "__main__":

    # Test configuration
    input_dimension = 513
    n_qubits = 6
    largest_kernel_dimension = n_qubits
    n_var_layers = 8
    n_out = 1
    max_number_conv_layers = 1

    # Get kernel dimensions and outputs per layer
    list_kernel_dimensions, list_outputs_per_conv_layer = compute_kernel_width_per_layer(
        input_dimension=input_dimension,
        largest_kernel_dimension=largest_kernel_dimension,
        max_number_conv_layers=max_number_conv_layers,
    )
    print("Kernel dimensions:", list_kernel_dimensions)
    print("Outputs per layer:", list_outputs_per_conv_layer)

    # Construct model
    list_conv_layers = construct_convolutional_model(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        max_number_conv_layers=max_number_conv_layers,
    )

    # Create test input
    grids = jnp.ones(input_dimension)

    #########################################################
    # Test MLP-based convolutional model (no QNNs)
    #########################################################
    print("\nBuilding conv with only batched global MLP")
    init_fn, apply_fn = build_conv_mlp(
        n_qubits=3,
        n_var_layers=2,
        n_out=1,
        input_dimension=input_dimension,
        largest_kernel_width=20,
    )
    # Note: 150 parameters in total vs 144 for GlobalQNN

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))

    # Count parameters
    param_count = count_parameters(params)
    print("Number of parameters:", param_count)

    # Test with JIT compilation
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)

    # Test each layer individually
    current_input = grids
    for i, (init_fn, apply_fn) in enumerate(list_conv_layers):
        _, params = init_fn(jax.random.PRNGKey(i), (input_dimension,))
        current_input = apply_fn(params, current_input)
        print(f"Layer {i} output shape:", current_input.shape)

    # Test full model using stax
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network
    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))

    # Count parameters
    param_count = count_parameters(params)
    print("Number of parameters:", param_count)

    # Test forward pass
    output = apply_fn(params, grids)
    print("\nFull model output shape:", output.shape)
    print("Output:", output)

    # Test with JIT compilation
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("\nJIT output matches:", jnp.allclose(output, jit_output))

    # Test gradient computation
    grad_fn = jax.jit(jax.grad(lambda p, x: apply_fn(p, x).sum()))
    grad_output = grad_fn(params, grids)
    print(
        "\nGradient shapes:",
        jax.tree_map(
            lambda x: x.shape if hasattr(x, "shape") else None,
            grad_output,
        ),
    )

    # Test standard QNN (no noise)
    print("\nBuilding convolutional QNN via factory function")
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
    )

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    # Test QNN with noise
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    # Test with low noise (0.1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.001),)
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        noise=noise,
    )
    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output with 0.1% noise, shape:", jit_output.shape)
    print("JIT output with 0.1% noise:", jit_output)

    # Test with higher noise (1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),)
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        noise=noise,
    )
    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output with 1% noise, shape:", jit_output.shape)
    print("JIT output with 1% noise:", jit_output)

    # Test amplitude encoding QNN
    init_fn, apply_fn = build_conv_amplitude_encoding_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
    )

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("Amplitude encoding QNN output shape:", jit_output.shape)
    print("Amplitude encoding QNN output:", jit_output)

    # Differentiate the amplitude encoding QNN
    grad_fn = jax.jit(jax.grad(lambda p, x: apply_fn(p, x).sum()))
    grad_output = grad_fn(params, grids)
    print(
        "\nGradient shapes:",
        jax.tree_map(
            lambda x: x.shape if hasattr(x, "shape") else None,
            grad_output,
        ),
    )

    # Performance benchmarking
    print("\nStarting performance benchmarking " + "-" * 30)
    import time

    num_iterations = 10

    # Non-JIT timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = apply_fn(params, grids)
    non_jit_time = time.time() - start_time

    # JIT timing
    start_time = time.time()
    for _ in range(num_iterations):
        _ = jit_apply_fn(params, grids)
    jit_time = time.time() - start_time

    print(f"\nPerformance over {num_iterations} iterations:")
    print(f"Non-JIT time: {non_jit_time:.4f} seconds")
    print(f"JIT time: {jit_time:.4f} seconds")
    print(f"Speedup: {non_jit_time/jit_time:.1f}x")
