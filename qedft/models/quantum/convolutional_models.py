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
from qedft.models.wrappers import negativity_transform


def add_gaussian_noise_layer(network: tuple[Callable, Callable], noise_std: float = 0.1):
    """
    Wraps a network with a Gaussian noise layer.

    Args:
        network: A tuple of (init_fn, apply_fn)
        noise_std: Standard deviation of the Gaussian noise

    Returns:
        A tuple of (init_fn, apply_fn) with noise added
    """

    init_fn, apply_fn = network

    def noisy_apply_fn(params, inputs, rng_key=None, **kwargs):
        outputs = apply_fn(params, inputs, **kwargs)
        rng = kwargs.get('rng', jax.random.PRNGKey(0))
        if rng is not None and noise_std > 0:
            noise = jax.random.normal(rng, shape=outputs.shape) * noise_std
            return jnp.array(outputs + noise).reshape((-1,))
        # return jnp.array([outputs,])
        return jnp.array(outputs).reshape((-1,))

    return init_fn, noisy_apply_fn


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
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
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

                def apply_fn_noisy(params, inputs, rng_key=None, **kwargs):
                    output = apply_fn(params, inputs, **kwargs)
                    rng_key = kwargs.get('rng', jax.random.PRNGKey(0))
                    rng1, rng2 = jax.random.split(rng_key)
                    if add_gaussian_noise_to_qnn_output:
                        if gaussian_noise_std > 0:
                            noise = jax.random.normal(rng1, shape=output.shape) * gaussian_noise_std
                            output = output + noise
                    return output

                # Add gaussian noise to the output of QNN in addition to the gate noise
                if add_gaussian_noise_to_qnn_output:
                    logger.info(f"Adding gaussian noise to the output of QNN with std {gaussian_noise_std}")
                    apply_fn = apply_fn_noisy


                list_conv_layers.append((init_fn, apply_fn))
            else:
                # For the last layer, we might want different configuration
                qnn_config["wrap_with_negative_transform"] = True  # Equivalent to "swish_and_norm"
                qnn_config["n_features"] = kernel_width
                qnn = QNNLayer(config_dict=qnn_config)
                init_fn, apply_fn = qnn.build_network(grids, noise=noise)

                # Add gaussian noise to the output of QNN in addition to the gate noise
                def apply_fn_noisy(params, inputs, rng_key=None, **kwargs):
                    output = apply_fn(params, inputs, **kwargs)
                    rng_key = kwargs.get('rng', jax.random.PRNGKey(0))
                    rng1, rng2 = jax.random.split(rng_key)
                    if add_gaussian_noise_to_qnn_output:
                        if gaussian_noise_std > 0:
                            noise = jax.random.normal(rng1, shape=output.shape) * gaussian_noise_std
                            output = output + noise
                    return output

                # Add gaussian noise to the output of QNN in addition to the gate noise
                if add_gaussian_noise_to_qnn_output:
                    logger.info(f"Adding gaussian noise to the output of QNN with std {gaussian_noise_std}")
                    apply_fn = apply_fn_noisy

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

            # Add gaussian noise to the output of QNN in addition to the gate noise
            def apply_fn_noisy(params, inputs, rng_key=None, **kwargs):
                output = apply_fn(params, inputs, **kwargs)
                rng_key = kwargs.get('rng', jax.random.PRNGKey(0))
                rng1, rng2 = jax.random.split(rng_key)
                if add_gaussian_noise_to_qnn_output:
                    if gaussian_noise_std > 0:
                        noise = jax.random.normal(rng1, shape=output.shape) * gaussian_noise_std
                        output = output + noise
                return output

            # Add gaussian noise to the output of QNN in addition to the gate noise
            if add_gaussian_noise_to_qnn_output:
                logger.info(f"Adding gaussian noise to the output of QNN with std {gaussian_noise_std}")
                apply_fn = apply_fn_noisy

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


def construct_convolutional_model_reverse(
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
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
):
    """
    Construct a convolutional model with a given input dimension and largest kernel width
    This is the reverse of the construct_convolutional_model function

    So first classical layers, and the last is a single QNN layer.

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

        # BatchedGlobalMLP if you want to replace by a simple MLP
        logger.info(
            f"BatchedGlobalMLP Layer {i}: n_qubits_layer\neurons {n_qubits_layer}, kernel_width {kernel_width}",
        )

        # n_qubits_layer is the number of neurons in the layer
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

    # Add a final MLP layer that outputs n_qubits values
    if list_outputs_per_conv_layer[-1] > 1:
        logger.info(f"Adding final MLP layer to output {n_qubits} values for QNN input")

        # Create a dense layer that outputs n_qubits values
        def init_final_mlp(rng, input_shape):
            k1, k2 = jax.random.split(rng)
            output_size = n_qubits
            input_size = list_outputs_per_conv_layer[-1]  # Use the known output size from previous layer
            w = jax.random.normal(k1, (input_size, output_size)) * 0.1
            b = None
            if use_bias_mlp:
                b = jax.random.normal(k2, (output_size,)) * 0.1
            return (output_size,), (w, b)

        def apply_final_mlp(params, inputs, **kwargs):
            w, b = params
            # Ensure inputs have the right shape for matrix multiplication
            inputs_reshaped = inputs.reshape(-1, w.shape[0])
            out = jnp.dot(inputs_reshaped, w)
            if b is not None:
                out = out + b
            # Apply activation function (tanh to match the pattern)
            out = jnp.tanh(out)
            return out.reshape(-1)  # Flatten output

        list_conv_layers.append((init_final_mlp, apply_final_mlp))

    # Add the final QNN layer that takes n_qubits inputs and outputs 1 value
    logger.info(f"Adding final QNN layer with {n_qubits} qubits")

    # Create GlobalQNN configuration for the final layer
    qnn_config = {
        "n_qubits": n_qubits,
        "n_var_layers": n_var_layers,
        "normalization": normalization,
        "use_amplitude_encoding": use_amplitude_encoding,
        "n_features": n_qubits,  # Input features should match n_qubits
        "diff_mode": diff_mode,
        "n_shots": n_shots,
        "key": key,
        "wrap_with_negative_transform": True,  # Ensure single output
    }

    # Create GlobalQNN instance and get its build_network function
    qnn = QNNLayer(config_dict=qnn_config)
    init_fn, apply_fn = qnn.build_network(grids, noise=noise)

    # Add gaussian noise to the output of QNN in addition to the gate noise
    if add_gaussian_noise_to_qnn_output:
        init_fn, apply_fn = add_gaussian_noise_layer((init_fn, apply_fn), gaussian_noise_std)

    list_conv_layers.append((init_fn, apply_fn))

    return list_conv_layers


def construct_convolutional_model_classical_to_quantum(
    n_qubits,
    n_var_layers,
    n_out,
    input_dimension,
    normalization: float = 1.0,
    use_bias_mlp: bool = False,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    use_amplitude_encoding: bool = False,
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
):

    list_conv_layers = []

    # ====================
    # Classical encoder
    # ====================

    # NOTE: I think we want to avoid making classical too good here.
    # logger.info(
        # f"BatchedGlobalMLP Layer {i}: neurons {n_neurons}",
    # )
    # # n_qubits_layer is the number of neurons in the layer
    # bmlp_config = {
    #     "n_neurons": n_neurons,
    #     "n_layers": n_var_layers,
    #     "use_bias": use_bias_mlp,
    #     "density_normalization_factor": normalization,
    #     "n_features": input_dimension,
    #     "activation": "tanh",
    #     "n_outputs": n_qubits,
    # }
    # bmlp = BatchedGlobalMLP(bmlp_config)
    # # Create mock grids for GlobalQNN initialization
    # grids = jnp.ones(input_dimension)
    # init_fn, apply_fn = bmlp.build_network(grids)
    # list_conv_layers.append((init_fn, apply_fn))

    # Final layer to output n_qubits values
    mlp_layers = []

    # Create mock grids for GlobalQNN initialization
    grids = jnp.ones(input_dimension)
    logger.info(f"Using MLP layer from {input_dimension} inputs to {n_qubits} outputs")
    mlp_layers.append(stax.Dense(n_qubits))
    # Create the MLP network
    mlp_network = stax.serial(*mlp_layers)

    def mlp_init_fn(rng, input_shape):
        # input_shape has to be provided to work in stax
        del input_shape
        return mlp_network[0](rng, input_shape=(input_dimension,))

    def mlp_apply_fn(params, inputs, **kwargs):
        del kwargs
        normalized_inputs = inputs / normalization
        return mlp_network[1](params, normalized_inputs)

    list_conv_layers.append((mlp_init_fn, mlp_apply_fn))

    # ====================
    # Quantum decoder
    # ====================

    # Select the QNN layer type
    QNNLayer = AmplitudeEncodingGlobalQNNLayer if use_amplitude_encoding else GlobalQNNLayer
    logger.info(f"Using {QNNLayer.__name__} layer")
    # Add the final QNN layer that takes n_qubits inputs and outputs 1 value
    logger.info(f"Layer with {n_qubits} qubits")
    # Create GlobalQNN configuration for the final layer
    qnn_config = {
        "n_qubits": n_qubits,
        "n_var_layers": n_var_layers,
        "normalization": normalization,
        "use_amplitude_encoding": False,  # We use the angle encoding
        "n_features": n_qubits,  # Input features should match n_qubits
        "diff_mode": diff_mode,
        "n_shots": n_shots,
        "key": key,
        "wrap_with_negative_transform": False,  # Use at the end, more reasonable after the noise
        "layer_type": "DirectQNN",  # simplest feature map
    }
    # Create GlobalQNN instance and get its build_network function
    qnn = QNNLayer(config_dict=qnn_config)
    init_fn, apply_fn = qnn.build_network(grids, noise=noise)

    # Emulates sampling noise
    # Add gaussian noise to the output of QNN in addition to the gate noise
    if add_gaussian_noise_to_qnn_output:
        init_fn, apply_fn = add_gaussian_noise_layer((init_fn, apply_fn), gaussian_noise_std)

    # Wrap the output with a negativity transform to ensure the output is negative
    # May lessen the effect of noise on the output
    logger.info("Wrapping the output with a negativity transform")
    init_fn, apply_fn = stax.serial((init_fn, apply_fn), negativity_transform())

    list_conv_layers.append((init_fn, apply_fn))

    return list_conv_layers


def construct_convolutional_model_quantum_to_classical(
    n_qubits,
    n_var_layers,
    n_out,
    input_dimension,
    normalization: float = 1.0,
    use_bias_mlp: bool = False,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    use_amplitude_encoding: bool = False,
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
    n_features: int = 3,
):

    list_conv_layers = []

    # Final layer to output n_qubits values
    mlp_layers = []
    grids = jnp.ones(input_dimension)

    # ====================
    # Quantum decoder
    # ====================

    # Select the QNN layer type
    QNNLayer = AmplitudeEncodingGlobalQNNLayer if use_amplitude_encoding else GlobalQNNLayer
    logger.info(f"Using {QNNLayer.__name__} layer")
    # Add the final QNN layer that takes n_qubits inputs and outputs 1 value
    logger.info(f"Layer with {n_qubits} qubits")
    # Create GlobalQNN configuration for the final layer
    qnn_config = {
        "n_qubits": n_qubits,
        "n_var_layers": n_var_layers,
        "normalization": normalization,
        "use_amplitude_encoding": False,  # We use the angle encoding
        "n_features": n_features,  # Input features should match n_qubits
        "diff_mode": diff_mode,
        "n_shots": n_shots,
        "key": key,
        "wrap_with_negative_transform": False,  # Use at the end, more reasonable after the noise
        "layer_type": "DirectQNN",  # simplest feature map
    }
    # Create GlobalQNN instance and get its build_network function
    qnn = QNNLayer(config_dict=qnn_config)
    init_fn, apply_fn = qnn.build_network(grids, noise=noise)

    # Emulates sampling noise
    # Add gaussian noise to the output of QNN in addition to the gate noise
    if add_gaussian_noise_to_qnn_output:
        init_fn, apply_fn = add_gaussian_noise_layer((init_fn, apply_fn), gaussian_noise_std)

    list_conv_layers.append((init_fn, apply_fn))

    # ====================
    # Classical encoder
    # ====================

    # Create mock grids for GlobalQNN initialization
    logger.info(f"Using MLP layer from {input_dimension // n_qubits} inputs to {n_features} outputs")
    mlp_layers.append(stax.Dense(n_features))
    # Create the MLP network
    mlp_network = stax.serial(*mlp_layers)

    def mlp_init_fn(rng, input_shape):
        # input_shape has to be provided to work in stax
        del input_shape
        # ? Changed to intput // n_features to match the input dimension of the QNN
        return mlp_network[0](rng, input_shape=(input_dimension // n_features,))

    def mlp_apply_fn(params, inputs, **kwargs):
        del kwargs
        normalized_inputs = inputs / normalization
        return mlp_network[1](params, normalized_inputs)

    list_conv_layers.append((mlp_init_fn, mlp_apply_fn))

    # Wrap the output with a negativity transform to ensure the output is negative
    # May lessen the effect of noise on the output
    logger.info("Wrapping the output with a negativity transform")
    init_fn, apply_fn = stax.serial((init_fn, apply_fn), negativity_transform())

    list_conv_layers.append((init_fn, apply_fn))

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
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
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
        use_amplitude_encoding=False,  # NOTE: always use angle encoding for these models
        add_gaussian_noise_to_qnn_output=add_gaussian_noise_to_qnn_output,
        gaussian_noise_std=gaussian_noise_std,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


def build_conv_qnn_reverse(
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
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
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
    list_conv_layers = construct_convolutional_model_reverse(
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
        use_amplitude_encoding=False,  # NOTE: always use angle encoding for these models
        add_gaussian_noise_to_qnn_output=add_gaussian_noise_to_qnn_output,
        gaussian_noise_std=gaussian_noise_std,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


def build_conv_qnn_quantum_to_classical(
    n_qubits: int,
    n_var_layers: int,
    n_out: int,
    input_dimension: int,
    normalization: float = 1.0,
    use_bias_mlp: bool = False,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
    n_features: int = 3,
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
    list_conv_layers = construct_convolutional_model_quantum_to_classical(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        normalization=normalization,
        use_bias_mlp=use_bias_mlp,
        noise=noise,
        diff_mode=diff_mode,
        n_shots=n_shots,
        key=key,
        use_amplitude_encoding=False,  # NOTE: always use angle encoding
        add_gaussian_noise_to_qnn_output=add_gaussian_noise_to_qnn_output,
        gaussian_noise_std=gaussian_noise_std,
        n_features=n_features,
    )
    network = stax.serial(*list_conv_layers)
    init_fn, apply_fn = network

    return init_fn, apply_fn


def build_conv_qnn_classical_to_quantum(
    n_qubits: int,
    n_var_layers: int,
    n_out: int,
    input_dimension: int,
    normalization: float = 1.0,
    use_bias_mlp: bool = False,
    noise: NoiseProtocol | None = None,
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int = 0,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    add_gaussian_noise_to_qnn_output: bool = False,
    gaussian_noise_std: float = 0.5,
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
    list_conv_layers = construct_convolutional_model_classical_to_quantum(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        normalization=normalization,
        use_bias_mlp=use_bias_mlp,
        noise=noise,
        diff_mode=diff_mode,
        n_shots=n_shots,
        key=key,
        use_amplitude_encoding=False,  # NOTE: always use angle encoding
        add_gaussian_noise_to_qnn_output=add_gaussian_noise_to_qnn_output,
        gaussian_noise_std=gaussian_noise_std,
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

    def print_section(title):
        print("\n" + "=" * 60)
        print(f"= {title}")
        print("=" * 60)

    def print_subsection(title):
        print("\n" + "-" * 60)
        print(f"-- {title}")
        print("-" * 60)

    # Test configuration
    print_section("Test Configuration")
    input_dimension = 513
    n_qubits = 6
    largest_kernel_dimension = n_qubits
    n_var_layers = 8
    n_out = 1
    max_number_conv_layers = 1

    # Get kernel dimensions and outputs per layer
    print_subsection("Kernel Dimensions and Outputs per Layer")
    list_kernel_dimensions, list_outputs_per_conv_layer = compute_kernel_width_per_layer(
        input_dimension=input_dimension,
        largest_kernel_dimension=largest_kernel_dimension,
        max_number_conv_layers=max_number_conv_layers,
    )
    print("Kernel dimensions:", list_kernel_dimensions)
    print("Outputs per layer:", list_outputs_per_conv_layer)

    # Construct model
    print_subsection("Constructing Convolutional Model")
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

    print_section("MLP-based Convolutional Model (No QNNs)")
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
    print_subsection("Testing Each Layer Individually")
    current_input = grids
    for i, (init_fn, apply_fn) in enumerate(list_conv_layers):
        _, params = init_fn(jax.random.PRNGKey(i), (input_dimension,))
        current_input = apply_fn(params, current_input)
        print(f"Layer {i} output shape:", current_input.shape)

    # Test full model using stax
    print_subsection("Testing Full Model Using stax")
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
    print_subsection("Testing Gradient Computation")
    grad_fn = jax.jit(jax.grad(lambda p, x: apply_fn(p, x).sum()))
    grad_output = grad_fn(params, grids)
    print(
        "\nGradient shapes:",
        jax.tree_map(
            lambda x: x.shape if hasattr(x, "shape") else None,
            grad_output,
        ),
    )

    print_section("Quantum to Classical QNN with Gaussian Noise")
    # Test QNN with noise
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
    # Test with low noise (0.1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),
            DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, 0.1))
    init_fn, apply_fn = build_conv_qnn_quantum_to_classical(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        diff_mode=DiffMode.AD,
        add_gaussian_noise_to_qnn_output=True,
        gaussian_noise_std=0.5,
        noise=noise,
    )
    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    print_section("Classical to Quantum QNN with Gaussian Noise")
    # Test QNN with noise
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
    # Test with low noise (0.1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),
            DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, 0.1))
    init_fn, apply_fn = build_conv_qnn_classical_to_quantum(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        diff_mode=DiffMode.AD,
        add_gaussian_noise_to_qnn_output=True,
        gaussian_noise_std=0.5,
        noise=noise,
    )
    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    print_section("Reverse Convolutional QNN with Gaussian Noise")
    # Test QNN with noise
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
    # Test with low noise (0.1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),
            DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, 0.1))
    init_fn, apply_fn = build_conv_qnn_reverse(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        add_gaussian_noise_to_qnn_output=True,
        gaussian_noise_std=0.5,
        noise=noise,
    )

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    print_section("Standard Convolutional QNN (No Noise)")
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

    print_section("Standard Convolutional QNN with Gaussian Noise")
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        add_gaussian_noise_to_qnn_output=True,
        gaussian_noise_std=0.5,
    )

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    print_section("Standard Convolutional QNN with Bitflip Noise (0.1%)")
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

    print_section("Standard Convolutional QNN with Bitflip Noise (1%)")
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

    print_section("Standard Convolutional QNN with Gaussian, Bitflip, and Amplitude Damping Noise")
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
    # Test with low noise (0.1%)
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.1),
            DigitalNoiseInstance(DigitalNoiseType.AMPLITUDE_DAMPING, 0.1))
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=n_qubits,
        n_var_layers=n_var_layers,
        n_out=n_out,
        input_dimension=input_dimension,
        largest_kernel_width=largest_kernel_dimension,
        add_gaussian_noise_to_qnn_output=True,
        gaussian_noise_std=0.5,
        noise=noise,
    )

    _, params = init_fn(jax.random.PRNGKey(0), (input_dimension,))
    jit_apply_fn = jax.jit(apply_fn)
    jit_output = jit_apply_fn(params, grids)
    print("JIT output shape:", jit_output.shape)
    print("JIT output:", jit_output)

    print_section("Amplitude Encoding QNN")
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

    print_subsection("Differentiating the Amplitude Encoding QNN")
    grad_fn = jax.jit(jax.grad(lambda p, x: apply_fn(p, x).sum()))
    grad_output = grad_fn(params, grids)
    print(
        "\nGradient shapes:",
        jax.tree_map(
            lambda x: x.shape if hasattr(x, "shape") else None,
            grad_output,
        ),
    )

    print_section("Performance Benchmarking")
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
