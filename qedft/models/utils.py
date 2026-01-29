"""
Utility functions for the models.
"""

import jax
import jax.tree_util


def count_parameters(params):
    """Count the number of parameters in the model.
    Args:
        params: Model parameters from stax
    Returns:
        Number of parameters in the model
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params) if hasattr(x, "size"))


if __name__ == "__main__":
    from jax.example_libraries import stax

    # Example of a MLP with 2 layers and 10 neurons in each layer
    network = stax.serial(
        stax.Dense(10),
        stax.Dense(10),
    )
    init_fn, apply_fn = network
    shape, params = init_fn(rng=jax.random.PRNGKey(0), input_shape=(1, 10))
    print(count_parameters(params))

    # Example of a QNN with 6 qubits, 8 convolutional layers, and 1 output for the paper
    from qedft.models.quantum.convolutional_models import (
        construct_convolutional_model_classical_to_quantum,
        construct_convolutional_model_quantum_to_classical,
    )

    n_qubits = 6
    n_var_layers = 8
    n_out = 1
    input_dimension = 513
    normalization = 1.0
    use_bias_mlp = False
    noise = None

    list_conv_layers_c2q = construct_convolutional_model_classical_to_quantum(
        n_qubits,
        n_var_layers,
        n_out,
        input_dimension,
        normalization,
        use_bias_mlp,
        noise,
    )

    list_conv_layers_q2c = construct_convolutional_model_quantum_to_classical(
        n_qubits,
        n_var_layers,
        n_out,
        input_dimension,
        normalization,
        use_bias_mlp,
        noise,
    )

    network = stax.serial(*list_conv_layers_c2q)
    init_fn, apply_fn = network
    shape, params = init_fn(rng=jax.random.PRNGKey(0), input_shape=(1, input_dimension))
    print("Number of parameters in the classical-to-quantum QNN: ", count_parameters(params))

    network = stax.serial(*list_conv_layers_q2c)
    init_fn, apply_fn = network
    shape, params = init_fn(rng=jax.random.PRNGKey(0), input_shape=(1, input_dimension))
    print("Number of parameters in the quantum-to-classical QNN: ", count_parameters(params))
