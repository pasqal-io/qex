"""Setup the JAX environment."""

import jax
from loguru import logger


def setup_jax_environment(force_cpu=False):
    """Set up JAX environment with double precision and GPU if available. Configures JAX to use 64-bit precision and attempts to enable GPU if available.
    If force_cpu is True, then the GPU is not used.
    """
    # Enable double precision
    jax.config.update("jax_enable_x64", True)
    logger.info(
        "Double precision enabled, necessary for accurate calculations, NaNs otherwise can occur during training.",
    )

    # Check if GPU is available
    def jax_has_gpu():
        """Test that JAX has GPU and uses it.

        Returns:
            bool: True if JAX can use GPU, Fa-lse otherwise
        """
        try:
            _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
            return True
        except (RuntimeError, IndexError):  # Be explicit about expected exceptions
            return False

    # Try to enable GPU if available
    has_gpu = jax_has_gpu()
    if has_gpu and not force_cpu:
        jax.config.update("jax_platform_name", "gpu")
        logger.info("GPU detected and enabled")
    else:
        jax.config.update("jax_platform_name", "cpu")
        logger.info("No GPU found, using CPU")

    # Verify GPU status
    logger.info(f"JAX using GPU: {jax.default_backend() == 'gpu'}")
    logger.info(f"Available devices: {jax.devices()}")


if __name__ == "__main__":
    setup_jax_environment(force_cpu=False)
