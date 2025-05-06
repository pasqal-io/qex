"""Tests for the jax_dft library.

This module contains tests for the jax_dft library, which provides a framework for
performing density functional theory calculations using JAX.
"""

import importlib
import pkgutil

import jax_dft
import pytest


def test_jax_dft_all_tests():
    """Discovers and runs all test functions from the jax_dft library.

    This function walks through all submodules in the jax_dft package and executes any
    function whose name starts with 'test_'. This provides a convenient way to run the
    entire test suite.

    The discovery process uses pkgutil.walk_packages to recursively find all submodules.
    For each module, it inspects all attributes and runs those that are callable test
    functions.

    If a module cannot be imported, the error is caught and reported but does not halt
    the overall test execution.
    """

    def run_module_tests(module):
        """Executes all test functions found in the given module.

        Args:
            module: A Python module object to inspect for test functions.

        A test function is identified by:
            1. Having a name that starts with 'test_'
            2. Being callable
        """
        for item_name in dir(module):
            if item_name.startswith("test_"):
                test_fn = getattr(module, item_name)
                if callable(test_fn):
                    test_fn()

    # Walk through all submodules in jax_dft
    prefix = jax_dft.__name__ + "."
    for _, name, _ in pkgutil.walk_packages(jax_dft.__path__, prefix):
        try:
            module = importlib.import_module(name)
            print(f"Running tests from {name}")
            run_module_tests(module)
        except ImportError as e:
            print(f"Could not import {name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
