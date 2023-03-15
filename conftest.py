"""
This conftest.py provides default imports for doctests. For the main conftest.py,
see tests/conftest.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import liesel.goose as gs
import liesel.model as lsl


@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["jax"] = jax
    doctest_namespace["jnp"] = jnp
    doctest_namespace["gs"] = gs
    doctest_namespace["lsl"] = lsl
