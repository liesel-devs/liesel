"""Named Gaussian blocks retain their marginal or conditional interpretation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import liesel.optim as opt
from liesel.optim.types import Position


def gaussian_approximation():
    # For independent standard normals e, the coordinates are
    # (e0-e1+e2-e3, e1-e2+e3, e2-e3, e3). Their covariances follow directly.
    return opt.LaplaceApproximation(
        mean=Position(
            {"z": jnp.array(0.0), "matrix": jnp.zeros((1, 1)), "a": jnp.zeros(2)}
        ),
        precision_cholesky=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ),
        names=("z", "matrix", "a"),
        shapes=((), (1, 1), (2,)),
        valid=True,
        diagnostics={},
    )


def test_marginal_covariance_blocks_include_dependence_on_other_parameters():
    posterior = gaussian_approximation()
    blocks = posterior.marginal_covariance_blocks()
    assert list(blocks) == ["z", "matrix", "a"]
    np.testing.assert_allclose(blocks["z"], [[4.0]])
    np.testing.assert_allclose(blocks["matrix"], [[3.0]])
    np.testing.assert_allclose(blocks["a"], [[2.0, -1.0], [-1.0, 1.0]])

    selected = posterior.marginal_covariance_blocks(["a", "z"])
    assert list(selected) == ["a", "z"]
    np.testing.assert_allclose(selected["a"], [[2.0, -1.0], [-1.0, 1.0]])


def test_marginal_precision_factors_invert_marginal_covariance_blocks():
    posterior = gaussian_approximation()
    factors = posterior.marginal_precision_cholesky_blocks()
    np.testing.assert_allclose(factors["z"], [[0.5]], atol=1e-6)
    np.testing.assert_allclose(factors["matrix"], [[1 / np.sqrt(3)]], atol=1e-6)
    np.testing.assert_allclose(factors["a"], [[1.0, 0.0], [1.0, 1.0]], atol=1e-6)
    assert list(posterior.marginal_precision_cholesky_blocks(["matrix"])) == ["matrix"]


def test_conditional_precision_blocks_include_cross_block_factor_entries():
    posterior = gaussian_approximation()
    blocks = posterior.conditional_precision_blocks()
    assert list(blocks) == ["z", "matrix", "a"]
    np.testing.assert_allclose(blocks["z"], [[1.0]])
    np.testing.assert_allclose(blocks["matrix"], [[2.0]])
    np.testing.assert_allclose(blocks["a"], [[2.0, 1.0], [1.0, 2.0]])
    assert list(posterior.conditional_precision_blocks(["a", "z"])) == ["a", "z"]


@pytest.mark.parametrize(
    "method",
    [
        "marginal_covariance_blocks",
        "marginal_precision_cholesky_blocks",
        "conditional_precision_blocks",
    ],
)
def test_block_selection_rejects_ambiguous_or_unknown_names(method):
    select = getattr(gaussian_approximation(), method)
    assert select([]) == {}
    for keys, message in [
        ("z", "sequence"),
        (["z", "z"], "Duplicate"),
        (["missing"], "Unknown"),
    ]:
        with pytest.raises(ValueError, match=message):
            select(keys)


@pytest.mark.parametrize("x64", [False, True])
def test_blocks_preserve_precision_and_reject_invalid_approximations(x64):
    with jax.enable_x64(x64):
        posterior = gaussian_approximation()
        for method in [
            "marginal_covariance_blocks",
            "marginal_precision_cholesky_blocks",
            "conditional_precision_blocks",
        ]:
            blocks = getattr(posterior, method)()
            assert all(
                block.dtype == posterior.precision_cholesky.dtype
                for block in blocks.values()
            )
            for invalid in [
                replace(posterior, valid=False),
                replace(posterior, precision_cholesky=None),
            ]:
                with pytest.raises(RuntimeError, match="invalid"):
                    getattr(invalid, method)()
