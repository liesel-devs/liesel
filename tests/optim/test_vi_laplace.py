"""Fitted Gaussian initialization through the public VDist builder."""

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


def distribution_of(builder):
    assert builder.var is not None
    assert builder.var.dist_node is not None
    return builder.var.dist_node.init_dist()


def example():
    approximation = opt.LaplaceApproximation(
        mean={
            "z": jnp.array(10.0),
            "matrix": jnp.array([[20.0]]),
            "a": jnp.array([30.0, 40.0]),
        },
        precision_cholesky=jnp.array(
            [[1.0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        ),
        names=("z", "matrix", "a"),
        shapes=((), (1, 1), (2,)),
        valid=True,
        diagnostics={},
    )
    model = lsl.Model(
        [
            lsl.Var.new_param(jnp.zeros(shape), name=name)
            for name, shape in zip(
                approximation.names, approximation.shapes, strict=True
            )
        ]
    )
    return model, approximation


def test_grouped_conditional_initialization_preserves_order_and_inputs(monkeypatch):
    model, approximation = example()
    before = model.extract_position(approximation.names)

    def no_covariance(*args):
        raise AssertionError("Subset initialization must not construct full covariance")

    monkeypatch.setattr(opt.LaplaceApproximation, "covariance", no_covariance)
    builder = opt.VDist(["matrix", "a"], model)
    assert builder.mvn_tril_from_laplace(approximation) is builder
    assert builder.q is None
    builder.build()
    distribution = distribution_of(builder)
    np.testing.assert_allclose(distribution.mean(), [30.0, 40.0, 20.0])
    np.testing.assert_allclose(
        distribution.covariance(),
        [[1.0, -0.5, -0.5], [-0.5, 0.75, 0.25], [-0.5, 0.25, 0.75]],
        atol=2e-6,
    )
    for name, value in before.items():
        np.testing.assert_array_equal(model.vars[name].value, value)
    np.testing.assert_array_equal(approximation.mean["a"], [30.0, 40.0])


def test_full_block_reproduces_joint_covariance_and_sample_layout():
    model, approximation = example()
    vdist = (
        opt.VDist(["z", "matrix", "a"], model)
        .mvn_tril_from_laplace(approximation)
        .build()
    )
    distribution = distribution_of(vdist)
    np.testing.assert_allclose(distribution.mean(), [30.0, 40.0, 20.0, 10.0])
    np.testing.assert_allclose(
        distribution.covariance(),
        [[2, -1, -2, 2], [-1, 1, 1, -1], [-2, 1, 3, -3], [2, -1, -3, 4]],
        atol=3e-6,
    )
    samples = vdist.sample(jax.random.key(42), (2, 3))
    assert {name: value.shape for name, value in samples.items()} == {
        "a": (2, 3, 2),
        "matrix": (2, 3, 1, 1),
        "z": (2, 3),
    }


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"valid": False}, RuntimeError),
        ({"precision_cholesky": None}, RuntimeError),
        ({"names": ("z", "z", "a")}, ValueError),
        ({"shapes": ((), (1, 1))}, ValueError),
        ({"shapes": ((), (1,), (2,))}, ValueError),
        ({"mean": {"z": jnp.array(0.0)}}, ValueError),
        ({"precision_cholesky": jnp.eye(3)}, ValueError),
        ({"precision_cholesky": jnp.eye(4).at[0, 0].set(jnp.nan)}, RuntimeError),
        ({"precision_cholesky": jnp.eye(4).at[0, 1].set(1.0)}, RuntimeError),
        ({"precision_cholesky": jnp.eye(4).at[0, 0].set(0.0)}, RuntimeError),
    ],
)
def test_rejects_unusable_approximation(changes, error):
    model, approximation = example()
    builder = opt.VDist(["a"], model)
    with pytest.raises(error):
        builder.mvn_tril_from_laplace(replace(approximation, **changes))
    assert builder.var is None


def test_validates_type_selected_names_shapes_and_all_means():
    model, approximation = example()
    builder = opt.VDist(["a"], model)
    wrong_type: Any = object()
    with pytest.raises(TypeError, match="LaplaceApproximation"):
        builder.mvn_tril_from_laplace(wrong_type)
    other = lsl.Model(lsl.Var.new_param(0.0, name="other"))
    with pytest.raises(ValueError, match="Unknown"):
        opt.VDist(["other"], other).mvn_tril_from_laplace(approximation)
    scalar_a = lsl.Model(lsl.Var.new_param(0.0, name="a"))
    with pytest.raises(ValueError, match="Target shape"):
        opt.VDist(["a"], scalar_a).mvn_tril_from_laplace(approximation)
    with pytest.raises(RuntimeError, match="means must be finite"):
        builder.mvn_tril_from_laplace(
            replace(approximation, mean={**approximation.mean, "z": jnp.array(jnp.inf)})
        )


@pytest.mark.parametrize("to_float32", [False, True])
def test_inherits_dtype_and_subset_avoids_full_identity(monkeypatch, to_float32):
    with jax.enable_x64():
        model = lsl.Model(
            lsl.Var.new_param(jnp.zeros(2), name="a"), to_float32=to_float32
        )
        _, approximation = example()
        eye = jnp.eye

        def block_eye(size, *args, **kwargs):
            assert size == 2
            return eye(size, *args, **kwargs)

        monkeypatch.setattr(jnp, "eye", block_eye)
        vdist = opt.VDist(["a"], model).mvn_tril_from_laplace(approximation).build()
        distribution = distribution_of(vdist)
        assert distribution.dtype == (jnp.float32 if to_float32 else jnp.float64)
        np.testing.assert_allclose(
            distribution.covariance(), [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], atol=2e-6
        )


def test_laplace_fit_initializes_dense_and_blocked_vi():
    # Completing the square in (z, a0, a1) gives precision
    # [[3,-1,-1],[-1,2,1],[-1,1,2]] and information vector (0,2,2).
    z = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
    loc = lsl.Var.new_calc(lambda value: jnp.repeat(value, 2), z, name="a_loc")
    a = lsl.Var.new_param(
        jnp.zeros(2), lsl.Dist(tfd.MultivariateNormalDiag, loc, jnp.ones(2)), name="a"
    )
    total = lsl.Var.new_calc(jnp.sum, a, name="total")
    y = lsl.Var.new_obs(jnp.array([2.0]), lsl.Dist(tfd.Normal, total, 1.0), name="y")
    model = lsl.Model(y)
    split = opt.PositionSplit.from_model(model)
    loss = opt.LaplaceLoss(model, split, latent=["a"])
    result = opt.LieselOptim(
        model,
        loss=loss,
        optimizers="lbfgs",
        loss_monitor="train_full_data",
        show_progress=False,
        stopper=opt.Stopper(epochs=20, patience=3),
    ).fit()
    approximation = loss.approximate_joint_posterior(result)
    assert approximation.names == ("z", "a")
    dense = opt.VDist(["z", "a"], model).mvn_tril_from_laplace(approximation).build()
    np.testing.assert_allclose(
        distribution_of(dense).mean(), [6 / 7, 6 / 7, 4 / 7], atol=2e-5
    )
    np.testing.assert_allclose(
        distribution_of(dense).covariance(),
        np.array([[5, -2, 1], [-2, 5, 1], [1, 1, 3]]) / 7,
        atol=2e-5,
    )
    blocks = [
        opt.VDist([name], model).mvn_tril_from_laplace(approximation)
        for name in ["a", "z"]
    ]
    np.testing.assert_allclose(
        distribution_of(blocks[0]).covariance(),
        [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]],
        atol=2e-5,
    )
    np.testing.assert_allclose(
        distribution_of(blocks[1]).covariance(), [[1 / 3]], atol=2e-5
    )
    blocked = opt.CompositeVDist(*blocks).build()
    for family in [dense, blocked]:
        vi_loss = opt.NegElboLoss.from_vdist(family, nsamples=8)
        vi_result = opt.LieselVI(
            model,
            loss=vi_loss,
            optimizers=optax.adam(0.001),
            loss_monitor="train_full_data",
            show_progress=False,
            stopper=opt.Stopper(epochs=3, patience=3),
        ).fit()
        draws = vi_loss.approximate_joint_posterior(vi_result).sample(
            10, seed=jax.random.key(42)
        )
        assert all(np.isfinite(value).all() for value in draws.values())
        assert draws["a"].shape == (10, 2)
        assert draws["z"].shape == (10,)


def test_rejects_initialization_outside_default_bijector_domain():
    model = lsl.Model(lsl.Var.new_param(0.0, name="z"))
    approximation = opt.LaplaceApproximation(
        mean={"z": jnp.array(0.0)},
        precision_cholesky=jnp.array([[1e9]]),
        names=("z",),
        shapes=((),),
        valid=True,
        diagnostics={},
    )
    with pytest.raises(RuntimeError, match="represent"):
        opt.VDist(["z"], model).mvn_tril_from_laplace(approximation)
