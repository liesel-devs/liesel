"""Laplace objectives through public model, loss, and fitting interfaces."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.optimize import brentq, minimize
from scipy.special import gammaln

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.state import OptimCarry


def density_model(joint, **coordinates):
    variables = [
        lsl.Var.new_param(value, name=name) for name, value in coordinates.items()
    ]
    data = lsl.Var.new_obs(jnp.zeros(1), name="data")
    builder = lsl.GraphBuilder().add(*variables, data)
    builder.log_prob_node = lsl.Calc(lambda *args: -joint(*args), *variables)
    model = builder.build_model(validate_log_prob_decomposition=False)
    split = opt.PositionSplit.from_model(
        model, position_keys=[["data"]], infer_sample_sizes=False, shuffle=False
    )
    return model, split


def loss_carry(loss, keys):
    position = loss.position(keys)
    carry = OptimCarry.new(
        key=jax.random.key(1),
        epochs=1,
        position=position,
        batches=opt.Batches.from_split(loss.split, batch_size=None),
        optimizers=[],
        model_state=loss.model.state,
        save_position_history=False,
    )
    carry.loss_state = loss.init_state(position, carry)
    return position, carry


def test_scalar_gaussian_has_normalized_value_gradient_and_conditional_state():
    # theta ~ N(0, 1), z | theta ~ N(theta, 1), so theta's marginal is N(0, 1).
    model, split = density_model(
        lambda theta, z: (
            0.5 * theta**2 + 0.5 * (z - theta) ** 2 + math.log(2 * math.pi)
        ),
        theta=jnp.array(0.7),
        z=jnp.array(-2.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    position, carry = loss_carry(loss, ["theta"])
    assert int(carry.loss_state.status) == 0
    (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
    np.testing.assert_allclose(
        value, 0.5 * 0.7**2 + 0.5 * math.log(2 * math.pi), atol=2e-6
    )
    np.testing.assert_allclose(gradient["theta"], 0.7, atol=2e-6)
    np.testing.assert_allclose(state.latent_position["z"], 0.7, atol=2e-6)
    np.testing.assert_allclose(state.latent_precision_cholesky, [[1.0]], atol=2e-6)
    assert int(state.status) == 1
    assert state.latent_names == ("z",)
    assert state.latent_shapes == ((),)
    assert float(state.outer_position["theta"]) == float(position["theta"])
    assert float(carry.loss_state.latent_position["z"]) == -2.0


@pytest.mark.parametrize("x64", [False, True])
def test_nonlinear_mode_and_gradient_include_log_determinant_dependence(x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        model, split = density_model(
            lambda t, z: t**2 / 2 - t / 3 + z**4 / 4 - t * z,
            theta=jnp.array(1.728, dtype),
            z=jnp.array(1.0, dtype),
        )
        loss = opt.LaplaceLoss(
            model, split, latent=["z"], inner_tol=1e-14 if x64 else 1e-6
        )
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
        # Mode t**(1/3) = 1.2; this is a closed-form marginal Laplace objective.
        expected = (
            1.728**2 / 2
            - 1.728 / 3
            - 3 * 1.2**4 / 4
            + math.log(3 * 1.2**2) / 2
            - math.log(2 * math.pi) / 2
        )
        expected_grad = 1.728 - 1 / 3 - 1.2 + 1 / (3 * 1.728)
        accuracy = 1e-7 if x64 else 2e-4
        np.testing.assert_allclose(value, expected, atol=accuracy, rtol=0)
        np.testing.assert_allclose(
            gradient["theta"], expected_grad, atol=accuracy, rtol=0
        )
        np.testing.assert_allclose(
            state.latent_position["z"], 1.2, atol=accuracy, rtol=0
        )
        assert int(state.status) == 1
        assert float(state.newton_decrement_squared) / 2 <= loss.inner_tol


def test_coupled_scalar_vector_matrix_latents_have_ordered_curvature():
    def joint(theta, a, b, c):
        residual = jnp.concatenate([a[None], b, c.ravel()]) - theta * jnp.array(
            [1.0, 2.0, -1.0, 0.5]
        )
        return (
            theta**2 / 2
            + residual @ residual / 2
            + jnp.sum(residual) ** 2 / 4
            + 2.5 * math.log(2 * math.pi)
            - 0.5 * math.log(3.0)
        )

    model, split = density_model(
        joint,
        theta=jnp.array(0.7),
        a=jnp.array(2.0),
        b=jnp.array([-2.0, 1.0]),
        c=jnp.array([[3.0]]),
    )
    loss = opt.LaplaceLoss(model, split, latent=["c", "a", "b"])
    position, carry = loss_carry(loss, ["theta"])
    (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
    np.testing.assert_allclose(
        value, 0.7**2 / 2 + 0.5 * math.log(2 * math.pi), atol=2e-6
    )
    np.testing.assert_allclose(gradient["theta"], 0.7, atol=2e-6)
    assert state.latent_names == ("a", "b", "c")
    assert state.latent_shapes == ((), (2,), (1, 1))
    np.testing.assert_allclose(state.latent_position["a"], 0.7, atol=2e-6)
    np.testing.assert_allclose(state.latent_position["b"], [1.4, -0.7], atol=2e-6)
    np.testing.assert_allclose(state.latent_position["c"], [[0.35]], atol=2e-6)
    factor = state.latent_precision_cholesky
    np.testing.assert_allclose(
        factor @ factor.T,
        [
            [1.5, 0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5, 0.5],
            [0.5, 0.5, 1.5, 0.5],
            [0.5, 0.5, 0.5, 1.5],
        ],
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "selection",
    [[], ["missing"], ["z", "z"], ["z", "z_value"], ["computed"], ["count"]],
)
def test_latent_selection_rejects_non_coordinates_and_aliases(selection):
    theta = lsl.Var.new_param(jnp.array(0.0), name="theta")
    z = lsl.Var.new_param(jnp.array(1.0), name="z")
    computed = lsl.Var.new_calc(lambda x: 2 * x, z, name="computed")
    count = lsl.Var.new_param(jnp.array(1, jnp.int32), name="count")
    data = lsl.Var.new_obs(jnp.zeros(1), name="data")
    builder = lsl.GraphBuilder().add(theta, z, computed, count, data)
    model = builder.build_model()
    split = opt.PositionSplit.from_model(
        model, position_keys=[["data"]], infer_sample_sizes=False
    )
    with pytest.raises(ValueError, match="latent|coordinate"):
        opt.LaplaceLoss(model, split, latent=selection)


@pytest.mark.parametrize(
    "keys", [["z"], ["theta", "z_value"], ["theta", "theta_value"]]
)
def test_outer_selection_rejects_latent_overlap_and_duplicate_aliases(keys):
    model, split = density_model(
        lambda theta, z: theta**2 + z**2, theta=jnp.array(0.0), z=jnp.array(1.0)
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    with pytest.raises(ValueError, match="coordinate|overlap"):
        loss.position(keys)


@pytest.mark.parametrize("case", ["nonconcave", "overflow_trial"])
@pytest.mark.parametrize("x64", [False, True])
def test_safeguarded_solve_recovers_from_bad_curvature_and_nonfinite_trials(case, x64):
    with jax.enable_x64(x64):
        if case == "nonconcave":

            def joint(t, z):
                return (z * z - 1) ** 2 / 4 - 0.2 * z + t * t / 2

            seed, expected = 0.1, 1.08803391469129
        else:

            def joint(t, z):
                return jnp.exp(z) - 2 * z + t * t / 2

            seed, expected = -5.0, math.log(2.0)
        model, split = density_model(joint, theta=jnp.array(0.0), z=jnp.array(seed))
        loss = opt.LaplaceLoss(model, split, latent=["z"])
        position, carry = loss_carry(loss, ["theta"])
        value, state = jax.jit(loss.loss_train)(position, carry)
        assert int(state.status) == 1
        assert jnp.isfinite(value)
        np.testing.assert_allclose(
            state.latent_position["z"], expected, rtol=0, atol=0.002
        )
        assert float(state.newton_decrement_squared) / 2 <= loss.inner_tol


@pytest.mark.parametrize("x64", [False, True])
def test_inner_solver_fits_normal_location_and_scale_from_zero(x64):
    with jax.enable_x64(x64):
        mu = lsl.Var.new_param(0.0, name="mu")
        log_sd = lsl.Var.new_param(0.0, name="log_sd")
        sd = lsl.Var.new_calc(jnp.exp, log_sd)
        theta = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="theta")
        y = lsl.Var.new_obs(
            jnp.array([9.0, 11.0]), lsl.Dist(tfd.Normal, mu, sd), name="y"
        )
        model = lsl.Model([y, theta], to_float32=not x64)
        loss = opt.LaplaceLoss(model, latent=["mu", "log_sd"])
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)

        # The zero start has indefinite curvature. The Normal MLE is (10, 1),
        # with Hessian diag(4, 2) in (log_sd, mu) order.
        assert int(state.status) == 1
        accuracy = 2e-5 if x64 else 5e-3
        np.testing.assert_allclose(state.latent_position["mu"], 10.0, atol=accuracy)
        np.testing.assert_allclose(state.latent_position["log_sd"], 0.0, atol=accuracy)
        factor = state.latent_precision_cholesky
        np.testing.assert_allclose(
            factor @ factor.T, [[4.0, 0.0], [0.0, 2.0]], atol=accuracy
        )
        np.testing.assert_allclose(
            value, 1 + math.log(2 * math.pi) / 2 + math.log(8) / 2, atol=accuracy
        )
        np.testing.assert_allclose(gradient["theta"], 0.0, atol=accuracy)


@pytest.mark.parametrize(
    "case,status", [("saddle", 5), ("nonfinite", 4), ("budget", 2), ("backtracking", 3)]
)
@pytest.mark.parametrize("x64", [False, True])
def test_unsuccessful_inner_proposals_cannot_supply_a_finite_loss(case, status, x64):
    with jax.enable_x64(x64):
        if case == "saddle":

            def joint(t, z):
                return t * t / 2 - jnp.sum(z * z) / 2

            seed, max_iter = jnp.zeros(2), 100
        elif case == "nonfinite":

            def joint(t, z):
                return t * t / 2 + jnp.log(z)

            seed, max_iter = jnp.array(-1.0), 100
        elif case == "budget":

            def joint(t, z):
                return t * t / 2 + z**4 / 4 - 2 * z

            seed, max_iter = jnp.array(3.0), 1
        else:

            def joint(t, z):
                return t * t / 2 + jnp.where(z == 0, z, jnp.inf)

            seed, max_iter = jnp.array(0.0), 100
        model, split = density_model(joint, theta=jnp.array(0.0), z=seed)
        loss = opt.LaplaceLoss(model, split, latent=["z"], inner_max_iter=max_iter)
        position, carry = loss_carry(loss, ["theta"])
        value, proposed = jax.jit(loss.loss_train)(position, carry)
        assert int(proposed.status) == status
        assert not jnp.isfinite(value)
        assert int(carry.loss_state.status) == 0
        np.testing.assert_array_equal(carry.loss_state.latent_position["z"], seed)


@pytest.mark.parametrize("x64", [False, True])
def test_rounding_and_hidden_cancellation_preserve_convergence(x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        accuracy = 3e-5 if x64 else 0.002
        cases = [
            (rate, offset, cancelled)
            for rate in (0.3, 1.3, 2.0, math.e, 3.7, 7.0, 17.0, 30.0, 100.0)
            for offset, cancelled in ((0.0, 0.0), (1e6, 0.0), (1e6, 1e6))
        ]
        parameters = jnp.array(cases, dtype)
        expected_modes = np.log([case[0] for case in cases])
        resolution_steps = 0
        for seed in (-1.0, 0.0, 1.0, 3.0, 5.0):
            model, split = density_model(
                lambda theta, z: (jnp.exp(z) + theta[1]) - theta[0] * z - theta[2],
                theta=jnp.array([1.0, 0.0, 0.0], dtype),
                z=jnp.array(seed, dtype),
            )
            loss = opt.LaplaceLoss(model, split, latent=["z"])
            _, carry = loss_carry(loss, ["theta"])
            values, states = jax.jit(
                jax.vmap(
                    lambda p, loss=loss, carry=carry: loss.loss_train(
                        {"theta": p}, carry
                    )
                )
            )(parameters)
            assert np.all(np.asarray(states.status) == 1), (seed, states.status)
            assert np.all(np.isfinite(values))
            np.testing.assert_allclose(
                states.latent_position["z"], expected_modes, rtol=0, atol=accuracy
            )
            assert np.all(
                np.asarray(states.newton_decrement_squared) / 2 <= loss.inner_tol
            )
            resolution_steps += int(jnp.sum(states.n_resolution_steps))
        assert resolution_steps > 0


def test_resolution_acceptance_cannot_hide_a_large_value_increase():
    model, split = density_model(
        lambda theta, z: (
            1e6 + (z - 1.0) ** 2 / 2 + jnp.where(z > 0.75, 100.0, 0.0) + theta**2
        ),
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"], inner_max_iter=1)
    position, carry = loss_carry(loss, ["theta"])
    value, proposed = jax.jit(loss.loss_train)(position, carry)
    # The Newton candidate at 1 has zero decrement but an excessive value jump.
    # Backtracking chooses .5; a budget of one cannot yet yield a valid solve.
    assert int(proposed.status) == 2
    assert not jnp.isfinite(value)
    np.testing.assert_allclose(proposed.latent_position["z"], 0.5, atol=1e-6)


def test_equal_energy_trial_cannot_inflate_the_resolution_allowance():
    def joint(theta, z):
        return theta**2 + jnp.where(
            z < 0.75,
            (z - 1.0) ** 2 / 2,
            jnp.where(
                z < 1.2,
                0.5 + (z - 1.25) ** 2 / 2 - 0.25**2 / 2,
                1.0 + (z - 1.25) ** 2 / 2,
            ),
        )

    model, split = density_model(joint, theta=jnp.array(0.0), z=jnp.array(0.0))
    loss = opt.LaplaceLoss(model, split, latent=["z"], inner_max_iter=2)
    position, carry = loss_carry(loss, ["theta"])
    value, proposed = jax.jit(loss.loss_train)(position, carry)
    # The genuine equal-energy jump from 0 to 1 must not authorize the higher
    # value at 1.25. Both candidates have smaller true Newton decrements.
    assert int(proposed.status) == 2
    assert not jnp.isfinite(value)
    np.testing.assert_allclose(proposed.latent_position["z"], 1.125, atol=1e-6)


@pytest.mark.parametrize("method", ["forward_over_reverse", "reverse_over_reverse"])
def test_fitting_path_rejects_unsupported_second_derivatives(method):
    model, split = density_model(
        lambda t, z: t * t / 2 - t / 3 + z**4 / 4 - t * z,
        theta=jnp.array(1.0),
        z=jnp.array(1.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    _, carry = loss_carry(loss, ["theta"])

    def scalar(t):
        return loss.loss_train({"theta": t}, carry)[0]

    differentiate = jax.jacfwd if method == "forward_over_reverse" else jax.jacrev
    with pytest.raises(
        TypeError, match="first derivatives only.*approximate_joint_posterior"
    ):
        jax.jit(differentiate(jax.grad(scalar)))(jnp.array(1.0))


@pytest.mark.parametrize(
    "option,value",
    [
        ("inner_tol", 0.0),
        ("inner_tol", -1.0),
        ("inner_tol", math.inf),
        ("inner_tol", math.nan),
        ("inner_tol", True),
        ("inner_max_iter", 0),
        ("inner_max_iter", True),
        ("inner_max_iter", 1.5),
        ("warm_start", 1),
    ],
)
def test_invalid_solver_controls_fail_at_construction(option, value):
    model, split = density_model(
        lambda t, z: t * t + z * z, theta=jnp.array(0.0), z=jnp.array(0.0)
    )
    with pytest.raises(ValueError, match=option):
        opt.LaplaceLoss(model, split, latent=["z"], **{option: value})


@pytest.mark.parametrize("wrapped", [False, True])
def test_discrete_latent_density_is_rejected_even_with_float_values(wrapped):
    distribution = (
        (lambda: tfd.Independent(tfd.Poisson(jnp.ones(2)), reinterpreted_batch_ndims=1))
        if wrapped
        else (lambda: tfd.Poisson(jnp.ones(2)))
    )
    z = lsl.Var.new_param(jnp.ones(2), lsl.Dist(distribution), name="z")
    y = lsl.Var.new_obs(jnp.zeros(1), lsl.Dist(tfd.Normal, 0.0, 1.0), name="y")
    model = lsl.Model([z, y])
    with pytest.raises(ValueError, match="discrete"):
        opt.LaplaceLoss(model, latent=["z_value"])


def test_training_observations_cannot_also_be_integrated_or_optimized():
    model, split = density_model(
        lambda t, z: t * t + z * z, theta=jnp.array(0.0), z=jnp.array(0.0)
    )
    with pytest.raises(ValueError, match="training|observation"):
        opt.LaplaceLoss(model, split, latent=["data"])
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    with pytest.raises(ValueError, match="training|observation"):
        loss.position(["data_value"])


def test_joint_density_retains_latent_priors_and_transformation_jacobians():
    scale = lsl.Var.new_param(
        2.0, lsl.Dist(tfd.LogNormal, 0.0, 1.0), bijector=tfb.Exp(), name="scale"
    )
    z = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, scale), name="z")
    y = lsl.Var.new_obs(jnp.array([3.0]), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    model = lsl.Model([y])
    loss = opt.LaplaceLoss(model, latent=["z"])
    keys = list(loss.default_position_keys)
    assert len(keys) == 1
    position, carry = loss_carry(loss, keys)
    (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
    # s=log(scale) ~ N(0,1), y|s ~ N(0,1+exp(2s)).
    expected = (
        0.5 * math.log(2.0) ** 2 + math.log(2 * math.pi) + 0.5 * math.log(5.0) + 0.9
    )
    np.testing.assert_allclose(value, expected, atol=2e-6)
    np.testing.assert_allclose(gradient[keys[0]], math.log(2.0) - 0.64, atol=2e-6)
    np.testing.assert_allclose(state.latent_position["z"], 2.4, atol=2e-6)
    np.testing.assert_allclose(state.latent_precision_cholesky**2, [[1.25]], atol=2e-6)
    assert model.vars["z"].parameter


@pytest.mark.parametrize("warm_start", [True, False])
def test_direct_evaluations_use_frozen_warm_or_original_cold_seed(warm_start):
    model, split = density_model(
        lambda theta, z: theta * theta / 2 + (z - theta) ** 2 / 2,
        theta=jnp.array(0.7),
        z=jnp.array(-2.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"], warm_start=warm_start)
    position, carry = loss_carry(loss, ["theta"])
    value, state = jax.jit(loss.loss_train)(position, carry)
    committed = replace(carry, loss_state=state)
    # Evaluating a discarded trial cannot mutate the common committed seed.
    loss.loss_train({"theta": jnp.array(4.0)}, committed)
    again, repeated = jax.jit(loss.loss_train)(position, committed)
    assert int(repeated.n_iter) == (0 if warm_start else 1)
    np.testing.assert_allclose(again, value, atol=1e-6)
    np.testing.assert_allclose(
        committed.loss_state.latent_position["z"], 0.7, atol=1e-6
    )


def fit_engine(loss, optimizers="lbfgs", epochs=6, **kwargs):
    return opt.LieselOptim(
        loss.model,
        loss=loss,
        optimizers=optimizers,
        loss_monitor="train_full_data",
        stopper=opt.Stopper(epochs=epochs, patience=epochs),
        show_progress=False,
        **kwargs,
    ).build_engine()


@pytest.mark.parametrize("optimizer", ["lbfgs", optax.sgd(0.5)])
@pytest.mark.parametrize("warm_start", [False, True])
def test_gaussian_marginal_fit_excludes_latents_and_matches_saved_states(
    optimizer, warm_start
):
    model, split = density_model(
        lambda theta, z: (theta - 2) ** 2 / 2 + (z - 3 * theta) ** 2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(-4.0),
    )
    loss = opt.LaplaceLoss(
        model, split, latent=["z"], warm_start=warm_start, inner_tol=1e-10
    )
    result = fit_engine(loss, optimizer, epochs=14).fit()
    assert result.status == "max_epochs"
    assert set(result.position_final) == {"theta"}
    np.testing.assert_allclose(result.position_final["theta"], 2.0, atol=2e-4)
    assert model.vars["z"].parameter
    for position, state in (
        (result.position_final, result.loss_state_final),
        (result.position_min_monitor, result.loss_state_min_monitor),
    ):
        assert int(state.status) == 1
        np.testing.assert_allclose(state.outer_position["theta"], position["theta"])
        np.testing.assert_allclose(
            state.latent_position["z"], 3 * position["theta"], atol=2e-5
        )


def test_compiled_blocks_preserve_fixed_coordinates_and_full_state():
    model, split = density_model(
        lambda a, b, fixed, z: (
            (a - fixed) ** 2 / 2 + (b + fixed) ** 2 / 2 + (z - a - 2 * b) ** 2 / 2
        ),
        a=jnp.array(0.0),
        b=jnp.array(0.0),
        fixed=jnp.array(2.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(
        loss,
        [opt.Optimizer(["a"], optax.sgd(1.0)), opt.Optimizer(["b"], optax.sgd(1.0))],
        epochs=2,
    ).fit()
    assert set(result.position_final) == {"a", "b"}
    np.testing.assert_allclose(result.position_final["a"], 2.0, atol=1e-6)
    np.testing.assert_allclose(result.position_final["b"], -2.0, atol=1e-6)
    np.testing.assert_allclose(result.loss_state_final.latent_position["z"], -2.0)
    assert set(result.loss_state_final.outer_position) == {"a", "b"}


def test_laplace_best_and_final_states_match_distinct_conditional_curvatures():
    model, split = density_model(
        lambda theta, z: (
            theta**2 / 2
            + (1 + theta**2) * (z - theta) ** 2 / 2
            - jnp.log1p(theta**2) / 2
        ),
        theta=jnp.array(4.0),
        z=jnp.array(4.0),
    )
    result = fit_engine(
        opt.LaplaceLoss(model, split, latent=["z"]), optax.sgd(2.5), epochs=2
    ).fit()
    for name, theta in (("min_monitor", -6.0), ("final", 9.0)):
        state = getattr(result, "loss_state_" + name)
        np.testing.assert_allclose(getattr(result, "position_" + name)["theta"], theta)
        np.testing.assert_allclose(state.latent_position["z"], theta, atol=1e-5)
        np.testing.assert_allclose(
            state.latent_precision_cholesky, [[math.sqrt(1 + theta**2)]]
        )


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("initial_failure", [False, True])
def test_inner_failure_rolls_back_matched_state_and_preserves_disk_checkpoint(
    debug, initial_failure, tmp_path
):
    model, split = density_model(
        lambda theta, z: jnp.where(
            theta < 2, (theta - 4) ** 2 / 2 + (z - theta) ** 2 / 2, jnp.inf
        ),
        theta=jnp.array(3.0 if initial_failure else 0.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    engine = fit_engine(loss, optax.sgd(0.25), epochs=5)
    engine.debug_nans = debug
    path = tmp_path / "fit.pkl"
    result = engine.fit(checkpoint=path, checkpoint_every=1)
    assert result.status == "numerical_failure"
    assert "inner" in result.failure_reason.lower()
    assert result.checkpoint is None
    assert int(result.failed_loss_state.status) == 4
    assert result.n_epochs == (0 if initial_failure else 2)
    np.testing.assert_allclose(
        result.position_final["theta"], 3.0 if initial_failure else 1.75
    )
    if initial_failure:
        assert result.loss_state_final is None
        assert result.loss_state_min_monitor is None
        assert not path.exists()
    else:
        state = result.loss_state_final
        assert int(state.status) == 1
        np.testing.assert_allclose(state.outer_position["theta"], 1.75)
        np.testing.assert_allclose(state.latent_position["z"], 1.75)
        np.testing.assert_allclose(result.position_min_monitor["theta"], 1.75)
        checkpoint = opt.OptimCheckpoint.load(path)
        assert checkpoint.n_epochs == 2
        again = engine.fit(checkpoint=checkpoint)
        assert again.status == "numerical_failure"
        np.testing.assert_allclose(again.position_final["theta"], 1.75)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("optimizer", ["lbfgs", optax.sgd(0.1)])
def test_nonfinite_outer_gradient_is_handled_even_with_finite_inner_solution(
    debug, optimizer
):
    model, split = density_model(
        lambda theta, z: jnp.sqrt(theta) + z**2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    engine = fit_engine(opt.LaplaceLoss(model, split, latent=["z"]), optimizer)
    engine.debug_nans = debug
    result = engine.fit()
    assert result.status == "numerical_failure"
    assert "gradient" in result.failure_reason.lower()
    assert int(result.failed_loss_state.status) == 1
    assert result.loss_state_final is None
    assert result.n_epochs == 0
    assert result.checkpoint is None
    np.testing.assert_array_equal(result.position_final["theta"], 0.0)


def test_lbfgs_accepts_a_safe_finite_fallback_after_search_budget():
    model, split = density_model(
        lambda theta, z: (theta - 100) ** 2 / 2 + (z - theta) ** 2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    optimizer = opt.LBFGS(
        ["theta"],
        optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=1)),
    )
    result = fit_engine(opt.LaplaceLoss(model, split, latent=["z"]), [optimizer]).fit()
    assert result.status == "max_epochs"
    np.testing.assert_allclose(result.position_final["theta"], 100.0, atol=1e-5)
    assert result.failed_loss_state is None


@pytest.mark.parametrize("debug", [False, True])
def test_lbfgs_reports_failure_when_every_trial_is_invalid(debug):
    model, split = density_model(
        lambda theta, z: jnp.where(theta == 0, theta + z**2 / 2, jnp.inf),
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    engine = fit_engine(opt.LaplaceLoss(model, split, latent=["z"]))
    engine.debug_nans = debug
    result = engine.fit()
    assert result.status == "numerical_failure"
    assert "line search" in result.failure_reason.lower()
    assert result.n_epochs == 0
    np.testing.assert_array_equal(result.position_final["theta"], 0.0)
    assert result.loss_state_final is None
    assert result.checkpoint is None


def test_lbfgs_accepts_zero_step_at_valid_stationary_point():
    model, split = density_model(
        lambda theta, z: theta**2 / 2 + (z - theta) ** 2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(2.0),
    )
    result = fit_engine(opt.LaplaceLoss(model, split, latent=["z"]), epochs=2).fit()
    assert result.status == "max_epochs"
    assert int(result.loss_state_final.status) == 1
    np.testing.assert_allclose(result.position_final["theta"], 0.0)


def test_lbfgs_rejects_invalid_trials_and_accepts_a_valid_candidate():
    model, split = density_model(
        lambda theta, z: jnp.where(
            theta < 2, (theta - 1) ** 2 / 2 + (z - theta) ** 2 / 2, jnp.inf
        ),
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    optimizer = opt.LBFGS(["theta"], optax.lbfgs(learning_rate=10.0))
    result = fit_engine(
        opt.LaplaceLoss(model, split, latent=["z"]), [optimizer], epochs=5
    ).fit()
    assert result.status == "max_epochs"
    np.testing.assert_allclose(result.position_final["theta"], 1.0, atol=1e-5)
    assert int(result.loss_state_final.status) == 1
    assert result.failed_loss_state is None


def checkpoint_engine(latent_value=0.0, **loss_options):
    theta = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, 0.0, 2.0), name="theta")
    z = lsl.Var.new_param(latent_value, lsl.Dist(tfd.Normal, theta, 1.0), name="z")
    y = lsl.Var.new_obs(jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    model = lsl.Model([y])
    loss = opt.LaplaceLoss(model, latent=["z"], **loss_options)
    return fit_engine(loss, optax.adam(0.1), epochs=5)


def test_laplace_checkpoint_resumes_in_a_reconstructed_engine(tmp_path):
    expected = checkpoint_engine().fit()
    first = checkpoint_engine().fit(pause_after=2)
    path = tmp_path / "resume.pkl"
    first.checkpoint.save(path)
    resumed = checkpoint_engine().fit(checkpoint=path)
    assert resumed.n_epochs == 5
    assert first.n_epochs == 2
    for actual, reference in zip(
        jax.tree.leaves(
            (resumed.history, resumed.loss_state_final, resumed.loss_state_min_monitor)
        ),
        jax.tree.leaves(
            (
                expected.history,
                expected.loss_state_final,
                expected.loss_state_min_monitor,
            )
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference, atol=1e-6)
    assert len(first.history.loss_train) == 2


@pytest.mark.parametrize(
    "option,value", [("warm_start", False), ("inner_tol", 1e-5), ("inner_max_iter", 50)]
)
def test_laplace_checkpoint_rejects_changed_solver_configuration(option, value):
    first = checkpoint_engine().fit(pause_after=1)
    with pytest.raises(ValueError, match="loss configuration"):
        checkpoint_engine(**{option: value}).fit(checkpoint=first.checkpoint)


def test_laplace_checkpoint_rejects_changed_latent_shape():
    first = checkpoint_engine().fit(pause_after=1)
    with pytest.raises(ValueError, match="loss state"):
        checkpoint_engine(latent_value=jnp.zeros(2)).fit(checkpoint=first.checkpoint)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("progress", ["none", "epochs", "batches"])
def test_failure_rollback_is_consistent_in_all_progress_paths(debug, progress):
    model, split = density_model(
        lambda theta, z: theta * jnp.sqrt(theta) + z**2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    batches = opt.BatchManager(
        [opt.Batches(["data"], axis_size=1, batch_size=1, shuffle=True)], epoch_size=3
    )
    engine = fit_engine(loss, optax.sgd(0.1), batches=batches)
    engine.debug_nans = debug
    engine.show_progress = progress != "none"
    engine.show_step_progress = progress == "batches"
    engine.progress_update_every = 1
    engine.step_progress_update_every = 1
    result = engine.fit()
    assert result.status == "numerical_failure"
    assert result.n_epochs == 0
    assert result.loss_state_final is None
    np.testing.assert_array_equal(result.position_final["theta"], 0.0)
    if debug:
        assert result.nan_debug is not None
        assert jnp.isfinite(result.nan_debug.reproduce_loss(engine))
        assert jnp.isnan(result.nan_debug.reproduce_step(engine).position["theta"])


def test_full_data_repeated_updates_and_multiple_blocks_keep_epoch_commit_timing():
    class CountingLoss(opt.LossMixin):
        def __init__(self, model, split):
            self.model, self.split = model, split

        def position(self, keys):
            return self.model.extract_position(keys)

        def init_state(self, params, carry):
            return {"count": jnp.array(0)}

        def loss_train_batched(self, params, carry):
            p = carry.position | params
            return (p["a"] ** 2 + p["b"] ** 2) / 2, {
                "count": carry.loss_state["count"] + 1
            }

        loss_train = loss_train_batched

    model, split = density_model(
        lambda a, b: a * a + b * b, a=jnp.array(8.0), b=jnp.array(16.0)
    )
    loss = CountingLoss(model, split)
    batches = opt.BatchManager(
        [opt.Batches(["data"], axis_size=1, batch_size=1, shuffle=True)], epoch_size=3
    )
    result = fit_engine(
        loss,
        [opt.Optimizer(["a"], optax.sgd(0.5)), opt.Optimizer(["b"], optax.sgd(0.5))],
        epochs=2,
        batches=batches,
    ).fit()
    assert result.loss_state_final["count"] == 2
    np.testing.assert_allclose(result.position_final["a"], 0.125)
    np.testing.assert_allclose(result.position_final["b"], 0.25)


def test_nonfinite_outer_gradient_stops_before_the_next_update():
    model, split = density_model(
        lambda theta, z: (
            -theta
            + jnp.where(theta == 2, (theta - 2) * jnp.sqrt(jnp.abs(theta - 2)), 0.0)
            + (z - theta) ** 2 / 2
        ),
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    result = fit_engine(
        opt.LaplaceLoss(model, split, latent=["z"]), optax.sgd(1.0), epochs=4
    ).fit()
    assert result.status == "numerical_failure"
    assert "gradient" in result.failure_reason.lower()
    # Monitoring commits the valid conditional mode; the next update checks the
    # outer gradient. No extra backward pass is required solely for monitoring.
    assert result.n_epochs == 2
    np.testing.assert_allclose(result.position_final["theta"], 2.0)
    np.testing.assert_allclose(result.loss_state_final.latent_position["z"], 2.0)
    np.testing.assert_allclose(result.failed_loss_state.outer_position["theta"], 2.0)


@pytest.mark.parametrize(
    "configuration", ["minibatch", "replacement", "ema", "validation"]
)
def test_laplace_rejects_unsupported_training_and_monitoring(configuration):
    engine = checkpoint_engine()
    if configuration in ("minibatch", "replacement"):
        engine.batches = opt.Batches.from_split(
            engine.split,
            batch_size=1 if configuration == "minibatch" else 2,
            shuffle=True,
            sample_with_replacement=configuration == "replacement",
        )
    else:
        engine.loss_monitor = (
            opt.EmaTrainLossMonitor(1.0) if configuration == "ema" else "validation"
        )
    with pytest.raises(ValueError, match="full-data|validation"):
        engine.fit()


@pytest.mark.parametrize("progress", [False, True])
def test_laplace_checkpoint_survives_an_interruption(progress, monkeypatch, tmp_path):
    engine = checkpoint_engine()
    engine.show_progress = progress
    path = tmp_path / "interrupted.pkl"
    save = opt.OptimCheckpoint.save

    def interrupt_after_save(checkpoint, path):
        save(checkpoint, path)
        raise KeyboardInterrupt

    monkeypatch.setattr(opt.OptimCheckpoint, "save", interrupt_after_save)
    with pytest.raises(KeyboardInterrupt):
        engine.fit(checkpoint=path, checkpoint_every=1)
    monkeypatch.setattr(opt.OptimCheckpoint, "save", save)
    checkpoint = opt.OptimCheckpoint.load(path)
    assert checkpoint.n_epochs == 1
    resumed = checkpoint_engine().fit(checkpoint=checkpoint)
    expected = checkpoint_engine().fit()
    np.testing.assert_allclose(
        resumed.history.loss_monitor, expected.history.loss_monitor
    )
    np.testing.assert_allclose(
        resumed.loss_state_final.latent_position["z"],
        expected.loss_state_final.latent_position["z"],
    )


def test_float64_fit_and_checkpoint_keep_loss_state_dtype():
    with jax.enable_x64():
        model, split = density_model(
            lambda theta, z: (theta - 2) ** 2 / 2 + (z - theta) ** 2 / 2,
            theta=jnp.array(0.0, dtype=jnp.float64),
            z=jnp.array(0.0, dtype=jnp.float64),
        )
        loss = opt.LaplaceLoss(model, split, latent=["z"])
        engine = fit_engine(loss, epochs=3)
        first = engine.fit(pause_after=1)
        result = engine.fit(checkpoint=first.checkpoint)
        assert result.status == "max_epochs"
        assert result.loss_state_final.latent_precision_cholesky.dtype == jnp.float64
        np.testing.assert_allclose(result.position_final["theta"], 2.0, atol=1e-10)


def test_float32_coordinates_allow_float64_density_accumulation():
    with jax.enable_x64():
        coefficient = jnp.array(1.0, dtype=jnp.float64)
        model, split = density_model(
            lambda theta, z: coefficient * (theta**2 / 2 + (z - theta) ** 2 / 2),
            theta=jnp.array(0.7, dtype=jnp.float32),
            z=jnp.array(-2.0, dtype=jnp.float32),
        )
        loss = opt.LaplaceLoss(model, split, latent=["z"])
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
        assert value.dtype == jnp.float64
        assert state.latent_position["z"].dtype == jnp.float32
        np.testing.assert_allclose(
            value, 0.7**2 / 2 - math.log(2 * math.pi) / 2, atol=1e-6
        )
        np.testing.assert_allclose(gradient["theta"], 0.7, atol=1e-6)


def test_joint_gaussian_posterior_has_correct_cross_covariances():
    # theta ~ N(0,1), z|theta ~ N([theta,-theta/2], inverse([[2,1],[1,2]])).
    model, split = density_model(
        lambda theta, z: (
            theta**2 / 2
            + (z[0] - theta) ** 2
            + (z[1] + theta / 2) ** 2
            + (z[0] - theta) * (z[1] + theta / 2)
        ),
        theta=jnp.array(0.0),
        z=jnp.zeros(2),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    approximation = loss.approximate_joint_posterior(result)
    assert isinstance(approximation, opt.LaplaceApproximation)
    assert approximation.valid
    assert approximation.names == ("theta", "z")
    assert approximation.shapes == ((), (2,))
    np.testing.assert_allclose(approximation.mean["theta"], 0.0)
    np.testing.assert_allclose(approximation.mean["z"], [0.0, 0.0])
    np.testing.assert_allclose(
        approximation.covariance(),
        [[1.0, 1.0, -0.5], [1.0, 5 / 3, -5 / 6], [-0.5, -5 / 6, 11 / 12]],
        atol=2e-6,
    )


@pytest.mark.parametrize("x64", [False, True])
def test_nonlinear_joint_posterior_uses_true_implicit_second_derivative(x64):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        model, split = density_model(
            lambda theta, z: theta**2 / 2 - theta / 3 + z**4 / 4 - theta * z,
            theta=jnp.array(1.0, dtype),
            z=jnp.array(1.0, dtype),
        )
        loss = opt.LaplaceLoss(model, split, latent=["z"])
        result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
        approximation = loss.approximate_joint_posterior(result)
        accuracy = 1e-10 if x64 else 2e-5
        assert approximation.valid
        # At theta=z=1, the exact marginal Laplace curvature is 1/3.
        # Differentiating one Newton correction instead gives the wrong 5/9.
        np.testing.assert_allclose(
            approximation.diagnostics["outer_precision"],
            [[1 / 3]],
            atol=accuracy,
            rtol=0,
        )
        np.testing.assert_allclose(
            approximation.covariance(),
            [[3.0, 1.0], [1.0, 2 / 3]],
            atol=accuracy,
            rtol=0,
        )


def test_posterior_requires_stationarity_and_allows_explicit_failure_inspection():
    model, split = density_model(
        lambda theta, z: theta**2 / 2 + (z - theta) ** 2 / 2,
        theta=jnp.array(1.0),
        z=jnp.array(1.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    with pytest.raises(RuntimeError, match="stationarity"):
        loss.approximate_joint_posterior(result)
    failed = loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert failed.precision_cholesky is None
    assert "stationarity" in failed.diagnostics["reason"]
    np.testing.assert_allclose(failed.diagnostics["outer_gradient"], [1.0])
    np.testing.assert_allclose(failed.diagnostics["outer_precision"], [[1.0]])
    np.testing.assert_allclose(
        failed.diagnostics["outer_newton_decrement_squared"], 1.0
    )
    with pytest.raises(RuntimeError, match="invalid"):
        failed.covariance()
    with pytest.raises(RuntimeError, match="invalid"):
        failed.sample(seed=jax.random.key(0))
    assert loss.approximate_joint_posterior(result, stationarity_tol=1.0).valid


@pytest.mark.parametrize("curvature", ["outer", "latent"])
def test_posterior_rejects_two_negative_curvature_directions(curvature):
    # Positive determinant alone cannot establish positive definiteness.
    model, split = density_model(
        lambda theta, z: -jnp.sum(theta**2) / 2 + jnp.sum(z**2) / 2,
        theta=jnp.zeros(2),
        z=jnp.zeros(2),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    if curvature == "latent":
        changed, changed_split = density_model(
            lambda theta, z: jnp.sum(theta**2) / 2 - jnp.sum(z**2) / 2,
            theta=jnp.zeros(2),
            z=jnp.zeros(2),
        )
        loss = opt.LaplaceLoss(changed, changed_split, latent=["z"])
    failed = loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert "curvature" in failed.diagnostics["reason"]
    np.testing.assert_allclose(failed.diagnostics[curvature + "_precision"], -np.eye(2))
    assert failed.precision_cholesky is None


def test_posterior_rejects_singular_outer_precision():
    model, split = density_model(
        lambda theta, z: jnp.sum(theta) ** 2 + z**2 / 2,
        theta=jnp.zeros(2),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    with pytest.raises(RuntimeError, match="curvature"):
        loss.approximate_joint_posterior(result)
    failed = loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    np.testing.assert_allclose(
        failed.diagnostics["outer_precision"], [[2.0, 2.0], [2.0, 2.0]]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("at", "missing"),
        ("raise_on_failure", 1),
        ("stationarity_tol", 0),
        ("stationarity_tol", math.inf),
    ],
)
def test_posterior_configuration_errors_are_not_suppressed(field, value):
    engine = checkpoint_engine()
    result = engine.fit()
    with pytest.raises(ValueError, match=field):
        engine.loss.approximate_joint_posterior(result, **{field: value})


@pytest.mark.parametrize(
    "invalid",
    [
        "names",
        "shapes",
        "values",
        "dtype",
        "outer",
        "missing",
        "failed",
        "configuration",
    ],
)
def test_posterior_rejects_visibly_incompatible_snapshots(invalid):
    model, split = density_model(
        lambda theta, z: theta**2 / 2 + (z - theta) ** 2 / 2,
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    state = result.loss_state_min_monitor
    changes = {
        "names": {"latent_names": ("other",)},
        "shapes": {"latent_shapes": ((1,),)},
        "values": {"latent_position": {"z": jnp.ones(2)}},
        "dtype": {"latent_position": {"z": jnp.array(0, dtype=jnp.int32)}},
        "outer": {"outer_position": {"theta": jnp.array(2.0)}},
        "failed": {"status": jnp.array(2)},
    }
    if invalid == "missing":
        result.loss_state_min_monitor = None
    elif invalid == "configuration":
        loss.warm_start = False
    else:
        result.loss_state_min_monitor = replace(state, **changes[invalid])
    with pytest.raises(RuntimeError, match="state|coordinate|configuration"):
        loss.approximate_joint_posterior(result)
    failed = loss.approximate_joint_posterior(result, raise_on_failure=False)
    assert not failed.valid
    assert failed.diagnostics["reason"]


def test_posterior_can_inspect_a_fit_with_no_valid_snapshot():
    model, split = density_model(
        lambda theta, z: theta**2 + jnp.log(z),
        theta=jnp.array(0.0),
        z=jnp.array(-1.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(0.1)).fit()
    assert result.status == "numerical_failure"
    for at in ("min_monitor", "final"):
        failed = loss.approximate_joint_posterior(result, at=at, raise_on_failure=False)
        assert not failed.valid
        assert failed.diagnostics["reason"]


def test_posterior_samples_restore_matrix_shapes_and_gaussian_moments():
    model, split = density_model(
        lambda theta, z: (
            theta**2 / 2
            + (z[0, 0] - theta) ** 2
            + (z[0, 1] + theta / 2) ** 2
            + (z[0, 0] - theta) * (z[0, 1] + theta / 2)
        ),
        theta=jnp.array(0.0),
        z=jnp.zeros((1, 2)),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    approximation = loss.approximate_joint_posterior(
        fit_engine(loss, optax.sgd(0.0), epochs=1).fit()
    )
    key = jax.random.key(42)
    single = approximation.sample(seed=key)
    assert single["theta"].shape == ()
    assert single["z"].shape == (1, 2)
    for name, value in single.items():
        np.testing.assert_array_equal(value, approximation.sample(seed=key)[name])
    draws = approximation.sample((2, 3), seed=key)
    assert draws["theta"].shape == (2, 3)
    assert draws["z"].shape == (2, 3, 1, 2)
    draws = approximation.sample((60_000,), seed=key)
    flat = np.column_stack([draws["theta"], np.asarray(draws["z"]).reshape(60_000, 2)])
    # With 60k draws, these bounds exceed five Monte Carlo standard errors.
    np.testing.assert_allclose(flat.mean(axis=0), [0.0, 0.0, 0.0], atol=0.03, rtol=0)
    np.testing.assert_allclose(
        np.cov(flat, rowvar=False),
        [[1.0, 1.0, -0.5], [1.0, 5 / 3, -5 / 6], [-0.5, -5 / 6, 11 / 12]],
        atol=0.05,
        rtol=0,
    )


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
def test_posterior_draws_predict_transforms_and_keep_omitted_coordinates_fixed(shape):
    scale = lsl.Var.new_param(
        1.0, lsl.Dist(tfd.LogNormal, 0.0, 1.0), bijector=tfb.Exp(), name="scale"
    )
    fixed = lsl.Var.new_param(3.0, name="fixed")
    location = lsl.Var.new_calc(
        lambda scale, fixed: jnp.log(scale) + fixed - 3, scale, fixed, name="location"
    )
    z = lsl.Var.new_param(0.0, lsl.Dist(tfd.Normal, location, 1.0), name="z")
    y = lsl.Var.new_obs(jnp.array([0.0]), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    model = lsl.Model([y])
    loss = opt.LaplaceLoss(model, latent=["z"])
    keys = [k for k in loss.default_position_keys if k != "fixed"]
    result = fit_engine(loss, [opt.Optimizer(keys, optax.sgd(0.0))], epochs=1).fit()
    approximation = loss.approximate_joint_posterior(result)
    assert "fixed" not in approximation.mean
    draws = approximation.sample(shape, seed=jax.random.key(2))
    predicted = model.predict(draws, predict=["scale", "fixed"])
    assert predicted["scale"].shape == shape
    np.testing.assert_allclose(predicted["scale"], jnp.exp(draws[keys[0]]), rtol=1e-6)
    np.testing.assert_allclose(predicted["fixed"], 3.0)


def test_posterior_selects_matched_best_and_final_snapshots():
    with jax.enable_x64():
        model, split = density_model(
            lambda theta, z: (
                theta**2 / 2
                + (1 + theta**2) * (z - theta) ** 2 / 2
                - jnp.log1p(theta**2) / 2
            ),
            theta=jnp.array(4.0),
            z=jnp.array(4.0),
        )
        loss = opt.LaplaceLoss(model, split, latent=["z"])
        result = fit_engine(loss, optax.sgd(2.5), epochs=2).fit()
        for at, theta in (("min_monitor", -6.0), ("final", 9.0)):
            # An explicit loose stationarity threshold isolates selection here.
            approximation = loss.approximate_joint_posterior(
                result, at=at, stationarity_tol=100.0
            )
            np.testing.assert_allclose(approximation.mean["theta"], theta, atol=1e-10)
            np.testing.assert_allclose(approximation.mean["z"], theta, atol=1e-10)
            np.testing.assert_allclose(
                approximation.diagnostics["latent_precision"],
                [[1 + theta**2]],
                atol=1e-10,
            )
            np.testing.assert_allclose(
                approximation.covariance(),
                [[1, 1], [1, 1 + 1 / (1 + theta**2)]],
                atol=1e-10,
            )


def test_posterior_detects_nonfinite_derivatives_at_a_terminal_monitored_point():
    model, split = density_model(
        lambda theta, z: (
            -theta
            + jnp.where(theta == 2, (theta - 2) * jnp.sqrt(jnp.abs(theta - 2)), 0.0)
            + (z - theta) ** 2 / 2
        ),
        theta=jnp.array(0.0),
        z=jnp.array(0.0),
    )
    loss = opt.LaplaceLoss(model, split, latent=["z"])
    result = fit_engine(loss, optax.sgd(1.0), epochs=2).fit()
    assert result.status == "max_epochs"
    assert int(result.loss_state_final.status) == 1
    failed = loss.approximate_joint_posterior(
        result, at="final", raise_on_failure=False
    )
    assert not failed.valid
    assert "Non-finite" in failed.diagnostics["reason"]


@pytest.mark.parametrize("x64", [False, True])
def test_poisson_random_intercept_matches_independent_scalar_laplace_and_quadrature(
    x64,
):
    counts = np.array([0.0, 1.0, 3.0, 2.0])
    constant = math.log(2 * math.pi) + math.log(2) + gammaln(counts + 1).sum()

    def joint(theta, z):
        return (
            theta**2 / 8
            + z**2 / 2
            + len(counts) * math.exp(theta + z)
            - counts.sum() * (theta + z)
            + constant
        )

    def reference(theta):
        mode = brentq(
            lambda z: z + len(counts) * math.exp(theta + z) - counts.sum(),
            -20.0,
            10.0,
            xtol=1e-14,
        )
        precision = 1 + len(counts) * math.exp(theta + mode)
        value = joint(theta, mode) + math.log(precision / (2 * math.pi)) / 2
        return value, mode

    expected, mode = reference(0.4)
    h = 1e-4
    expected_gradient = (reference(0.4 + h)[0] - reference(0.4 - h)[0]) / (2 * h)
    # Normal tails outside +/-20 contribute less than 6e-89 before centering.
    integral, error = quad(
        lambda z: math.exp(joint(0.4, mode) - joint(0.4, z)),
        -20,
        20,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    exact = joint(0.4, mode) - math.log(integral)
    assert error < 1e-10
    # This is approximation error, distinct from the numerical checks below.
    assert 0.001 < abs(expected - exact) < 0.02

    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        theta = lsl.Var.new_param(
            jnp.array(0.4, dtype), lsl.Dist(tfd.Normal, 0.0, 2.0), name="theta"
        )
        z = lsl.Var.new_param(
            jnp.array(-1.0, dtype), lsl.Dist(tfd.Normal, 0.0, 1.0), name="z"
        )
        eta = lsl.Var.new_calc(lambda t, z: t + z, theta, z, name="eta")
        y = lsl.Var.new_obs(
            jnp.asarray(counts, dtype), lsl.Dist(tfd.Poisson, log_rate=eta), name="y"
        )
        model = lsl.Model([y], to_float32=False)
        loss = opt.LaplaceLoss(model, latent=["z"])
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
        assert int(state.status) == 1
        np.testing.assert_allclose(value, expected, atol=5e-6 if x64 else 2e-5, rtol=0)
        np.testing.assert_allclose(
            gradient["theta"], expected_gradient, atol=5e-6 if x64 else 2e-4, rtol=0
        )
        np.testing.assert_allclose(
            state.latent_position["z"], mode, atol=1e-5 if x64 else 5e-4, rtol=0
        )
        assert float(state.newton_decrement_squared) / 2 <= loss.inner_tol


@pytest.mark.parametrize("x64", [False, True])
def test_coupled_spline_penalty_matches_exact_gaussian_marginal(x64):
    x = np.linspace(0, 1, 16)
    knots = np.r_[np.zeros(4), [0.25, 0.5, 0.75], np.ones(4)]
    design = BSpline.design_matrix(x, knots, k=3).toarray()
    n, d = design.shape
    difference = np.diff(np.eye(d), n=2, axis=0)
    # A proper prior: second-difference penalty with its null space regularized.
    precision = difference.T @ difference + 0.2 * np.eye(d)
    prior_covariance = np.linalg.inv(precision)
    y_value = 0.3 + np.sin(2 * math.pi * x)
    covariance = 0.25 * np.eye(n) + design @ prior_covariance @ design.T
    residual = y_value - 0.2
    solved = np.linalg.solve(covariance, residual)
    expected = (
        (
            residual @ solved
            + np.linalg.slogdet(covariance)[1]
            + n * math.log(2 * math.pi)
        )
        / 2
        + 0.2**2 / 8
        + math.log(2)
        + math.log(2 * math.pi) / 2
    )
    expected_gradient = 0.2 / 4 - solved.sum()
    conditional_precision = precision + design.T @ design / 0.25
    expected_mode = np.linalg.solve(conditional_precision, design.T @ residual / 0.25)

    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        theta = lsl.Var.new_param(
            jnp.array(0.2, dtype), lsl.Dist(tfd.Normal, 0.0, 2.0), name="theta"
        )
        coef = lsl.Var.new_param(
            jnp.ones(d, dtype),
            lsl.Dist(
                tfd.MultivariateNormalTriL,
                loc=jnp.zeros(d, dtype),
                scale_tril=jnp.asarray(np.linalg.cholesky(prior_covariance), dtype),
            ),
            name="coef",
        )
        mean = lsl.Var.new_calc(
            lambda t, c: t + jnp.asarray(design, dtype) @ c, theta, coef, name="mean"
        )
        y = lsl.Var.new_obs(
            jnp.asarray(y_value, dtype), lsl.Dist(tfd.Normal, mean, 0.5), name="y"
        )
        model = lsl.Model([y], to_float32=False)
        loss = opt.LaplaceLoss(model, latent=["coef"])
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
        assert int(state.status) == 1
        accuracy = 1e-8 if x64 else 5e-4
        np.testing.assert_allclose(value, expected, atol=accuracy, rtol=0)
        np.testing.assert_allclose(
            gradient["theta"], expected_gradient, atol=accuracy, rtol=0
        )
        np.testing.assert_allclose(
            state.latent_position["coef"], expected_mode, atol=accuracy, rtol=0
        )
        np.testing.assert_allclose(
            state.latent_precision_cholesky,
            np.linalg.cholesky(conditional_precision),
            atol=accuracy,
            rtol=0,
        )


@pytest.mark.parametrize("x64", [False, True])
def test_large_poisson_count_converges_from_a_remote_start(x64):
    count, theta_value = 100_000.0, 3.0
    mode = brentq(lambda z: z + math.exp(theta_value + z) - count, 0, 20, xtol=1e-14)
    rate = math.exp(theta_value + mode)
    expected_gradient = theta_value / 4 - mode + rate / (2 * (1 + rate) ** 2)
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        theta = lsl.Var.new_param(
            jnp.array(theta_value, dtype), lsl.Dist(tfd.Normal, 0.0, 2.0), name="theta"
        )
        z = lsl.Var.new_param(
            jnp.array(-3.0, dtype), lsl.Dist(tfd.Normal, 0.0, 1.0), name="z"
        )
        eta = lsl.Var.new_calc(lambda t, z: t + z, theta, z, name="eta")
        y = lsl.Var.new_obs(
            jnp.array([count], dtype), lsl.Dist(tfd.Poisson, log_rate=eta), name="y"
        )
        loss = opt.LaplaceLoss(lsl.Model([y], to_float32=False), latent=["z"])
        position, carry = loss_carry(loss, ["theta"])
        (value, state), gradient = jax.jit(loss.value_and_grad)(position, carry)
        assert int(state.status) == 1
        assert jnp.isfinite(value)
        assert float(state.newton_decrement_squared) / 2 <= loss.inner_tol
        np.testing.assert_allclose(
            state.latent_position["z"], mode, atol=1e-7 if x64 else 1e-5, rtol=0
        )
        np.testing.assert_allclose(
            gradient["theta"], expected_gradient, atol=1e-7 if x64 else 2e-4, rtol=0
        )
        np.testing.assert_allclose(
            state.latent_precision_cholesky,
            [[math.sqrt(1 + rate)]],
            atol=1e-5 if x64 else 0.002,
            rtol=0,
        )


def test_multigroup_poisson_fit_estimates_scale_near_exact_integrated_posterior():
    counts = np.array(
        [
            [1, 2, 1, 0],
            [3, 2, 4, 3],
            [6, 5, 4, 5],
            [9, 7, 10, 8],
            [13, 16, 12, 15],
            [2, 4, 3, 2],
            [5, 8, 6, 7],
            [20, 23, 19, 21],
        ],
        dtype=float,
    )
    sums = counts.sum(axis=1)

    def exact_marginal(outer):
        mu, log_tau = outer
        tau = math.exp(log_tau)
        value = mu**2 / 8 + (log_tau - math.log(0.5)) ** 2 / (2 * 0.7**2)
        for total in sums:

            def joint(b, total=total):
                return b * b / (2 * tau * tau) + 4 * math.exp(mu + b) - total * (mu + b)

            mode = brentq(
                lambda b, total=total: b / (tau * tau) + 4 * math.exp(mu + b) - total,
                -40,
                40,
            )
            at_mode = joint(mode)
            # Conditional curvature >= 1/tau**2 bounds these omitted tails.
            integral, error = quad(
                lambda b, at_mode=at_mode, joint=joint: math.exp(at_mode - joint(b)),
                mode - 12 * tau,
                mode + 12 * tau,
                epsabs=1e-11,
                epsrel=1e-11,
            )
            assert error < 1e-8
            value += at_mode + log_tau - math.log(integral)
        return value

    reference = minimize(
        exact_marginal,
        [1.0, math.log(0.5)],
        method="Powell",
        bounds=[(-3, 5), (-3, 2)],
        options={"xtol": 1e-8, "ftol": 1e-10},
    )
    assert reference.success
    with jax.enable_x64():
        mu = lsl.Var.new_param(1.0, lsl.Dist(tfd.Normal, 0.0, 2.0), name="mu")
        tau = lsl.Var.new_param(
            0.5,
            lsl.Dist(tfd.LogNormal, math.log(0.5), 0.7),
            bijector=tfb.Exp(),
            name="tau",
        )
        b = lsl.Var.new_param(jnp.zeros(8), lsl.Dist(tfd.Normal, 0.0, tau), name="b")
        groups = jnp.repeat(jnp.arange(8), 4)
        eta = lsl.Var.new_calc(lambda mu, b: mu + b[groups], mu, b, name="eta")
        y = lsl.Var.new_obs(
            jnp.asarray(counts.ravel()), lsl.Dist(tfd.Poisson, log_rate=eta), name="y"
        )
        model = lsl.Model([y], to_float32=False)
        loss = opt.LaplaceLoss(model, latent=["b"])
        result = opt.LieselOptim(
            model,
            loss=loss,
            optimizers="lbfgs",
            loss_monitor="train_full_data",
            stopper=opt.Stopper(epochs=60, patience=10, rtol=1e-10),
            show_progress=False,
        ).fit()
        assert result.status in ("early_stopping", "max_epochs")
        fitted = model.predict(result.position_min_monitor, predict=["mu", "tau"])
        # Fixed before running: permit Laplace approximation error, not merely
        # numerical optimizer error, relative to exactly integrated posterior.
        np.testing.assert_allclose(
            fitted["tau"], math.exp(reference.x[1]), atol=0.02, rtol=0
        )
        np.testing.assert_allclose(fitted["mu"], reference.x[0], atol=0.02, rtol=0)
        assert loss.approximate_joint_posterior(result).valid
