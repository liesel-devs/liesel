"""The built-in objective must agree structurally with the model probability."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


@pytest.mark.parametrize(
    "kind", ["likelihood", "probability", "prior", "latent", "standalone"]
)
def test_default_loss_rejects_unsupported_model_factors(kind):
    loc = lsl.Var.new_param(0.0, name="loc")
    dist = lsl.Dist(tfd.Normal, loc=loc, scale=1.0)
    y = lsl.Var.new_obs(jnp.arange(4.0), dist, name="y")
    builder = lsl.GraphBuilder().add(y)
    if kind == "likelihood":
        weighted = lsl.Calc(
            lambda values: (values * jnp.array([1.0, 1.0, 4.0, 4.0])).sum(), dist
        )
        builder.log_lik_node = weighted
        builder.log_prob_node = weighted
    elif kind == "probability":
        # Value AND gradient agree at loc=0; a one-point numerical check misses this.
        builder.log_prob_node = lsl.Calc(
            lambda values, x: values.sum() - 10 * x**2, dist, loc
        )
    elif kind == "prior":
        builder.log_prior_node = lsl.Calc(lambda x: -10 * x**2, loc)
    elif kind == "latent":
        builder.add(
            lsl.Var(3.0, lsl.Dist(tfd.Normal, loc=loc, scale=0.1), name="latent")
        )
    else:
        extra = lsl.Dist(tfd.Normal, loc=loc, scale=0.1, _name="extra")
        extra.at = lsl.Value(3.0)
        builder.add(extra)
    model = builder.build_model()
    split = opt.PositionSplit.from_model(model, infer_sample_sizes=False)
    with pytest.raises(ValueError, match="cannot decompose.*custom Loss"):
        opt.NegLogProbLoss(model, split)
    with pytest.raises(ValueError, match="cannot decompose.*custom Loss"):
        opt.LieselOptim(
            model, split=split, optimizers="lbfgs", loss_monitor="train_full_data"
        )
    if kind == "likelihood":
        with pytest.raises(ValueError, match="custom Loss"):
            opt.LieselOptim(model, optimizers="lbfgs", loss_monitor="train_full_data")
    else:
        with pytest.raises(ValueError, match="cannot decompose.*custom Loss"):
            opt.LieselOptim(model, optimizers="lbfgs", loss_monitor="train_full_data")

    class ModelLoss(opt.LossMixin):
        def __init__(self):
            self.split = split

        def position(self, position_keys):
            return model.extract_position(position_keys)

        def loss_train_batched(self, params, carry):
            return -model.update_state(params, carry.model_state)[
                "_model_log_prob"
            ].value

        loss_train = loss_train_batched
        loss_monitor = loss_train_batched

    engine = opt.LieselOptim(
        model,
        loss=ModelLoss(),
        optimizers=optax.sgd(0.001),
        loss_monitor="train_full_data",
        show_progress=False,
        stopper=opt.Stopper(epochs=1, patience=1),
    ).build_engine()
    carry = engine._init_carry(1)
    value = engine.loss.loss_train({"loc": jnp.array(1.0)}, carry)
    assert (
        value
        == -model.update_state({"loc": jnp.array(1.0)}, model.state)[
            "_model_log_prob"
        ].value
    )
    assert engine.fit().n_epochs == 1


@pytest.mark.parametrize("per_obs", [True, False])
def test_supported_objective_and_gradient_match_model(per_obs):
    scale = lsl.Var.new_param(
        1.0, lsl.Dist(tfd.LogNormal, 0.0, 1.0), bijector=tfb.Exp(), name="scale"
    )
    ys = [
        lsl.Var.new_obs(
            jnp.arange(float(n)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=scale),
            name=f"y{n}",
        )
        for n in (4, 6)
    ]
    for var in ys:
        assert var.dist_node is not None
        var.dist_node.per_obs = per_obs
    model = lsl.Model(ys)
    split = opt.PositionSplit.from_model(
        model, multi_size="manager", infer_sample_sizes=False
    )
    engine = opt.LieselOptim(
        model,
        split=split,
        optimizers="lbfgs",
        scale_loss=False,
        loss_monitor="train_full_data",
        show_progress=False,
    ).build_engine()
    carry = engine._init_carry(1)
    for value in (0.0, 0.7):
        key = next(iter(model.parameters))
        position = {key: jnp.asarray(value, dtype=model.vars[key].value.dtype)}
        actual, grad = jax.value_and_grad(engine.loss.loss_train)(position, carry)
        expected, expected_grad = jax.value_and_grad(
            lambda p: -model.update_state(p, model.state)["_model_log_prob"].value
        )(position)
        assert jnp.allclose(actual, expected)
        assert all(
            jnp.allclose(a, b)
            for a, b in zip(
                jax.tree.leaves(grad), jax.tree.leaves(expected_grad), strict=True
            )
        )


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_weak_parameter_prior_requires_explicit_strong_optimization_keys(optimizer):
    source = lsl.Var(0.0, name="source")
    location = lsl.Var.new_calc(
        lambda value: 2 * value,
        source,
        dist=lsl.Dist(tfd.Normal, 0.0, 1.0),
        name="location",
    )
    location.parameter = True
    response = lsl.Var.new_obs(
        jnp.array([1.0, 2.0, 3.0]),
        lsl.Dist(tfd.Normal, location, 1.0),
        name="response",
    )
    model = lsl.Model([response])
    automatic = optax.adam(0.05) if optimizer == "adam" else "lbfgs"
    with pytest.raises(ValueError, match="weak parameters.*Explicitly name.*strong"):
        opt.LieselOptim(model, optimizers=automatic, loss_monitor="train_full_data")
    explicit = (
        opt.Optimizer(["source"], optax.adam(0.05))
        if optimizer == "adam"
        else opt.LBFGS(["source"])
    )
    engine = opt.LieselOptim(
        model,
        optimizers=[explicit],
        loss_monitor="train_full_data",
        scale_loss=False,
        stopper=opt.Stopper(epochs=150, patience=150),
        show_progress=False,
    ).build_engine()
    carry = engine._init_carry(150)
    for value in (0.0, 0.5):
        position = {
            "source": jnp.asarray(
                value, dtype=jnp.asarray(model.vars["source"].value).dtype
            )
        }
        actual, grad = jax.value_and_grad(engine.loss.loss_train)(position, carry)
        expected, ref_grad = jax.value_and_grad(
            lambda p: -model.update_state(p, model.state)["_model_log_prob"].value
        )(position)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        np.testing.assert_allclose(grad["source"], ref_grad["source"], rtol=1e-6)
    # Normal prior precision 1 plus three observations: location MAP = 6 / 4.
    np.testing.assert_allclose(engine.fit().position_final["source"], 0.75, atol=5e-4)
