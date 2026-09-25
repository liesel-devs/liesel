"""Construction guards for the supported variational inference contract."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
import liesel.optim as opt


def target_model():
    z = lsl.Var.new_param(0.0, name="z")
    y = lsl.Var.new_obs(jnp.array([1.0, 2.0]), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    return lsl.Model(y)


def test_explicit_loss_must_belong_to_supplied_target():
    model = target_model()
    loss = opt.NegElboLoss.mvn_diag(model)
    with pytest.raises(ValueError, match="loss.p.*model"):
        opt.LieselVI(
            target_model(),
            loss=loss,
            optimizers=optax.adam(0.01),
            loss_monitor="train_full_data",
        )
    opt.LieselVI(
        model, loss=loss, optimizers=optax.adam(0.01), loss_monitor="train_full_data"
    )


def test_computed_q_parameters_require_strong_free_inputs():
    raw = lsl.Var.new_value(0.0, name="raw")
    scale = lsl.Var.new_calc(jnp.exp, raw, name="scale")
    scale.parameter = True
    z = lsl.Var.new_obs(0.0, lsl.Dist(tfd.Normal, 0.0, scale), name="z")
    q = lsl.Model(z)
    with pytest.raises(ValueError, match="scale.*strong"):
        opt.NegElboLoss(target_model(), q)
    q.vars["scale"].parameter = False
    q.vars["raw"].parameter = True
    loss = opt.NegElboLoss(target_model(), q)
    assert list(loss.q.parameters) == ["raw"]


@pytest.mark.parametrize("kind", ["discrete", "unclassified"])
def test_rejects_unsupported_sampled_q_distributions(kind):
    if kind == "discrete":
        z = lsl.Var.new_obs(0, lsl.Dist(tfd.Bernoulli, probs=0.5), name="z")
    else:
        z = lsl.Var(0.0, lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
    with pytest.raises(ValueError, match="observed|fully reparameterized"):
        opt.NegElboLoss(target_model(), lsl.Model(z))


@pytest.mark.parametrize("family", ["normal", "mvn_diag", "mvn_tril"])
def test_gaussian_builders_reject_nonfinite_locations(family):
    with pytest.raises(ValueError, match="finite"):
        getattr(opt.VDist(["z"], target_model()), family)(loc=jnp.nan)


@pytest.mark.parametrize(
    "family,argument",
    [("normal", "scale"), ("mvn_diag", "scale_diag"), ("mvn_tril", "scale_tril")],
)
@pytest.mark.parametrize("value", [0.0, jnp.nan, jnp.inf])
def test_gaussian_builders_reject_unusable_scales(family, argument, value):
    with pytest.raises(ValueError, match="scale|factor"):
        getattr(opt.VDist(["z"], target_model()), family)(
            **{argument: value, f"{argument}_bijector": None}
        )


@pytest.mark.parametrize(
    "family,argument", [("normal", "scale"), ("mvn_diag", "scale_diag")]
)
def test_diagonal_scales_must_be_positive_without_transform(family, argument):
    with pytest.raises(ValueError, match="positive"):
        getattr(opt.VDist(["z"], target_model()), family)(
            **{argument: -1.0, f"{argument}_bijector": None}
        )


def test_dense_factor_must_be_lower_triangular():
    model = lsl.Model(lsl.Var.new_param(jnp.zeros(2), name="z"))
    with pytest.raises(ValueError, match="lower triangular"):
        opt.VDist(["z"], model).mvn_tril(
            scale_tril=jnp.array([[1.0, 0.2], [0.0, 1.0]]), scale_tril_bijector=None
        )


@pytest.mark.parametrize(
    "family,argument",
    [("normal", "scale"), ("mvn_diag", "scale_diag"), ("mvn_tril", "scale_tril")],
)
def test_gaussian_scale_must_be_representable_by_bijector(family, argument):
    with pytest.raises(ValueError, match="bijector"):
        getattr(opt.VDist(["z"], target_model()), family)(**{argument: 1e-12})


@pytest.mark.parametrize("bijector", [None, tfb.Identity()])
def test_dense_negative_diagonal_allowed_with_compatible_bijector(bijector):
    model = lsl.Model(lsl.Var.new_param(jnp.zeros(2), name="z"))
    factor = jnp.array([[-1.0, 0.0], [0.5, -2.0]])
    q = (
        opt.VDist(["z"], model)
        .mvn_tril(scale_tril=factor, scale_tril_bijector=bijector)
        .build()
    )
    assert q.var is not None and q.var.dist_node is not None
    np.testing.assert_allclose(
        q.var.dist_node.init_dist().covariance(), [[1.0, -0.5], [-0.5, 4.25]]
    )
    with pytest.raises(ValueError, match="bijector"):
        opt.VDist(["z"], model).mvn_tril(scale_tril=factor)


@pytest.mark.parametrize("family", ["normal", "mvn_diag", "mvn_tril"])
def test_laplace_initializer_rejects_nonfinite_curvature(family):
    z = lsl.Var.new_param(0.0, name="z")
    scale = lsl.Var.new_calc(lambda value: jnp.sqrt(value), z, name="scale")
    y = lsl.Var.new_obs(jnp.zeros(2), lsl.Dist(tfd.Normal, z, scale), name="y")
    model = lsl.Model(y)
    argument = {"normal": "scale", "mvn_diag": "scale_diag", "mvn_tril": "scale_tril"}[
        family
    ]
    with pytest.raises(ValueError, match="curvature"):
        getattr(opt.VDist(["z"], model), family)(**{argument: "laplace"})


@pytest.mark.parametrize(
    "family,argument",
    [("normal", "scale"), ("mvn_diag", "scale_diag"), ("mvn_tril", "scale_tril")],
)
@pytest.mark.parametrize("laplace", [False, True])
def test_scalar_location_remains_one_shared_parameter(family, argument, laplace):
    z = lsl.Var.new_param(jnp.zeros(2), lsl.Dist(tfd.Normal, 0.0, 1.0), name="z")
    y = lsl.Var.new_obs(jnp.zeros(2), lsl.Dist(tfd.Normal, z, 1.0), name="y")
    model = lsl.Model(y)
    q = getattr(opt.VDist(["z"], model), family)(
        loc=0.25, **{argument: "laplace" if laplace else 0.5}
    ).build()
    assert q.q is not None and q.var is not None and q.var.dist_node is not None
    assert q.q.vars["(z)_loc"].value.shape == ()
    distribution = q.var.dist_node.init_dist()
    np.testing.assert_allclose(distribution.mean(), [0.25, 0.25])
    np.testing.assert_allclose(
        distribution.stddev(), jnp.sqrt(0.5) if laplace else 0.5, atol=2e-6
    )
    assert q.sample(jax.random.key(42), (5,))["z"].shape == (5, 2)


def test_custom_scale_bijector_must_have_finite_inverse():
    with pytest.raises(ValueError, match="bijector"):
        opt.VDist(["z"], target_model()).normal(
            scale=0.5, scale_bijector=tfb.Softplus(low=1.0)
        )


@pytest.mark.parametrize(
    "family,argument",
    [("normal", "scale"), ("mvn_diag", "scale_diag"), ("mvn_tril", "scale_tril")],
)
def test_gaussian_initial_shapes_must_match_governed_parameters(family, argument):
    model = lsl.Model(lsl.Var.new_param(jnp.zeros(2), name="z"))
    with pytest.raises(ValueError):
        getattr(opt.VDist(["z"], model), family)(loc=jnp.ones((2, 2)))
    wrong_scale = jnp.eye(3) if family == "mvn_tril" else jnp.ones(3)
    with pytest.raises(ValueError):
        getattr(opt.VDist(["z"], model), family)(**{argument: wrong_scale})
