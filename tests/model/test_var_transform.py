import pickle

import dill
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.numpy.distributions as nd

import liesel.model as lsl


class TestVarTransform:
    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_distribtution_inst(self, name) -> None:
        """
        Tests transformation of a weak var with distribution when the bijector is passed
        as an instance.
        """
        x = lsl.Var.new_value(jnp.linspace(0.1, 2, 5), name="all_x")
        batch_index = lsl.Var.new_value(1, name="index")

        x_batched = lsl.Var(
            value=lsl.Calc(lambda i, x: x[i], batch_index, x),
            distribution=lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="x_batched",
        )

        x_batched_transformed = x_batched.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert x_batched_transformed.name == f"{x_batched.name}_transformed"
        else:
            assert x_batched_transformed.name == name

        assert x_batched_transformed.value == pytest.approx(jnp.log(x.value[1]))

        batch_index.value = 2
        x_batched_transformed.update()
        x_batched.update()
        assert x_batched_transformed.value == pytest.approx(jnp.log(x.value[2]))

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_distribtution_class(self, name) -> None:
        """
        Tests transformation of a weak var with distribution when the bijector is passed
        as a class.
        """
        x = lsl.Var.new_value(jnp.linspace(0.1, 2, 5), name="all_x")
        batch_index = lsl.Var.new_value(1, name="index")

        x_batched = lsl.Var(
            value=lsl.Calc(lambda i, x: x[i], i=batch_index, x=x),
            distribution=lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="x_batched",
        )

        x_batched_transformed = x_batched.transform(
            tfp.bijectors.Scale, scale=2.0, name=name
        )

        if name is None:
            assert x_batched_transformed.name == f"{x_batched.name}_transformed"
        else:
            assert x_batched_transformed.name == name

        assert x_batched_transformed.value == pytest.approx(x.value[1] / 2.0)

        batch_index.value = 2
        x_batched_transformed.update()
        x_batched.update()
        assert x_batched_transformed.value == pytest.approx(x.value[2] / 2.0)

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_bijector_instance(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        tau_sqrt = lsl.Var.new_calc(jnp.sqrt, tau)
        log_tau_sqrt = tau_sqrt.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert log_tau_sqrt.name == f"{tau_sqrt.name}_transformed"
        else:
            assert log_tau_sqrt.name == name

        assert tau.value == pytest.approx(10.0)
        assert tau_sqrt.value == pytest.approx(jnp.sqrt(10.0))
        assert log_tau_sqrt.value == pytest.approx(jnp.log(jnp.sqrt(10.0)))

        assert tau.strong
        assert tau_sqrt.weak
        assert log_tau_sqrt.weak
        assert tau.parameter
        assert not log_tau_sqrt.parameter
        assert not tau_sqrt.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_weak_var_with_bijector_class(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        tau_sqrt = lsl.Var.new_calc(jnp.sqrt, tau)

        scale = lsl.Var.new_param(2.0, name="bijector_scale")
        scaled_tau_sqrt = tau_sqrt.transform(
            tfp.bijectors.Scale, scale=scale, name=name
        )

        if name is None:
            assert scaled_tau_sqrt.name == f"{tau_sqrt.name}_transformed"
        else:
            assert scaled_tau_sqrt.name == name

        assert tau.value == pytest.approx(10.0)
        assert tau_sqrt.value == pytest.approx(jnp.sqrt(10.0))
        assert scaled_tau_sqrt.value == pytest.approx(jnp.sqrt(10.0) / 2)

        assert tau.strong
        assert tau_sqrt.weak
        assert scaled_tau_sqrt.weak
        assert tau.parameter
        assert not scaled_tau_sqrt.parameter
        assert not tau_sqrt.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_without_dist_with_bijector_instance(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")
        log_tau = tau.transform(tfp.bijectors.Exp(), name=name)

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(jnp.log(10.0))

        assert tau.weak
        assert log_tau.strong
        assert not tau.parameter
        assert log_tau.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_without_dist_with_bijector_class(self, name) -> None:
        tau = lsl.Var.new_param(10.0, name="tau")

        scale = lsl.Var.new_param(2.0, name="bijector_scale")
        log_tau = tau.transform(tfp.bijectors.Scale, scale=scale, name=name)

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(5.0)

        assert tau.weak
        assert log_tau.strong
        assert not tau.parameter
        assert log_tau.parameter

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_instance(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        log_tau = tau.transform(tfp.bijectors.Exp(), name=name)
        tau.update()

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

    def test_transform_class_no_args(self) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        with pytest.raises(ValueError):
            tau.transform(tfp.bijectors.Exp)

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_class_with_args(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        transformed_tau = tau.transform(
            tfp.bijectors.Softplus, hinge_softness=lsl.Var(0.9), name=name
        )
        tau.update()

        if name is None:
            assert transformed_tau.name == f"{tau.name}_transformed"
        else:
            assert transformed_tau.name == name

        bijector = tfp.bijectors.Softplus(hinge_softness=0.9)

        assert tau.weak
        assert not transformed_tau.weak

        assert tau.value == pytest.approx(bijector.forward(transformed_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert transformed_tau.value == pytest.approx(bijector.inverse(10.0))

    @pytest.mark.parametrize("name", ("newname", None))
    def test_transform_default(self, name) -> None:
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        log_tau = tau.transform(name=name)
        tau.update()

        if name is None:
            assert log_tau.name == f"{tau.name}_transformed"
        else:
            assert log_tau.name == name

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

    def test_pickle_model_with_transformed_var(self, tmp_path):
        prior = lsl.Dist(tfp.distributions.HalfCauchy, loc=0.0, scale=25.0)
        tau = lsl.Var(10.0, prior, name="tau")
        _ = tau.transform()

        model = lsl.Model([tau])

        with pytest.raises(AttributeError):
            with open(tmp_path / "model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open(tmp_path / "model.pkl", "rb") as f:
                model2 = pickle.load(f)

        with open(tmp_path / "model.pkl", "wb") as f:
            dill.dump(model, f)

        with open(tmp_path / "model.pkl", "rb") as f:
            model2 = dill.load(f)

        assert len(model2.vars) == len(model.vars)
        assert len(model2.nodes) == len(model.nodes)
        assert model2.log_prob == pytest.approx(model.log_prob)


def test_transform() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    rate = 0.5
    d_true = lsl.Value(True, "true")
    dist = lsl.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lsl.Var(val, dist)
    var.parameter = True
    var.update()
    assert var.log_prob == pytest.approx(
        tfp.distributions.Exponential(rate).log_prob(val)
    )

    # the bijector is a softplus transformation
    # x = softplus(y) = log(exp(y) + 1) # forward
    # y = softplus_inv(x) = log(exp(x) - 1) # backward

    # def forward(x):
    #     return jnp.log1p(jnp.exp(x))

    def backward(x):
        return jnp.log(jnp.exp(x) - 1)

    def forward_dx(x):
        return jnp.reciprocal(1 + jnp.exp(-x))

    var_trans = var.transform()

    assert var.weak
    assert var.dist_node is None

    assert var_trans.strong
    assert var_trans.dist_node is not None

    assert isinstance(var_trans.dist_node.init_dist(), tfp.distributions.Distribution)
    assert not isinstance(var_trans.dist_node.init_dist(), nd.Distribution)

    var_trans.update()
    var.update()

    trans_value = backward(val)
    trans_log_prob = tfp.distributions.Exponential(rate, validate_args=True).log_prob(
        val
    ) + jnp.log(jnp.abs(forward_dx(trans_value)))

    assert var_trans.value == pytest.approx(trans_value)
    assert var.log_prob == 0.0

    assert var.value == pytest.approx(val)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)

    # parameter flag has moved
    assert not var.parameter
    assert var_trans


def test_transform_with_bijector_instance() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    rate = 0.5
    d_true = lsl.Data(True, "true")
    dist = lsl.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lsl.Var(val, dist)
    var.parameter = True
    var.update()
    assert var.log_prob == pytest.approx(
        tfp.distributions.Exponential(rate).log_prob(val)
    )

    # the bijector is a softplus transformation
    # x = softplus(y) = log(exp(y) + 1) # forward
    # y = softplus_inv(x) = log(exp(x) - 1) # backward

    # def forward(x):
    #     return jnp.log1p(jnp.exp(x))

    def backward(x):
        return jnp.log(jnp.exp(x) - 1)

    def forward_dx(x):
        return jnp.reciprocal(1 + jnp.exp(-x))

    var_trans = var.transform(tfp.bijectors.Softplus())

    assert var.weak
    assert var.dist_node is None

    assert var_trans.strong
    assert var_trans.dist_node is not None

    assert isinstance(var_trans.dist_node.init_dist(), tfp.distributions.Distribution)
    assert not isinstance(var_trans.dist_node.init_dist(), nd.Distribution)

    var_trans.update()
    var.update()

    trans_value = backward(val)
    trans_log_prob = tfp.distributions.Exponential(rate, validate_args=True).log_prob(
        val
    ) + jnp.log(jnp.abs(forward_dx(trans_value)))

    assert var_trans.value == pytest.approx(trans_value)
    assert var.log_prob == 0.0

    assert var.value == pytest.approx(val)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)

    # parameter flag has moved
    assert not var.parameter
    assert var_trans


@pytest.mark.xfail
def test_transform_with_param_outdated() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    real_rate = 1.5
    rate = lsl.Calc(lambda x, y: x + y, 0.5, 1.0)
    d_true = lsl.Value(True, "true")
    dist = lsl.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lsl.Var(val, dist)

    # the bijector is a softplus transformation
    # x = softplus(y) = log(exp(y) + 1) # forward
    # y = softplus_inv(x) = log(exp(x) - 1) # backward

    # def forward(x):
    #     return jnp.log1p(jnp.exp(x))

    def backward(x):
        return jnp.log(jnp.exp(x) - 1)

    def forward_dx(x):
        return jnp.reciprocal(1 + jnp.exp(-x))

    var_trans = var.transform()

    assert var.weak
    assert var.dist_node is None

    assert var_trans.strong
    assert var_trans.dist_node is not None

    # var_trans cannot be set since rate is outdated
    assert var_trans.value is None
    var_trans.value = backward(val)

    rate.update()
    var_trans.update()
    var.update()

    trans_value = backward(val)
    trans_log_prob = tfp.distributions.Exponential(
        real_rate, validate_args=True
    ).log_prob(val) + jnp.log(jnp.abs(forward_dx(trans_value)))

    assert var_trans.value == pytest.approx(trans_value)
    assert var.log_prob == 0.0

    assert var.value == pytest.approx(val)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)


def test_transform_with_param() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    real_rate = 1.5
    rate = lsl.Calc(lambda x, y: x + y, 0.5, 1.0)
    d_true = lsl.Value(True, "true")
    dist = lsl.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lsl.Var(val, dist)

    # the bijector is a softplus transformation
    # x = softplus(y) = log(exp(y) + 1) # forward
    # y = softplus_inv(x) = log(exp(x) - 1) # backward

    # def forward(x):
    #     return jnp.log1p(jnp.exp(x))

    def backward(x):
        return jnp.log(jnp.exp(x) - 1)

    def forward_dx(x):
        return jnp.reciprocal(1 + jnp.exp(-x))

    rate.update()

    # rate is not outdated. transform should also set value
    var_trans = var.transform()

    assert var.weak
    assert var.dist_node is None

    assert var_trans.strong
    assert var_trans.dist_node is not None

    # var_trans cannot be set since rate is outdated
    assert var_trans.value is not None

    rate.update()
    var_trans.update()
    var.update()

    trans_value = backward(val)
    trans_log_prob = tfp.distributions.Exponential(
        real_rate, validate_args=True
    ).log_prob(val) + jnp.log(jnp.abs(forward_dx(trans_value)))

    assert var_trans.value == pytest.approx(trans_value)
    assert var.log_prob == 0.0

    assert var.value == pytest.approx(val)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)


def test_transform_user() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    rate = 0.5
    d_true = lsl.Value(True, "true")
    dist = lsl.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lsl.Var(val, dist)
    var.update()
    assert var.log_prob == pytest.approx(
        tfp.distributions.Exponential(rate).log_prob(val)
    )

    var_trans = var.transform(tfp.bijectors.Exp())

    assert var.weak
    assert var.dist_node is None

    assert var_trans.strong
    assert var_trans.dist_node is not None

    var_trans.update()
    var.update()

    assert var.log_prob == 0.0
    assert var.value == pytest.approx(val)

    trans_value = jnp.log(val)
    trans_log_prob = (
        tfp.distributions.Exponential(rate, validate_args=True).log_prob(val)
        + trans_value
    )
    assert var_trans.value == pytest.approx(trans_value)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)


def test_transform_user_bijector_with_input() -> None:
    # the bijector is a softplus transformation
    # x = softplus(y, z) = z * log(exp(y/z) + 1) # forward
    # y = softplus_inv(x, z) = z * log(exp(x/z) - 1) # backward

    def backward(x, z):
        return z * jnp.log(jnp.exp(x / z) - 1)

    def forward_dx(x, z):
        return jnp.reciprocal(1 + jnp.exp(-x / z))

    val = jnp.array((0.1, 1.0, 2.0))
    rate = 0.5
    sp_param = 2.0

    # positional argument
    dist = lsl.Dist(tfp.distributions.Exponential, rate)
    var = lsl.Var(val, dist)
    var_trans = var.transform(tfp.bijectors.Softplus, sp_param)

    var_trans.update()
    var.update()

    trans_value = backward(val, sp_param)
    trans_log_prob = tfp.distributions.Exponential(rate).log_prob(val) + jnp.log(
        jnp.abs(forward_dx(trans_value, sp_param))
    )
    assert var.log_prob == 0.0
    assert var.value == pytest.approx(val)
    assert var_trans.value == pytest.approx(trans_value)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)

    # keyword argument
    dist = lsl.Dist(tfp.distributions.Exponential, rate)
    var = lsl.Var(val, dist)
    var_trans = var.transform(tfp.bijectors.Softplus, hinge_softness=sp_param)

    var_trans.update()
    var.update()

    trans_value = backward(val, sp_param)
    trans_log_prob = tfp.distributions.Exponential(rate).log_prob(val) + jnp.log(
        jnp.abs(forward_dx(trans_value, sp_param))
    )
    assert var.log_prob == 0.0
    assert var.value == pytest.approx(val)
    assert var_trans.value == pytest.approx(trans_value)
    assert var_trans.log_prob == pytest.approx(trans_log_prob)


def test_transform_twice_with_no_bijector_given() -> None:
    dist = lsl.Dist(tfp.distributions.Exponential, 1.0)
    var = lsl.Var(1.0, dist)
    var.transform()

    # fails when trying to use a default bijector, because a weak variable does not
    # have a default bijector
    with pytest.raises(RuntimeError):
        var.transform()


def test_transform_twice() -> None:
    """
    While this may not always be a good idea, it is allowed.
    """
    dist = lsl.Dist(tfp.distributions.Exponential, 1.0)
    var = lsl.Var(1.0, dist)
    var.transform(tfp.bijectors.Exp())
    var.transform(tfp.bijectors.Exp())


def test_transform_no_bijector() -> None:
    dist = lsl.Dist(tfp.distributions.Poisson, 1.0)
    var = lsl.Var(1, dist)
    with pytest.raises(RuntimeError):
        var.transform()


@pytest.mark.xfail
def test_transform_no_bijector_delayed_check() -> None:
    lamb = lsl.Calc(lambda x, y: x + y, 1.0, 1.0)
    dist = lsl.Dist(tfp.distributions.Poisson, lamb)
    var = lsl.Var(1.0, dist)
    var_trans = var.transform()

    lamb.update()
    with pytest.raises(RuntimeError):
        var_trans.update()


def test_transform_no_dist() -> None:
    var = lsl.Var(1, None)
    with pytest.raises(RuntimeError):
        var.transform()


def test_t1() -> None:
    assert tfp.bijectors.Exp().forward(1.0) == jnp.exp(1.0)
