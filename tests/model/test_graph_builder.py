import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.numpy.distributions as nd

import liesel.model.model as lmodel
import liesel.model.nodes as lnodes


def test_transform() -> None:
    val = jnp.array((0.1, 1.0, 2.0))
    rate = 0.5
    d_true = lnodes.Data(True, "true")
    dist = lnodes.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lnodes.Var(val, dist)
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

    var_trans = lmodel.GraphBuilder().transform(var)

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
    rate = lnodes.Calc(lambda x, y: x + y, 0.5, 1.0)
    d_true = lnodes.Data(True, "true")
    dist = lnodes.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lnodes.Var(val, dist)

    # the bijector is a softplus transformation
    # x = softplus(y) = log(exp(y) + 1) # forward
    # y = softplus_inv(x) = log(exp(x) - 1) # backward

    # def forward(x):
    #     return jnp.log1p(jnp.exp(x))

    def backward(x):
        return jnp.log(jnp.exp(x) - 1)

    def forward_dx(x):
        return jnp.reciprocal(1 + jnp.exp(-x))

    var_trans = lmodel.GraphBuilder().transform(var)

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
    rate = lnodes.Calc(lambda x, y: x + y, 0.5, 1.0)
    d_true = lnodes.Data(True, "true")
    dist = lnodes.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lnodes.Var(val, dist)

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
    var_trans = lmodel.GraphBuilder().transform(var)

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
    d_true = lnodes.Data(True, "true")
    dist = lnodes.Dist(tfp.distributions.Exponential, rate, validate_args=d_true)
    dist.per_obs = True
    var = lnodes.Var(val, dist)
    var.update()
    assert var.log_prob == pytest.approx(
        tfp.distributions.Exponential(rate).log_prob(val)
    )

    var_trans = lmodel.GraphBuilder().transform(var, tfp.bijectors.Exp)

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
    dist = lnodes.Dist(tfp.distributions.Exponential, rate)
    var = lnodes.Var(val, dist)
    var_trans = lmodel.GraphBuilder().transform(var, tfp.bijectors.Softplus, sp_param)

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
    dist = lnodes.Dist(tfp.distributions.Exponential, rate)
    var = lnodes.Var(val, dist)
    var_trans = lmodel.GraphBuilder().transform(
        var, tfp.bijectors.Softplus, hinge_softness=sp_param
    )

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


def test_transform_twice() -> None:
    dist = lnodes.Dist(tfp.distributions.Exponential, 1.0)
    var = lnodes.Var(1.0, dist)
    lmodel.GraphBuilder().transform(var)

    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().transform(var)


def test_transform_no_bijector() -> None:
    dist = lnodes.Dist(tfp.distributions.Poisson, 1.0)
    var = lnodes.Var(1, dist)
    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().transform(var)


@pytest.mark.xfail
def test_transform_no_bijector_delayed_check() -> None:
    lamb = lnodes.Calc(lambda x, y: x + y, 1.0, 1.0)
    dist = lnodes.Dist(tfp.distributions.Poisson, lamb)
    var = lnodes.Var(1.0, dist)
    var_trans = lmodel.GraphBuilder().transform(var)

    lamb.update()
    with pytest.raises(RuntimeError):
        var_trans.update()


def test_transform_no_dist() -> None:
    var = lnodes.Var(1, None)
    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().transform(var)


def test_transform_weak() -> None:
    dist = lnodes.Dist(tfp.distributions.Poisson, 1.0)
    x = lnodes.Calc(lambda x, y: x + y, 1.0, 1.0)
    var = lnodes.Var(x, dist)
    assert var.weak
    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().transform(var)


def test_groups() -> None:
    v1 = lnodes.Var(0.0, name="v1")
    g1 = lnodes.Group("g1", var1=v1)
    gb = lmodel.GraphBuilder().add_groups(g1)
    assert v1 in gb.vars


def test_add_group_with_duplicate_name() -> None:
    v1 = lnodes.Var(0.0, name="v1")
    v2 = lnodes.Var(0.0, name="v2")
    g1 = lnodes.Group("g1", var1=v1)
    g2 = lnodes.Group("g1", var1=v2)
    with pytest.raises(RuntimeError):
        lmodel.GraphBuilder().add_groups(g1, g2)
