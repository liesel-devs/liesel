import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel.experimental.optim as opt
import liesel.model as lsl


def _laplace_model():
    loc = lsl.Var.new_param(
        jnp.array(0.0),
        lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
        name="loc",
    )
    y = lsl.Var.new_obs(
        jnp.zeros(2),
        lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
        name="y",
    )
    return lsl.Model([y])


def _two_branch_param_model():
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    y1 = lsl.Var.new_obs(
        jnp.arange(8.0),
        lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
        name="y1",
    )
    y2 = lsl.Var.new_obs(
        jnp.arange(5.0),
        lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
        name="y2",
    )
    return lsl.Model([y1, y2])


def _two_parameter_model():
    alpha = lsl.Var.new_param(jnp.array(0.0), name="alpha")
    beta = lsl.Var.new_param(jnp.array([0.0, 1.0]), name="beta")
    y = lsl.Var.new_obs(
        jnp.zeros(2),
        lsl.Dist(tfp.distributions.Normal, loc=alpha, scale=1.0),
        name="y",
    )
    return lsl.Model([y, beta])


def test_neg_elbo_from_vdist_scale_uses_total_branch_training_size():
    model = _two_branch_param_model()
    split = opt.PositionSplitManager.from_model(model, position_keys=["y1", "y2"])
    vdist = opt.VDist(["loc"], model).mvn_diag().build()

    elbo = opt.NegElboLoss.from_vdist(vdist, split, scale=True)

    assert elbo.scalar == sum(split.n_trains)


def test_neg_elbo_mvn_diag_forwards_custom_initialization():
    p = _laplace_model()

    elbo = opt.NegElboLoss.mvn_diag(
        p,
        loc=jnp.array([1.5]),
        scale_diag=0.2,
        scale_diag_bijector=None,
    )

    params = elbo.vdist.var.dist_node.kwinputs
    assert jnp.allclose(params["loc"].value, jnp.array([1.5]))
    assert jnp.allclose(params["scale_diag"].value, jnp.array([0.2]))


def test_neg_elbo_mvn_tril_forwards_laplace_initialization():
    p = _laplace_model()

    elbo = opt.NegElboLoss.mvn_tril(
        p,
        scale_tril="laplace",
        scale_tril_bijector=None,
    )

    scale_tril = elbo.vdist.var.dist_node.kwinputs["scale_tril"].value
    assert jnp.allclose(scale_tril, jnp.sqrt(jnp.array([[1.0 / 3.0]])), rtol=1e-5)


def test_neg_elbo_mvn_blocked_forwards_shared_scale_initialization():
    p = _two_parameter_model()

    elbo = opt.NegElboLoss.mvn_blocked(p, scale_tril=0.2, scale_tril_bijector=None)

    scale_trils = [
        vdist.var.dist_node.kwinputs["scale_tril"].value
        for vdist in elbo.vdist.vi_dists
    ]
    assert jnp.allclose(scale_trils[0], jnp.array([[0.2]]))
    assert jnp.allclose(scale_trils[1], 0.2 * jnp.eye(2))


def test_neg_elbo_mvn_diag_inherits_target_model_to_float32():
    p = _laplace_model()

    elbo = opt.NegElboLoss.mvn_diag(p)

    assert elbo.q.to_float32 is p.to_float32


def test_vdist_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0)

    assert not q.p.to_float32
    assert q._to_float32 is False


def test_vdist_default_inherits_target_model_to_float32():
    x = lsl.Var.new_obs(
        jnp.array(0.0),
        name="x",
    )
    model = lsl.Model([x], to_float32=True)

    q = opt.VDist(["x"], model).normal(0.0, 1.0).build()

    assert q.p.to_float32 is True
    assert q.q is not None
    assert q.q.to_float32 is True


def test_vdist_can_override_target_model_to_float32():
    x = lsl.Var.new_obs(
        jnp.array(0.0),
        name="x",
    )
    model = lsl.Model([x], to_float32=True)

    q = opt.VDist(["x"], model, to_float32=False).normal(0.0, 1.0).build()

    assert q.p.to_float32 is True
    assert q.q is not None
    assert q.q.to_float32 is False


def test_vdist_can_still_force_variational_model_to_float32():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(
            jnp.array(0.0),
            name="x",
        )
        model = lsl.Model([x], to_float32=False)

        q = opt.VDist(["x"], model, to_float32=True).normal(0.0, 1.0).build()

    assert q.q is not None
    assert q.q.to_float32 is True
    assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float32


def test_vdist_uses_float64_under_x64_when_not_converting():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(
            jnp.array(0.0),
            name="x",
        )
        model = lsl.Model([x], to_float32=False)

        q = opt.VDist(["x"], model).normal().build()

    assert q.q is not None
    assert q.q.to_float32 is False
    assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float64


def test_compositevdist_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0)

    vi_dist = opt.CompositeVDist(q).build()

    assert not vi_dist._to_float32()


def test_vdist_exp_bijector_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0, scale_bijector=tfb.Exp())

    assert not q.p.to_float32


def test_compositevdist_exp_bijector_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0, scale_bijector=tfb.Exp())

    vi_dist = opt.CompositeVDist(q).build()

    assert not vi_dist._to_float32()


class TestVDist:
    def test_normal_laplace_uses_standard_deviation(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).normal(scale="laplace", scale_bijector=None)
        scale = q.var.dist_node.kwinputs["scale"].value

        assert jnp.allclose(scale, jnp.sqrt(jnp.array([1.0 / 3.0])), rtol=1e-5)

    def test_mvn_diag_laplace_uses_standard_deviation(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).mvn_diag(
            scale_diag="laplace", scale_diag_bijector=None
        )
        scale_diag = q.var.dist_node.kwinputs["scale_diag"].value

        assert jnp.allclose(scale_diag, jnp.sqrt(jnp.array([1.0 / 3.0])), rtol=1e-5)

    def test_mvn_tril_laplace_uses_cholesky_factor(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).mvn_tril(
            scale_tril="laplace", scale_tril_bijector=None
        )
        scale_tril = q.var.dist_node.kwinputs["scale_tril"].value

        assert jnp.allclose(scale_tril, jnp.sqrt(jnp.array([[1.0 / 3.0]])), rtol=1e-5)

    def test_rejects_non_reparameterized_custom_distribution(self):
        p = _laplace_model()
        dist = lsl.Dist(tfp.distributions.Categorical, probs=jnp.ones(2) / 2)

        with pytest.raises(ValueError, match="fully reparameterized"):
            opt.VDist(["loc"], p).init(dist)

    def test_rejects_shape_incompatible_custom_distribution(self):
        p = _laplace_model()
        dist = lsl.Dist(
            tfp.distributions.MultivariateNormalDiag,
            loc=jnp.zeros(2),
            scale_diag=jnp.ones(2),
        )

        with pytest.raises(ValueError, match="shape"):
            opt.VDist(["loc"], p).init(dist)

    def test_sample_shapes(self):
        loc = lsl.Var.new_param(jnp.array([0.0]), name="loc")
        scale = lsl.Var.new_param(0.0, name="scale", bijector=tfp.bijectors.Exp())
        y = lsl.Var.new_obs(
            jnp.linspace(-2, 2, 50),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
            name="y",
        )
        p = lsl.Model([y])
        q = opt.VDist(p.parameters, p).mvn_tril().build()

        key = jax.random.key(0)
        samples = q.sample(key)
        assert samples["loc"].shape == (1,)
        assert samples["h(scale)"].shape == ()

        samples = q.sample(key, (2,))
        assert samples["loc"].shape == (2, 1)
        assert samples["h(scale)"].shape == (2,)

        samples = q.sample(key, (1, 2))
        assert samples["loc"].shape == (1, 2, 1)
        assert samples["h(scale)"].shape == (1, 2)

        samples = q.sample(key, (1, 2, 3))
        assert samples["loc"].shape == (1, 2, 3, 1)
        assert samples["h(scale)"].shape == (1, 2, 3)

    def test_sample_at_position_shapes(self):
        loc = lsl.Var.new_param(jnp.array([0.0]), name="loc")
        scale = lsl.Var.new_param(0.0, name="scale", bijector=tfp.bijectors.Exp())
        y = lsl.Var.new_obs(
            jnp.linspace(-2, 2, 50),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
            name="y",
        )
        p = lsl.Model([y])
        q = opt.VDist(p.parameters, p).mvn_tril().build()

        at_position = q.q.extract_position(q.parameters)

        key = jax.random.key(0)
        samples = q.sample(key, at_position=at_position)
        assert samples["loc"].shape == (1,)
        assert samples["h(scale)"].shape == ()

        samples = q.sample(key, (2,), at_position=at_position)
        assert samples["loc"].shape == (2, 1)
        assert samples["h(scale)"].shape == (2,)

        samples = q.sample(key, (1, 2), at_position=at_position)
        assert samples["loc"].shape == (1, 2, 1)
        assert samples["h(scale)"].shape == (1, 2)

        samples = q.sample(key, (1, 2, 3), at_position=at_position)
        assert samples["loc"].shape == (1, 2, 3, 1)
        assert samples["h(scale)"].shape == (1, 2, 3)


class TestCompositeVDist:
    def test_rejects_empty_composite(self):
        with pytest.raises(ValueError, match="at least one"):
            opt.CompositeVDist()

    def test_rejects_overlapping_position_keys(self):
        p = _laplace_model()
        q1 = opt.VDist(["loc"], p).mvn_diag()
        q2 = opt.VDist(["loc"], p).mvn_diag()

        with pytest.raises(ValueError, match="duplicates"):
            opt.CompositeVDist(q1, q2)

    def test_rejects_different_target_models(self):
        p1 = _laplace_model()
        p2 = _laplace_model()
        q1 = opt.VDist(["loc"], p1).mvn_diag()
        q2 = opt.VDist(["loc"], p2).mvn_diag()

        with pytest.raises(ValueError, match="share one p"):
            opt.CompositeVDist(q1, q2)

    def test_sample_shapes(self):
        loc = lsl.Var.new_param(jnp.array([0.0]), name="loc")
        scale = lsl.Var.new_param(1.0, name="scale", bijector=tfp.bijectors.Exp())
        y = lsl.Var.new_obs(
            jnp.linspace(-2, 2, 50),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
            name="y",
        )
        p = lsl.Model([y])
        q1 = opt.VDist(["loc"], p).mvn_diag()
        q2 = opt.VDist(["h(scale)"], p).mvn_diag()
        q = opt.CompositeVDist(q1, q2).build()

        key = jax.random.key(0)
        samples = q.sample(key)
        assert samples["loc"].shape == (1,)
        assert samples["h(scale)"].shape == ()

        samples = q.sample(key, (2,))
        assert samples["loc"].shape == (2, 1)
        assert samples["h(scale)"].shape == (2,)

        samples = q.sample(key, (1, 2))
        assert samples["loc"].shape == (1, 2, 1)
        assert samples["h(scale)"].shape == (1, 2)

        samples = q.sample(key, (1, 2, 3))
        assert samples["loc"].shape == (1, 2, 3, 1)
        assert samples["h(scale)"].shape == (1, 2, 3)


class TestNegElboLoss:
    def test_rejects_non_positive_sample_counts(self):
        p = _laplace_model()

        with pytest.raises(ValueError, match="nsamples"):
            opt.NegElboLoss.mvn_diag(p, nsamples=0)

    def test_from_vdist_requires_built_distribution(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p)
        vdist = opt.VDist(["loc"], p).mvn_diag()

        with pytest.raises(ValueError, match="build"):
            opt.NegElboLoss.from_vdist(vdist, split)

    def test_rejects_split_with_validation_data(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p, share_validate=0.5)

        with pytest.raises(ValueError, match="validation data"):
            opt.NegElboLoss.mvn_diag(p, split=split)

    def test_regularize_q_prior_controls_variational_prior_contribution(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p)
        q_loc = lsl.Var.new_param(
            jnp.zeros(1),
            lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="q_loc",
        )
        q_scale = lsl.Var.new_param(jnp.ones(1), name="q_scale")
        dist = lsl.Dist(
            tfp.distributions.MultivariateNormalDiag,
            loc=q_loc,
            scale_diag=q_scale,
        )
        vdist = opt.VDist(["loc"], p).init(dist).build()
        params = vdist.q.extract_position(vdist.parameters)
        key = jax.random.key(1)

        elbo_with_prior = opt.NegElboLoss.from_vdist(
            vdist, split, nsamples=2, regularize_q_prior=True
        )
        elbo_without_prior = opt.NegElboLoss.from_vdist(
            vdist, split, nsamples=2, regularize_q_prior=False
        )
        value_with_prior = elbo_with_prior.estimate_elbo(params, key, p.state)
        value_without_prior = elbo_without_prior.estimate_elbo(params, key, p.state)
        q_state = vdist.q.update_state(params, vdist.q.state)

        assert jnp.allclose(
            value_with_prior - value_without_prior,
            q_state["_model_log_prior"].value,
        )
