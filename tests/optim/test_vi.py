import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel.model as lsl
import liesel.optim as opt
from liesel.optim.types import Position


def _dist_node(vdist: opt.VDist):
    assert vdist.var is not None
    assert vdist.var.dist_node is not None
    return vdist.var.dist_node


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


def test_weighted_elbo_expectation_and_gradient_match_full_data():
    model = _laplace_model()
    model.vars["y"].value = jnp.array([1.0, 4.0])
    loss = opt.NegElboLoss.mvn_diag(model, nsamples=3)
    params = loss.position(list(loss.q.parameters))
    batches = opt.Batches(
        ["y"], 2, 1, sample_with_replacement=True, sampling_weights=[1.0, 3.0]
    )
    batches.indices = jnp.array([0, 1])
    key = jax.random.key(12)

    def estimate(position, index):
        return loss.estimate_elbo(
            position,
            key,
            model.state,
            obs=batches.get_batched_position(loss.split.train, index),
            batches=batches,
            batch_index=index,
        )

    def weighted(position):
        values = jax.vmap(estimate, in_axes=(None, 0))(position, jnp.arange(2))
        return jnp.dot(jnp.array([0.25, 0.75]), values)

    expected = jax.value_and_grad(
        lambda position: loss.estimate_elbo(position, key, model.state)
    )(params)
    actual = jax.jit(jax.value_and_grad(weighted))(params)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_neg_elbo_from_vdist_scale_uses_total_branch_training_size():
    model = _two_branch_param_model()
    split = opt.PositionSplitManager.from_model(model, position_keys=["y1", "y2"])
    vdist = opt.VDist(["loc"], model).mvn_diag().build()

    elbo = opt.NegElboLoss.from_vdist(vdist, split, scale=True)

    assert elbo.scalar == sum(split.train_axis_sizes)


def test_neg_elbo_mvn_diag_forwards_custom_initialization():
    p = _laplace_model()

    elbo = opt.NegElboLoss.mvn_diag(
        p,
        loc=jnp.array([1.5]),
        scale_diag=0.2,
        scale_diag_bijector=None,
    )

    assert isinstance(elbo.vdist, opt.VDist)
    params = _dist_node(elbo.vdist).kwinputs
    assert jnp.allclose(params["loc"].value, jnp.array([1.5]))
    assert jnp.allclose(params["scale_diag"].value, jnp.array([0.2]))


def test_neg_elbo_mvn_tril_forwards_laplace_initialization():
    p = _laplace_model()

    elbo = opt.NegElboLoss.mvn_tril(
        p,
        scale_tril="laplace",
        scale_tril_bijector=None,
    )

    assert isinstance(elbo.vdist, opt.VDist)
    scale_tril = _dist_node(elbo.vdist).kwinputs["scale_tril"].value
    assert jnp.allclose(scale_tril, jnp.sqrt(jnp.array([[1.0 / 3.0]])), rtol=1e-5)


def test_neg_elbo_mvn_blocked_forwards_shared_scale_initialization():
    p = _two_parameter_model()

    elbo = opt.NegElboLoss.mvn_blocked(p, scale_tril=0.2, scale_tril_bijector=None)

    assert isinstance(elbo.vdist, opt.CompositeVDist)
    scale_trils = [
        _dist_node(vdist).kwinputs["scale_tril"].value for vdist in elbo.vdist.vi_dists
    ]
    assert jnp.allclose(scale_trils[0], jnp.array([[0.2]]))
    assert jnp.allclose(scale_trils[1], 0.2 * jnp.eye(2))


def test_model_float32_policy_has_no_public_getter():
    assert not hasattr(_laplace_model(), "to_float32")


@pytest.mark.parametrize("family", ["mvn_diag", "mvn_tril", "mvn_blocked"])
@pytest.mark.parametrize("to_float32", [True, False])
def test_neg_elbo_family_inherits_target_model_to_float32(family, to_float32):
    with jax.enable_x64(True):
        loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
        y = lsl.Var.new_obs(
            jnp.zeros(2),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=1.0),
            name="y",
        )
        p = lsl.Model([y], to_float32=to_float32)
        loss = getattr(opt.NegElboLoss, family)(p)
        expected = jnp.float32 if to_float32 else jnp.float64
        assert all(
            value.dtype == expected
            for value in loss.q.extract_position(list(loss.q.parameters)).values()
        )
        assert loss.vdist is not None
        assert loss.vdist.sample(jax.random.key(1), (3,))["loc"].dtype == expected


def test_vdist_float64():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(0.0, name="x")
        model = lsl.Model([x], to_float32=False)
        q = opt.VDist(["x"], model).normal(0.0, 1.0).build()
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float64
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


def test_vdist_default_inherits_target_model_to_float32():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(jnp.array(0.0), name="x")
        model = lsl.Model([x], to_float32=True)
        q = opt.VDist(["x"], model).normal(0.0, 1.0).build()
        assert model.vars["x"].value.dtype == jnp.float32
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float32
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float32


def test_vdist_can_override_target_model_to_float32():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(jnp.array(0.0), name="x")
        model = lsl.Model([x], to_float32=True)
        q = opt.VDist(["x"], model, to_float32=False).normal(0.0, 1.0).build()
        assert model.vars["x"].value.dtype == jnp.float32
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float64
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


@pytest.mark.parametrize("initializer", ["normal", "mvn_diag", "mvn_tril"])
def test_vdist_can_still_force_variational_model_to_float32(initializer):
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(jnp.array(0.0, dtype=jnp.float32), name="x")
        model = lsl.Model([x], to_float32=False)
        vdist = opt.VDist(["x"], model, to_float32=True)
        loc = jnp.array([0.0], dtype=jnp.float64)
        q = getattr(vdist, initializer)(loc=loc).build()
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float32
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float32


@pytest.mark.parametrize("initializer", ["normal", "mvn_diag", "mvn_tril"])
def test_vdist_uses_float64_under_x64_when_not_converting(initializer):
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(jnp.array(0.0), name="x")
        model = lsl.Model([x], to_float32=False)
        vdist = opt.VDist(["x"], model)
        q = getattr(vdist, initializer)().build()
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float64
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


def test_compositevdist_float64():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(0.0, name="x")
        model = lsl.Model([x], to_float32=False)
        q = opt.VDist(["x"], model).normal(0.0, 1.0)
        vi_dist = opt.CompositeVDist(q).build()
        assert vi_dist.q is not None
        assert (
            vi_dist.q.extract_position(vi_dist.parameters)["(x)_loc"].dtype
            == jnp.float64
        )
        assert vi_dist.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


def test_vdist_exp_bijector_float64():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(0.0, name="x")
        model = lsl.Model([x], to_float32=False)
        q = opt.VDist(["x"], model).normal(0.0, 1.0, scale_bijector=tfb.Exp()).build()
        assert q.q is not None
        assert q.q.extract_position(q.parameters)["(x)_loc"].dtype == jnp.float64
        assert q.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


def test_compositevdist_exp_bijector_float64():
    with jax.enable_x64(True):
        x = lsl.Var.new_obs(0.0, name="x")
        model = lsl.Model([x], to_float32=False)
        q = opt.VDist(["x"], model).normal(0.0, 1.0, scale_bijector=tfb.Exp())
        vi_dist = opt.CompositeVDist(q).build()
        assert vi_dist.q is not None
        assert (
            vi_dist.q.extract_position(vi_dist.parameters)["(x)_loc"].dtype
            == jnp.float64
        )
        assert vi_dist.sample(jax.random.key(1), (3,))["x"].dtype == jnp.float64


def test_compositevdist_rejects_inconsistent_dtype_policy():
    model = _two_parameter_model()
    q1 = opt.VDist(["alpha"], model, to_float32=True).normal()
    q2 = opt.VDist(["beta"], model, to_float32=False).mvn_diag()
    with pytest.raises(ValueError, match="setting must be consistent"):
        opt.CompositeVDist(q1, q2).build()


class TestVDist:
    @pytest.mark.parametrize("family", ["normal", "mvn_diag", "mvn_tril"])
    def test_laplace_initialization_preserves_correlated_curvature(self, family):
        theta = lsl.Var.new_param(
            jnp.zeros(2),
            lsl.Dist(
                tfp.distributions.MultivariateNormalTriL,
                jnp.zeros(2),
                jnp.array([[2.0, 0.0], [1.0, 1.0]]),
            ),
            name="theta",
        )
        vdist = opt.VDist(["theta"], lsl.Model(theta))
        scale_arg = {
            "normal": "scale",
            "mvn_diag": "scale_diag",
            "mvn_tril": "scale_tril",
        }
        getattr(vdist, family)(**{scale_arg[family]: "laplace"}).build()
        distribution = _dist_node(vdist).init_dist()
        if family == "mvn_tril":
            actual = distribution.covariance()
            expected = np.array([[4.0, 2.0], [2.0, 2.0]])
        else:
            actual = distribution.variance()
            expected = np.array([4.0, 2.0])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_normalizes_tuple_position_keys(self):
        p = _laplace_model()

        vdist = opt.VDist(("loc",), p)

        assert vdist.position_keys == ["loc"]
        assert jnp.allclose(
            vdist.p_to_q_array(Position({"loc": jnp.array(1.5)})), jnp.array([1.5])
        )

    @pytest.mark.parametrize("position_keys", [[], ["loc", "loc"]])
    def test_rejects_empty_or_duplicate_position_keys(self, position_keys):
        p = _laplace_model()

        with pytest.raises(ValueError, match="position_key"):
            opt.VDist(position_keys, p)

    def test_normal_laplace_uses_standard_deviation(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).normal(scale="laplace", scale_bijector=None)
        scale = _dist_node(q).kwinputs["scale"].value

        assert jnp.allclose(scale, jnp.sqrt(jnp.array([1.0 / 3.0])), rtol=1e-5)

    def test_mvn_diag_laplace_uses_standard_deviation(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).mvn_diag(
            scale_diag="laplace", scale_diag_bijector=None
        )
        scale_diag = _dist_node(q).kwinputs["scale_diag"].value

        assert jnp.allclose(scale_diag, jnp.sqrt(jnp.array([1.0 / 3.0])), rtol=1e-5)

    def test_mvn_tril_laplace_uses_cholesky_factor(self):
        p = _laplace_model()
        q = opt.VDist(["loc"], p).mvn_tril(
            scale_tril="laplace", scale_tril_bijector=None
        )
        scale_tril = _dist_node(q).kwinputs["scale_tril"].value

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
        scale = lsl.Var.new_param(1.0, name="scale", bijector=tfp.bijectors.Exp())
        y = lsl.Var.new_obs(
            jnp.linspace(-2, 2, 50),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
            name="y",
        )
        p = lsl.Model([y])
        q = opt.VDist(list(p.parameters), p).mvn_tril().build()

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
        scale = lsl.Var.new_param(1.0, name="scale", bijector=tfp.bijectors.Exp())
        y = lsl.Var.new_obs(
            jnp.linspace(-2, 2, 50),
            lsl.Dist(tfp.distributions.Normal, loc=loc, scale=scale),
            name="y",
        )
        p = lsl.Model([y])
        q = opt.VDist(list(p.parameters), p).mvn_tril().build()

        assert q.q is not None
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

        assert q.q is not None
        at_position = q.q.extract_position(q.parameters)
        samples = q.sample(key, (2,), at_position=at_position)
        assert samples["loc"].shape == (2, 1)
        assert samples["h(scale)"].shape == (2,)


class TestNegElboLoss:
    def test_ill_conditioned_gaussian_has_finite_elbo_and_gradients(self):
        n = 64
        theta = lsl.Var.new_param(
            jnp.zeros(n),
            lsl.Dist(tfp.distributions.Normal, loc=0.0, scale=1.0),
            name="theta",
        )
        y = lsl.Var.new_obs(
            jnp.zeros(n),
            lsl.Dist(tfp.distributions.Normal, loc=theta, scale=1.0),
            name="y",
        )
        p = lsl.Model([y])
        scale = 0.01 * jnp.eye(n) + 0.04 * jnp.eye(n, k=-1)
        loss = opt.NegElboLoss.mvn_tril(p, scale_tril=scale, nsamples=4)
        params = loss.position(list(loss.q.parameters))

        value, grad = jax.jit(
            jax.value_and_grad(
                lambda position: loss.estimate_elbo(
                    position, jax.random.key(84), p.state
                )
            )
        )(params)

        assert jnp.isfinite(value)
        assert all(jnp.isfinite(v).all() for v in grad.values())

    @pytest.mark.parametrize("nsamples", [0, True, 1.5, "2"])
    def test_rejects_invalid_sample_counts(self, nsamples):
        p = _laplace_model()

        with pytest.raises(ValueError, match="nsamples"):
            opt.NegElboLoss.mvn_diag(p, nsamples=nsamples)

    def test_from_vdist_rejects_invalid_sample_count(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p)
        vdist = opt.VDist(["loc"], p).mvn_diag().build()

        with pytest.raises(ValueError, match="nsamples"):
            opt.NegElboLoss.from_vdist(
                vdist,
                split,
                nsamples=1.5,  # ty: ignore[invalid-argument-type]
            )

    def test_from_vdist_builds_default_split(self):
        p = _laplace_model()
        vdist = opt.VDist(["loc"], p).mvn_diag().build()

        loss = opt.NegElboLoss.from_vdist(vdist)

        assert isinstance(loss.split, opt.PositionSplit)
        assert loss.split.train_axis_size == 2
        assert loss.split.validate_axis_size == 0
        assert loss.split.test_axis_size == 0

    def test_estimate_elbo_rejects_invalid_sample_count_override(self):
        p = _laplace_model()
        elbo = opt.NegElboLoss.mvn_diag(p, nsamples=1)
        params = elbo.position(list(elbo.q.parameters))

        with pytest.raises(ValueError, match="nsamples"):
            elbo.estimate_elbo(
                params,
                jax.random.key(1),
                p.state,
                nsamples=1.5,  # ty: ignore[invalid-argument-type]
            )

    def test_from_vdist_requires_built_distribution(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p)
        vdist = opt.VDist(["loc"], p).mvn_diag()

        with pytest.raises(ValueError, match="build"):
            opt.NegElboLoss.from_vdist(vdist, split)

    def test_rejects_split_with_validation_data(self):
        p = _laplace_model()
        split = opt.PositionSplit.from_model(p, validate_axis_share=0.5, seed=1)

        with pytest.raises(ValueError, match="validation data"):
            opt.NegElboLoss.mvn_diag(p, split=split)

    @pytest.mark.parametrize("entropy", ["auto", "mc"])
    def test_regularize_q_prior_controls_variational_prior_contribution(self, entropy):
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
        assert vdist.q is not None
        params = vdist.q.extract_position(vdist.parameters)
        key = jax.random.key(1)

        elbo_with_prior = opt.NegElboLoss.from_vdist(
            vdist, split, nsamples=2, regularize_q_prior=True, entropy=entropy
        )
        elbo_without_prior = opt.NegElboLoss.from_vdist(
            vdist, split, nsamples=2, regularize_q_prior=False, entropy=entropy
        )
        value_with_prior = elbo_with_prior.estimate_elbo(params, key, p.state)
        value_without_prior = elbo_without_prior.estimate_elbo(params, key, p.state)
        q_state = vdist.q.update_state(params, vdist.q.state)

        assert jnp.allclose(
            value_with_prior - value_without_prior,
            q_state["_model_log_prior"].value,
        )


def _entropy_test_loss(q, **kwargs):
    # A constant target isolates the entropy contribution of the actual ELBO.
    return opt.NegElboLoss(_laplace_model(), q, q_to_p=lambda sample: {}, **kwargs)


@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("position_kind", ["current", "partial", "full", "alias"])
def test_sampling_keeps_variational_parameters_fixed(composite, position_kind):
    mean = lsl.Var.new_param(
        jnp.array([3.0]),
        lsl.Dist(tfp.distributions.Normal, 10.0, 1.0),
        name="mean",
    )
    scale = lsl.Var.new_param(jnp.array([0.2]), name="scale")
    block = opt.VDist(["loc"], _laplace_model()).init(
        lsl.Dist(tfp.distributions.Normal, mean, scale)
    )
    vdist = opt.CompositeVDist(block).build() if composite else block.build()
    position = None
    expected_mean, expected_scale = 3.0, 0.2
    if position_kind != "current":
        position = Position({"scale": jnp.array([0.5])})
        expected_scale = 0.5
    if position_kind in ("full", "alias"):
        assert position is not None
        mean_key = "mean" if position_kind == "full" else mean.value_node.name
        position[mean_key] = jnp.array([-2.0])
        expected_mean = -2.0
    draws = vdist.sample(jax.random.key(82), (2000,), at_position=position)["loc"]
    np.testing.assert_allclose(draws.mean(), expected_mean, atol=0.04)
    np.testing.assert_allclose(draws.std(), expected_scale, atol=0.04)


@pytest.mark.parametrize("entropy", ["auto", "mc"])
@pytest.mark.parametrize("optimize_mean", [True, False])
def test_variational_parameter_priors_do_not_resample_parameters(
    entropy, optimize_mean
):
    p = _laplace_model()

    def evaluate(with_prior):
        prior = lsl.Dist(tfp.distributions.Normal, 10.0, 1.0) if with_prior else None
        mean = lsl.Var.new_param(3.0, prior, name="mean")
        scale = lsl.Var.new_param(0.2, name="scale")
        z = lsl.Var.new_obs(
            0.0, lsl.Dist(tfp.distributions.Normal, mean, scale), name="z"
        )
        loss = opt.NegElboLoss(
            p,
            lsl.Model(z),
            q_to_p=lambda pos: Position({"loc": pos["z"]}),
            regularize_q_prior=False,
            entropy=entropy,
            nsamples=2048,
        )
        params = Position(
            {"mean": jnp.array(3.0)} if optimize_mean else {"scale": jnp.array(0.2)}
        )
        return jax.jit(
            jax.value_and_grad(
                lambda params: loss.estimate_elbo(params, jax.random.key(82), p.state)
            )
        )(params)

    expected = evaluate(False)
    actual = evaluate(True)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("entropy", ["auto", "mc"])
def test_supplied_variational_state_controls_draws_without_mutating_model(entropy):
    p = _laplace_model()

    def variational_model(mean_value):
        mean = lsl.Var.new_value(mean_value, name="mean")
        scale = lsl.Var.new_param(0.2, name="scale")
        z = lsl.Var.new_obs(
            0.0, lsl.Dist(tfp.distributions.Normal, mean, scale), name="z"
        )
        return lsl.Model(z)

    q = variational_model(0.0)
    loss = opt.NegElboLoss(
        p, q, q_to_p=lambda pos: Position({"loc": pos["z"]}), entropy=entropy
    )
    reference = opt.NegElboLoss(
        p, variational_model(3.0), q_to_p=loss.q_to_p, entropy=entropy
    )
    state = q.update_state({"mean": jnp.array(3.0)})
    params = Position({"scale": jnp.array(0.2)})
    key = jax.random.key(82)
    actual = jax.jit(
        jax.value_and_grad(
            lambda params: loss.estimate_elbo(params, key, p.state, q_state=state)
        )
    )(params)
    expected = jax.jit(
        jax.value_and_grad(lambda params: reference.estimate_elbo(params, key, p.state))
    )(params)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    assert q.vars["mean"].value == 0.0


def _estimated_entropy(loss, params):
    return (
        loss.estimate_elbo(params, jax.random.key(84), loss.p.state) - loss.p.log_prob
    )


@pytest.mark.parametrize("family", ["scalar_normal", "batched_normal", "gamma"])
def test_entropy_values_and_gradients_with_replication(family):
    parameter = lsl.Var.new_param(1.2, name="parameter")
    if family == "gamma":
        distribution = tfp.distributions.Gamma
        kwargs = {"concentration": jnp.array([2.0, 3.0])}
        parameter_name = "rate"
        value = jnp.ones((3, 2))
    else:
        distribution = tfp.distributions.Normal
        kwargs = {"loc": 0.0 if family == "scalar_normal" else jnp.zeros(2)}
        parameter_name = "scale"
        value = jnp.zeros((3, 2))
    z = lsl.Var.new_obs(
        value, lsl.Dist(distribution, *kwargs.values(), parameter), name="z"
    )
    loss = _entropy_test_loss(lsl.Model([z]))
    actual = jax.jit(
        jax.value_and_grad(lambda x: _estimated_entropy(loss, {"parameter": x}))
    )(jnp.array(1.2))
    repeats = 6 if family == "scalar_normal" else 3
    expected = jax.value_and_grad(
        lambda x: (
            repeats * jnp.sum(distribution(**kwargs, **{parameter_name: x}).entropy())
        )
    )(jnp.array(1.2))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("factory", ["mvn_diag", "mvn_tril", "mvn_blocked"])
def test_gaussian_factories_forward_entropy_and_sum_blocks(factory):
    built = getattr(opt.NegElboLoss, factory)(_two_parameter_model(), entropy="mc")
    assert built.entropy == "mc"
    loss = _entropy_test_loss(built.q)
    params = loss.position(list(loss.q.parameters))
    actual = jax.jit(lambda x: _estimated_entropy(loss, x))(params)
    expected = sum(
        jnp.sum(v.dist_node.init_dist().entropy()) for v in built.q.observed.values()
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_mixed_entropy_fallback_and_gradient(monkeypatch):
    scale = lsl.Var.new_param(1.2, name="scale")
    normal = lsl.Var.new_obs(
        0.0, lsl.Dist(tfp.distributions.Normal, 0.0, scale), name="normal"
    )
    gamma = lsl.Var.new_obs(
        1.0, lsl.Dist(tfp.distributions.Gamma, 2.0, scale), name="gamma"
    )
    q = lsl.Model([normal, gamma])
    loss = _entropy_test_loss(q, nsamples=4)

    def unsupported(self):
        raise NotImplementedError

    monkeypatch.setattr(tfp.distributions.Normal, "_entropy", unsupported)

    def expected(x):
        samples = q.sample((4,), seed=jax.random.key(84), newdata={"scale": x})
        return tfp.distributions.Gamma(2.0, x).entropy() - jnp.mean(
            tfp.distributions.Normal(0.0, x).log_prob(samples["normal"])
        )

    actual = jax.jit(
        jax.value_and_grad(lambda x: _estimated_entropy(loss, {"scale": x}))
    )(jnp.array(1.2))
    expected_value = jax.jit(jax.value_and_grad(expected))(jnp.array(1.2))
    np.testing.assert_allclose(actual, expected_value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("behavior", ["error", "nan", "inf"])
def test_entropy_failures_are_not_silently_replaced(monkeypatch, behavior):
    loss = opt.NegElboLoss.mvn_diag(_laplace_model())

    def broken(self, *args, **kwargs):
        if behavior == "error":
            raise ValueError("invalid entropy")
        return jnp.asarray(float(behavior))

    monkeypatch.setattr(tfp.distributions.MultivariateNormalDiag, "entropy", broken)
    params = loss.position(list(loss.q.parameters))
    evaluate = jax.jit(lambda x: loss.estimate_elbo(x, jax.random.key(0), loss.p.state))
    if behavior == "error":
        with pytest.raises(ValueError, match="invalid entropy"):
            evaluate(params)
    else:
        assert not jnp.isfinite(evaluate(params))


def test_conditional_entropy_averages_over_sampled_parents():
    mean = lsl.Var.new_param(0.2, name="mean")
    parent = lsl.Var.new_obs(
        0.0, lsl.Dist(tfp.distributions.Normal, mean, 1.0), name="parent"
    )
    scale = lsl.Var.new_calc(jnp.exp, parent, name="scale")
    child = lsl.Var.new_obs(
        0.0, lsl.Dist(tfp.distributions.Normal, 0.0, scale), name="child"
    )
    q = lsl.Model([child])
    loss = _entropy_test_loss(q, nsamples=1024)
    params = {"mean": jnp.array(0.2)}
    actual, grad = jax.jit(jax.value_and_grad(lambda x: _estimated_entropy(loss, x)))(
        params
    )
    samples = q.sample((1024,), seed=jax.random.key(84), newdata=params)
    base_entropy = 2 * tfp.distributions.Normal(0.0, 1.0).entropy()
    np.testing.assert_allclose(
        actual, base_entropy + samples["parent"].mean(), atol=1e-5
    )
    # E[parent] = mean, so H(q) = 2 H(N(0, 1)) + mean and dH/dmean = 1.
    np.testing.assert_allclose(actual, base_entropy + params["mean"], atol=0.1)
    np.testing.assert_allclose(grad["mean"], 1.0, atol=1e-5)


@pytest.mark.parametrize("entropy", ["auto", "mc"])
def test_custom_variational_likelihood_uses_mc(entropy):
    loc = lsl.Var.new_param(0.3, name="q_loc")
    scale = lsl.Var.new_param(1.2, name="q_scale")
    z = lsl.Var.new_obs(0.0, lsl.Dist(tfp.distributions.Normal, loc, scale), name="z")
    gb = lsl.GraphBuilder().add(z)
    # A custom aggregate that is still the normalized density of the draws.
    gb.log_lik_node = lsl.Calc(jnp.sum, z.dist_node)
    q = gb.build_model()
    p = _laplace_model()
    loss = opt.NegElboLoss(
        p, q, q_to_p=lambda sample: {"loc": sample["z"]}, entropy=entropy
    )
    key = jax.random.key(84)
    draws = q.sample((10,), seed=key)["z"]
    epsilon = (draws - 0.3) / 1.2
    # log q(z) = -log(scale)-log(2*pi)/2-epsilon**2/2;
    # p has a standard Normal prior and two zero Normal observations:
    # log p(z) = -3*log(2*pi)/2-3*z**2/2. Differentiate z=loc+scale*epsilon.
    expected = jnp.mean(
        jnp.log(1.2) + 0.5 * epsilon**2 - 1.5 * draws**2 - jnp.log(2 * jnp.pi)
    )
    value, gradients = jax.value_and_grad(
        lambda params: loss.estimate_elbo(params, key, p.state)
    )(loss.position(["q_loc", "q_scale"]))
    np.testing.assert_allclose(value, expected, atol=1e-5)
    np.testing.assert_allclose(gradients["q_loc"], -3 * draws.mean(), atol=1e-5)
    np.testing.assert_allclose(
        gradients["q_scale"], 1 / 1.2 - 3 * jnp.mean(draws * epsilon), atol=1e-5
    )


def test_mc_mode_matches_original_elbo_and_gradients():
    p = _laplace_model()
    vdist = opt.VDist(["loc"], p).mvn_tril().build()
    loss = opt.NegElboLoss.from_vdist(vdist, entropy="mc", nsamples=4)
    assert loss.entropy == "mc"
    key = jax.random.key(84)
    params = loss.position(list(loss.q.parameters))

    def original(position):
        samples = loss.q.sample((4,), seed=key, newdata=position)

        def log_ratio(sample):
            ps = p.update_state(loss.q_to_p(sample), p.state)
            qs = loss.q.update_state(sample | position, loss.q.state)
            return (
                ps["_model_log_prob"].value
                - qs["_model_log_lik"].value
                + qs["_model_log_prior"].value
            )

        return jax.vmap(log_ratio)(samples).mean()

    expected = jax.jit(jax.value_and_grad(original))(params)
    actual = jax.jit(jax.value_and_grad(lambda x: loss.estimate_elbo(x, key, p.state)))(
        params
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("entropy", [True, None, "analytic", 1])
def test_rejects_invalid_entropy_mode(entropy):
    with pytest.raises(ValueError, match="entropy"):
        opt.NegElboLoss.mvn_diag(_laplace_model(), entropy=entropy)


def _computed_mean_model(beta_is_parameter=True, transient=False):
    x = lsl.Var.new_value(jnp.arange(1.0, 5.0), name="x")
    beta = lsl.Var.new_value(1.0, name="beta")
    beta.parameter = beta_is_parameter
    mean = lsl.Var.new_calc(
        lambda b, v: b * v, beta, x, name="mean", cache=not transient
    )
    alpha = lsl.Var.new_param(0.0, name="alpha")
    loc = lsl.Var.new_calc(lambda a, m: a + m, alpha, mean)
    y = lsl.Var.new_obs(
        2.0 + x.value, lsl.Dist(tfp.distributions.Normal, loc, 1.0), name="y"
    )
    return lsl.Model([y])


@pytest.mark.parametrize("factory", ["vdist", "composite", "direct"])
@pytest.mark.parametrize("beta_is_parameter", [False, True])
@pytest.mark.parametrize("infer_beta", [False, True])
def test_computed_data_dependency_uses_inferred_target_keys(
    factory, beta_is_parameter, infer_beta
):
    model = _computed_mean_model(beta_is_parameter)
    split = opt.PositionSplit.from_model(model, position_keys=["mean", "y"])
    keys = ["alpha", "beta"] if infer_beta else ["alpha"]

    def make_loss():
        if factory == "direct":
            loc = lsl.Var.new_param(jnp.zeros(len(keys)), name="q_loc")
            draw = lsl.Var.new_obs(
                jnp.zeros(len(keys)),
                lsl.Dist(
                    tfp.distributions.MultivariateNormalDiag, loc, jnp.ones(len(keys))
                ),
                name="draw",
            )
            q = lsl.Model([draw])
            return opt.NegElboLoss(
                model,
                q,
                split=split,
                q_to_p=lambda sample: Position(
                    {key: sample["draw"][i] for i, key in enumerate(keys)}
                ),
            )
        vdist = (
            opt.CompositeVDist(*(opt.VDist([key], model).normal() for key in keys))
            if factory == "composite"
            else opt.VDist(keys, model).mvn_diag()
        ).build()
        return opt.NegElboLoss.from_vdist(vdist, split)

    if infer_beta:
        with pytest.raises(
            ValueError, match="Computed data key 'mean'.*parameter 'beta'"
        ):
            make_loss()
    else:
        loss = make_loss()
        value = loss.estimate_elbo(
            loss.position(list(loss.q.parameters)),
            jax.random.key(1),
            model.state,
            obs=split.train,
        )
        assert jnp.isfinite(value)


@pytest.mark.parametrize("target_kind", ["variable", "node"])
def test_computed_data_cannot_overlap_inferred_target(target_kind):
    model = _computed_mean_model()
    split = opt.PositionSplit.from_model(model, position_keys=["mean", "y"])
    key = "mean" if target_kind == "variable" else model.vars["mean"].value_node.name
    vdist = opt.VDist([key], model).mvn_diag().build()
    with pytest.raises(ValueError, match="Computed data key 'mean'.*inferred target"):
        opt.NegElboLoss.from_vdist(vdist, split)


@pytest.mark.parametrize("invalid", ["ancestor", "transient", "node_alias"])
def test_neg_elbo_validates_manual_computed_data_splits(invalid):
    model = _computed_mean_model(transient=invalid == "transient")
    keys = ["mean", "y"]
    if invalid == "ancestor":
        keys.append("x")
    elif invalid == "node_alias":
        keys[0] = model.vars["mean"].value_node.name
    split = opt.Split(keys, axis_size=4).split_position(
        Position({key: jnp.zeros(4) for key in keys})
    )
    vdist = opt.VDist(["alpha"], model).normal().build()
    message = {
        "ancestor": "Ambiguous",
        "transient": "transient",
        "node_alias": "variable name",
    }[invalid]
    with pytest.raises(ValueError, match=message):
        opt.NegElboLoss.from_vdist(vdist, split)
