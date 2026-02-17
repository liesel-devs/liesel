import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.bijectors as tfb

import liesel.experimental.optim as opt
import liesel.model as lsl


def test_vdist_float64():
    x = lsl.Var.new_obs(
        0.0,
        name="x",
    )
    model = lsl.Model([x], to_float32=False)
    q = opt.VDist(["x"], model).normal(0.0, 1.0)

    assert not q.p.to_float32


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
