import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates import jax as tfp

from liesel.experimental.vi import builder

tfd = tfp.distributions


class DummyVar:
    def __init__(self, value, observed=True):
        self.value = jnp.asarray(value)
        self.observed = observed


class DummyModel:
    def __init__(self, params, n_obs=10):
        self.vars = {name: DummyVar(jnp.asarray(val)) for name, val in params.items()}
        self.vars["__obs__"] = DummyVar(jnp.zeros((n_obs,)), observed=True)


class DummyInterface:
    def __init__(self, params):
        self._params = {k: jnp.asarray(v) for k, v in params.items()}
        self.model = DummyModel(params)

    def get_params(self):
        return self._params


def test_validate_latent_variable_keys_rejects_invalid():
    b = builder.OptimizerBuilder()
    with pytest.raises(ValueError):
        b._validate_latent_variable_keys(
            tfd.Normal,
            parameter_bijectors=None,
            variational_params={"loc": 0.0, "scale": 1.0, "foo": 1.0},
        )

    with pytest.raises(ValueError):
        b._validate_latent_variable_keys(
            tfd.Normal,
            parameter_bijectors={"foo": tfb.Identity()},
            variational_params={"loc": 0.0, "scale": 1.0},
        )


def test_obtain_and_merge_bijectors_override():
    b = builder.OptimizerBuilder()
    default = b._obtain_parameter_default_bijectors(tfd.Normal)
    assert "scale" in default
    assert "loc" in default

    custom = {"scale": tfb.Identity()}
    merged = b._merge_parameter_bijectors(default, custom)
    assert isinstance(merged["scale"], tfb.Identity) or isinstance(
        merged["scale"], tfb.Identity.__mro__[0]
    )


def test_add_variational_dist_builds_and_records_config():
    params = {"beta": jnp.zeros((4,)), "sigma": jnp.array(1.0)}
    interface = DummyInterface(params)
    b = builder.OptimizerBuilder(seed=0, n_epochs=10, S=8)
    b.set_model(interface)

    opt_chain = optax.chain(optax.clip(1.0), optax.adam(1e-2))

    b.add_variational_dist(
        ["sigma"],
        dist_class=tfd.LogNormal,
        variational_params={"loc": 0.0, "scale": 0.1},
        optimizer_chain=opt_chain,
        variational_param_bijectors={"scale": tfb.Softplus(), "loc": tfb.Identity()},
    )

    b.add_variational_dist(
        ["beta"],
        dist_class=tfd.MultivariateNormalDiag,
        variational_params={"loc": jnp.zeros((4,)), "scale_diag": jnp.ones((4,))},
        optimizer_chain=opt_chain,
        variational_param_bijectors={
            "scale_diag": tfb.Softplus(),
            "loc": tfb.Identity(),
        },
    )

    assert len(b.latent_variables) == 2

    cfg1 = b.latent_variables[0]
    assert cfg1["names"] == ["sigma"]
    assert "event_shape" in cfg1 and isinstance(cfg1["event_shape"], int)
    assert "variable_dims" in cfg1 and cfg1["variable_dims"]["sigma"] == 1
    assert cfg1["split_indices"] == []

    cfg2 = b.latent_variables[1]
    assert cfg2["names"] == ["beta"]
    assert cfg2["variable_dims"]["beta"] == 4
    assert cfg2["split_indices"] == []


def test_build_returns_optimizer_instance():
    params = {"w": jnp.zeros((3,)), "b": jnp.array(0.0)}
    interface = DummyInterface(params)
    b = builder.OptimizerBuilder(seed=123, n_epochs=5, S=4)
    b.set_model(interface)

    opt_chain = optax.adam(1e-2)

    b.add_variational_dist(
        ["w"],
        dist_class=tfd.MultivariateNormalDiag,
        variational_params={"loc": jnp.zeros((3,)), "scale_diag": jnp.ones((3,))},
        optimizer_chain=opt_chain,
        variational_param_bijectors={
            "scale_diag": tfb.Softplus(),
            "loc": tfb.Identity(),
        },
    )

    opt = b.build()
    from liesel.experimental.vi.optimizer import Optimizer

    assert isinstance(opt, Optimizer)
    assert "w" in opt.latent_vars_config
