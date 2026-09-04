import logging
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import random

import liesel.goose as gs
import liesel.model as lsl
from liesel.goose.mcmc_spec import MCMCSpec

type Array = jax.Array


class FixedDistribution(tfd.Distribution):
    def __init__(self, fixed_value: Array):
        super().__init__(
            dtype=jnp.float32,
            reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=False,
        )
        self.fixed_value = fixed_value

    def _sample_n(self, n, seed=None, **kwargs):
        return jnp.tile(self.fixed_value, (n, 1))

    def _batch_shape_tensor(self, **parameter_kwargs):
        return jnp.array([])

    def _batch_shape(self):
        return jnp.array([])

    def _event_shape_tensor(self):
        return self.fixed_value.shape

    def _event_shape(self):
        return self.fixed_value.shape


class DummyKernel:
    def __init__(self, position_keys, **kwargs):
        self.position_keys = position_keys
        self.kwargs = kwargs


# Create a dummy kernel factory for testing
def dummy_kernel_factory(position_keys, **kwargs):
    return DummyKernel(position_keys, **kwargs)


class _FakeResults:
    def __init__(self):
        self.saved_paths = []

    def pkl_save(self, path):
        self.saved_paths.append(path)


class _FakeEngine:
    def __init__(self, owner, results=None, error=None, time_error=None):
        self.owner = owner
        self.results = results or _FakeResults()
        self.error = error
        self.time_error = time_error
        self.exposed_while_sampling = False
        self.wall_time = None

    def _sample(self):
        self.exposed_while_sampling = self.owner.engine is self
        if self.error is not None:
            raise self.error

    def sample_all_epochs(self):
        self._sample()

    def sample_for_time(self, wall_time):
        self.wall_time = wall_time
        if self.time_error is not None:
            raise self.time_error
        self._sample()

    def get_results(self):
        return self.results


class _FakeEngineBuilder:
    def __init__(self, engine):
        self.engine = engine
        self.epochs = []
        self.store_kernel_states = False
        self.show_progress = True
        self.positions_included: list[str] = []
        self.positions_excluded: list[str] = []
        self.jit_block_size: int | None = None
        self.max_wall_time: float | None = None

    def add_adaptation(self, duration, thinning):
        self.epochs.append(("adaptation", duration, thinning))

    def add_burnin(self, duration, thinning):
        self.epochs.append(("burnin", duration, thinning))

    def add_posterior(self, duration, thinning):
        self.epochs.append(("posterior", duration, thinning))

    def build(self):
        return self.engine


class TestMCMCSpec:
    def setup_method(self):
        # Setup a basic MCMCSpec for testing
        self.kernel_factory = dummy_kernel_factory
        self.key = random.PRNGKey(42)
        self.value = jnp.ones((3,))

    def test_no_jitter(self):
        """Test that no jitter is applied when jitter_dist is None."""
        spec = MCMCSpec(kernel=self.kernel_factory)
        result = spec.apply_jitter(self.key, self.value)
        assert jnp.array_equal(result, self.value)

    def test_jitter_type_none_is_rejected(self):
        """Test that the removed jitter method "none" is rejected."""
        invalid_jitter_method: Any = "none"

        with pytest.raises(ValueError, match="Invalid jitter method"):
            MCMCSpec(
                kernel=self.kernel_factory,
                jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
                jitter_method=invalid_jitter_method,
            )

    def test_additive_jitter(self):
        """Test additive jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([0.1, 0.2, 0.3])
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(jnp.array([0.1, 0.2, 0.3])),
            jitter_method="additive",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = self.value + fixed_jitter
        assert jnp.allclose(result, expected)

    def test_multiplicative_jitter(self):
        """Test multiplicative jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([2.0, 3.0, 4.0])

        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(fixed_jitter),
            jitter_method="multiplicative",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = self.value * fixed_jitter
        assert jnp.allclose(result, expected)

    def test_replacement_jitter(self):
        """Test replacement jitter application."""
        # Using a fixed distribution for deterministic testing
        fixed_jitter = jnp.array([5.0, 6.0, 5.0])
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=FixedDistribution(fixed_jitter),
            jitter_method="replacement",
        )

        result = spec.apply_jitter(self.key, self.value)
        expected = fixed_jitter
        assert jnp.allclose(result, expected)

    def test_with_normal_distribution(self):
        """Test with a standard TFP distribution (Normal)."""
        # Using a seeded jitter for reproducible testing
        seed = 42
        value = jnp.ones(3)

        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
            jitter_method="additive",
        )

        # Generate jitter separately to compare
        key = random.PRNGKey(seed)
        jitter = tfd.Normal(loc=0.0, scale=1.0).sample(sample_shape=(3,), seed=key)
        expected = value + jitter

        # Apply jitter via the MCMCSpec
        result = spec.apply_jitter(key, value)

        assert jnp.allclose(result, expected)

    def test_shape_compatibility(self):
        """Test jitter with different shape configurations."""
        # Test with scalar value and distribution
        scalar_value = jnp.array(1.0)
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.Normal(loc=0.0, scale=1.0),
            jitter_method="replacement",
        )
        result = spec.apply_jitter(self.key, scalar_value)
        assert result.shape == scalar_value.shape

        # Test with vector value and scalar distribution
        vector_value = jnp.ones(3)
        result = spec.apply_jitter(self.key, vector_value)
        assert result.shape == vector_value.shape
        assert result[0] != result[1]  # Check that we have multiple draws

        # Test with multivariate distribution
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(3), scale_diag=jnp.ones(3)
            ),
        )
        result = spec.apply_jitter(self.key, vector_value)
        assert result.shape == vector_value.shape

    def test_incompatible_shapes(self):
        """Test error handling for incompatible shapes."""
        # Distribution shape (2,) for value shape (3,)
        value = jnp.ones(3)
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            jitter_dist=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(2), scale_diag=jnp.ones(2)
            ),
            jitter_method="additive",
        )

        with pytest.raises(ValueError, match="do not match variable shape"):
            spec.apply_jitter(self.key, value)

    def test_batch_shape(self):
        """Test jitter application with batched distributions."""
        value = jnp.ones((2, 3))
        spec = MCMCSpec(
            kernel=self.kernel_factory,
            # Create a batch of 2 normal distributions
            jitter_dist=tfd.Independent(
                tfd.Normal(loc=jnp.zeros((2, 3)), scale=jnp.ones((2, 3))),
                reinterpreted_batch_ndims=1,
            ),
            jitter_method="additive",
        )

        result = spec.apply_jitter(self.key, value)
        assert result.shape == value.shape


class TestLieselMCMC:
    def test_engine_is_none_before_sampling(self):
        model = lsl.Model([])

        assert gs.LieselMCMC(model).engine is None

    @pytest.mark.parametrize(
        ("method_name", "timing_arguments", "expected_wall_time"),
        [
            (
                "run_for_epochs",
                {"jit_block_size": 2, "max_wall_time": 9.5},
                None,
            ),
            (
                "run_for_time",
                {
                    "wall_time": 6.5,
                    "jit_block_size": 2,
                    "max_wall_time": 9.5,
                },
                6.5,
            ),
        ],
    )
    def test_run_methods_configure_schedule_retain_engine_and_save(
        self,
        monkeypatch,
        tmp_path,
        method_name,
        timing_arguments,
        expected_wall_time,
    ):
        mcmc = gs.LieselMCMC(lsl.Model([]))
        mcmc.engine = cast(Any, _FakeEngine(mcmc))
        engine = _FakeEngine(mcmc)
        builder = _FakeEngineBuilder(engine)
        builder_arguments = None

        def get_engine_builder(**kwargs):
            nonlocal builder_arguments
            builder_arguments = kwargs
            return builder

        monkeypatch.setattr(mcmc, "get_engine_builder", get_engine_builder)
        save_path = tmp_path / "results.pkl"

        result = getattr(mcmc, method_name)(
            seed=5,
            num_chains=3,
            adaptation=12,
            posterior=8,
            burnin=4,
            adaptation_thinning=3,
            burnin_thinning=2,
            posterior_thinning=4,
            apply_jitter=False,
            store_kernel_states=True,
            show_progress=False,
            positions_included=["included"],
            positions_excluded=["excluded"],
            save_path=save_path,
            **timing_arguments,
        )

        assert result is engine.results
        assert mcmc.engine is engine
        assert engine.exposed_while_sampling
        assert engine.wall_time == expected_wall_time
        assert builder_arguments == {
            "seed": 5,
            "num_chains": 3,
            "apply_jitter": False,
        }
        assert builder.epochs == [
            ("adaptation", 12, 3),
            ("burnin", 4, 2),
            ("posterior", 8, 4),
        ]
        assert builder.store_kernel_states is True
        assert builder.show_progress is False
        assert builder.positions_included == ["included"]
        assert builder.positions_excluded == ["excluded"]
        assert builder.jit_block_size == 2
        assert builder.max_wall_time == 9.5
        assert engine.results.saved_paths == [save_path]

    @pytest.mark.parametrize(
        ("method_name", "timing_arguments"),
        [
            ("run_for_epochs", {}),
            ("run_for_time", {"wall_time": 1.0, "jit_block_size": 1}),
        ],
    )
    def test_safety_failure_retains_engine_without_saving(
        self, monkeypatch, tmp_path, method_name, timing_arguments
    ):
        mcmc = gs.LieselMCMC(lsl.Model([]))
        engine = _FakeEngine(mcmc, error=TimeoutError("safety limit"))
        builder = _FakeEngineBuilder(engine)
        monkeypatch.setattr(mcmc, "get_engine_builder", lambda **_: builder)
        save_path = tmp_path / "partial.pkl"

        with pytest.raises(TimeoutError, match="safety limit"):
            getattr(mcmc, method_name)(
                seed=1,
                num_chains=2,
                adaptation=0,
                posterior=2,
                save_path=save_path,
                **timing_arguments,
            )

        assert mcmc.engine is engine
        assert engine.exposed_while_sampling
        assert not save_path.exists()
        assert engine.results.saved_paths == []

    def test_run_for_time_delegates_none_to_engine_validation(self, monkeypatch):
        mcmc = gs.LieselMCMC(lsl.Model([]))
        engine = _FakeEngine(
            mcmc, time_error=ValueError("wall_time must be greater than zero")
        )
        builder = _FakeEngineBuilder(engine)
        monkeypatch.setattr(mcmc, "get_engine_builder", lambda **_: builder)

        with pytest.raises(ValueError, match="wall_time"):
            mcmc.run_for_time(
                wall_time=cast(Any, None),
                jit_block_size=1,
                seed=1,
                num_chains=2,
                adaptation=0,
                posterior=2,
            )

        assert engine.wall_time is None

    @pytest.mark.parametrize(
        ("method_name", "timing_arguments"),
        [
            ("run_for_epochs", {}),
            ("run_for_time", {"wall_time": 1.0, "jit_block_size": 1}),
        ],
    )
    def test_cache_hit_clears_stale_engine(
        self, monkeypatch, tmp_path, method_name, timing_arguments
    ):
        mcmc = gs.LieselMCMC(lsl.Model([]))
        mcmc.engine = cast(Any, _FakeEngine(mcmc))
        cached_results = _FakeResults()
        cache_path = tmp_path / "cached.pkl"
        cache_path.touch()
        monkeypatch.setattr(gs.SamplingResults, "pkl_load", lambda path: cached_results)
        monkeypatch.setattr(
            mcmc,
            "get_engine_builder",
            lambda **_: pytest.fail("cache hit must not construct an Engine"),
        )

        result = getattr(mcmc, method_name)(
            seed=1,
            num_chains=2,
            adaptation=0,
            posterior=2,
            save_path=cache_path,
            **timing_arguments,
        )

        assert result is cached_results
        assert mcmc.engine is None

    def test_engine_validation(self, local_caplog):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)
        with local_caplog() as caplog:
            mcmc.get_engine_builder(seed=1, num_chains=4)
            assert caplog.records[0].levelno == logging.WARNING
            assert "No inference specification" in caplog.records[0].msg

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)
        with local_caplog() as caplog:
            mcmc.get_engine_builder(seed=1, num_chains=4)
            assert len(caplog.records) == 0

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu",
        )

        mu2 = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            name="mu2",
        )

        model = lsl.Model([mu, mu2])

        mcmc = gs.LieselMCMC(model)
        with local_caplog() as caplog:
            mcmc.get_engine_builder(seed=1, num_chains=4)
            assert len(caplog.records) == 2

    def test_engine(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(gs.NUTSKernel),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)
        eb = mcmc.get_engine_builder(seed=1, num_chains=4)
        eb.set_duration(warmup_duration=200, posterior_duration=100)
        engine = eb.build()

        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        assert "mu" in samples

    def test_multiple_specs(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference={
                "a": gs.MCMCSpec(gs.NUTSKernel),
                "b": gs.MCMCSpec(gs.IWLSKernel),
            },
            name="mu",
        )

        model = lsl.Model([mu])

        with pytest.raises(ValueError):
            gs.LieselMCMC(model).get_kernel_list()

        kernels = gs.LieselMCMC(model, which="a").get_kernel_list()
        assert isinstance(kernels[0], gs.NUTSKernel)

        kernels = gs.LieselMCMC(model, which="b").get_kernel_list()
        assert isinstance(kernels[0], gs.IWLSKernel)

    def test_multiple_and_single_specs(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference={
                "a": gs.MCMCSpec(gs.NUTSKernel),
                "b": gs.MCMCSpec(gs.IWLSKernel),
            },
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        with pytest.raises(ValueError):
            gs.LieselMCMC(model).get_kernel_list()

        kernels = gs.LieselMCMC(model, which="a").get_kernel_list()
        assert isinstance(kernels[1], gs.IWLSKernel)
        assert isinstance(kernels[0], gs.NUTSKernel)

        kernels = gs.LieselMCMC(model, which="b").get_kernel_list()
        assert isinstance(kernels[0], gs.IWLSKernel)
        assert isinstance(kernels[1], gs.IWLSKernel)

    def test_kernel_group(self):
        spec = gs.MCMCSpec(gs.NUTSKernel, kernel_group="a")

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=spec,
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=spec,
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        kernels = mcmc.get_kernel_list()

        assert len(kernels) == 1
        assert kernels[0].position_keys == ("mu", "sigma")

    def test_kernel_group_equal_kwargs(self):
        """
        Uses equal kernel kwargs for two MCMCSpecs.
        Since they are not the same object, we get an error.
        """
        spec1 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs={"mm_diag": True, "da_target_accept": 0.8},
        )

        spec2 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs={"mm_diag": True, "da_target_accept": 0.8},
        )

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=spec1,
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=spec2,
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        with pytest.raises(ValueError):
            mcmc = gs.LieselMCMC(model)
            mcmc.get_kernel_list()

    def test_kernel_group_kwargs_same_object(self):
        """
        Uses identical objects for the kernel kwargs.
        """
        kwargs = {"mm_diag": True, "da_target_accept": 0.8}
        spec1 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs=kwargs,
        )

        spec2 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs=kwargs,
        )

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=spec1,
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=spec2,
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        kernels = mcmc.get_kernel_list()
        assert len(kernels) == 1
        assert kernels[0].position_keys == ("mu", "sigma")

    def test_kernel_group_kwargs_defined_once(self):
        """
        Only one spec defines the kernel kwargs, they get used.
        """
        kwargs = {"mm_diag": False, "da_target_accept": 0.5}
        spec1 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
            kernel_kwargs=kwargs,
        )

        spec2 = gs.MCMCSpec(
            gs.NUTSKernel,
            kernel_group="a",
        )

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=spec1,
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=spec2,
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        kernels = mcmc.get_kernel_list()
        assert len(kernels) == 1
        kernel = kernels[0]
        assert isinstance(kernel, gs.NUTSKernel)
        assert kernel.position_keys == ("mu", "sigma")
        assert kernel.da_target_accept == pytest.approx(kwargs["da_target_accept"])
        assert kernel.mm_diag == pytest.approx(kwargs["mm_diag"])

    def test_incoherent_kernel_group(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(gs.NUTSKernel, kernel_group="a"),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(gs.IWLSKernel, kernel_group="a"),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        with pytest.raises(ValueError):
            mcmc.get_kernel_list()

        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel, {"da_target_accept": 0.6}, kernel_group="a"
            ),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(
                gs.NUTSKernel, {"da_target_accept": 0.7}, kernel_group="a"
            ),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        with pytest.raises(ValueError):
            mcmc.get_kernel_list()

    def test_jitter_functions(self):
        mu = lsl.Var.new_param(
            0.0,
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=1.0, scale=0.5),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=0.0, high=1.0),
            ),
            name="sigma",
        )

        model = lsl.Model([mu, sigma])

        mcmc = gs.LieselMCMC(model)
        jitter_fns = mcmc.get_jitter_functions()

        assert len(jitter_fns) == 2

        eb = mcmc.get_engine_builder(1, 4)
        assert len(eb.jitter_fns.expect("")) == 2

    def test_jitter_draw_shape(self):
        # 1d array
        mu = lsl.Var.new_param(
            jnp.zeros(3),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)

        jitter_funs = mcmc.get_jitter_functions()
        jitter_draw = jitter_funs["mu"](jax.random.key(0), mu.value)
        assert not jnp.all(jitter_draw == jitter_draw[0])  # not all equal
        # no two are equal
        assert len(jnp.unique(jitter_draw)) == len(jitter_draw.flatten())

        # 2d array
        mu = lsl.Var.new_param(
            jnp.zeros((3, 3)),
            lsl.Dist(tfd.Normal, loc=0.0, scale=1.0),
            inference=gs.MCMCSpec(
                gs.NUTSKernel,
                kernel_group="a",
                jitter_dist=tfd.Uniform(low=-1.0, high=1.0),
            ),
            name="mu",
        )

        model = lsl.Model([mu])

        mcmc = gs.LieselMCMC(model)

        jitter_funs = mcmc.get_jitter_functions()
        jitter_draw = jitter_funs["mu"](jax.random.key(0), mu.value)
        assert not jnp.all(jitter_draw == jitter_draw[0])  # not all equal
        # no two are equal
        assert len(jnp.unique(jitter_draw)) == len(jitter_draw.flatten())

    def test_transform_var_with_inference_new(self):
        """
        It is allowed to pass a new inferece object during transformation.
        In this case, the inference object of the original variable is removed.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )
        inference = sigma.inference

        log_sigma = sigma.transform(tfb.Exp(), inference=gs.MCMCSpec(gs.NUTSKernel))
        assert log_sigma.inference is not inference
        assert sigma.inference is None
        assert log_sigma.inference.kernel is gs.NUTSKernel

    def test_transform_var_with_inference_none(self):
        """
        Default behavior when trying to transform a variable *with* inference
        information: Error. You need to declare explicitly, what you want to do.
        In this case, ``"drop"`` means the inference information is deleted.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            inference=gs.MCMCSpec(gs.IWLSKernel),
            name="sigma",
        )
        inference = sigma.inference

        with pytest.raises(ValueError):
            sigma.transform(tfb.Exp())

        log_sigma = sigma.transform(tfb.Exp(), inference="drop")
        assert log_sigma.inference is not inference
        assert sigma.inference is None
        assert log_sigma.inference is None

    def test_transform_var_without_inference(self):
        """
        Default when the original variable has no inference information:
        Everything works smoothly.
        """
        sigma = lsl.Var.new_param(
            1.0,
            lsl.Dist(tfd.InverseGamma, concentration=2.0, scale=1.0),
            name="sigma",
        )

        log_sigma = sigma.transform(tfb.Exp())

        assert log_sigma.inference is None
        assert sigma.inference is None
        assert log_sigma.inference is None

    def test_default_order_of_kernels(self):
        layer3 = lsl.Var.new_param(
            1.0, name="layer3", inference=gs.MCMCSpec(gs.NUTSKernel)
        )
        layer2_loc = lsl.Var.new_param(
            1.0,
            distribution=lsl.Dist(tfd.Normal, loc=layer3, scale=1.0),
            name="layer2_loc",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        layer2_scale = lsl.Var.new_param(
            1.0,
            name="layer2_scale",
            bijector=tfb.Exp(),
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        layer1 = lsl.Var.new_param(
            1.0,
            distribution=lsl.Dist(tfd.Normal, loc=layer2_loc, scale=layer2_scale),
            name="layer1",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        model = lsl.Model([layer1])
        klist = gs.LieselMCMC(model).get_kernel_list()
        position_keys = [k.position_keys for k in klist]
        assert position_keys[0][0] == "layer1"
        assert position_keys[1][0] == "layer2_loc"
        assert position_keys[2][0] == "layer3"
        assert position_keys[3][0] == "h(layer2_scale)"

    def test_custom_order_of_kernels(self):
        layer3 = lsl.Var.new_param(
            1.0, name="layer3", inference=gs.MCMCSpec(gs.NUTSKernel, order=1)
        )
        layer2_loc = lsl.Var.new_param(
            1.0,
            distribution=lsl.Dist(tfd.Normal, loc=layer3, scale=1.0),
            name="layer2_loc",
            inference=gs.MCMCSpec(gs.NUTSKernel, order=2),
        )
        layer2_scale = lsl.Var.new_param(
            1.0,
            name="layer2_scale",
            bijector=tfb.Exp(),
            inference=gs.MCMCSpec(gs.NUTSKernel, order=3),
        )
        layer1 = lsl.Var.new_param(
            1.0,
            distribution=lsl.Dist(tfd.Normal, loc=layer2_loc, scale=layer2_scale),
            name="layer1",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        model = lsl.Model([layer1])
        klist = gs.LieselMCMC(model).get_kernel_list()
        position_keys = [k.position_keys for k in klist]
        assert position_keys[3][0] == "layer1"
        assert position_keys[2][0] == "h(layer2_scale)"
        assert position_keys[1][0] == "layer2_loc"
        assert position_keys[0][0] == "layer3"

    # @pytest.mark.mcmc
    def test_run_mcmc(self):
        loc = lsl.Var.new_param(1.0, name="loc", inference=gs.MCMCSpec(gs.NUTSKernel))
        scale = lsl.Var.new_param(
            1.0,
            name="scale",
            bijector=tfb.Exp(),
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        y = lsl.Var.new_param(
            jnp.linspace(-2, 2, 50),
            distribution=lsl.Dist(tfd.Normal, loc=loc, scale=scale),
            name="layer1",
            inference=gs.MCMCSpec(gs.NUTSKernel),
        )
        model = lsl.Model([y])
        result = gs.LieselMCMC(model).run_for_epochs(
            seed=1, num_chains=4, adaptation=250, posterior=250
        )
        assert isinstance(result, gs.SamplingResults)
