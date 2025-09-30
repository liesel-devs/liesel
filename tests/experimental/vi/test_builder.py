"""Unit tests for the OptimizerBuilder class."""

from unittest.mock import Mock

import jax.numpy as jnp
import optax
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel.experimental.vi import LieselInterface, OptimizerBuilder

# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def mock_model_interface():
    """Create a mock LieselInterface parameters for testing."""
    interface = Mock(spec=LieselInterface)
    interface.get_params.return_value = {
        "param1": jnp.array([1.0, 2.0]),
        "param2": jnp.array(3.0),
        "param3": jnp.ones((4,)),
        "sigma_sq_transformed": jnp.array(1.0),
        "b": jnp.array([1.0, 2.0, 3.0, 4.0]),
    }

    # Create a mock observed variable with required attributes
    mock_observed_var = Mock()
    mock_observed_var.observed = True
    mock_observed_var.value = Mock()
    mock_observed_var.value.shape = (100,)  # Mock shape for observed data

    # Create mock unobserved variables
    mock_unobserved_var1 = Mock()
    mock_unobserved_var1.observed = False
    mock_unobserved_var1.value = Mock()
    mock_unobserved_var1.value.shape = (2,)

    mock_unobserved_var2 = Mock()
    mock_unobserved_var2.observed = False
    mock_unobserved_var2.value = Mock()
    mock_unobserved_var2.value.shape = (1,)

    # Create mock model with vars that can be iterated
    mock_model = Mock()
    mock_model.vars = {
        "y_obs": mock_observed_var,
        "param1": mock_unobserved_var1,
        "param2": mock_unobserved_var2,
    }
    interface.model = mock_model
    return interface


@pytest.fixture
def builder_with_defaults():
    """Create an OptimizerBuilder with default parameters."""
    return OptimizerBuilder()


@pytest.fixture
def builder_with_custom_params():
    """Create an OptimizerBuilder with custom parameters."""
    return OptimizerBuilder(
        seed=42, n_epochs=5000, S=64, patience_tol=0.01, window_size=50, batch_size=32
    )


@pytest.fixture
def optimizer_chain():
    """Create a reusable optimizer chain."""
    return optax.adam(learning_rate=0.001)


# --- Unit Tests - Initialization ------------------------------------------


class TestOptimizerBuilderInitialization:
    """Test OptimizerBuilder initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        builder = OptimizerBuilder()

        assert builder.seed == 0
        assert builder.n_epochs == 10_000
        assert builder.S == 32
        assert builder.patience_tol is None
        assert builder.window_size is None
        assert builder.batch_size is None
        assert builder._model_interface is None
        assert builder.latent_variables == []

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        builder = OptimizerBuilder(
            seed=123,
            n_epochs=5000,
            S=64,
            patience_tol=0.001,
            window_size=100,
            batch_size=16,
        )

        assert builder.seed == 123
        assert builder.n_epochs == 5000
        assert builder.S == 64
        assert builder.patience_tol == 0.001
        assert builder.window_size == 100
        assert builder.batch_size == 16

    @pytest.mark.skip(reason="Functionality not implemented yet")
    @pytest.mark.parametrize(
        "seed, n_epochs, S, patience_tol, window_size, batch_size",
        [
            # invalid seed
            (-1, 10, 32, 0.1, 50, 16),
            # invalid n_epochs
            (1, -1, 32, 0.1, 50, 16),
            (1, 0, 32, 0.1, 50, 16),
            # invalid S
            (1, 10, -1, 0.1, 50, 16),
            (1, 10, 0, 0.1, 50, 16),
            # invalid patience_tol
            (1, 10, 32, -0.001, 50, 16),
            # invalid window_size
            (1, 10, 32, 0.1, -5, 16),
            (1, 10, 32, 0.1, 0, 16),
            # invalid batch_size
            (1, 10, 32, 0.1, 50, -16),
            (1, 10, 32, 0.1, 50, 0),
        ],
    )
    def test_init_parameter_variations(
        self, seed, n_epochs, S, patience_tol, window_size, batch_size
    ):
        """Test initialization with invalid parameter combinations."""
        builder = OptimizerBuilder(
            seed=seed,
            n_epochs=n_epochs,
            S=S,
            patience_tol=patience_tol,
            window_size=window_size,
            batch_size=batch_size,
        )

        assert builder.seed == seed
        assert builder.n_epochs == n_epochs
        assert builder.S == S
        assert builder.patience_tol == patience_tol
        assert builder.window_size == window_size
        assert builder.batch_size == batch_size


# -- Unit Tests - set_model Method ------------------------------------------


class TestSetModel:
    """Test the set_model method."""

    def test_set_model_with_valid_interface(
        self, builder_with_defaults, mock_model_interface
    ):
        """Test setting a valid model interface."""
        builder_with_defaults.set_model(mock_model_interface)
        assert builder_with_defaults._model_interface is mock_model_interface

    def test_set_model_overwrites_previous(self, builder_with_defaults):
        """Test that setting a new model overwrites the previous one."""
        first_interface = Mock(spec=LieselInterface)
        second_interface = Mock(spec=LieselInterface)

        builder_with_defaults.set_model(first_interface)
        assert builder_with_defaults._model_interface is first_interface

        builder_with_defaults.set_model(second_interface)
        assert builder_with_defaults._model_interface is second_interface


# --- Unit Tests - add_variational_dist Method --------------------------------


class TestAddVariationalDist:
    """Test the add_variational_dist method."""

    def test_add_single_variational_dist(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding a single variational distribution."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        assert len(builder_with_defaults.latent_variables) == 1
        config = builder_with_defaults.latent_variables[0]
        assert config["names"] == ["param1"]
        assert config["dist_class"] == tfd.Normal
        assert config["full_rank_key"] == "param1"
        assert config["dims_list"] == [2]
        assert config["split_indices"] == []

    def test_add_variational_dist_string_name(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with string name instead of list."""
        builder_with_defaults.set_model(mock_model_interface)

        # Pass string instead of list
        builder_with_defaults.add_variational_dist(
            "param2",  # String instead of list
            dist_class=tfd.Normal,
            variational_params={"loc": 0.0, "scale": 1.0},
            optimizer_chain=optimizer_chain,
        )

        assert len(builder_with_defaults.latent_variables) == 1
        config = builder_with_defaults.latent_variables[0]
        assert config["names"] == ["param2"]  # Should be converted to list

    def test_add_multiple_variational_dists(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding multiple variational distributions."""
        builder_with_defaults.set_model(mock_model_interface)

        # Add first distribution
        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        # Add second distribution
        builder_with_defaults.add_variational_dist(
            ["param2"],
            dist_class=tfd.Gamma,
            variational_params={"concentration": 1.0, "rate": 1.0},
            optimizer_chain=optimizer_chain,
        )

        assert len(builder_with_defaults.latent_variables) == 2

    def test_add_variational_dist_with_fixed_params(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with fixed parameters."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2)},
            fixed_distribution_params={"scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        config = builder_with_defaults.latent_variables[0]
        # Use array_equal for JAX array comparison
        assert jnp.array_equal(
            config["fixed_distribution_params"]["scale"], jnp.ones(2)
        )

    def test_add_variational_dist_with_custom_bijectors(
        self,
        builder_with_defaults,
        mock_model_interface,
        optimizer_chain,
    ):
        """Test adding variational distribution with custom bijectors."""
        builder_with_defaults.set_model(mock_model_interface)

        # Use Exp for loc and Identity for scale since they cannot be default bijectors,
        # making them identifiable as custom.
        custom_bijectors = {"loc": tfb.Exp(), "scale": tfb.Identity()}

        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
            variational_param_bijectors=custom_bijectors,
        )

        config = builder_with_defaults.latent_variables[0]
        assert (
            config["variational_param_bijectors"]["scale"] is custom_bijectors["scale"]
        )

    def test_add_variational_dist_multiple_latent_vars_two(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution for two latent variables (full rank)."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1", "param2"],
            dist_class=tfd.MultivariateNormalFullCovariance,
            variational_params={"loc": jnp.zeros(3), "covariance_matrix": jnp.eye(3)},
            optimizer_chain=optimizer_chain,
        )

        config = builder_with_defaults.latent_variables[0]
        assert config["names"] == ["param1", "param2"]
        assert config["full_rank_key"] == "param1_param2"
        assert config["dims_list"] == [2, 1]  # param1: 2, param2: 1
        assert config["split_indices"] == [2]  # Split at index 2

    def test_add_variational_dist_multiple_latent_vars_three(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution for three latent variables.

        This test verifies the correct splits are calculated.
        """
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1", "param2", "param3"],  # Three variables
            dist_class=tfd.MultivariateNormalFullCovariance,
            variational_params={"loc": jnp.zeros(7), "covariance_matrix": jnp.eye(7)},
            optimizer_chain=optimizer_chain,
        )

        config = builder_with_defaults.latent_variables[0]
        assert config["names"] == ["param1", "param2", "param3"]
        assert config["full_rank_key"] == "param1_param2_param3"
        assert config["dims_list"] == [2, 1, 4]  # param1: 2, param2: 1, param3: 4
        assert config["split_indices"] == [2, 3]  # Split at indices 2 and 3

    def test_add_variational_dist_invalid_param_key(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with invalid parameter key."""
        builder_with_defaults.set_model(mock_model_interface)

        with pytest.raises(
            ValueError, match=r"Invalid key\(s\) in 'variational_params'"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={"invalid_param": 0.0},  # Invalid parameter name
                optimizer_chain=optimizer_chain,
            )

    def test_add_variational_dist_invalid_bijector_key(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with invalid bijector key."""
        builder_with_defaults.set_model(mock_model_interface)

        with pytest.raises(
            ValueError, match=r"Invalid key\(s\) in 'parameter_bijectors'"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
                optimizer_chain=optimizer_chain,
                variational_param_bijectors={"invalid_key": tfb.Identity()},
            )

    def test_add_variational_dist_unknown_variable(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with unknown latent variable."""
        builder_with_defaults.set_model(mock_model_interface)

        with pytest.raises(KeyError, match="Parameter does_not_exist not found"):
            builder_with_defaults.add_variational_dist(
                ["does_not_exist"],
                dist_class=tfd.Normal,
                variational_params={"loc": 0.0, "scale": 1.0},
                optimizer_chain=optimizer_chain,
            )

    def test_add_variational_dist_param_shape_mismatch(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with mismatched parameter shapes."""
        builder_with_defaults.set_model(mock_model_interface)

        with pytest.raises(
            ValueError, match="Failed to build variational distribution"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={
                    "loc": jnp.zeros(2),
                    "scale": jnp.ones(3),
                },  # Shape mismatch
                optimizer_chain=optimizer_chain,
            )

    def test_add_variational_dist_wrong_bijector_type(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding variational distribution with invalid bijector type."""
        builder_with_defaults.set_model(mock_model_interface)

        class NotABijector:
            """Invalid bijector without forward/inverse methods."""

            pass

        with pytest.raises(
            ValueError, match="Failed to build variational distribution"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
                optimizer_chain=optimizer_chain,
                variational_param_bijectors={"scale": NotABijector()},
            )

    def test_add_variational_dist_non_reparameterized(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding non-reparameterized distribution raises error."""
        builder_with_defaults.set_model(mock_model_interface)

        # Bernoulli is not fully reparameterized
        with pytest.raises(NotImplementedError, match="Only fully reparameterized"):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Bernoulli,
                variational_params={"probs": 0.5},
                optimizer_chain=optimizer_chain,
            )

    def test_add_variational_dist_missing_required_param(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test adding distribution with missing required parameter."""
        builder_with_defaults.set_model(mock_model_interface)

        with pytest.raises(
            ValueError, match="Failed to build variational distribution"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={"loc": jnp.zeros(2)},  # Missing scale
                optimizer_chain=optimizer_chain,
            )

    def test_add_variational_dist_maintains_order(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test that latent variables are stored in order of addition."""
        builder_with_defaults.set_model(mock_model_interface)

        # Add three variables in specific order
        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        builder_with_defaults.add_variational_dist(
            ["param2"],
            dist_class=tfd.Gamma,
            variational_params={"concentration": 1.0, "rate": 1.0},
            optimizer_chain=optimizer_chain,
        )

        builder_with_defaults.add_variational_dist(
            ["param3"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(4), "scale": jnp.ones(4)},
            optimizer_chain=optimizer_chain,
        )

        # Verify order is preserved
        assert len(builder_with_defaults.latent_variables) == 3
        assert builder_with_defaults.latent_variables[0]["names"] == ["param1"]
        assert builder_with_defaults.latent_variables[1]["names"] == ["param2"]
        assert builder_with_defaults.latent_variables[2]["names"] == ["param3"]

    def test_latent_variable_config_has_all_required_keys(
        self, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Ensure all required keys are present in latent variable config."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        config = builder_with_defaults.latent_variables[0]
        required_keys = [
            "names",
            "dist_class",
            "variational_params",
            "fixed_distribution_params",
            "optimizer_chain",
            "variational_param_bijectors",
            "full_rank_key",
            "event_shape",
            "variable_dims",
            "dims_list",
            "split_indices",
        ]

        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

        # Additional assertions for key types/values
        assert isinstance(config["names"], list)
        assert config["dist_class"] == tfd.Normal
        assert isinstance(config["variational_params"], dict)
        assert isinstance(config["dims_list"], list)
        assert isinstance(config["split_indices"], list)

    @pytest.mark.xfail(
        strict=True,
        reason="Don't store Bijectors for fixed dist params (TODO: implement)",
    )
    def test_no_bijectors_stored_for_fixed_params(
        builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test that bijectors are not stored for fixed distribution parameters.

        Currently fails because the builder stores bijectors for all parameters,
        even fixed ones. Fixed parameters don't need bijector transformations
        since they're not optimized.
        """
        builder_with_defaults.set_model(mock_model_interface)

        # Add distribution with both variational and fixed parameters
        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.array(1.0)},  # Variational parameter
            fixed_distribution_params={"scale": jnp.array(2.0)},  # Fixed parameter
            optimizer_chain=optimizer_chain,
            variational_param_bijectors={
                "loc": tfb.Identity(),  # Should be kept for variational param
                "scale": tfb.Softplus(),  # Should be removed for fixed param
            },
        )

        # Check the stored configuration
        config = builder_with_defaults.latent_variables[0]

        # The bijector for 'loc' should be present (it's a variational param)
        assert "loc" in config["variational_param_bijectors"]

        # The bijector for 'scale' should NOT be present (it's a fixed param)
        assert "scale" not in config["variational_param_bijectors"]

    @pytest.mark.xfail(
        strict=True,
        reason="Reject bijectors for fixed distribution parameters (TODO: validation)",
    )
    def test_rejects_bijectors_for_fixed_params(
        builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test that providing bijectors for fixed parameters raises an error.

        When a parameter is specified as fixed, users should not be allowed to
        provide bijectors for it, as it won't be optimized.
        """
        builder_with_defaults.set_model(mock_model_interface)

        # This should raise an error because 'scale' is fixed but has a bijector
        with pytest.raises(
            ValueError, match=r"Cannot specify bijectors for fixed.*scale"
        ):
            builder_with_defaults.add_variational_dist(
                ["param1"],
                dist_class=tfd.Normal,
                variational_params={"loc": jnp.array(1.0)},
                fixed_distribution_params={"scale": jnp.array(2.0)},  # scale is fixed
                optimizer_chain=optimizer_chain,
                variational_param_bijectors={
                    "loc": tfb.Identity(),
                    "scale": tfb.Softplus(),  # This should cause an error
                },
            )

    @pytest.mark.xfail(
        strict=True,
        reason="Don't apply default bijectors to fixed dist params (TODO: implement)",
    )
    def test_default_bijectors_not_applied_to_fixed_params(
        builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test that default bijectors are not applied to fixed distribution parameters.

        When users don't provide custom bijectors, the builder uses default bijectors.
        However, default bijectors should only be applied to variational parameters,
        not fixed ones.
        """
        builder_with_defaults.set_model(mock_model_interface)

        # Add distribution with fixed scale parameter, no custom bijectors provided
        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.array(1.0)},  # Variational parameter
            fixed_distribution_params={"scale": jnp.array(2.0)},  # Fixed parameter
            optimizer_chain=optimizer_chain,
            # No variational_param_bijectors provided -
            # should use defaults only for variational params
        )

        # Check the stored configuration
        config = builder_with_defaults.latent_variables[0]

        # Default bijector for 'loc' should be present (it's a variational param)
        assert "loc" in config["variational_param_bijectors"]

        # Default bijector for 'scale' should NOT be present (it's a fixed param)
        assert "scale" not in config["variational_param_bijectors"]


# -- Unit Tests - build Method --


class TestBuildMethod:
    """Test the build method with decoupled Optimizer testing."""

    def test_build_without_model_raises_error(self, builder_with_defaults):
        """Test that building without setting model raises error."""
        with pytest.raises(ValueError, match="Model interface not set"):
            builder_with_defaults.build()

    def test_build_passes_config_to_optimizer_no_latent_vars(
        self, monkeypatch, builder_with_defaults, mock_model_interface
    ):
        """Test build passes correct configuration with no latent variables."""
        builder_with_defaults.set_model(mock_model_interface)

        captured = {}

        class DummyOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                # Store these as attributes so existing assertions work
                for key, value in kwargs.items():
                    setattr(self, key, value)
                # Special handling for latent_vars_config
                self.latent_vars_config = kwargs.get("latent_variables", {})

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        builder_with_defaults.build()

        assert captured["seed"] == 0
        assert captured["n_epochs"] == 10_000
        assert captured["S"] == 32
        assert captured["model_interface"] is mock_model_interface
        assert captured["latent_variables"] == {}

    def test_build_passes_config_to_optimizer_single_latent_var(
        self, monkeypatch, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test build passes correct configuration with single latent variable."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        captured = {}

        class DummyOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        builder_with_defaults.build()

        assert list(captured["latent_variables"].keys()) == ["param1"]
        config = captured["latent_variables"]["param1"]
        assert config["names"] == ["param1"]
        assert config["dist_class"] == tfd.Normal

    def test_build_passes_config_to_optimizer_multiple_latent_vars(
        self, monkeypatch, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test build passes correct configuration with multiple latent variables."""
        builder_with_defaults.set_model(mock_model_interface)

        # Add first variable
        builder_with_defaults.add_variational_dist(
            ["param1"],
            dist_class=tfd.Normal,
            variational_params={"loc": jnp.zeros(2), "scale": jnp.ones(2)},
            optimizer_chain=optimizer_chain,
        )

        # Add second variable
        builder_with_defaults.add_variational_dist(
            ["param2"],
            dist_class=tfd.Normal,
            variational_params={"loc": 0.0, "scale": 1.0},
            optimizer_chain=optimizer_chain,
        )

        captured = {}

        class DummyOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.latent_vars_config = kwargs.get("latent_variables", {})

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        builder_with_defaults.build()

        assert set(captured["latent_variables"].keys()) == {"param1", "param2"}

    def test_build_passes_config_to_optimizer_full_rank(
        self, monkeypatch, builder_with_defaults, mock_model_interface, optimizer_chain
    ):
        """Test build passes correct configuration with full rank variables."""
        builder_with_defaults.set_model(mock_model_interface)

        builder_with_defaults.add_variational_dist(
            ["param1", "param2"],
            dist_class=tfd.MultivariateNormalFullCovariance,
            variational_params={"loc": jnp.zeros(3), "covariance_matrix": jnp.eye(3)},
            optimizer_chain=optimizer_chain,
        )

        captured = {}

        class DummyOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.latent_vars_config = kwargs.get("latent_variables", {})

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        builder_with_defaults.build()

        assert list(captured["latent_variables"].keys()) == ["param1_param2"]

    def test_build_preserves_custom_configuration(
        self, monkeypatch, builder_with_custom_params, mock_model_interface
    ):
        """Test that build preserves all custom configuration parameters."""
        builder_with_custom_params.set_model(mock_model_interface)

        captured = {}

        class DummyOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)
                self.latent_vars_config = kwargs.get("latent_variables", {})

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        builder_with_custom_params.build()

        assert captured["seed"] == 42
        assert captured["n_epochs"] == 5000
        assert captured["S"] == 64
        assert captured["patience_tol"] == 0.01
        assert captured["window_size"] == 50
        assert captured["batch_size"] == 32

    def test_build_multiple_times_returns_new_instances(
        self, monkeypatch, builder_with_defaults, mock_model_interface
    ):
        """Document that calling build() multiple times returns new Optimizer instances.

        Note: This documents current behavior. The builder doesn't prevent or
        explicitly handle multiple builds, so each call creates a new Optimizer.
        """
        builder_with_defaults.set_model(mock_model_interface)

        class DummyOptimizer:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr("liesel.experimental.vi.builder.Optimizer", DummyOptimizer)

        # Build twice
        optimizer1 = builder_with_defaults.build()
        optimizer2 = builder_with_defaults.build()

        # Each call returns a new instance
        assert optimizer1 is not optimizer2
        assert isinstance(optimizer1, DummyOptimizer)
        assert isinstance(optimizer2, DummyOptimizer)


# -- Unit Tests - Minimal tests for default bijectors -----------------------


class TestDefaultBijectors:
    """Minimal tests for default bijector retrieval."""

    @pytest.mark.parametrize(
        "dist_class,expected_params",
        [
            (tfd.Normal, ["loc", "scale"]),
            (tfd.Gamma, ["concentration", "rate"]),
            (tfd.Beta, ["concentration1", "concentration0"]),
        ],
    )
    def test_obtain_default_bijectors_has_all_params(
        self, builder_with_defaults, dist_class, expected_params
    ):
        """Test that default bijectors contain all distribution parameters."""
        bijectors = builder_with_defaults._obtain_parameter_default_bijectors(
            dist_class
        )

        for param in expected_params:
            assert param in bijectors
            assert bijectors[param] is not None


# -- Unit Tests - Dimensionality validation placeholder --


@pytest.mark.xfail(
    strict=True, reason="Dimensionality validation not implemented yet (TODO in code)"
)
def test_validate_dimensionality_will_check_total_dims():
    """Test that dimensionality validation will check total dimensions."""
    builder = OptimizerBuilder()

    # When implemented, this should raise ValueError for dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        builder._validate_dimensionality(
            latent_variable_names=["param1"],
            event_shape=5,  # Intentionally mismatched
            variable_dims={"param1": 2},
        )


# -- Integration Tests - Returning Optimizer ---------------------------------


def test_build_returns_optimizer_instance(
    builder_with_defaults, mock_model_interface, optimizer_chain
):
    """Test that build returns an Optimizer instance using proper mocks."""
    from liesel.experimental.vi.optimizer import Optimizer

    # Set the mock model interface
    builder_with_defaults.set_model(mock_model_interface)

    # Add variational distributions with proper structure
    builder_with_defaults.add_variational_dist(
        ["sigma_sq_transformed"],
        dist_class=tfd.Normal,
        variational_params={"loc": 1.0, "scale": 0.5},
        optimizer_chain=optimizer_chain,
        variational_param_bijectors={
            "loc": tfb.Identity(),
            "scale": tfb.Softplus(),
        },
    )

    builder_with_defaults.add_variational_dist(
        ["b"],
        dist_class=tfd.Normal,
        variational_params={"loc": jnp.zeros(4), "scale": jnp.ones(4)},
        optimizer_chain=optimizer_chain,
    )

    # Build and verify the optimizer is returned
    optimizer = builder_with_defaults.build()
    assert isinstance(optimizer, Optimizer)
