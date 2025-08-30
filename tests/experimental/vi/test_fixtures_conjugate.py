"""Fixtures for conjugate prior scenarios.

Each fixture provides data and configuration for a specific conjugate pair.
"""

import jax.numpy as jnp
import pytest
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


# ============================================================================
# Bernoulli/Binomial - Beta Conjugate Pair
# ============================================================================


@pytest.fixture
def bernoulli_beta_data():
    """Data for Bernoulli/Binomial - Beta conjugate pair.

    Returns data with 40 successes and 10 failures (p=0.8).
    """
    return jnp.array([1] * 40 + [0] * 10)


@pytest.fixture
def bernoulli_beta_prior_params():
    """Prior parameters for Beta distribution.

    Beta(1, 1) - uniform prior.
    """
    return {"concentration1": 1.0, "concentration0": 1.0}


@pytest.fixture
def bernoulli_beta_posterior_params(bernoulli_beta_data):
    """Analytic posterior parameters for Beta distribution.

    alpha' = alpha + sum(x_i)
    beta' = beta + n - sum(x_i)
    """
    n = len(bernoulli_beta_data)
    sum_x = jnp.sum(bernoulli_beta_data)
    return {
        "concentration1": 1.0 + sum_x,  # 1 + 40 = 41
        "concentration0": 1.0 + n - sum_x,  # 1 + 50 - 40 = 11
    }


@pytest.fixture
def bernoulli_beta_config():
    """Configuration for Beta variational distribution with narrow variance."""
    return {
        "dist_class": tfd.Beta,
        "variational_params": {"concentration1": 0.01, "concentration0": 100.0},
    }


# ============================================================================
# Poisson - Gamma Conjugate Pair
# ============================================================================


@pytest.fixture
def poisson_gamma_data():
    """Data for Poisson - Gamma conjugate pair.

    Returns data with variability but mean = 5.0 (integer).
    Deterministic pattern: [3, 4, 5, 6, 7] repeated 10 times.
    Mean = (3+4+5+6+7)/5 = 5.0
    """
    pattern = [3, 4, 5, 6, 7]
    return jnp.array(pattern * 10)


@pytest.fixture
def poisson_gamma_prior_params():
    """Prior parameters for Gamma distribution.

    Gamma(1, 1) - weak prior with mean 1.0.
    """
    return {"concentration": 1.0, "rate": 1.0}


@pytest.fixture
def poisson_gamma_posterior_params(poisson_gamma_data):
    """Analytic posterior parameters for Gamma distribution.

    alpha' = alpha + sum(x_i)
    beta' = beta + n
    """
    n = len(poisson_gamma_data)
    sum_x = jnp.sum(poisson_gamma_data)
    return {
        "concentration": 1.0 + sum_x,  # 1 + 250 = 251
        "rate": 1.0 + n,  # 1 + 50 = 51
    }


@pytest.fixture
def poisson_gamma_config():
    """Configuration for Gamma variational distribution with narrow variance."""
    return {
        "dist_class": tfd.Gamma,
        "variational_params": {"concentration": 1.0, "rate": 20.0},
    }


# ============================================================================
# Normal (known variance) - Normal Conjugate Pair
# ============================================================================


@pytest.fixture
def normal_known_var_data():
    """Data for Normal with known variance - Normal conjugate pair.

    Returns data with variability but mean = 10.0 (integer).
    Deterministic pattern: [8, 9, 10, 11, 12] repeated 10 times.
    Mean = (8+9+10+11+12)/5 = 10.0
    """
    pattern = [8.0, 9.0, 10.0, 11.0, 12.0]
    return jnp.array(pattern * 10)


@pytest.fixture
def normal_known_var_sigma():
    """Known standard deviation for Normal distribution."""
    return 1.0


@pytest.fixture
def normal_known_var_prior_params():
    """Prior parameters for Normal distribution.

    Normal(0, 10) - weak prior centered at 0.
    """
    return {"loc": 0.0, "scale": 10.0}


@pytest.fixture
def normal_known_var_posterior_params(normal_known_var_data, normal_known_var_sigma):
    """Analytic posterior parameters for Normal distribution.

    sigma'^2 = (sigma_0^{-2} + n*sigma^{-2})^{-1}
    mu' = sigma'^2 * (mu_0/sigma_0^2 + sum(x_i)/sigma^2)
    """
    n = len(normal_known_var_data)
    sum_x = jnp.sum(normal_known_var_data)

    sigma0 = 10.0
    sigma = normal_known_var_sigma

    var0 = sigma0**2
    var = sigma**2

    var_prime = 1.0 / (1.0 / var0 + n / var)
    mu_prime = var_prime * (0.0 / var0 + sum_x / var)

    return {"loc": mu_prime, "scale": jnp.sqrt(var_prime)}


@pytest.fixture
def normal_known_var_config():
    """Configuration for Normal variational distribution with narrow variance."""
    return {
        "dist_class": tfd.Normal,
        "variational_params": {"loc": 0.0, "scale": 0.01},  # Very narrow around 10
    }


# ============================================================================
# Normal (known mean) - Inverse Gamma Conjugate Pair
# ============================================================================


@pytest.fixture
def normal_known_mean_data():
    """Data for Normal with known mean - Inverse Gamma conjugate pair.

    Returns data with variance = 4.0 around mean = 5.0.
    """
    return jnp.array([3.0] * 25 + [7.0] * 25)


@pytest.fixture
def normal_known_mean_mu():
    """Known mean for Normal distribution."""
    return 5.0


@pytest.fixture
def normal_known_mean_prior_params():
    """Prior parameters for Inverse Gamma distribution.

    InverseGamma(0.01, 0.01) - weak prior.
    """
    return {"concentration": 0.01, "scale": 0.01}


@pytest.fixture
def normal_known_mean_posterior_params(normal_known_mean_data, normal_known_mean_mu):
    """Analytic posterior parameters for Inverse Gamma distribution.

    alpha' = alpha + n/2
    beta' = beta + 0.5 * sum((x_i - mu)^2)
    """
    n = len(normal_known_mean_data)
    sum_sq_diff = jnp.sum((normal_known_mean_data - normal_known_mean_mu) ** 2)

    return {
        "concentration": 0.01 + n / 2,  # 0.01 + 25 = 25.01
        "scale": 0.01 + 0.5 * sum_sq_diff,  # 0.01 + 0.5 * 200 = 100.01
    }


@pytest.fixture
def normal_known_mean_config():
    """Configuration for Inverse Gamma variational distribution with narrow variance."""
    return {
        "dist_class": tfd.InverseGamma,
        "variational_params": {"concentration": 10.0, "scale": 1.0},
    }


# ============================================================================
# Categorical/Multinomial - Dirichlet Conjugate Pair
# ============================================================================


@pytest.fixture
def categorical_dirichlet_data():
    """Data for Categorical/Multinomial - Dirichlet conjugate pair.

    Returns data with category counts [5, 15, 30] (probabilities [0.1, 0.3, 0.6]).
    """
    return jnp.array([0] * 5 + [1] * 15 + [2] * 30)


@pytest.fixture
def categorical_dirichlet_prior_params():
    """Prior parameters for Dirichlet distribution.

    Dirichlet(1, 1, 1) - uniform prior over 3 categories.
    """
    return {"concentration": jnp.array([1.0, 1.0, 1.0])}


@pytest.fixture
def categorical_dirichlet_posterior_params(categorical_dirichlet_data):
    """Analytic posterior parameters for Dirichlet distribution.

    alpha_i' = alpha_i + c_i (count in category i)
    """
    counts = jnp.array(
        [
            jnp.sum(categorical_dirichlet_data == 0),
            jnp.sum(categorical_dirichlet_data == 1),
            jnp.sum(categorical_dirichlet_data == 2),
        ]
    )

    return {
        "concentration": jnp.array([1.0, 1.0, 1.0]) + counts  # [6, 16, 31]
    }


@pytest.fixture
def categorical_dirichlet_config():
    """Configuration for Dirichlet variational distribution with narrow variance."""
    return {
        "dist_class": tfd.Dirichlet,
        "variational_params": {"concentration": jnp.array([10.0, 10.0, 10.0])},
    }


# ============================================================================
# MV-Normal (known covariance) - MV-Normal Conjugate Pair
# ============================================================================


@pytest.fixture
def mvn_known_cov_data():
    """Data for MV-Normal with known covariance - MV-Normal conjugate pair.

    Returns data with mean = [10.0, -10.0].
    """
    return jnp.array([[10.0, -10.0]] * 50)


@pytest.fixture
def mvn_known_cov_sigma():
    """Known covariance matrix (identity)."""
    return jnp.eye(2)


@pytest.fixture
def mvn_known_cov_prior_params():
    """Prior parameters for MV-Normal distribution.

    MVN([0, 0], 100*I) - weak prior centered at origin.
    """
    return {"loc": jnp.array([0.0, 0.0]), "covariance_matrix": 100.0 * jnp.eye(2)}


@pytest.fixture
def mvn_known_cov_posterior_params(mvn_known_cov_data, mvn_known_cov_sigma):
    """Analytic posterior parameters for MV-Normal distribution.

    Sigma' = (Sigma_0^{-1} + n*Sigma^{-1})^{-1}
    mu' = Sigma' * (Sigma_0^{-1}*mu_0 + n*Sigma^{-1}*x_bar)
    """
    n = len(mvn_known_cov_data)
    x_bar = jnp.mean(mvn_known_cov_data, axis=0)

    sigma0 = 100.0 * jnp.eye(2)
    sigma = mvn_known_cov_sigma
    mu0 = jnp.array([0.0, 0.0])

    sigma0_inv = jnp.linalg.inv(sigma0)
    sigma_inv = jnp.linalg.inv(sigma)

    sigma_prime = jnp.linalg.inv(sigma0_inv + n * sigma_inv)
    mu_prime = sigma_prime @ (sigma0_inv @ mu0 + n * sigma_inv @ x_bar)

    return {"loc": mu_prime, "covariance_matrix": sigma_prime}


@pytest.fixture
def mvn_known_cov_config():
    """Configuration for MV-Normal variational distribution using TriL
    parameterization with narrow variance."""
    return {
        "dist_class": tfd.MultivariateNormalTriL,
        "variational_params": {
            "loc": jnp.array([0.0, 0.0]),
            "scale_tril": 0.01 * jnp.eye(2),
        },
    }
