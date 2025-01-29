"""
Tests for the multivariate normal degenerate distribution.
"""

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.random as jrd
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import jit
from scipy.interpolate import BSpline

import liesel.goose as gs
import liesel.model as lsl
from liesel.distributions.mvn_degen import (
    Array,
    MultivariateNormalDegenerate,
    _log_pdet,
    _rank,
)

key = jrd.PRNGKey(24605)


def determinant_structure_degen(R: Array, tol: float = 1e-6) -> tuple[float, float]:
    """
    Calculates the generalized log determinant based on the eigenvalue decomposition.
    """
    evals = np.linalg.eigvalsh(R)
    mask = evals > tol
    selected = np.where(mask, evals, 1.0)
    lgdet = np.sum(np.log(selected))
    rank = mask.sum()
    return lgdet, rank


def mvnorm_degen_logpdf(
    x: Array,
    mu: Array,
    precision: Array,
    log_det: tuple[float, float],
    sigma2: float,
    prop_for_x_mu: bool = False,
):
    """
    Calculates the log pdf of a potentially degenerated multivariate normal
    distribution.

    This is a baseline implementation for testing.

    If ``log_det`` is not ``None``, the generalized log determinant is calculated by
    ``log_det[0] - log_det[1] * np.log(sigma2) = log(|sigma2 * R|)`` where ``R`` is the
    'structure' of the precision. Therefore ``log_det[0]`` is the log determinant of
    ``R`` and ``log_det[1]`` the rank of ``R``.
    """
    assert len(x.shape) == 1
    assert x.shape == mu.shape

    xmu = x - mu
    t1 = (-0.5 * xmu @ precision @ xmu).squeeze()
    if prop_for_x_mu:
        return t1

    if log_det:
        rank_prec = log_det[1]
        gen_log_det = log_det[0]
        gen_log_det += -rank_prec * np.log(sigma2)  # type: ignore
    else:
        gen_log_det, rank_prec = determinant_structure_degen(precision)

    t0 = 0.5 * (gen_log_det - rank_prec * np.log(2.0 * np.pi))

    return t0 + t1


def partial_first_diff_matrix(ncol: int) -> Array:
    return jnp.eye(ncol - 2, ncol, k=1) - jnp.eye(ncol - 2, ncol, k=2)


@pytest.fixture(scope="module")
def beta():
    yield jrd.normal(key, (5,))


@pytest.fixture(scope="module")
def K(beta):
    D = partial_first_diff_matrix(len(beta))
    yield D.T @ D


@pytest.fixture(scope="module")
def tau2():
    yield 3.0


@pytest.fixture(scope="module")
def log_prob_baseline(beta, K, tau2):
    log_det, rank = determinant_structure_degen(K)
    log_prob = mvnorm_degen_logpdf(
        beta,
        mu=jnp.zeros(len(beta)),
        precision=K / tau2,
        sigma2=tau2,
        log_det=(log_det, rank),
    )
    yield log_prob


@pytest.fixture(scope="module")
def mvn_batch():
    K = jnp.eye(5)
    prec1 = K / 1.0
    prec2 = K / 2.0
    prec3 = K / 3.0

    prec = jnp.array([[prec1, prec2, prec3]])  # shape (1, 3, 5, 5)
    loc = jnp.array([[[0.0]], [[5.0]]])  # shape (2, 1, 1)

    yield MultivariateNormalDegenerate(loc, prec=prec)


class TestComputePseudoLogDet:
    def test_log_pdet_with_and_without_rank(self) -> None:
        """
        Test that pseudo-log-determinant computation works with and without given rank.
        """
        K = jnp.eye(5)
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec3 = K / 3.0

        ldet1a = _log_pdet(jnpla.eigvalsh(prec1))
        ldet2a = _log_pdet(jnpla.eigvalsh(prec2))
        ldet3a = _log_pdet(jnpla.eigvalsh(prec3))

        ldet1b = _log_pdet(jnpla.eigvalsh(prec1), rank=5)
        ldet2b = _log_pdet(jnpla.eigvalsh(prec2), rank=5)
        ldet3b = _log_pdet(jnpla.eigvalsh(prec3), rank=5)

        assert ldet1a == pytest.approx(ldet1b)
        assert ldet2a == pytest.approx(ldet2b)
        assert ldet3a == pytest.approx(ldet3b)

    def test_jit_log_pdet(self) -> None:
        """Test that pseudo-log-determinant computation can be jitted."""
        K = jnp.eye(5)
        prec = K / 2.0
        evals = jnpla.eigvalsh(prec)
        jitted_log_pdet = jit(_log_pdet)
        ldet1 = _log_pdet(evals)
        ldet2 = jitted_log_pdet(evals)
        ldet3 = jitted_log_pdet(evals, rank=5)

        assert ldet1 == pytest.approx(ldet2)
        assert ldet1 == pytest.approx(ldet3)

    def test_log_pdet_batch_full_computation(self) -> None:
        """Test that pseudo-log-determinant computation works for batches."""
        K1 = jnp.diag(jnp.array([2.0, 0.0, 0.0]))
        K2 = jnp.diag(jnp.array([2.0, 3.0, 0.0]))

        evals1 = jnpla.eigvalsh(K1)
        evals2 = jnpla.eigvalsh(K2)

        ldet1 = _log_pdet(evals1)
        ldet2 = _log_pdet(evals2)

        evals_batch = jnpla.eigvalsh(jnp.array([K1, K2]))

        ldet = _log_pdet(evals_batch)

        assert ldet1 == pytest.approx(ldet[0])  # type: ignore
        assert ldet2 == pytest.approx(ldet[1])  # type: ignore

    def test_log_pdet_batch_given_rank(self) -> None:
        """
        Test that pseudo-log-determinant computation with given ranks works for simple
        batches.
        """
        K1 = jnp.diag(jnp.array([2.0, 0.0, 0.0]))
        K2 = jnp.diag(jnp.array([2.0, 3.0, 0.0]))

        evals1 = jnpla.eigvalsh(K1)
        evals2 = jnpla.eigvalsh(K2)

        ldet1 = _log_pdet(evals1)
        ldet2 = _log_pdet(evals2)

        evals_batch = jnpla.eigvalsh(jnp.array([K1, K2]))
        rank_batch = _rank(evals_batch)

        ldet = jit(_log_pdet)(evals_batch, rank=rank_batch)

        assert ldet1 == pytest.approx(ldet[0])
        assert ldet2 == pytest.approx(ldet[1])

    def test_rank_and_log_pdet_big_batch_given_rank(self) -> None:
        """
        Test that pseudo-log-determinant computation with given ranks works for nested
        batches.
        """
        K1 = jnp.diag(jnp.array([2.0, 0.0, 0.0]))
        K2 = jnp.diag(jnp.array([2.0, 3.0, 0.0]))
        K3 = jnp.diag(jnp.array([4.0, 3.0, 1.0]))
        K4 = jnp.diag(jnp.array([8.0, 1.0, 3.0]))

        evals1 = jnpla.eigvalsh(K1)
        evals2 = jnpla.eigvalsh(K2)
        evals3 = jnpla.eigvalsh(K3)
        evals4 = jnpla.eigvalsh(K4)

        ldet1 = _log_pdet(evals1)
        ldet2 = _log_pdet(evals2)
        ldet3 = _log_pdet(evals3)
        ldet4 = _log_pdet(evals4)

        evals_batch = jnpla.eigvalsh(jnp.array([[K1, K2], [K3, K4]]))
        _rank_batch = _rank(evals_batch)

        ldet = _log_pdet(evals_batch, rank=_rank_batch)

        assert ldet1 == pytest.approx(ldet[0, 0])  # type: ignore
        assert ldet2 == pytest.approx(ldet[0, 1])  # type: ignore
        assert ldet3 == pytest.approx(ldet[1, 0])  # type: ignore
        assert ldet4 == pytest.approx(ldet[1, 1])  # type: ignore


class TestMVNDegenerateValues:
    """
    Tests that :class:`.MultivariateNormalDegenerate` computes the log probability
    correctly.
    """

    def test_log_prob_from_penalty(self, beta, K, tau2, log_prob_baseline) -> None:
        """
        Log prob obtained from MultivariateNormalDegenerate constructed from penalty
        matrix should be equal to the baseline log prob.
        """
        log_det, rank = determinant_structure_degen(K)

        mvn = MultivariateNormalDegenerate.from_penalty(
            loc=0.0, var=tau2, pen=K, rank=rank, log_pdet=log_det
        )
        log_prob = mvn.log_prob(beta)

        assert log_prob == pytest.approx(log_prob_baseline)

    def test_log_prob_from_penalty_smooth(
        self, beta, K, tau2, log_prob_baseline
    ) -> None:
        """
        Log prob obtained from MultivariateNormalDegenerate constructed from penalty
        matrix should be equal to the baseline log prob.
        """
        log_det, rank = determinant_structure_degen(K)

        mvn = MultivariateNormalDegenerate.from_penalty_smooth(
            loc=0.0, smooth=1 / tau2, pen=K, rank=rank, log_pdet=log_det
        )
        log_prob = mvn.log_prob(beta)

        assert log_prob == pytest.approx(log_prob_baseline)

    def test_log_prob_from_init(self, beta, K, tau2, log_prob_baseline) -> None:
        """
        Log prob obtained from MultivariateNormalDegenerate constructed directly from
        the init should be equal to the baseline log prob.
        """
        prec = K / tau2
        log_det, rank = determinant_structure_degen(prec)

        mvn = MultivariateNormalDegenerate(
            loc=0.0, prec=prec, rank=rank, log_pdet=log_det
        )

        log_prob = mvn.log_prob(beta)

        assert log_prob == pytest.approx(log_prob_baseline)

        lp = MultivariateNormalDegenerate(loc=0.0, prec=prec).log_prob(beta)
        assert lp == pytest.approx(log_prob_baseline)


class TestMVNDegenerateBatches:
    def test_shape_single_sample(self, beta, K, tau2) -> None:
        """Log prob should be a scalar."""
        mvn = MultivariateNormalDegenerate(loc=0.0, prec=K / tau2)
        log_prob = mvn.log_prob(beta)
        assert jnp.ndim(log_prob) == 0

    def test_shape_two_samples(self, beta, K, tau2) -> None:
        """Log prob should have shape (2,)."""
        mvn = MultivariateNormalDegenerate(loc=0.0, prec=K / tau2)
        log_prob = mvn.log_prob([beta, beta])
        assert log_prob.shape == (2,)

    def test_shape_nested_samples(self, beta, K, tau2) -> None:
        """Log prob should have shape (2, 2)."""
        mvn = MultivariateNormalDegenerate(loc=0.0, prec=K / tau2)
        log_prob = mvn.log_prob([[beta, beta], [beta, beta]])
        assert log_prob.shape == (2, 2)

    def test_shape_unequal_sample_shapes(self, beta, K, tau2) -> None:
        """Input samples must be of the same size."""
        mvn = MultivariateNormalDegenerate(loc=0.0, prec=K / tau2)
        with pytest.raises(ValueError):
            mvn.log_prob([[beta, beta], [beta]])

        with pytest.raises(ValueError):
            mvn.log_prob([[beta, beta], beta])

    def test_from_penalty_batch(self, beta, K, tau2) -> None:
        """The from_penalty constructor should be able to deal with batches."""
        rank, log_det = determinant_structure_degen(K)
        rank2, log_det2 = determinant_structure_degen(jnp.eye(5))
        mvn = MultivariateNormalDegenerate.from_penalty(
            loc=0.0,
            var=jnp.array([tau2, tau2]),
            pen=jnp.array([K, jnp.eye(5)]),
            rank=jnp.array([rank, rank2]),
            log_pdet=jnp.array([log_det, log_det2]),
        )

        assert mvn.log_prob(beta).shape == (2,)

    def test_batch_precision(self, beta, K) -> None:
        """
        Tests the distribution with two precision matrices while the location remains
        constant.

        Event shape: 5
        Batch shape: 2

        The output of the ``log_prob`` method has shape (2,).
        """
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec = jnp.array([prec1, prec2])
        mvn = MultivariateNormalDegenerate(0.0, prec)

        # assert correct batch shape
        assert mvn.batch_shape_tensor() == jnp.array([2])

        lp = mvn.log_prob(beta)

        # assert correct output shape
        assert lp.shape == (2,)

        # assert correct output values
        mvn1 = MultivariateNormalDegenerate(0.0, prec1)
        lp1 = mvn1.log_prob(beta)
        assert lp[0] == lp1

        mvn2 = MultivariateNormalDegenerate(0.0, prec2)
        lp2 = mvn2.log_prob(beta)
        assert lp[1] == lp2

    def test_batch_precision_2_beta_samples(self, beta, K) -> None:
        """
        Tests the distribution with two precision matrices while the location remains
        constant.

        Supplying two beta samples means that the first sample's log probability is
        evaluated for the first set of distribution parameters and the second sample's
        log probability is evaluated for the second set.

        - Event shape: 5
        - Batch shape: 2

        The output of the ``log_prob`` method has shape (2,).
        """
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec = jnp.array([prec1, prec2])
        mvn = MultivariateNormalDegenerate(0.0, prec)

        lp = mvn.log_prob([beta, beta])
        assert lp.shape == (2,)
        assert jnp.allclose(lp, mvn.log_prob(beta))

    def test_batch_loc(self, beta, K) -> None:
        """
        Tests the distribution with two location values while the precision matrix
        remains constant.

        Event shape: 5
        Batch shape: 2

        The output of the ``log_prob`` method has shape (2,).
        """
        mvn = MultivariateNormalDegenerate(jnp.array([[0.0], [5.0]]), prec=K)

        # assert correct batch shape
        assert mvn.batch_shape_tensor() == jnp.array([2])

        lp = mvn.log_prob(beta)

        # assert correct output shape
        assert lp.shape == (2,)

        # assert correct output values
        mvn1 = MultivariateNormalDegenerate(0.0, K)
        lp1 = mvn1.log_prob(beta)
        assert jnp.allclose(lp[0], lp1)

        mvn2 = MultivariateNormalDegenerate(5.0, K)
        lp2 = mvn2.log_prob(beta)
        assert jnp.allclose(lp[1], lp2)

    def test_batch_loc_prec(self, beta, K) -> None:
        """
        Tests the distribution with two location values and two precision matrices.

        This leads to two distinct distributions.

        Event shape: 5
        Batch shape: 2

        The output of the ``log_prob`` method has shape (2,).
        """
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec = jnp.array([prec1, prec2])
        loc = jnp.array([[0.0], [5.0]])

        mvn = MultivariateNormalDegenerate(loc, prec)

        # assert correct batch shape
        assert mvn.batch_shape_tensor() == jnp.array([2])

        lp = mvn.log_prob(beta)

        # assert correct output shape
        assert lp.shape == (2,)

        # assert correct output values
        mvn1 = MultivariateNormalDegenerate(0.0, prec1)
        lp1 = mvn1.log_prob(beta)
        assert jnp.allclose(lp[0], lp1)

        mvn2 = MultivariateNormalDegenerate(5.0, prec2)
        lp2 = mvn2.log_prob(beta)
        assert jnp.allclose(lp[1], lp2)

    def test_batch_2loc_3prec(self, beta, mvn_batch) -> None:
        """
        Tests the distribution with three precision matrices and two locations.

        Each location value is combined with each precision matrix, such that we have
        six distributions in total.

        Event shape: 5
        Batch shape: (2, 3)

        The output of the ``log_prob`` method has shape (2, 3).
        """
        mvn = mvn_batch

        K = jnp.eye(5)
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec3 = K / 3.0

        # assert correct batch shape
        assert jnp.all(mvn.batch_shape_tensor() == jnp.array([2, 3]))

        lp = mvn.log_prob(beta)

        # assert correct output shape
        assert lp.shape == (2, 3)

        # assert correct output values with loc[0]
        lp1 = MultivariateNormalDegenerate(0.0, prec1).log_prob(beta)
        lp2 = MultivariateNormalDegenerate(0.0, prec2).log_prob(beta)
        lp3 = MultivariateNormalDegenerate(0.0, prec3).log_prob(beta)

        assert lp[0, 0] == pytest.approx(lp1)
        assert lp[0, 1] == pytest.approx(lp2)
        assert lp[0, 2] == pytest.approx(lp3)

        # assert correct output values with loc[1]
        lp4 = MultivariateNormalDegenerate(5.0, prec1).log_prob(beta)
        lp5 = MultivariateNormalDegenerate(5.0, prec2).log_prob(beta)
        lp6 = MultivariateNormalDegenerate(5.0, prec3).log_prob(beta)

        assert lp[1, 0] == pytest.approx(lp4)
        assert lp[1, 1] == pytest.approx(lp5)
        assert lp[1, 2] == pytest.approx(lp6)

    def test_batch_2loc_3prec_multiple_samples(self, beta, mvn_batch) -> None:
        """
        Sample shape has to be (1,) or be broadcastable with batch shape.

        In this case, the distribution does not know how four samples should be
        broadcasted.

        Once the samples are brought to a fitting shape, the distribution can deal with
        them and returns six log probabilities for each sample, as expected.

        Six log probabilities are expected, because the distribution's batch shape
        is (2, 3).
        """

        mvn = mvn_batch

        samples = jnp.array([beta] * 4)

        with pytest.raises(ValueError):
            mvn.log_prob(samples)

        samples_reshaped = jnp.expand_dims(samples, axis=(1, 2))  # shape: (4, 1, 1, 5)
        lp = mvn.log_prob(samples_reshaped)
        assert lp.shape == (4, 2, 3)

    def test_batch_2loc_3prec_multiple_samples_loc(self, beta, mvn_batch) -> None:
        """
        Tests a distribution with batch size (2, 3) and multiple samples.

        Here, the sample shape corresponds to the shape of the distribution's location.

        In this case the first sample will be evaluated for the first location and all
        three precision matrices.

        The second sample will be evaluated for the second location and all three
        precision matrices.
        """

        mvn = mvn_batch

        b1 = beta
        b2 = beta.at[2].set(50)

        # sample shape corresponds to loc.shape: (2, 1)
        sample1 = [[b1], [b2]]
        lp1 = mvn.log_prob(sample1)
        assert lp1.shape == (2, 3)

        # assert correct values
        assert jnp.allclose(lp1[0], mvn.log_prob(b1)[0])
        assert jnp.allclose(lp1[1], mvn.log_prob(b2)[1])

        # two samples, both correspond to loc.shape: (2, 1)
        lp2 = mvn.log_prob([sample1, sample1])  # shape: (2, 2, 1)
        assert lp2.shape == (2, 2, 3)

        # assert correct values
        assert jnp.allclose(lp2[0], lp1)
        assert jnp.allclose(lp2[1], lp1)

    def test_batch_2loc_3prec_multiple_samples_prec(self, beta, mvn_batch) -> None:
        """
        Tests a distribution with batch size (2, 3) and multiple samples.

        Here, the sample shape corresponds to the shape of the distribution's precision.

        In this case the first sample will be evaluated for the first precision matrix
        and both locations.

        The second sample will be evaluated for the second precision matrix and both
        locations.

        The third sample will be evaluated for the third precision matrix and both
        locations.
        """
        mvn = mvn_batch

        b1 = beta
        b2 = beta.at[2].set(50)
        b3 = beta.at[3].set(20)

        # sample shape corresponds to prec.shape: (1, 3)
        sample3 = [[b1, b2, b3]]
        lp3 = mvn.log_prob(sample3)
        assert lp3.shape == (2, 3)

        # assert correct values
        assert lp3[:, 0] == pytest.approx(mvn.log_prob(b1)[:, 0])
        assert lp3[:, 1] == pytest.approx(mvn.log_prob(b2)[:, 1])
        assert lp3[:, 2] == pytest.approx(mvn.log_prob(b3)[:, 2])

        # two samples, both correspond to prec.shape: (1, 3)
        lp4 = mvn.log_prob([sample3, sample3])  # shape: (2, 1, 3)
        assert lp4.shape == (2, 2, 3)

        # assert correct values
        assert jnp.allclose(lp4[0], lp3)
        assert jnp.allclose(lp4[1], lp3)

    def test_batch_2by2_loc_3_prec(self, beta, K) -> None:
        """
        Tests the distribution with 2x2 locations and three precision matrices.

        Event shape: 5
        Batch shape: (2, 2, 3)

        The output of the ``log_prob`` method has shape (2, 2, 3).
        """
        prec1 = K / 1.0
        prec2 = K / 2.0
        prec3 = K / 3.0

        prec = jnp.array([[prec1, prec2, prec3]])  # shape (1, 3, 5, 5)
        # shape (2, 2, 1)
        loc = jnp.expand_dims(jnp.array([[[0.0], [5.0]], [[0.0], [5.0]]]), axis=(-1))

        mvn = MultivariateNormalDegenerate(loc, prec=prec)

        # assert correct batch shape
        assert jnp.all(mvn.batch_shape_tensor() == jnp.array([2, 2, 3]))

        lp = mvn.log_prob(beta)

        # assert correct output shape
        assert lp.shape == (2, 2, 3)

    def test_batch_of_smoothing_parameters(self, beta, K) -> None:
        """Tests that batches of smoothing parameters can be handled."""
        var = jnp.array([1.0, 2.0, 3.0])
        mvnd1 = MultivariateNormalDegenerate.from_penalty(0.0, var=var, pen=K)

        prec = K / jnp.expand_dims(var, axis=(-2, -1))
        mvnd2 = MultivariateNormalDegenerate(0.0, prec=prec)
        assert jnp.allclose(mvnd1.log_prob(beta), mvnd2.log_prob(beta))


class TestMVNDegenerateAgainstTensorFlow:
    """
    Compares shapes and log probability values of :class:`.MultivariateNormalDegenerate`
    with the ``MultivariateNormalFullCovariance`` distribution class from TensorFlow
    probability.
    """

    def test_scalar_location(self, beta) -> None:
        """When location is a scalar."""
        vcov = jnp.eye(5)
        mvn = tfd.MultivariateNormalFullCovariance(loc=0.0, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=0.0, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(beta).shape == mvnd.log_prob(beta).shape
        assert jnp.allclose(mvn.log_prob(beta), mvnd.log_prob(beta))

    def test_1darray_location(self, beta) -> None:
        """When location is a 1d array."""
        vcov = jnp.eye(5)
        mvn = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(5), covariance_matrix=vcov
        )
        mvnd = MultivariateNormalDegenerate(loc=jnp.zeros(5), prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(beta).shape == mvnd.log_prob(beta).shape
        assert jnp.allclose(mvn.log_prob(beta), mvnd.log_prob(beta))

    def test_2darray_location_1batch(self, beta) -> None:
        """
        When location is a 2d array, defining one batch.

        This is one distribution with event shape 5.
        """
        vcov = jnp.eye(5)
        loc = jnp.zeros((1, 5))
        sample = beta

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_2darray_location_2batches(self, beta) -> None:
        """
        When location is a 2d array, defining two batches.

        This is two distributions, each with event shape 5.
        """
        vcov = jnp.eye(5)
        loc = jnp.arange(0, 10, dtype=jnp.float32).reshape((2, 5))
        sample = beta

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_x(self) -> None:
        """
        When location is a 1d array, it must have the dimension of the event size.
        """
        vcov = jnp.eye(5)
        loc = jnp.zeros(3)

        with pytest.raises(ValueError):
            tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)

        with pytest.raises(ValueError):
            MultivariateNormalDegenerate(loc=loc, prec=vcov)

    def test_3darray_location(self, beta) -> None:
        """
        When location is a 3d array.

        Three distributions with event shape 5.
        """
        vcov = jnp.eye(5)
        loc = jnp.arange(0, 15, dtype=jnp.float32).reshape((3, 5))
        sample = beta

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_3darray_location_multiple_samples(self, beta) -> None:
        """
        When location is a 3d array.

        Three distributions with event shape 5.

        Since the samples have shape (2, 1, 5), both samples get evaluated for all
        three distributions.
        """
        vcov = jnp.eye(5)
        loc = jnp.arange(0, 15, dtype=jnp.float32).reshape((3, 5))
        sample = jnp.array([[beta], [beta]])

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_multiple_precisions(self, beta) -> None:
        """
        When two precision matrices are used.

        Since location is a 3d array, we get six distributions in this case.
        """
        vcov = jnp.eye(5)
        vcov = jnp.array([vcov, vcov])
        loc = jnp.zeros((3, 1, 5))
        sample = beta

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_multiple_precisions_nested(self, beta) -> None:
        """
        When two x two precision matrices are used.

        Since location is a 1d array here, we get four distributions.
        """
        vcov = jnp.eye(5)
        vcov = jnp.array([[vcov, vcov], [vcov, vcov]])
        loc = jnp.zeros(5)
        sample = beta

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

    def test_incompatible_shapes(self) -> None:
        """
        The :class:`.MultivariateNormalDegenerate` should behave similar to the
        TensorFlow probability distribution when initialized with parameter arrays that
        cannot be broadcasted together.

        Since the TFD class throws an error on init,
        :class:`.MultivariateNormalDegenerate` should do the same.
        """
        vcov = jnp.eye(5)
        vcov = jnp.array([vcov, vcov])
        loc = jnp.zeros((3, 5))

        with pytest.raises(ValueError):
            tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)

        with pytest.raises(ValueError):
            MultivariateNormalDegenerate(loc=loc, prec=vcov)

    def test_multiple_samples(self, beta) -> None:
        """
        Tests that :class:`.MultivariateNormalDegenerate` handles multiple samples in
        the same way as the TFD baseline class ``MultivariateNormalFullCovariance``.

        Also makes sure that the resulting arrays are structured in exactly the same
        way.
        """

        # I construct different covariace matrices and corresponding precision matrices
        # to ensure that the results are actually the same for both distribution
        # classes
        Imat = jnp.eye(5)
        tau2 = jnp.expand_dims(jnp.array([[1.0, 2.0], [3.0, 4.0]]), axis=(-2, -1))
        vcov = jnp.array([[Imat, Imat], [Imat, Imat]]) * tau2
        prec = jnp.array([[Imat, Imat], [Imat, Imat]]) / tau2
        loc = jnp.zeros(5)

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=prec)

        assert jnp.all(mvn.batch_shape_tensor() == mvnd.batch_shape_tensor())
        assert jnp.all(mvn.event_shape_tensor() == mvnd.event_shape_tensor())

        # Error is raised, if sample shape does not fit to batch shape
        # Sample shape: (3,5)
        # Batch shape: (2, 2)
        sample = jnp.array([beta, beta, beta])
        with pytest.raises(ValueError):
            mvn.log_prob(sample)

        with pytest.raises(ValueError):
            mvnd.log_prob(sample)

        # The tfd version apparently does not always raise the same error if shapes
        # are not compatible for broadcasting.
        # Here, it raises a TypeError for TFP v0.19 and smaller and a ValueError from
        # TFP v0.20.1.
        # Sample shape: (1, 3, 5)
        # Batch shape: (2, 2)
        sample = jnp.array([[beta, beta, beta]])
        with pytest.raises((TypeError, ValueError)):
            mvn.log_prob(sample)

        with pytest.raises(ValueError):
            mvnd.log_prob(sample)

        # Sample shape: (3, 1, 1, 5)
        # Batch shape: (2, 2)
        # log prob shape: (3, 2, 2)
        # Each sample gets evaluated for all four distributions
        sample = jnp.array([[[beta]], [[beta]], [[beta]]])
        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

        # Sample shape: (1, 2, 5)
        # Batch shape: (2, 2)
        # log prob shape: (2, 2)
        # sample[:,0] gets evaluated using vcov[:,0]
        # sample[:,1] gets evaluated using vcov[:,1]
        sample = jnp.array([[beta, beta + 2.0]])
        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

        # Sample shape: (2,5)
        # Batch shape: (2, 2)
        # log prob shape: (2, 2)
        # sample[0] gets evaluated using vcov[:,0]
        # sample[1] gets evaluated using vcov[:,1]
        sample = jnp.array([beta, beta + 2.0])
        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

        # Sample shape: (2, 2, 5)
        # Batch shape: (2, 2)
        # log prob shape: (2, 2)
        # One sample for each distribution
        sample = jnp.array([[beta, beta + 1.0], [beta + 2.0, beta + 3.0]])
        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))

        # Sample shape: (2, 2, 1, 1, 5)
        # Batch shape: (2, 2)
        # log prob shape: (2, 2, 2, 2)
        # Each of the four samples gets evaluated for each of the four distributions
        sample = jnp.array([[beta, beta + 1.0], [beta + 2.0, beta + 3.0]])
        sample = jnp.expand_dims(sample, axis=(2, 3))
        assert mvn.log_prob(sample).shape == mvnd.log_prob(sample).shape
        assert jnp.allclose(mvn.log_prob(sample), mvnd.log_prob(sample))


class TestSampleFromMVNDegenerate:
    def test_seed_required(self, K, tau2) -> None:
        """Validates that we MUST supply a seed."""
        mvnd = MultivariateNormalDegenerate.from_penalty(loc=0.0, var=tau2, pen=K)

        with pytest.raises(ValueError):
            mvnd.sample()

    def test_one_sample(self, K, tau2) -> None:
        """Tests that we can draw one sample of correct shape."""
        mvn = tfd.MultivariateNormalFullCovariance(covariance_matrix=jnp.eye(5))
        mvnd = MultivariateNormalDegenerate.from_penalty(loc=0.0, var=tau2, pen=K)

        key = jax.random.PRNGKey(42)
        s1 = mvn.sample(seed=key)
        s2 = mvnd.sample(seed=key)

        assert s2.shape == (5,)
        assert s1.shape == s2.shape

    def test_two_samples(self, K, tau2) -> None:
        """Validates that we can draw two samples of correct shape."""
        mvn = tfd.MultivariateNormalFullCovariance(covariance_matrix=jnp.eye(5))
        mvnd = MultivariateNormalDegenerate.from_penalty(loc=0.0, var=tau2, pen=K)

        key = jax.random.PRNGKey(42)
        s1 = mvn.sample(2, seed=key)
        s2 = mvnd.sample(2, seed=key)

        assert s2.shape == (2, 5)
        assert s1.shape == s2.shape

    def test_two_samples_2d_location(self) -> None:
        """
        When location is a 2d array, defining one batch.

        This is one distribution with event shape 5. So we get a return array of shape
        [2, 1, 5], which corresponds to [n, batch_shape, event_shape].

        Apart from the shape, this test also validates that the location of the samples
        is plausible.
        """
        vcov = jnp.diag(jnp.full(5, 0.01))
        prec = jnp.diag(jnp.full(5, 1 / 0.01))
        loc = jnp.array([jnp.zeros(5), jnp.full(5, 100.0)])

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=prec)

        key = jax.random.PRNGKey(42)
        s1 = mvn.sample(2, seed=key).round()
        s2 = mvnd.sample(2, seed=key).round()

        assert s1.shape == s2.shape
        assert np.allclose(s1[0, 0, :], 0.0)
        assert np.allclose(s1[0, 1, :], 100.0)
        assert np.allclose(s2[0, 0, :], 0.0)
        assert np.allclose(s2[0, 1, :], 100.0)

    def test_two_samples_3d_location(self) -> None:
        """
        When location is a 3d array, defining one batch.

        This is one distribution with event shape 5. So we get a return array of shape
        [2, 3, 1, 5], which corresponds to [n, batch_shape, event_shape].
        """
        vcov = jnp.eye(5)
        loc = jnp.zeros((3, 1, 5))

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=vcov)

        key = jax.random.PRNGKey(42)
        s1 = mvn.sample(2, seed=key)
        s2 = mvnd.sample(2, seed=key)

        assert s1.shape == s2.shape

    def test_two_samples_2d_prec(self) -> None:
        """
        When we have two precision matrices, i.e. two batches.

        This is one distribution with event shape 5. So we get a return array of shape
        [3, 2, 5], which corresponds to [n, batch_shape, event_shape].

        Apart from the shape, this test also validates that the variance of the samples
        is plausible.
        """
        vcov_low = jnp.diag(jnp.full(5, 0.01))
        vcov_high = jnp.diag(jnp.full(5, 100.0))
        vcov = jnp.array([vcov_low, vcov_high])

        prec_high = jnp.diag(jnp.full(5, 100.0))
        prec_low = jnp.diag(jnp.full(5, 0.01))
        prec = jnp.array([prec_high, prec_low])

        loc = 0.0

        mvn = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=vcov)
        mvnd = MultivariateNormalDegenerate(loc=loc, prec=prec)

        key = jax.random.PRNGKey(42)
        s1 = mvn.sample(10, seed=key)
        s2 = mvnd.sample(10, seed=key)

        assert s1.shape == s2.shape
        assert jnp.var(s1[:, 0, :]) < 0.5
        assert jnp.var(s1[:, 1, :]) > 80
        assert jnp.var(s2[:, 0, :]) < 0.5
        assert jnp.var(s2[:, 1, :]) > 80


def test_jit1() -> None:
    D = jnp.diff(jnp.eye(5))

    @jax.jit
    def fn(x):
        mvnd = MultivariateNormalDegenerate.from_penalty(loc=0.0, var=1.0, pen=D.T @ D)
        return mvnd.log_prob(x)

    fn(jnp.zeros(4))


def test_jit2() -> None:
    D = jnp.diff(jnp.eye(5))

    @jax.jit
    def fn(x, loc, var, pen):
        mvnd = MultivariateNormalDegenerate.from_penalty(loc=loc, var=var, pen=pen)
        return mvnd.log_prob(x)

    fn(jnp.zeros(4), loc=0.0, var=1.0, pen=D.T @ D)


def _create_equidistant_knots(x: Array, order: int, internal_k: int) -> Array:
    min_x = jnp.min(x)
    max_x = jnp.max(x)

    internal_knots = jnp.linspace(min_x, max_x, internal_k)

    step = internal_knots[1] - internal_knots[0]

    left_knots = jnp.linspace(min_x - (step * (order - 1)), min_x - step, order - 1)
    right_knots = jnp.linspace(max_x + step, max_x + (step * (order - 1)), order - 1)

    return jnp.concatenate((left_knots, internal_knots, right_knots))


def test_sampling() -> None:
    key = jrd.PRNGKey(42)
    x = jrd.uniform(key, shape=(100,))
    y = jnp.exp(x) + jrd.normal(key, shape=(100,))

    D = jnp.diff(jnp.eye(6))
    K = D.T @ D

    mvnd = lsl.Dist(
        MultivariateNormalDegenerate.from_penalty,
        loc=lsl.Var(0.0),
        var=lsl.Var(1.0),
        pen=lsl.Var(K),
    )

    beta = lsl.Var.new_param(jnp.zeros(5), mvnd, name="beta")

    knots = _create_equidistant_knots(x, order=4, internal_k=4)
    basis_mat = BSpline.design_matrix(x, knots, 3).toarray()
    X = lsl.Var.new_obs(basis_mat[:, 1:])

    smooth = lsl.Smooth(X, beta)
    Y = lsl.Var.new_obs(
        y, distribution=lsl.Dist(tfd.Normal, loc=smooth, scale=lsl.Var(1.0))
    )
    model = lsl.Model([Y])

    builder = gs.EngineBuilder(1337, 1)
    builder.set_model(gs.LieselInterface(model))
    builder.set_initial_values(model.state)
    builder.add_kernel(gs.IWLSKernel(["beta"]))
    builder.set_duration(warmup_duration=500, posterior_duration=20)
    engine = builder.build()
    engine.sample_all_epochs()
