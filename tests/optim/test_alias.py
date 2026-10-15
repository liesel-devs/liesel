"""Deterministic numerical checks for the internal alias sampling boundary."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from liesel.optim import _alias


def test_coin_encoding_preserves_binary64_thresholds_without_device_float64():
    thresholds = np.array(
        [0.0, 1.0, 0.25, 0.7, 2.0**-130, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)]
    )
    high, low, zeros = map(np.asarray, _alias.encode_thresholds(thresholds))
    assert high.dtype == low.dtype == np.uint32
    assert zeros.dtype == np.int32
    for q, hi, lo, exponent in zip(thresholds, high, low, zeros, strict=True):
        significand = (int(hi) << 32) | int(lo)
        encoded = Fraction(significand, 2**53) * Fraction(2) ** -int(exponent)
        assert encoded == Fraction(float(q))


def test_significand_comparison_resolves_low_word_and_strict_boundary():
    # A 53-bit uniform integer must be strictly below the threshold integer.
    high = jnp.array([1, 1, 1, 1], dtype=jnp.uint32)
    low = jnp.array([17, 17, 17, 17], dtype=jnp.uint32)
    words = jnp.array([[0, 1, 1, 2], [0xFFFFFFFF, 16, 17, 0]], dtype=jnp.uint32)
    result = jax.jit(_alias.compare_significand)(words, high, low)
    np.testing.assert_array_equal(result, [True, True, False, False])


def test_zero_prefix_comparison_handles_word_boundaries_and_multiple_words():
    remaining = jnp.array([0, 1, 1, 31, 31, 32, 32, 33, 129, 1073])
    words = jnp.array(
        [0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 1, 2, 0, 1, 0, 0, 0],
        dtype=jnp.uint32,
    )
    accepted, left = jax.jit(_alias.consume_zero_bits)(
        words, jnp.ones(10, dtype=bool), remaining
    )
    np.testing.assert_array_equal(
        accepted, [True, True, False, True, False, True, False, True, True, True]
    )
    np.testing.assert_array_equal(left, [0, 0, 0, 0, 0, 0, 0, 1, 97, 1041])
    # An all-zero prefix can reach even the smallest binary64 threshold.
    for _ in range(33):
        accepted, left = _alias.consume_zero_bits(
            jnp.zeros(10, jnp.uint32), accepted, left
        )
    assert bool(accepted[-1]) and int(left[-1]) == 0
    assert not bool(accepted[2])  # A rejection never becomes an acceptance.


def test_coin_endpoints_and_ordinary_frequencies_under_jit():
    thresholds = np.broadcast_to(np.array([0.0, 1.0, 0.25, 0.7]), (20000, 4))
    result = jax.jit(_alias.bernoulli)(
        jax.random.key(41), _alias.encode_thresholds(thresholds)
    )
    frequencies = np.asarray(result).mean(axis=0)
    np.testing.assert_array_equal(frequencies[:2], [0.0, 1.0])
    np.testing.assert_allclose(frequencies[2:], [0.25, 0.7], atol=0.015)


def test_integer_rejection_excludes_the_incomplete_modulo_cycle():
    words = jnp.array([0, 1, 2, 3, 4, 0xFFFFFFFF], dtype=jnp.uint32)
    indices, accepted = _alias.reduce_words(words, 3)
    np.testing.assert_array_equal(indices, [0, 1, 2, 0, 1, 0])
    np.testing.assert_array_equal(accepted, [False, True, True, True, True, True])
    _, accepted = _alias.reduce_words(words, 2**31 - 1)
    np.testing.assert_array_equal(accepted, [False, False, True, True, True, True])
    _, accepted = _alias.reduce_words(words, 4)
    assert bool(jnp.all(accepted))


@pytest.mark.parametrize("upper", [1, 7, 2**30 + 1])
def test_uniform_integer_sampling_is_jittable_and_in_range(upper):
    sample = jax.jit(lambda key: _alias.uniform_indices(key, 35000, upper))
    indices = np.asarray(sample(jax.random.key(17)))
    assert indices.dtype == np.int32
    assert np.all((indices >= 0) & (indices < upper))
    np.testing.assert_array_equal(indices, sample(jax.random.key(17)))
    if upper == 7:
        np.testing.assert_allclose(
            np.bincount(indices) / len(indices), 1 / 7, atol=0.01
        )


@pytest.mark.parametrize(
    "weights",
    [[1], [1, 1, 1, 1, 1, 1, 1], [1, 2, 7], [1, 1e-8, 1], [1, 2.0**-100, 2.0**-500]],
)
def test_alias_table_preserves_distribution_including_tiny_branches(weights):
    weights = np.asarray(weights, dtype=np.float64)
    aliases, coin = _alias.build_table(weights / weights.sum())
    high, low, zeros = map(np.asarray, coin)
    n = len(weights)
    distribution = [Fraction(0)] * n
    for index, (alias, hi, lo, exponent) in enumerate(
        zip(np.asarray(aliases), high, low, zeros, strict=True)
    ):
        q = Fraction((int(hi) << 32) | int(lo), 2**53) * Fraction(2) ** -int(exponent)
        assert 0 < q <= 1
        assert 0 <= alias < n
        distribution[index] += q / n
        distribution[alias] += (1 - q) / n
    assert sum(distribution) == 1
    exact_weights = [Fraction(float(w)) for w in weights]
    expected = [float(w / sum(exact_weights)) for w in exact_weights]
    np.testing.assert_allclose(
        list(map(float, distribution)), expected, rtol=1e-14, atol=0
    )


def test_alias_draws_are_independent_reproducible_and_follow_weights():
    table = _alias.build_table(np.array([0.1, 0.2, 0.7]))
    draw = jax.jit(lambda key: _alias.sample(key, 20000, table))
    indices = np.asarray(draw(jax.random.key(13)))
    np.testing.assert_array_equal(indices, draw(jax.random.key(13)))
    assert not np.array_equal(indices, draw(jax.random.key(14)))
    np.testing.assert_allclose(
        np.bincount(indices) / len(indices), [0.1, 0.2, 0.7], atol=0.015
    )
