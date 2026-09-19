"""Alias sampling with integer randomness, including tiny binary64 thresholds.

Host-side preparation uses NumPy float64. Device-side sampling uses only 32-bit
integers, so its coin resolution does not depend on JAX's float64 setting.
"""

import jax
import jax.numpy as jnp
import numpy as np

type Coin = tuple[jax.Array, jax.Array, jax.Array]
type AliasTable = tuple[jax.Array, Coin]


def build_table(probabilities: np.ndarray) -> AliasTable:
    """Build Vose's table from a normalized positive float64 probability vector.

    Preparation is O(N) and runs only on the host. See
    https://www.keithschwarz.com/darts-dice-coins/ for the algorithm.
    """
    n = len(probabilities)
    if not 1 <= n <= np.iinfo(np.int32).max:
        raise ValueError("Alias sampling requires between 1 and 2**31-1 indices.")
    scaled = np.asarray(probabilities, dtype=np.float64) * n
    thresholds = np.ones(n, dtype=np.float64)
    aliases = np.arange(n, dtype=np.int32)
    small = np.flatnonzero(scaled < 1).tolist()
    large = np.flatnonzero(scaled >= 1).tolist()
    while small and large:
        low, high = small.pop(), large.pop()
        thresholds[low] = scaled[low]
        aliases[low] = high
        # Subtract first to preserve a tiny residual when scaled[high] is 1.
        scaled[high] = (scaled[high] - 1) + scaled[low]
        (small if scaled[high] < 1 else large).append(high)
    # Rounding can leave either list nonempty. Remaining columns keep q=1
    # and their own alias; this is Vose's floating-point completion step.
    return jnp.asarray(aliases), encode_thresholds(thresholds)


def encode_thresholds(thresholds: np.ndarray) -> Coin:
    """Encode q = significand / 2**53 * 2**(-zeros), including q=0 and q=1."""
    mantissa, exponent = np.frexp(thresholds)
    significand = (mantissa * 2.0**53).astype(np.uint64)
    return (
        jnp.asarray((significand >> 32).astype(np.uint32)),
        jnp.asarray((significand & 0xFFFFFFFF).astype(np.uint32)),
        jnp.asarray(-exponent.astype(np.int32)),
    )


def compare_significand(words: jax.Array, high: jax.Array, low: jax.Array) -> jax.Array:
    """Compare 53 independent random bits with a split 53-bit significand."""
    random_high = words[0] & jnp.uint32(2**21 - 1)
    return (random_high < high) | ((random_high == high) & (words[1] < low))


def consume_zero_bits(
    word: jax.Array, accepted: jax.Array, remaining: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Require up to 32 leading zero bits, preserving previous rejections."""
    count = jnp.clip(remaining, 1, 32)
    zero_prefix = (word >> (32 - count).astype(jnp.uint32)) == 0
    return accepted & ((remaining <= 0) | zero_prefix), jnp.maximum(
        remaining - count, 0
    )


def bernoulli(key: jax.Array, coin: Coin) -> jax.Array:
    """Sample exactly at stored thresholds, assuming uniform independent bits.

    For q < 1, first accept with probability significand / 2**53, then
    require ``zeros`` independent zero bits. A binary64 threshold needs at most
    1073 prefix bits, usually rejected after the first word. q=1 has zeros=-1.
    """
    high, low, zeros = coin
    key, draw_key = jax.random.split(key)
    words = jax.random.bits(draw_key, (2, *high.shape), dtype=jnp.uint32)
    accepted = compare_significand(words, high, low)

    def cond(state):
        _, passed, remaining = state
        return jnp.any(passed & (remaining > 0))

    def body(state):
        rng, passed, remaining = state
        rng, word_key = jax.random.split(rng)
        word = jax.random.bits(word_key, high.shape, dtype=jnp.uint32)
        passed, remaining = consume_zero_bits(word, passed, remaining)
        return rng, passed, remaining

    _, accepted, _ = jax.lax.while_loop(cond, body, (key, accepted, zeros))
    return (zeros < 0) | accepted


def reduce_words(words: jax.Array, upper: int) -> tuple[jax.Array, jax.Array]:
    """Reject the incomplete modulo cycle so every index has equal mass."""
    rejected_prefix = jnp.uint32(2**32 % upper)
    return (words % jnp.uint32(upper)).astype(jnp.int32), words >= rejected_prefix


def uniform_indices(key: jax.Array, size: int, upper: int) -> jax.Array:
    """Uniform integer draws without modulo bias, with expected O(1) retries."""
    key, draw_key = jax.random.split(key)
    words = jax.random.bits(draw_key, (size,), dtype=jnp.uint32)
    indices, accepted = reduce_words(words, upper)

    def cond(state):
        return jnp.any(~state[2])

    def body(state):
        rng, previous, accepted = state
        rng, draw_key = jax.random.split(rng)
        words = jax.random.bits(draw_key, (size,), dtype=jnp.uint32)
        fresh, valid = reduce_words(words, upper)
        return rng, jnp.where(accepted, previous, fresh), accepted | valid

    _, indices, _ = jax.lax.while_loop(cond, body, (key, indices, accepted))
    return indices


def sample(key: jax.Array, size: int, table: AliasTable) -> jax.Array:
    """Draw independent indices with replacement using the prepared table."""
    aliases, coin = table
    index_key, coin_key = jax.random.split(key)
    columns = uniform_indices(index_key, size, aliases.size)
    high, low, zeros = coin
    selected_coin = high[columns], low[columns], zeros[columns]
    return jnp.where(bernoulli(coin_key, selected_coin), columns, aliases[columns])
