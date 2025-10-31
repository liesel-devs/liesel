import jax.numpy as jnp

from liesel.experimental.optim.state import OptimHistory


class TestOptimHistory:
    def test_position_df(self):
        pos = {"a": jnp.full((3, 2), fill_value=1.0)}
        hist = OptimHistory.new(niter=20, position=pos, tracked=None)
        df = hist.position_df()
        assert df.shape == (20, 7)
