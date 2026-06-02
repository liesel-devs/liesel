import jax
import jax.numpy as jnp
import numpy as np
import pytest

from liesel.optim import Stopper


class TestStopper:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"epochs": 0, "patience": 1}, "epochs must be at least 1"),
            ({"epochs": 10, "patience": 0}, "patience must be at least 1"),
            (
                {"epochs": 5, "patience": 6},
                "patience must be less than or equal to epochs",
            ),
            ({"epochs": 10, "patience": 5, "atol": -1.0}, "atol"),
            ({"epochs": 10, "patience": 5, "rtol": -1.0}, "rtol"),
        ],
    )
    def test_invalid_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Stopper(**kwargs)

    def test_max_iter_alias(self):
        stopper = Stopper(epochs=10, patience=5)
        assert stopper.max_iter == stopper.epochs

    def test_stopper_does_not_stop(self):
        stopper = Stopper(patience=5, epochs=100)

        # case 1: continuous decrease
        loss_history = np.linspace(100, 0, 100)
        stop = stopper.stop_early(i=7, loss_history=loss_history)
        assert not bool(stop)

        # case 2: current loss is not the best
        loss_history = np.zeros((100,))
        loss_history[5] = -1.0
        stop = stopper.stop_early(i=7, loss_history=loss_history)
        assert not bool(stop)

    def test_stopper_stops_atol(self):
        stopper = Stopper(patience=5, epochs=100, atol=0.1)

        loss_history = np.zeros((100,))
        loss_history[2:7] = np.array([0.0, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=7, loss_history=loss_history)
        assert bool(stop)

    def test_stopper_stops_rtol(self):
        stopper = Stopper(patience=5, epochs=100, atol=0.0, rtol=0.05)

        # 0.0 gets compared to -0.05.
        # The relative difference is 1, so early stopping does not trigger.
        loss_history = np.zeros((100,))
        loss_history[2:7] = np.array([0.0, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=7, loss_history=loss_history)
        assert not bool(stop)

        # -0.049 gets compared to -0.05.
        # The relative difference is about 0.02, so early stopping triggers.
        loss_history[2:7] = np.array([-0.049, 1.0, -0.05, -0.05, -0.01])
        stop = stopper.stop_early(i=7, loss_history=loss_history)
        assert bool(stop)

    def test_stopper_jitted(self):
        stopper = Stopper(patience=5, epochs=100, atol=0.1, rtol=0.1)
        stop_jit = jax.jit(stopper.stop_early)

        loss_history = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        stop = stop_jit(i=7, loss_history=loss_history)
        assert bool(stop)

    def test_stop_at_jitted(self):
        stopper = Stopper(patience=5, epochs=100, atol=0.1, rtol=0.1)
        stop_at_jit = jax.jit(stopper.which_best_in_recent_history)

        key = jax.random.PRNGKey(42)
        loss_history = 2.0 + jax.random.uniform(
            key, shape=(15,), minval=0.0, maxval=0.1
        )

        stop = stopper.stop_early(i=7, loss_history=loss_history)
        stop_at = stop_at_jit(i=7, loss_history=loss_history)

        assert bool(stop)
        assert loss_history[stop_at] == pytest.approx(jnp.min(loss_history[2:7]))

    def test_stop_at_start(self):
        stopper = Stopper(epochs=100, patience=5, atol=0.1, rtol=0.1)
        loss_history = jnp.zeros(shape=(10,))

        stop = stopper.stop_now(i=0, loss_history=loss_history)

        assert not bool(stop)

    def test_patience(self):
        stopper = Stopper(epochs=100, patience=100)
        loss_history = jnp.zeros(shape=(100,))

        for i in range(100):
            assert not bool(stopper.stop_early(i, loss_history))

    def test_stops_after_maximum_epochs(self):
        stopper = Stopper(epochs=10, patience=5)
        loss_history = jnp.linspace(10.0, 0.0, 10)

        assert not bool(stopper.stop_now(i=9, loss_history=loss_history))
        assert bool(stopper.stop_now(i=10, loss_history=loss_history))
