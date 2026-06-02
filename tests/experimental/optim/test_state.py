import jax
import jax.numpy as jnp
import optax
import pytest

from liesel.optim import Batches, Optimizer
from liesel.optim.state import OptimCarry, OptimHistory, OptimResult
from liesel.optim.types import Position


class TestOptimHistory:
    def test_position_df(self):
        pos = {"a": jnp.full((3, 2), fill_value=1.0)}
        hist = OptimHistory.from_epochs(epochs=20, position=pos, tracked=None)
        df = hist.position_df()
        assert df.shape == (20, 7)
        assert hist.position_df(subset=[]).to_dict("list") == {
            "epoch": [float(i) for i in range(20)]
        }

    def test_position_df_handles_one_epoch_scalar_and_vector_histories(self):
        hist = OptimHistory(
            loss_train=jnp.array([1.0]),
            loss_validate=jnp.array([2.0]),
            position=Position(
                {
                    "theta": jnp.array([[1.0, 2.0]]),
                    "sigma": jnp.array([3.0]),
                }
            ),
            tracked=None,
        )

        df = hist.position_df()

        assert df.columns.tolist() == ["epoch", "theta0", "theta1", "sigma"]
        assert df.to_dict("list") == {
            "epoch": [0.0],
            "theta0": [1.0],
            "theta1": [2.0],
            "sigma": [3.0],
        }

    def test_tracked_df_flattens_multidimensional_histories(self):
        tracked = Position({"matrix": jnp.arange(4.0).reshape(2, 2)})
        hist = OptimHistory.from_epochs(epochs=1, position=None, tracked=tracked)
        assert hist.tracked is not None
        hist.tracked = OptimHistory.update_position_history(0, hist.tracked, tracked)

        df = hist.tracked_df()

        assert df.columns.tolist() == [
            "epoch",
            "matrix0",
            "matrix1",
            "matrix2",
            "matrix3",
        ]
        assert df.iloc[0].to_dict() == {
            "epoch": 0.0,
            "matrix0": 0.0,
            "matrix1": 1.0,
            "matrix2": 2.0,
            "matrix3": 3.0,
        }


class TestOptimCarry:
    def test_new_uses_position_dtype_for_losses_and_history(self):
        position = Position({"theta": jnp.array(0.0, dtype=jnp.float32)})

        with jax.enable_x64(True):
            carry = OptimCarry.new(
                key=jax.random.key(0),
                epochs=2,
                position=position,
                tracked=None,
                batches=Batches(["y"], n=4, batch_size=2),
                optimizers=[Optimizer(["theta"], optax.sgd(0.1))],
                model_state={},
                save_position_history=True,
            )

        assert carry.best_loss.dtype == jnp.float32
        assert carry.loss_train.dtype == jnp.float32
        assert carry.loss_validate.dtype == jnp.float32
        assert carry.history.loss_train.dtype == jnp.float32
        assert carry.history.loss_validate.dtype == jnp.float32
        assert carry.history.position is not None
        assert carry.history.position["theta"].dtype == jnp.float32

    def test_new_rejects_duplicate_optimizer_identifiers(self):
        position = Position({"theta": jnp.array(0.0)})
        optimizers = [
            Optimizer(["theta"], optax.sgd(0.1), identifier="same"),
            Optimizer(["theta"], optax.sgd(0.1), identifier="same"),
        ]

        with pytest.raises(ValueError, match="identifiers"):
            OptimCarry.new(
                key=jax.random.key(0),
                epochs=2,
                position=position,
                tracked=None,
                batches=Batches(["y"], n=4, batch_size=2),
                optimizers=optimizers,
                model_state={},
                save_position_history=True,
            )


class TestOptimResult:
    def test_plot_loss_labels_monitoring_loss(self):
        history = OptimHistory.from_epochs(epochs=2, position=None, tracked=None)
        history.loss_train = jnp.array([1.0, 0.5])
        history.loss_validate = jnp.array([1.2, 0.7])
        result = OptimResult(
            history=history,
            final_epoch=1,
            best_position=Position({}),
            best_epoch=1,
            duration=0.0,
        )

        plot = result.plot_loss()

        assert set(plot.data["Loss Type"].unique()) == {"Training", "Monitoring"}

    def test_plot_methods_reject_invalid_window(self):
        position = Position({"theta": jnp.array(0.0)})
        history = OptimHistory.from_epochs(epochs=2, position=position, tracked=None)
        result = OptimResult(
            history=history,
            final_epoch=1,
            best_position=position,
            best_epoch=0,
            duration=0.0,
        )

        with pytest.raises(ValueError, match="window"):
            result.plot_loss(window=0)

        with pytest.raises(ValueError, match="window"):
            result.plot_params(window=-1)
