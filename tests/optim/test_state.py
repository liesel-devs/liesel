from typing import Literal

import jax
import jax.numpy as jnp
import optax
import pytest

from liesel.optim import Batches, Optimizer
from liesel.optim.state import OptimCarry, OptimHistory, OptimResult
from liesel.optim.types import Position


class TestOptimHistory:
    def test_position_df(self):
        pos = Position({"a": jnp.full((3, 2), fill_value=1.0)})
        hist = OptimHistory.from_epochs(epochs=20, position=pos, tracked=None)
        df = hist.position_df()
        assert df.shape == (20, 7)
        assert hist.position_df(subset=[]).to_dict("list") == {
            "epoch": [float(i) for i in range(20)]
        }

    def test_position_df_handles_one_epoch_scalar_and_vector_histories(self):
        hist = OptimHistory(
            loss_train=jnp.array([1.0]),
            loss_monitor=jnp.array([2.0]),
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
                batches=Batches(["y"], axis_size=4, batch_size=2),
                optimizers=[Optimizer(["theta"], optax.sgd(0.1))],
                model_state={},
                save_position_history=True,
            )

        assert carry.min_monitor_loss.dtype == jnp.float32
        assert carry.loss_train.dtype == jnp.float32
        assert carry.loss_monitor.dtype == jnp.float32
        assert carry.history.loss_train.dtype == jnp.float32
        assert carry.history.loss_monitor.dtype == jnp.float32
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
                batches=Batches(["y"], axis_size=4, batch_size=2),
                optimizers=optimizers,
                model_state={},
                save_position_history=True,
            )


class TestOptimResult:
    def test_n_epochs_is_completed_epoch_count(self):
        history = OptimHistory.from_epochs(epochs=2, position=None, tracked=None)
        position = Position({})
        result = OptimResult(
            history=history,
            position=position,
            position_final=position,
            position_min_monitor=position,
            n_epochs=2,
            min_monitor_epoch=1,
            monitor_source="validation",
            duration=0.0,
        )

        assert result.n_epochs == len(result.history.loss_train)
        assert result.n_epochs - 1 == 1

    @pytest.mark.parametrize(
        ("monitor_source", "monitor_label"),
        [
            ("train_ema", "Monitoring (training EMA)"),
            ("validation", "Monitoring (validation)"),
            ("train_full_data", "Monitoring (full training data)"),
        ],
    )
    def test_plot_loss_uses_source_aware_labels(
        self,
        monitor_source: Literal["train_ema", "validation", "train_full_data"],
        monitor_label: str,
    ):
        history = OptimHistory.from_epochs(epochs=2, position=None, tracked=None)
        history.loss_train = jnp.array([1.0, 0.5])
        history.loss_monitor = jnp.array([1.2, 0.7])
        position = Position({})
        result = OptimResult(
            history=history,
            position=position,
            position_final=position,
            position_min_monitor=position,
            n_epochs=2,
            min_monitor_epoch=1,
            monitor_source=monitor_source,
            duration=0.0,
        )

        plot = result.plot_loss()

        assert set(plot.data["Loss Type"].unique()) == {
            "Training (epoch mean)",
            monitor_label,
        }
        assert len(plot.layers) == 2
        assert repr(result) == (
            "OptimResult(n_epochs=2, min_monitor_epoch=1, "
            f"monitor_source={monitor_source!r}, duration=0.0s)"
        )

    def test_plot_methods_reject_invalid_window(self):
        position = Position({"theta": jnp.array(0.0)})
        history = OptimHistory.from_epochs(epochs=2, position=position, tracked=None)
        result = OptimResult(
            history=history,
            position=position,
            position_final=position,
            position_min_monitor=position,
            n_epochs=2,
            min_monitor_epoch=0,
            monitor_source="train_ema",
            duration=0.0,
        )

        with pytest.raises(ValueError, match="window"):
            result.plot_loss(window=0)

        with pytest.raises(ValueError, match="window"):
            result.plot_params(window=-1)

        assert len(result.plot_params().layers) == 2
