"""Checkpoint recovery through the public optimization API."""

import os
import pickle
import subprocess
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel.model as lsl
from liesel.optim import (
    Batches,
    EmaTrainLossMonitor,
    LieselOptim,
    OptimCheckpoint,
    OptimEngine,
    Optimizer,
    PositionSplit,
    Stopper,
)
from liesel.optim.loss import LossMixin
from liesel.optim.types import Position


class StochasticLoss(LossMixin):
    def __init__(self):
        self.split = PositionSplit(
            Position({"y": jnp.arange(8.0)}), Position({}), Position({}), 8, 0, 0
        )

    def position(self, position_keys):
        return Position({key: jnp.array(0.0) for key in position_keys})

    def loss_train_batched(self, params, carry):
        target = jnp.mean(carry.batch["y"]) + jax.random.normal(carry.key)
        return (params["theta"] - target) ** 2

    def loss_monitor(self, params, carry):
        raise NotImplementedError("This test loss uses EMA monitoring.")


def make_engine(epochs=6, **kwargs):
    return OptimEngine(
        loss=StochasticLoss(),
        batches=Batches(["y"], axis_size=8, batch_size=2),
        optimizers=[Optimizer(["theta"], optax.adam(0.05))],
        stopper=Stopper(epochs=epochs, patience=2, min_epochs=epochs),
        seed=42,
        initial_state={},
        loss_monitor=EmaTrainLossMonitor(1.0),
        show_progress=False,
        **kwargs,
    )


def assert_same_run(actual, expected):
    assert actual.n_epochs == expected.n_epochs
    for a, b in zip(
        jax.tree.leaves(
            (actual.history, actual.position_final, actual.position_min_monitor)
        ),
        jax.tree.leaves(
            (expected.history, expected.position_final, expected.position_min_monitor)
        ),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    assert actual.min_monitor_epoch == expected.min_monitor_epoch


@pytest.mark.parametrize("save_position_history", [False, True])
def test_pause_and_resume_matches_uninterrupted_stochastic_optimization(
    save_position_history,
):
    expected = make_engine(save_position_history=save_position_history).fit()
    engine = make_engine(save_position_history=save_position_history)
    first = engine.fit(pause_after=2)
    assert first.n_epochs == 2
    assert first.status == "paused"

    second = engine.fit(checkpoint=first.checkpoint, pause_after=1)
    assert second.n_epochs == 3
    actual = engine.fit(checkpoint=second.checkpoint)

    assert_same_run(actual, expected)
    assert actual.status == "max_epochs"


def test_stopper_changes_are_validated_before_resuming():
    engine = make_engine()
    result = engine.fit(pause_after=2)
    engine.stopper.epochs = 0

    with pytest.raises(ValueError, match="epochs"):
        engine.fit(checkpoint=result.checkpoint)


@pytest.mark.parametrize("progress", ["off", "epochs", "batches"])
@pytest.mark.parametrize("prune", [True, False])
def test_snapshots_survive_continuation_and_budget_extension(progress, prune):
    engine = make_engine(epochs=3, prune_history=prune)
    engine.show_progress = progress != "off"
    engine.progress_update_every = 2
    engine.show_step_progress = progress == "batches"
    engine.step_progress_update_every = 2

    first = engine.fit()
    saved = np.array(first.history.position["theta"])
    assert first.history is not first.checkpoint.history
    assert first.history.position is not first.checkpoint.history.position
    assert first.history.position["theta"] is first.checkpoint.history.position["theta"]
    assert_same_run(engine.fit(checkpoint=first.checkpoint), first)

    engine.stopper.epochs = 6
    resumed = engine.fit(checkpoint=first.checkpoint)
    reference = make_engine(prune_history=prune)
    reference.stopper.min_epochs = 3
    assert_same_run(resumed, reference.fit())
    np.testing.assert_array_equal(first.history.position["theta"], saved)

    # Returning to the same checkpoint repeats the same continuation.
    assert_same_run(engine.fit(checkpoint=first.checkpoint), resumed)


def test_resuming_rejects_incompatible_optimizer_state_before_execution():
    checkpoint = make_engine().fit(pause_after=2).checkpoint
    engine = make_engine()
    engine.optimizers = [Optimizer(["theta"], optax.sgd(0.05))]
    with pytest.raises(ValueError, match="optimizer state"):
        engine.fit(checkpoint=checkpoint)


def test_version_override_warns_and_only_bypasses_version_checks():
    engine = make_engine()
    first = engine.fit(pause_after=2)
    assert set(first.checkpoint.versions) == {
        "liesel",
        "jax",
        "jaxlib",
        "optax",
        "numpy",
    }
    checkpoint = replace(
        first.checkpoint, versions={**first.checkpoint.versions, "jax": "different"}
    )
    with pytest.raises(ValueError, match="version.*jax"):
        engine.fit(checkpoint=checkpoint)
    with pytest.warns(UserWarning, match="version.*jax"):
        actual = engine.fit(checkpoint=checkpoint, allow_version_mismatch=True)
    assert_same_run(actual, make_engine().fit())

    engine.optimizers = [Optimizer(["theta"], optax.sgd(0.05))]
    with (
        pytest.warns(UserWarning, match="version.*jax"),
        pytest.raises(ValueError, match="optimizer state"),
    ):
        engine.fit(checkpoint=checkpoint, allow_version_mismatch=True)


def test_persistent_run_recovers_on_a_reconstructed_engine(tmp_path):
    path = tmp_path / "optim.pkl"
    first = make_engine().fit(checkpoint=path, pause_after=2)
    loaded = OptimCheckpoint.load(path)
    assert loaded.n_epochs == first.n_epochs == 2
    assert_same_run(make_engine().fit(checkpoint=loaded), make_engine().fit())
    # Using an object selects memory only; the file remains at epoch 2.
    assert OptimCheckpoint.load(path).n_epochs == 2

    actual = make_engine().fit(checkpoint=path)
    assert_same_run(actual, make_engine().fit())
    assert actual.duration > first.duration
    assert OptimCheckpoint.load(path).n_epochs == actual.n_epochs

    first.checkpoint.save(path)
    assert OptimCheckpoint.load(path).n_epochs == 2


@pytest.mark.parametrize("progress", ["off", "epochs", "batches"])
def test_periodic_saves_survive_interruption_independently_of_progress(
    tmp_path, monkeypatch, progress
):
    path = tmp_path / "optim.pkl"
    engine = make_engine(epochs=7)
    engine.show_progress = progress != "off"
    engine.show_step_progress = progress == "batches"
    engine.progress_update_every = 3
    engine.step_progress_update_every = 2
    save = OptimCheckpoint.save
    saved_epochs = []

    def interrupt_after_save(checkpoint, path):
        save(checkpoint, path)
        saved_epochs.append(checkpoint.n_epochs)
        if checkpoint.n_epochs == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(OptimCheckpoint, "save", interrupt_after_save)
    with pytest.raises(KeyboardInterrupt):
        engine.fit(checkpoint=path, checkpoint_every=2)
    assert saved_epochs == [2, 4]
    assert OptimCheckpoint.load(path).n_epochs == 4

    monkeypatch.setattr(OptimCheckpoint, "save", save)
    assert_same_run(
        engine.fit(checkpoint=path, checkpoint_every=2), make_engine(7).fit()
    )
    assert OptimCheckpoint.load(path).n_epochs == 7


@pytest.mark.parametrize("operation", ["serialize", "replace"])
def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch, operation):
    path = tmp_path / "optim.pkl"
    make_engine().fit(checkpoint=path, pause_after=2)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        if operation == "serialize":
            args[1].write(b"partial write")
        raise OSError("Disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            pickle if operation == "serialize" else os,
            "dump" if operation == "serialize" else "replace",
            fail,
        )
        with pytest.raises(OSError, match="Disk failure"):
            make_engine().fit(checkpoint=path, checkpoint_every=1)

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert OptimCheckpoint.load(path).n_epochs == 2
    assert_same_run(make_engine().fit(checkpoint=path), make_engine().fit())


@pytest.mark.parametrize("contents", [b"garbage", b"liesel.optim.checkpoint\x00\x02\n"])
def test_invalid_file_is_not_overwritten(tmp_path, contents):
    path = tmp_path / "optim.pkl"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="format"):
        make_engine().fit(checkpoint=path, allow_version_mismatch=True)
    assert path.read_bytes() == contents


@pytest.mark.parametrize("argument", ["pause_after", "checkpoint_every"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_execution_limits_are_rejected(argument, value):
    with pytest.raises(ValueError, match=argument):
        make_engine().fit(**{argument: value})


class NaNLoss(StochasticLoss):
    def loss_train_batched(self, params, carry):
        return jnp.where(
            carry.epoch >= 3, jnp.nan, super().loss_train_batched(params, carry)
        )


@pytest.mark.parametrize("progress", ["off", "epochs", "batches"])
@pytest.mark.parametrize("debug_nans", [False, True])
def test_nan_failure_preserves_last_successful_checkpoint(
    tmp_path, progress, debug_nans
):
    path = tmp_path / "optim.pkl"
    engine = make_engine(debug_nans=debug_nans)
    engine.loss = NaNLoss()
    engine.show_progress = progress != "off"
    engine.show_step_progress = progress == "batches"
    engine.progress_update_every = 3
    engine.step_progress_update_every = 2
    result = engine.fit(checkpoint=path, checkpoint_every=2)
    assert result.status == "nan"
    assert result.checkpoint is None
    assert (result.nan_debug is not None) == debug_nans
    assert OptimCheckpoint.load(path).n_epochs == 2


def test_early_stop_takes_precedence_over_pause_and_survives_budget_extension(tmp_path):
    path = tmp_path / "optim.pkl"
    engine = make_engine()
    engine.stopper.min_epochs = 0
    engine.stopper.atol = 1e9
    result = engine.fit(checkpoint=path, pause_after=3)
    assert result.status == "early_stopping"
    assert result.n_epochs == 3
    assert OptimCheckpoint.load(path).n_epochs == 3
    engine.stopper.epochs = 20
    resumed = engine.fit(checkpoint=path)
    assert resumed.status == "early_stopping"
    assert_same_run(resumed, result)


def make_model_engine(optimizer):
    loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    observations = [
        lsl.Var.new_obs(
            jnp.arange(float(n)),
            lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
            name=f"y{n}",
        )
        for n in (8, 5)
    ]
    model = lsl.Model(observations)
    batches = (
        Batches.from_model(model, batch_size=2, multi_size="manager")
        if optimizer == "adam"
        else None
    )
    return LieselOptim(
        model,
        loss_monitor="train_full_data",
        batches=batches,
        optimizers=optax.adam(0.02) if optimizer == "adam" else optimizer,
        stopper=Stopper(epochs=6, patience=2, min_epochs=6),
        seed=21,
        show_progress=False,
    ).build_engine()


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_model_checkpoint_recovers_in_a_fresh_python_process(tmp_path, optimizer):
    path = tmp_path / "optim.pkl"
    engine = make_model_engine(optimizer)
    engine.fit(checkpoint=path, pause_after=2)
    recovery = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from tests.optim.test_checkpoint import make_model_engine; "
                "make_model_engine(sys.argv[2]).fit(checkpoint=sys.argv[1])"
            ),
            str(path),
            optimizer,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert recovery.returncode == 0, recovery.stderr
    actual = make_model_engine(optimizer).fit(checkpoint=path)
    assert_same_run(actual, make_model_engine(optimizer).fit())


def test_model_recovery_rejects_changed_observation_shapes():
    first = make_model_engine("adam").fit(pause_after=2)
    engine = make_model_engine("adam")
    engine.split.train["y8"] = engine.split.train["y8"].reshape(8, 1)
    with pytest.raises(ValueError, match="data.*structure"):
        engine.fit(checkpoint=first.checkpoint)


def test_reconstructed_model_recovery_with_nan_debugging(tmp_path):
    def engine():
        engine = make_model_engine("adam")
        engine.debug_nans = True
        return engine

    path = tmp_path / "optim.pkl"
    engine().fit(checkpoint=path, pause_after=2)
    assert_same_run(engine().fit(checkpoint=path), engine().fit())


class StatefulLoss(StochasticLoss):
    def loss_train_batched(self, params, carry):
        return (
            super().loss_train_batched(params, carry)
            + carry.model_state["offset"] * params["theta"]
        )


class StatefulOptimizer(Optimizer):
    def step(self, position, loss, carry):
        carry, value = super().step(position, loss, carry)
        carry.model_state["offset"] += 0.1
        return carry, value


def test_custom_evolving_model_state_survives_recovery(tmp_path):
    def engine():
        engine = make_engine()
        engine.loss = StatefulLoss()
        engine.optimizers = [StatefulOptimizer(["theta"], optax.adam(0.05))]
        engine.initial_state = {"offset": jnp.array(0.0)}
        return engine

    path = tmp_path / "optim.pkl"
    first = engine().fit(checkpoint=path, pause_after=2)
    assert_same_run(engine().fit(checkpoint=first.checkpoint), engine().fit())
    assert_same_run(engine().fit(checkpoint=path), engine().fit())


def test_duration_includes_writes_and_excludes_time_between_calls(
    tmp_path, monkeypatch
):
    elapsed = 0.0
    monotonic = time.monotonic
    dump = pickle.dump

    def timed_dump(value, *args, **kwargs):
        nonlocal elapsed
        if isinstance(value, OptimCheckpoint):
            elapsed += 5.0
        return dump(value, *args, **kwargs)

    monkeypatch.setattr(time, "monotonic", lambda: monotonic() + elapsed)
    monkeypatch.setattr(pickle, "dump", timed_dump)
    path = tmp_path / "optim.pkl"
    first = make_engine().fit(checkpoint=path, pause_after=2)
    saved = OptimCheckpoint.load(path)
    assert first.duration >= 5.0
    assert saved.duration >= 5.0

    elapsed += 3600.0
    second = make_engine().fit(checkpoint=saved)
    assert saved.duration <= second.duration < saved.duration + 60.0
