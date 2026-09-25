"""Opinionated variational inference setup for Liesel models."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import optax

from ..model import Model
from ._engine_utils import (
    BatchConfig,
    SplitConfig,
    _validate_optimizer_batches,
    _validate_positive_int,
)
from .batch import Batches
from .engine import EmaTrainLossMonitor, LossMonitor
from .loss import _validate_bool
from .optimizer import Optimizer, OptimizerLike
from .split import PositionSplit
from .stop import Stopper
from .vi import NegElboLoss

if TYPE_CHECKING:
    from .engine import OptimEngine
    from .state import OptimResult


class LieselVI:
    """
    Builds an :class:`.OptimEngine` for variational inference.

    ``LieselVI`` is the quick-start wrapper for ELBO optimization. It constructs one
    of the standard :class:`.NegElboLoss` variational families and default training
    batches, and wraps a supplied Optax transformation over all variational
    parameters. Variational-family initialization belongs to
    :class:`.NegElboLoss` and :class:`.VDist`; pass a custom ``NegElboLoss`` when you
    need Laplace or custom initialization.

    Parameters
    ----------
    model
        Target Liesel model.
    loss
        Either one of ``"mvn_diag"``, ``"mvn_tril"``, and ``"mvn_blocked"``, or an
        explicit :class:`.NegElboLoss` instance.
    batches
        Optional explicit batch configuration. Cannot be combined with
        ``batch_size``.
    batch_size
        Mini-batch size used to construct default batches. ``None`` means full-data
        batches.
    split
        Optional split. If omitted and ``loss`` is not an explicit
        :class:`.NegElboLoss`,
        all observed data is used for training. Multi-size observed data
        automatically uses :class:`.PositionSplitManager`. Validation data is not
        supported for ELBO losses.
    optimizers
        A configured Optax transformation, such as ``optax.adam(learning_rate=0.01)``,
        applied to all variational q parameters, or a sequence of explicit
        :class:`.Optimizer` objects selecting q parameter blocks. Pass a
        transformation, not an optimizer factory. L-BFGS has no string shortcut
        because ELBO estimates are usually stochastic; an explicit :class:`.LBFGS`
        must be the sole optimizer and requires full-data batches and a
        deterministic objective.
    stopper
        Maximum-epoch and early-stopping configuration. ``None`` creates a fresh
        :class:`.Stopper` with ``epochs=1000``, ``patience=10``, and ``rtol=1e-6``.
    seed
        Integer seed, defaulting to zero. If ``None``, the current Unix time is used.
    nsamples
        Monte Carlo sample count for internally constructed training ELBOs.
    scale_loss
        Whether internally constructed ELBO losses should be divided by the training
        sample size. Must be a boolean. This setting has no effect when
        ``loss`` is an explicit :class:`.NegElboLoss`.
    regularize_q_prior
        Whether internally constructed ELBOs should include priors in the
        variational model as regularization terms.
    loss_monitor
        Source for the epoch-level stopping and progress loss. Pass
        :class:`.EmaTrainLossMonitor` for a continuous EMA of pre-update losses or
        ``"train_full_data"`` for one complete training-loss evaluation after each
        epoch. Validation monitoring is unavailable because ELBO losses do not
        support validation splits.
    entropy
        Entropy estimator for internally constructed losses: ``"auto"`` uses
        analytic entropy where supported with per-term Monte Carlo fallback;
        ``"mc"`` estimates all entropy terms by sampling. An explicit loss retains
        its own entropy setting.
    save_position_history
        Whether to save parameter positions after each epoch. Disabling this saves
        memory; final and best-monitor positions and loss histories remain available.
    show_progress
        Whether the built engine should show ``tqdm`` progress bars.
    progress_update_every
        Update the epoch progress bar after this many completed epochs. When batch
        progress is active, the epoch bar advances after every epoch.
    show_step_progress
        Whether to show an additional progress bar for batches within each epoch
        when ``show_progress`` is enabled.
    step_progress_update_every
        Update the batch progress bar after this many completed batches.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import optax
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.optim import EmaTrainLossMonitor, LieselVI
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([0.0, 1.0]),
    ...     lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> engine = LieselVI(
    ...     model,
    ...     optimizers=optax.adam(learning_rate=1e-3),
    ...     loss_monitor=EmaTrainLossMonitor(effective_window=1.0),
    ...     seed=1,
    ... ).build_engine()
    >>> type(engine).__name__
    'OptimEngine'
    >>> type(engine.loss).__name__
    'NegElboLoss'
    """

    def __init__(
        self,
        model: Model,
        *,
        loss_monitor: LossMonitor,
        optimizers: optax.GradientTransformation | Sequence[OptimizerLike],
        stopper: Stopper | None = None,
        seed: int | None = 0,
        split: SplitConfig | None = None,
        batch_size: int | None = None,
        batches: BatchConfig | None = None,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss = "mvn_diag",
        nsamples: int = 10,
        scale_loss: bool = True,
        regularize_q_prior: bool = True,
        entropy: Literal["auto", "mc"] = "auto",
        save_position_history: bool = True,
        show_progress: bool = True,
        show_step_progress: bool = False,
        progress_update_every: int = 10,
        step_progress_update_every: int = 10,
    ) -> None:
        if batches is not None and batch_size is not None:
            raise ValueError("Pass either batch_size or batches, not both.")

        self.model = model
        self.seed = int(time.time()) if seed is None else seed
        self.stopper = (
            Stopper(epochs=1000, patience=10, rtol=1e-6) if stopper is None else stopper
        )
        self.split = self._resolve_split(loss, split)
        self.loss_monitor = loss_monitor
        if loss_monitor == "validation":
            raise ValueError(
                "LieselVI does not support loss_monitor='validation' because "
                "NegElboLoss does not support validation splits."
            )
        if not isinstance(loss_monitor, EmaTrainLossMonitor) and (
            loss_monitor != "train_full_data"
        ):
            raise ValueError(
                "loss_monitor must be EmaTrainLossMonitor(effective_window=...) "
                "or 'train_full_data', "
                f"but got {loss_monitor!r}."
            )
        self.loss = self._resolve_loss(
            loss,
            nsamples=nsamples,
            scale_loss=scale_loss,
            regularize_q_prior=regularize_q_prior,
            entropy=entropy,
        )
        self.batches = (
            Batches.from_split(self.split, batch_size=batch_size)
            if batches is None
            else batches
        )
        self.optimizers = self._resolve_optimizers(optimizers)
        self.save_position_history = save_position_history
        _validate_optimizer_batches(self.optimizers, self.batches)
        self.show_progress = show_progress
        self.progress_update_every = progress_update_every
        self.show_step_progress = show_step_progress
        self.step_progress_update_every = step_progress_update_every
        _validate_positive_int(self.progress_update_every, "progress_update_every")
        _validate_positive_int(
            self.step_progress_update_every, "step_progress_update_every"
        )

    def _resolve_split(
        self,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss,
        split: SplitConfig | None,
    ) -> SplitConfig:
        if isinstance(loss, NegElboLoss):
            if split is not None and split is not loss.split:
                raise ValueError(
                    "When both loss and split are provided, split must be loss.split."
                )

            return loss.split

        if split is not None:
            return split

        return PositionSplit.from_model(
            self.model,
            shuffle=False,
            multi_size="manager",
        )

    def _resolve_loss(
        self,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss,
        nsamples: int,
        scale_loss: bool,
        regularize_q_prior: bool,
        entropy: Literal["auto", "mc"],
    ) -> NegElboLoss:
        if isinstance(loss, NegElboLoss):
            return loss

        _validate_bool(scale_loss, "scale_loss")

        match loss:
            case "mvn_diag":
                return NegElboLoss.mvn_diag(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    scale=scale_loss,
                    regularize_q_prior=regularize_q_prior,
                    entropy=entropy,
                )
            case "mvn_tril":
                return NegElboLoss.mvn_tril(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    scale=scale_loss,
                    regularize_q_prior=regularize_q_prior,
                    entropy=entropy,
                )
            case "mvn_blocked":
                return NegElboLoss.mvn_blocked(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    scale=scale_loss,
                    regularize_q_prior=regularize_q_prior,
                    entropy=entropy,
                )
            case _:
                raise ValueError(
                    "loss must be 'mvn_diag', 'mvn_tril', 'mvn_blocked', or a "
                    "NegElboLoss instance."
                )

    def _resolve_optimizers(
        self, optimizers: optax.GradientTransformation | Sequence[OptimizerLike]
    ) -> Sequence[OptimizerLike]:
        if isinstance(optimizers, optax.GradientTransformation):
            return [Optimizer(list(self.loss.q.parameters), optimizers)]
        if isinstance(optimizers, str):
            if optimizers == "lbfgs":
                raise ValueError(
                    "LieselVI does not provide optimizers='lbfgs' because ELBO "
                    "estimates are usually stochastic. Pass an explicit LBFGS "
                    "optimizer sequence if you want to run a deterministic "
                    "L-BFGS experiment."
                )
            raise ValueError(  # noqa: TRY004
                "Pass a configured Optax transformation as "
                "optimizers=optax.adam(learning_rate=...) or a sequence of optimizers."
            )
        if isinstance(optimizers, Sequence):
            for index, optimizer in enumerate(optimizers):
                if isinstance(optimizer, optax.GradientTransformation):
                    raise TypeError(
                        f"optimizers[{index}] is a bare Optax transformation. Pass "
                        "a single transformation directly for all parameters, or "
                        "wrap each one in Optimizer(keys, transformation) for "
                        "separate parameter blocks."
                    )
            return optimizers
        raise TypeError(
            "optimizers must be a configured Optax transformation such as "
            "optax.adam(learning_rate=...) or a sequence of optimizers. "
            "Pass the transformation, not the optimizer factory."
        )

    def build_engine(self) -> OptimEngine:
        """
        Builds the low-level optimization engine.

        Returns
        -------
        OptimEngine
            Configured engine. Users may modify engine attributes before calling
            :meth:`OptimEngine.fit`.
        """
        from .engine import OptimEngine

        return OptimEngine(
            loss=self.loss,
            batches=self.batches,
            optimizers=self.optimizers,
            stopper=self.stopper,
            initial_state=self.model.state,
            seed=self.seed,
            loss_monitor=self.loss_monitor,
            save_position_history=self.save_position_history,
            show_progress=self.show_progress,
            progress_update_every=self.progress_update_every,
            show_step_progress=self.show_step_progress,
            step_progress_update_every=self.step_progress_update_every,
        )

    def fit(self) -> OptimResult:
        """
        Builds an engine and runs variational inference immediately.

        Returns
        -------
        OptimResult
            Result returned by :meth:`OptimEngine.fit`.
        """
        return self.build_engine().fit()
