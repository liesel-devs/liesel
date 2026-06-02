"""Opinionated variational inference setup for Liesel models."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import optax

from ...model import Model
from ._engine_utils import BatchConfig, SplitConfig
from .batch import Batches, BatchManager
from .optimizer import Optimizer
from .split import PositionSplit, PositionSplitManager
from .stop import Stopper
from .vi import NegElboLoss

if TYPE_CHECKING:
    from .engine import OptimEngine
    from .state import OptimResult


class LieselVI:
    """
    Builds an :class:`.OptimEngine` for variational inference.

    ``LieselVI`` is the quick-start wrapper for ELBO optimization. It constructs one
    of the standard :class:`.NegElboLoss` variational families, default training
    batches, and an Adam optimizer over the variational parameters unless these
    pieces are supplied explicitly. Variational-family initialization belongs to
    :class:`.NegElboLoss` and :class:`.VDist`; pass a custom ``NegElboLoss`` when you
    need Laplace or custom initialization.

    Parameters
    ----------
    model
        Target Liesel model.
    loss
        Either one of ``"mvn_diag"``, ``"mvn_tril"``, and ``"mvn_blocked"``, or an
        explicit :class:`.NegElboLoss` instance.
    elbo
        Deprecated alias for ``loss``.
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
        automatically uses :class:`.PositionSplitManager`.
    optimizers
        Either explicit optimizers or the string shortcut ``"adam"``. L-BFGS is not
        provided as a string shortcut because ELBO estimates are usually stochastic.
    stopper
        Maximum-epoch and early-stopping configuration.
    seed
        Integer seed. If ``None``, the current Unix time is used.
    n
        Optional scalar observation count for scalar default splitting.
    axes
        Optional mapping from observed position key to split/batch axis.
    default_axis
        Split/batch axis for observed keys missing from ``axes``.
    shuffle_batches
        Whether default mini-batches should shuffle observations.
    batch_mode
        Mode used when default batches require a :class:`.BatchManager`.
    epoch_size
        Joint epoch length used by default :class:`.BatchManager` objects in
        ``mode="resample"``.
    nsamples
        Monte Carlo sample count for internally constructed training ELBOs.
    nsamples_validate
        Monte Carlo sample count for internally constructed validation ELBOs.
    scale_loss
        Whether internally constructed ELBO losses should be divided by the training
        sample size. ``"auto"`` scales the loss. This setting has no effect when
        ``loss`` is an explicit :class:`.NegElboLoss`.
    regularize_q_prior
        Whether internally constructed ELBOs should include priors in the
        variational model as regularization terms.
    train_monitor
        Training-data monitor used by :class:`.OptimEngine` when no validation split
        exists.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import liesel.model as lsl
    >>> import tensorflow_probability.substrates.jax.distributions as tfd
    >>> from liesel.experimental.optim import LieselVI
    >>> loc = lsl.Var.new_param(jnp.array(0.0), name="loc")
    >>> y = lsl.Var.new_obs(
    ...     jnp.array([0.0, 1.0]),
    ...     lsl.Dist(tfd.Normal, loc=loc, scale=1.0),
    ...     name="y",
    ... )
    >>> model = lsl.Model([y])
    >>> engine = LieselVI(model, seed=1).build_engine()
    >>> type(engine).__name__
    'OptimEngine'
    >>> type(engine.loss).__name__
    'NegElboLoss'
    """

    def __init__(
        self,
        model: Model,
        *,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss = "mvn_diag",
        elbo: Literal["mvn_diag", "mvn_tril", "mvn_blocked"]
        | NegElboLoss
        | None = None,
        batches: BatchConfig | None = None,
        batch_size: int | None = None,
        split: SplitConfig | None = None,
        optimizers: Sequence[Optimizer] | Literal["adam"] = "adam",
        stopper: Stopper = Stopper(epochs=1000, patience=10, rtol=1e-6),
        seed: int | None = None,
        n: int | None = None,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle_batches: bool = True,
        batch_mode: Literal["strict", "resample"] = "resample",
        epoch_size: Literal["max", "min"] | int = "max",
        nsamples: int = 10,
        nsamples_validate: int = 50,
        scale_loss: bool | Literal["auto"] = "auto",
        regularize_q_prior: bool = True,
        train_monitor: Literal[
            "auto", "epoch_average", "weighted_epoch_average", "full_data"
        ] = "auto",
    ) -> None:
        if batches is not None and batch_size is not None:
            raise ValueError("Pass either batches or batch_size, not both.")

        if elbo is not None:
            if loss != "mvn_diag":
                raise ValueError("Pass either loss or elbo, not both.")
            loss = elbo

        self.model = model
        self.seed = int(time.time()) if seed is None else seed
        self.stopper = stopper
        self.split = self._resolve_split(loss, split, n, axes, default_axis)
        self.loss = self._resolve_loss(
            loss,
            nsamples=nsamples,
            nsamples_validate=nsamples_validate,
            scale_loss=scale_loss,
            regularize_q_prior=regularize_q_prior,
        )
        self.batches = self._resolve_batches(
            batches=batches,
            batch_size=batch_size,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle_batches,
            mode=batch_mode,
            epoch_size=epoch_size,
        )
        self.optimizers = self._resolve_optimizers(optimizers)
        self.train_monitor = train_monitor

    def _resolve_split(
        self,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss,
        split: SplitConfig | None,
        n: int | None,
        axes: dict[str, int] | None,
        default_axis: int,
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
            n=n,
            axes=axes,
            default_axis=default_axis,
            multi_size="manager",
        )

    def _resolve_loss(
        self,
        loss: Literal["mvn_diag", "mvn_tril", "mvn_blocked"] | NegElboLoss,
        nsamples: int,
        nsamples_validate: int,
        scale_loss: bool | Literal["auto"],
        regularize_q_prior: bool,
    ) -> NegElboLoss:
        if isinstance(loss, NegElboLoss):
            return loss

        if scale_loss == "auto":
            scale = True
        elif isinstance(scale_loss, bool):
            scale = scale_loss
        else:
            raise ValueError("scale_loss must be True, False, or 'auto'.")

        match loss:
            case "mvn_diag":
                return NegElboLoss.mvn_diag(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    nsamples_validate=nsamples_validate,
                    scale=scale,
                    regularize_q_prior=regularize_q_prior,
                )
            case "mvn_tril":
                return NegElboLoss.mvn_tril(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    nsamples_validate=nsamples_validate,
                    scale=scale,
                    regularize_q_prior=regularize_q_prior,
                )
            case "mvn_blocked":
                return NegElboLoss.mvn_blocked(
                    self.model,
                    split=self.split,
                    nsamples=nsamples,
                    nsamples_validate=nsamples_validate,
                    scale=scale,
                    regularize_q_prior=regularize_q_prior,
                )
            case _:
                raise ValueError(
                    "loss must be 'mvn_diag', 'mvn_tril', 'mvn_blocked', or a "
                    "NegElboLoss instance."
                )

    def _resolve_batches(
        self,
        batches: BatchConfig | None,
        batch_size: int | None,
        axes: dict[str, int] | None,
        default_axis: int,
        shuffle: bool,
        mode: Literal["strict", "resample"],
        epoch_size: Literal["max", "min"] | int,
    ) -> BatchConfig:
        if batches is not None:
            return batches

        shuffle = False if batch_size is None else shuffle
        if isinstance(self.split, PositionSplitManager):
            children = [
                Batches(
                    position_keys=child.position_keys,
                    n=child.n_train,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    axes=axes,
                    default_axis=default_axis,
                    sample_with_replacement=(
                        mode == "resample"
                        and batch_size is not None
                        and batch_size > child.n_train
                    ),
                )
                for child in self.split.splits
            ]
            return BatchManager(children, mode=mode, epoch_size=epoch_size)

        position_keys = self.split.position_keys or list(self.model.observed)
        return Batches(
            position_keys=position_keys,
            n=self.split.n_train,
            batch_size=batch_size,
            shuffle=shuffle,
            axes=axes,
            default_axis=default_axis,
        )

    def _resolve_optimizers(
        self, optimizers: Sequence[Optimizer] | str
    ) -> Sequence[Optimizer]:
        if not isinstance(optimizers, str):
            return optimizers

        if optimizers == "adam":
            return [
                Optimizer(
                    list(self.loss.q.parameters),
                    optimizer=optax.adam(learning_rate=1e-3),
                )
            ]

        if optimizers == "lbfgs":
            raise ValueError(
                "LieselVI does not provide optimizers='lbfgs' because ELBO "
                "estimates are usually stochastic. Pass an explicit LBFGS optimizer "
                "sequence if you want to run a deterministic L-BFGS experiment."
            )

        raise ValueError("optimizers must be 'adam' or a sequence.")

    @property
    def elbo(self) -> NegElboLoss:
        """Alias for :attr:`loss`."""
        return self.loss

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
            train_monitor=self.train_monitor,
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
