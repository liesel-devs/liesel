"""Train, validation, and test splitting utilities for optimizer positions."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...model import Model
from .types import ModelInterface, ModelState, Position
from .util import guess_n


@dataclass
class PositionSplit:
    """
    Container for train, validation, and test position dictionaries.

    A ``PositionSplit`` is usually returned by :meth:`Split.split_position` or
    :meth:`PositionSplit.from_model`. It stores the split observed position entries
    together with the corresponding split sizes.

    Parameters
    ----------
    train
        Position entries used for optimization.
    validate
        Position entries used for validation or early stopping.
    test
        Position entries reserved for testing.
    n_train
        Number of training observations.
    n_validate
        Number of validation observations.
    n_test
        Number of test observations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import PositionSplit
    >>> from liesel.experimental.optim.types import Position
    >>> split = PositionSplit(
    ...     train=Position({"x": jnp.arange(3)}),
    ...     validate=Position({"x": jnp.arange(3, 5)}),
    ...     test=Position({"x": jnp.arange(5, 6)}),
    ...     n_train=3,
    ...     n_validate=2,
    ...     n_test=1,
    ... )
    >>> split
    PositionSplit(train=3, validate=2, test=1)
    >>> split.validate["x"].tolist()
    [3, 4]
    """

    train: Position
    validate: Position
    test: Position

    n_train: int
    n_validate: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out

    @staticmethod
    def from_model(
        model: Model,
        position_keys: Sequence[str] | None = None,
        n: int | None = None,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
    ) -> PositionSplit:
        """
        Builds a :class:`PositionSplit` from the observed variables in a model.

        Parameters
        ----------
        model
            Model containing the observed variables to split.
        position_keys
            Names of observed position entries to split. If ``None``, all observed
            variables in ``model`` are used.
        n
            Number of observations along the split axis. If ``None``, the number is
            guessed from ``model`` along ``default_axis``.
        share_validate
            Share of observations assigned to the validation split.
        share_test
            Share of observations assigned to the test split.
        axes
            Optional mapping from position key to split axis. Keys missing from this
            mapping use ``default_axis``.
        default_axis
            Split axis for all position keys not listed in ``axes``.
        shuffle
            Whether observations are shuffled before splitting.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        PositionSplit
            Split observed position entries extracted from ``model``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import PositionSplit
        >>> y = lsl.Var.new_obs(jnp.arange(10.0), name="y")
        >>> model = lsl.Model([y])
        >>> split = PositionSplit.from_model(
        ...     model,
        ...     position_keys=["y"],
        ...     share_validate=0.2,
        ...     share_test=0.1,
        ... )
        >>> split.n_train, split.n_validate, split.n_test
        (7, 2, 1)
        >>> split.train["y"].tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        """
        pos_keys = position_keys or list(model.observed)
        n = n or guess_n(model, axis=default_axis)
        splitter = Split.from_share(
            position_keys=pos_keys,
            n=n,
            share_validate=share_validate,
            share_test=share_test,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle,
            seed=seed,
        )

        pos = model.extract_position(pos_keys)
        return splitter.split_position(pos)


@dataclass
class StateSplit:
    """
    Container for a split model state and held-out observed positions.

    ``StateSplit`` is returned by :meth:`Split.split_state`. The training
    observations are written back into ``train_state``; validation and test
    observations are kept as positions so they can be inserted later.

    Parameters
    ----------
    train_state
        Model state updated with training observations.
    pos_validate
        Validation observed position entries.
    pos_test
        Test observed position entries.
    n_train
        Number of training observations.
    n_validate
        Number of validation observations.
    n_test
        Number of test observations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim.split import StateSplit
    >>> from liesel.experimental.optim.types import Position
    >>> split = StateSplit(
    ...     train_state={},
    ...     pos_validate=Position({"x": jnp.arange(2)}),
    ...     pos_test=Position({"x": jnp.arange(0)}),
    ...     n_train=4,
    ...     n_validate=2,
    ...     n_test=0,
    ... )
    >>> split
    StateSplit(train=4, validate=2, test=0)
    """

    train_state: ModelState
    pos_validate: Position
    pos_test: Position

    n_train: int
    n_validate: int
    n_test: int

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out


@dataclass
class Split:
    """
    Defines how observed position entries are split into train, validation, and test.

    ``Split`` stores a vector of observation indices. The first ``n_train`` indices
    become the training split, the next ``n_validate`` indices become the validation
    split, and the final ``n_test`` indices become the test split. If ``shuffle=True``,
    the index vector is permuted once during initialization.

    Parameters
    ----------
    position_keys
        Names of position entries that should be split.
    n
        Number of observations along each split axis.
    n_validate
        Number of validation observations.
    n_test
        Number of test observations.
    n_train
        Number of training observations. If left at ``-1``, it is computed as
        ``n - n_validate - n_test``.
    axes
        Optional mapping from position key to split axis. Keys missing from this
        mapping use ``default_axis``.
    default_axis
        Split axis for all position keys not listed in ``axes``.
    shuffle
        Whether to shuffle observations during initialization.
    seed
        Seed or JAX pseudo-random key used when ``shuffle=True``. If ``None``, the
        current time is used.

    Attributes
    ----------
    indices
        Current observation order. Initialized as ``jnp.arange(n)`` and optionally
        shuffled in :meth:`__post_init__`.

    Raises
    ------
    ValueError
        If the requested split sizes imply a negative number of training
        observations, if their sum exceeds ``n``, or if validation or test shares are
        negative.

    Examples
    --------
    Split one vector without shuffling:

    >>> import jax.numpy as jnp
    >>> from liesel.experimental.optim import Split
    >>> splitter = Split(["x"], n=10, n_validate=2, n_test=1, shuffle=False)
    >>> splitter
    Split(train=7, validate=2, test=1)
    >>> splitter.indices_train.tolist()
    [0, 1, 2, 3, 4, 5, 6]
    >>> split = splitter.split_position({"x": jnp.arange(10)})
    >>> (
    ...     split.train["x"].tolist(),
    ...     split.validate["x"].tolist(),
    ...     split.test["x"].tolist(),
    ... )
    ([0, 1, 2, 3, 4, 5, 6], [7, 8], [9])

    Split different entries along different axes:

    >>> splitter = Split(
    ...     ["x", "y"],
    ...     n=4,
    ...     n_validate=1,
    ...     n_test=1,
    ...     axes={"x": 1},
    ...     shuffle=False,
    ... )
    >>> position = {
    ...     "x": jnp.arange(8).reshape(2, 4),
    ...     "y": jnp.arange(12).reshape(4, 3),
    ... }
    >>> split = splitter.split_position(position)
    >>> split.train["x"].tolist()
    [[0, 1], [4, 5]]
    >>> split.validate["y"].tolist()
    [[6, 7, 8]]
    """

    position_keys: Sequence[str]
    n: int
    n_validate: int = 0
    n_test: int = 0
    n_train: int = -1
    axes: dict[str, int] | None = None
    default_axis: int = 0
    shuffle: bool = False
    seed: jax.Array | int | None = None

    def __post_init__(self):
        if self.axes is None:
            self.axes = {}

        if self.n_train == -1:
            self.n_train = self.n - self.n_validate - self.n_test

        self.indices = jnp.arange(self.n)

        if self.n_train < 0:
            raise ValueError(
                f"The given {self.n_validate=} and {self.n_test=} imply "
                f"a total of {self.n_train=}, which < 0."
            )

        n = self.n_train + self.n_validate + self.n_test
        if n > self.n:
            raise ValueError(
                f"The given {self.n_train=}, {self.n_validate=}, and {self.n_test=} "
                f"imply a total of {n=}, which is > the provided {self.n=}."
            )

        if self.share_validate < 0.0 or self.share_test < 0.0:
            raise ValueError(
                f"One of {self.share_validate=} or {self.share_test=} is negative, "
                "which is not allowed."
            )

        if self.shuffle:
            if isinstance(self.seed, jax.Array):
                key = self.seed
            else:
                seed = int(time.time()) if self.seed is None else self.seed
                key = jax.random.key(seed)
            self.indices = self.permute_indices(key)

    @property
    def share_validate(self) -> float:
        """
        Share of observations assigned to validation.

        Returns
        -------
        float
            The ratio ``n_validate / n``.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10, n_validate=2).share_validate
        0.2
        """
        return self.n_validate / self.n

    @property
    def share_test(self) -> float:
        """
        Share of observations assigned to testing.

        Returns
        -------
        float
            The ratio ``n_test / n``.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=10, n_test=3).share_test
        0.3
        """
        return self.n_test / self.n

    @classmethod
    def from_share(
        cls,
        position_keys: Sequence[str],
        n: int,
        share_validate: float = 0.0,
        share_test: float = 0.0,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
        shuffle: bool = False,
        seed: jax.Array | int | None = None,
    ) -> Split:
        """
        Builds a :class:`Split` from validation and test proportions.

        The number of validation and test observations is computed with
        ``int(n * share)``. Any fractional remainder is therefore truncated.

        Parameters
        ----------
        position_keys
            Names of position entries that should be split.
        n
            Number of observations along each split axis.
        share_validate
            Share of observations assigned to validation.
        share_test
            Share of observations assigned to testing.
        axes
            Optional mapping from position key to split axis.
        default_axis
            Split axis for all position keys not listed in ``axes``.
        shuffle
            Whether to shuffle observations during initialization.
        seed
            Seed or JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        Split
            Splitter with integer split sizes derived from the shares.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split.from_share(
        ...     ["x"],
        ...     n=10,
        ...     share_validate=0.25,
        ...     share_test=0.25,
        ... )
        >>> splitter.n_train, splitter.n_validate, splitter.n_test
        (6, 2, 2)
        >>> splitter.share_validate, splitter.share_test
        (0.2, 0.2)
        """
        n_validate = int(n * share_validate)
        n_test = int(n * share_test)

        return cls(
            position_keys=position_keys,
            n=n,
            n_validate=n_validate,
            n_test=n_test,
            axes=axes,
            default_axis=default_axis,
            shuffle=shuffle,
            seed=seed,
        )

    def permute_indices(self, key: jax.Array) -> jax.Array:
        """
        Returns a random permutation of the current index vector.

        This method does not mutate :attr:`indices`.

        Parameters
        ----------
        key
            JAX pseudo-random key.

        Returns
        -------
        jax.Array
            Permuted copy of :attr:`indices`.

        Examples
        --------
        >>> import jax
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split(["x"], n=5)
        >>> permuted = splitter.permute_indices(jax.random.key(0))
        >>> sorted(permuted.tolist())
        [0, 1, 2, 3, 4]
        >>> splitter.indices.tolist()
        [0, 1, 2, 3, 4]
        """
        return jax.random.permutation(key, self.indices)

    @property
    def indices_train(self) -> jax.Array:
        """
        Observation indices for the training split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_train.tolist()
        [0, 1, 2]
        """
        return self.indices[: self.n_train]

    @property
    def indices_validate(self) -> jax.Array:
        """
        Observation indices for the validation split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_validate.tolist()
        [3, 4]
        """
        start = self.n_train
        end = self.n_train + self.n_validate
        return self.indices[start:end]

    @property
    def indices_test(self) -> jax.Array:
        """
        Observation indices for the test split.

        Examples
        --------
        >>> from liesel.experimental.optim import Split
        >>> Split(["x"], n=6, n_validate=2, n_test=1).indices_test.tolist()
        [5]
        """
        start = self.n_train + self.n_validate
        end = self.n_train + self.n_validate + self.n_test
        return self.indices[start:end]

    def split_position(self, position: Position) -> PositionSplit:
        """
        Splits position entries into train, validation, and test positions.

        Parameters
        ----------
        position
            Mapping containing every key in :attr:`position_keys`. Each selected
            entry must have length ``n`` along its split axis.

        Returns
        -------
        PositionSplit
            Split position entries and split sizes.

        Raises
        ------
        ValueError
            If a selected position entry has an incompatible length along its split
            axis.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import Split
        >>> splitter = Split(["x"], n=5, n_validate=1, n_test=1)
        >>> split = splitter.split_position({"x": jnp.arange(5)})
        >>> split.train["x"].tolist()
        [0, 1, 2]
        >>> split.validate["x"].tolist(), split.test["x"].tolist()
        ([3], [4])

        Use ``axes`` to split an array along a non-leading axis:

        >>> splitter = Split(["x"], n=4, n_validate=1, axes={"x": 1})
        >>> split = splitter.split_position({"x": jnp.arange(8).reshape(2, 4)})
        >>> split.train["x"].tolist()
        [[0, 1, 2], [4, 5, 6]]
        >>> split.validate["x"].tolist()
        [[3], [7]]
        """
        train_position = {}
        validation_position = {}
        test_position = {}

        assert isinstance(self.axes, dict)
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)

            n_this_key = jnp.shape(position[key])[axis]
            if not jnp.shape(position[key])[axis] == self.n:
                raise ValueError(
                    f"{key} has n={n_this_key}, which is incompatible with the "
                    f"given sample size of n={self.n}."
                )

            train_values = jnp.take(position[key], self.indices_train, axis=axis)
            validation_values = jnp.take(
                position[key], self.indices_validate, axis=axis
            )
            test_values = jnp.take(position[key], self.indices_test, axis=axis)

            train_position[key] = train_values
            validation_position[key] = validation_values
            test_position[key] = test_values

        split = PositionSplit(
            train=Position(train_position),
            validate=Position(validation_position),
            test=Position(test_position),
            n_train=self.n_train,
            n_validate=self.n_validate,
            n_test=self.n_test,
        )
        return split

    def split_state(
        self, interface: ModelInterface | Model, model_state: ModelState
    ) -> StateSplit:
        """
        Splits observed entries from a model state.

        The training observations are written into a new model state. Validation and
        test observations are returned separately as positions.

        Parameters
        ----------
        interface
            Model or model interface used to extract and update observed position
            entries.
        model_state
            Model state containing the full observed data.

        Returns
        -------
        StateSplit
            Training state plus validation and test positions.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import Split
        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([y])
        >>> splitter = Split(["y"], n=6, n_validate=2, n_test=1)
        >>> split = splitter.split_state(model, model.state)
        >>> model.extract_position(["y"], split.train_state)["y"].tolist()
        [0.0, 1.0, 2.0]
        >>> split.pos_validate["y"].tolist(), split.pos_test["y"].tolist()
        ([3.0, 4.0], [5.0])
        """
        obs = interface.extract_position(self.position_keys, model_state)
        split = self.split_position(obs)
        train_state = interface.update_state(split.train, model_state)
        return StateSplit(
            train_state,
            split.validate,
            split.test,
            n_train=self.n_train,
            n_validate=self.n_validate,
            n_test=self.n_test,
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(train={self.n_train}, "
            f"validate={self.n_validate}, test={self.n_test})"
        )
        return out
