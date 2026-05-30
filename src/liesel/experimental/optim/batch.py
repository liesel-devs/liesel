from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...model import Model
from .types import ModelInterface, ModelState, Position
from .util import guess_n


@dataclass
class Batches:
    """
    Defines mini-batches for observed entries in an optimizer position.

    ``Batches`` stores an index vector of length ``n`` and reshapes the first
    complete part of that vector into batches. The observed position entries named in
    ``position_keys`` are sliced with these indices. By default, every entry is sliced
    along axis ``0``; use ``default_axis`` or ``axes`` for arrays where observations
    live on another axis.

    Parameters
    ----------
    position_keys
        Names of the position entries that should be batched.
    n
        Number of observations along each batched axis.
    batch_size
        Number of observations per batch. If ``None``, batching is disabled by using
        a single batch with all ``n`` observations.
    shuffle
        Whether :meth:`permute_indices` should return a random permutation of the
        indices. If ``False``, :meth:`permute_indices` returns the indices unchanged.
    axes
        Optional mapping from position key to batching axis. Keys missing from this
        mapping use ``default_axis``.
    default_axis
        Batching axis for all position keys not listed in ``axes``.

    Attributes
    ----------
    indices
        Current ordering of the observations. Initialized as ``jnp.arange(n)`` and
        used by :attr:`batch_indices`. Assign the result of :meth:`permute_indices`
        to this attribute to use a fresh order.

    Notes
    -----
    If ``n`` is not divisible by ``batch_size``, only full batches are used and the
    final incomplete batch is dropped.

    Examples
    --------
    Create two batches of size four from ten observations:

    >>> from liesel.experimental.optim import Batches
    >>> batches = Batches(["y"], n=10, batch_size=4, shuffle=False)
    >>> batches.batch_indices.tolist()
    [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> batches.n_full_batches
    2

    With ``batch_size=None``, the object represents one full-data batch:

    >>> full_data = Batches(["y"], n=5, batch_size=None)
    >>> full_data.batch_size
    5
    >>> full_data.batch_indices.tolist()
    [[0, 1, 2, 3, 4]]

    ``axes`` can batch different entries along different axes:

    >>> import jax.numpy as jnp
    >>> batches = Batches(["x", "y"], n=5, batch_size=2, axes={"x": 1}, shuffle=False)
    >>> position = {
    ...     "x": jnp.arange(15).reshape(3, 5),
    ...     "y": jnp.arange(20).reshape(5, 4),
    ... }
    >>> batched = batches.get_batched_position(position, batch_index=0)
    >>> batched["x"].shape, batched["y"].shape
    ((3, 2), (2, 4))
    """

    position_keys: Sequence[str]
    n: int
    batch_size: int | None
    shuffle: bool = True
    axes: dict[str, int] | None = None
    default_axis: int = 0

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.n

        if self.batch_size < 1:
            raise ValueError(f"{self.batch_size=} is < 1, which is not allowed.")

        if self.n < self.batch_size:
            raise ValueError(
                f"{self.n=} is < {self.batch_size=}, which is not allowed."
            )

        if self.axes is None:
            self.axes = {}

        self.indices = jnp.arange(self.n)

    @classmethod
    def from_model(
        cls,
        model: Model,
        batch_size: int | None,
        position_keys: Sequence[str] | None = None,
        n: int | None = None,
        shuffle: bool = True,
        axes: dict[str, int] | None = None,
        default_axis: int = 0,
    ) -> Batches:
        """
        Builds a :class:`Batches` object from a Liesel model.

        Parameters
        ----------
        model
            Model containing the observed variables to batch.
        batch_size
            Number of observations per batch. If ``None``, batching is disabled and
            the returned object uses one full-data batch.
        position_keys
            Names of the observed position entries to batch. If ``None``, all observed
            variables in ``model`` are used.
        n
            Number of observations. If ``None``, the number is guessed from the model's
            observed variables along ``default_axis``.
        shuffle
            Whether epoch-wise calls to :meth:`permute_indices` should shuffle the
            indices. This is forced to ``False`` when ``batch_size`` is ``None``.
        axes
            Optional mapping from position key to batching axis.
        default_axis
            Axis used for guessing ``n`` and for position keys missing from ``axes``.

        Returns
        -------
        Batches
            Batch configuration for the model's observed data.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import Batches

        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([y])
        >>> batches = Batches.from_model(model, batch_size=2, position_keys=["y"])
        >>> batches.n, batches.batch_size, batches.position_keys
        (6, 2, ['y'])

        Passing ``batch_size=None`` disables shuffling and creates one full-data batch:

        >>> full_data = Batches.from_model(model, batch_size=None, position_keys=["y"])
        >>> full_data.shuffle, full_data.batch_indices.tolist()
        (False, [[0, 1, 2, 3, 4, 5]])
        """
        pos_keys = position_keys or list(model.observed)
        n = n or guess_n(model, axis=default_axis)

        if batch_size is None:
            shuffle = False

        batches = cls(
            pos_keys,
            batch_size=batch_size,
            n=n,
            shuffle=shuffle,
            axes=axes,
            default_axis=default_axis,
        )

        return batches

    @property
    def batch_share(self) -> float:
        """
        Scaling factor for turning a batch likelihood into a full-data likelihood.

        Returns
        -------
        float
            The ratio ``n / batch_size``.

        Examples
        --------
        >>> from liesel.experimental.optim import Batches
        >>> Batches(["y"], n=10, batch_size=4).batch_share
        2.5
        """
        assert self.batch_size is not None
        return self.n / self.batch_size

    @property
    def n_full_batches(self) -> int:
        """
        Number of complete batches.

        Returns
        -------
        int
            The integer quotient ``n // batch_size``.

        Examples
        --------
        >>> from liesel.experimental.optim import Batches
        >>> Batches(["y"], n=10, batch_size=4).n_full_batches
        2
        """
        assert self.batch_size is not None
        return int(self.n // self.batch_size)

    @property
    def likelihood_scalar(self) -> float:
        """
        Alias for the mini-batch likelihood scaling factor.

        Returns
        -------
        float
            The ratio ``n / batch_size``.

        Examples
        --------
        >>> from liesel.experimental.optim import Batches
        >>> Batches(["y"], n=10, batch_size=4).likelihood_scalar
        2.5
        """
        assert self.batch_size is not None
        return float(self.n / self.batch_size)

    def permute_indices(self, key: jax.Array) -> jax.Array:
        """
        Returns epoch indices, optionally shuffled.

        This method does not mutate :attr:`indices`. Assign the return value to
        ``indices`` if the object should use the new order.

        Parameters
        ----------
        key
            JAX pseudo-random key used when ``shuffle=True``.

        Returns
        -------
        jax.Array
            A vector of indices from ``0`` to ``n - 1``. The order is random if
            ``shuffle=True`` and unchanged otherwise.

        Examples
        --------
        >>> import jax
        >>> from liesel.experimental.optim import Batches

        >>> batches = Batches(["y"], n=6, batch_size=3, shuffle=False)
        >>> batches.permute_indices(jax.random.key(0)).tolist()
        [0, 1, 2, 3, 4, 5]

        >>> shuffled = Batches(["y"], n=6, batch_size=3, shuffle=True)
        >>> shuffled.indices = shuffled.permute_indices(jax.random.key(0))
        >>> sorted(shuffled.batch_indices.ravel().tolist())
        [0, 1, 2, 3, 4, 5]
        """
        if self.shuffle:
            all_indices = jax.random.permutation(key, self.indices)
        else:
            all_indices = self.indices

        return all_indices

    @property
    def batch_indices(self) -> jax.Array:
        """
        Batch index matrix.

        Returns
        -------
        jax.Array
            Integer array with shape ``(n_full_batches, batch_size)``. Each row gives
            the observation indices for one full batch.

        Examples
        --------
        >>> from liesel.experimental.optim import Batches
        >>> Batches(["y"], n=7, batch_size=3, shuffle=False).batch_indices.tolist()
        [[0, 1, 2], [3, 4, 5]]
        """
        assert self.batch_size is not None
        idx = self.indices[: self.n_full_batches * self.batch_size]
        batch_indices = jnp.reshape(idx, (self.n_full_batches, self.batch_size))
        return batch_indices

    def get_batched_position(self, position: Position, batch_index: int) -> Position:
        """
        Slices observed position entries for one batch.

        Parameters
        ----------
        position
            Mapping from position key to array. Every key listed in
            ``position_keys`` must be present and have length ``n`` along its batching
            axis.
        batch_index
            Row number in :attr:`batch_indices`.

        Returns
        -------
        Position
            Position containing only the batched entries named in ``position_keys``.

        Raises
        ------
        ValueError
            If an entry's length along its batching axis is not equal to ``n``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from liesel.experimental.optim import Batches

        >>> batches = Batches(["y"], n=6, batch_size=2, shuffle=False)
        >>> position = {"y": jnp.arange(6)}
        >>> batches.get_batched_position(position, batch_index=1)["y"].tolist()
        [2, 3]

        Batch along a non-leading axis:

        >>> batches = Batches(["x"], n=4, batch_size=2, default_axis=1, shuffle=False)
        >>> position = {"x": jnp.arange(12).reshape(3, 4)}
        >>> batches.get_batched_position(position, batch_index=0)["x"].tolist()
        [[0, 1], [4, 5], [8, 9]]
        """
        idx = self.batch_indices[batch_index]
        batched_position = {}
        assert isinstance(self.axes, dict)
        for key in self.position_keys:
            axis = self.axes.get(key, self.default_axis)

            n_this_key = jnp.shape(position[key])[axis]
            if not jnp.shape(position[key])[axis] == self.n:
                raise ValueError(
                    f"{key} has n={n_this_key}, which is incompatible with the "
                    f"given sample size of n={self.n}."
                )

            batched = jnp.take(position[key], idx, axis=axis)
            batched_position[key] = batched

        return Position(batched_position)

    def extract_batched_position(
        self,
        interface: ModelInterface | Model,
        model_state: ModelState,
        batch_number: int,
    ) -> Position:
        """
        Extracts observed data from a model state and returns one batch.

        Parameters
        ----------
        interface
            Model or model interface used to extract the observed position entries.
        model_state
            State from which ``position_keys`` are extracted.
        batch_number
            Row number in :attr:`batch_indices`.

        Returns
        -------
        Position
            Batched position extracted from ``model_state``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import liesel.model as lsl
        >>> from liesel.experimental.optim import Batches

        >>> y = lsl.Var.new_obs(jnp.arange(6.0), name="y")
        >>> model = lsl.Model([y])
        >>> batches = Batches(["y"], n=6, batch_size=2, shuffle=False)
        >>> batches.extract_batched_position(model, model.state, 2)["y"].tolist()
        [4.0, 5.0]
        """
        obs = interface.extract_position(self.position_keys, model_state)
        return self.get_batched_position(obs, batch_number)

    def _tree_flatten(self):
        children = (self.indices,)
        aux_data = {
            "position_keys": self.position_keys,
            "n": self.n,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "axes": self.axes,
            "default_axis": self.default_axis,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        bi = cls(**aux_data)
        bi.indices = children[0]
        return bi

    def __repr__(self) -> str:
        name = type(self).__name__
        out = (
            f"{name}(n={self.n}, "
            f"batch_size={self.batch_size}, default_axis={self.default_axis})"
        )
        return out


jax.tree_util.register_pytree_node(
    Batches, Batches._tree_flatten, Batches._tree_unflatten
)
