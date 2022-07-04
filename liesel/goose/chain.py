"""
# MCMC chains

This module is experimental. Expect API changes.
"""

from typing import Callable, Generic, Protocol, Sequence, TypeVar

import jax
import numpy as np

from liesel.option import Option

from .epoch import EpochConfig
from .pytree import concatenate_leaves, slice_leaves
from .types import PyTree

TPyTree = TypeVar("TPyTree", bound=PyTree)


class Chain(Protocol[TPyTree]):
    """
    A `Chain` stores multiple chucks of pytrees and concatenates them along a time
    axis.

    A `Chain` operates on the assumption that all chunks are pytrees with the same
    structure.

    The `Chain` can either represent one chain or multiple independent chains.
    For a single chain, the first dimension of each leaf represents the time.
    For a multi-chain, the first dimension is the chain index and the second dimension
    represents the time. Consequently, the leafs in the pytree must have a dimension of
    at least one (single chain) or two (multi-chain).

    For a multi-chain, the size of the first dimension (the chain index) is not
    allowed to vary among chunks.
    """

    @property
    def multichain(self) -> bool:
        """Returns true if the chain is a multi-chain, and false otherwise."""

    def append(self, chunk: TPyTree) -> None:
        """Appends a chunk to the chain."""

    def get(self) -> Option[TPyTree]:
        """
        Returns all chunks combined into one pytree.

        The option is none if no samples are in the chain.
        """


class EpochChain(Chain[TPyTree]):
    """
    An `EpochChain` is a `Chain` with an associated `EpochConfig`.

    The implementation must implement thinning. That is,
    if epoch.thinning > 1 and enabled in contructor, the chain must
    safe only every epch.thinning element
    """

    @property
    def epoch(self) -> EpochConfig:
        """Returns the associated `EpochConfig`."""


# implementations


class ListChain(Generic[TPyTree]):
    """Implements the `Chain` protocol with a list as storage."""

    def __init__(self, multichain: bool):
        self._multichain = multichain
        self._chunks_list: list[TPyTree] = []

    @property
    def multichain(self) -> bool:
        return self._multichain

    def append(self, chunk: TPyTree) -> None:
        return self._chunks_list.append(chunk)

    def _concatenate(self) -> None:
        time_axis = 1 if self.multichain else 0
        combined = concatenate_leaves(self._chunks_list, time_axis)
        if combined is not None:
            self._chunks_list = [combined]

    def get(self) -> Option[TPyTree]:
        if len(self._chunks_list) == 0:
            return Option(None)
        else:
            self._concatenate()
            return Option(self._chunks_list[0])


class ListEpochChain(ListChain[TPyTree]):
    """Implements the `EpochChain` protocol with a list as storage."""

    def __init__(
        self, multichain: bool, epoch: EpochConfig, apply_thinning: bool = False
    ):
        super().__init__(multichain)
        self._epoch = epoch
        self._apply_thinning = apply_thinning
        self._states_counter = 1

    @property
    def epoch(self) -> EpochConfig:
        return self._epoch

    def append(self, chunk: TPyTree) -> None:
        if self._apply_thinning:
            th = self._epoch.thinning
            size = jax.tree_leaves(chunk)[0].shape[1]
            idx = np.arange(size)[(self._states_counter + np.arange(size)) % th == 0]

            self._states_counter += size

            if len(idx) > 0:
                chunk = slice_leaves(chunk, np.s_[:, idx, ...])
                return super().append(chunk)
        else:
            return super().append(chunk)


class EpochChainManager(Generic[TPyTree]):
    """
    An `EpochChainManager` is a container for multiple epoch chains.

    The chains can be concatenated over multiple epochs. Thinning defined in epochs
    can be switched on or of with the constructor flag
    """

    def __init__(self, multichain: bool, apply_thinning: bool = False) -> None:
        self._multichain = multichain
        self._chains: list[ListEpochChain[TPyTree]] = []
        self._apply_thinning = apply_thinning

    @property
    def current_epoch(self) -> EpochConfig:
        return self._chains[-1].epoch

    def advance_epoch(self, epoch: EpochConfig) -> None:
        new_chain: ListEpochChain[TPyTree] = ListEpochChain(
            self._multichain, epoch, self._apply_thinning
        )
        self._chains.append(new_chain)

    def append(self, chunk: TPyTree) -> None:
        self._chains[-1].append(chunk)

    def get_epochs(self) -> Sequence[EpochConfig]:
        return [c.epoch for c in self._chains]

    def get_specific_chain(self, epoch_number: int) -> ListEpochChain[TPyTree]:
        return self._chains[epoch_number]

    def get_current_chain(self) -> ListEpochChain[TPyTree]:
        return self._chains[-1]

    def get_current_epoch(self) -> EpochConfig:
        return self._chains[-1].epoch

    def combine(self, epoch_numbers: Sequence[int]) -> Option[TPyTree]:
        chain: ListChain[TPyTree] = ListChain(self._multichain)
        for num in epoch_numbers:
            epoch_chain = self._chains[num]
            chunk = epoch_chain.get()
            if chunk.is_some():
                chain.append(chunk.unwrap())
        return chain.get()

    def combine_all(self) -> Option[TPyTree]:
        chain: ListChain[TPyTree] = ListChain(self._multichain)
        for epoch_chain in self._chains:
            chunk = epoch_chain.get()
            if chunk.is_some():
                chain.append(chunk.unwrap())
        return chain.get()

    def combine_filtered(
        self, predicate: Callable[[EpochConfig], bool]
    ) -> Option[TPyTree]:
        chain: ListChain[TPyTree] = ListChain(self._multichain)
        for epoch_chain in self._chains:
            if predicate(epoch_chain.epoch):
                chunk = epoch_chain.get()
                if chunk.is_some():
                    chain.append(chunk.unwrap())
        return chain.get()
