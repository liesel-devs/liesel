"""Private read-only mapping helpers with IPython key completion."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Never, Self, overload


class _KeyCompletableMapping[V](Mapping[str, V]):
    """
    A read-only mapping view whose string keys can be completed by IPython.

    This class is used for named collections such as ``Model.vars`` and
    ``Group.nodes``. IPython recognizes the ``_ipython_key_completions_`` hook
    when completing an expression such as ``model.vars["`` and obtains the
    suggestions from the mapping at runtime. This complements static language
    servers, which cannot infer names created dynamically while building a model.

    Parameters
    ----------
    mapping
        The dictionary exposed through the read-only view.

    Notes
    -----
    The view prevents mutation *through the view*; it does not make the backing
    dictionary immutable. Changes made to the same dictionary by its owner remain
    visible. If the owner replaces the dictionary, an already-held view continues
    to refer to the old dictionary, matching ``MappingProxyType`` semantics.

    Callers that pass a newly created dictionary, as ``Model.parameters`` and
    ``Model.observed`` do, intentionally create a snapshot instead of a live view.

    This private type implements the commonly used ``MappingProxyType`` operations
    needed by Liesel, but it is not intended to reproduce every concrete-type or
    introspection detail. Public code should depend on the ``Mapping`` interface
    rather than this class.
    """

    def __init__(self, mapping: dict[str, V]) -> None:
        self._mapping = mapping

    def __getitem__(self, key: str) -> V:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __or__(self, other: Mapping[str, Any]) -> dict[str, Any]:
        return dict(self) | dict(other)

    def __ror__(self, other: Mapping[str, Any]) -> dict[str, Any]:
        return dict(other) | dict(self)

    def __repr__(self) -> str:
        return f"mappingproxy({self._mapping!r})"

    def __str__(self) -> str:
        return str(self._mapping)

    def __reversed__(self) -> Iterator[str]:
        return reversed(self._mapping)

    def copy(self) -> dict[str, V]:
        return dict(self)

    def _ipython_key_completions_(self) -> list[str]:
        """Return the runtime keys offered by IPython during key completion."""
        return list(self)


class _KeyCompletableProperty[O, V]:
    """
    A read-only descriptor for properties returning key-completable mappings.

    IPython's default ``limited`` completion evaluator rejects ordinary
    ``property`` access before it can inspect the returned mapping's
    ``_ipython_key_completions_`` hook. This descriptor preserves property-like
    access and assignment protection while allowing IPython to reach that hook.
    It is intended only for inexpensive getters returning
    :class:`_KeyCompletableMapping`.

    Notes
    -----
    IPython may execute the getter whenever completion is requested. Decorated
    getters must therefore be deterministic, inexpensive, and free of side
    effects. Do not use this as a general replacement for ``property``.

    The solution relies on how IPython currently distinguishes ``property`` from
    other descriptors in its guarded evaluator. A future IPython release could
    change that behavior, so the limited-evaluator regression test should remain
    in place. If that test fails after an IPython upgrade, prefer an explicit
    upstream completion API over broadening what the evaluator may execute.

    The ``__set__`` method makes this a data descriptor, preventing instance
    assignment from shadowing it. Its ``Never`` annotation also makes such
    assignment invalid to static type checkers.
    """

    def __init__(self, getter: Callable[[O], _KeyCompletableMapping[V]]) -> None:
        self._getter = getter
        self.__doc__ = getter.__doc__

    @overload
    def __get__(self, instance: None, owner: type[O] | None = None) -> Self: ...

    @overload
    def __get__(
        self, instance: O, owner: type[O] | None = None
    ) -> _KeyCompletableMapping[V]: ...

    def __get__(
        self, instance: O | None, owner: type[O] | None = None
    ) -> Self | _KeyCompletableMapping[V]:
        if instance is None:
            return self
        return self._getter(instance)

    def __set__(self, instance: O, value: Never) -> None:
        raise AttributeError("mapping is read-only")
