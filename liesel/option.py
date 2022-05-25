"""
# A Rust-inspired Option type for Liesel and Goose
"""

from __future__ import annotations

import weakref
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Option(Generic[T]):
    """An Option type inspired by the Rust stdlib.

    See <https://doc.rust-lang.org/std/option/enum.Option.html> for descriptions
    of the methods.
    """

    def __init__(self, value: T | None) -> None:
        self._value = value

    @property
    def value(self) -> T | None:
        return self._value

    @value.setter
    def value(self, value: T | None) -> None:
        self._value = value

    @staticmethod
    def some(value: T) -> Option[T]:
        return Option(value)

    @staticmethod
    def none() -> Option[T]:
        return Option(None)

    def is_some(self) -> bool:
        return self.value is not None

    def is_none(self) -> bool:
        return self.value is None

    def expect(self, msg: str) -> T:
        if self.value is None:
            raise RuntimeError(msg)

        return self.value

    def unwrap(self) -> T:
        return self.expect(f"Trying to unwrap None from {repr(self)}")

    def unwrap_or(self, default: T) -> T:
        if self.value is None:
            return default

        return self.value

    def map(self, f: Callable[[T], U]) -> Option[U]:
        if self.value is None:
            return Option.none()

        return Option.some(f(self.value))

    def map_or(self, default: U, f: Callable[[T], U]) -> U:
        if self.value is None:
            return default

        return f(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Option):
            return self.value == other.value
        else:
            raise NotImplementedError(
                f"Option cannot be compared to {type(other).__name__}"
            )

    def __repr__(self) -> str:
        return f"Option({repr(self.value)})"

    def __str__(self) -> str:
        return f"Option({self.value})"


class WeakOption(Option[T]):
    def __init__(self, value: T | None) -> None:
        self._ref = Option(value).map(lambda x: weakref.ref(x))

    @property
    def value(self) -> T | None:
        return self._ref.map_or(None, lambda x: x())

    @value.setter
    def value(self, value: T | None) -> None:
        self._ref = Option(value).map(lambda x: weakref.ref(x))

    def __repr__(self) -> str:
        return f"WeakOption({repr(self.value)})"

    def __str__(self) -> str:
        return f"WeakOption({self.value})"
