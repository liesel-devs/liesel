"""Types shared by Liesel's modeling and sampling interfaces."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NewType

PyTree = Any
"""An arbitrary JAX pytree, including registered custom containers and leaves."""

type PositionInput = Mapping[str, PyTree]
"""Read-only input of variable or node names and their values, including pytrees."""

if TYPE_CHECKING:
    Position = dict[str, PyTree]
else:
    # Accept plain dictionaries statically while preserving Position(d) is d.
    Position = NewType("Position", dict[str, PyTree])
