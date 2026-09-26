"""Types shared by Liesel's modeling and sampling interfaces."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NewType

type PositionInput = Mapping[str, Any]
"""Read-only input of variable or node names and their values, including pytrees."""

if TYPE_CHECKING:
    Position = dict[str, Any]
else:
    # Accept plain dictionaries statically while preserving Position(d) is d.
    Position = NewType("Position", dict[str, Any])
