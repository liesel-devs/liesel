"""Types shared by Liesel's modeling and sampling interfaces."""

from typing import TYPE_CHECKING, Any, NewType

if TYPE_CHECKING:
    Position = dict[str, Any]
else:
    # Accept plain dictionaries statically while preserving Position(d) is d.
    Position = NewType("Position", dict[str, Any])
