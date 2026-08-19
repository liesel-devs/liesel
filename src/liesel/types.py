"""Types shared by Liesel's modeling and sampling interfaces."""

from typing import Any, NewType

Position = NewType("Position", dict[str, Any])
