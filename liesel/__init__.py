from .__version__ import __version__, __version_info__  # isort: skip  # noqa: F401

from . import goose, liesel, tfp
from .logging import setup_logger

# Because logger setup takes place after importing the submodules, it only affects
# log messages emitted at runtime.
setup_logger()

__all__ = ["goose", "liesel", "tfp"]
