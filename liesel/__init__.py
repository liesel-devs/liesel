"""
# Probabilistic Programming Framework

Welcome to the API documentation of Liesel, a probabilistic programming framework with
a focus on semi-parametric regression. It includes:

- [**Liesel**][1], a library to express statistical models as Probabilistic Graphical
  Models (PGMs). Through the PGM representation, the user can build and update models
  in a natural way.
- **Goose**, a library to build custom MCMC algorithms with several parameter blocks
  and MCMC kernels such as the No U-Turn Sampler (NUTS), the Iteratively Weighted Least
  Squares (IWLS) sampler, or different Gibbs samplers. Goose also takes care of the
  MCMC bookkeeping and the chain post-processing.
- [**RLiesel**][2], an R interface for Liesel which assists the user with the
  configuration of semi-parametric regression models such as Generalized Additive
  Models for Location, Scale and Shape (GAMLSS) with different response distributions,
  spline-based smooth terms and shrinkage priors.

The name "Liesel" is an homage to the [Gänseliesel fountain][3], landmark of Liesel's
birth city [Göttingen][4].

## Installation

For installation instructions, see the [README][5] in the main repository.

## Further reading

For a scientific discussion of the software, see our paper on arXiv (in preparation).
If you are looking for code examples, the [tutorial book][6] might come in handy.

[1]: https://github.com/liesel-devs/liesel
[2]: https://github.com/liesel-devs/rliesel
[3]: https://en.wikipedia.org/wiki/G%C3%A4nseliesel
[4]: https://en.wikipedia.org/wiki/G%C3%B6ttingen
[5]: https://github.com/liesel-devs/liesel#installation
[6]: https://liesel-devs.github.io/liesel-tutorials
"""

from .__version__ import __version__, __version_info__  # isort: skip  # noqa: F401

from . import goose, liesel, tfp
from .logging import setup_logger

# because logger setup takes place after importing the submodules, it only affects
# log messages emitted at runtime
setup_logger()

__all__ = ["goose", "liesel", "tfp"]
