"""
Extra distributions for JAX-TFP.
"""

from .copulas import GaussianCopula
from .mvn_degen import MultivariateNormalDegenerate

__all__ = ["GaussianCopula", "MultivariateNormalDegenerate"]
