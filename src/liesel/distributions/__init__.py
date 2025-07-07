"""
Extra distributions for JAX-TFP.
"""

from .copulas import GaussianCopula
from .mvn_degen import MultivariateNormalDegenerate
from .mvn_log_cholesky_param import MultivariateNormalLogCholeskyParametrization

__all__ = ["GaussianCopula", "MultivariateNormalDegenerate", "MultivariateNormalLogCholeskyParametrization"]
