import tensorflow_probability.substrates.numpy.distributions as nd
from tensorflow_probability.python.internal import reparameterization


class NoDistribution(nd.Distribution):
    def __init__(self):
        super().__init__(
            dtype=float,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True,
        )

    def _default_event_space_bijector(self):
        return None
