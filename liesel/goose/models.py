import warnings

from .interface import DataclassInterface, DictInterface, LogProbFunction


class DictModel(DictInterface):
    """
    Alias for :class:`.DictInterface`, provided for backwards compatibility.

    .. deprecated:: v0.2.6
        Use :class:`.DictInterface` instead. This alias will be removed in v0.4.0.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        super().__init__(log_prob_fn)

        warnings.warn(
            "Use gs.DictInterface instead. This alias will be removed in v0.4.0.",
            FutureWarning,
        )


class DataClassModel(DataclassInterface):
    """
    Alias for :class:`.DataclassInterface`, provided for backwards compatibility.

    .. deprecated:: v0.2.6
        Use :class:`.DataclassInterface` instead. This alias will be removed in v0.4.0.
    """

    def __init__(self, log_prob_fn: LogProbFunction):
        super().__init__(log_prob_fn)

        warnings.warn(
            "Use gs.DataclassInterface instead. This alias will be removed in v0.4.0.",
            FutureWarning,
        )
