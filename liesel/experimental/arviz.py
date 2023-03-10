import arviz as az

from .. import __version__
from ..goose.engine import SamplingResults


def to_arviz_inference_data(
    results: SamplingResults, include_warmup: bool = False
) -> az.InferenceData:
    """Converts goose's SamplingResults into InferenceData from arviz"""

    posterior_samples = results.get_posterior_samples()

    warmup_posterior_samples = None
    if include_warmup:
        warmup_posterior_samples = results.positions.combine_filtered(
            lambda ec: ec.type.is_warmup(ec.type)
        ).expect("No warmup samples found.")

    inference_data = az.from_dict(
        posterior=posterior_samples,
        warmup_posterior=warmup_posterior_samples,
        posterior_attrs={
            "inference_library": "liesel",
            "inference_library_version": __version__,
            "creation_library": "liesel",
            "creation_library_version": __version__,
        },
        save_warmup=include_warmup,
    )

    return inference_data
