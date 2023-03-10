import arviz as az
from ..goose.engine import SamplingResults
from .. import __version__


def to_arviz_inference_data(results: SamplingResults) -> az.InferenceData:

    posterior_samples = results.get_posterior_samples()

    inference_data = az.from_dict(
        posterior=posterior_samples,
        # warmup_posterior=wps,
        posterior_attrs = {
            "inference_library": "liesel",
            "inference_library_version": __version__,
            "creation_library": "liesel",
            "creation_library_version": __version__,
        },
    )
    
    return inference_data





