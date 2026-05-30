import arviz as az
from xarray import DataTree

from .. import __version__
from ..goose.engine import SamplingResults


def to_arviz_inference_data(
    results: SamplingResults, include_warmup: bool = False
) -> DataTree:
    """
    Converts goose's SamplingResults into an ArviZ ``DataTree``.

    Posterior and warmup samples are stored in the groups ``"posterior"`` and
    ``"warmup_posterior"``. By default, all summaries and plots use only the data
    in the group ``"posterior"``.

    Parameters
    ----------
    results
        The sampling results.
    include_warmup
        Whether to include the warmup in the returned object.


    Returns
    -------
    The ArviZ ``DataTree``.


    Notes
    -----
    The returned object has a variable for each position key included in the
    SamplingResult object. These are usually the position keys of the sampled
    parameters. Goose can track more values if specified in the field
    ``position_included``. This might be helpful to let arviz calculate
    information criteria like WAIC. Assuming that the position key
    ``loglik_pointwise`` corresponds to the point-wise evaluated log-likelihood,
    move it into an ArviZ ``"log_likelihood"`` group before computing
    information criteria such as WAIC.

    """

    posterior_samples = results.get_posterior_samples()

    warmup_posterior_samples = None
    if include_warmup:
        warmup_posterior_samples = results.positions.combine_filtered(
            lambda ec: ec.type.is_warmup(ec.type)
        ).expect("No warmup samples found.")

    inference_data = az.from_dict(
        {
            "posterior": posterior_samples,
            "warmup_posterior": warmup_posterior_samples,
        },
        attrs={
            "posterior": {
                "inference_library": "liesel",
                "inference_library_version": __version__,
                "creation_library": "liesel",
                "creation_library_version": __version__,
            }
        },
        save_warmup=include_warmup,
    )

    return inference_data
