import arviz as az

from .. import __version__
from ..goose.engine import SamplingResults


def to_arviz_inference_data(
    results: SamplingResults, include_warmup: bool = False
) -> az.InferenceData:
    """
    Converts goose's SamplingResults into InferenceData from arviz.

    Arviz' InferenceData seperates samples from the posterior and the warmup in
    the groups 'posterior' and 'warmup_posterior'. By default, all summaries and
    plots use only the data in the group 'posterior'.

    Parameters
    ----------
    results
        The sampling results.
    include_warmup
        Whether to include the warmup in the returned object.


    Returns
    -------
    The inference data.


    Notes
    -----
    The inference data has a variable for each position key included in the
    SamplingResult object. These are usually the position keys of the sampled
    parameters. Goose can track more values if specified in the field
    ``position_included``. This might be helpful to let arviz calculate
    information criteria like WAIC. Assuming that the position key
    ``loglik_pointwise`` corresponds to the point-wise evaluated log-likelihood
    then the inference data can be slightly changed so that arviz interprets
    these values as intended.

    .. code-block:: python

        # re-interpret the data
        llpw_extracted = idat.posterior['loglik_pointwise']
        idat.posterior = idat.posterior.drop('loglik_pointwise')
        idat.add_groups({'log_likelihood': {'observed': llpw_extracted}})
        az.waic(idat)

        # Computed from 300 posterior samples and 10 observations log-likelihood matrix.
        #              Estimate     SE
        # elpd_waic.     -17.43   3.40
        # p_waic           0.16      -

    """

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
