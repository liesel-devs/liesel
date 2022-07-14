"""
# Posterior statistics and diagnostics
"""

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import arviz as az
import jax.numpy as jnp
import jaxlib.xla_extension
import numpy as np
import pandas as pd
import xarray

from liesel.goose.engine import ErrorLog, SamplingResult
from liesel.goose.pytree import slice_leaves, stack_leaves
from liesel.goose.types import Position
from liesel.option import Option


def raise_chain_indices_error(
    chain_indices: Sequence[int], num_original_chains: int
) -> None:
    """Display informative error message with valid `chain_indices` inputs."""
    if any(
        chain_index not in range(num_original_chains) for chain_index in chain_indices
    ):
        raise ValueError(
            f"All chain indices must be between 0 and {num_original_chains-1} "
            "(bounds inclusive)."
        )


def validate_chain_indices(
    chain_indices: int | Sequence[int] | None,
    num_original_chains: int,
) -> Sequence[int]:
    """Convert `int` or `None` input of `chain_indices` to sequence of integers."""
    if chain_indices is None:
        return list(range(num_original_chains))

    if isinstance(chain_indices, int):
        chain_indices = [chain_indices]

    raise_chain_indices_error(chain_indices, num_original_chains)
    return chain_indices


def numpy_to_arviz(subparam_chains: np.ndarray) -> xarray.DataArray:
    """Convert data structure of `arviz` package to numpy array."""
    arviz_data = az.convert_to_inference_data(subparam_chains)
    return arviz_data["posterior"]["x"]


def add_rhat(arviz_array: xarray.DataArray, round_digits: int) -> np.ndarray:
    """Compute a single Rhat value for multiple chains of the same subparameter."""
    return np.round(az.rhat(arviz_array)["x"].values, decimals=round_digits)


def combine_chains(subparam_chains: np.ndarray) -> tuple[np.ndarray, xarray.DataArray]:
    """Concatenate all separate chains to a single chain for multi-chain sampling."""
    subparam_chains = subparam_chains.reshape(1, -1)
    arviz_array = numpy_to_arviz(subparam_chains)

    return subparam_chains, arviz_array


def add_num_effective(arviz_array: xarray.DataArray, round_digits: int) -> np.ndarray:
    """
    Compute the effective sample size of one specific subparameter (e.g. `beta_0`).
    """
    return np.array(
        [az.ess(arviz_array[[i]])["x"].values for i in range(arviz_array.shape[0])]
    ).round(decimals=round_digits)


def compute_quantiles(
    subparam_chains: np.ndarray, quantiles: Sequence[float], round_digits: int
) -> np.ndarray:
    """
    Compute posterior quantiles of one specific subparameter (e.g. `beta_0`).
    """
    # i'th row contains i'th input quantile for all chains
    # j'th column contains all input quantiles for j'th chain
    return np.quantile(subparam_chains, q=quantiles, axis=1).round(round_digits)


def add_quantiles(
    subparam_chains: np.ndarray,
    subparam_stats: dict[str, int | float | np.ndarray],
    quantiles: Sequence[float],
    round_digits: int,
) -> dict[str, int | float | np.ndarray]:
    """
    Add posterior quantiles to dictionary which collects all summary statistics and
    diagnostics for one specific subparameter (e.g. `beta_0`).
    """
    # number of rows equals length of quantiles input
    quantiles_array = compute_quantiles(subparam_chains, quantiles, round_digits)

    for ind, q in enumerate(quantiles):
        subparam_stats[f"q_{100 * q:.5g}"] = quantiles_array[ind]

    return subparam_stats


def compute_single_hdi(
    arviz_subarray: xarray.DataArray, hdi_prob: float, round_digits: int
) -> np.ndarray:
    """Compute Highest-Density Interval for a single chain."""
    # compute separate hdi for each chain to stay consistent with credible intervals
    # from `add_quantile()`
    return np.round(
        az.hdi(arviz_subarray, hdi_prob=hdi_prob)["x"].values, decimals=round_digits
    )


def compute_hdi(
    arviz_array: xarray.DataArray, hdi_prob: float, round_digits: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Highest-Density Intervals for all chains. Returns two numpy arrays of lower
    and upper bounds of Highest-Density Intervals.
    """
    # numpy array where i'th row contains lower and upper hdi bound of i'th chain
    hdi = np.vstack(
        [
            # subset with list to keep first dimension, arviz needs 2d arrays
            compute_single_hdi(arviz_array[[i]], hdi_prob, round_digits)
            for i in range(arviz_array.shape[0])
        ]
    )

    return hdi[:, 0], hdi[:, 1]


def add_hdi(
    arviz_array: xarray.DataArray,
    subparam_stats: dict[str, int | float | np.ndarray],
    hdi_prob: float,
    round_digits: int,
) -> dict[str, int | float | np.ndarray]:
    """
    Add Highest Density Intervals to dictionary which collects all summary statistics
    and diagnostics for one specific subparameter (e.g. `beta_0`).
    """
    hdi_lower, hdi_upper = compute_hdi(arviz_array, hdi_prob, round_digits)

    subparam_stats[f"hdi_{100 * hdi_prob:.5g}_low"] = hdi_lower
    subparam_stats[f"hdi_{100 * hdi_prob:.5g}_high"] = hdi_upper

    return subparam_stats


def subparam_stats_to_df(
    subparam_stats: dict[str, int | float | np.ndarray], per_chain: bool
) -> pd.DataFrame:
    """
    Convert dictionary which collects all summary statistics and diagnostics for one
    specific subparameter (e.g. `beta_0`) to a `pandas` data frame.
    """
    if not per_chain:
        # Must be dropped BEFORE conversion to data frame due to unequal lengths
        del subparam_stats["chain_index"]

    return pd.DataFrame(subparam_stats).drop(columns="num_chains")


def get_subparam_stats(
    subparam_chains: np.ndarray,
    per_chain: bool,
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
    param_index: int,
    as_dataframe: bool,
) -> dict[str, int | float | np.ndarray] | pd.DataFrame:
    """
    Collect all summary statistics and diagnostics for one specific subparameter (e.g.
    `beta_0`) into one dictionary or `pandas` data frame.
    """
    num_original_chains = subparam_chains.shape[0]
    chain_indices = validate_chain_indices(chain_indices, num_original_chains)

    subparam_chains = subparam_chains[chain_indices]
    num_filtered_chains = subparam_chains.shape[0]
    num_samples: int | list[int] = [subparam_chains.shape[1]] * num_filtered_chains

    arviz_array = numpy_to_arviz(subparam_chains)
    rhat = add_rhat(arviz_array, round_digits)

    if not per_chain:
        subparam_chains, arviz_array = combine_chains(subparam_chains)
        num_samples = sum(num_samples)  # type: ignore

    num_effective = add_num_effective(arviz_array, round_digits)
    posterior_means = subparam_chains.mean(axis=1).round(round_digits)
    posterior_sds = subparam_chains.std(axis=1).round(round_digits)

    subparam_stats = {
        "param_index": param_index,
        "num_chains": num_filtered_chains,
        "chain_index": chain_indices,
        "num_samples": num_samples,
        "num_effective": num_effective,
        "mean": posterior_means,
        "sd": posterior_sds,
        "rhat": rhat,
    }

    subparam_stats = add_quantiles(
        subparam_chains, subparam_stats, quantiles, round_digits
    )
    subparam_stats = add_hdi(arviz_array, subparam_stats, hdi_prob, round_digits)

    if not as_dataframe:
        return subparam_stats

    return subparam_stats_to_df(subparam_stats, per_chain)


def raise_dimension_error(param: str, num_dim: int) -> None:
    """Check for correct array dimensions of posterior samples."""
    if num_dim not in (2, 3):
        raise ValueError(
            f"Array of posterior samples for {param} has the wrong number of"
            f"dimensions.\nExpected 2 or 3, got {num_dim}."
        )


def adjust_dimensions(param_chains: np.ndarray, num_dim: int) -> np.ndarray:
    """
    Make shape of posterior samples for one dimensional parameters (e.g. `log_sigma`)
    consistent with multi-dimensional parameters.
    """
    if num_dim == 2:
        param_chains = np.expand_dims(param_chains, axis=-1)
    return param_chains


def raise_param_indices_error(
    param_indices: Sequence[int], num_original_subparams: int, param: str
) -> None:
    """
    Display informative error message with valid `param_indices` inputs for this
    specific `param`.
    """
    if any(
        param_index not in range(num_original_subparams)
        for param_index in param_indices
    ):
        raise ValueError(
            f"All param indices for {param} must be between "
            f"0 and {num_original_subparams-1} (bounds inclusive)."
        )


def validate_param_indices(
    param_indices: int | Sequence[int] | None, num_original_subparams: int, param: str
) -> Sequence[int]:
    """Convert `int` or `None` input of `param_indices` to sequence of integers."""
    if param_indices is None:
        return list(range(num_original_subparams))

    if isinstance(param_indices, int):
        param_indices = [param_indices]

    raise_param_indices_error(param_indices, num_original_subparams, param)

    return param_indices


def collect_subparam_dicts(
    param_chains: np.ndarray,
    per_chain: bool,
    param_indices: Sequence[int],
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
) -> list[dict]:
    """
    Combine dictionaries with summary statistics and diagnostics within one parameter
    vector (e.g. `beta_0` and `beta_1`) into a list.
    """
    return [
        get_subparam_stats(
            param_chains[..., i],
            per_chain,
            chain_indices,
            quantiles,
            hdi_prob,
            round_digits,
            param_index=i,
            as_dataframe=False,
        )
        for i in param_indices
    ]


def move_col_first(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Move last column of a `pandas` data frame to the first column."""
    return df[[colname] + [col for col in df.columns if col != colname]]


def collect_subparam_dfs(
    param_chains: np.ndarray,
    per_chain: bool,
    param_indices: Sequence[int],
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
) -> pd.DataFrame:
    """
    Combine data frames with summary statistics and diagnostics within one parameter
    vector (e.g. `beta_0` and `beta_1`) into a single `pandas` data frame.
    """
    param_df = pd.concat(
        [
            get_subparam_stats(
                param_chains[..., i],
                per_chain,
                chain_indices,
                quantiles,
                hdi_prob,
                round_digits,
                param_index=i,
                as_dataframe=True,
            )
            for i in param_indices
        ]
    ).reset_index(drop=True)

    return move_col_first(param_df, "param_index")


def get_param_stats(
    param: str,
    posterior_samples: dict[str, jnp.DeviceArray],
    per_chain: bool,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
    as_dataframe: bool,
) -> list[dict] | pd.DataFrame:
    """
    Collect all summary statistics and diagnostics for one specific parameter vector
    (e.g. `beta`) into one dictionary or `pandas` data frame.
    """
    # arviz package requires numpy arrays instead of jax numpy arrays
    param_chains = np.array(posterior_samples[param])
    num_dim = param_chains.ndim

    # check that each param_chain has either two (scalar) or three (vector) dimension
    raise_dimension_error(param, num_dim)

    # add last dimension if original parameter is a scalar instead of a vector
    param_chains = adjust_dimensions(param_chains, num_dim)

    num_original_subparams = param_chains.shape[-1]
    param_indices = validate_param_indices(param_indices, num_original_subparams, param)

    if not as_dataframe:
        return collect_subparam_dicts(
            param_chains,
            per_chain,
            param_indices,
            chain_indices,
            quantiles,
            hdi_prob,
            round_digits,
        )

    return collect_subparam_dfs(
        param_chains,
        per_chain,
        param_indices,
        chain_indices,
        quantiles,
        hdi_prob,
        round_digits,
    )


def validate_params(
    posterior_samples: dict[str, jnp.DeviceArray], params: str | list[str] | None
) -> list[str]:
    """Convert `str` or `None` input of `params` to sequence of strings."""
    posterior_keys = list(posterior_samples.keys())
    if params is None:
        return posterior_keys

    if isinstance(params, str):
        params = [params]

    if any(param not in posterior_keys for param in params):
        raise KeyError(f"All params must be in {posterior_keys}.")

    return params


def collect_param_dicts(
    posterior_samples: dict[str, jnp.DeviceArray],
    per_chain: bool,
    params: list[str],
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
) -> dict[str, list[dict]]:
    """
    Combine dictionaries with summary statistics and diagnostics of all model parameters
    into a single dictionary.
    """
    return {
        param: get_param_stats(
            param,
            posterior_samples,
            per_chain,
            param_indices,
            chain_indices,
            quantiles,
            hdi_prob,
            round_digits,
            as_dataframe=False,
        )
        for param in params
    }


def collect_param_dfs(
    posterior_samples: dict[str, jnp.DeviceArray],
    per_chain: bool,
    params: list[str],
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    quantiles: Sequence[float],
    hdi_prob: float,
    round_digits: int,
) -> pd.DataFrame:
    """
    Combine data frames with summary statistics and diagnostics of all model parameters
    into a single `pandas` data frame.
    """
    param_dfs: list[pd.DataFrame] = []
    for param in params:
        param_df: pd.DataFrame = get_param_stats(
            param,
            posterior_samples,
            per_chain,
            param_indices,
            chain_indices,
            quantiles,
            hdi_prob,
            round_digits,
            as_dataframe=True,
        )
        param_df = param_df.assign(param=param).set_index("param")

        param_df.index.name = None
        param_dfs.append(param_df)

    return pd.concat(param_dfs)


def summary(
    results: SamplingResult,
    per_chain: bool = True,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    hdi_prob: float = 0.9,
    round_digits: int = 3,
    as_dataframe: bool = True,
) -> dict[str, list[dict]] | pd.DataFrame:
    """
    Compute summary statistics and diagnostic measures of posterior samples.

    ## Parameters

    - `results`:
      Result object of the sampling process. Must have a method
      `get_posterior_samples()` which extracts all samples from the posterior
      distribution.

    - `per_chain`:
      If `True`, all statistics and diagnostics (except the Rhat value) are computed for
      each chain separately. If `False`, one metric is computed for each subparameter
      after concatenating all chains.

    - `params`:
      Names of the model parameters that are contained in the summary output. Must
      coincide with the dictionary keys of the `Position` with the posterior samples.
      If `None`, all parameters are included.

    - `param_indices`:
      Indices of each model parameter that are contained in the summary output. Selects
      e.g. `beta_0` out of a `beta` parameter vector.A single index can be specified as
      an integer or a sequence containing one integer. If `None`, all subparameters are
      included.

    - `chain_indices`:
      Indices of chains for each model subparameter that are contained in the summary
      output. Selects e.g. chain 0 and chain 2 out of multiple chains. A single index
      can be specified as an integer or a sequence containing one integer. If `None`,
      all chains are included.

    - `quantiles`:
      Quantiles of the posterior distribution that are contained in the summary output.

    - `hdi_prob`:
      Coverage level of the Highest Density Interval of the posterior distribution.
      Summary output contains lower and upper bound of this interval.

    - `round_digits`:
      Number of decimals for each float value within the summary output.

    - `as_dataframe`:
      If `True`, all statistics and diagnostics are embedded into a `pandas` data frame.
      If `False`, a dictionary with the same keys as `Position` with the posterior
      samples is returned.

    ## Returns

    Dictionary if `as_dataframe` is False, `pandas.DataFrame` if `as_dataframe` is
    True.
    """
    # NOTE: Docstring Duplications
    # The entries `results`, `params`, `param_indices` and `chain_indices` are shared
    # with all user plotting functions.
    posterior_samples = results.get_posterior_samples()
    params = validate_params(posterior_samples, params)

    if not as_dataframe:
        return collect_param_dicts(
            posterior_samples,
            per_chain,
            params,
            param_indices,
            chain_indices,
            quantiles,
            hdi_prob,
            round_digits,
        )

    return collect_param_dfs(
        posterior_samples,
        per_chain,
        params,
        param_indices,
        chain_indices,
        quantiles,
        hdi_prob,
        round_digits,
    )


class ErrorSummaryForOneCode(NamedTuple):
    error_code: int
    error_msg: str
    count_per_chain: np.ndarray
    count_per_chain_posterior: None


ErrorSummary = dict[str, dict[int, ErrorSummaryForOneCode]]
"""
See docstring of `_make_error_summary`.
"""


def _make_error_summary(
    error_log: ErrorLog,
    posterior_error_log: Option[ErrorLog],
) -> ErrorSummary:
    """
    Creates an error summary from the error log.

    The returned value looks like this:

    ```
    {
        kernel_identifier: {
            error_code: (error_code, error_msg, count, count_in_posterior),
            error_code: (error_code, error_msg, count, count_in_posterior),
            ...
        },
        ...
    }
    ```

    The `error_msg` is the empty string if the kernel class is not supplied in the
    `error_log`.
    """
    error_summary = {}
    for kel in error_log.values():
        counter_dict: dict[int, np.ndarray] = {}

        # calculate the overall counts
        ec_unique = np.unique(kel.error_codes)
        for ec in ec_unique:
            if ec == 0:
                continue
            occurences_per_chain = np.sum(kel.error_codes == ec, axis=1)
            counter_dict[ec] = occurences_per_chain

        krnl_summary: dict[int, ErrorSummaryForOneCode] = {}
        for key, count in counter_dict.items():
            ec = key
            # type ignore is ok since the type must implement the kernel protocol.
            error_msg = kel.kernel_cls.map_or(
                "", lambda krn_cls: krn_cls.error_book[ec]  # type: ignore
            )
            krnl_summary[ec] = ErrorSummaryForOneCode(ec, error_msg, count, None)

        # calculate the counts in the posterior
        if posterior_error_log.is_some():
            posterior_error_log_unwrapped = posterior_error_log.unwrap()
            kel_post = posterior_error_log_unwrapped[kel.kernel_ident]
            for ec in ec_unique:
                if ec == 0:
                    continue
                occurences_per_chain = np.sum(kel_post.error_codes == ec, axis=1)
                krnl_summary[ec] = krnl_summary[ec]._replace(
                    count_per_chain_posterior=occurences_per_chain
                )

        error_summary[kel.kernel_ident] = krnl_summary

    return error_summary


@dataclass
class Summary:
    """
    A summary object.

    Allows easy programmatic access via `quantities[quantity_name][var_name]`.
    The array has a similar shape as the parameter with `var_name`. However,
    if `per_chain` is `True`, the first dimension refers to the chain index.
    Additionally, for `hdi` and `quantile` the second dimension refers to the
    quantile.

    The summary object can be turned into a `pd.DataFrame` using the function
    `to_dataframe`.

    Experimental.
    """

    quantities: dict[str, dict[str, np.ndarray]]
    config: dict
    sample_info: dict
    error_summary: ErrorSummary

    def to_dataframe(self) -> pd.DataFrame:
        """Turns Summary object into a DataFrame object."""

        # don't change the original data
        quants = self.quantities.copy()

        # make new entries for the quantiles
        for i, q in enumerate(self.config["quantiles"]):
            quants[f"q_{q}"] = {k: v[i, ...] for k, v in quants["quantile"].items()}
        quants["hdi_low"] = {k: v[0, ...] for k, v in quants["hdi"].items()}
        quants["hdi_high"] = {k: v[1, ...] for k, v in quants["hdi"].items()}

        # remove the old entries
        del quants["quantile"]
        del quants["hdi"]

        # create one row per entry
        df_dict = {}
        for var in quants["mean"].keys():
            it = np.nditer(quants["mean"][var], flags=["multi_index"])
            for _ in it:
                var_fqn = (
                    var if len(it.multi_index) == 0 else f"{var}{list(it.multi_index)}"
                )
                quant_per_elem: dict[str, Any] = {}
                quant_per_elem["variable"] = var
                if self.config["chains_merged"]:
                    quant_per_elem["var_index"] = it.multi_index
                    quant_per_elem["sample_size"] = (
                        self.sample_info["sample_size_per_chain"]
                        * self.sample_info["num_chains"]
                    )
                else:
                    quant_per_elem["chain_index"] = it.multi_index[0]
                    quant_per_elem["var_index"] = it.multi_index[1:]
                    quant_per_elem["sample_size"] = self.sample_info[
                        "sample_size_per_chain"
                    ]

                for quant_name, quant_dict in quants.items():
                    quant_per_elem[quant_name] = quant_dict[var][it.multi_index]

                # convert DeviceArrays (scalar) to floats so that
                # pandas treats them correctly

                for key, val in quant_per_elem.items():
                    if type(val) == jaxlib.xla_extension.DeviceArray:
                        # make mypy happy
                        val = typing.cast(jnp.ndarray, val)
                        # value should be a scalar
                        assert val.shape == ()

                        # convert to float32
                        val = np.atleast_1d(np.asarray(val))[0]

                        quant_per_elem[key] = val

                df_dict[var_fqn] = quant_per_elem

        # convert to dataframe and use varname as index
        df = pd.DataFrame.from_dict(df_dict, orient="index")
        df = df.reset_index()
        df = df.rename(columns={"index": "var_fqn"})
        df = df.set_index("variable")

        return df

    def _param_df(self):
        df = self.to_dataframe()

        df.index.name = "parameter"
        df = df.rename(columns={"var_index": "index"})
        df = df.set_index("index", append=True)

        qtls = [f"q_{qtl}" for qtl in self.config["quantiles"]]
        cols = ["mean", "sd"] + qtls + ["sample_size", "ess_bulk", "ess_tail", "rhat"]
        df = df[cols]

        return df

    def _error_df(self, per_chain=False):
        # fmt: off
        df = pd.concat({
            kernel: pd.DataFrame.from_dict(code_summary, orient="index")
            for kernel, code_summary in self.error_summary.items()
        })
        # fmt: on

        df = df.reset_index(level=1, drop=True)
        df["error_code"] = df["error_code"].astype(int)
        df = df.set_index(["error_code", "error_msg"], append=True)
        df.index.names = ["kernel", "error_code", "error_msg"]

        # fmt: off
        df = df.rename(columns={
            "count_per_chain": "total",
            "count_per_chain_posterior": "posterior",
        })
        # fmt: on

        df = df.explode(["total", "posterior"])
        df["warmup"] = df["total"] - df["posterior"]
        df = df.drop(columns="total")

        df = df.melt(
            value_vars=["warmup", "posterior"],
            var_name="phase",
            value_name="count",
            ignore_index=False,
        )

        df["phase"] = pd.Categorical(df["phase"], categories=["warmup", "posterior"])

        df = df.set_index("phase", append=True)
        df["chain"] = df.groupby(level=[0, 1, 2, 3]).cumcount()
        df = df.set_index("chain", append=True)
        df = df.sort_index()

        df["sample_size"] = None
        warmup_size = self.sample_info["warmup_size_per_chain"]
        posterior_size = self.sample_info["sample_size_per_chain"]
        df.loc[pd.IndexSlice[:, :, :, "warmup"], "sample_size"] = warmup_size
        df.loc[pd.IndexSlice[:, :, :, "posterior"], "sample_size"] = posterior_size
        df["relative"] = df["count"] / df["sample_size"]
        df = df.drop(columns="sample_size")

        if not per_chain:
            df = df.groupby(level=[0, 1, 2, 3], observed=True)
            df = df.aggregate({"count": "sum", "relative": "mean"})
            df = df.sort_index()

        return df

    def __repr__(self):
        param_df = self._param_df()
        error_df = self._error_df()

        txt = (
            "Parameter summary:\n\n"
            + repr(param_df)
            + "\n\nError summary:\n\n"
            + repr(error_df)
        )

        return txt

    def _repr_html_(self):
        param_df = self._param_df()
        error_df = self._error_df()

        html = (
            "\n<p><strong>Parameter summary:</strong></p>\n"
            + param_df.to_html()
            + "\n<p><strong>Error summary:</strong></p>\n"
            + error_df.to_html()
            + "\n"
        )

        return html

    def _repr_markdown_(self):
        param_df = self._param_df()
        error_df = self._error_df()

        try:
            param_md = param_df.to_markdown()
            error_md = error_df.to_markdown()
        except ImportError:
            param_md = f"```\n{repr(param_df)}\n```"
            error_md = f"```\n{repr(error_df)}\n```"

        md = (
            "\n\n**Parameter summary:**\n\n"
            + param_md
            + "\n\n**Error summary:**\n\n"
            + error_md
            + "\n\n"
        )

        return md

    def __str__(self):
        return str(self.to_dataframe())

    @staticmethod
    def from_result(
        result: SamplingResult,
        additional_chain: Position | None = None,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        hdi_prob: float = 0.9,
        selected: list[str] | None = None,
        deselected: list[str] | None = None,
        per_chain=False,
    ) -> "Summary":
        """
        Creates a `Summary` object from a result object.

        An optional `additional_chain` can be supplied to add more parameters to
        the summary output. `additional_chain` must be a position chain which
        matches chain and time dimension of the posterior chain as returned by
        `result.get_posterior_samples()`.

        The arguments `selected` and `deselected` allow to get a summary only
        for a subset of the position keys.

        When using `per_chain`, the summary is calculated on a per-chain basis.
        Certain measures like `rhat` are not available if `per_chain` is true.

        Experimental.
        """

        posterior_chain = result.get_posterior_samples()
        if additional_chain:
            for k, v in additional_chain.items():
                posterior_chain[k] = v

        if selected:
            posterior_chain = Position(
                {
                    key: value
                    for key, value in posterior_chain.items()
                    if key in selected
                }
            )

        if deselected is not None:
            for key in deselected:
                del posterior_chain[key]

        # get some general infos on the sampling
        param_chain = next(iter(posterior_chain.values()))
        epochs = result.positions.get_epochs()

        warmup_size = np.sum(
            [epoch.duration for epoch in epochs if epoch.type.is_warmup(epoch.type)]
        )

        sample_info = {
            "num_chains": param_chain.shape[0],
            "sample_size_per_chain": param_chain.shape[1],
            "warmup_size_per_chain": warmup_size,
        }

        # convert everything to numpy array
        for key in posterior_chain:
            posterior_chain[key] = np.asarray(posterior_chain[key])

        # calculate quantiles either per chain and merge the results or all at once
        single_chain_summaries = []
        if per_chain:
            for chain_idx in range(sample_info["num_chains"]):
                single_chain = slice_leaves(
                    posterior_chain, jnp.s_[None, chain_idx, ...]
                )
                single_chain_summaries.append(
                    _create_quantity_dict(single_chain, quantiles, hdi_prob)
                )
            quantities = stack_leaves(single_chain_summaries, axis=0)
        else:
            quantities = _create_quantity_dict(posterior_chain, quantiles, hdi_prob)

        config = {
            "quantiles": quantiles,
            "hdi_prob": hdi_prob,
            "chains_merged": not per_chain,
        }

        error_summary = _make_error_summary(
            result.get_error_log(False).unwrap(), result.get_error_log(True)
        )

        return Summary(
            quantities=quantities,
            config=config,
            sample_info=sample_info,
            error_summary=error_summary,
        )


def _create_quantity_dict(
    chain: Position, quantiles: Sequence[float], hdi_prob: float
) -> dict[str, dict[str, np.ndarray]]:
    azchain = az.convert_to_inference_data(chain).posterior

    # calculate quantities
    mean = azchain.mean(dim=["chain", "draw"])
    var = azchain.var(dim=["chain", "draw"])
    sd = azchain.std(dim=["chain", "draw"])
    quantile = azchain.quantile(q=quantiles, dim=["chain", "draw"])
    hdi = az.hdi(azchain, hdi_prob=hdi_prob)

    ess_bulk = az.ess(azchain, method="bulk")
    ess_tail = az.ess(azchain, method="tail")
    mcse_mean = az.mcse(azchain, method="mean")
    mcse_sd = az.mcse(azchain, method="sd")

    # place quantities in dict
    quantities = {
        "mean": mean,
        "var": var,
        "sd": sd,
        "quantile": quantile,
        "hdi": hdi,
        "rhat": None,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "mcse_mean": mcse_mean,
        "mcse_sd": mcse_sd,
    }

    if azchain.chain.size > 1:
        quantities["rhat"] = az.rhat(azchain)
    else:
        del quantities["rhat"]

    # convert to simple dict[str, np.ndarray]
    for key, val in quantities.items():
        quantities[key] = {k: v.values for k, v in val.data_vars.items()}

    # special treatment for hdi since the function uses the last axis to refer
    # to the quantile
    for k, v in quantities["hdi"].items():
        quantities["hdi"][k] = np.moveaxis(v, -1, 0)

    return quantities
