"""
Posterior statistics and diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from deprecated.sphinx import deprecated

from liesel.goose.engine import ErrorLog, SamplingResults
from liesel.goose.pytree import slice_leaves, stack_leaves
from liesel.goose.types import Position
from liesel.option import Option


class ErrorSummaryForOneCode(NamedTuple):
    error_code: int
    error_msg: str
    count_per_chain: np.ndarray
    count_per_chain_posterior: None


ErrorSummary = dict[str, dict[int, ErrorSummaryForOneCode]]
"""
See docstring of ``_make_error_summary``.
"""


def _make_error_summary(
    error_log: ErrorLog,
    posterior_error_log: Option[ErrorLog],
) -> ErrorSummary:
    """
    Creates an error summary from the error log.

    The returned value looks like this::

        {
            kernel_identifier: {
                error_code: (error_code, error_msg, count, count_in_posterior),
                error_code: (error_code, error_msg, count, count_in_posterior),
                ...
            },
            ...
        }

    The ``error_msg`` is the empty string if the kernel class is not supplied in the
    ``error_log``.
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


class Summary:
    """
    A summary object.

    Offers two main use cases:

    1. View an overall summary by printing a summary instance, including a summary table
       of the posterior samples and a summary of sammpling errors.
    2. Programmatically access summary statistics via
       ``quantities[quantity_name][var_name]``. Please refer to the documentation of the
       attribute :attr:`.quantities` for details.

    Additionally, the summary object can be turned into a :class:`~pandas.DataFrame`
    using :meth:`.to_dataframe`.

    Parameters
    ----------
    results
        The sampling results to summarize.
    additional_chain
        can be supplied to add more parameters to the summary output. Must be a position
        chain which matches chain and time dimension of the posterior chain as returned
        by :meth:`~.goose.SamplingResults.get_posterior_samples`.
    hdi_prob
        Level on which to return posterior highest density intervals.
    selected, deselected
        Allow to get a summary only for a subset of the position keys.
    per_chain
        If *True*, the summary is calculated on a per-chain basis. Certain measures like
        ``rhat`` are not available if ``per_chain`` is *True*.

    Notes
    -----
    This class is still considered experimental. The API may still undergo larger
    changes.
    """

    per_chain: bool
    """
    Whether results are summarized for individual chains (*True*), or aggregated
    over chains (*False*).
    """
    quantities: dict[str, dict[str, np.ndarray]]
    """
    Dict of summarizing quantities.

    Built up in hierarchies as. Let ``summary`` be a :class:`.Summary` instance. The
    hierarchy is::

        q = summary.quantities["quantity_name"]["parameter_name"]

    The extracted object is an ``np.ndarray``. If ``per_chain=True``, the arrays for
    the ``"quantile"`` and ``"hdi"`` quantities have the following dimensions:

    1. First index refers to the chain
    2. Second index refers to the quantile/interval
    3. Third and subsequent indices refer to individual parameters.

    If ``per_chain=True``, the arrays for the other quantiles have the dimensions:

    1. First index refers to the chain
    2. Second and subsequent indices refer to individual parameters.

    If ``per_chain=False``, the first index is removed for all quantities.
    """
    config: dict
    """
    A dictionary of config settings for this summary object. Should NOT be changed
    after initialization; such changes have no effect on the computed summary values.
    """
    sample_info: dict
    """
    Dictionary of meta-information about the mcmc samples used to create this summary
    object.

    Contains ``num_chains``, ``sample_size_per_chain``, and ``warmup_size_per_chain``.
    """
    error_summary: ErrorSummary
    """
    Contains error information for each kernel.
    """
    kernels_by_pos_key: dict[str, str]
    """
    A dict, linking parameter names (the keys) to the kernel identifier (the values).
    The identifier refers to the kernel that was used to sample the respective
    parameter.
    """

    def __init__(
        self,
        results: SamplingResults,
        additional_chain: Position | None = None,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        hdi_prob: float = 0.9,
        selected: list[str] | None = None,
        deselected: list[str] | None = None,
        per_chain: bool = False,
    ):
        posterior_chain = results.get_posterior_samples()
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
        epochs = results.positions.get_epochs()

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

                qdict = _create_quantity_dict(single_chain, quantiles, hdi_prob)

                single_chain_summaries.append(qdict)

            quantities = stack_leaves(single_chain_summaries, axis=0)

        else:
            quantities = _create_quantity_dict(posterior_chain, quantiles, hdi_prob)

        config = {
            "quantiles": quantiles,
            "hdi_prob": hdi_prob,
            "chains_merged": not per_chain,
        }

        error_summary = _make_error_summary(
            results.get_error_log(False).unwrap(), results.get_error_log(True)
        )

        self.per_chain = per_chain
        self.quantities = quantities
        self.config = config
        self.sample_info = sample_info
        self.error_summary = error_summary
        self.kernels_by_pos_key = results.get_kernels_by_pos_key()

    def to_dataframe(self) -> pd.DataFrame:
        """Turns Summary object into a :class:`~pandas.DataFrame` object."""

        # don't change the original data
        quants = self.quantities.copy()

        # make new entries for the quantiles
        if self.per_chain:
            for i, q in enumerate(self.config["quantiles"]):
                quants[f"q_{q}"] = {
                    k: v[:, i, ...] for k, v in quants["quantile"].items()
                }

            quants["hdi_low"] = {k: v[:, 0, ...] for k, v in quants["hdi"].items()}
            quants["hdi_high"] = {k: v[:, 1, ...] for k, v in quants["hdi"].items()}
        else:
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
                quant_per_elem["kernel"] = self.kernels_by_pos_key.get(var, "-")
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

                # convert jax.Arrays (scalar) to floats so that pandas treats them
                # correctly
                for key, val in quant_per_elem.items():
                    if isinstance(val, jax.Array):
                        # value should be a scalar
                        assert val.shape == ()

                        # replace dict element with value casted to float32
                        quant_per_elem[key] = float(val)

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
        cols = (
            ["kernel", "mean", "sd"]
            + qtls
            + ["sample_size", "ess_bulk", "ess_tail", "rhat"]
        )
        cols = [col for col in cols if col in df.columns]
        df = df[cols]

        return df

    def error_df(self, per_chain: bool = False) -> pd.DataFrame:
        """
        Returns an overview of the errors recorded during sampling as a dataframe.
        """
        return self._error_df(per_chain=per_chain)

    def _error_df(self, per_chain: bool = False) -> pd.DataFrame:
        # fmt: off
        df = pd.concat({
            kernel: pd.DataFrame.from_dict(code_summary, orient="index")
            for kernel, code_summary in self.error_summary.items()
        })
        # fmt: on

        if df.empty:
            return df

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
        df["chain"] = df.groupby(level=[0, 1, 2, 3], observed=True).cumcount()
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

        txt = "Parameter summary:\n\n" + repr(param_df)

        if not error_df.empty:
            txt += "\n\nError summary:\n\n" + repr(error_df)

        return txt

    def _repr_html_(self):
        param_df = self._param_df()
        error_df = self._error_df()

        html = "\n<p><strong>Parameter summary:</strong></p>\n" + param_df.to_html()

        if not error_df.empty:
            html += "\n<p><strong>Error summary:</strong></p>\n" + error_df.to_html()

        html += "\n"
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

        md = "\n\n**Parameter summary:**\n\n" + param_md

        if not error_df.empty:
            md += "\n\n**Error summary:**\n\n" + error_md

        md += "\n\n"
        return md

    def __str__(self):
        return str(self.to_dataframe())

    @classmethod
    @deprecated(
        reason=(
            "Functionality moved directly to the __init__. Will be removed in v0.4.0."
        ),
        version="0.1.4",
    )
    def from_result(
        cls,
        result: SamplingResults,
        additional_chain: Position | None = None,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        hdi_prob: float = 0.9,
        selected: list[str] | None = None,
        deselected: list[str] | None = None,
        per_chain=False,
    ) -> Summary:
        """
        Alias for :meth:`.from_results` for backwards compatibility.

        In addition to the name, there are two further subtle differences to
        :meth:`.from_results`.

        - The argument ``result`` is in singular. The method :meth:`.from_results` uses
          the plural instead.
        - The argument ``result`` is of type :class:`.SamplingResult`, which itself is
          an alias for :meth:`.SamplingResults`.
        """

        return cls.from_results(
            results=result,
            additional_chain=additional_chain,
            quantiles=quantiles,
            hdi_prob=hdi_prob,
            selected=selected,
            deselected=deselected,
            per_chain=per_chain,
        )

    @staticmethod
    @deprecated(
        reason=(
            "Functionality moved directly to the __init__. Will be removed in v0.4.0."
        ),
        version="0.1.4",
    )
    def from_results(
        results: SamplingResults,
        additional_chain: Position | None = None,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        hdi_prob: float = 0.9,
        selected: list[str] | None = None,
        deselected: list[str] | None = None,
        per_chain=False,
    ) -> Summary:
        """
        Creates a :class:`.Summary` object from a results object.
        """

        return Summary(
            results=results,
            additional_chain=additional_chain,
            quantiles=quantiles,
            hdi_prob=hdi_prob,
            selected=selected,
            deselected=deselected,
            per_chain=per_chain,
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

    # hdi shape BEFORE
    # VarIDX --- HDI

    # special treatment for hdi since the function uses the last axis to refer
    # to the quantile
    for k, v in quantities["hdi"].items():
        quantities["hdi"][k] = np.moveaxis(v, -1, 0)

    # hdi shape AFTER
    # HDI --- VarIDX

    return quantities
