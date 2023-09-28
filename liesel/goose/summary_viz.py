"""
Diagnostic plots of the posterior samples.
"""

from collections.abc import Sequence
from typing import Any

import arviz
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from liesel.goose.engine import SamplingResults


def _raise_chain_indices_error(
    chain_indices: Sequence[int], num_original_chains: int
) -> None:
    """Display informative error message with valid ``chain_indices`` inputs."""
    if any(
        chain_index not in range(num_original_chains) for chain_index in chain_indices
    ):
        raise ValueError(
            f"All chain indices must be between 0 and {num_original_chains-1} "
            "(bounds inclusive)."
        )


def _validate_chain_indices(
    chain_indices: int | Sequence[int] | None,
    num_original_chains: int,
) -> Sequence[int]:
    """
    Convert ``int`` or ``None`` input of ``chain_indices`` to sequence of integers.
    """
    if chain_indices is None:
        return list(range(num_original_chains))

    if isinstance(chain_indices, int):
        chain_indices = [chain_indices]

    _raise_chain_indices_error(chain_indices, num_original_chains)
    return chain_indices


def _raise_dimension_error(param: str, num_dim: int) -> None:
    """Check for correct array dimensions of posterior samples."""
    if num_dim not in (2, 3):
        raise ValueError(
            f"Array of posterior samples for {param} has the wrong number of"
            f"dimensions.\nExpected 2 or 3, got {num_dim}."
        )


def _adjust_dimensions(param_chains: np.ndarray, num_dim: int) -> np.ndarray:
    """
    Make shape of posterior samples for one dimensional parameters (e.g. ``log_sigma``)
    consistent with multi-dimensional parameters.
    """
    if num_dim == 2:
        param_chains = np.expand_dims(param_chains, axis=-1)
    return param_chains


def _raise_param_indices_error(
    param_indices: Sequence[int], num_original_subparams: int, param: str
) -> None:
    """
    Display informative error message with valid ``param_indices`` inputs for this
    specific ``param``.
    """
    if any(
        param_index not in range(num_original_subparams)
        for param_index in param_indices
    ):
        raise ValueError(
            f"All param indices for {param} must be between "
            f"0 and {num_original_subparams-1} (bounds inclusive)."
        )


def _validate_param_indices(
    param_indices: int | Sequence[int] | None, num_original_subparams: int, param: str
) -> Sequence[int]:
    """
    Convert ``int`` or ``None`` input of ``param_indices`` to sequence of integers.
    """
    if param_indices is None:
        return list(range(num_original_subparams))

    if isinstance(param_indices, int):
        param_indices = [param_indices]

    _raise_param_indices_error(param_indices, num_original_subparams, param)

    return param_indices


def _move_col_first(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Move last column of a :class:`~pandas.DataFrame` to the first column."""
    return df[[colname] + [col for col in df.columns if col != colname]]


def _validate_params(
    posterior_samples: dict[str, jax.Array], params: str | list[str] | None
) -> list[str]:
    """Convert ``str`` or ``None`` input of ``params`` to sequence of strings."""
    posterior_keys = list(posterior_samples.keys())
    if params is None:
        return posterior_keys

    if isinstance(params, str):
        params = [params]

    if any(param not in posterior_keys for param in params):
        raise KeyError(f"All params must be in {posterior_keys}.")

    return params


def _subparam_chains_to_df(
    subparam_chains: np.ndarray, param_index: int
) -> pd.DataFrame:
    """
    Convert array of posterior samples for a single subparameter (e.g. ``beta[0]``) to a
    pandas data frame.
    """

    subparam_df = (
        pd.DataFrame(subparam_chains)
        .melt(ignore_index=False)
        .rename(columns={"variable": "iteration"})
        .reset_index()
        .rename(columns={"index": "chain_index"})
        .sort_values(by=["chain_index", "iteration"], ignore_index=True)
        .assign(param_index=param_index)
    )

    return _move_col_first(subparam_df, colname="param_index")


def _preprocess_param_chains(
    posterior_samples: dict[str, jax.Array], param: str
) -> np.ndarray:
    """Convert array of posteror samples for each parameter to equal dimensions."""

    param_chains = np.array(posterior_samples[param])
    num_dim = param_chains.ndim

    _raise_dimension_error(param, num_dim)
    param_chains = _adjust_dimensions(param_chains, num_dim)
    return param_chains


def _convert_to_sequence(indices: int | Sequence[int]) -> Sequence[int]:
    """Convert integer parameter and chain indices to list or tuple."""

    if isinstance(indices, int):
        indices = [indices]

    return indices


def _filter_param_df(
    param_df: pd.DataFrame,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    max_chains: int | None,
) -> pd.DataFrame:
    """
    Filters the plotting data to contain only the specified parameter and chain indices.
    Output data contains not more than `max_chains` chain indices.
    """

    if chain_indices is not None:
        chain_indices = _convert_to_sequence(chain_indices)
        param_df = param_df.loc[param_df["chain_index"].isin(chain_indices)]

    if param_indices is not None:
        param_indices = _convert_to_sequence(param_indices)
        param_df = param_df.loc[param_df["param_index"].isin(param_indices)]

    if max_chains is not None and param_df["chain_index"].nunique() > max_chains:
        last_chain_index = sorted(param_df["chain_index"].unique())[max_chains - 1]
        param_df = param_df.loc[param_df["chain_index"] <= last_chain_index]

    return param_df


def _postprocess_param_df(
    param_df: pd.DataFrame,
    param: str,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    max_chains: int | None,
) -> pd.DataFrame:
    """
    Combines parameter and parameter index column and filters chain and parameter
    indices.
    """

    num_original_chains = param_df["chain_index"].nunique()
    num_original_subparams = param_df["param_index"].nunique()

    if num_original_subparams > 1:
        name = param_df["param"].astype(str)
        index = param_df["param_index"].astype(str)
        param_df.loc[:, "param_label"] = name + "[" + index + "]"
    else:
        name = param_df["param"].astype(str)
        param_df["param_label"] = name

    chain_indices = _validate_chain_indices(chain_indices, num_original_chains)
    param_indices = _validate_param_indices(
        param_indices, num_original_subparams, param
    )
    param_df = _filter_param_df(param_df, param_indices, chain_indices, max_chains)

    return param_df


def _collect_subparam_dfs(
    posterior_samples: dict[str, jax.Array],
    param: str,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    max_chains: int | None,
) -> pd.DataFrame:
    """
    Combines individual data frames for each subparameter into a single data frame for
    each parameter.
    """

    param_chains = _preprocess_param_chains(posterior_samples, param)

    param_df = (
        pd.concat(
            [
                _subparam_chains_to_df(param_chains[..., param_index], param_index)
                for param_index in range(param_chains.shape[-1])
            ]
        )
        .assign(param=param)
        .reset_index(drop=True)
    )

    param_df = _move_col_first(param_df, colname="param")

    return _postprocess_param_df(
        param_df, param, param_indices, chain_indices, max_chains
    )


def _collect_param_dfs(
    results: SamplingResults,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    include_warmup: bool = False,
) -> pd.DataFrame:
    """Combines individual data frames for each parameter into a single data frame."""

    if include_warmup:
        samples = results.get_samples()
    else:
        samples = results.get_posterior_samples()
    params = _validate_params(samples, params)

    return pd.concat(
        [
            _collect_subparam_dfs(
                samples, param, param_indices, chain_indices, max_chains
            )
            for param in params
        ]
    ).reset_index(drop=True)


def _setup_plot_df(
    results: SamplingResults,
    params: str | list[str] | None,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    max_chains: int | None,
    include_warmup: bool = False,
) -> pd.DataFrame:
    """Provides data input for all plotting functions."""

    return _collect_param_dfs(
        results,
        params,
        param_indices,
        chain_indices,
        max_chains,
        include_warmup,
    ).astype({"chain_index": "category"})


def _setup_scatterplot_df(
    results: SamplingResults,
    params: str | list[str] | None,
    param_indices: int | Sequence[int] | None,
    chain_indices: int | Sequence[int] | None,
    max_chains: int | None,
    include_warmup: bool = False,
) -> pd.DataFrame:
    """
    Provides bespoke data input for plot_scatter. If two indices *and* two params are
    specified, the first index refers the first param and the second index to the
    second.
    """
    if (
        isinstance(params, str)
        or params is None
        or isinstance(param_indices, int)
        or param_indices is None
    ):
        return _setup_plot_df(
            results, params, param_indices, chain_indices, max_chains, include_warmup
        )
    if not isinstance(params, str) and len(params) > 1:
        param0_df = _setup_plot_df(
            results,
            params[0],
            param_indices[0],
            chain_indices,
            max_chains,
            include_warmup,
        )
        param1_df = _setup_plot_df(
            results,
            params[1],
            param_indices[1],
            chain_indices,
            max_chains,
            include_warmup,
        )
        plot_df = pd.concat([param0_df, param1_df], ignore_index=True)
    else:
        plot_df = _setup_plot_df(
            results, params, param_indices, chain_indices, max_chains, include_warmup
        )
    return plot_df


def _set_plot_cols(plot_df: pd.DataFrame, ncol: int) -> int:
    """Determines number of facets within each row of the grid."""

    num_subparams = plot_df["param_label"].nunique()
    return min(ncol, num_subparams)


def _set_aesthetics(
    g: sns.FacetGrid, title: str | None, title_spacing: float, xlabel: str, ylabel: str
) -> sns.FacetGrid:
    """Adds titles, labels and correct spacing between facets."""

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(x_var=xlabel, y_var=ylabel)
    g.tight_layout()

    g.legend.set_title("Chain")

    if title is not None:
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=title_spacing)

    return g


def save_figure(g: sns.FacetGrid | None = None, save_path: str | None = None) -> None:
    """Saves plot to file."""

    if save_path is not None:
        if g is not None:
            g.fig.savefig(save_path)
        else:
            plt.savefig(save_path)
    else:
        plt.show()


def plot_trace(
    results: SamplingResults,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    title: str | None = None,
    title_spacing: float = 0.85,
    xlabel: str = "Iteration",
    style: str = "whitegrid",
    color_palette: str | list[str] | dict[int, str] | None = None,
    ncol: int = 3,
    height: int = 3,
    aspect_ratio: int = 1,
    save_path: str | None = None,
    include_warmup: bool = False,
    **kwargs,
) -> sns.FacetGrid:
    """
    Visualizes posterior samples over time with a trace plot.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    params
        Names of the model parameters that are contained in the plot. Must coincide
        with the dictionary keys of the `Position` with the posterior samples. If
        `None`, all parameters are included.
    param_indices
        Indices of each model parameter that are contained in the plot. Selects e.g.
        ``beta[0]`` out of a ``beta`` parameter vector. A single index can be specified
        as an integer or a sequence containing one integer. If ``None``, all
        subparameters are included.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    xlabel
        Label of the x-axis.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``"darkgrid"``, ``"whitegrid"``, ``"dark"``, ``"white"``, and ``"ticks"``.
    color_palette
        Passed to the palette argument of ``sns.relplot()``. String values must be
        valid inputs of ``sns.color_palette()`` such as a seaborn color palette or a
        matplotlib colormap. Custom colors can be set with a list of color strings or a
        dictionary with the chain indices as keys and color strings as values. The
        number of color strings must coincide with the number of plotted chains. If
        ``None``, the default ``tab10`` matplotlib colormap is chosen.
    ncol
        Number of subplots/facets within each row of the grid.
    height
        Height in inches of each subplot/facet within the grid.
    aspect_ratio
        Ratio of width / height of each subplot/facet within the grid, i.e. ``width =
        aspect_ratio * height``.
    save_path
        File path where the plot is saved.
    include_warmup
        Include the warmup samples in the trace plot.
    **kwargs
        Further keyword arguments passed to the seaborn ``relplot()`` function.

    Returns
    -------
    A seaborn ``FacetGrid``.
    """

    # NOTE: Docstring duplications
    # The entries `results` to `max_chains` are shared with the `summary()`
    # and all user plotting functions.
    # The entries `title` to `save_path` are shared with `plot_density()`, `plot_cor()`
    # and partially with `plot_param()`.

    sns.set_theme(style=style)
    plot_df = _setup_plot_df(
        results, params, param_indices, chain_indices, max_chains, include_warmup
    )

    g = sns.relplot(
        data=plot_df,
        kind="line",
        x="iteration",
        y="value",
        hue="chain_index",
        col="param_label",
        col_wrap=_set_plot_cols(plot_df, ncol),
        facet_kws=dict(sharex=True, sharey=False),
        palette=color_palette,
        height=height,
        aspect=aspect_ratio,
        **kwargs,
    )

    g = _set_aesthetics(g, title, title_spacing, xlabel, ylabel="")

    save_figure(g, save_path)
    return g


def plot_density(
    results: SamplingResults,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    title: str | None = None,
    title_spacing: float = 0.85,
    xlabel: str = "Value",
    style: str = "whitegrid",
    color_palette: str | list[str] | dict[int, str] | None = None,
    ncol: int = 3,
    height: int = 3,
    aspect_ratio: int = 1,
    save_path: str | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """
    Visualizes posterior distributions with a density plot.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    params
        Names of the model parameters that are contained in the plot. Must coincide
        with the dictionary keys of the ``Position`` with the posterior samples. If
        ``None``, all parameters are included.
    param_indices
        Indices of each model parameter that are contained in the plot. Selects e.g.
        ``beta[0]`` out of a ``beta`` parameter vector. A single index can be specified
        as an integer or a sequence containing one integer. If ``None``, all
        subparameters are included.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    xlabel
        Label of the x-axis.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``.
    color_palette
        Passed to the palette argument of ``sns.displot()``. String values must be
        valid inputs of ``sns.color_palette()`` such as a seaborn color palette or a
        matplotlib colormap. Custom colors can be set with a list of color strings or a
        dictionary with the chain indices as keys and color strings as values. The
        number of color strings must coincide with the number of plotted chains. If
        ``None``, the default ``tab10`` matplotlib colormap is chosen.
    ncol
        Number of subplots/facets within each row of the grid.
    height
        Height in inches of each subplot/facet within the grid.
    aspect_ratio
        Ratio of width / height of each subplot/facet within the grid, i.e. ``width =
        aspect_ratio * height``.
    save_path
        File path where the plot is saved.
    **kwargs
        Further keyword arguments passed to the seaborn ``displot()`` function.

    Returns
    -------
    A seaborn ``FacetGrid``.
    """

    # NOTE: Docstring duplications
    # The entries `results` to `max_chains` are shared with the `summary()`
    # and all user plotting functions.
    # The entries `title` to `save_path` are shared with `plot_trace()`, `plot_cor()`
    # and partially with `plot_param()`.

    sns.set_theme(style=style)
    plot_df = _setup_plot_df(results, params, param_indices, chain_indices, max_chains)

    g = sns.displot(
        data=plot_df,
        kind="kde",
        x="value",
        y=None,
        hue="chain_index",
        col="param_label",
        col_wrap=_set_plot_cols(plot_df, ncol),
        facet_kws=dict(sharex=False, sharey=False),
        palette=color_palette,
        height=height,
        aspect=aspect_ratio,
        **kwargs,
    )

    g = _set_aesthetics(g, title, title_spacing, xlabel, ylabel="")

    save_figure(g, save_path)
    return g


def _compute_max_lags(
    plot_df: pd.DataFrame,
    max_lags: int | None,
) -> int:
    """
    Determines number time lags that are shown on the x-axis of the autocorrelation
    plot.
    """

    num_iterations = plot_df["iteration"].max()
    max_lags = np.min([num_iterations, 30]) if max_lags is None else max_lags
    return max_lags


def plot_cor(
    results: SamplingResults,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    max_lags: int | None = None,
    title: str | None = None,
    title_spacing: float = 0.85,
    xlabel: str = "Lag",
    style: str = "whitegrid",
    color_palette: str | list[str] | dict[int, str] | None = None,
    ncol: int = 3,
    height: int = 3,
    aspect_ratio: int = 1,
    save_path: str | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """
    Visualizes autocorrelations of posterior samples.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    params
        Names of the model parameters that are contained in the plot. Must coincide
        with the dictionary keys of the ``Position`` with the posterior samples. If
        ``None``, all parameters are included.
    param_indices
        Indices of each model parameter that are contained in the plot. Selects e.g.
        ``beta[0]`` out of a ``beta`` parameter vector. A single index can be specified
        as an integer or a sequence containing one integer. If ``None``, all
        subparameters are included.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    max_lags
        Maximum number of time lags shown on the x-axis of the autocorrelation plot. If
        ``None``, the minimum of the chain lengths and 30 is chosen.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    xlabel
        Label of the x-axis.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``.
    color_palette
        Passed to the palette argument of ``sns.FacetGrid()``. String values must be
        valid inputs of ``sns.color_palette()`` such as a seaborn color palette or a
        matplotlib colormap. Custom colors can be set with a list of color strings or a
        dictionary with the chain indices as keys and color strings as values. The
        number of color strings must coincide with the number of plotted chains. If
        ``None``, the default ``tab10`` matplotlib colormap is chosen.
    ncol
        Number of subplots/facets within each row of the grid.
    height
        Height in inches of each subplot/facet within the grid.
    aspect_ratio
        Ratio of width / height of each subplot/facet within the grid, i.e. ``width =
        aspect_ratio * height``.
    save_path
        File path where the plot is saved.
    **kwargs
        Further keyword arguments passed to the seaborn ``FacetGrid()`` function.


    Returns
    -------
    A seaborn ``FacetGrid``.
    """

    # NOTE: Docstring duplications
    # The entries `results` to `max_chains` are shared with the `summary()`
    # and all user plotting functions.
    # The entries `title` to `save_path` are shared with `plot_trace()`,
    # `plot_density()` and partially with `plot_param()`.
    # The entry `max_lags` is shared with `plot_param()`.

    sns.set_theme(style=style)
    plot_df = _setup_plot_df(results, params, param_indices, chain_indices, max_chains)
    max_lags = _compute_max_lags(plot_df, max_lags)

    def do_acor_plot(x, maxlags, **kwargs):
        x = np.asarray(x)
        acor = arviz.autocorr(x)[..., 0:maxlags]
        return sns.lineplot(x=range(maxlags), y=acor, **kwargs)

    g = (
        sns.FacetGrid(
            data=plot_df,
            hue="chain_index",
            col="param_label",
            col_wrap=_set_plot_cols(plot_df, ncol),
            palette=color_palette,
            height=height,
            aspect=aspect_ratio,
            **kwargs,
        )
        .map(
            do_acor_plot,
            "value",
            maxlags=max_lags,
        )
        .set(
            xlim=(0, max_lags),
            ylim=(-0.2, 1.1),
        )
        .add_legend()
    )

    g = _set_aesthetics(g, title, title_spacing, xlabel, ylabel="Autocorrelation")

    save_figure(g, save_path)
    return g


def _raise_multi_param_error(plot_df: pd.DataFrame, param: str) -> None:
    """
    :func:`.plot_param` function can only display all three diagnostic plots for a
    single subparameter. Throws an informative error otherwise.
    """

    if plot_df["param_label"].nunique() > 1:
        raise ValueError(
            f"{param} has more than one index. "
            "Please specify a single `param_index` for plotting."
        )


def _set_colors(plot_df: pd.DataFrame, color_list: list[str] | None) -> list[str]:
    """Determines colors of different chains in each plot."""

    num_chains = plot_df["chain_index"].nunique()

    if color_list is None:
        # default matplotlib and seaborn colors with 10 elements
        color_list = sns.color_palette()

        # make default color list sufficiently long
        if num_chains > 10:
            color_list = color_list * (num_chains // 10 + 1)

        color_list = color_list[:num_chains]  # type: ignore

    return color_list


def _setup_grid(
    figure_size: tuple[int | float, int | float]
) -> tuple[plt.Figure, Any, Any, Any]:
    """
    Initializes plotting grid with one large subplot for the trace plot and two smaller
    subplots for the density and autocorrelation plot.
    """

    fig = plt.figure(figsize=figure_size)
    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
    ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
    return fig, ax1, ax2, ax3


def _add_lineplot(plot_df: pd.DataFrame, ax: Any, color_list: list[str]) -> None:
    """Adds trace plot to plotting grid."""

    sns.lineplot(
        data=plot_df,
        x="iteration",
        y="value",
        hue="chain_index",
        palette=color_list,
        ax=ax,
        legend="full",
    ).set(xlabel="Iteration", ylabel="")


def _add_kdeplot(plot_df: pd.DataFrame, ax: Any, color_list: list[str]) -> None:
    """Adds density plot to plotting grid."""

    sns.kdeplot(
        data=plot_df,
        x="value",
        hue="chain_index",
        palette=color_list,
        ax=ax,
        legend=False,
    ).set(xlabel="Value", ylabel="")


def _add_corplot(
    plot_df: pd.DataFrame, ax: Any, max_lags: int | None, color_list: list[str]
) -> None:
    """Adds correlation plot to plotting grid."""

    max_lags = _compute_max_lags(plot_df, max_lags)

    for chain_index, col in zip(plot_df["chain_index"].unique(), color_list):
        x = np.asarray(plot_df.loc[plot_df["chain_index"] == chain_index]["value"])
        acor = arviz.autocorr(x)[0:max_lags]

        sns.lineplot(
            x=range(max_lags),
            y=acor,
            marker="",
            linestyle="-",
            color=col,
            ax=ax,
        )

        ax.set(xlim=(0, max_lags), ylim=(-0.2, 1.1), xlabel="Lag", ylabel="")


def _get_title(plot_df: pd.DataFrame, title: str | None) -> str:
    """Sets either a custom or the default plot title."""

    default_title = f"Diagnostic plots for '{plot_df['param'][0]}'"
    return title if title is not None else default_title


def plot_param(
    results: SamplingResults,
    param: str,
    param_index: int | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    max_lags: int | None = None,
    title: str | None = None,
    title_spacing: float = 0.9,
    style: str = "whitegrid",
    color_list: list[str] | None = None,
    figure_size: tuple[int | float, int | float] = (9, 6),
    # default values chosen for default figure size of (9, 6)
    legend_position: tuple[float, float] = (1.2, 0.4),
    save_path: str | None = None,
) -> None:
    """
    Visualizes trace plot, density plot and autocorrelation plot of a single
    subparameter.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    param
        Name of a single model parameter that is contained in the plot. Must coincide
        with one dictionary key of the ``Position`` with the posterior samples.
    param_index
        A single index of the selected model parameter that is contained in the plot.
        Selects e.g. ``beta[0]`` out of a ``beta`` parameter vector. Can be specified
        as an integer or as a sequence containing one integer. If ``None``, the
        parameter is assumed to have only a single index.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    max_lags
        Maximum number of time lags shown on the x-axis of the autocorrelation plot. If
        ``None``, the minimum of the chain lengths and 30 is chosen.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``.
    color_list
        Determines the chain colors for all three subplots. Custom colors can be passed
        with a list of color strings. The length of the list must match the number of
        chains. If ``None``, the default ``tab10`` matplotlib colormap is chosen.
    figure_size
        Size of the entire plot grid. Passed to the ``figsize`` argument of
        ``plt.figure()``. When changing the figure size consider changing the
        ``legend_position`` as well. Generally, a ratio of 3
    legend_position
        Determines the color legend position. Coordinates are relative to the upper
        panel within the plot grid. The first coordinate specifies the horizontal, the
        second coordinate the vertical position. Might require an adjustment when
        changing the ``figure_size`` values or the number of chains.
    save_path
        File path where the plot is saved.
    """

    # NOTE: Docstring duplications
    # The entries `results`, `chain_indices` and `max_chains` are shared with the
    # `summary()` and all user plotting functions.
    # The entries `title`, `title_spacing`, `style` and `save_path` are shared with
    # `plot_trace()`, `plot_density()` and `plot_cor()`.
    # The entry `max_lags` is shared with `plot_cor()`.

    sns.set_theme(style=style)
    plot_df = _setup_plot_df(results, param, param_index, chain_indices, max_chains)
    _raise_multi_param_error(plot_df, param)
    color_list = _set_colors(plot_df, color_list)

    fig, ax1, ax2, ax3 = _setup_grid(figure_size)
    _add_lineplot(plot_df, ax1, color_list)
    _add_kdeplot(plot_df, ax2, color_list)
    _add_corplot(plot_df, ax3, max_lags, color_list)
    ax1.legend(title="Chain", bbox_to_anchor=legend_position, frameon=False)

    fig.tight_layout()
    fig.suptitle(_get_title(plot_df, title))
    fig.subplots_adjust(top=title_spacing)

    save_figure(save_path=save_path)


def plot_scatter(
    results: SamplingResults,
    params: list[str],
    param_indices: tuple[int, int],
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    alpha: float = 0.2,
    title: str | None = None,
    title_spacing: float = 0.9,
    style: str = "whitegrid",
    color_list: list[str] | None = None,
    figure_size: tuple[int | float, int | float] = (9, 6),
    legend_position: tuple[float, float] | str = "best",
    save_path: str | None = None,
    include_warmup: bool = False,
):
    """
    Produces a scatterplot of two parameters.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    params
        Names of the model parameters that are contained in the plot. Must coincide with
        the dictionary keys of the ``Position`` with the posterior samples.
    param_indices
        Indices of each model parameter that are contained in the plot. Selects e.g.
        ``beta[0]`` out of a ``beta`` parameter vector. If only one string is supplied
        as the value of ``params``, ``param_indices`` must contain two indices. If a
        sequence of two strings is supplied to ``params``, you can supply either a
        single integer or a tuple of two integers. A single integer will be used as the
        index for *both* parameters. If you use a tuple of two integers, the first
        element will be used as the index for the first parameter, and the second
        element will be used as the index for the second parameter.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    alpha
        Amount of transparency; a float between 0 and 1.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``.
    color_list
        Determines the chain colors for all three subplots. Custom colors can be passed
        with a list of color strings. The length of the list must match the number of
        chains. If ``None``, the default ``tab10`` matplotlib colormap is chosen.
    figure_size
        Size of the entire plot grid. Passed to the ``figsize`` argument of
        ``plt.figure()``. When changing the figure size consider changing the
        ``legend_position`` as well. Generally, a ratio of 3
    legend_position
        Determines the color legend position. Coordinates are relative to the upper
        panel within the plot grid. The first coordinate specifies the horizontal, the
        second coordinate the vertical position. Might require an adjustment when
        changing the ``figure_size`` values or the number of chains.
    save_path
        File path where the plot is saved.
    """
    # NOTE: Docstring duplications
    # Multiple arguments in this docstring are shared with other plotting functions.

    sns.set_theme(style=style)
    plot_df = _setup_scatterplot_df(
        results, params, param_indices, chain_indices, max_chains, include_warmup
    )

    labels = plot_df.param_label.unique()
    if len(labels) != 2:
        raise ValueError(
            "'plot_scatter' can only plot exactly two parameters. Use 'plot_pairs'"
            " instead to plot more."
        )

    plot_df = (
        plot_df.drop(["param_index", "param"], axis=1)
        .pivot(
            index=["chain_index", "iteration"], columns="param_label", values="value"
        )
        .reset_index()
        .drop(["iteration"], axis=1)
    )

    color_list = _set_colors(plot_df, color_list)
    fig, axis = plt.subplots(1, 1, figsize=figure_size)
    sns.scatterplot(
        data=plot_df,
        x=labels[0],
        y=labels[1],
        alpha=alpha,
        hue="chain_index",
        palette=color_list,
        ax=axis,
    )
    if title is not None:
        fig.suptitle(title)
        fig.subplots_adjust(top=title_spacing)

    axis.legend(title="Chain", loc=legend_position, frameon=False)

    save_figure(save_path=save_path)


def plot_pairs(
    results: SamplingResults,
    params: str | list[str] | None = None,
    param_indices: int | Sequence[int] | None = None,
    chain_indices: int | Sequence[int] | None = None,
    max_chains: int | None = 5,
    alpha: float = 0.2,
    title: str | None = None,
    title_spacing: float = 0.9,
    style: str = "whitegrid",
    diag_kind: str = "kde",
    color_palette: str | list[str] | dict[int, str] | None = None,
    height: int = 3,
    aspect_ratio: int = 1,
    save_path: str | None = None,
    include_warmup: bool = False,
):
    """
    Produces a pairplot panel.

    Parameters
    ----------
    results
        Result object of the sampling process. Must have a method
        ``get_posterior_samples()`` which extracts all samples from the posterior
        distribution.
    params
        Names of the model parameters that are contained in the plot. Must coincide with
        the dictionary keys of the ``Position`` with the posterior samples. If ``None``,
        all parameters are included.
    param_indices
        Indices of each model parameter that are contained in the plot. Selects e.g.
        ``beta[0]`` out of a ``beta`` parameter vector. A single index can be specified
        as an integer or a sequence containing one integer. If ``None``, all
        subparameters are included.
    chain_indices
        Indices of chains for each model subparameter that are contained in the plot.
        Selects e.g. chain 0 and chain 2 out of multiple chains. A single index can be
        specified as an integer or a sequence containing one integer. If ``None``, all
        chains are included.
    max_chains
        Upper bound how many chains are included within each subplot/facet. Avoids
        overplotting. If ``None``, all chains contained in the ``results`` input are
        plotted. Always starts chain selection from the lowest chain index upwards. For
        selecting specific chains use the argument ``chain_indices``.
    alpha
        Amount of transparency; a float between 0 and 1.
    title
        Plot title.
    title_spacing
        Determines the margin/whitespace between the plot title (set with
        ``fig.suptitle()``) and the first row of subplots/facets. Passed to the ``top``
        argument of ``fig.subplots_adjust()``.
    style
        Passed to the ``style`` argument of ``sns.set_theme()``. Valid options are
        ``darkgrid``, ``whitegrid``, ``dark``, ``white``, and ``ticks``.
    diag_kind
        Kind of plot for the diagonal subplots. Can be 'kde' (default) for kernel
        density estimates or 'hist' for histograms.
    color_palette
        Passed to the palette argument of ``sns.pairplot()``. String values must be
        valid inputs of ``sns.color_palette()`` such as a seaborn color palette or a
        matplotlib colormap. Custom colors can be set with a list of color strings or a
        dictionary with the chain indices as keys and color strings as values. The
        number of color strings must coincide with the number of plotted chains. If
        ``None``, the default ``tab10`` matplotlib colormap is chosen.
    height
        Height in inches of each subplot/facet within the grid.
    aspect_ratio
        Ratio of width / height of each subplot/facet within the grid, i.e. ``width =
        aspect_ratio * height``.
    legend_position
        Determines the color legend position. Coordinates are relative to the upper
        panel within the plot grid. The first coordinate specifies the horizontal, the
        second coordinate the vertical position. Might require an adjustment when
        changing the ``figure_size`` values or the number of chains.
    save_path
        File path where the plot is saved.
    include_warmup
        Include the warmup samples in the trace plot.
    """
    # NOTE: Docstring duplications
    # Multiple arguments in this docstring are shared with other plotting functions.

    sns.set_theme(style=style)
    plot_df = _setup_plot_df(
        results, params, param_indices, chain_indices, max_chains, include_warmup
    )

    plot_df = (
        plot_df.drop(["param_index", "param"], axis=1)
        .pivot(
            index=["chain_index", "iteration"], columns="param_label", values="value"
        )
        .reset_index()
        .drop(["iteration"], axis=1)
    )

    g = sns.pairplot(
        data=plot_df,
        hue="chain_index",
        plot_kws={"alpha": alpha},
        diag_kind=diag_kind,
        height=height,
        aspect=aspect_ratio,
        palette=color_palette,
    )

    if title is not None:
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=title_spacing)

    save_figure(save_path=save_path)
