import os.path

import matplotlib
import pytest

from liesel.goose.engine import SamplingResults
from liesel.goose.summary_viz import (
    plot_cor,
    plot_density,
    plot_param,
    plot_trace,
    setup_plot_df,
)

# use non-interactive backend
matplotlib.use("template")

# load results from file
# file was generated with files/files_for_test_summary.py
path_module_dir = os.path.dirname(__file__)
path = os.path.join(path_module_dir, "files", "summary_viz_res.pkl")
results = SamplingResults.pkl_load(path)


def test_data_complete():
    data_complete = setup_plot_df(
        results, params=None, param_indices=None, chain_indices=None, max_chains=None
    )

    assert len(data_complete) == 90000
    assert len(data_complete.columns) == 6
    assert data_complete["param"].nunique() == 2
    assert data_complete["param_label"].nunique() == 6

    assert data_complete["param_label"].unique().tolist() == [
        "beta[0]",
        "beta[1]",
        "beta[2]",
        "beta[3]",
        "beta[4]",
        "log_sigma",
    ]

    assert data_complete["chain_index"].nunique() == 15
    assert data_complete["iteration"].nunique() == 1000


def test_data_subset():
    data_subset = setup_plot_df(
        results,
        params="beta",
        param_indices=[1, 3, 4],
        chain_indices=[0, 1, 10],
        max_chains=None,
    )

    assert len(data_subset) == 9000
    assert len(data_subset.columns) == 6
    assert data_subset["param"].nunique() == 1

    assert data_subset["param_label"].unique().tolist() == [
        "beta[1]",
        "beta[3]",
        "beta[4]",
    ]

    assert data_subset["chain_index"].nunique() == 3
    assert data_subset["iteration"].nunique() == 1000


def test_max_chains():
    data_max_chains_large = setup_plot_df(
        results, params=None, param_indices=None, chain_indices=None, max_chains=100
    )

    assert data_max_chains_large["chain_index"].nunique() == 15

    data_max_chains_small = setup_plot_df(
        results,
        params=None,
        param_indices=None,
        chain_indices=[0, 3, 5, 6, 7, 8, 10],
        max_chains=2,
    )

    assert data_max_chains_small["chain_index"].nunique() == 2
    assert data_max_chains_small["chain_index"].unique().tolist() == [0, 3]


def test_plot_defaults():
    plot_trace(results)
    plot_density(results)
    plot_cor(results)
    plot_param(results, param="beta", param_index=3)
    plot_param(results, param="log_sigma")


def test_plot_aesthetics():
    plot_trace(
        results,
        max_chains=3,
        title="Custom Title",
        title_spacing=0.86,
        xlabel="Custom xlabel",
        style="white",
        color_palette=["blue", "red", "green"],
        ncol=4,
        height=4,
        aspect_ratio=1.2,
    )

    plot_density(
        results,
        max_chains=3,
        title="Custom Title",
        title_spacing=0.86,
        xlabel="Custom xlabel",
        style="white",
        color_palette={1: "blue", 0: "green", 2: "red"},
        ncol=3,
        height=4,
        aspect_ratio=1.2,
    )

    plot_cor(
        results,
        max_chains=10,
        max_lags=10,
        title="Custom Title",
        title_spacing=0.86,
        xlabel="Custom xlabel",
        style="white",
        ncol=3,
        height=3,
        aspect_ratio=0.8,
    )

    plot_param(
        results,
        param="beta",
        param_index=1,
        max_chains=3,
        max_lags=10,
        title_spacing=0.91,
        style="white",
        color_list=["blue", "red", "green"],
        figure_size=(15, 10),
        legend_position=(1.1, 0.2),
    )


def test_correct_error_message():
    with pytest.raises(ValueError) as exc_info:
        plot_param(results, param="beta")

    assert exc_info.type is ValueError

    assert (
        exc_info.value.args[0]
        == "beta has more than one index. Please specify a single `param_index` "
        "for plotting."
    )
