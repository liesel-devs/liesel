import os.path

import numpy as np
import pandas as pd
import pytest

from liesel.goose.engine import SamplingResults
from liesel.goose.summary_m import get_param_stats, get_subparam_stats, summary

# ---------------------------------------------------------------------------- #
#              Load results from file.                                         #
#              file was generated with files/files_for_test_summary.py         #
# ---------------------------------------------------------------------------- #
path_module_dir = os.path.dirname(__file__)
path0 = os.path.join(path_module_dir, "files", "summary_res.pkl")
results = SamplingResults.pkl_load(path0)
path1 = os.path.join(path_module_dir, "files", "summary_res_long.pkl")
results_long = SamplingResults.pkl_load(path1)
posterior_samples = results.get_posterior_samples()

# ---------------------------------------------------------------------------- #
#              Test Output for a single Subparameter (e.g. beta_0)             #
# ---------------------------------------------------------------------------- #

chains_beta_0 = np.array(posterior_samples["beta"][..., 0])
beta_0_dict_per_chain = get_subparam_stats(
    chains_beta_0,
    per_chain=True,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=False,
)
beta_0_dict_aggregated = get_subparam_stats(
    chains_beta_0,
    per_chain=False,
    chain_indices=1,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=False,
)

chains_log_sigma = np.array(posterior_samples["log_sigma"])
log_sigma_subparam_dict_per_chain = get_subparam_stats(
    chains_log_sigma,
    per_chain=True,
    chain_indices=[0, 2],
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=False,
)
log_sigma_subparam_dict_aggregated = get_subparam_stats(
    chains_log_sigma,
    per_chain=False,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=False,
)

subparam_dict_keys = [
    "param_index",
    "num_chains",
    "chain_index",
    "num_samples",
    "num_effective",
    "mean",
    "sd",
    "rhat",
    "q_5",
    "q_50",
    "q_95",
    "hdi_90_low",
    "hdi_90_high",
]


def test_subparam_dict_per_chain():
    assert list(beta_0_dict_per_chain.keys()) == subparam_dict_keys

    assert beta_0_dict_per_chain["param_index"] == 0

    num_chains = beta_0_dict_per_chain["num_chains"]
    assert num_chains == 3
    assert beta_0_dict_per_chain["chain_index"] == list(range(num_chains))
    assert beta_0_dict_per_chain["num_samples"] == [1000] * num_chains

    assert isinstance(beta_0_dict_per_chain["mean"], np.ndarray)
    assert len(beta_0_dict_per_chain["mean"]) == num_chains
    assert isinstance(beta_0_dict_per_chain["q_50"], np.ndarray)
    assert len(beta_0_dict_per_chain["q_50"]) == num_chains
    assert isinstance(beta_0_dict_per_chain["hdi_90_low"], np.ndarray)
    assert len(beta_0_dict_per_chain["hdi_90_low"]) == num_chains

    assert log_sigma_subparam_dict_per_chain["param_index"] == 0
    assert log_sigma_subparam_dict_per_chain["chain_index"] == [0, 2]


def test_subparam_dict_aggregated():
    assert beta_0_dict_aggregated["param_index"] == 0
    assert beta_0_dict_aggregated["num_chains"] == 1
    assert beta_0_dict_aggregated["chain_index"] == [1]

    assert isinstance(beta_0_dict_aggregated["mean"], np.ndarray)
    assert len(beta_0_dict_aggregated["mean"]) == 1
    assert np.isnan(beta_0_dict_aggregated["rhat"])


beta_0_df_per_chain = get_subparam_stats(
    chains_beta_0,
    per_chain=True,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=True,
)
beta_0_df_aggregated = get_subparam_stats(
    chains_beta_0,
    per_chain=False,
    chain_indices=1,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=True,
)

log_sigma_subparam_df_per_chain = get_subparam_stats(
    chains_log_sigma,
    per_chain=True,
    chain_indices=[0, 2],
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=True,
)
log_sigma_subparam_df_aggregated = get_subparam_stats(
    chains_log_sigma,
    per_chain=False,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    param_index=0,
    as_dataframe=True,
)


def test_subparam_df_per_chain():
    assert isinstance(beta_0_df_per_chain, pd.DataFrame)

    subparam_dict_keys.remove("num_chains")
    assert beta_0_df_per_chain.columns.to_list() == subparam_dict_keys

    assert len(beta_0_df_per_chain) == 3
    assert beta_0_df_per_chain["num_samples"].nunique() == 1
    assert beta_0_df_per_chain["rhat"].nunique() == 1
    assert beta_0_df_per_chain["num_effective"].nunique() == 3
    assert beta_0_df_per_chain["hdi_90_low"].nunique() == 3

    assert len(log_sigma_subparam_df_per_chain) == 2
    assert log_sigma_subparam_df_per_chain["chain_index"].to_list() == [0, 2]


def test_subparam_df_aggregated():
    assert all(
        beta_0_df_aggregated.columns
        == beta_0_df_per_chain.drop(columns="chain_index").columns
    )
    assert len(beta_0_df_aggregated) == 1
    assert len(log_sigma_df_aggregated) == 1
    assert (
        log_sigma_subparam_df_aggregated["num_samples"][0]
        == 3 * log_sigma_subparam_df_per_chain["num_samples"][0]
    )


# ---------------------------------------------------------------------------- #
#              Test Output for a single Parameter (e.g. beta)                  #
# ---------------------------------------------------------------------------- #

beta_dict_per_chain = get_param_stats(
    param="beta",
    posterior_samples=posterior_samples,
    per_chain=True,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=False,
)
beta_dict_aggregated = get_param_stats(
    param="beta",
    posterior_samples=posterior_samples,
    per_chain=False,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=False,
)

log_sigma_dict_per_chain = get_param_stats(
    param="log_sigma",
    posterior_samples=posterior_samples,
    per_chain=True,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=False,
)
log_sigma_dict_aggregated = get_param_stats(
    param="log_sigma",
    posterior_samples=posterior_samples,
    per_chain=False,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=False,
)


def test_param_dict_per_chain():
    assert isinstance(beta_dict_per_chain, list)
    assert len(beta_dict_per_chain) == 2
    assert beta_dict_per_chain[0]["param_index"] == 0
    assert beta_dict_per_chain[1]["param_index"] == 1

    assert isinstance(log_sigma_dict_per_chain, list)
    assert len(log_sigma_dict_per_chain) == 1
    assert log_sigma_dict_per_chain[0]["param_index"] == 0


def test_param_dict_aggregated():
    assert isinstance(beta_dict_aggregated, list)
    assert len(beta_dict_aggregated) == 2
    assert beta_dict_aggregated[0]["param_index"] == 0
    assert beta_dict_aggregated[1]["param_index"] == 1

    assert isinstance(log_sigma_dict_aggregated, list)
    assert len(log_sigma_dict_aggregated) == 1
    assert log_sigma_dict_aggregated[0]["param_index"] == 0


beta_df_per_chain = get_param_stats(
    param="beta",
    posterior_samples=posterior_samples,
    per_chain=True,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=True,
)
beta_df_aggregated = get_param_stats(
    param="beta",
    posterior_samples=posterior_samples,
    per_chain=False,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=True,
)

log_sigma_df_per_chain = get_param_stats(
    param="log_sigma",
    posterior_samples=posterior_samples,
    per_chain=True,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=True,
)
log_sigma_df_aggregated = get_param_stats(
    param="log_sigma",
    posterior_samples=posterior_samples,
    per_chain=False,
    param_indices=None,
    chain_indices=None,
    quantiles=[0.05, 0.5, 0.95],
    hdi_prob=0.9,
    round_digits=3,
    as_dataframe=True,
)


def test_param_df_per_chain():
    assert isinstance(beta_df_per_chain, pd.DataFrame)
    assert len(beta_df_per_chain) == 6
    assert beta_df_per_chain["param_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert beta_df_per_chain["chain_index"].tolist() == [0, 1, 2, 0, 1, 2]
    assert beta_dict_per_chain[1]["param_index"] == 1

    assert isinstance(log_sigma_dict_per_chain, list)
    assert len(log_sigma_dict_per_chain) == 1
    assert log_sigma_dict_per_chain[0]["param_index"] == 0


def test_param_df_aggregated():
    assert all(
        beta_df_aggregated.columns
        == beta_df_per_chain.drop(columns="chain_index").columns
    )
    assert isinstance(beta_df_aggregated, pd.DataFrame)
    assert len(beta_df_aggregated) == 2
    assert beta_df_aggregated["param_index"].tolist() == [0, 1]

    assert len(log_sigma_df_aggregated) == 1


# ---------------------------------------------------------------------------- #
#                           Test Output of summary()                           #
# ---------------------------------------------------------------------------- #


def test_dict_output():
    assert isinstance(summary(results, as_dataframe=False), dict)

    assert list(summary(results, as_dataframe=False).keys()) == ["beta", "log_sigma"]
    assert list(summary(results, per_chain=False, as_dataframe=False).keys()) == [
        "beta",
        "log_sigma",
    ]
    assert list(summary(results, params="beta", as_dataframe=False).keys()) == ["beta"]
    assert list(summary(results, params=["beta"], as_dataframe=False).keys()) == [
        "beta"
    ]
    assert list(
        summary(results, params=["beta", "log_sigma"], as_dataframe=False).keys()
    ) == ["beta", "log_sigma"]


def test_df_output():
    assert isinstance(summary(results), pd.DataFrame)

    assert len(summary(results)) == 9
    assert summary(results).index.unique().tolist() == ["beta", "log_sigma"]

    assert len(summary(results, per_chain=False)) == 3

    assert len(summary(results, param_indices=0)) == 6
    assert summary(results, param_indices=0)["param_index"].unique().item() == 0
    assert summary(results, param_indices=[0])["param_index"].unique().item() == 0
    assert summary(results, param_indices=None)["param_index"].nunique() == 2

    assert len(summary(results, chain_indices=0)) == 3
    assert summary(results, chain_indices=0)["chain_index"].unique().item() == 0
    assert summary(results, chain_indices=[0])["chain_index"].unique().item() == 0
    assert summary(results, chain_indices=None)["chain_index"].nunique() == 3

    rhat_per_chain = summary(results)["rhat"].unique()
    rhat_aggregated = summary(results, per_chain=False)["rhat"].unique()
    assert (rhat_per_chain == rhat_aggregated).all()

    assert "q_1" in summary(results, quantiles=(0.01,))
    assert all(
        col in summary(results, quantiles=[0.8, 0.99]).columns
        for col in ["q_80", "q_99"]
    )

    assert all(
        col in summary(results, hdi_prob=0.1).columns
        for col in ["hdi_10_low", "hdi_10_high"]
    )

    assert summary(results, round_digits=0)["num_effective"][0] == 57.0
    assert summary(results, round_digits=1)["num_effective"][0] == 57.3


def test_correct_error_messages():
    with pytest.raises(KeyError) as exc_info:
        summary(results, params="beta2")
    assert exc_info.type is KeyError
    assert exc_info.value.args[0] == "All params must be in ['beta', 'log_sigma']."

    with pytest.raises(ValueError) as exc_info:
        summary(results, chain_indices=[0, 3])
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "All chain indices must be between 0 and 2 (bounds inclusive)."
    )

    with pytest.raises(ValueError) as exc_info:
        summary(results, param_indices=[0, 2])
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "All param indices for beta must be between 0 and 1 (bounds inclusive)."
    )

    with pytest.raises(ValueError) as exc_info:
        summary(results, param_indices=[0, 1])
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == "All param indices for log_sigma must be between 0 and 0 (bounds inclusive)."
    )


def test_higher_param_dim():
    summary_long = summary(results_long)

    assert len(summary_long) == 33
    assert summary_long["param_index"].nunique() == 10
    assert summary_long["chain_index"].nunique() == 3
    assert summary_long["rhat"].nunique() == 11
    assert summary_long["num_effective"].nunique() == 33
