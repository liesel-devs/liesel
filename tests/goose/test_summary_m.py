from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from liesel.__version__ import __version__
from liesel.goose.engine import SamplingResults
from liesel.goose.summary_m import SamplesSummary, Summary

# TODO: add tests to test correctness of quantities
# TODO: speed up tests


def test_shapes(result: SamplingResults):
    summary = Summary(result)

    # combined chains
    assert summary.quantities["mean"]["foo"].shape == (3,)
    assert summary.quantities["mean"]["bar"].shape == (3, 5, 7)
    assert summary.quantities["mean"]["baz"].shape == ()

    assert summary.quantities["quantile"]["foo"].shape == (
        3,
        3,
    )
    assert summary.quantities["quantile"]["bar"].shape == (3, 3, 5, 7)
    assert summary.quantities["quantile"]["baz"].shape == (3,)

    assert summary.quantities["hdi"]["foo"].shape == (
        2,
        3,
    )
    assert summary.quantities["hdi"]["bar"].shape == (2, 3, 5, 7)
    assert summary.quantities["hdi"]["baz"].shape == (2,)

    # combined chains
    summary = Summary(result, quantiles=(0.2, 0.4), per_chain=True)

    assert summary.quantities["mean"]["foo"].shape == (
        3,
        3,
    )
    assert summary.quantities["mean"]["bar"].shape == (3, 3, 5, 7)
    assert summary.quantities["mean"]["baz"].shape == (3,)

    assert summary.quantities["quantile"]["foo"].shape == (
        3,
        2,
        3,
    )
    assert summary.quantities["quantile"]["bar"].shape == (3, 2, 3, 5, 7)
    assert summary.quantities["quantile"]["baz"].shape == (3, 2)

    assert summary.quantities["hdi"]["foo"].shape == (
        3,
        2,
        3,
    )
    assert summary.quantities["hdi"]["bar"].shape == (3, 2, 3, 5, 7)
    assert summary.quantities["hdi"]["baz"].shape == (
        3,
        2,
    )


def test_additional_chain(result: SamplingResults):
    chain = result.get_posterior_samples()
    chain["expbaz"] = jnp.log(chain["baz"] + 1)
    chain["expbar"] = jnp.log(chain["bar"] + 1)

    summary = Summary(result, chain)

    assert summary.quantities["mean"]["expbar"].shape == (3, 5, 7)
    assert summary.quantities["mean"]["expbaz"].shape == ()


def test_selected(result: SamplingResults):
    summary = Summary(result, selected=["foo"])

    assert "foo" in summary.quantities["mean"]
    assert "bar" not in summary.quantities["mean"]
    assert "baz" not in summary.quantities["mean"]


def test_deselected(result: SamplingResults):
    summary = Summary(result, deselected=["baz"])

    assert "foo" in summary.quantities["mean"]
    assert "bar" in summary.quantities["mean"]
    assert "baz" not in summary.quantities["mean"]


def test_mean(result: SamplingResults):
    summary = Summary(result)
    assert jnp.allclose(
        summary.quantities["mean"]["foo"], jnp.array([175.5, 176.5, 177.5])
    )

    assert jnp.allclose(summary.quantities["mean"]["bar"], 175.5 * jnp.ones((3, 5, 7)))

    assert jnp.allclose(summary.quantities["mean"]["baz"], jnp.array(176.5))


def test_config(result: SamplingResults):
    summary = Summary(result, quantiles=(0.4, 0.6), hdi_prob=0.5)
    assert summary.config["chains_merged"]
    assert summary.config["quantiles"] == (0.4, 0.6)
    assert summary.config["hdi_prob"] == 0.5


def test_sample_info(result: SamplingResults):
    summary = Summary(result)
    print(summary.sample_info)
    assert summary.sample_info["num_chains"] == 3
    assert summary.sample_info["sample_size_per_chain"] == 250


def test_df_sample_info(result: SamplingResults):
    summary = Summary(result, selected=["baz"]).to_dataframe()
    assert summary["sample_size"].iloc[0] == 3 * 250

    summary = Summary(result, per_chain=True, selected=["baz"]).to_dataframe()
    assert summary["sample_size"].iloc[0] == 250


def test_error_summary(result: SamplingResults):
    # add some error codes to the chain

    # epoch 1 - warmup
    ecs = np.array(
        result.transition_infos.get_specific_chain(1)
        ._chunks_list[0]["kernel_00"]
        .error_code
    )
    ecs[0, 0] = 1
    ecs[:, 3] = 1
    ecs[2, 3:5] = 2
    result.transition_infos.get_specific_chain(1)._chunks_list[0][
        "kernel_00"
    ].error_code = ecs

    # posterior
    ecs = np.array(
        result.transition_infos.get_current_chain()
        ._chunks_list[0]["kernel_00"]
        .error_code
    )
    ecs[0, 0] = 1
    ecs[:, 230] = 1
    ecs[1, 220:225] = 2
    result.transition_infos.get_current_chain()._chunks_list[0][
        "kernel_00"
    ].error_code = ecs

    # create the summary object
    summary = Summary(result, selected=["baz"])
    es = summary.error_summary["kernel_00"]

    # check error_code correspond
    assert es[1].error_code == 1
    assert es[2].error_code == 2

    # check that these error codes don't have entries
    assert 3 not in es
    assert 0 not in es

    # check that error messages are correct
    assert es[1].error_msg == "error 1"
    assert es[2].error_msg == "error 2"

    # check that counts are correct - overall
    assert np.all(es[1].count_per_chain == np.array([4, 2, 1]))
    assert np.all(es[2].count_per_chain == np.array([0, 5, 2]))

    # check that counts are correct - only in the posterior
    assert np.all(es[1].count_per_chain_posterior == np.array([2, 1, 1]))
    assert np.all(es[2].count_per_chain_posterior == np.array([0, 5, 0]))


def test_single_chain_repr_fs_return(single_chain_result: SamplingResults):
    summary = Summary(single_chain_result)
    md = summary._repr_markdown_()
    html = summary._repr_html_()
    assert isinstance(md, str)
    assert isinstance(html, str)


def test_per_chain_quantiles(result: SamplingResults):
    summary = Summary(result, per_chain=True)
    cols = [
        "var_fqn",
        "chain_index",
        "mean",
        "q_0.05",
        "q_0.5",
        "q_0.95",
        "hdi_low",
        "hdi_high",
    ]
    df = summary.to_dataframe().loc["baz"][cols]

    assert np.allclose(df["q_0.05"], 64.449997)
    assert np.allclose(df["q_0.5"], 176.5)
    assert np.allclose(df["q_0.95"], 288.549988)
    assert np.allclose(df["hdi_low"], 52.0)
    assert np.allclose(df["hdi_high"], 277.0)


def test_quantity_shape(result_for_quants: SamplingResults):
    """
    Confirms that the HDI and quantile quantities at the summary object have the
    intended shape:

    - First index refers to the chain
    - Second index refers to the quantile/hdi
    - Following indices refer to individual parameters
    """
    summary = Summary(result_for_quants, per_chain=True)

    assert summary.quantities["quantile"]["foo"].shape == (4, 3, 5)
    assert summary.quantities["quantile"]["bar"].shape == (4, 3, 3, 5, 7)
    assert summary.quantities["quantile"]["baz"].shape == (4, 3)

    assert summary.quantities["hdi"]["foo"].shape == (4, 2, 5)
    assert summary.quantities["hdi"]["bar"].shape == (4, 2, 3, 5, 7)
    assert summary.quantities["hdi"]["baz"].shape == (4, 2)


def test_liesel_version(result: SamplingResults):
    summary = Summary(result, per_chain=True)

    assert summary.liesel_version == __version__


@pytest.mark.parametrize(
    ("per_chain", "which_"),
    list(
        product(
            (True, False),
            [
                ("mean", "sd"),
                ("hdi",),
                ("mean", "ess_bulk"),
                ("mean", "mcse_mean"),
                ("mean", "mcse_sd"),
                ("quantiles", "var"),
                ("mean", "sd", "var", "quantiles"),
                ("mean", "ess_bulk", "ess_tail"),
                ("mean", "ess_bulk", "ess_tail", "rhat"),
            ],
        )
    ),
)
def test_deselect_quantities(result: SamplingResults, per_chain, which_):
    summary = Summary(result, which=which_, per_chain=per_chain)
    df = summary.to_dataframe()
    for key in which_:
        if key == "rhat" and per_chain:
            continue

        if key not in ["quantiles", "hdi"]:
            assert key in df.columns

        if key == "hdi":
            assert "hdi_low" in df.columns
            assert "hdi_high" in df.columns
        if key == "quantiles":
            assert "q_0.05" in df.columns
            assert "q_0.5" in df.columns
            assert "q_0.95" in df.columns

    n_quantity_columns = len(which_)
    if "quantiles" in which_:
        n_quantity_columns += len(summary.config["quantiles"]) - 1
    if "hdi" in which_:
        n_quantity_columns += 1
    if per_chain:
        n_quantity_columns += 1

    if per_chain and "rhat" in which_:
        n_quantity_columns -= 1

    assert (len(df.columns) - n_quantity_columns) == 4


class TestSamplesSummary:
    def test_shapes(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples)

        # combined chains
        assert summary.quantities["mean"]["foo"].shape == (3,)
        assert summary.quantities["mean"]["bar"].shape == (3, 5, 7)
        assert summary.quantities["mean"]["baz"].shape == ()

        assert summary.quantities["quantile"]["foo"].shape == (
            3,
            3,
        )
        assert summary.quantities["quantile"]["bar"].shape == (3, 3, 5, 7)
        assert summary.quantities["quantile"]["baz"].shape == (3,)

        assert summary.quantities["hdi"]["foo"].shape == (
            2,
            3,
        )
        assert summary.quantities["hdi"]["bar"].shape == (2, 3, 5, 7)
        assert summary.quantities["hdi"]["baz"].shape == (2,)

        # combined chains
        summary = SamplesSummary(samples, quantiles=(0.2, 0.4), per_chain=True)

        assert summary.quantities["mean"]["foo"].shape == (
            3,
            3,
        )
        assert summary.quantities["mean"]["bar"].shape == (3, 3, 5, 7)
        assert summary.quantities["mean"]["baz"].shape == (3,)

        assert summary.quantities["quantile"]["foo"].shape == (
            3,
            2,
            3,
        )
        assert summary.quantities["quantile"]["bar"].shape == (3, 2, 3, 5, 7)
        assert summary.quantities["quantile"]["baz"].shape == (3, 2)

        assert summary.quantities["hdi"]["foo"].shape == (
            3,
            2,
            3,
        )
        assert summary.quantities["hdi"]["bar"].shape == (3, 2, 3, 5, 7)
        assert summary.quantities["hdi"]["baz"].shape == (
            3,
            2,
        )

    def test_selected(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, selected=["foo"])

        assert "foo" in summary.quantities["mean"]
        assert "bar" not in summary.quantities["mean"]
        assert "baz" not in summary.quantities["mean"]

    def test_deselected(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, deselected=["baz"])

        assert "foo" in summary.quantities["mean"]
        assert "bar" in summary.quantities["mean"]
        assert "baz" not in summary.quantities["mean"]

    def test_mean(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples)
        assert jnp.allclose(
            summary.quantities["mean"]["foo"], jnp.array([175.5, 176.5, 177.5])
        )

        assert jnp.allclose(
            summary.quantities["mean"]["bar"], 175.5 * jnp.ones((3, 5, 7))
        )

        assert jnp.allclose(summary.quantities["mean"]["baz"], jnp.array(176.5))

    def test_config(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, quantiles=(0.4, 0.6), hdi_prob=0.5)
        assert summary.config["chains_merged"]
        assert summary.config["quantiles"] == (0.4, 0.6)
        assert summary.config["hdi_prob"] == 0.5

    def test_sample_info(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples)
        print(summary.sample_info)
        assert summary.sample_info["num_chains"] == 3
        assert summary.sample_info["sample_size_per_chain"] == 250

    def test_df_sample_info(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, selected=["baz"]).to_dataframe()
        assert summary["sample_size"].iloc[0] == 3 * 250

        summary = Summary(result, per_chain=True, selected=["baz"]).to_dataframe()
        assert summary["sample_size"].iloc[0] == 250

    def test_per_chain_quantiles(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, per_chain=True)
        cols = [
            "var_fqn",
            "chain_index",
            "mean",
            "q_0.05",
            "q_0.5",
            "q_0.95",
            "hdi_low",
            "hdi_high",
        ]
        df = summary.to_dataframe().loc["baz"][cols]

        assert np.allclose(df["q_0.05"], 64.449997)
        assert np.allclose(df["q_0.5"], 176.5)
        assert np.allclose(df["q_0.95"], 288.549988)
        assert np.allclose(df["hdi_low"], 52.0)
        assert np.allclose(df["hdi_high"], 277.0)

    def test_quantity_shape(self, result_for_quants: SamplingResults):
        """
        Confirms that the HDI and quantile quantities at the summary object have the
        intended shape:

        - First index refers to the chain
        - Second index refers to the quantile/hdi
        - Following indices refer to individual parameters
        """
        samples = result_for_quants.get_posterior_samples()
        summary = SamplesSummary(samples, per_chain=True)

        assert summary.quantities["quantile"]["foo"].shape == (4, 3, 5)
        assert summary.quantities["quantile"]["bar"].shape == (4, 3, 3, 5, 7)
        assert summary.quantities["quantile"]["baz"].shape == (4, 3)

        assert summary.quantities["hdi"]["foo"].shape == (4, 2, 5)
        assert summary.quantities["hdi"]["bar"].shape == (4, 2, 3, 5, 7)
        assert summary.quantities["hdi"]["baz"].shape == (4, 2)

    def test_from_array(self, result: SamplingResults):
        samples = result.get_posterior_samples()
        summary = SamplesSummary.from_array(samples["bar"], name="test")
        assert summary.to_dataframe().shape == (105, 16)

    @pytest.mark.parametrize(
        ("per_chain", "which_"),
        list(
            product(
                (True, False),
                [
                    ("mean", "sd"),
                    ("hdi",),
                    ("mean", "ess_bulk"),
                    ("mean", "mcse_mean"),
                    ("mean", "mcse_sd"),
                    ("quantiles", "var"),
                    ("mean", "sd", "var", "quantiles"),
                    ("mean", "ess_bulk", "ess_tail"),
                    ("mean", "ess_bulk", "ess_tail", "rhat"),
                ],
            )
        ),
    )
    def test_deselect_quantities(self, result: SamplingResults, per_chain, which_):
        samples = result.get_posterior_samples()
        summary = SamplesSummary(samples, which=which_, per_chain=per_chain)
        df = summary.to_dataframe()
        for key in which_:
            if key == "rhat" and per_chain:
                continue

            if key not in ["quantiles", "hdi"]:
                assert key in df.columns

            if key == "hdi":
                assert "hdi_low" in df.columns
                assert "hdi_high" in df.columns
            if key == "quantiles":
                assert "q_0.05" in df.columns
                assert "q_0.5" in df.columns
                assert "q_0.95" in df.columns

        n_quantity_columns = len(which_)
        if "quantiles" in which_:
            n_quantity_columns += len(summary.config["quantiles"]) - 1
        if "hdi" in which_:
            n_quantity_columns += 1
        if per_chain:
            n_quantity_columns += 1

        if per_chain and "rhat" in which_:
            n_quantity_columns -= 1

        assert (len(df.columns) - n_quantity_columns) == 3
