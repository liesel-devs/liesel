import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection

import liesel.goose as gs
from liesel.goose.chain import EpochChainManager
from liesel.goose.engine import SamplingResults
from liesel.goose.epoch import EpochConfig, EpochType
from liesel.goose.hmc import HMCTransitionInfo
from liesel.goose.kernel import DefaultTransitionInfo
from liesel.goose.nuts import NUTSTransitionInfo
from liesel.option import Option


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def nuts_info(codes):
    codes = np.asarray(codes)
    zeros = np.zeros_like(codes)
    # Transition-info annotations describe single draws; stored fields are batched.
    return NUTSTransitionInfo(
        error_code=codes,
        acceptance_prob=np.ones_like(codes, dtype=float),
        position_moved=zeros,
        divergent=np.isin(codes, [1, 3]),  # ty: ignore[invalid-argument-type]
        turning=zeros,  # ty: ignore[invalid-argument-type]
        treedepth=zeros,  # ty: ignore[invalid-argument-type]
        leapfrog=zeros,  # ty: ignore[invalid-argument-type]
    )


def results_from_epochs(*epochs, kernel_classes=None):
    positions = EpochChainManager(apply_thinning=True)
    transition_infos = EpochChainManager()
    for config, samples, infos in epochs:
        positions.advance_epoch(config)
        transition_infos.advance_epoch(config)
        positions.append(samples)
        if infos is not None:
            transition_infos.append(infos)
    return SamplingResults(
        positions=positions,
        transition_infos=transition_infos,
        generated_quantities=Option(None),
        tuning_infos=Option(None),
        kernel_states=Option(None),
        full_model_states=Option(None),
        kernel_classes=Option(kernel_classes),
        kernels_by_pos_key=Option(None),
    )


def test_scatter_marks_nuts_divergences_including_combined_tree_depth_error():
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])},
            {"nuts": nuts_info([[0, 1, 2, 3]])},
        )
    )

    gs.plot_scatter(results, ["x", "y"], (0, 0), highlight_divergences=True, show=False)

    markers = [c for c in plt.gca().collections if c.get_label() == "Divergent"]
    assert len(markers) == 1
    np.testing.assert_array_equal(markers[0].get_offsets(), [[2, 12], [4, 14]])


def test_scatter_combines_hmc_and_nuts_but_not_unrelated_kernel_errors():
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])},
            {
                "nuts": nuts_info([[0, 0, 0, 1]]),
                "hmc": HMCTransitionInfo(
                    error_code=np.array([[0, 1, 0, 0]]),
                    acceptance_prob=np.ones((1, 4)),
                    position_moved=np.ones((1, 4)),
                    divergent=np.array([[False, True, False, False]]),  # ty: ignore[invalid-argument-type]
                ),
                "other": DefaultTransitionInfo(
                    error_code=np.array([[0, 0, 1, 0]]),
                    acceptance_prob=np.ones((1, 4)),
                    position_moved=np.ones((1, 4)),
                ),
            },
        )
    )

    gs.plot_scatter(results, ["x", "y"], (0, 0), highlight_divergences=True, show=False)

    markers = next(c for c in plt.gca().collections if c.get_label() == "Divergent")
    np.testing.assert_array_equal(markers.get_offsets(), [[2, 12], [4, 14]])


@pytest.fixture
def thinned_results():
    epochs = []
    for kind, thinning, values, codes in [
        (EpochType.INITIAL_VALUES, 1, [[-1], [-11], [-21]], None),
        (
            EpochType.BURNIN,
            2,
            [[1, 2, 3], [11, 12, 13], [21, 22, 23]],
            [[1, 1, 1], [0, 1, 0], [1, 1, 1]],
        ),
        (
            EpochType.POSTERIOR,
            2,
            [[4, 5, 6, 7], [14, 15, 16, 17], [24, 25, 26, 27]],
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
        ),
        (
            EpochType.POSTERIOR,
            3,
            [
                [8, 9, 10, 11, 12, 13],
                [18, 19, 20, 21, 22, 23],
                [28, 29, 30, 31, 32, 33],
            ],
            [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]],
        ),
    ]:
        x = np.array(values)
        epochs.append(
            (
                EpochConfig(kind, x.shape[1], thinning, None),
                {"x": x, "y": x + 100},
                {"nuts": nuts_info(codes)} if codes is not None else None,
            )
        )
    return results_from_epochs(*epochs)


@pytest.mark.parametrize(
    "include_warmup, expected",
    [(False, [[17, 117], [20, 120]]), (True, [[12, 112], [17, 117], [20, 120]])],
)
def test_scatter_aligns_divergences_with_thinned_epochs_and_selected_chains(
    thinned_results, include_warmup, expected
):
    gs.plot_scatter(
        thinned_results,
        ["x", "y"],
        (0, 0),
        chain_indices=[2, 1],
        max_chains=1,
        include_warmup=include_warmup,
        highlight_divergences=True,
        show=False,
    )

    markers = next(c for c in plt.gca().collections if c.get_label() == "Divergent")
    np.testing.assert_array_equal(markers.get_offsets(), expected)


def test_pairs_marks_both_scatter_triangles_with_one_shared_legend(thinned_results):
    gs.plot_pairs(
        thinned_results,
        params=["x", "y"],
        chain_indices=[2, 1],
        max_chains=1,
        diag_kind="hist",
        highlight_divergences=True,
        show=False,
    )

    figure = plt.gcf()
    markers = [
        c
        for ax in figure.axes
        for c in ax.collections
        if isinstance(c, PathCollection) and c.get_label() == "Divergent"
    ]
    assert len(markers) == 2
    np.testing.assert_array_equal(markers[0].get_offsets(), [[117, 17], [120, 20]])
    np.testing.assert_array_equal(markers[1].get_offsets(), [[17, 117], [20, 120]])
    assert len(figure.legends) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "1",
        "Divergent",
    ]


@pytest.mark.parametrize("params", [["x"], ["x", "y"]])
@pytest.mark.parametrize("diag_kind", ["hist", "kde"])
def test_pair_diagonals_show_one_combined_divergence_rug(
    thinned_results, params, diag_kind
):
    gs.plot_pairs(
        thinned_results,
        params=params,
        chain_indices=[1, 2],
        max_chains=None,
        diag_kind=diag_kind,
        highlight_divergences=True,
        show=False,
    )

    figure = plt.gcf()
    rugs = [
        c
        for ax in figure.axes
        for c in ax.collections
        if isinstance(c, LineCollection) and c.get_label() == "Divergent"
    ]
    assert len(rugs) == len(params)
    np.testing.assert_array_equal(
        np.asarray(rugs[0].get_segments())[:, 0, 0], [17, 20, 25, 27, 30, 33]
    )
    if len(params) == 2:
        np.testing.assert_array_equal(
            np.asarray(rugs[1].get_segments())[:, 0, 0], [117, 120, 125, 127, 130, 133]
        )
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "1",
        "2",
        "Divergent",
    ]


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
def test_highlighting_requires_sampling_results(plot):
    samples = {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])}
    with pytest.raises(TypeError, match="SamplingResults"):
        plot(
            samples,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=True,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize(
    "missing", ["epoch", "info", "supported_kernel", "initial_only"]
)
def test_highlighting_rejects_unavailable_diagnostics(plot, missing):
    infos = (
        None
        if missing != "supported_kernel"
        else {
            "other": DefaultTransitionInfo(
                error_code=np.ones((1, 1)),
                acceptance_prob=np.ones((1, 1)),
                position_moved=np.ones((1, 1)),
            )
        }
    )
    kind = (
        EpochType.INITIAL_VALUES if missing == "initial_only" else EpochType.POSTERIOR
    )
    results = results_from_epochs(
        (
            EpochConfig(kind, 1, 1, None),
            {"x": np.array([[1]]), "y": np.array([[11]])},
            infos,
        )
    )
    if missing == "epoch":
        results.transition_infos = EpochChainManager()

    with pytest.raises(ValueError, match="HMC or NUTS transition information"):
        plot(
            results,
            params=["x", "y"],
            param_indices=(0, 0),
            include_warmup=True,
            highlight_divergences=True,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("info_shape", [(2, 4), (1, 5), (1, 4, 1), (1, 2)])
def test_highlighting_rejects_diagnostics_that_do_not_align_with_draws(
    plot, info_shape
):
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])},
            {"nuts": nuts_info(np.ones(info_shape, dtype=int))},
        )
    )
    with pytest.raises(ValueError, match="align"):
        plot(
            results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=True,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
def test_highlighting_does_not_silently_omit_a_known_hmc_kernel(plot):
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])},
            {"nuts": nuts_info([[0, 0, 0, 0]])},
        ),
        kernel_classes={"nuts": gs.NUTSKernel, "hmc": gs.HMCKernel},
    )
    with pytest.raises(ValueError, match="HMC or NUTS transition information"):
        plot(
            results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=True,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("case", ["zero", "unselected", "disabled", "dictionary"])
def test_ordinary_plots_have_no_divergence_markers_or_legend(plot, case):
    samples = {
        "x": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "y": np.array([[11, 12, 13, 14], [15, 16, 17, 18]]),
    }
    codes = (
        [[0, 0, 0, 0], [0, 0, 0, 0]] if case == "zero" else [[1, 1, 1, 1], [0, 0, 0, 0]]
    )
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            samples,
            {"nuts": nuts_info(codes)},
        )
    )
    kwargs = {"highlight_divergences": True} if case in ["zero", "unselected"] else {}
    plot(
        samples if case == "dictionary" else results,
        params=["x", "y"],
        param_indices=(0, 0),
        chain_indices=[1] if case == "unselected" else None,
        show=False,
        **kwargs,
    )

    figure = plt.gcf()
    assert all(
        c.get_label() != "Divergent" for ax in figure.axes for c in ax.collections
    )
    legends = figure.legends + [ax.get_legend() for ax in figure.axes]
    assert all(
        text.get_text() != "Divergent"
        for legend in legends
        if legend is not None
        for text in legend.get_texts()
    )


def test_scatter_highlights_selected_vector_components():
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {
                "beta": np.array(
                    [[[1, 101, 11], [2, 102, 12], [3, 103, 13], [4, 104, 14]]]
                )
            },
            {"nuts": nuts_info([[0, 1, 0, 1]])},
        )
    )
    gs.plot_scatter(
        results,
        params=["beta"],
        param_indices=(2, 0),
        highlight_divergences=True,
        show=False,
    )

    axis = plt.gca()
    markers = next(c for c in axis.collections if c.get_label() == "Divergent")
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("beta[0]", "beta[2]")
    np.testing.assert_array_equal(markers.get_offsets(), [[2, 12], [4, 14]])
