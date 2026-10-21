import warnings

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


def results_from_epochs(*epochs, kernel_classes=None, kernels_by_pos_key=None):
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
        kernels_by_pos_key=Option(
            kernels_by_pos_key
            if kernels_by_pos_key is not None
            else {param: "nuts" for param in epochs[0][1]}
        ),
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


@pytest.fixture
def separate_kernel_results():
    return results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {
                "x": np.array([[1, 2, 3, 4]]),
                "y": np.array([[11, 12, 13, 14]]),
                "z": np.array([[21, 22, 23, 24]]),
                "hidden": np.array([[31, 32, 33, 34]]),
            },
            {
                "x_kernel": nuts_info([[0, 1, 0, 0]]),
                "y_kernel": nuts_info([[0, 0, 0, 0]]),
                "z_kernel": nuts_info([[0, 0, 0, 3]]),
                "hidden_kernel": nuts_info([[1, 0, 0, 0]]),
            },
        ),
        kernels_by_pos_key={
            "x": "x_kernel",
            "y": "y_kernel",
            "z": "z_kernel",
            "hidden": "hidden_kernel",
        },
    )


@pytest.mark.parametrize("scope", ["parameter", "any"])
def test_trace_marks_divergent_iterations_only_in_affected_parameter_panels(
    separate_kernel_results,
    scope,
):
    grid = gs.plot_trace(
        separate_kernel_results,
        params=["x", "y", "z"],
        divergence_scope=scope,
        show=False,
    )
    for label, expected in {"x": [1], "y": [], "z": [3]}.items():
        if scope == "any":
            expected = [0, 1, 3]
        axis = grid.axes_dict[label]
        rugs = [
            c
            for c in axis.collections
            if isinstance(c, LineCollection) and c.get_label() == "Divergent"
        ]
        if expected:
            assert len(rugs) == 1
            segments = np.asarray(rugs[0].get_segments())
            np.testing.assert_array_equal(segments[:, 0, 0], expected)
            np.testing.assert_array_equal(
                segments[:, :, 1], [[0, 0.05]] * len(expected)
            )
        else:
            assert rugs == []
    assert len(grid.figure.legends) == 1
    assert [text.get_text() for text in grid.legend.get_texts()] == ["0", "Divergent"]


@pytest.mark.parametrize("scope, expected", [("parameter", [1]), ("any", [0, 1, 3])])
def test_param_marks_trace_iterations_and_density_values(
    separate_kernel_results, scope, expected
):
    gs.plot_param(separate_kernel_results, "x", divergence_scope=scope, show=False)

    figure = plt.gcf()
    rugs = [
        c
        for c in figure.axes[0].collections
        if isinstance(c, LineCollection) and c.get_label() == "Divergent"
    ]
    assert len(rugs) == 1
    np.testing.assert_array_equal(np.asarray(rugs[0].get_segments())[:, 0, 0], expected)
    density_rugs = [
        c for c in figure.axes[1].collections if c.get_label() == "Divergent"
    ]
    assert len(density_rugs) == 1
    assert isinstance(density_rugs[0], LineCollection)
    expected_values = [2] if scope == "parameter" else [1, 2, 4]
    np.testing.assert_array_equal(
        np.asarray(density_rugs[0].get_segments())[:, 0, 0], expected_values
    )
    assert all(c.get_label() != "Divergent" for c in figure.axes[2].collections)
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "0",
        "Divergent",
    ]


@pytest.mark.parametrize("plot", [gs.plot_trace, gs.plot_density])
def test_omit_divergence_legend_when_only_unaffected_facets_are_displayed(
    separate_kernel_results,
    plot,
):
    grid = plot(
        separate_kernel_results,
        params=["x", "y"],
        col_order=["y"],
        show=False,
    )
    assert not any(
        c.get_label() == "Divergent" for c in grid.axes_dict["y"].collections
    )
    assert "Divergent" not in [text.get_text() for text in grid.legend.get_texts()]


@pytest.mark.parametrize("plot", [gs.plot_trace, gs.plot_param, gs.plot_density])
def test_trace_rugs_follow_selected_vector_components(plot):
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {"beta": np.arange(12).reshape(1, 4, 3)},
            {"nuts": nuts_info([[0, 1, 0, 1]])},
        )
    )
    if plot is not gs.plot_param:
        grid = plot(results, params=["beta"], param_indices=(2, 0), show=False)
        assert set(grid.axes_dict) == {"beta[0]", "beta[2]"}
        panels = (
            [
                (axis, [3, 9] if label == "beta[0]" else [5, 11])
                for label, axis in grid.axes_dict.items()
            ]
            if plot is gs.plot_density
            else [(axis, [1, 3]) for axis in grid.axes_dict.values()]
        )
    else:
        plot(results, "beta", param_index=2, show=False)
        panels = [(plt.gcf().axes[0], [1, 3]), (plt.gcf().axes[1], [5, 11])]
    for axis, expected in panels:
        rug = next(c for c in axis.collections if c.get_label() == "Divergent")
        assert isinstance(rug, LineCollection)
        np.testing.assert_array_equal(np.asarray(rug.get_segments())[:, 0, 0], expected)


@pytest.mark.parametrize("plot", [gs.plot_trace, gs.plot_param, gs.plot_density])
@pytest.mark.parametrize("mode", ["auto", True, False])
@pytest.mark.parametrize("source", ["available", "unaffected", "dictionary", "missing"])
def test_rug_highlighting_modes(separate_kernel_results, plot, mode, source):
    results = separate_kernel_results
    if source == "dictionary":
        results = results.get_posterior_samples()
    elif source == "missing":
        results.kernels_by_pos_key = Option(None)
    param = "y" if source == "unaffected" else "x"
    kwargs = {"param": param} if plot is gs.plot_param else {"params": [param]}
    if mode is True and source in ["dictionary", "missing"]:
        error = TypeError if source == "dictionary" else ValueError
        with pytest.raises(error):
            plot(results, highlight_divergences=mode, show=False, **kwargs)
        return
    if mode == "auto" and source == "missing":
        with pytest.warns(
            UserWarning, match="Divergence highlighting skipped"
        ) as caught:
            plot(results, highlight_divergences=mode, show=False, **kwargs)
        assert len(caught) == 1
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            plot(results, highlight_divergences=mode, show=False, **kwargs)
    figure = plt.gcf()
    rugs = [
        c for ax in figure.axes for c in ax.collections if c.get_label() == "Divergent"
    ]
    expected_count = int(source == "available" and mode is not False)
    assert len(rugs) == expected_count * (2 if plot is gs.plot_param else 1)
    legends = figure.legends + [ax.get_legend() for ax in figure.axes]
    assert (
        sum(
            text.get_text() == "Divergent"
            for legend in legends
            if legend is not None
            for text in legend.get_texts()
        )
        == expected_count
    )


@pytest.mark.parametrize(
    "plot, include_warmup, expected",
    [
        (gs.plot_trace, False, [1, 2]),
        (gs.plot_trace, True, [1, 3, 4]),
        (gs.plot_param, False, [1, 2]),
    ],
)
def test_trace_rugs_align_with_thinning_epochs_and_selected_chains(
    thinned_results, plot, include_warmup, expected
):
    kwargs = (
        {"params": ["x"], "include_warmup": include_warmup}
        if plot is gs.plot_trace
        else {"param": "x"}
    )
    plot(thinned_results, chain_indices=[2, 1], max_chains=1, show=False, **kwargs)
    rugs = [
        c
        for c in plt.gcf().axes[0].collections
        if isinstance(c, LineCollection) and c.get_label() == "Divergent"
    ]
    assert len(rugs) == 1
    np.testing.assert_array_equal(np.asarray(rugs[0].get_segments())[:, 0, 0], expected)


@pytest.mark.parametrize("plot", [gs.plot_trace, gs.plot_param])
def test_combined_trace_rug_preserves_trace_lines_and_limits(thinned_results, plot):
    kwargs = {"params": ["x"]} if plot is gs.plot_trace else {"param": "x"}
    plot(
        thinned_results,
        chain_indices=[1, 2],
        highlight_divergences=False,
        show=False,
        **kwargs,
    )
    axis = plt.gcf().axes[0]
    before = [np.asarray(line.get_xydata()).copy() for line in axis.lines]
    limits = axis.get_xlim(), axis.get_ylim()
    plt.close("all")

    plot(thinned_results, chain_indices=[1, 2], show=False, **kwargs)
    axis = plt.gcf().axes[0]
    assert (axis.get_xlim(), axis.get_ylim()) == limits
    assert len(axis.lines) == len(before)
    for line, expected in zip(axis.lines, before):
        np.testing.assert_array_equal(line.get_xydata(), expected)
    rugs = [
        c
        for c in axis.collections
        if isinstance(c, LineCollection) and c.get_label() == "Divergent"
    ]
    assert len(rugs) == 1
    np.testing.assert_array_equal(
        np.asarray(rugs[0].get_segments())[:, 0, 0], [0, 1, 2, 3]
    )


@pytest.mark.parametrize("plot", [gs.plot_param, gs.plot_density])
@pytest.mark.parametrize(
    "max_chains, expected_values", [(1, [17, 20]), (None, [17, 20, 25, 27, 30, 33])]
)
def test_density_rug_preserves_kde_and_uses_selected_thinned_values(
    thinned_results,
    plot,
    max_chains,
    expected_values,
):
    plot(
        thinned_results,
        "x",
        chain_indices=[2, 1],
        max_chains=max_chains,
        highlight_divergences=False,
        show=False,
    )
    axis_index = 1 if plot is gs.plot_param else 0
    axis = plt.gcf().axes[axis_index]
    before = [np.asarray(line.get_xydata()).copy() for line in axis.lines]
    limits = axis.get_xlim(), axis.get_ylim()
    plt.close("all")

    plot(thinned_results, "x", chain_indices=[2, 1], max_chains=max_chains, show=False)
    axis = plt.gcf().axes[axis_index]
    assert (axis.get_xlim(), axis.get_ylim()) == limits
    assert len(axis.lines) == len(before)
    for line, expected in zip(axis.lines, before):
        np.testing.assert_array_equal(line.get_xydata(), expected)
    rugs = [c for c in axis.collections if c.get_label() == "Divergent"]
    assert len(rugs) == 1
    assert isinstance(rugs[0], LineCollection)
    np.testing.assert_array_equal(
        np.asarray(rugs[0].get_segments())[:, 0, 0], expected_values
    )


@pytest.mark.parametrize("scope", ["parameter", "any"])
def test_density_rugs_mark_values_from_the_relevant_kernels(
    separate_kernel_results, scope
):
    if scope == "any":
        separate_kernel_results.kernels_by_pos_key = Option(None)
    grid = gs.plot_density(
        separate_kernel_results,
        params=["x", "y", "z"],
        divergence_scope=scope,
        show=False,
    )
    expected = (
        {"x": [2], "y": [], "z": [24]}
        if scope == "parameter"
        else {"x": [1, 2, 4], "y": [11, 12, 14], "z": [21, 22, 24]}
    )
    for label, values in expected.items():
        rugs = [
            c for c in grid.axes_dict[label].collections if c.get_label() == "Divergent"
        ]
        assert len(rugs) == int(bool(values))
        if values:
            assert isinstance(rugs[0], LineCollection)
            segments = np.asarray(rugs[0].get_segments())
            np.testing.assert_array_equal(segments[:, 0, 0], values)
            np.testing.assert_array_equal(segments[:, :, 1], [[0, 0.05]] * len(values))
    assert len(grid.figure.legends) == 1
    assert [text.get_text() for text in grid.legend.get_texts()] == ["0", "Divergent"]


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("mode", ["default", "auto"])
@pytest.mark.parametrize("scope", ["parameter", "any"])
def test_auto_highlights_available_divergences(
    separate_kernel_results, plot, mode, scope
):
    kwargs = {} if mode == "default" else {"highlight_divergences": "auto"}
    plot(
        separate_kernel_results,
        params=["x", "y"],
        param_indices=(0, 0),
        divergence_scope=scope,
        show=False,
        **kwargs,
    )
    markers = [
        c
        for ax in plt.gcf().axes
        for c in ax.collections
        if isinstance(c, PathCollection) and c.get_label() == "Divergent"
    ]
    assert len(markers) == (1 if plot is gs.plot_scatter else 2)
    expected = [[2, 12]] if scope == "parameter" else [[1, 11], [2, 12], [4, 14]]
    np.testing.assert_array_equal(markers[-1].get_offsets(), expected)


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("source", ["dictionary", "non_hmc"])
def test_auto_plots_inputs_without_applicable_diagnostics_normally(plot, source):
    samples = {"x": np.array([[1, 2, 3, 4]]), "y": np.array([[11, 12, 13, 14]])}
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            samples,
            {
                "rw": DefaultTransitionInfo(
                    error_code=np.zeros((1, 4)),
                    acceptance_prob=np.ones((1, 4)),
                    position_moved=np.ones((1, 4)),
                )
            },
        ),
        kernel_classes={"rw": gs.RWKernel},
        kernels_by_pos_key={"x": "rw", "y": "rw"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plot(
            samples if source == "dictionary" else results,
            params=["x", "y"],
            param_indices=(0, 0),
            show=False,
        )
    assert all(
        c.get_label() != "Divergent" for ax in plt.gcf().axes for c in ax.collections
    )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("missing", ["mapping", "info", "incomplete", "shape"])
def test_auto_warns_and_skips_unusable_diagnostics(
    separate_kernel_results, plot, missing
):
    results = separate_kernel_results
    if missing == "mapping":
        results.kernels_by_pos_key = Option(None)
    elif missing == "info":
        results.transition_infos = EpochChainManager()
    elif missing == "incomplete":
        results.kernel_classes = Option({"missing_hmc": gs.HMCKernel})
    else:
        infos = results.get_posterior_transition_infos()
        infos["x_kernel"] = nuts_info([[0, 1, 0, 0, 0]])
        results = results_from_epochs(
            (
                EpochConfig(EpochType.POSTERIOR, 4, 1, None),
                results.get_posterior_samples(),
                infos,
            ),
            kernels_by_pos_key=results.get_kernels_by_pos_key(),
        )
    with pytest.warns(UserWarning, match="Divergence highlighting skipped") as caught:
        plot(results, params=["x", "y"], param_indices=(0, 0), show=False)
    assert len(caught) == 1
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


def test_pairs_marks_only_kernels_updating_each_panels_parameters(
    separate_kernel_results,
):
    gs.plot_pairs(
        separate_kernel_results,
        params=["x", "y", "z"],
        diag_kind="hist",
        highlight_divergences=True,
        show=False,
    )

    axes = plt.gcf().axes
    expected = {
        1: [[12, 2]],
        2: [[22, 2], [24, 4]],
        3: [[2, 12]],
        5: [[24, 14]],
        6: [[2, 22], [4, 24]],
        7: [[14, 24]],
    }
    for index, coordinates in expected.items():
        markers = [c for c in axes[index].collections if c.get_label() == "Divergent"]
        assert len(markers) == 1
        np.testing.assert_array_equal(markers[0].get_offsets(), coordinates)
    for axis, values in zip(axes[9:], [[2], [], [24]]):
        rugs = [
            c
            for c in axis.collections
            if isinstance(c, LineCollection) and c.get_label() == "Divergent"
        ]
        if values:
            assert len(rugs) == 1
            np.testing.assert_array_equal(
                np.asarray(rugs[0].get_segments())[:, 0, 0], values
            )
        else:
            assert rugs == []


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("mode", [True, "auto"])
def test_any_scope_marks_divergences_from_unplotted_kernels(
    separate_kernel_results, plot, mode
):
    # Any-kernel scope does not need parameter-to-kernel metadata.
    separate_kernel_results.kernels_by_pos_key = Option(None)
    plot(
        separate_kernel_results,
        params=["x", "y"],
        param_indices=(0, 0),
        highlight_divergences=mode,
        divergence_scope="any",
        show=False,
    )

    figure = plt.gcf()
    markers = [
        c
        for ax in figure.axes
        for c in ax.collections
        if isinstance(c, PathCollection) and c.get_label() == "Divergent"
    ]
    assert len(markers) == (1 if plot is gs.plot_scatter else 2)
    np.testing.assert_array_equal(
        markers[-1].get_offsets(), [[1, 11], [2, 12], [4, 14]]
    )
    if plot is gs.plot_pairs:
        rugs = [
            c
            for ax in figure.axes
            for c in ax.collections
            if isinstance(c, LineCollection) and c.get_label() == "Divergent"
        ]
        assert len(rugs) == 2
        np.testing.assert_array_equal(
            np.asarray(rugs[0].get_segments())[:, 0, 0], [1, 2, 4]
        )
        np.testing.assert_array_equal(
            np.asarray(rugs[1].get_segments())[:, 0, 0], [11, 12, 14]
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("mode", [True, "auto"])
def test_highlighting_rejects_unknown_divergence_scope(
    separate_kernel_results, plot, mode
):
    with pytest.raises(ValueError, match="divergence_scope"):
        plot(
            separate_kernel_results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=mode,
            divergence_scope="unknown",
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("mode", ["yes", None, 1])
def test_highlighting_rejects_invalid_modes(separate_kernel_results, plot, mode):
    with pytest.raises(ValueError, match="highlight_divergences"):
        plot(
            separate_kernel_results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=mode,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
def test_parameter_scope_requires_kernel_associations(separate_kernel_results, plot):
    separate_kernel_results.kernels_by_pos_key = Option(None)
    with pytest.raises(ValueError, match="position-kernel associations"):
        plot(
            separate_kernel_results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=True,
            show=False,
        )


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
def test_parameter_scope_rejects_missing_mapped_kernel_diagnostics(
    separate_kernel_results, plot
):
    separate_kernel_results.kernels_by_pos_key = Option(
        {"x": "missing_kernel", "y": "y_kernel"}
    )
    with pytest.raises(ValueError, match="transition information"):
        plot(
            separate_kernel_results,
            params=["x", "y"],
            param_indices=(0, 0),
            highlight_divergences=True,
            show=False,
        )


def test_scatter_ignores_divergences_from_unplotted_kernels(separate_kernel_results):
    gs.plot_scatter(
        separate_kernel_results,
        params=["x", "y"],
        param_indices=(0, 0),
        highlight_divergences=True,
        show=False,
    )
    markers = [c for c in plt.gca().collections if c.get_label() == "Divergent"]
    assert len(markers) == 1
    np.testing.assert_array_equal(markers[0].get_offsets(), [[2, 12]])


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
@pytest.mark.parametrize("association", ["nondivergent", "unmapped", "other_kernel"])
def test_unaffected_parameters_have_no_markers_or_divergent_legend(
    separate_kernel_results, plot, association
):
    if association == "unmapped":
        separate_kernel_results.kernels_by_pos_key = Option({})
    else:
        separate_kernel_results.kernels_by_pos_key = Option(
            {"x": "y_kernel", "y": "y_kernel"}
        )
        if association == "other_kernel":
            infos = separate_kernel_results.get_posterior_transition_infos()
            infos["y_kernel"] = DefaultTransitionInfo(
                error_code=np.ones((1, 4)),
                acceptance_prob=np.ones((1, 4)),
                position_moved=np.ones((1, 4)),
            )
            samples = separate_kernel_results.get_posterior_samples()
            separate_kernel_results = results_from_epochs(
                (EpochConfig(EpochType.POSTERIOR, 4, 1, None), samples, infos),
                kernels_by_pos_key={"x": "y_kernel", "y": "y_kernel"},
            )
    plot(
        separate_kernel_results,
        params=["x", "y"] if plot is gs.plot_scatter else ["y"],
        param_indices=(0, 0),
        highlight_divergences=True,
        show=False,
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
        ),
        kernels_by_pos_key={"x": "nuts", "y": "hmc"},
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
@pytest.mark.parametrize(
    "case", ["zero", "unselected", "disabled", "disabled_missing", "dictionary"]
)
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
    if case == "disabled_missing":
        results.kernels_by_pos_key = Option(None)
        results.transition_infos = EpochChainManager()
    kwargs = {"highlight_divergences": False} if case.startswith("disabled") else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
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


@pytest.mark.parametrize("plot", [gs.plot_scatter, gs.plot_pairs])
def test_highlighting_uses_the_joint_kernel_for_selected_vector_components(plot):
    results = results_from_epochs(
        (
            EpochConfig(EpochType.POSTERIOR, 4, 1, None),
            {
                "beta": np.array(
                    [[[1, 101, 11], [2, 102, 12], [3, 103, 13], [4, 104, 14]]]
                )
            },
            {
                "nuts": nuts_info([[0, 1, 0, 1]]),
                "unrelated": nuts_info([[1, 0, 1, 0]]),
            },
        )
    )
    plot(
        results,
        params=["beta"],
        param_indices=(2, 0),
        highlight_divergences=True,
        show=False,
    )

    axis = plt.gca() if plot is gs.plot_scatter else plt.gcf().axes[2]
    markers = next(c for c in axis.collections if c.get_label() == "Divergent")
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("beta[0]", "beta[2]")
    np.testing.assert_array_equal(markers.get_offsets(), [[2, 12], [4, 14]])
