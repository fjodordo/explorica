from collections import deque

import pandas as pd
import numpy as np

import pytest

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from explorica.visualizations import (distplot, boxplot, scatterplot,
                                      heatmap, hexbin, piechart, barchart, mapbox)
from explorica.visualizations.plots import _make_autopct

EXAMPLE_PLOT = plt.subplots(figsize=(10, 6))
plt.plot(ax=EXAMPLE_PLOT[1])

SET1_RED_COLOR = tuple(list(sns.color_palette("Set1")[0]) + [1.0])

PLOT_PROPERTIES = {
    "distplot": {"req_categorical_input": False, "req_second_numerical_input": False,
                 "supports_style": True, "supports_palette": True, "supports_cmap": False},
    "boxplot": {"req_categorical_input": False, "req_second_numerical_input": False,
                "supports_style": False, "supports_palette": False, "supports_cmap": False},
    "hexbin": {"req_categorical_input": False, "req_second_numerical_input": True,
               "supports_style": False, "supports_palette": False, "supports_cmap": False},
    "piechart": {"req_categorical_input": True, "req_second_numerical_input": False,
                 "supports_style": False, "supports_palette": False, "supports_cmap": False},
    "barchart": {"req_categorical_input": True, "req_second_numerical_input": False,
                 "supports_style": True, "supports_palette": True, "supports_cmap": False},
    "scatterplot": {"req_categorical_input": False, "req_second_numerical_input": True,
                    "supports_style": False, "supports_palette": False, "supports_cmap": False}
}

ALL_PLOTS = [distplot, boxplot, hexbin, piechart, barchart, scatterplot]
CAT_PLOTS = [boxplot, piechart, barchart]

TEST_SEQUENCES = [
    ([1, 2, 3], ["a", "b", "c"]),                                        # list[int/float], list[str]
    (np.array([1.1, 2.2, 3.3], dtype=np.float32), np.array(["x", "y", "z"], dtype=object)), # np.array, np.array[str]
    (pd.Series([1, 2, 3], name="numbers"), pd.Series(["cat", "dog", "mouse"], dtype="category")), # pd.Series, pd.Series[category]
    (pd.DataFrame({"val": [1, 2, 3]}), pd.DataFrame({"cat": ["x", "y", "z"]})), # pd.DataFrame (1 column), pd.DataFrame (1 column)
    (deque([7, 8, 9]), list("LMN")),                                     # deque, list
]

def get_input_features(plot,  first_numerical_data, categorical_data, second_numerical_data):
    """Internal helper for defining parametrization args."""
    plot_name = plot.__name__
    if PLOT_PROPERTIES[plot_name]["req_categorical_input"]:
        return [first_numerical_data, categorical_data]
    elif PLOT_PROPERTIES[plot_name]["req_second_numerical_input"]:
        return [first_numerical_data, second_numerical_data]
    else:
        return [first_numerical_data]

# API contracts tests

@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_empty_input(plot):
    empty_first = []
    empty_second = []

    input_features = get_input_features(plot, empty_first, empty_second, empty_second)

    expected_warning_message = f"Data for '{plot.__name__}' is empty. Returning empty figure with message."
    
    with pytest.warns(UserWarning, match=expected_warning_message):
        fig, ax = plot(*input_features) 

    assert len(ax.texts) == 1
    
    assert len(ax.lines) == 0
    assert len(ax.patches) == 0
    assert len(ax.collections) == 0

@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_returns_figure_and_axes(plot, tmp_path, mocker):
    """Checks that each plot method returns tuple[Figure, Axes]."""

    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    first_numerical_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    categorical_data = ["A", "A", "A", "A", "B", "B", "C", "C", "C"]
    second_numerical_data = [4, 2, 5, 1, 3, 5, 2, 4, 5]
    input_features = get_input_features(plot, first_numerical_data, categorical_data, second_numerical_data)
    
    fig, ax = plot(*input_features)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_saves_to_directory(plot, mocker, tmp_path):

    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    first_numerical_data = [1, 2, 3]
    categorical_data = ["A", "B", "C"]
    second_numerical_data = [4, 5, 6]
    input_features = get_input_features(plot, first_numerical_data, categorical_data, second_numerical_data)
    output_path = tmp_path / f"{plot.__name__}.png"
    
    fig, ax = plot(*input_features, directory=str(output_path))
    
    assert output_path.exists()
    plt.close(fig)

@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_metadata_and_figsize(plot):
    title = "Test Plot Title"
    xlabel = "Test X-Label"
    ylabel = "Test Y-Label"
    figsize = (12, 8)

    first_numerical_data = [1, 2, 3]
    categorical_data = ["A", "B", "C"]
    second_numerical_data = [4, 5, 6]
    input_features = get_input_features(plot, first_numerical_data, categorical_data, second_numerical_data)

    fig, ax = plot(*input_features, 
                   title=title, 
                   xlabel=xlabel, 
                   ylabel=ylabel, 
                   figsize=figsize)
    assert fig.get_size_inches().tolist() == list(figsize)

    assert ax.get_title() == title
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel

    plt.close(fig)


@pytest.mark.parametrize("plot", ALL_PLOTS)
@pytest.mark.parametrize("seq_pair", TEST_SEQUENCES)
def test_plot_different_sequences_and_dtypes(plot, seq_pair, mocker):

    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    val_seq, cat_seq = seq_pair
    if PLOT_PROPERTIES[plot.__name__]["req_categorical_input"]:
        input_features = [val_seq, cat_seq]
    elif PLOT_PROPERTIES[plot.__name__]["req_second_numerical_input"]:
        input_features = [val_seq, val_seq] 
    else:
        input_features = [val_seq]

    fig, ax = plot(*input_features)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


@pytest.mark.parametrize("plot", ALL_PLOTS)
@pytest.mark.parametrize("nan_policy, should_raise", [
    ('raise', True),
    ('drop', False)
])
def test_plot_nan_policy(plot, nan_policy, should_raise, mocker):
    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    first_numerical_data_with_nan = [1, 2, 3, 4, pd.NA, 6]
    second_numerical_data_with_none = [1, 3, 5, 7, 9, None]
    categorical_data_with_nan = ["A", "A", "A", "B", np.nan, "B"]

    input_features = get_input_features(plot, 
                                        first_numerical_data_with_nan, 
                                        categorical_data_with_nan, 
                                        second_numerical_data_with_none)
    if should_raise:
        with pytest.raises(ValueError):
            plot(*input_features, nan_policy=nan_policy)
    else:
        fig, ax = plot(*input_features, nan_policy=nan_policy)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

# tests for visualizations.distplot()

def test_distplot_palette_and_style():
    expected_rgb = SET1_RED_COLOR
    x = [1, 2, 3, 4, 5]

    fig, ax = distplot(x, style="whitegrid", palette="Set1", kde=True, opacity=1.0)
    
    assert len(ax.patches) > 0, "Distplot should have patches (histogram bars)."
    actual_patch_rgb = ax.patches[0].get_facecolor()
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Patches: Expected RGB {expected_rgb}, got {actual_patch_rgb}"

    assert len(ax.lines) == 1, "Distplot with kde=True should have one KDE line."
    actual_line_rgb = ax.lines[0].get_color()
    assert np.allclose(actual_line_rgb, expected_rgb), f"Lines: Expected RGB {expected_rgb}, got {actual_line_rgb}"
    
    assert ax.xaxis.get_gridlines(), "Style 'whitegrid' failed: X-axis grid lines not found."
    assert ax.yaxis.get_gridlines(), "Style 'whitegrid' failed: Y-axis grid lines not found."

    plt.close(fig)

def test_distplot_kde():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, ax = distplot(x, kde=True)

    n_lines = len(ax.lines)
    assert n_lines == 1
    plt.close(fig)

    fig, ax = distplot(x, kde=False)
    n_lines = len(ax.lines)
    assert n_lines == 0
    plt.close(fig)

def test_distplot_bins():
    bins = 10
    x = [i for i in range(10)]
    fig, ax = distplot(x, bins=bins)

    n_bins = len(ax.patches)
    assert n_bins == bins
    plt.close(fig)

def test_distplot_invalid_bins_raises_error():
    x = [1, 2, 3]
    with pytest.raises(ValueError):
        distplot(x, bins=0)
    with pytest.raises(ValueError):
        distplot(x, bins=-5)

# tests for visualizations.barchart()

def test_barchart_palette_and_style():
    expected_rgb = SET1_RED_COLOR
    values = [10, 20, 30]
    labels = ['A', 'B', 'C']

    fig, ax = barchart(values, labels, style="whitegrid", palette="Set1", opacity=1.0)

    assert len(ax.patches) > 0, "Barchart should have patches (bars)."
    actual_patch_rgb = ax.patches[0].get_facecolor()
    
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Barchart Patch: Expected RGB {expected_rgb}, got {actual_patch_rgb}"

    assert ax.xaxis.get_gridlines(), "Style 'whitegrid' failed: X-axis grid lines not found."
    assert ax.yaxis.get_gridlines(), "Style 'whitegrid' failed: Y-axis grid lines not found."

    plt.close(fig)

def test_barchart_sorting_ascending_descending():
    values = [25, 40, 15]
    labels = ['C', 'A', 'B']
    
    # 1. (ascending=True): B(15), C(25), A(40)
    fig_asc, ax_asc = barchart(values, labels, ascending=True)
    sorted_labels_asc = [t.get_text() for t in ax_asc.get_xticklabels()] 
    assert sorted_labels_asc == ['B', 'C', 'A']
    plt.close(fig_asc)

    # 2. (ascending=False): A(40), C(25), B(15)
    fig_desc, ax_desc = barchart(values, labels, ascending=False)
    sorted_labels_desc = [t.get_text() for t in ax_desc.get_xticklabels()]
    assert sorted_labels_desc == ['A', 'C', 'B']
    plt.close(fig_desc)

def test_barchart_horizontal_orientation():
    values = [10, 20]
    labels = ['Cat', 'Dog']
    
    # 1. Vertical (default)
    fig_vert, ax_vert = barchart(values, labels)
    assert [t.get_text() for t in ax_vert.get_xticklabels()] == labels
    plt.close(fig_vert)

    # 2. Horizontal
    fig_horiz, ax_horiz = barchart(values, labels, horizontal=True)
    assert [t.get_text() for t in ax_horiz.get_yticklabels()] == labels
    plt.close(fig_horiz)

def test_barchart_drops_nan_correctly():
    values = [10, 20, np.nan, 40, 50]
    labels = ['A', 'B', 'C', 'D', 'E']

    fig, ax = barchart(values, labels, nan_policy='drop', horizontal=True, ascending=None)
    
    assert len(ax.patches) == 4

    final_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert 'C' not in final_labels
    assert len(final_labels) == 4
    plt.close(fig)

def test_barchart_length_mismatch_raises_error():
    values = [10, 20, 30]
    labels = ['A', 'B'] # Mismatch
    
    with pytest.raises(ValueError):
        barchart(values, labels)

# tests for visualizations.piechart()

def test_piechart_legend_visibility():
    # With legend
    fig1, ax1 = piechart([1, 2, 3], ['A', 'B', 'C'], show_legend=True)
    assert ax1.get_legend() is not None
    plt.close(fig1)

    # Without legend
    fig2, ax2 = piechart([1, 2, 3], ['A', 'B', 'C'], show_legend=False)
    assert ax2.get_legend() is None
    plt.close(fig2)

def test_piechart_labels_visibility():
    # With labels
    fig1, ax1 = piechart([1, 2, 3], ['A', 'B', 'C'], show_labels=True)
    all_texts_1 = [t for t in ax1.get_children() if isinstance(t, plt.Text) and t.get_visible() and t.get_text()]

    assert len(all_texts_1) == 6, f"Expected 6 text objects (labels+values), got {len(all_texts_1)}"
    plt.close(fig1)
    
    # Without labels
    fig2, ax2 = piechart([1, 2, 3], ['A', 'B', 'C'], show_labels=False)
    all_texts_2 = [t for t in ax2.get_children() if isinstance(t, plt.Text) and t.get_visible() and t.get_text()]
    assert len(all_texts_2) == 3, f"Expected 3 text objects (only values), got {len(all_texts_2)}"
    plt.close(fig2)


def test_piechart_palette():
    """Color palette usage test."""
    expected_rgb = SET1_RED_COLOR
    # With palette
    fig1, ax1 = piechart(
        [1, 2, 3], 
        ['A', 'B', 'C'], 
        palette="Set1"
    )
    actual_patch_rgb = ax1.patches[0].get_facecolor()
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Piechart Patch: Expected RGB {expected_rgb}, got {actual_patch_rgb}"  
    plt.close(fig1)
    # With different palette
    fig2, ax2 = piechart([1, 2, 3], ['A', 'B', 'C'], palette="viridis")
    actual_patch_rgb = ax2.patches[0].get_facecolor()
    assert not np.allclose(actual_patch_rgb, expected_rgb)   
    plt.close(fig2)

def test_piechart_autopct_output_content():
    data = [10, 20, 30]
    categories = ['X', 'Y', 'Z']

    def get_autotexts(ax):
        all_texts = [t for t in ax.get_children() if isinstance(t, plt.Text) and t.get_visible() and t.get_text()]
        
        # When show_labels=True, total text objects = 2 * len(data). Autotexts are the second half.
        if len(all_texts) == 2 * len(data):
            return [t.get_text() for t in all_texts[len(data):]]
        
        # When show_labels=False, total text objects = len(data). Autotexts are all of them.
        if len(all_texts) == len(data):
            return [t.get_text() for t in all_texts]
            
        return []


    # 1. Method 'percent'
    fig1, ax1 = piechart(data, categories, autopct_method="percent", show_labels=False)
    texts_percent = get_autotexts(ax1)
    # Expected pcts: 16.7%, 33.3%, 50.0%
    assert sorted(texts_percent) == ['16.7%', '33.3%', '50.0%']
    plt.close(fig1)

    # 2. Method 'value'
    fig2, ax2 = piechart(data, categories, autopct_method="value", show_labels=False)
    texts_value = get_autotexts(ax2)
    # Expected values: 10, 20, 30
    assert sorted(texts_value) == ['10', '20', '30']
    plt.close(fig2)

    # 3. Method 'both'
    fig3, ax3 = piechart(data, categories, autopct_method="both", show_labels=False)
    texts_both = get_autotexts(ax3)
    # Expected values (pct\n(value)): 16.7%\n(10), 20 (33.3%), 30 (50.0%)
    expected_both = ['16.7%\n(10)', '33.3%\n(20)', '50.0%\n(30)']
    assert sorted(texts_both) == sorted(expected_both)
    plt.close(fig3)

def test_piechart_invalid_autopct_method_raises_error():
    with pytest.raises(ValueError, match="Unsupported method"):
        piechart([1, 2], ['A', 'B'], autopct_method="invalid")

def test_piechart_empty_data_warns_and_returns_empty_plot():
    with pytest.warns(UserWarning, match="Data for 'piechart' is empty"):
        fig, ax = piechart([], [])
    assert len(ax.patches) == 0
    assert len(ax.texts) == 1 # Текст, установленный get_empty_plot
    plt.close(fig)

def test_piechart_nan_policy_drop():
    data = [10, 20, np.nan, 40]
    categories = ['A', 'B', 'C', 'D']
    
    fig, ax = piechart(data, categories, nan_policy='drop')

    assert len(ax.patches) == 3
    plt.close(fig)

def test_piechart_length_mismatch_raises_error():
    with pytest.raises(ValueError, match="must match length"):
        piechart([10, 20, 30], ['A', 'B']) # Mismatch

def test_piechart_title_setting():
    title = "My Pie Chart Title"
    data = [1, 1, 1]
    categories = ['a', 'b', 'c']
    
    fig, ax = piechart(data, categories, title=title)
    
    assert ax.get_title() == title
    plt.close(fig)

def test_piechart_xlabel_ylabel_setting():
    xlabel = "Count"
    ylabel = "Categories"
    data = [1, 1, 1]
    categories = ['a', 'b', 'c']
    
    fig, ax = piechart(data, categories, xlabel=xlabel, ylabel=ylabel)
    
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel
    plt.close(fig)