from collections import deque
from pathlib import Path

import pandas as pd
import numpy as np

import pytest

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import plotly.express as px

from explorica.visualizations import (distplot, boxplot, scatterplot,
                                      heatmap, hexbin, piechart, barchart, mapbox)

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
        result = plot(*input_features) 

    assert len(result.axes.texts) == 1
    
    assert len(result.axes.lines) == 0
    assert len(result.axes.patches) == 0
    assert len(result.axes.collections) == 0
    plt.close(result.figure)

@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_returns_figure_and_axes(plot, mocker):
    """Checks that each plot method returns tuple[Figure, Axes]."""

    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    first_numerical_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    categorical_data = ["A", "A", "A", "A", "B", "B", "C", "C", "C"]
    second_numerical_data = [4, 2, 5, 1, 3, 5, 2, 4, 5]
    input_features = get_input_features(plot, first_numerical_data, categorical_data, second_numerical_data)
    
    result = plot(*input_features)
    assert isinstance(result.figure, Figure)
    assert isinstance(result.axes, Axes)
    plt.close(result.figure)


@pytest.mark.parametrize("plot", ALL_PLOTS)
def test_plot_saves_to_directory(plot, mocker, tmp_path):

    mocker.patch("matplotlib.pyplot.subplots", return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

    first_numerical_data = [1, 2, 3]
    categorical_data = ["A", "B", "C"]
    second_numerical_data = [4, 5, 6]
    input_features = get_input_features(plot, first_numerical_data, categorical_data, second_numerical_data)
    output_path = tmp_path / f"{plot.__name__}.png"
    
    plot(*input_features, directory=str(output_path))
    
    assert output_path.exists()

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

    result = plot(*input_features, 
                   title=title, 
                   xlabel=xlabel, 
                   ylabel=ylabel, 
                   figsize=figsize)
    assert result.figure.get_size_inches().tolist() == list(figsize)

    assert result.axes.get_title() == title
    assert result.axes.get_xlabel() == xlabel
    assert result.axes.get_ylabel() == ylabel

    plt.close(result.figure)


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

    result = plot(*input_features)
    assert isinstance(result.figure, Figure)
    assert isinstance(result.axes, Axes)
    plt.close(result.figure)


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
        result = plot(*input_features, nan_policy=nan_policy)
        assert isinstance(result.figure, Figure)
        assert isinstance(result.axes, Axes)
        plt.close(result.figure)

# tests for visualizations.distplot()

def test_distplot_palette_and_style():
    expected_rgb = SET1_RED_COLOR
    x = [1, 2, 3, 4, 5]

    result = distplot(x, style="whitegrid", palette="Set1", kde=True, opacity=1.0)
    
    assert len(result.axes.patches) > 0, "Distplot should have patches (histogram bars)."
    actual_patch_rgb = result.axes.patches[0].get_facecolor()
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Patches: Expected RGB {expected_rgb}, got {actual_patch_rgb}"

    assert len(result.axes.lines) == 1, "Distplot with kde=True should have one KDE line."
    actual_line_rgb = result.axes.lines[0].get_color()
    assert np.allclose(actual_line_rgb, expected_rgb), f"Lines: Expected RGB {expected_rgb}, got {actual_line_rgb}"
    
    assert result.axes.xaxis.get_gridlines(), "Style 'whitegrid' failed: X-axis grid lines not found."
    assert result.axes.yaxis.get_gridlines(), "Style 'whitegrid' failed: Y-axis grid lines not found."

    plt.close(result.figure)

def test_distplot_kde():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = distplot(x, kde=True)

    n_lines = len(result.axes.lines)
    assert n_lines == 1
    plt.close(result.figure)

    result = distplot(x, kde=False)
    n_lines = len(result.axes.lines)
    assert n_lines == 0
    plt.close(result.figure)

def test_distplot_bins():
    bins = 10
    x = [i for i in range(10)]
    result = distplot(x, bins=bins)

    n_bins = len(result.axes.patches)
    assert n_bins == bins
    plt.close(result.figure)

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

    result = barchart(values, labels, style="whitegrid", palette="Set1", opacity=1.0)

    assert len(result.axes.patches) > 0, "Barchart should have patches (bars)."
    actual_patch_rgb = result.axes.patches[0].get_facecolor()
    
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Barchart Patch: Expected RGB {expected_rgb}, got {actual_patch_rgb}"

    assert result.axes.xaxis.get_gridlines(), "Style 'whitegrid' failed: X-axis grid lines not found."
    assert result.axes.yaxis.get_gridlines(), "Style 'whitegrid' failed: Y-axis grid lines not found."

    plt.close(result.figure)

def test_barchart_sorting_ascending_descending():
    values = [25, 40, 15]
    labels = ['C', 'A', 'B']
    
    # 1. (ascending=True): B(15), C(25), A(40)
    result_ascending = barchart(values, labels, ascending=True)
    sorted_labels_asc = [t.get_text() for t in result_ascending.axes.get_xticklabels()] 
    assert sorted_labels_asc == ['B', 'C', 'A']
    plt.close(result_ascending.figure)

    # 2. (ascending=False): A(40), C(25), B(15)
    result_descending = barchart(values, labels, ascending=False)
    sorted_labels_desc = [t.get_text() for t in result_descending.axes.get_xticklabels()]
    assert sorted_labels_desc == ['A', 'C', 'B']
    plt.close(result_descending.figure)

def test_barchart_horizontal_orientation():
    values = [10, 20]
    labels = ['Cat', 'Dog']
    
    # 1. Vertical (default)
    result_vertical = barchart(values, labels)
    assert [t.get_text() for t in result_vertical.axes.get_xticklabels()] == labels
    plt.close(result_vertical.figure)

    # 2. Horizontal
    result_horizontal = barchart(values, labels, horizontal=True)
    assert [t.get_text() for t in result_horizontal.axes.get_yticklabels()] == labels
    plt.close(result_horizontal.figure)

def test_barchart_drops_nan_correctly():
    values = [10, 20, np.nan, 40, 50]
    labels = ['A', 'B', 'C', 'D', 'E']

    result = barchart(values, labels, nan_policy='drop', horizontal=True, ascending=None)
    
    assert len(result.axes.patches) == 4

    final_labels = [t.get_text() for t in result.axes.get_yticklabels()]
    assert 'C' not in final_labels
    assert len(final_labels) == 4
    plt.close(result.figure)

def test_barchart_length_mismatch_raises_error():
    values = [10, 20, 30]
    labels = ['A', 'B'] # Mismatch
    
    with pytest.raises(ValueError):
        barchart(values, labels)

# tests for visualizations.piechart()

def test_piechart_legend_visibility():
    # With legend
    result_legend = piechart([1, 2, 3], ['A', 'B', 'C'], show_legend=True)
    assert result_legend.axes.get_legend() is not None
    plt.close(result_legend.figure)

    # Without legend
    result_no_legend = piechart([1, 2, 3], ['A', 'B', 'C'], show_legend=False)
    assert result_no_legend.axes.get_legend() is None
    plt.close(result_no_legend.figure)

def test_piechart_labels_visibility():
    # With labels
    result_labels = piechart([1, 2, 3], ['A', 'B', 'C'], show_labels=True)
    all_texts_1 = [t for t in result_labels.axes.get_children() if isinstance(t, plt.Text) and t.get_visible() and t.get_text()]

    assert len(all_texts_1) == 6, f"Expected 6 text objects (labels+values), got {len(all_texts_1)}"
    plt.close(result_labels.figure)
    
    # Without labels
    result_no_labels = piechart([1, 2, 3], ['A', 'B', 'C'], show_labels=False)
    all_texts_2 = [t for t in result_no_labels.axes.get_children() if isinstance(t, plt.Text) and t.get_visible() and t.get_text()]
    assert len(all_texts_2) == 3, f"Expected 3 text objects (only values), got {len(all_texts_2)}"
    plt.close(result_no_labels.figure)


def test_piechart_palette():
    """Color palette usage test."""
    expected_rgb = SET1_RED_COLOR
    # With palette
    result_palette = piechart(
        [1, 2, 3], 
        ['A', 'B', 'C'], 
        palette="Set1"
    )
    actual_patch_rgb = result_palette.axes.patches[0].get_facecolor()
    assert np.allclose(actual_patch_rgb, expected_rgb), f"Piechart Patch: Expected RGB {expected_rgb}, got {actual_patch_rgb}"  
    plt.close(result_palette.figure)

    # With different palette
    result_no_palette = piechart([1, 2, 3], ['A', 'B', 'C'], palette="viridis")
    actual_patch_rgb = result_no_palette.axes.patches[0].get_facecolor()
    assert not np.allclose(actual_patch_rgb, expected_rgb)   
    plt.close(result_no_palette.figure)

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
    result_percent = piechart(data, categories, autopct_method="percent", show_labels=False)
    texts_percent = get_autotexts(result_percent.axes)
    # Expected pcts: 16.7%, 33.3%, 50.0%
    assert sorted(texts_percent) == ['16.7%', '33.3%', '50.0%']
    plt.close(result_percent.figure)

    # 2. Method 'value'
    result_value = piechart(data, categories, autopct_method="value", show_labels=False)
    texts_value = get_autotexts(result_value.axes)
    # Expected values: 10, 20, 30
    assert sorted(texts_value) == ['10', '20', '30']
    plt.close(result_value.figure)

    # 3. Method 'both'
    result_both = piechart(data, categories, autopct_method="both", show_labels=False)
    texts_both = get_autotexts(result_both.axes)
    # Expected values (pct\n(value)): 16.7%\n(10), 20 (33.3%), 30 (50.0%)
    expected_both = ['16.7%\n(10)', '33.3%\n(20)', '50.0%\n(30)']
    assert sorted(texts_both) == sorted(expected_both)
    plt.close(result_both.figure)

def test_piechart_invalid_autopct_method_raises_error():
    with pytest.raises(ValueError, match="Unsupported method"):
        piechart([1, 2], ['A', 'B'], autopct_method="invalid")


def test_piechart_nan_policy_drop():
    data = [10, 20, np.nan, 40]
    categories = ['A', 'B', 'C', 'D']
    
    result = piechart(data, categories, nan_policy='drop')

    assert len(result.axes.patches) == 3
    plt.close(result.figure)

def test_piechart_length_mismatch_raises_error():
    with pytest.raises(ValueError, match="must match length"):
        piechart([10, 20, 30], ['A', 'B']) # Mismatch

def test_piechart_title_setting():
    title = "My Pie Chart Title"
    data = [1, 1, 1]
    categories = ['a', 'b', 'c']
    
    result = piechart(data, categories, title=title)
    
    assert result.axes.get_title() == title
    plt.close(result.figure)

def test_piechart_xlabel_ylabel_setting():
    xlabel = "Count"
    ylabel = "Categories"
    data = [1, 1, 1]
    categories = ['a', 'b', 'c']

    result = piechart(data, categories, xlabel=xlabel, ylabel=ylabel)

    assert result.axes.get_xlabel() == xlabel
    assert result.axes.get_ylabel() == ylabel
    plt.close(result.figure)

# tests for visualizations.boxplot()

def test_boxplot_single_list_input():
    data = [1.0, 2.0, 3.0]
    result = boxplot(data)

    assert isinstance(result.figure, Figure)
    assert isinstance(result.axes, Axes)

    assert result.axes.get_title() == ""
    assert result.axes.get_xlabel() == ""

    assert list(result.figure.get_size_inches()) == [10, 6]


def test_boxplot_kwargs_applied():
    custom_figsize = (8, 4)
    custom_title = "My Custom Plot"
    result = boxplot([1, 2],
                         title=custom_title,
                         xlabel="Data Points",
                         figsize=custom_figsize)

    assert result.axes.get_title() == custom_title
    assert result.axes.get_xlabel() == "Data Points"

    assert list(result.figure.get_size_inches()) == [8, 4]
    plt.close(result.figure)

def test_piechart_empty_data_warns_and_returns_empty_plot():
    with pytest.warns(UserWarning, match="Data for 'piechart' is empty"):
        result = piechart([], [])
    assert len(result.axes.patches) == 0
    assert len(result.axes.texts) == 1
    plt.close(result.figure)

# tests for visualizations.hexbin()

def test_hexbin_simple_input():
    data = [1, 2, 3, 4, 5]
    target = [10, 20, 30, 40, 50]
    
    result = hexbin(data, target)

    assert isinstance(result.figure, Figure)
    assert isinstance(result.axes, Axes)

    assert result.axes.get_title() == ""
    assert result.axes.get_xlabel() == ""

    assert list(result.figure.get_size_inches()) == [10, 6]
    plt.close(result.figure)

def test_hexbin_kwargs_applied(mocker):
    custom_figsize = (12, 8)
    custom_title = "Density Plot"
    custom_gridsize = 50
    custom_opacity = 0.5
    
    data = [1, 2, 3]
    target = [1, 2, 3]
    
    result = hexbin(data, target,
                     title=custom_title,
                     xlabel="X Data",
                     ylabel="Y Data",
                     gridsize=custom_gridsize,
                     opacity=custom_opacity,
                     figsize=custom_figsize)

    assert result.axes.get_title() == custom_title
    assert result.axes.get_xlabel() == "X Data"
    assert result.axes.get_ylabel() == "Y Data"

    assert list(result.figure.get_size_inches()) == list(custom_figsize)
    
    assert len(result.axes.collections) > 0, "`hexbin` must contains PolyCollection for check of alpha."
    
    poly_collection = result.axes.collections[0]

    assert poly_collection.get_alpha() == custom_opacity

    plt.close(result.figure)

def test_hexbin_valueerror_mismatched_lengths():
    data = [1, 2, 3]
    target = [1, 2]
    
    with pytest.raises(ValueError, match = "must match length"):
        hexbin(data, target)


def test_hexbin_valueerror_invalid_gridsize():
    with pytest.raises(ValueError) as excinfo:
        hexbin([1, 2], [1, 2], gridsize=0)
    assert "'gridsize' must be a positive integer." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        hexbin([1, 2], [1, 2], gridsize="large")

    assert "'gridsize' must be a positive integer." in str(excinfo.value)

# tests for visualizations.scatterplot()

def test_scatterplot_basic_output():
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    result = scatterplot(x, y)

    assert isinstance(result.figure, Figure)
    assert isinstance(result.axes, Axes)
    assert len(result.axes.collections) == 1
    plt.close(result.figure)

def test_scatterplot_with_categories():
    x = [1, 2, 3, 4, 5, 6]
    y = [2, 3, 5, 7, 11, 13]
    cat = ["A", "A", "B", "B", "C", "C"]

    result = scatterplot(x, y, category=cat, show_legend=True)

    assert len(result.axes.collections) == len(set(cat))
    legend_texts = [text.get_text() for text in result.axes.get_legend().get_texts()]
    assert set(legend_texts) == set(cat)
    plt.close(result.figure)

def test_scatterplot_linear_trendline():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 2, 4, 6, 8])

    result = scatterplot(x, y, trendline="linear")

    assert any(line.get_linestyle() == '-' or line.get_linestyle() == '--' for line in result.axes.lines)
    trendline_x = result.axes.lines[0].get_xdata()
    assert np.isclose(trendline_x.min(), min(x))
    assert np.isclose(trendline_x.max(), max(x))
    plt.close(result.figure)

def test_scatterplot_polynomial_trendline():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])

    result = scatterplot(x, y, trendline="polynomial", trendline_kws={"degree": 2})

    assert len(result.axes.lines) == 1
    y_pred = result.axes.lines[0].get_ydata()
    assert np.isclose(y_pred[0], 0)
    assert np.isclose(y_pred[-1], 16)
    plt.close(result.figure)

def test_scatterplot_custom_callable_trendline():
    x = np.linspace(0, 4, 5)
    y = np.array([1, 2, 3, 4, 5])

    def custom_line(x):
        return x + 1

    result = scatterplot(x, y, trendline=custom_line)
    line = result.axes.lines[0]
    x_domain = line.get_xdata()
    y_pred = line.get_ydata()

    assert len(x_domain) == 1000
    assert len(y_pred) == 1000
    
    # check several control dots
    idx_check = [0, 500, -1]
    for i in idx_check:
        assert np.isclose(y_pred[i], custom_line(x_domain[i]))
    plt.close(result.figure)

def test_scatterplot_empty_input_warns():
    result = scatterplot([], [])
    assert len(result.axes.texts) == 1
    plt.close(result.figure)

def test_scatterplot_nan_handling_drop():
    x = [1, 2, 3, np.nan, 5]
    y = [2, 4, 6, 8, np.nan]

    result = scatterplot(x, y, nan_policy="drop")
    assert len(result.axes.collections[0].get_offsets()) == 3
    plt.close(result.figure)

def test_scatterplot_nan_handling_raise():
    x = [1, 2, np.nan]
    y = [1, 2, 3]
    with pytest.raises(ValueError):
        scatterplot(x, y, nan_policy="raise")

# tests for visualizations.heatmap()

def test_heatmap_basic_plot():
    data = np.array([[1, 2], [3, 4]])
    result = heatmap(data)
    assert isinstance(result.figure, plt.Figure)
    assert isinstance(result.axes, plt.Axes)
    plt.close(result.figure)

def test_heatmap_empty_data_warning():
    data = np.array([]).reshape(0, 0)
    with pytest.warns(UserWarning, match="empty"):
        result = heatmap(data)
    assert isinstance(result.figure, plt.Figure)
    assert isinstance(result.axes, plt.Axes)
    plt.close(result.figure)

def test_heatmap_plot_kws_override():
    data = np.array([[1, 2], [3, 4]])
    result = heatmap(data, plot_kws={"annot": False, "cbar": False})
    assert len(result.axes.texts) == 0
    plt.close(result.figure)

def test_heatmap_directory(tmp_path):
    data = np.array([[1, 2], [3, 4]])
    heatmap(data, directory=str(tmp_path), title="TestSave")
    files = [f.name for f in Path(tmp_path).iterdir()]
    assert any("heatmap" in f for f in files)

# tests for visualizations.mapbox()

def test_mapbox_default_palette():
    result = mapbox(
    lat=[0, 1, 2],
    lon=[0, 1, 2],
    category=['A', 'B', 'C'],
    opacity=1.0
    )

    default_colors = px.colors.qualitative.Plotly[:3]
    actual_colors = [trace.marker.color for trace in result.figure.data]
    assert actual_colors == default_colors

def test_mapbox_basic():
    lat = [10, 20, 30]
    lon = [100, 110, 120]
    result = mapbox(lat, lon)
    assert result.figure is not None
    assert hasattr(result.figure, "data")
    assert len(result.figure.data) == 1

def test_mapbox_with_category_and_palette():
    lat = [10, 20, 30]
    lon = [100, 110, 120]
    category = ["A", "B", "A"]
    palette = ["#ff0000", "#00ff00"]
    result = mapbox(lat, lon, category=category, palette=palette)
    assert result.figure is not None
    categories_in_fig = (set([d.name for d in result.figure.data])
                         if hasattr(result.figure.data[0], "name") else set(category))
    assert len(categories_in_fig) == len(set(category))

def test_mapbox_empty_dataframe_returns_placeholder():
    lat = []
    lon = []
    result = mapbox(lat, lon)
    assert hasattr(result.figure, "layout")
    assert any("No data" in ann.text for ann in result.figure.layout.annotations)

def test_mapbox_nan_handling_drop():
    lat = [10, None, 30]
    lon = [100, 110, None]
    result = mapbox(lat, lon, nan_policy="drop")
    assert result is not None

def test_mapbox_custom_dimensions():
    lat = [10, 20]
    lon = [100, 110]
    result = mapbox(lat, lon, width=500, height=400)
    assert result.figure.layout.width == 500
    assert result.figure.layout.height == 400
