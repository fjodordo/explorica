from collections import deque

import pandas as pd
import numpy as np

import pytest

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from explorica.visualizations import (distplot, boxplot, scatterplot,
                                      heatmap, hexbin, piechart, barchart)


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
CAT_PLOTS = [piechart, barchart]

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

    mocker.patch("matplotlib.pyplot.subplots",
                 return_value=(EXAMPLE_PLOT[0], EXAMPLE_PLOT[1]))

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