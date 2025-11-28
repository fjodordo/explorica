from datetime import datetime, timedelta
from itertools import chain
from collections import deque

import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from explorica.visualizations import (distplot, boxplot, scatterplot,
                                      heatmap, hexbin, piechart, barchart, mapbox)

plot_properties = {
    "distplot": {
        "req_cat": False,
        "req_num2": False
    },
    "boxplot": {
        "req_cat": False,
        "req_num2": False
    },
    "hexbin": {
        "req_cat": False,
        "req_num2": True
    },
    "piechart": {
        "req_cat": True,
        "req_num2": False
    },
    "barchart": {
        "req_cat": True,
        "req_num2": False
    },
    "scatterplot": {
        "req_cat": False,
        "req_num2": True
    }
}

# API contracts tests

@pytest.mark.parametrize("plot", [distplot,
                                  boxplot,
                                  hexbin,
                                  piechart,
                                  barchart,
                                  scatterplot])
def test_plot_exists(plot):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = ["A", "A", "A", "A", "B", "B", "C", "C", "C"]
    z = [4, 2, 5, 1, 3, 5, 2, 4, 5]
    if plot_properties[plot.__name__]["req_cat"]:
        input_features = [y, x]
    elif plot_properties[plot.__name__]["req_num2"]:
        input_features = [x, z]
    else:
        input_features = [x]
    with patch("matplotlib.pyplot.show") as mock_show:
        plot(*input_features, show_plot=True)
        mock_show.assert_called_once()
    with patch.object(plt.Figure, "savefig") as mock_save:
        plot(*input_features, dir="plot.png", show_plot=False)
        mock_save.assert_called_once_with("plot.png")
    figure = plot(*input_features, show_plot=False, return_plot=True)
    assert isinstance(figure, plt.Figure)

@pytest.mark.parametrize("plot", [distplot,
                                  boxplot,
                                  hexbin,
                                  piechart,
                                  barchart])
def test_plot_different_sequences_and_dtypes(plot):
    test_sequences = [
    ([1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"]),  # list[int], list[str]
    ([1.1, 2.2, 3.3], [10, 20, 30]),               # list[float], list[int]

    # --- numpy ---
    (np.array([1, 2, 3], dtype=np.int64), np.array(["x", "y", "z"], dtype=object)), # np.int64, np.array[str]
    (np.array([0.1, 0.5, 0.9], dtype=np.float32), np.array([pd.Timestamp("2020-01-01"),
                                                           pd.Timestamp("2020-01-02"),
                                                           pd.Timestamp("2020-01-03")],
                                                          dtype="datetime64[ns]")), # np.float32, np.datetime64

    # --- pandas.Series ---
    (pd.Series([1, 2, 3], name="numbers"), pd.Series(["cat", "dog", "mouse"], dtype="category")), # Series[int], Series[category]
    (pd.Series([0.5, 1.5, 2.5], dtype="float64"), pd.Series(pd.date_range("2021-01-01", periods=3, freq="D"))), # Series[float], Series[datetime]

    # --- pandas.DataFrame (одна колонка) ---
    (pd.DataFrame({"val": [1, 2, 3]}), pd.DataFrame({"cat": ["x", "y", "z"]})), # DataFrame[int], DataFrame[str]
    (pd.DataFrame({"val": np.array([0.1, 0.2, 0.3], dtype=np.float64)}),
     pd.DataFrame({"cat": pd.Series(["L", "M", "N"], dtype="category")})),     # DataFrame[float], DataFrame[category]

    # --- dict -> single array ---
    ({"nums": np.array([10, 20, 30])}, {"cats": np.array(["AA", "BB", "CC"])}), # dict[str, np.array]
    ({"nums": [7, 8, 9]}, {"cats": pd.Series([1, 2, 3], dtype="category")}),    # dict[str, list[int] / category series]
    ]
    for seq_pair in test_sequences:
        if plot_properties[plot.__name__]["req_cat"]:
            input_features = [seq_pair[1], seq_pair[0]]
        elif plot_properties[plot.__name__]["req_num2"]:
            input_features = [seq_pair[0], seq_pair[0]]
        else:
            input_features = [seq_pair[0]]
        figure = plot(*input_features, show_plot=False, return_plot = True)
        assert isinstance(figure, plt.Figure)
        plt.close(figure)

@pytest.mark.parametrize("plot", [distplot,
                                  boxplot,
                                  hexbin,
                                  piechart,
                                  barchart,
                                  scatterplot])
def test_plot_input_contains_nans(plot):
    array_num = [1, 2, 3, 4, 5, 6, 7, 8, pd.NA, 10]
    array_num2 = [1, 3, 5, 7, None, 1, 4, 19, 1, 0]
    array_cat = ["A", "A", "A", "B", np.nan, "B", "B", "C", "C", "C"]
    if plot_properties[plot.__name__]["req_cat"]:
        input_features = [array_cat, array_num]
    elif plot_properties[plot.__name__]["req_num2"]:
        input_features = [array_num, array_num2]
    else:
        input_features = [array_num]
    with pytest.raises(ValueError):
        plot(*input_features, show_plot=False)

# DataVisualizer.distplot tests

def test_distplot_kde():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    fig = distplot(x, kde=True, show_plot=False, return_plot=True)
    ax = fig.axes[0]

    n_lines = len(ax.lines)

    assert n_lines == 1
    plt.close(fig)
    fig = distplot(x, kde=False, show_plot=False, return_plot=True)
    ax = fig.axes[0]

    n_lines = len(ax.lines)

    assert n_lines == 0
    plt.close(fig)

def test_distplot_bins():
    bins = 10

    x = [i for i in range(10)]
    fig = distplot(x, bins=bins, show_plot=False, return_plot=True)
    ax = fig.axes[0]

    n_bins = len(ax.patches)
    assert n_bins == bins
    plt.close(fig)

