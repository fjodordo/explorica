import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from explorica.reports.utils import normalize_visualization, normalize_table
from explorica.types import VisualizationResult, TableResult

# -------------------------------
# Tests for normalize_visualization
# -------------------------------

def test_normalize_matplotlib_figure():
    fig, ax = plt.subplots()
    try:
        ax.plot([1, 2, 3], [4, 5, 6])
        fig.suptitle("Test Title")

        result = normalize_visualization(fig)

        assert isinstance(result, VisualizationResult)
        assert result.engine == "matplotlib"
        assert result.axes == fig.get_axes()
        assert result.width == pytest.approx(fig.get_size_inches()[0])
        assert result.height == pytest.approx(fig.get_size_inches()[1])
        assert result.figure == fig
    finally:
        plt.close(fig)


def test_normalize_plotly_figure():
    fig = go.Figure(data=go.Bar(y=[1, 2, 3]))
    fig.update_layout(title="Plotly Title", width=800, height=600)

    result = normalize_visualization(fig)

    assert isinstance(result, VisualizationResult)
    assert result.engine == "plotly"
    assert result.axes is None
    assert result.width == 800
    assert result.height == 600
    assert result.figure == fig


def test_normalize_unsupported_type():
    with pytest.raises(TypeError):
        normalize_visualization("not a figure")


def test_result_attributes_exist():
    fig, ax = plt.subplots()
    try:
        result = normalize_visualization(fig)
        expected_attrs = ["figure", "engine", "axes", "width", "height", "title"]
        for attr in expected_attrs:
            assert hasattr(result, attr)
    finally:
        plt.close(fig)

# -------------------------------
# Tests for normalize_table
# -------------------------------

def test_normalize_table_from_dict():
    data = {"a": [1, 2], "b": [3, 4]}
    result = normalize_table(data)
    assert isinstance(result, TableResult)
    assert isinstance(result.table, pd.DataFrame)
    assert list(result.table.columns) == ["a", "b"]
    assert result.table.shape == (2, 2)

def test_normalize_table_from_list_of_lists():
    data = [[1, 2], [3, 4]]
    result = normalize_table(data)
    assert isinstance(result, TableResult)
    assert isinstance(result.table, pd.DataFrame)
    assert result.table.shape == (2, 2)

def test_normalize_table_from_1d_list():
    data = [1, 2, 3]
    result = normalize_table(data)
    assert isinstance(result, TableResult)
    assert isinstance(result.table, pd.DataFrame)
    assert result.table.shape == (3, 1)

def test_normalize_table_raises_on_multiindex_rows():
    # Create a DataFrame with MultiIndex rows
    index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
    df = pd.DataFrame({"col1": [10, 20]}, index=index)

    with pytest.raises(ValueError, match="MultiIndex in rows is not supported"):
        normalize_table(df)

def test_normalize_table_raises_on_multiindex_columns():
    # Create a DataFrame with MultiIndex columns
    columns = pd.MultiIndex.from_tuples([("A", "x"), ("B", "y")])
    df = pd.DataFrame([[1, 2]], columns=columns)

    with pytest.raises(ValueError, match="MultiIndex in columns is not supported"):
        normalize_table(df)