import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from explorica.reports import normalize_visualization
from explorica.types import VisualizationResult


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
