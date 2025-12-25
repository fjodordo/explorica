import pytest
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from reportlab.platypus import Image, Paragraph

from explorica.reports.renderers.pdf import (
    _save_pdf,
    _get_build_pdf,
    _preprocess_font,
    render_pdf)
from explorica.reports.renderers.html import (
    _save_html,
    render_block_html,
    _render_block_html_build_visualizations,
    render_html)
from explorica.reports import render_block_pdf, BlockConfig, Block, Report
from explorica.types import VisualizationResult

# -------------------------------
# Consts and fixtures
# -------------------------------

@pytest.fixture
def simple_block():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    # Plotly figure
    plotly_fig = go.Figure(data=go.Bar(y=[2, 3, 1]))

    block_config = BlockConfig(
        title="Test Block",
        description="Block with Matplotlib and Plotly figures",
        metrics=[{"name": "Metric A", "value": 42}],
        visualizations=[fig, plotly_fig]
    )

    yield Block(block_config)
    plt.close(fig)

@pytest.fixture
def simple_report(simple_block):
    blocks = [simple_block, simple_block]
    report = Report(blocks, title="Report Title", description="Report Desc")
    yield report
    report.close_figures()

@pytest.fixture
def empty_report():
    report = Report([], title="Report Title", description="Report without blocks")
    yield report
    report.close_figures()


# -------------------------------
# Tests for render_html
# -------------------------------

def test_render_html_invalid_type():
    with pytest.raises(TypeError):
        render_html(data=object())

def test_render_html_empty_report_inserts_placeholder(empty_report):
    html = render_html(empty_report)
    assert "There are no render blocks in this report." in html

# -------------------------------
# Tests for render_block_html
# -------------------------------

def test_render_block_html_font_family(simple_block):
    html_output = render_block_html(
        simple_block,
        add_css_style=True,
        font=["Arial", "sans-serif"]
    )
    assert "font-family: Arial, sans-serif" in html_output

def test_render_block_html_without_visualizations():
    block_config = BlockConfig(
        title="No Vis",
        description="Block without visualizations",
        metrics=[]
    )
    block = Block(block_config)

    html_output = render_block_html(block)

    assert '<div class="visualizations">' not in html_output

def test_render_block_html_plotly_without_dimensions(simple_block):
    # Plotly figure without explicit width/height
    plotly_fig = go.Figure(data=go.Bar(y=[1, 2, 3]))

    block_config = BlockConfig(
        title="Plotly No Size",
        description="Plotly without dimensions",
        visualizations=[plotly_fig]
    )
    block = Block(block_config)

    html_output = render_block_html(block)

    assert isinstance(html_output, str)
    assert "Plotly.newPlot" in html_output

def test_render_block_html_mpl_scaling(simple_block):
    html_small = render_block_html(simple_block, mpl_fig_scale=50.0)
    html_large = render_block_html(simple_block, mpl_fig_scale=200.0)

    # width/height must mismatch
    assert html_small != html_large

def test_render_block_html_contains_plotly_html(simple_block):
    html_output = render_block_html(simple_block)
    # Plotly always injects a script
    assert "plotly.js" in html_output or "Plotly.newPlot" in html_output

def test_render_block_html_contains_matplotlib_image(simple_block):
    html_output = render_block_html(simple_block)
    assert '<img src="data:image/png;base64,' in html_output
    assert 'width="' in html_output
    assert 'height="' in html_output

def test_render_block_html_custom_css_names(simple_block):
    html_output = render_block_html(
        simple_block,
        add_css_style=True,
        block_name="custom-block",
        vis_name_css="custom-vis"
    )

    assert '<div class=\'custom-block\'>' in html_output
    assert '<div class="custom-vis">' in html_output

    # CSS uses the same names
    assert ".custom-block {" in html_output
    assert ".custom-vis {" in html_output

def test_render_block_html_without_css(simple_block):
    html_output = render_block_html(simple_block, add_css_style=False)
    assert "<style>" not in html_output
    assert "</style>" not in html_output

def test_render_block_html_smoke(simple_block):
    html_output = render_block_html(simple_block, add_css_style=True)
    # Check, that returned type is string
    assert isinstance(html_output, str)
    # Check, that header and description are enabled
    assert "<h2>Test Block</h2>" in html_output
    assert "<p>Block with Matplotlib and Plotly figures</p>" in html_output
    # Check availability of visualiasations
    assert '<div class="visualizations">' in html_output
    # Check CSS wrap
    assert "<style>" in html_output and "</style>" in html_output

# -------------------------------
# Tests for _render_block_html_build_visualizations
# -------------------------------

def test_render_visualizations_smoke(simple_block):
    html_snippets = _render_block_html_build_visualizations(
        simple_block, mpl_fig_scale=80.0, plotly_fig_scale=1.0, name_css="visualizations"
    )
    assert isinstance(html_snippets, list)
    assert any('<div class="visualizations">' in s for s in html_snippets) or html_snippets[0].startswith('<div class="visualizations">')
    joined_html = "\n".join(html_snippets)
    assert "<img" in joined_html or "<iframe" in joined_html

# -------------------------------
# Tests for _save_html
# -------------------------------

def test_save_html_to_directory(tmp_path):
    html_content = "<html><body><h1>Test</h1></body></html>"
    _save_html(html_content, tmp_path, report_name="my_report")


    # Check that file created with the correct filename
    expected_file = tmp_path / "my_report.html"
    assert expected_file.exists()

    # Check content
    content = expected_file.read_text(encoding="utf-8")
    assert "<h1>Test</h1>" in content

def test_save_html_to_file(tmp_path):
    html_content = "<html><body><h1>File Test</h1></body></html>"
    file_path = tmp_path / "custom.html"
    _save_html(html_content, file_path)

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "<h1>File Test</h1>" in content

def test_save_html_overwrite(tmp_path):
    html_content = "<html><body>Original</body></html>"
    file_path = tmp_path / "overwrite.html"
    file_path.write_text("<html>Old</html>", encoding="utf-8")

    # If overwrite=True, overwriting will occur
    _save_html(html_content, file_path, overwrite=True)
    assert "Original" in file_path.read_text(encoding="utf-8")

def test_save_html_no_overwrite(tmp_path):
    html_content = "<html><body>New</body></html>"
    file_path = tmp_path / "no_overwrite.html"
    file_path.write_text("<html>Existing</html>", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _save_html(html_content, file_path, overwrite=False)

def test_save_html_invalid_extension(tmp_path):
    html_content = "<html></html>"
    bad_path = tmp_path / "not_html.txt"

    with pytest.raises(ValueError):
        _save_html(html_content, bad_path)

# -------------------------------
# Tests for render_pdf
# -------------------------------

def test_render_pdf_invalid_type(simple_block, simple_report):
    with pytest.raises(TypeError):
        render_pdf(data="not_a_block_or_report")
    assert render_pdf(data=simple_block) is not None
    assert render_pdf(data=simple_report) is not None

def test_render_pdf_block_calls(simple_block, mocker):
    # Mock helpers
    mock_preprocess = mocker.patch("explorica.reports.renderers.pdf._preprocess_font",
                                   return_value=("DejaVuSans", {"Normal": None, "Heading1": None, "BodyText": None}))
    mock_render_block = mocker.patch("explorica.reports.renderers.pdf.render_block_pdf",
                                     return_value=["mock_story"])
    mock_get_pdf = mocker.patch("explorica.reports.renderers.pdf._get_build_pdf",
                                return_value=b"pdf_bytes")
    mock_save_pdf = mocker.patch("explorica.reports.renderers.pdf._save_pdf")
    pdf_bytes = render_pdf(simple_block,
                           path=None,
                           font="DejaVuSans", report_name="my_block")

    # Check calls
    mock_preprocess.assert_called_once_with("DejaVuSans")
    mock_render_block.assert_called_once()
    mock_get_pdf.assert_called_once_with(["mock_story"], doc_template_kws=None)
    mock_save_pdf.assert_not_called()  # path=None -> do not save

    # Returned type check
    assert pdf_bytes == b"pdf_bytes"

def test_render_pdf_report_params(simple_report, mocker):
    mock_preprocess = mocker.patch("explorica.reports.renderers.pdf._preprocess_font",
                                   return_value=("DejaVuSans", {"Normal": None, "Heading1": None, "BodyText": None}))
    mock_render_block = mocker.patch("explorica.reports.renderers.pdf.render_block_pdf",
                                     return_value=["block_story"])
    mock_get_pdf = mocker.patch("explorica.reports.renderers.pdf._get_build_pdf",
                                return_value=b"pdf_bytes")

    pdf_bytes = render_pdf(simple_report,
                           path=None,
                           font="DejaVuSans", report_name="report_name",
                           mpl_fig_scale=50, plotly_fig_scale=2.0)

    # Check calls with deterministic parameter values
    for render_block_args in mock_render_block.call_args_list:
        args, kwargs = render_block_args
        assert kwargs["mpl_fig_scale"] == 50
        assert kwargs["plotly_fig_scale"] == 2.0

    for preprocess_args in mock_preprocess.call_args_list:
        args, kwargs = preprocess_args
        assert args[0] == "DejaVuSans"

    for build_args in mock_get_pdf.call_args_list:
        args, kwargs = build_args
        assert "block_story" in args[0]
        assert kwargs["doc_template_kws"] is None
    # Return type check
    assert pdf_bytes == b"pdf_bytes"

def test_render_pdf_logging(simple_block, mocker, caplog):
    mocker.patch("explorica.reports.renderers.pdf._preprocess_font",
                                   return_value=("DejaVuSans", {"Normal": None, "Heading1": None, "BodyText": None}))
    mocker.patch("explorica.reports.renderers.pdf.render_block_pdf",
                                     return_value=["mock_story"])
    mocker.patch("explorica.reports.renderers.pdf._get_build_pdf",
                                return_value=b"pdf_bytes")

    with caplog.at_level("INFO"):
        render_pdf(simple_block, path=None, report_name="my_block", verbose=True)

    # Check, that logs about 'start rendering' & 'success rendering' are exists
    assert any("Rendering 'my_block'" in r.message for r in caplog.records)
    assert any("'my_block' was successfully rendered" in r.message for r in caplog.records)

# -------------------------------
# Tests for _preprocess_font
# -------------------------------

def test_preprocess_font_builtin():
    font, styles = _preprocess_font("DejaVuSans")
    assert font == "DejaVuSans"
    assert "Normal" in styles

def test_preprocess_font_user_path(tmp_path):
    font_path = Path(__file__).parent.parent.parent.parent / "src/explorica/assets/fonts/DejaVuSans.ttf"
    font, styles = _preprocess_font(str(font_path))
    assert font == "UserProvidedFont"

def test_preprocess_font_invalid_path():
    with pytest.raises(ValueError):
        _preprocess_font("/non/existing/font.ttf")

# -------------------------------
# Tests for _get_build_pdf
# -------------------------------

def test_get_build_pdf_basic():
    story = [Paragraph("Test", _preprocess_font("DejaVuSans")[1]["Normal"])]
    pdf_bytes = _get_build_pdf(story)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0

def test_get_build_pdf_with_doc_template_kws():
    story = [Paragraph("Test", _preprocess_font("DejaVuSans")[1]["Normal"])]
    pdf_bytes = _get_build_pdf(story, doc_template_kws={"rightMargin": 10})
    assert isinstance(pdf_bytes, bytes)

# -------------------------------
# Tests for render_block_pdf
# -------------------------------

def test_render_block_basic_metrics_and_description():
    """Block with title, description, and metrics renders correctly."""
    block_cfg = BlockConfig(
        title="Test Block",
        description="This is a test block",
        metrics=[{"name": "sum", "value": 42, "description": "total"}],
        visualizations=[]
    )
    block = Block(block_cfg)
    flowables = render_block_pdf(block)
    
    # Should contain Flowables: title, description, metric
    assert len(flowables) >= 3
    # First Flowable should be title
    assert str(flowables[0]).find("Test Block") != -1
    # Second Flowable should be description
    assert str(flowables[2]).find("This is a test block") != -1
    # Metric text should be present
    metric_text = str(flowables[4])
    assert "sum" in metric_text and "42" in metric_text and "total" in metric_text

def test_render_block_with_matplotlib_fig():
    """Matplotlib figure is converted to Flowable."""
    fig, ax = plt.subplots()
    try:
        ax.plot([1, 2], [3, 4])
        vis = VisualizationResult(engine="matplotlib", figure=fig, width=6, height=4)
        
        block_cfg = BlockConfig(
            title="Plot Block",
            visualizations=[vis]
        )
        block = Block(block_cfg)
        flowables = render_block_pdf(block)
        
        # Should include at least one Flowable which is an Image
        assert any(isinstance(f, Image) for f in flowables)
    finally:
        plt.close(fig)

def test_render_block_with_plotly_fig_placeholder(monkeypatch):
    """Plotly figure is replaced by a placeholder image."""
    vis = VisualizationResult(engine="plotly", figure=None, width=5, height=3)
    
    block_cfg = BlockConfig(
        title="Plotly Block",
        visualizations=[vis]
    )
    block = Block(block_cfg)
    flowables = render_block_pdf(block)
    
    assert any(isinstance(f, Image) for f in flowables)

def test_render_block_mpl_figure_size():
    fig, ax = plt.subplots(figsize=(2, 1))  # 2x1 inches
    try:
        vis = VisualizationResult(engine="matplotlib", figure=fig, width=2, height=1)

        block_cfg = BlockConfig(
            title="MPL Block",
            description="Test MPL figure size",
            metrics=[],
            visualizations=[vis]
        )
        block = Block(block_cfg)
        flowables = render_block_pdf(block, mpl_fig_scale=100.0)  # scale = 100

        # Check that Image with the correct size exists
        images = [f for f in flowables if isinstance(f, Image)]
        assert len(images) == 1
        img = images[0]
        assert img._width == pytest.approx(200.0)  # 2 * 100
        assert img._height == pytest.approx(100.0) # 1 * 100
    finally:
        plt.close(fig)

def test_render_block_plotly_figure_placeholder_size():
    fig = go.Figure(data=go.Bar(y=[1, 2, 3]))
    vis = VisualizationResult(engine="plotly", figure=fig, width=5, height=3)

    block_cfg = BlockConfig(
        title="Plotly Block",
        description="Test Plotly placeholder size",
        metrics=[],
        visualizations=[vis]
    )
    block = Block(block_cfg)
    flowables = render_block_pdf(block, plotly_fig_scale=10.0)  # тестовый scale

    images = [f for f in flowables if isinstance(f, Image)]
    assert len(images) == 1
    img = images[0]
    # Check, that size match scale
    assert img._width == pytest.approx(50.0)  # 5 * 10
    assert img._height == pytest.approx(30.0) # 3 * 10

# -------------------------------
# Tests for _save_pdf
# -------------------------------

def test_save_pdf_to_directory(tmp_path):
    pdf_bytes = b"%PDF-1.4\n%..."
    # Save to dir, using default name
    expected_file = tmp_path / "report.pdf"
    _save_pdf(pdf_bytes, tmp_path)
    assert expected_file.exists()
    assert expected_file.read_bytes() == pdf_bytes

def test_save_pdf_to_full_path(tmp_path):
    pdf_bytes = b"%PDF-1.4\n%..."
    file_path = tmp_path / "my_report.pdf"
    _save_pdf(pdf_bytes, file_path)
    assert file_path.exists()
    assert file_path.read_bytes() == pdf_bytes

def test_overwrite_false_raises(tmp_path):
    pdf_bytes = b"%PDF-1.4\n%..."
    file_path = tmp_path / "existing.pdf"
    file_path.write_bytes(pdf_bytes)
    # overwrite=False -> must raise error
    with pytest.raises(FileExistsError):
        _save_pdf(pdf_bytes, file_path, overwrite=False)

def test_overwrite_true_allows(tmp_path):
    pdf_bytes = b"%PDF-1.4\n%..."
    file_path = tmp_path / "existing.pdf"
    file_path.write_bytes(b"OLD")
    # overwrite=True -> will overwrite
    _save_pdf(pdf_bytes, file_path, overwrite=True)
    assert file_path.read_bytes() == pdf_bytes

def test_non_pdf_path_raises(tmp_path):
    pdf_bytes = b"%PDF-1.4\n%..."
    file_path = tmp_path / "report.html"
    with pytest.raises(ValueError):
        _save_pdf(pdf_bytes, file_path)
