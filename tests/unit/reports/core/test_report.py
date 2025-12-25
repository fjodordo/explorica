import pytest
from explorica.reports import Report, Block, BlockConfig
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_block():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    figly = go.Figure(data=go.Bar(y=[2, 3, 1]))
    blk_cfg = BlockConfig(title="Test Block", visualizations=[fig, figly], metrics=[])
    blk = Block(blk_cfg)
    plt.close(fig)
    yield blk
    plt.close(fig)

@pytest.fixture
def another_block():
    fig, ax = plt.subplots()
    ax.plot([10, 20], [3, 6])
    blk_cfg = BlockConfig(title="Another Block", visualizations=[fig], metrics=[])
    blk = Block(blk_cfg)
    plt.close(fig)
    yield blk
    plt.close(fig)

# -------------------------------
# Tests for Report init
# -------------------------------

def test_report_init_empty():
    rpt = Report()
    try:
        assert rpt.blocks == []
        assert rpt.title is None
        assert rpt.description is None
    finally:
        rpt.close_figures()

def test_report_init_with_blocks(sample_block):
    rpt = Report(blocks=[sample_block], title="Title", description="Desc")
    try:
        assert len(rpt.blocks) == 1
        assert rpt.title == "Title"
        assert rpt.description == "Desc"
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report's Block management
# -----------------------------

def test_insert_block(sample_block, another_block):
    sample_block.block_config.title = "First Block"
    another_block.block_config.title = "Second Block"
    rpt = Report(blocks=[sample_block])
    try:
        rpt.insert_block(another_block, 0)
        assert rpt.blocks[0].block_config.title == another_block.block_config.title
        assert rpt.blocks[1].block_config.title == sample_block.block_config.title
        assert rpt.blocks[0].block_config.title != sample_block.block_config.title
    finally:
        rpt.close_figures()

def test_remove_block(sample_block, another_block):
    sample_block.block_config.title = "this block will be deleted"
    another_block.block_config.title = "shouldn't touch this block"
    rpt = Report(blocks=[sample_block, another_block])
    try:
        removed = rpt.remove_block(0)
        assert removed.block_config.title == sample_block.block_config.title
        assert len(rpt.blocks) == 1
        with pytest.raises(IndexError):
            rpt.remove_block(5)
    finally:
        rpt.close_figures()

def test_iadd_with_single_block(sample_block, another_block):
    rpt = Report(blocks=[sample_block])
    try:
        rpt += another_block
        assert len(rpt.blocks) == 2
        assert rpt.blocks[1] == another_block
    finally:
        rpt.close_figures()

def test_iadd_with_list_blocks(sample_block, another_block):
    rpt = Report(blocks=[sample_block])
    try:
        rpt += [another_block]
        assert len(rpt.blocks) == 2
        assert rpt.blocks[1] == another_block
    finally:
        rpt.close_figures()

def test_add_returns_new_report(sample_block, another_block):
    rpt = Report(blocks=[sample_block])
    try:
        rpt2 = rpt + [another_block]
        assert len(rpt2.blocks) == 2
        assert len(rpt.blocks) == 1  # original unchanged
        assert rpt2.blocks[1] == another_block
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report.__iter__
# -----------------------------

def test_len_and_iter(sample_block, another_block):
    block1, block2 = sample_block, another_block
    block1.block_config.title = "First Block"
    block2.block_config.title = "Second Block"
    rpt = Report(blocks=[block1, block2])
    try:
        assert len(rpt) == 2
        blocks = list(iter(rpt))

        # blocks are not equal, because Report deep-copies blocks during init
        # so compare their titles
        assert blocks[0] is not block1
        assert blocks[1] is not block2
        assert blocks[0].block_config.title == block1.block_config.title
        assert blocks[1].block_config.title == block2.block_config.title
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report.render_html
# -----------------------------

def test_render_html_routes_parameters(mocker, sample_block):
    rpt = Report(blocks=[sample_block])
    try:
        mock_render = mocker.patch("explorica.reports.core.report.render_html", return_value="<html></html>")
        html = rpt.render_html(path="some/path", report_name="myreport", max_width=500)
        mock_render.assert_called_once()
        called_args = mock_render.call_args[1]
        assert called_args["path"] == "some/path"
        assert called_args["report_name"] == "myreport"
        assert called_args["max_width"] == 500
        assert html == "<html></html>"
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report.render_pdf
# -----------------------------

def test_render_pdf_routes_parameters(mocker, sample_block):
    rpt = Report(blocks=[sample_block])
    try:
        mock_render = mocker.patch("explorica.reports.core.report.render_pdf", return_value=b"%PDF-1.4")
        pdf = rpt.render_pdf(path="some/path", report_name="myreport", doc_template_kws={"pagesize": "A4"})
        mock_render.assert_called_once()
        called_args = mock_render.call_args[1]
        assert called_args["path"] == "some/path"
        assert called_args["report_name"] == "myreport"
        assert called_args["doc_template_kws"] == {"pagesize": "A4"}
        assert pdf == b"%PDF-1.4"
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report.typename
# -----------------------------
def test_typename(sample_block):
    rpt = Report(blocks=[sample_block])
    try:
        assert rpt.typename == "Report"
    finally:
        rpt.close_figures()

# -----------------------------
# Tests for Report.close_figures
# -----------------------------

def test_close_figures_removes_matplotlib_figures_from_registry():
    """
    Ensure that Report.close_figures() only closes figures owned by the Report.

    The Report constructor deep-copies provided blocks, which results in
    new Matplotlib Figure instances being created internally. This test
    verifies that:
    - external user-created figures remain open
    - figures belonging to the Report are properly closed
    """
    plt.close("all")

    fig, ax = plt.subplots()
    try:
        # Figure created by the user (external)
        external_fig_num = fig.number

        # Figure copied and normalized inside Report
        block = Block({"visualizations": [fig], "metrics": []})
        report = Report(blocks=[block])
        report_fig_num = (
            report.blocks[0]
            .block_config.visualizations[0]
            .figure.number
        )

        report.close_figures()

        assert external_fig_num in plt.get_fignums()
        assert report_fig_num not in plt.get_fignums()
    finally:
        plt.close(fig)

def test_close_figures_same_figure_in_multiple_blocks():
    """
    Ensure that the same external Figure added to multiple blocks is handled
    correctly by Report.close_figures().

    Even if the same Matplotlib Figure instance is provided multiple times
    when constructing a Report, each block receives its own internal copy.
    This test verifies that:
    - all internal figures created by the Report are closed
    - the original external figure remains open
    """
    plt.close("all")
    fig, ax = plt.subplots()
    try:
        external_fig_num = fig.number

        block1 = Block({"visualizations": [fig], "metrics": []})
        block2 = Block({"visualizations": [fig], "metrics": []})

        report = Report(blocks=[block1, block2])
        report_fig1_num = (report.blocks[0]
            .block_config.visualizations[0]
            .figure.number
        )
        report_fig2_num = (report.blocks[1]
            .block_config.visualizations[0]
            .figure.number
        )
        report.close_figures()

        fignums = plt.get_fignums()
        assert external_fig_num in fignums
        assert report_fig1_num not in fignums
        assert report_fig2_num not in fignums
    finally:
        plt.close(fig)

def test_close_figures_ignores_plotly_figures():
    """
    Ensure that non-Matplotlib visualizations do not affect the behavior of
    Report.close_figures().

    This test verifies that:
    - Plotly figures are ignored by close_figures()
    - Matplotlib figures inside the Report are still properly closed
    - external Matplotlib figures remain open
    - the presence of Plotly visualizations does not alter or break the
      expected closing logic
    """
    plt.close("all")

    fig, ax = plt.subplots()
    try:
        figly = go.Figure(data=go.Bar(y=[1, 2, 3]))
        external_fig_num = fig.number

        # Figure copied and normalized inside Report
        block = Block({"visualizations": [fig, figly], "metrics": []})
        report = Report(blocks=[block])
        report_fig_num = (
            report.blocks[0]
            .block_config.visualizations[0]
            .figure.number
        )

        report.close_figures()

        assert external_fig_num in plt.get_fignums()
        assert report_fig_num not in plt.get_fignums()
    finally:
        plt.close(fig)

def test_close_figures_empty_report_is_noop():
    """
    Ensure that calling Report.close_figures() on an empty report
    does not raise any exceptions or errors.
    
    This confirms that the method is safe to call regardless of
    whether any blocks or visualizations exist.
    """
    report = Report(blocks=[])
    report.close_figures()