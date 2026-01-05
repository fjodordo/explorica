import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from explorica.reports.core.block import Block, BlockConfig
from explorica.types import VisualizationResult, TableResult


# -------------------------------
# Fixtures & helper functions
# -------------------------------

@pytest.fixture
def block() -> Block:
    return Block()

@pytest.fixture
def simple_visualization():
    fig, ax = plt.subplots(figsize=(10, 6))
    vr = VisualizationResult(figure=fig, axes=ax)
    yield vr
    plt.close(vr.figure)

@pytest.fixture
def simple_metric() -> dict:
    metric = {
        "name": "metric1",
        "value": 14.2,
        "description": "metric1 description"
    }
    return metric



# -------------------------------
# Tests for Block init
# -------------------------------

def test_block_init_with_none():
    """Block can be initialized with None (default)"""
    block = Block()
    assert isinstance(block.block_config, BlockConfig)
    assert block.block_config.visualizations == []
    assert block.block_config.metrics == []

def test_block_init_with_empty_dict():
    """Block can be initialized with empty dict"""
    block = Block({})
    assert isinstance(block.block_config, BlockConfig)
    assert block.block_config.visualizations == []
    assert block.block_config.metrics == []

def test_block_init_with_blockconfig_instance():
    """Block can be initialized with BlockConfig instance"""
    cfg = BlockConfig(
        title="Test Block",
        description="Testing",
        metrics=[{"name": "mean", "value": 1}],
        visualizations=[]
    )
    block = Block(cfg)
    assert block.block_config is cfg
    assert block.block_config.title == "Test Block"
    assert block.block_config.metrics[0]["name"] == "mean"

def test_init_invalid_type_raises():
    """Block raises ValueError if block_config is invalid type"""
    with pytest.raises(ValueError):
        Block(block_config=123)  # int is invalid type
    with pytest.raises(ValueError):
        Block(block_config="string")  # str is invalid type

def test_block_normalizes_visualizations_and_tables():
    fig_mpl, ax = plt.subplots()
    try:
        ax.plot([1, 2, 3], [4, 5, 6])
        fig_plotly = go.Figure(data=go.Bar(y=[1, 2, 3]))

        table1 = [[1, 2], [3, 4]]
        table2 = {"col1": [10, 20], "col2": [30, 40]}

        block_cfg = BlockConfig(
            title="Test Block",
            visualizations=[fig_mpl, fig_plotly],
            tables=[table1, table2],
            metrics=[]
        )

        block = Block(block_cfg)

        for vis in block.block_config.visualizations:
            assert isinstance(vis, VisualizationResult)

        for tbl in block.block_config.tables:
            assert isinstance(tbl, TableResult)
    finally:
        plt.close(fig_mpl)

# -------------------------------
# Tests for Block.add_visualization
# -------------------------------

def test_add_matplotlib_figure(block):
    """Add a matplotlib figure and check normalization"""
    fig, ax = plt.subplots()
    try:
        block.add_visualization(fig)
        vis = block.block_config.visualizations[-1]
        assert isinstance(vis, VisualizationResult)
        assert len(block.block_config.visualizations) == 1
    finally:
        plt.close(fig)

def test_add_plotly_figure(block):
    """Add a plotly figure and check normalization"""
    fig = go.Figure(data=go.Bar(y=[1, 2, 3]))
    block.add_visualization(fig)
    vis = block.block_config.visualizations[-1]
    assert isinstance(vis, VisualizationResult)
    assert len(block.block_config.visualizations) == 1

def test_add_visualization_multiple(block):
    """Add multiple figures sequentially"""
    fig1, ax1 = plt.subplots()
    fig2 = go.Figure(data=go.Bar(y=[1, 2, 3]))
    try:
        block.add_visualization(fig1)
        block.add_visualization(fig2)
        vis_list = block.block_config.visualizations
        assert len(vis_list) == 2
        assert vis_list[0].figure is fig1
        assert vis_list[1].figure is fig2
    finally:
        plt.close(fig1)

def test_add_visualization_result_object(block):
    """Add an already normalized VisualizationResult"""
    fig, ax = plt.subplots()
    try:
        dummy_vis = VisualizationResult(fig, engine="matplotlib")
        block.add_visualization(dummy_vis)
        vis = block.block_config.visualizations[-1]
        assert vis is dummy_vis
        assert len(block.block_config.visualizations) == 1
    finally:
        plt.close(fig)

# -------------------------------
# Tests for Block.insert_visualization
# -------------------------------

def test_insert_visualization_into_empty_block():
    block = Block()
    fig, ax = plt.subplots()
    try:
        block.insert_visualization(fig, index=0)

        assert len(block.block_config.visualizations) == 1
        assert isinstance(
            block.block_config.visualizations[0],
            VisualizationResult
        )
    finally:
        plt.close(fig)


def test_insert_visualization_at_specific_index():
    block = Block()
    fig1, ax1 = plt.subplots()
    try:
        fig2 = go.Figure(data=go.Bar(y=[1, 2, 3]))

        result_mpl = VisualizationResult(fig1, engine="matplotlib")
        result_plotly = VisualizationResult(fig2, engine="plotly")
    
        block.insert_visualization(result_mpl, index=0)
        block.insert_visualization(result_plotly, index=0)

        vis_list = block.block_config.visualizations

        assert vis_list[0].engine == "plotly"
        assert vis_list[1].engine == "matplotlib"
    finally:
        plt.close(fig1)


def test_insert_visualization_negative_index():
    block = Block()
    fig1, ax1 = plt.subplots()
    try:
        fig2 = go.Figure(data=go.Bar(y=[1, 2, 3]))

        result_mpl = VisualizationResult(fig1, engine="matplotlib")
        result_plotly = VisualizationResult(fig2, engine="plotly")

        block.add_visualization(result_mpl)
        block.insert_visualization(result_plotly, index=-1)

        vis_list = block.block_config.visualizations

        assert len(vis_list) == 2
        # index = -1 -> insert before the last element
        assert vis_list[0].engine == "plotly"
        assert vis_list[1].engine == "matplotlib"
    finally:
        plt.close(fig1)


def test_insert_already_normalized_visualization():
    block = Block()
    vis = VisualizationResult(figure=go.Figure(data=go.Bar(y=[1, 2, 3])))

    block.insert_visualization(vis, index=0)

    assert block.block_config.visualizations[0] is vis

# -------------------------------
# Tests for Block.remove_visualization
# -------------------------------


def test_remove_visualization_returns_removed_item():
    fig, _ = plt.subplots()
    try:
        block = Block(BlockConfig(visualizations=[fig]))
        removed = block.remove_visualization(0)

        assert removed.figure is fig
        assert block.block_config.visualizations == []
    finally:
        plt.close(fig)

def test_remove_visualization_invalid_index_raises_index_error():
    block = Block(BlockConfig(visualizations=[]))

    with pytest.raises(IndexError, match="No visualization at index 0"):
        block.remove_visualization(0)

# -------------------------------
# Tests for Block.clear_visualizations
# -------------------------------

def test_clear_visualizations_removes_all_items():

    fig, _ = plt.subplots()
    try:
        block = Block(BlockConfig(visualizations=[fig]))
        assert len(block.block_config.visualizations) == 1

        block.clear_visualizations()

        assert block.block_config.visualizations == []
    finally:
        plt.close(fig)

# -------------------------------
# Tests for Block.add_metric
# -------------------------------

def test_add_metric_appends_metric():
    block = Block()

    block.add_metric(name="mean", value=5.0, description="Mean value")

    assert len(block.block_config.metrics) == 1
    metric = block.block_config.metrics[0]

    assert metric["name"] == "mean"
    assert metric["value"] == 5.0
    assert metric["description"] == "Mean value"

def test_add_metric_raises_if_name_not_hashable():
    block = Block()

    with pytest.raises(TypeError):
        block.add_metric(name=["mean"], value=5.0)

def test_add_metric_raises_if_description_not_hashable():
    block = Block()

    with pytest.raises(TypeError):
        block.add_metric(name="mean", value=5.0, description=["desc"])

def test_add_metric_raises_invalid_value():
    block = Block()

    with pytest.raises(TypeError):
        block.add_metric(name="mean", value=set(1, 2, 3))

# -------------------------------
# Tests for Block.insert_metric
# -------------------------------

def test_insert_metric_inserts_at_index():
    block = Block(BlockConfig(metrics=[
        {"name": "A", "value": 1.2, "description": None},
        {"name": "C", "value": 2.1, "description": None}]))

    block.insert_metric(1, "B", 2)

    names = [m["name"] for m in block.block_config.metrics]
    assert names == ["A", "B", "C"]

def test_insert_metric_raises_for_invalid_value():
    block = Block()

    with pytest.raises(TypeError):
        block.insert_metric(0, "bad", [1, 2, 3])

# -------------------------------
# Tests for Block.remove_metric
# -------------------------------

def test_remove_metric_removes_correct_item():
    block = Block(BlockConfig(metrics=[
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
        {"name": "c", "value": 3}
    ]))

    removed = block.remove_metric(1)
    assert removed["name"] == "b"
    assert len(block.block_config.metrics) == 2
    assert [m["name"] for m in block.block_config.metrics] == ["a", "c"]

def test_remove_metric_supports_negative_index():
    block = Block(BlockConfig(metrics=[
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
    ]))

    removed = block.remove_metric(-1)
    assert removed["name"] == "b"
    assert len(block.block_config.metrics) == 1
    assert block.block_config.metrics[0]["name"] == "a"

def test_remove_metric_raises_index_error():
    block = Block()
    with pytest.raises(IndexError):
        block.remove_metric(0)

# -------------------------------
# Tests for Block.add_table
# -------------------------------

def test_block_add_table_normalizes_to_tableresult():
    block = Block(BlockConfig())

    data = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }

    block.add_table(
        table=data,
        title="Test table",
        description="Some description",
    )

    assert len(block.block_config.tables) == 1

    table = block.block_config.tables[0]
    assert isinstance(table, TableResult)
    assert isinstance(table.table, pd.DataFrame)

    assert table.title == "Test table"
    assert table.description == "Some description"

    pd.testing.assert_frame_equal(
        table.table,
        pd.DataFrame(data),
    )

# -------------------------------
# Tests for Block.insert_table
# -------------------------------

def test_block_insert_table_at_index():
    block = Block(BlockConfig(tables=[{"x": [1, 2]}, {"y": [3, 4]}]))

    insert_data = {"z": [10, 20]}

    block.insert_table(
        index=1,
        table=insert_data,
        title="Inserted",
    )

    assert len(block.block_config.tables) == 3

    inserted = block.block_config.tables[1]
    assert isinstance(inserted, TableResult)
    assert inserted.title == "Inserted"

    pd.testing.assert_frame_equal(
        inserted.table,
        pd.DataFrame(insert_data),
    )

# -------------------------------
# Tests for Block.remove_table
# -------------------------------

def test_block_remove_table_by_index():
    block = Block(BlockConfig(tables=[{"a": [1]}, {"b": [2]}]))

    removed = block.remove_table(0)

    assert isinstance(removed, TableResult)
    assert len(block.block_config.tables) == 1

    remaining = block.block_config.tables[0]
    pd.testing.assert_frame_equal(
        remaining.table,
        pd.DataFrame({"b": [2]}),
    )

def test_block_remove_table_invalid_index_raises():
    block = Block(BlockConfig())

    with pytest.raises(IndexError, match="No metric at index"):
        block.remove_table(0)

# -------------------------------
# Tests for Block.render_html
# -------------------------------

def test_render_html_forwards_parameters(mocker):
    block = Block()
    block.add_metric("mean", 5.0)

    extra_kwargs = {"max_width": 800, "verbose": True}

    # Mock render_html function
    mock_render = mocker.patch("explorica.reports.core.block.render_html",
                               return_value="<html></html>")

    result = block.render_html(path="./reports", font=("Arial", "Sans"), **extra_kwargs)

    # Check, that render_html has been called once
    mock_render.assert_called_once()

    # Check parameters of call
    called_args, called_kwargs = mock_render.call_args
    assert called_args[0] == block
    assert called_kwargs["path"] == "./reports"
    assert called_kwargs["font"] == ("Arial", "Sans")
    assert called_kwargs["max_width"] == 800
    assert called_kwargs["verbose"] is True

# -------------------------------
# Tests for Block.render_pdf
# -------------------------------

def test_render_pdf_forwards_parameters(mocker):
    block = Block()
    block.add_metric("mean", 5.0)

    extra_kwargs = {"verbose": True}

    # Mock render_pdf function
    mock_render = mocker.patch("explorica.reports.core.block.render_pdf",
                               return_value="<html></html>")

    result = block.render_pdf(path="./reports", font="DejaVuSans", **extra_kwargs)

    # Check, that render_html has been called once
    mock_render.assert_called_once()

    # Check parameters of call
    called_args, called_kwargs = mock_render.call_args
    assert called_args[0] == block
    assert called_kwargs["path"] == "./reports"
    assert called_kwargs["font"] == "DejaVuSans"
    assert called_kwargs["verbose"] is True

# -----------------------------
# Tests for Block.typename
# -----------------------------

def test_typename():
    block = Block()
    assert block.typename == "Block"

# -----------------------------
# Tests for Block.empty
# -----------------------------

def test_empty_block():
    block = Block(BlockConfig(title="Test empty"))
    assert block.empty is True

def test_block_with_table():
    block = Block(BlockConfig(title="Test table"))
    block.add_table(TableResult(table=pd.DataFrame()))
    assert block.empty is False

def test_block_with_visualization(simple_visualization):
    block = Block(BlockConfig(title="Test viz"))
    block.add_visualization(simple_visualization)
    assert block.empty is False

def test_block_with_metrics(simple_metric):
    block = Block(BlockConfig(title="Test metric"))
    block.add_metric(**simple_metric)
    assert block.empty is False

# -----------------------------
# Tests for Report.close_figures
# -----------------------------

def test_close_figures_really_closes_figures():
    # Make figures 
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    block = Block(BlockConfig(visualizations=[fig1, fig2]))

    # Check that figures're opened
    open_figs_before = plt.get_fignums()
    assert fig1.number in open_figs_before
    assert fig2.number in open_figs_before

    # Call close method
    block.close_figures()

    # Check that figures're closed
    open_figs_after = plt.get_fignums()
    assert fig1.number not in open_figs_after
    assert fig2.number not in open_figs_after
