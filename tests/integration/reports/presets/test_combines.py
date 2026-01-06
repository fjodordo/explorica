import pytest
import pandas as pd

from explorica.reports.presets.data_overview import get_data_overview_report, get_data_overview_blocks
from explorica.reports.presets.data_quality import get_data_quality_blocks, get_data_quality_report
from explorica.reports.presets.interactions import get_interactions_blocks, get_interactions_report
from explorica.reports.presets.eda import get_eda_blocks, get_eda_report
from explorica.reports.core.block import Block, BlockConfig
from explorica.reports.core.report import Report
from explorica.types import FeatureAssignment

# -------------------------------
# Fixtures & helper functions
# -------------------------------

@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "x1": [1, 2, 3, 4],
        "x2": [10, 20, 30, 40],
        "c1": ["a", "b", "a", "b"],
        "y": [0, 1, 0, 1],
    })

# -------------------------------
# Tests for get_eda_blocks
# -------------------------------

def test_get_eda_blocks_basic_blocks_generation(df_mixed):

    blocks = []
    try:
        blocks = get_eda_blocks(df_mixed)
        # Check that `blocks` are returned as a list of Blocks
        assert isinstance(blocks, list)
        assert all(isinstance(b, Block) for b in blocks)
        # Check availability of main blocks
        titles = [b.block_config.title for b in blocks]
        assert "Data Overview" in titles
        assert "Data Quality" in titles
        assert "Feature Interactions" in titles
    finally:
        for block in blocks:
            block.close_figures()

def test_get_eda_blocks_target_assignment_heuristics():
    df = pd.DataFrame({
        "num_small_unique": [0, 1, 0, 1],
        "num_large_unique": [10, 20, 30, 40],
        "cat": ["a", "b", "a", "b"]
    })
    blocks = []
    try:
        blocks = get_eda_blocks(
            df,
            numerical_names=["num_large_unique"],
            target_name="num_small_unique",
            categorical_threshold=3
        )
        assert len(blocks) > 0
        titles = [b.block_config.title for b in blocks]
        assert "Feature Interactions" in titles
        assert "Linear relations" in titles
        assert "Non-linear relations" in titles
    finally:
        for block in blocks:
            block.close_figures()

def test_get_eda_blocks_nan_policy_include_vs_drop():
    df = pd.DataFrame({
        "x1": [1, 2, None, 4],
        "c1": ["a", "b", None, "b"],
        "y": [0, 1, 0, None]
    })
    blocks_drop = []
    blocks_include = []
    try:
        # nan_policy='drop'
        blocks_drop = get_eda_blocks(df, nan_policy="drop")
        assert all(isinstance(b, Block) for b in blocks_drop)
        # nan_policy='include'
        blocks_include = get_eda_blocks(df, nan_policy="include")
        assert all(isinstance(b, Block) for b in blocks_include)
    finally:
        for block in blocks_drop + blocks_include:
            block.close_figures()

# -------------------------------
# Tests for get_eda_report
# -------------------------------

def test_get_eda_report_smoke(mocker):
    df = pd.DataFrame({"x": [1, 2, 3]})

    fake_blocks = []

    mocked_blocks = mocker.patch(
        "explorica.reports.presets.eda.get_eda_blocks",
        return_value=fake_blocks,
    )

    report = get_eda_report(
        df,
        numerical_names=["x"],
        target_name="x",
        categorical_threshold=42,
        round_digits=7,
        nan_policy="raise",
    )

    try:
        # Report is constructed correctly
        assert isinstance(report, Report)
        assert report.title == "Exploratory Data Analysis Report"
        assert len(report.blocks) == len(fake_blocks)

        # Parameters are forwarded correctly
        mocked_blocks.assert_called_once_with(
            df,
            ["x"],
            None,
            "x",
            categorical_threshold=42,
            round_digits=7,
            nan_policy="raise",
        )
    finally:
        report.close_figures()


# -------------------------------
# Tests for get_interactions_blocks
# -------------------------------

def test_get_interactions_blocks_linear_block_always_created(df_mixed):
    blocks = get_interactions_blocks(df_mixed)
    try:
        assert len(blocks) >= 1
        assert blocks[0].block_config.title == "Linear relations"
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_nonlinear_block_conditionally_added(df_mixed):
    blocks = get_interactions_blocks(df_mixed)
    try:
        titles = [b.block_config.title for b in blocks]

        assert "Linear relations" in titles
        # Non-linear may be included, and may be not included. check consistency.
        assert len(titles) in (1, 2)
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_feature_assignment_priority(df_mixed):

    blocks = get_interactions_blocks(
        df_mixed, numerical_names=["x1"], categorical_names=["c1"],
        target_numerical_name="y")
    try:
        linear_block = blocks[0]

        # x2 should not be included in linear-relations tables
        for table in linear_block.block_config.tables:
            assert "x2" not in table.table.to_string()
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_mapping_input_supported():
    data = {
        "x1": [1, 2, 3],
        "x2": [4, 5, 6],
        "y": [0, 1, 0],
    }

    blocks = get_interactions_blocks(data)
    try:
        assert len(blocks) >= 1
        assert blocks[0].block_config.title == "Linear relations"
    finally:
        for block in blocks:
            block.close_figures()

def test_get_interactions_blocks_no_numeric_features():
    df = pd.DataFrame({"c1": ["a", "b", "c"]})

    blocks = get_interactions_blocks(df)
    try:
        # block should not be included
        assert all(b.block_config.title != "Linear relations" for b in blocks)
    finally:
        for block in blocks:
            block.close_figures()

# -------------------------------
# Tests for get_interactions_report
# -------------------------------

def test_get_interactions_report_smoke(mocker):
    df = pd.DataFrame({"x": [1, 2, 3]})

    fake_blocks = []

    mocked_blocks = mocker.patch(
        "explorica.reports.presets.interactions.get_interactions_blocks",
        return_value=fake_blocks,
    )

    report = get_interactions_report(
        df,
        categorical_threshold=42,
        round_digits=7,
        nan_policy="raise",
    )
    try:
        # Report is constructed correctly
        assert isinstance(report, Report)
        assert report.title == "Interaction analysis"
        assert len(report.blocks) == len(fake_blocks)

        # Parameters are forwarded
        mocked_blocks.assert_called_once_with(
            df,
            numerical_names=None,
            categorical_names=None,
            target_name=None,
            categorical_threshold=42,
            round_digits=7,
            nan_policy="raise",
        )
    finally:
        report.close_figures()

# -------------------------------
# Tests for get_data_quality_blocks
# -------------------------------

def test_get_data_quality_blocks_smoke(mocker):
    data = {"a": [1, 2, 3]}
    round_digits = 3

    card_block = Block({"title": "cardinality"})
    dist_block = Block({"title": "distributions"})
    out_block = Block({"title": "outliers"})

    card_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_cardinality_block",
        return_value=card_block,
    )
    dist_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_distributions_block",
        return_value=dist_block,
    )
    out_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_outliers_block",
        return_value=out_block,
    )

    blocks = get_data_quality_blocks(data, round_digits=round_digits)
    try:
        # Result
        assert blocks[0].block_config.title == "outliers"
        assert blocks[1].block_config.title == "distributions"
        assert blocks[2].block_config.title == "cardinality"

        # Calls
        card_mock.assert_called_once_with(data, round_digits=round_digits, nan_policy="drop")
        dist_mock.assert_called_once_with(data, round_digits=round_digits, nan_policy="drop")
        out_mock.assert_called_once_with(data, nan_policy="drop")
    finally:
        for block in blocks:
            block.close_figures()


# -------------------------------
# Tests for get_data_quality_report
# -------------------------------

def test_get_data_quality_report_smoke(mocker):
    df = pd.DataFrame({"a": [1, 2, 3]})

    fake_blocks = [
        Block({"title": "block0"}),
        Block({"title": "block1"}),
        Block({"title": "block2"}),
    ]

    blocks_mock = mocker.patch(
        "explorica.reports.presets.data_quality.get_data_quality_blocks",
        return_value=fake_blocks,
    )
    report = get_data_quality_report(df, 3)
    try:
        # orchestration
        blocks_mock.assert_called_once_with(df, 3, nan_policy="drop")

        # report object
        assert isinstance(report, Report)
        assert report.blocks[0].block_config.title == "block0"
        assert report.blocks[1].block_config.title == "block1" 
        assert report.blocks[2].block_config.title == "block2" 
        assert report.title == "Data quality"
    finally:
        report.close_figures()

# -------------------------------
# Tests for get_data_overview_blocks
# -------------------------------

def test_get_data_overview_blocks_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })

    blocks = get_data_overview_blocks(df, round_digits=3)
    try:
        # Check block type and count
        assert isinstance(blocks, list)
        assert len(blocks) == 3
        assert all(isinstance(b, Block) for b in blocks)

        # Check block headers
        titles = [b.block_config.title for b in blocks]
        assert "Basic statistics for the dataset." in titles
        assert "Dataset shape" in titles
        assert "Data quality quick summary" in titles
    finally:
        for block in blocks:
            block.close_figures()

# -------------------------------
# Tests for get_data_overview_report
# -------------------------------

def test_get_data_overview_report_with_mocked_blocks(mocker):
    # make fake data
    dummy_blocks = [
        Block(BlockConfig(title="block-1")),
        Block(BlockConfig(title="block-2"))
    ]

    # mock get_data_overview_blocks
    mocker.patch(
        "explorica.reports.presets.data_overview.get_data_overview_blocks",
        return_value=dummy_blocks
    )

    report = get_data_overview_report(data={"a": [1, 2, 3]})
    try:
        # check type and attrs
        assert isinstance(report, Report)
        assert report.blocks[0].block_config.title == dummy_blocks[0].block_config.title
        assert report.blocks[1].block_config.title == dummy_blocks[1].block_config.title
        assert report.title == "Data overview"
        assert "Short overview of the dataset" in report.description
    finally:
        report.close_figures()