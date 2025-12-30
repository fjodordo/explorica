import pandas as pd

from explorica.reports.presets.data_overview import get_data_overview_report, get_data_overview_blocks
from explorica.reports.presets.data_quality import get_data_quality_blocks, get_data_quality_report
from explorica.reports.core.block import Block, BlockConfig
from explorica.reports.core.report import Report


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

    # Result
    assert blocks[0].block_config.title == "outliers"
    assert blocks[1].block_config.title == "distributions"
    assert blocks[2].block_config.title == "cardinality"

    # Calls
    card_mock.assert_called_once_with(data, round_digits=round_digits)
    dist_mock.assert_called_once_with(data, round_digits=round_digits)
    out_mock.assert_called_once_with(data)


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
        blocks_mock.assert_called_once_with(df, 3)

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

    # Check block type and count
    assert isinstance(blocks, list)
    assert len(blocks) == 3
    assert all(isinstance(b, Block) for b in blocks)

    # Check block headers
    titles = [b.block_config.title for b in blocks]
    assert "Basic statistics for the dataset." in titles
    assert "Dataset shape" in titles
    assert "Data quality quick summary" in titles

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