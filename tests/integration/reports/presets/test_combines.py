import pandas as pd

from explorica.reports.presets.data_overview import get_data_overview_report, get_data_overview_blocks
from explorica.reports.core.block import Block, BlockConfig
from explorica.reports.core.report import Report

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

    # check type and attrs
    assert isinstance(report, Report)
    assert report.blocks[0].block_config.title == dummy_blocks[0].block_config.title
    assert report.blocks[1].block_config.title == dummy_blocks[1].block_config.title
    assert report.title == "Data overview"
    assert "Short overview of the dataset" in report.description
