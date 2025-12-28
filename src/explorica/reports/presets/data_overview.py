"""
Short data overview presets for Explorica reports.

This module provides utilities to quickly build a short overview of a dataset,
including basic statistics, data shape, and a brief data quality summary.
It is intended for use with the Explorica reports framework.

Functions
---------
get_data_overview_blocks(data, round_digits=4)
    Build blocks for a short data overview. Returns a list of Block instances
    containing basic statistics, data shape, and data quality overview.
get_data_overview_report(data, round_digits=4)
    Generate a short data overview report. Returns a Report instance
    composed of the blocks returned by `get_data_overview_blocks`.

Notes
-----
- These functions are designed for quick, high-level EDA and should not be
  used as a replacement for full data quality or interactions analysis.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.data_overview import get_data_overview_report
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
>>> report = get_data_overview_report(df)
>>> report.title
'Data overview'
>>> len(report.blocks)
3
"""

from typing import Sequence, Mapping, Any

from ..core.block import Block
from ..core.report import Report
from .blocks import get_ctm_block, get_data_quality_overview_block, get_data_shape_block


def get_data_overview_blocks(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
) -> list[Block]:
    """
    Build blocks for a short data overview.

    This function constructs a sequence of Explorica `Block` objects that
    together provide a concise exploratory overview of a dataset. The resulting
    blocks can be combined with other blocks before rendering.

    The overview includes:
    - basic descriptive statistics,
    - dataset shape and data types,
    - a brief data quality summary.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Can be any structure convertible to a pandas DataFrame,
        such as a dictionary of sequences or a sequence of records.
    round_digits : int, default=4
        Number of decimal digits to use when rounding numerical statistics.

    Returns
    -------
    list of Block
        A list of Explorica blocks representing a short data overview.

    Notes
    -----
    This function does not return a `Report` instance and does not perform any
    rendering. It is intended for compositional use, allowing users to merge
    the returned blocks with other presets before building a final report.

    Examples
    --------
    >>> blocks = get_data_overview_blocks(df)
    >>> len(blocks)
    3
    >>> blocks[0].block_config.title
    'Basic statistics for the dataset.'
    """
    blocks = [
        get_ctm_block(data, round_digits=round_digits),
        get_data_shape_block(data),
        get_data_quality_overview_block(data, round_digits=round_digits),
    ]
    return blocks


def get_data_overview_report(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
) -> list[Block] | Report:
    """
    Generate a short data overview report.

    This function creates a complete Explorica `Report` that provides a concise
    exploratory overview of a dataset. Internally, it builds the same blocks as
    `get_data_overview_blocks` and wraps them into a report ready for rendering
    (e.g., to HTML or PDF).

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Can be any structure convertible to a pandas DataFrame,
        such as a dictionary of sequences or a sequence of records.
    round_digits : int, default 4
        Number of decimal digits to use when rounding numerical statistics.

    Returns
    -------
    Report
        An Explorica report containing:
        - basic descriptive statistics,
        - dataset shape and data types,
        - a brief data quality summary.

    Notes
    -----
    This is a convenience wrapper around `get_data_overview_blocks`. Use
    `get_data_overview_blocks` instead if you need fine-grained control over
    block composition before report creation.

    Examples
    --------
    >>> report = get_data_overview_report(df)
    >>> report.title
    'Data overview'
    >>> len(report.blocks)
    3
    """
    return Report(
        blocks=get_data_overview_blocks(data, round_digits=round_digits),
        title="Data overview",
        description=(
            "Short overview of the dataset. "
            "Includes basic statistics, data shape, "
            "and a brief data quality summary."
        ),
    )
