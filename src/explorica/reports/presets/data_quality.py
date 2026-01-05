"""
Data quality presets for Explorica reports.

Provides high-level preset builders for composing comprehensive
data quality reports in Explorica. These presets orchestrate multiple
specialized blocks into a single, ordered analysis pipeline suitable
for exploratory data analysis (EDA).

Functions
---------
get_data_quality_blocks(data, round_digits=4)
    Build blocks for a detailed data quality analysis.
get_data_quality_report(data, round_digits=4)
    Generate a detailed data quality analysis report.

Notes
-----
- This module acts as an orchestration layer and does not perform
  computations directly.
- Each block included in the report is responsible for a distinct
  data quality dimension (e.g. cardinality, distributions, outliers).
- The order of blocks is intentional and reflects a typical EDA flow,
  from coarse feature screening to more detailed statistical inspection.
- All blocks share common formatting and rounding conventions
  via the `round_digits` parameter.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.data_quality import (
...     get_data_quality_blocks,
...     get_data_quality_report
... )
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 1, 1]})
>>> blocks = get_data_quality_blocks(df)
>>> len(blocks) > 0
True
>>> report = get_data_quality_report(df)
>>> report.title
'Data quality'
"""

from typing import Sequence, Mapping, Any

from ..core.block import Block
from ..core.report import Report
from .blocks import get_cardinality_block, get_distributions_block, get_outliers_block


def get_data_quality_blocks(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
) -> list[Block]:
    """
    Build a set of blocks providing a detailed data quality analysis.

    This function orchestrates multiple data quality blocks, including
    cardinality, distributions, and outliers, into a coherent sequence
    that can be used to assemble a full data quality report.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Can be any structure convertible to a pandas DataFrame,
        such as a dictionary of sequences or a sequence of records.
    round_digits : int, default 4
        Number of decimal digits to use when rounding numerical statistics.

    Returns
    -------
    list[Block]
        A list of `Block` instances for data quality analysis.

    Examples
    --------
    >>> blocks = get_data_quality_blocks(df)
    >>> len(blocks)
    3
    """
    blocks = [
        get_outliers_block(data),
        get_distributions_block(data, round_digits=round_digits),
        get_cardinality_block(data, round_digits=round_digits),
    ]
    return blocks


def get_data_quality_report(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
) -> Report:
    """
    Generate a full data quality report from multiple quality blocks.

    This function orchestrates the assembly of a complete `Report` containing
    cardinality, distributions, and outlier analysis blocks. It provides
    an overview of the dataset's quality characteristics in a structured format.

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
        - outlier summary (IQR and Z-score counts, zero/near-zero variance features),
        - distribution characteristics
          (skewness, kurtosis, normality flag, boxplots, histograms),
        - cardinality metrics
          (unique values, top value ratio, entropy, is_constant/is_unique flags).

    Notes
    -----
    This is a convenience wrapper around `get_data_overview_blocks`. Use
    `get_data_overview_blocks` instead if you need fine-grained control over
    block composition before report creation.

    Examples
    --------
    >>> report = get_data_quality_report(df)
    >>> report.title
    'Data quality'
    >>> len(report.blocks)
    3
    """
    return Report(
        blocks=get_data_quality_blocks(data, round_digits),
        title="Data quality",
        description=(
            "A comprehensive summary of dataset quality, including missing values, "
            "outliers, cardinality, and distribution characteristics."
        ),
    )
