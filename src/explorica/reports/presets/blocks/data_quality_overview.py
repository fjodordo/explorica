"""
Data quality overview block preset.

Provides a quick summary of a dataset's quality, including duplicated rows
and counts/ratios of missing values. Designed for a fast, high-level exploratory
analysis in Explorica reports.

Functions
---------
get_data_quality_overview_block(data, round_digits=4)
    Build a `Block` instance containing metrics and a table summarizing
    duplicated rows and NaN counts/ratios.

Notes
-----
- Intended for quick inspection; use `data_quality` module for more detailed analysis.
- `round_digits` controls numeric precision in NaN ratio calculations.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.data_quality_overview import (
...     get_data_quality_overview_block,)
>>> df = pd.DataFrame({'a': [1, None, 2], 'b': ['x', 'y', None]})
>>> block = get_data_quality_overview_block(df)
>>> block.block_config.title
'Data quality quick summary'
>>> [m['name'] for m in block.block_config.metrics]
['Duplicates rows', 'Duplicates ratio']
>>> block.block_config.tables[0].title
"NaN's count & ratio"
"""

from typing import Sequence, Mapping, Any
import numpy as np

from ....data_quality import get_missing
from ...._utils import convert_dataframe
from ....types import TableResult
from ...core.block import Block, BlockConfig


def get_data_quality_overview_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
) -> Block:
    """
    Generate a quick data quality overview block.

    This block provides a concise summary of the dataset's quality, including
    duplicated rows and missing values. It is intended for fast exploratory analysis
    without going into detailed data quality checks.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        The input dataset. Can be a list of sequences, a dictionary of sequences,
        or any pandas-compatible structure convertible to a DataFrame.
    round_digits : int, default 4
        Number of decimal places to round ratios (e.g., NaN ratios, duplicate ratio).

    Returns
    -------
    Block
        An Explorica Block containing:
        - Metrics:
            - "Duplicates rows": number of duplicated rows in the dataset.
            - "Duplicates ratio": ratio of duplicated rows to total rows.
        - Table:
            - "NaN's count & ratio": table summarizing count and ratio of missing
              values per column.

    Notes
    -----
    - The "Duplicates ratio" is np.nan for empty datasets.
    - NaN ratios and counts are rounded to `round_digits` decimal places.
    - This block is meant to give a quick overview and does not replace
      a detailed data quality analysis.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_data_quality_overview_block
    >>> df = pd.DataFrame({
    ...     "a": [1, 2, 2, None],
    ...     "b": ["x", "y", "y", "z"]
    ... })
    >>> block = get_data_quality_overview_block(df)
    >>> metrics = {m['name']: m['value'] for m in block.block_config.metrics}
    >>> metrics["Duplicates rows"]
    1
    >>> metrics["Duplicates ratio"]
    0.25
    >>> block.block_config.tables[0].table
       count_of_nans  nan_ratio
    a              1       0.25
    b              0       0.00
    """
    df = convert_dataframe(data)

    block = Block(BlockConfig(title="Data quality quick summary"))
    block.add_metric(value=df[df.duplicated()].shape[0], name="Duplicates rows")
    if df.shape[0] == 0:
        # ensures protection in 0 rows case
        duplicated_ratio = np.nan
    else:
        duplicated_ratio = np.round(
            df[df.duplicated()].shape[0] / df.shape[0], round_digits
        )
    block.add_metric(value=duplicated_ratio, name="Duplicates ratio")
    nans = TableResult(
        get_missing(df, round_digits=round_digits).rename(
            {"pct_of_nans": "nan_ratio", "count_of_nans": "nan_count"}, axis=1
        ),
        title="NaN's count & ratio",
    )
    block.add_table(nans)
    return block
