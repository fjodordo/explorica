"""
Central tendency & dispersion block preset.

Provides a summary of the dataset's central tendency (mean, median, mode)
and dispersion (std, min, max, range) as an Explorica `Block`.

Functions
---------
get_ctm_block(data, nan_policy='drop', round_digits=4)
    Build a `Block` instance containing tables of basic statistics for the dataset.

Notes
-----
- Designed for quick, high-level overview in Explorica reports.
- Mode calculation includes categorical columns.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.ctm import get_ctm_block
>>> df = pd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']})
>>> block = get_ctm_block(df)
>>> block.block_config.title
'Basic statistics for the dataset.'
>>> [table.title for table in block.block_config.tables]
['Central tendency measures', 'Includes categorical columns', 'Dispersion measures']
"""

from typing import Sequence, Mapping, Any, Literal
import numpy as np
import pandas as pd

from ...._utils import convert_dataframe, handle_nan
from ....types import TableResult
from ...core.block import Block, BlockConfig


def get_ctm_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    nan_policy: Literal["drop", "raise"] = "drop",
    round_digits: int = 4,
) -> Block:
    """
    Generate a `Block` containing central tendency and dispersion statistics
    for a dataset.

    Parameters
    ----------
    data : Sequence or Mapping of sequences
        Input dataset. Can be a list of sequences or a dictionary of column names
        to sequences. Will be converted to a pandas DataFrame internally.
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy to handle missing values:
        - 'drop' : drop rows containing NaNs before computing statistics.
        - 'raise' : raise an error if NaNs are present.
    round_digits : int, default=4
        Number of decimal places to round numerical results.

    Returns
    -------
    Block
        A `Block` object containing the following tables:
        - "Central tendency measures": mean and median for numerical columns.
        - "Mode": mode for all columns, including categorical.
        - "Dispersion measures": standard deviation, minimum, maximum,
          and range for numerical columns.

    Notes
    -----
    - The function automatically separates numerical and categorical columns.

    Examples
    --------
    >>> from explorica.reports.presets.blocks import get_ctm_block
    >>> data = {'a': [1, 2, 3, 4], 'b': [5, 5, 6, 6]}
    >>> block = get_ctm_block(data)
    >>> block.tables  # contains central tendency and dispersion tables
    """
    df = convert_dataframe(data)
    df = handle_nan(df, nan_policy=nan_policy)
    number_columns = df.select_dtypes("number")
    block = Block(BlockConfig(title="Basic statistics for the dataset."))
    table_ctm1 = TableResult(
        pd.DataFrame(
            {
                "mean": np.round(np.mean(number_columns, axis=0), round_digits),
                "median": np.median(number_columns, axis=0),
            }
        ),
        title="Central tendency measures",
    )
    table_ctm2 = TableResult(
        pd.DataFrame(
            {
                "mode": df.mode(dropna=False).iloc[0],
            }
        ),
        description="Includes categorical columns",
    )
    table_var = TableResult(
        pd.DataFrame(
            {
                "std": np.round(np.std(number_columns, axis=0), round_digits),
                "min": np.min(number_columns, axis=0),
                "max": np.max(number_columns, axis=0),
                "range": np.round(
                    np.max(number_columns, axis=0) - np.min(number_columns, axis=0),
                    round_digits,
                ),
            }
        ),
        title="Dispersion measures",
    )
    block.add_table(table_ctm1)
    block.add_table(table_ctm2)
    block.add_table(table_var)
    return block
