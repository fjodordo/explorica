"""
Data shape block preset.

Provides a quick summary of the dataset shape, including number of rows,
columns, column types, and positional index check.

Functions
---------
get_data_shape_block(data, nan_policy='drop')
    Build a Block instance summarizing the dataset shape.

Notes
-----
- Designed for quick, high-level overview in Explorica reports.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.data_shape import get_data_shape_block
>>> df = pd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']})
>>> block = get_data_shape_block(df)
>>> block.block_config.title
'Dataset shape'
"""

from typing import Sequence, Mapping, Any, Literal
import numpy as np

from ...._utils import convert_dataframe, handle_nan
from ....types import TableResult
from ...core.block import Block, BlockConfig


def get_data_shape_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    nan_policy: Literal["drop", "raise", "include"] = "drop",
) -> Block:
    """
    Generate a data shape block.

    This block provides an overview of the dataset's structural properties,
    including the number of rows and columns, the distribution of column types,
    and information about the dataset index.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        The input dataset. Can be a list of sequences, a dictionary of sequences,
        or a pandas-compatible structure convertible to a DataFrame.
    nan_policy : {'drop', 'raise', 'include'}, default 'drop'
        Policy for handling missing values:
        - 'drop' : remove rows with NaN values before computing metrics.
        - 'raise': raise an error if NaN values are present.
        - 'include': keep rows with NaN values; they do not interfere with
          computation of structural metrics or column type counts.

    Returns
    -------
    Block
        An Explorica Block containing:
        - Metrics:
            - "Rows": number of rows in the dataset.
            - "Columns": number of columns in the dataset.
            - "Index is positional": boolean indicating if the index behaves as
              a non-negative integer positional index (unique, integer, starting at 0).
        - Table:
            - "Data types": a table summarizing the count of columns per data type,
              sorted descending by number of features.

    Notes
    -----
    The "Index is positional" metric uses a heuristic to determine if the index
    can be interpreted as a simple positional index, which is robust to missing
    rows or non-consecutive integer indices.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets import get_data_shape_block
    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': ['x', 'y', 'z']
    ... })
    >>> block = get_data_shape_block(df)
    >>> block.block_config.metrics
    [{'name': 'Rows', 'value': 3, 'description': None},
     {'name': 'Columns', 'value': 2, 'description': None},
     ...,
    ]
    >>> block.block_config.tables[0].table
       dtype  n_features
    0  int64           1
    1  object          1
    """
    df = convert_dataframe(data)
    df = handle_nan(
        df, nan_policy=nan_policy, supported_policy=("drop", "raise", "include")
    )
    block = Block(BlockConfig(title="Dataset shape"))
    block.add_metric("Rows", df.shape[0])
    block.add_metric("Columns", df.shape[1])
    data_types = TableResult(
        df.dtypes.astype("str")
        .reset_index()
        .groupby(0)["index"]
        .count()
        .sort_values(ascending=False)
        .reset_index()
        .rename({0: "dtype", "index": "n_features"}, axis=1),
        title="Data types",
        render_extra={"show_index": False},
    )
    block.add_table(data_types)
    is_index_default = (
        df.index.is_unique
        and np.issubdtype(df.index.dtype, np.integer)
        and df.index.min() >= 0
    )
    block.add_metric(
        "Index is positional",
        value=is_index_default,
        description="Index behaves as a non-negative integer positional index",
    )
    return block
