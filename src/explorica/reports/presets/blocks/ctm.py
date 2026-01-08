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
    nan_policy: Literal["drop_with_split", "raise"] = "drop_with_split",
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
    nan_policy : {'drop_with_split', 'raise'}, default='drop'
        Policy to handle missing values:
        - 'drop_with_split' :
          Missing values are handled independently for each feature.
          For every column, NaNs are dropped column-wise before computing
          statistics. As a result, different features may be evaluated
          on different numbers of observations.
          This behavior is semantically correct in an EDA context, where
          preserving per-feature statistics is preferred over strict
          row-wise alignment.
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
    feature_dict: dict[pd.Series] = handle_nan(
        df, nan_policy=nan_policy, supported_policy=["drop_with_split", "raise"]
    )
    number_keys: list[str] = [
        key
        for key in feature_dict.keys()
        if (
            pd.api.types.is_numeric_dtype(feature_dict[key])
            and not pd.api.types.is_bool_dtype(feature_dict[key])
        )
    ]

    block = Block(BlockConfig(title="Basic statistics for the dataset"))
    table_ctm1 = TableResult(
        pd.DataFrame(
            {
                "mean": np.round(
                    pd.Series(
                        [np.mean(feature_dict[key], axis=0) for key in number_keys],
                        index=number_keys,
                    ),
                    round_digits,
                ),
                "median": np.round(
                    pd.Series(
                        [np.median(feature_dict[key], axis=0) for key in number_keys],
                        index=number_keys,
                    ),
                    round_digits,
                ),
            }
        ),
        title="Central tendency measures",
    )
    table_ctm2 = TableResult(
        pd.DataFrame(
            {
                "mode": pd.Series(
                    [
                        feature_dict[key].mode(dropna=False).iloc[0]
                        for key in list(feature_dict)
                    ],
                    index=list(feature_dict),
                ),
            }
        ),
        description="Includes categorical columns",
    )
    table_var = TableResult(
        pd.DataFrame(
            {
                "std": np.round(
                    pd.Series(
                        [
                            np.std(feature_dict[key], axis=0, ddof=0)
                            for key in number_keys
                        ],
                        index=number_keys,
                    ),
                    round_digits,
                ),
                "min": pd.Series(
                    [np.min(feature_dict[key], axis=0) for key in number_keys],
                    index=number_keys,
                ),
                "max": pd.Series(
                    [np.max(feature_dict[key], axis=0) for key in number_keys],
                    index=number_keys,
                ),
                "range": np.round(
                    pd.Series(
                        [
                            np.max(feature_dict[key], axis=0)
                            - np.min(feature_dict[key], axis=0)
                            for key in number_keys
                        ],
                        index=number_keys,
                    ),
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
