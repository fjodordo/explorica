"""
Data cardinality block preset.

Provides a summary of feature cardinality, uniqueness, and constancy
characteristics as an Explorica `Block`. The block helps identify constant,
unique, and low-information features using multiple complementary metrics.

Functions
---------
get_cardinality_block(data, round_digits=4, nan_policy="drop")
    Build a Block instance summarizing feature cardinality and constancy metrics.

Notes
-----
- Cardinality is described using both absolute and relative measures,
  including number of unique values and their ratios.
- Constant and unique features are flagged explicitly via boolean indicators.
- Entropy is reported in normalized form to allow comparison across
  features with different cardinalities.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.cardinality import get_cardinality_block
>>> df = pd.DataFrame({
...     'a': [1, 1, 1, 1],
...     'b': [1, 2, 3, 4],
...     'c': ['x', 'x', 'y', 'y']
... })
>>> block = get_cardinality_block(df)
>>> block.block_config.title
'Cardinality'
>>> [table.title for table in block.block_config.tables]
['Constancy | uniqueness metrics']
"""

from typing import Sequence, Mapping, Any, Literal

import numpy as np
import pandas as pd

from ...._utils import handle_nan, convert_dataframe
from ....types import TableResult
from ...core.block import Block, BlockConfig
from ....data_quality import get_entropy


def get_cardinality_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
    nan_policy: (
        str | Literal["drop_with_split", "raise", "include"]
    ) = "drop_with_split",
) -> Block:
    """
    Generate a `Block` summarizing feature cardinality and constancy metrics.

    This block provides an overview of uniqueness and constant-like behavior
    in the dataset. It is intended for exploratory data analysis and data
    quality assessment, helping identify features with low variability,
    redundant values, or high uniqueness.

    The block contains a single table with the following columns:
    - `is_unique` : indicates if a feature has all unique values.
    - `is_constant` : indicates if a feature has a single unique value.
    - `n_unique` : number of unique values in the feature.
    - `unique_ratio` : ratio of unique values to the number of rows.
    - `top_value_ratio` : proportion of the most frequent value.
    - `entropy (normalized)` : Shannon entropy of the feature
      normalized by log2(n_unique), measuring information content
      and effective cardinality.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Must be convertible to a pandas DataFrame.
        Both numeric and categorical columns are supported.
    round_digits : int, default=4
        Number of decimal places to round ratio and entropy metrics.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling missing values:
        - 'drop_with_split' :
          Missing values are handled independently for each feature.
          For every column, NaNs are dropped column-wise before computing
          statistics. As a result, different features may be evaluated
          on different numbers of observations.
          This behavior is semantically correct in an EDA context, where
          preserving per-feature statistics is preferred over strict
          row-wise alignment.
        - 'raise': raise an error if any missing values are present.
        - 'include': treat NaN as a valid category (counts towards uniqueness
          and entropy calculations).

    Returns
    -------
    Block
        An Explorica `Block` containing a single table with cardinality
        and constancy metrics for each feature.

    Notes
    -----
    - Features with zero variance or all missing values will appear as NaN
      in relevant metrics.
    - This block complements data quality overview blocks by providing
      a deeper view of feature redundancy and variability.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_cardinality_block
    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 3, 4],
    ...     "B": [5, 5, 5, 5],
    ...     "C": [1, 2, 2, 3],
    ...     "D": [None, 1, None, 1]
    ... })
    >>> block = get_cardinality_block(df, nan_policy="include")
    >>> block.block_config.tables[0].table
       is_unique  is_constant  n_unique ... top_value_ratio  entropy (normalized)
    A       True        False         2 ...             0.5                   1.0
    B      False         True         1 ...             1.0                   NaN
    C       True        False         2 ...             0.5                   1.0
    D      False         True         1 ...             1.0                   NaN
    """
    # NaN handling is fully delegated to nan_policy.
    # At this stage, NaNs are either intentionally preserved ("include")
    # or already removed per-feature ("drop_with_split").
    dict_of_series = handle_nan(
        convert_dataframe(data),
        nan_policy,
        supported_policy=("drop_with_split", "raise", "include"),
    )
    if nan_policy == "include":
        dict_of_series = {feat: dict_of_series[feat] for feat in dict_of_series.columns}
    # From this point on, dict_of_series is guaranteed to be dict[str, pd.Series]

    nunique = {key: dict_of_series[key].nunique(dropna=False) for key in dict_of_series}
    cardinality_table = pd.DataFrame(
        {
            "is_unique": pd.Series(
                {
                    feat: (
                        len(dict_of_series[feat]) == nunique[feat]
                        if nunique[feat] != 0
                        else np.nan
                    )
                    for feat in dict_of_series
                }
            ),
            "is_constant": pd.Series(
                {
                    feat: nunique[feat] == 1 if nunique[feat] != 0 else np.nan
                    for feat in nunique
                }
            ),
            "n_unique": pd.Series({feat: nunique[feat] for feat in nunique}),
            "unique_ratio": np.round(
                pd.Series(
                    {
                        key: (
                            nunique[key] / len(dict_of_series[key])
                            if len(dict_of_series[key]) != 0
                            else np.nan
                        )
                        for key in dict_of_series
                    }
                ),
                round_digits,
            ),
            "top_value_ratio": np.round(
                pd.Series(
                    {
                        feat: (
                            dict_of_series[feat].value_counts(dropna=False).max()
                            / len(dict_of_series[feat])
                            if len(dict_of_series[feat]) != 0
                            else np.nan
                        )
                        for feat in dict_of_series
                    }
                ),
                round_digits,
            ),
            "entropy (normalized)": np.round(
                pd.Series(
                    {
                        feat: (
                            get_entropy(dict_of_series[feat]) / np.log2(nunique[feat])
                            if nunique[feat] > 1
                            else np.nan
                        )
                        for feat in dict_of_series
                    }
                ),
                round_digits,
            ),
        }
    )

    cardinality_table = TableResult(cardinality_table, "Constancy | uniqueness metrics")
    block = Block(BlockConfig(title="Cardinality"))
    block.add_table(cardinality_table)
    return block
