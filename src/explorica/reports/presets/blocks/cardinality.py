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

from ...._utils import handle_nan
from ....types import TableResult
from ...core.block import Block, BlockConfig
from ....data_quality import get_entropy


def get_cardinality_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
    nan_policy: str | Literal["drop", "raise", "include"] = "drop",
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
        - 'drop' : remove rows with missing values.
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
    df = handle_nan(
        data,
        nan_policy,
        supported_policy=("drop", "raise", "include"),
        is_dataframe=False,
    )
    if df.shape[0] == 0:
        # Add placeholder table
        cardinality_table = pd.DataFrame(
            {
                "is_unique": np.nan,
                "is_constant": np.nan,
                "n_unique": 0,
                "unique_ratio": np.nan,
                "top_value_ratio": np.nan,
                "entropy (normalized)": np.nan,
            },
            index=df.columns,
        )
    else:
        nunique = df.nunique(axis=0, dropna=False)
        entropy = get_entropy(df, nan_policy=nan_policy)
        if not isinstance(entropy, pd.Series):
            entropy = pd.Series(entropy, index=df.columns)
        cardinality_table = pd.DataFrame(
            {
                "is_unique": nunique.map(lambda row: row == df.shape[0]),
                "is_constant": nunique.map(lambda row: row == 1),
                "n_unique": nunique,
                "unique_ratio": np.round(nunique / df.shape[0], round_digits),
                "top_value_ratio": np.round(
                    df.apply(
                        (
                            lambda col: col.value_counts(dropna=False).max()
                            / df.shape[0]
                        ),
                        axis=0,
                    ),
                    round_digits,
                ),
                "entropy (normalized)": np.round(
                    (
                        pd.Series(
                            {
                                col: (
                                    entropy[col] / np.log2(nunique[col])
                                    if nunique[col] > 1
                                    else np.nan
                                )
                                for col in df.columns
                            }
                        )
                    ),
                    round_digits,
                ),
            }
        )

    cardinality_table = TableResult(cardinality_table, "Constancy | uniqueness metrics")
    block = Block(BlockConfig(title="Cardinality"))
    block.add_table(cardinality_table)
    return block
