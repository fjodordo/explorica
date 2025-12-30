"""
Distributions block preset.

Provides an overview of feature distributions as an Explorica `Block`,
including numerical distribution descriptors and visual diagnostics.
The block focuses on skewness, kurtosis, normality flags, and per-feature
distribution visualizations.

Functions
---------
get_distributions_block(data, threshold_skewness=0.25, threshold_kurtosis=0.25,
    round_digits=4, nan_policy="drop"
)
    Build a Block instance summarizing feature distributions in a dataset.

Notes
-----
- The block operates on numerical columns only.
- Skewness and excess kurtosis are used to assess distribution shape
  and deviation from normality.
- Normality flags are derived using user-defined skewness and kurtosis
  thresholds.
- Boxplots and distribution plots are rendered per feature; the first
  visualization in each group provides a group-level title.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.distributions import (
...     get_distributions_block
... )
>>> df = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [10, 10, 10, 10, 10]
... })
>>> block = get_distributions_block(df)
>>> block.block_config.title
'Distributions'
>>> [table.title for table in block.block_config.tables]
['Skewness and excess kurtosis']
"""

from typing import Sequence, Mapping, Any, Literal
import warnings

import numpy as np

from ...._utils import convert_dataframe, handle_nan
from ....types import TableResult
from ...core.block import Block, BlockConfig
from ....data_quality import describe_distributions
from ....visualizations import distplot, boxplot


def get_distributions_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    threshold_skewness: float = 0.25,
    threshold_kurtosis: float = 0.25,
    round_digits: int = 4,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> Block:
    """
    Generate a `Block` summarizing feature distributions in a dataset.

    This block provides an overview of numeric features, including:
    - Skewness and excess kurtosis metrics in a table,
      with an `is_normal` flag according to provided thresholds.
    - Boxplots for all numeric features, plus individual boxplots per feature.
    - Distribution plots (histograms + optional KDE) for all numeric features.

    The block is intended for exploratory data analysis and can be
    combined with other blocks (e.g., data quality, outliers) in reports.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Must be convertible to a pandas DataFrame.
        Only numeric columns are used for analysis.
    threshold_skewness : float, default=0.25
        Maximum absolute skewness value
        for a feature to be considered approximately normal.
    threshold_kurtosis : float, default=0.25
        Maximum absolute excess kurtosis value
        for a feature to be considered approximately normal.
    round_digits : int, default=4
        Number of decimal digits for skewness and kurtosis values in the table.
    nan_policy : {'drop', 'raise'}, default 'drop'
        Policy for handling missing values:
        - 'drop' : remove rows with missing values.
        - 'raise': raise an error if missing values are present.

    Returns
    -------
    Block
        An Explorica `Block` containing:
        - A table with skewness, excess kurtosis, and `is_normal`
          flags for numeric features.
        - Boxplots for all numeric features and individual boxplots per feature.
        - Distribution plots (histograms + optional KDE) for all numeric features.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_distributions_block
    >>> df = pd.DataFrame({
    ...     "a": [1, 2, 3, 4, 5],
    ...     "b": [2, 2, 3, 4, 5]
    ... })
    >>> block = get_distributions_block(df)
    >>> block.block_config.tables[0].table
       skewness  kurtosis  is_normal                       desc
    a    0.0000    -1.300      False                low-pitched
    b    0.3632    -1.372      False  right-skewed, low-pitched
    """
    df = handle_nan(
        convert_dataframe(data).select_dtypes("number"), nan_policy=nan_policy
    )

    # Describe skewness-kurtosis
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="explorica.data_quality.outliers.stats",
        )
        sk_describe = describe_distributions(
            df,
            threshold_skewness=threshold_skewness,
            threshold_kurtosis=threshold_kurtosis,
        )
    sk_describe["kurtosis"] = np.round(
        sk_describe["kurtosis"],
        decimals=round_digits,
    )
    sk_describe["skewness"] = np.round(
        sk_describe["skewness"],
        decimals=round_digits,
    )
    sk_describe["is_normal"] = sk_describe["is_normal"].astype("bool")
    sk_table = TableResult(sk_describe, title="Skewness and excess kurtosis")
    block = Block(BlockConfig(title="Distributions"))
    block.add_table(sk_table)

    # Add boxplots
    boxplots = [
        boxplot(
            df[feature],
            title=f"boxplot for '{feature}' feature",
            figsize=(5, 3),
            palette="Set1",
            style="whitegrid",
        )
        for feature in df.columns
    ]
    for i, vr in enumerate(boxplots):
        if i == 0:
            vr.title = "Boxplots for all features"
        block.add_visualization(vr)

    # Add distplots
    boxplots = [
        distplot(
            df[feature],
            title=f"distplot for '{feature}' feature",
            figsize=(5, 3),
            palette="Set1",
            xlabel="value",
            ylabel="frequency",
            style="whitegrid",
        )
        for feature in df.columns
    ]
    for i, vr in enumerate(boxplots):
        if i == 0:
            vr.title = "Distplots for all features"
        block.add_visualization(vr)

    return block
