"""
Outliers block preset.

Provides an overview of outliers in numerical features as an Explorica
`Block`. The block summarizes outliers detected using multiple statistical
methods, allowing comparison of their sensitivity and coverage.

Functions
---------
get_outliers_block(data, iqr_factor=1.5, zscore_factor=3.0, nan_policy="drop")
    Build a `Block` instance containing a table summarizing
    outliers detected by different methods.

Notes
-----
- The block operates on numerical columns only.
- Outliers are detected independently for each feature.
- The interquartile range (IQR) method uses `iqr_factor` to control
  sensitivity to extreme values.
- The Z-score method uses `zscore_factor` as a threshold for standardized
  deviation.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.outliers import get_outliers_block
>>> df = pd.DataFrame({
...     'a': [1, 2, 3, 100],
...     'b': [10, 11, 12, 13]
... })
>>> block = get_outliers_block(df)
>>> block.block_config.title
'Outliers'
>>> [table.title for table in block.block_config.tables]
['Count of outliers by different detection methods']
"""

from typing import Sequence, Mapping, Any, Literal
import warnings

import numpy as np
import pandas as pd

from ...._utils import handle_nan
from ....types import TableResult
from ...core.block import Block, BlockConfig
from ....data_quality import detect_iqr, detect_zscore


def get_outliers_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    iqr_factor: float = 1.5,
    zscore_factor: float = 3.0,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> Block:
    """
    Generate a `Block` summarizing outliers detected by different methods.

    This block provides a compact overview of potential outliers in numeric
    features using multiple detection strategies. Currently, it includes
    counts of outliers detected by the IQR and Z-score methods.

    If features with zero or near-zero variance are present, an additional
    table is included to explicitly report such features, as outliers cannot
    exist in constant series.

    The resulting block is intended for exploratory data analysis and can be
    composed with other blocks (e.g., distribution or data quality blocks)
    in higher-level reports.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Must be convertible to a pandas DataFrame.
        Only numeric columns are considered for outlier detection.
    iqr_factor : float, default 1.5
        Scaling factor used for the IQR-based outlier detection.
    zscore_factor : float, default 3.0
        Threshold (in standard deviations) used for Z-score-based
        outlier detection.
    nan_policy : {'drop', 'raise'}, default 'drop'
        Policy for handling missing values prior to outlier detection:
        - 'drop' : remove rows with missing values.
        - 'raise': raise an error if missing values are present.

    Returns
    -------
    Block
        An Explorica `Block` containing a single table:
        - "Count of outliers by different detection methods":
          a table indexed by feature name, where each column corresponds
          to an outlier detection method.
        - 'Features with zero or near zero variance"' (optional):
          A table listing features whose variance is zero or numerically
          close to zero. This table is included only if such features are
          detected in the dataset.

    Notes
    -----
    The block is intentionally minimal and currently focuses on outlier
    counts only. It is designed to be extensible, allowing additional
    detection methods or related summaries to be added in the future.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_outliers_block
    >>> df = pd.DataFrame({"x": [1, 2, 3, 100]})
    >>> block = get_outliers_block(df)
    >>> block.block_config.tables[0].table
          IQR (1.5)  Z-Score (3σ)
    x              1             1
    """
    df = handle_nan(data, nan_policy, is_dataframe=False).select_dtypes("number")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="explorica.data_quality.outliers.detection",
        )
        zscore_outliers = detect_zscore(df, threshold=zscore_factor)
        iqr_outliers = detect_iqr(df, iqr_factor=iqr_factor)
    zscore_outliers = zscore_outliers.count()
    iqr_outliers = iqr_outliers.count()
    if not isinstance(zscore_outliers, pd.Series):
        zscore_outliers = pd.Series([zscore_outliers], index=df.columns)
    if not isinstance(iqr_outliers, pd.Series):
        iqr_outliers = pd.Series([iqr_outliers], index=df.columns)

    outliers_table = TableResult(
        pd.DataFrame(
            {
                f"IQR ({iqr_factor})": iqr_outliers,
                f"Z-Score ({zscore_factor}σ)": zscore_outliers,
            },
        )
    )

    variance_series = np.var(df, axis=0, ddof=0)
    variance_series = variance_series[np.isclose(variance_series, 0.0, atol=1e-10)]

    block = Block(BlockConfig(title="Outliers"))
    block.add_table(
        outliers_table, title="Count of outliers by different detection methods"
    )
    if variance_series.size > 0:
        variance_table = TableResult(
            pd.DataFrame({"feature_name": variance_series.index}),
            title="Features with zero or near zero variance",
            description=(
                "If any features have constant or nearly constant values, "
                "as outliers cannot exist in such series."
            ),
            render_extra={"show_index": False},
        )
        block.add_table(variance_table)

    return block
