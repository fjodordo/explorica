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

from typing import Sequence, Mapping, Any, Literal
import warnings

from ..core.block import Block
from ..core.report import Report
from .blocks import get_cardinality_block, get_distributions_block, get_outliers_block


def get_data_quality_blocks(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
    nan_policy: str | Literal["drop", "raise", "include"] = "drop",
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
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling missing values in the input data.
        - 'drop' : rows containing missing values are removed before
          analysis.
        - 'raise' : an error is raised if missing values are present.
        - 'include' : missing values are preserved where supported.
        Note that not all child blocks support nan_policy='include'.
        In such cases, the policy is internally downgraded to 'drop' for
        those blocks.

    Returns
    -------
    list[Block]
        A list of `Block` instances for data quality analysis.

    Notes
    -----
    - During the construction of EDA or interaction reports, many matplotlib figures
      may be opened (one per plot or table visualization). This is expected behavior
      when the dataset contains many features.
    - To prevent runtime warnings about too many open figures, these warnings are
      ignored internally.
    - To free memory after rendering, it is recommended to explicitly close figures:

      >>> report = get_eda_report(df)
      >>> report.render()
      >>> report.close_figures()

      or for individual blocks:

      >>> block = some_block
      >>> block.render()
      >>> block.close_figures()

    Examples
    --------
    >>> blocks = get_data_quality_blocks(df)
    >>> len(blocks)
    3
    """
    # We ignore mpl runtime warnings because EDA reports may open many figures.
    # It's assumed, that the user use ``Report.close_figures()``
    # and ``Block.close_figures`` after rendering
    blocks = []
    with warnings.catch_warnings():

        warnings.filterwarnings(
            action="ignore",
            module="explorica.visualizations",
            message="More than 20 figures have been opened.",
            category=RuntimeWarning,
        )
        outliers_policy = nan_policy if nan_policy != "include" else "drop"
        outliers_policy = (
            "drop_with_split" if outliers_policy == "drop" else outliers_policy
        )
        blocks.extend(
            [
                get_outliers_block(data, nan_policy=outliers_policy),
                get_distributions_block(
                    data,
                    round_digits=round_digits,
                    nan_policy=outliers_policy,
                ),
                get_cardinality_block(
                    data, round_digits=round_digits, nan_policy=nan_policy
                ),
            ]
        )
    return blocks


def get_data_quality_report(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    round_digits: int = 4,
    nan_policy: str | Literal["drop", "raise", "include"] = "drop",
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
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling missing values in the input data.
        - 'drop' : rows containing missing values are removed before
          analysis.
        - 'raise' : an error is raised if missing values are present.
        - 'include' : missing values are preserved where supported.
        Note that not all child blocks support nan_policy='include'.
        In such cases, the policy is internally downgraded to 'drop' for
        those blocks.

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
    - This is a convenience wrapper around `get_data_overview_blocks`. Use
      `get_data_overview_blocks` instead if you need fine-grained control over
      block composition before report creation.
    - During the construction of EDA or interaction reports, many matplotlib figures
      may be opened (one per plot or table visualization). This is expected behavior
      when the dataset contains many features.
    - To prevent runtime warnings about too many open figures, these warnings are
      ignored internally.
    - To free memory after rendering, it is recommended to explicitly close figures:

      >>> report = get_eda_report(df)
      >>> report.render()
      >>> report.close_figures()

      or for individual blocks:

      >>> block = some_block
      >>> block.render()
      >>> block.close_figures()

    Examples
    --------
    >>> report = get_data_quality_report(df)
    >>> report.title
    'Data quality'
    >>> len(report.blocks)
    3
    """
    return Report(
        blocks=get_data_quality_blocks(data, round_digits, nan_policy=nan_policy),
        title="Data quality",
        description=(
            "A comprehensive summary of dataset quality, including missing values, "
            "outliers, cardinality, and distribution characteristics."
        ),
    )
