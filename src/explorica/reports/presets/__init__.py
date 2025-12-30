"""
Explorica report presets.

This module aggregates preset functions for quickly building Explorica reports.
It exposes both atomic blocks and higher-level report builders, enabling
flexible creation of data overview reports or standalone blocks.

Functions
---------
get_data_overview_blocks(data, round_digits=4)
    Build blocks for a short data overview.
get_data_overview_report(data, round_digits=4)
    Generate a short data overview report.
get_data_overview_blocks(data, round_digits=4)
    Build blocks for a detailed data quality analysis.
get_data_overview_report(data, round_digits=4)
    Generate a detailed data quality analysis report.
get_ctm_block(data, nan_policy='drop', round_digits=4)
    Build a `Block` instance containing tables of basic statistics for the dataset.
get_data_shape_block(data, nan_policy='drop')
    Build a Block instance summarizing the dataset shape.
get_data_quality_overview_block(data, round_digits=4)
    Build a `Block` instance containing metrics and a table summarizing
    duplicated rows and NaN counts/ratios.
get_outliers_block(data, iqr_factor=1.5, zscore_factor=3.0, nan_policy="drop")
    Build a `Block` instance containing a table summarizing
    outliers detected by different methods.
get_distributions_block(data, threshold_skewness=0.25, threshold_kurtosis=0.25,
    round_digits=4, nan_policy="drop"
)
    Build a Block instance summarizing feature distributions in a dataset.
get_cardinality_block(data, round_digits=4, nan_policy="drop")
    Build a Block instance summarizing feature cardinality and constancy metrics.

Notes
-----
- Presets are designed to be building blocks for Explorica reports.
- Both atomic blocks and higher-level reports can be used independently
  or combined to create customized EDA reports.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets import (
...     get_data_overview_blocks,
...     get_data_overview_report,
...     get_ctm_block,
...     get_data_shape_block,
...     get_data_quality_overview_block
... )
>>> df = pd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']})
>>> blocks = get_data_overview_blocks(df)
>>> report = get_data_overview_report(df)
>>> [b.block_config.title for b in blocks]
['Basic statistics for the dataset.', 'Dataset shape', 'Data quality quick summary']
>>> report.title
"""

from .blocks import (
    get_ctm_block,
    get_data_quality_overview_block,
    get_data_shape_block,
    get_outliers_block,
    get_distributions_block,
    get_cardinality_block,
)

from .data_overview import get_data_overview_blocks, get_data_overview_report
from .data_quality import get_data_quality_blocks, get_data_quality_report

__all__ = [
    "get_data_overview_blocks",
    "get_data_overview_report",
    "get_data_quality_blocks",
    "get_data_quality_report",
    "get_ctm_block",
    "get_data_quality_overview_block",
    "get_data_shape_block",
    "get_outliers_block",
    "get_distributions_block",
    "get_cardinality_block",
]
