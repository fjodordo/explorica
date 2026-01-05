"""
Atomic block presets for Explorica reports.

This module exposes individual block functions that generate Explorica `Block`
instances for various aspects of dataset exploration, including central
tendency, dataset shape, and a quick data quality overview. Each function is
designed to be used standalone or as part of a larger report.

Functions
---------
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
get_linear_relations_block(data, target, round_digits=4, nan_policy="drop")
    Build a Block instance summarizing linear relationships in a dataset.
get_nonlinear_relations_block(
    numerical_data, categorical_data, numerical_target=None,
    categorical_target=None, **kwargs
)
    Build a Block instance summarizing non-linear dependencies between features.

Notes
-----
- These blocks are intended as building blocks for reports and can be
  combined in any order.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks import (
...     get_ctm_block,
...     get_data_shape_block,
...     get_data_quality_overview_block
... )
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
>>> block1 = get_ctm_block(df)
>>> block2 = get_data_shape_block(df)
>>> block3 = get_data_quality_overview_block(df)
>>> [b.block_config.title for b in [block1, block2, block3]]
['Basic statistics for the dataset.', 'Dataset shape', 'Data quality quick summary']
"""

from .ctm import get_ctm_block
from .data_shape import get_data_shape_block
from .data_quality_overview import get_data_quality_overview_block
from .outliers import get_outliers_block
from .distributions import get_distributions_block
from .cardinality import get_cardinality_block
from .relations_linear import get_linear_relations_block
from .relations_nonlinear import get_nonlinear_relations_block

__all__ = [
    "get_ctm_block",
    "get_data_shape_block",
    "get_data_quality_overview_block",
    "get_outliers_block",
    "get_distributions_block",
    "get_cardinality_block",
    "get_nonlinear_relations_block",
    "get_linear_relations_block",
]
