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

__all__ = [
    "get_ctm_block",
    "get_data_shape_block",
    "get_data_quality_overview_block",
]
