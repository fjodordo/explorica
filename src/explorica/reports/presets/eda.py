"""
Exploratory Data Analysis (EDA) presets.

This module provides high-level orchestration functions for building
Explorica exploratory data analysis (EDA) reports. It defines convenient
entry points that assemble multiple lower-level analysis blocks into
coherent EDA workflows.

The public API is designed for *casual usage*: users may specify feature
groups and targets explicitly, or rely on heuristic inference based on
data types and cardinality.

Functions
---------
get_eda_blocks(
    data,
    numerical_names=None,
    categorical_names=None,
    target_name=None,
    **kwargs
)
    Build a full exploratory data analysis (EDA) report as a list of blocks.
get_eda_report(
    data,
    numerical_names=None,
    categorical_names=None,
    target_name=None,
    **kwargs
)
    Build a full exploratory data analysis (EDA) report.

Notes
-----
- Feature and target assignments can be provided explicitly or inferred
  automatically using heuristics.
- Missing value handling is controlled via `nan_policy` and may be
  adapted internally for blocks that do not support all policies.
- These functions do not perform statistical analysis directly; they
  orchestrate and compose lower-level preset blocks.
- The API is intended as a stable, high-level entry point for EDA in
  Explorica.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.eda import get_eda_report
>>> df = pd.DataFrame({
...     "x1": [1, 2, 3, 4],
...     "x2": [10, 20, 30, 40],
...     "c1": ["a", "b", "a", "b"],
...     "y": [0, 1, 0, 1]
... })
>>> report = get_eda_report(df)
>>> report.title
'Exploratory Data Analysis Report'
"""

from typing import Sequence, Mapping, Hashable
import warnings

import pandas as pd

from ..._utils import handle_nan
from ..core.block import Block
from ..core.report import Report
from ..utils import normalize_assignment, _split_features_by_assignment
from .interactions import get_interactions_blocks
from .data_overview import get_data_overview_blocks
from .data_quality import get_data_quality_blocks


def get_eda_blocks(
    data: pd.DataFrame | Mapping[Hashable, Sequence],
    numerical_names: list[Hashable] = None,
    categorical_names: list[Hashable] = None,
    target_name: Hashable = None,
    **kwargs,
) -> list[Block]:
    """
    Build a full exploratory data analysis (EDA) report as a list of blocks.

    This function orchestrates the creation of multiple EDA-related blocks,
    including data overview, data quality, and feature interactions. It
    handles feature assignment, target detection, missing value policy, and
    automatic block composition.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[Hashable, Sequence]
        Input dataset. Can be a DataFrame or a mapping (e.g., dict of lists)
        convertible to a DataFrame.
    numerical_names : list[Hashable], optional
        Explicit names of numerical feature columns. If not provided, numerical
        features are inferred heuristically.
    categorical_names : list[Hashable], optional
        Explicit names of categorical feature columns. If not provided, categorical
        features are inferred heuristically using `categorical_threshold`.
    target_name : Hashable, optional
        Name of the target column. If provided, heuristics determine whether it
        is a numerical or categorical target based on its type and number of unique
        values.

    Other Parameters
    ----------------
    target_numerical_name : Hashable, optional
        Explicit numerical target name. Has priority over `target_name`.
    target_categorical_name : Hashable, optional
        Explicit categorical target name. Has priority over `target_name`.
    round_digits : int, default=4
        Number of decimal places for rounding statistics in tables and plots.
    categorical_threshold : int, default=30
        Maximum number of unique values for a column to be considered categorical
        when inferred automatically.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling missing values:
        - 'drop' : remove rows containing NaNs.
        - 'raise': raise an error if missing values are present.
        - 'include': keep NaNs where supported; for blocks that do not support
          'include', this defaults to 'drop'.

    Returns
    -------
    list[Block]
        A list of `Block` objects representing the EDA report. Blocks included:
        - Data overview blocks: cardinality, basic statistics, distributions.
        - Data quality blocks: missing values, outliers, data types.
        - Feature interactions blocks: linear and non-linear interactions
          (correlations, η², Cramer's V). Only non-empty interaction blocks
          are included.

    Notes
    -----
    - User-specified feature and target assignments take precedence over
      heuristic inference.
    - If `numerical_names` or `categorical_names` are not provided, they will
      be inferred automatically from the data.
    - If `target_name` is provided, the function may assign it as both
      `numerical_target` and `categorical_target` based on type and cardinality.
    - `nan_policy='include'` is only supported in blocks that allow missing values;
      for other blocks, it is automatically converted to `'drop'`.
    - This function is intended as a high-level entry point for casual API users.
      It does not perform analysis itself, but assembles lower-level blocks.
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
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_eda_blocks
    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3, 4],
    ...     "x2": [10, 20, 30, 40],
    ...     "c1": ["a", "b", "a", "b"],
    ...     "y": [0, 1, 0, 1]
    ... })
    >>> blocks = get_eda_blocks(df)
    >>> len(blocks)
    9
    >>> blocks[0].title
    'Exploratory Data Analysis Overview'
    >>> blocks[5].title
    'Feature Interactions'
    >>> blocks[-1].title  # last block may vary depending on data
    'Relations Non-linear'
    """
    other_params = {
        "target_numerical_name": kwargs.get("target_numerical_name", None),
        "target_categorical_name": kwargs.get("target_categorical_name", None),
        "round_digits": kwargs.get("round_digits", 4),
        "categorical_threshold": kwargs.get("categorical_threshold", 30),
        "nan_policy": kwargs.get("nan_policy", "drop"),
    }
    df = handle_nan(
        data,
        nan_policy=other_params["nan_policy"],
        supported_policy=("drop", "raise", "include"),
        is_dataframe=False,
    )
    feature_assignment = normalize_assignment(
        df,
        numerical_names,
        categorical_names,
        target_name=target_name,
        numerical_target=other_params["target_numerical_name"],
        categorical_target=other_params["target_categorical_name"],
        nan_policy=other_params["nan_policy"],
        categorical_threshold=other_params["categorical_threshold"],
    )
    num_features, cat_features, _, _ = _split_features_by_assignment(
        df, feature_assignment
    )
    # We ignore mpl runtime warnings because EDA reports may open many figures.
    # It's assumed, that the user use ``Report.close_figures()``
    # and ``Block.close_figures`` after rendering
    blocks = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="explorica.visualizations",
            message="More than 20 figures have been opened.",
        )
        # Data Overview Block
        blocks.append(Block({"title": "Data Overview"}))
        blocks.extend(
            get_data_overview_blocks(df, round_digits=other_params["round_digits"])
        )
        # Data Quality Block
        blocks.append(Block({"title": "Data Quality"}))
        blocks.extend(
            get_data_quality_blocks(df, round_digits=other_params["round_digits"])
        )
        # Interactions Block
        interaction_blocks = get_interactions_blocks(
            df,
            numerical_names=num_features.columns,
            categorical_names=cat_features.columns,
            target_numerical_name=other_params["target_numerical_name"],
            target_categorical_name=other_params["target_categorical_name"],
            round_digits=other_params["round_digits"],
            nan_policy=(
                other_params["nan_policy"]
                if other_params["nan_policy"] != "include"
                else "drop"
            ),
        )
        # Interaction may contain no blocks
        if len(interaction_blocks) > 0:
            blocks.append(Block({"title": "Feature Interactions"}))
            blocks.extend(interaction_blocks)
    return blocks


def get_eda_report(
    data: pd.DataFrame | Mapping[Hashable, Sequence],
    numerical_names: list[Hashable] = None,
    categorical_names: list[Hashable] = None,
    target_name: Hashable = None,
    **kwargs,
) -> Report:
    """
    Build a full exploratory data analysis (EDA) report.

    This function is a thin wrapper around :func:`get_eda_blocks`. It constructs
    a high-level Explorica :class:`Report` by assembling EDA-related blocks
    (data overview, data quality, and feature interactions) and assigning a
    predefined report title.

    All feature selection, target inference, missing value handling, and block
    composition logic is delegated to :func:`get_eda_blocks`.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[Hashable, Sequence]
        Input dataset. Can be a DataFrame or any mapping convertible to a
        DataFrame (e.g., a dictionary of columns).
    numerical_names : list[Hashable], optional
        Explicit names of numerical feature columns. If not provided, numerical
        features are inferred heuristically.
    categorical_names : list[Hashable], optional
        Explicit names of categorical feature columns. If not provided,
        categorical features are inferred heuristically using
        `categorical_threshold`.
    target_name : Hashable, optional
        Name of the target column. If provided, heuristics are used to determine
        whether it should be treated as a numerical or categorical target.

    Other Parameters
    ----------------
    target_numerical_name : Hashable, optional
        Explicit numerical target name. Has priority over `target_name`.
    target_categorical_name : Hashable, optional
        Explicit categorical target name. Has priority over `target_name`.
    round_digits : int, default=4
        Number of decimal places for rounding numerical statistics in all blocks.
    categorical_threshold : int, default=30
        Maximum number of unique values for a column to be considered categorical
        when inferred automatically.
    nan_policy : {'drop', 'raise', 'include'}, default='drop'
        Policy for handling missing values:
        - 'drop' : remove rows containing NaNs.
        - 'raise': raise an error if missing values are present.
        - 'include': keep NaNs where supported; for blocks that do not support
          'include', this policy is internally converted to 'drop'.

    Returns
    -------
    Report
        An Explorica :class:`Report` titled ``"Exploratory Data Analysis Report"``,
        containing zero or more EDA blocks.

        Only non-empty blocks are included. If no blocks can be constructed from
        the provided data and assignments, the report may be empty.

    See Also
    --------
    get_eda_blocks
        Constructs the individual EDA blocks used in the report.
    Report
        Container object used to assemble and render blocks.

    Notes
    -----
    - This function does not perform analysis itself; it only wraps
      :func:`get_eda_blocks` into a report object.
    - User-specified feature and target assignments always take precedence over
      heuristic inference.
    - The behavior and contents of the report are entirely determined by
      :func:`get_eda_blocks`.
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
    """
    return Report(
        get_eda_blocks(data, numerical_names, categorical_names, target_name, **kwargs),
        title="Exploratory Data Analysis Report",
    )
