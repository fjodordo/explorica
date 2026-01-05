"""
Interactions presets for Explorica reports.

This module provides high-level orchestration utilities for building
interaction-focused Explorica reports. It does not implement statistical
methods itself; instead, it coordinates feature assignment, heuristic
inference, and composition of lower-level analytical blocks.

Functions
---------
get_interactions_blocks(
    data,
    feature_assignment = None,
    category_threshold = 30,
    round_digits = 4,
    nan_policy="drop"
)
    Build linear and non-linear interaction blocks for Explorica reports.
def get_interactions_report(
    data,
    feature_assignment = None,
    category_threshold = 30,
    round_digits = 4,
    nan_policy = "drop"
)
    Generate an interaction analysis report.

Notes
-----
- This module is pandas-based and expects tabular, column-addressable
  input data (``DataFrame`` or mapping of column names to sequences).
- User-provided ``FeatureAssignment`` objects always take precedence over
  heuristic feature and target inference.
- Only non-empty blocks are included in the final report.
- An empty result indicates insufficient information for interaction
  analysis rather than an execution error.

Examples
--------
>>> from explorica.reports.presets.interactions import get_interactions_report
>>> report = get_interactions_report(df)
>>> report.title
'Interaction analysis'
"""

from typing import Sequence, Mapping, Any, Literal
from copy import deepcopy

import pandas as pd

from ...types import FeatureAssignment
from ..._utils import convert_dataframe, handle_nan
from ...data_quality import get_categorical_features
from ..core.block import Block
from ..core.report import Report
from .blocks import get_linear_relations_block, get_nonlinear_relations_block


def get_interactions_blocks(
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    feature_assignment: FeatureAssignment = None,
    category_threshold: int = 30,
    round_digits: int = 4,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> list[Block]:
    """
    Generate linear and non-linear interaction blocks for Explorica reports.

    This function orchestrates the creation of two main blocks:
    1. Linear relations block,
       summarizing correlations and multicollinearity diagnostics.
    2. Non-linear relations block, summarizing eta² (numerical-categorical)
       and Cramer's V (categorical-categorical) dependencies.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[str, Sequence[Any]]
        Input dataset containing features and optionally target columns.
    feature_assignment : FeatureAssignment, optional
        Explicit assignment of features and target columns. If provided, it
        takes precedence over heuristic detection.
    category_threshold : int, default=30
        Maximum number of unique values for a numerical column to be
        considered categorical if no assignment is provided.
    round_digits : int, default=4
        Number of decimal places to round coefficients in tables.
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy for handling missing values:
        - 'drop' : remove rows containing NaNs.
        - 'raise': raise an error if missing values are present.

    Returns
    -------
    list[Block]
        List of generated Explorica `Block` instances:
        - The linear relations block is always included.
        - The non-linear relations block is included only if it contains
          metrics, visualizations, or tables (otherwise it is omitted).

    Notes
    -----
    - User-provided feature assignments have higher priority than heuristics.
    - Features may appear in both numerical and categorical sets if applicable.
    - This function is intended for EDA and interaction analysis purposes.

    Examples
    --------
    >>> from explorica.reports.presets.blocks.interactions import (
    ...     get_interactions_blocks)
    >>> from explorica.types import FeatureAssignment
    >>> blocks = get_interactions_blocks(df)
    >>> blocks = get_interactions_blocks(df, feature_assignment=FeatureAssignment(
    ...     numerical_features=['x1', 'x2'],
    ...     categorical_features=['c1'],
    ...     numerical_target='y'
    ... ))
    """
    if feature_assignment is None:
        feature_assignment = FeatureAssignment()
    df = handle_nan(convert_dataframe(data), nan_policy)

    # Split df by assignments
    df_num, df_cat, target_num, target_cat = _split_features_by_assignment(
        df, feature_assignment, category_threshold=category_threshold
    )

    blocks = []

    if not df_num.empty:
        blocks.append(
            get_linear_relations_block(df_num, target_num, round_digits=round_digits)
        )
    block_nonlinear_rels = get_nonlinear_relations_block(
        df_num,
        df_cat,
        numerical_target=target_num,
        categorical_target=target_cat,
        round_digits=round_digits,
    )
    # block_nonlinear_rels can be empty
    if not block_nonlinear_rels.empty:
        blocks.append(block_nonlinear_rels)
    return blocks


def _split_features_by_assignment(
    df: pd.DataFrame,
    feature_assignment: FeatureAssignment,
    category_threshold: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    """
    Split a DataFrame into numerical and categorical features and extract targets.

    This function separates the features of a dataset into numerical and
    categorical subsets based on a provided `FeatureAssignment` or, if some
    assignments are missing, inferred heuristically. It also extracts numerical
    and categorical target variables if assigned, and ensures that targets are
    removed from the feature DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing all features and targets.
    feature_assignment : FeatureAssignment
        Object specifying which columns are numerical features, categorical
        features, and target variables. User-specified assignments have the
        highest priority; heuristics are applied only if some feature lists
        are empty.
    category_threshold : int, default=30
        Maximum number of unique values for a column to be considered categorical
        when features are inferred automatically.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]
        A 4-tuple containing:
        - df_num : DataFrame of numerical features (targets removed)
        - df_cat : DataFrame of categorical features (targets removed)
        - target_num : Numerical target Series or None if not assigned
        - target_cat : Categorical target Series or None if not assigned

    Notes
    -----
    - Columns may appear in both `df_num` and `df_cat` if they are both numeric
      and categorical according to the heuristics or assignments.
    - Targets are always removed from feature DataFrames to avoid leakage in
      downstream analysis.
    - Empty DataFrames are returned if no features are found for a given type.
    - User-provided `FeatureAssignment` has priority over heuristic inference.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.types import FeatureAssignment
    >>> df = pd.DataFrame({
    ...     'x1': [1, 2, 3],
    ...     'x2': [4, 5, 6],
    ...     'c1': ['a', 'b', 'a'],
    ...     'target': [0, 1, 0]
    ... })
    >>> assignment = FeatureAssignment(
    ...     numerical_features=['x1', 'x2'],
    ...     categorical_features=['c1'],
    ...     numerical_target='target'
    ... )
    >>> df_num, df_cat, target_num, target_cat = (
    _split_features_by_assignment(df, assignment))
    >>> df_num.columns
    Index(['x1', 'x2'], dtype='object')
    >>> df_cat.columns
    Index(['c1'], dtype='object')
    >>> target_num.name
    'target'
    >>> target_cat is None
    True
    """
    assignment_copy = deepcopy(feature_assignment)
    # Define categorical features if it's not assigned
    if len(assignment_copy.categorical_features) == 0:
        mask = get_categorical_features(
            df, include_all=True, threshold=category_threshold
        )["is_category"].astype("bool")
        assignment_copy.categorical_features = list(df.columns[mask])
    # Define numerical features if it's not assigned
    if len(assignment_copy.numerical_features) == 0:
        assignment_copy.numerical_features = list(df.select_dtypes("number").columns)

    df_num = (
        df[assignment_copy.numerical_features]
        if len(assignment_copy.numerical_features) != 0
        else pd.DataFrame(columns=[])
    )
    df_cat = (
        df[assignment_copy.categorical_features]
        if len(assignment_copy.categorical_features) != 0
        else pd.DataFrame(columns=[])
    )

    # Define targets if it's not provided
    if assignment_copy.numerical_target is not None:
        target_num = df[assignment_copy.numerical_target]
    else:
        target_num = None
    if assignment_copy.categorical_target is not None:
        target_cat = df[assignment_copy.categorical_target]
    else:
        target_cat = None

    # Remove targets from feature dataframes
    for target in [target_num, target_cat]:
        if target is None:
            continue
        if target.name in df_num.columns:
            df_num = df_num.drop(target.name, axis=1)
        if target.name in df_cat.columns:
            df_cat = df_cat.drop(target.name, axis=1)

    return df_num, df_cat, target_num, target_cat


def get_interactions_report(
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    feature_assignment: FeatureAssignment = None,
    category_threshold: int = 30,
    round_digits: int = 4,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> Report:
    """
    Generate an interaction analysis report.

    This function is a high-level orchestrator that constructs an Explorica
    `Report` focused on feature interactions. It delegates feature selection,
    target assignment, and block composition to `get_interactions_blocks`,
    and wraps the resulting blocks into a single report.

    Parameters
    ----------
    data : pd.DataFrame or Mapping[str, Sequence[Any]]
        Input dataset containing features and optionally target columns.
    feature_assignment : FeatureAssignment, optional
        Explicit feature and target assignment specification.
        User-defined assignments have higher priority than heuristic-based
        inference. If a particular group of features or targets is not
        specified, it may be inferred automatically from the data.
    category_threshold : int, default=30
        Threshold on the number of unique values used to infer categorical
        features when they are not explicitly assigned.
    round_digits : int, default=4
        Number of decimal places for rounding statistical coefficients
        in all included blocks.
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy for handling missing values across all blocks:
        - 'drop' : remove rows with missing values.
        - 'raise': raise an error if missing values are present.

    Returns
    -------
    Report
        An Explorica `Report` titled ``"Interaction analysis"`` containing
        zero or more blocks describing linear and non-linear feature
        interactions.

        The report may include:
        - A linear relations block (correlations, multicollinearity diagnostics,
          and feature–target visualizations).
        - A non-linear relations block (η² and Cramer's V dependency analysis).

        Only non-empty blocks are included in the report. If no interaction
        blocks can be constructed from the provided data and assignments,
        the report may be empty.

    Notes
    -----
    - This function does not perform any analysis itself; it only orchestrates
      block construction and report assembly.
    - Feature and target assignments provided by the user always take precedence
      over automatically inferred heuristics.
    - The presence and contents of each block depend on the availability of
      numerical and categorical features and on whether target variables are
      provided.
    - An empty report indicates insufficient information to compute interaction
      metrics, not an execution error.

    See Also
    --------
    get_interactions_blocks
        Constructs the individual interaction blocks used in the report.
    FeatureAssignment
        Defines explicit feature and target assignments.
    """
    blocks = get_interactions_blocks(
        data,
        feature_assignment=feature_assignment,
        category_threshold=category_threshold,
        round_digits=round_digits,
        nan_policy=nan_policy,
    )
    return Report(blocks, title="Interaction analysis")
