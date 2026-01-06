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

from typing import Sequence, Mapping, Any, Hashable
import warnings

import pandas as pd

from ..._utils import convert_dataframe, handle_nan
from ..core.block import Block
from ..core.report import Report
from ..utils import _split_features_by_assignment, normalize_assignment
from .blocks import get_linear_relations_block, get_nonlinear_relations_block


def get_interactions_blocks(
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    numerical_names: list[Hashable] = None,
    categorical_names: list[Hashable] = None,
    target_name: Hashable = None,
    **kwargs,
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
    numerical_names : list[Hashable], optional
        Names of numerical feature columns. If not provided, numerical features
        are inferred from column dtypes.
    categorical_names : list[Hashable], optional
        Names of categorical feature columns. If not provided, categorical
        features are inferred using cardinality-based heuristics.
    target_name : Hashable, optional
        Name of the target column in `data`. If provided and explicit target
        names are not specified, its type and cardinality are used to infer
        whether it should be treated as numerical, categorical, or both.

    Other Parameters
    ----------------
    target_numerical_name : Hashable, optional
        Explicit name of the numerical target column.
        Takes precedence over heuristic inference.
    target_categorical_name : Hashable, optional
        Explicit name of the categorical target column.
        Takes precedence over heuristic inference.
    categorical_threshold : int, default=30
        Maximum number of unique values for a column to be considered
        categorical during heuristic inference.
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
    - Explicitly provided feature and target names always take precedence over
      heuristic inference.
    - Features may appear in both numerical and categorical sets if applicable.
    - This function is intended for EDA and interaction analysis purposes.
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
    >>> from explorica.reports.presets.blocks.interactions import (
    ...     get_interactions_blocks)
    >>> blocks = get_interactions_blocks(df)
    >>> blocks = get_interactions_blocks(
    ...     df,
    ...     numerical_names=["x1", "x2"],
    ...     categorical_names=["c1"],
    ...     target_name="y"
    ... )
    """
    other_params = {
        "target_numerical_name": kwargs.get("target_numerical_name", None),
        "target_categorical_name": kwargs.get("target_categorical_name", None),
        "categorical_threshold": kwargs.get("categorical_threshold", 30),
        "round_digits": kwargs.get("round_digits", 4),
        "nan_policy": kwargs.get("nan_policy", "drop"),
    }

    df = handle_nan(convert_dataframe(data), other_params["nan_policy"])
    feature_assignment = normalize_assignment(
        df,
        numerical_names,
        categorical_names,
        numerical_target=other_params["target_numerical_name"],
        categorical_target=other_params["target_categorical_name"],
        target_name=target_name,
    )
    # Split df by assignments
    df_num, df_cat, target_num, target_cat = _split_features_by_assignment(
        df,
        feature_assignment,
        categorical_threshold=other_params["categorical_threshold"],
    )
    # We ignore mpl runtime warnings because EDA reports may open many figures.
    # It's assumed, that the user use ``Report.close_figures()``
    # and ``Block.close_figures`` after rendering
    blocks = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            module="explorica.visualizations",
            category=RuntimeWarning,
            message="More than 20 figures have been opened.",
        )

        if not df_num.empty:
            blocks.append(
                get_linear_relations_block(
                    df_num,
                    target_num,
                    round_digits=other_params["round_digits"],
                    nan_policy=other_params["nan_policy"],
                )
            )
        block_nonlinear_rels = get_nonlinear_relations_block(
            df_num,
            df_cat,
            numerical_target=target_num,
            categorical_target=target_cat,
            round_digits=other_params["round_digits"],
            nan_policy=other_params["nan_policy"],
        )
        # block_nonlinear_rels can be empty
        if not block_nonlinear_rels.empty:
            blocks.append(block_nonlinear_rels)
    return blocks


def get_interactions_report(
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    numerical_names: list[Hashable] = None,
    categorical_names: list[Hashable] = None,
    target_name: Hashable = None,
    **kwargs,
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
    numerical_names : list[Hashable], optional
        Names of numerical feature columns. If not provided, numerical features
        are inferred from column dtypes.
    categorical_names : list[Hashable], optional
        Names of categorical feature columns. If not provided, categorical
        features are inferred using cardinality-based heuristics.
    target_name : Hashable, optional
        Name of the target column in `data`. If provided and explicit target
        names are not specified, its type and cardinality are used to infer
        whether it should be treated as numerical, categorical, or both.

    Other Parameters
    ----------------
    target_numerical_name : Hashable, optional
        Explicit name of the numerical target column.
        Takes precedence over heuristic inference.
    target_categorical_name : Hashable, optional
        Explicit name of the categorical target column.
        Takes precedence over heuristic inference.
    categorical_threshold : int, default=30
        Maximum number of unique values for a column to be considered
        categorical during heuristic inference.
    round_digits : int, default=4
        Number of decimal places to round coefficients in all included blocks.
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy for handling missing values across all blocks:
        - 'drop' : remove rows containing NaNs.
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

    See Also
    --------
    get_interactions_blocks
        Constructs the individual interaction blocks used in the report.

    Notes
    -----
    - This function does not perform any analysis itself; it only orchestrates
      block construction and report assembly.
    - Explicitly provided feature and target names always take precedence over
      heuristic inference.
    - The presence and contents of each block depend on the availability of
      numerical and categorical features and on whether target variables are
      provided.
    - An empty report indicates insufficient information to compute interaction
      metrics, not an execution error.
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
    >>> from explorica.reports.presets import get_interactions_report

    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3, 4],
    ...     "x2": [10, 20, 30, 40],
    ...     "c1": ["a", "b", "a", "b"],
    ...     "y": [0, 1, 0, 1],
    ... })

    # Automatic feature and target inference
    >>> report = get_interactions_report(df, target_name="y")
    >>> len(report.blocks) > 0
    True

    # Explicit feature assignment
    >>> report = get_interactions_report(
    ...     df,
    ...     numerical_names=["x1", "x2"],
    ...     categorical_names=["c1"],
    ...     target_name="y",
    ... )
    >>> report.title
    'Interaction analysis'

    # Explicit target specification via kwargs
    >>> report = get_interactions_report(
    ...     df,
    ...     numerical_names=["x1", "x2"],
    ...     categorical_names=["c1"],
    ...     target_numerical_name="y",
    ... )
    >>> report.blocks
    [...]
    """
    blocks = get_interactions_blocks(
        data,
        numerical_names=numerical_names,
        categorical_names=categorical_names,
        target_name=target_name,
        **kwargs,
    )
    return Report(blocks, title="Interaction analysis")
