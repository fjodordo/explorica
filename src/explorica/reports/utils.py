"""
Low-level utilities for Explorica's reports module

Low-level utility functions for standardizing visualization objects
into the `VisualizationResult` format. This module provides a
single public function that can be used in user code to normalize
Matplotlib and Plotly figures for downstream report generation
or further processing.

Functions
---------
normalize_visualization(figure)
    Convert a Matplotlib or Plotly figure into a standardized
    `VisualizationResult` dataclass with extracted metadata.
normalize_table(data)
    Normalize tabular data into a standardized TableResult object.
normalize_assignment(
    data, numerical_names = None, categorical_names = None,
    target_name = None, **kwargs
)
    Normalize feature and target assignment into a `FeatureAssignment` object.

Examples
--------
>>> from explorica.reports.utils import normalize_visualization

# Usage with matplotlib figure
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>> result = normalize_visualization(fig)
>>> result.engine
'matplotlib'
>>> result.width, result.height
(6.0, 4.0)

# Usage with plotly figure
>>> import plotly.graph_objects as go
>>> fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
>>> result = normalize_visualization(fig)
>>> result.engine
'plotly'
>>> result.width, result.height
(800, 600)  # default if not specified in layout
"""

from typing import Union, Sequence, Any, Mapping, Hashable
from copy import deepcopy

import pandas as pd
import matplotlib.figure
import plotly.graph_objects

from ..data_quality import get_categorical_features
from .._utils import convert_dataframe
from ..types import VisualizationResult, TableResult, FeatureAssignment


def normalize_visualization(
    figure: Union[
        matplotlib.figure.Figure, plotly.graph_objects.Figure, VisualizationResult
    ],
) -> VisualizationResult:
    """
    Normalize a visualization object into a standardized `VisualizationResult`.

    This function converts a Matplotlib or Plotly figure into a
    `VisualizationResult` dataclass, extracting common metadata
    such as engine, axes, width, height, and title. This allows
    downstream rendering or report composition functions to work
    with a uniform interface.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The input figure to normalize. Can be:
        - A Matplotlib figure
        - A Plotly figure
        - A pre-normalized `VisualizationResult` (in which
          case it is returned as-is)

    Returns
    -------
    VisualizationResult
        A dataclass containing:
        - figure : The original figure object.
        - engine : str, either "matplotlib" or "plotly".
        - axes : List of Matplotlib axes (for Matplotlib) or None (for Plotly).
        - width : Figure width in inches (Matplotlib) or pixels (Plotly).
        - height : Figure height in inches (Matplotlib) or pixels (Plotly).
        - title : Optional figure title.

    Raises
    ------
    TypeError
        If `figure` is not an instance of Matplotlib,
        Plotly Figure, or `VisualizationResult`.

    Notes
    -----
    - For Matplotlib figures, `width` and `height` are measured in inches.
    - For Plotly figures, `width` and `height` are measured in pixels.
    - The original figure is preserved in the `figure` attribute of the
      returned `VisualizationResult`.

    Examples
    --------
    >>> from explorica.reports.utils import normalize_visualization

    # Usage with matplotlib figure
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> result = normalize_visualization(fig)
    >>> result.engine
    'matplotlib'
    >>> result.width, result.height
    (6.0, 4.0)

    # Usage with plotly figure
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    >>> result = normalize_visualization(fig)
    >>> result.engine
    'plotly'
    >>> result.width, result.height
    (800, 600)  # default if not specified in layout
    """
    if isinstance(figure, matplotlib.figure.Figure):
        vis_result = VisualizationResult(figure=figure)
        vis_result.engine = "matplotlib"
        vis_result.axes = vis_result.figure.get_axes()
        w, h = figure.get_size_inches()
        vis_result.width = w
        vis_result.height = h
    elif isinstance(figure, plotly.graph_objects.Figure):
        vis_result = VisualizationResult(figure=figure)
        vis_result.engine = "plotly"
        vis_result.axes = None  # should be None
        vis_result.width = figure.layout.width
        vis_result.height = figure.layout.height
    elif isinstance(figure, VisualizationResult):
        vis_result = figure
    else:
        raise TypeError(
            f"Unsupported figure type: {type(figure)}."
            f"Please provide 'matplotlib.figure.Figure',"
            f"'plotly.graph_objects.Figure', or 'VisualizationResult'."
        )
    return vis_result


def normalize_table(
    data: (
        Sequence[float]
        | Sequence[Sequence[float]]
        | Mapping[str, Sequence[Any]]
        | TableResult
    ),
) -> TableResult:
    """
    Normalize tabular data into a standardized TableResult object.

    This function converts input data of various formats (1D/2D sequences, mappings
    or `TableResult`) into a `TableResult` instance containing a Pandas DataFrame.
    This ensures consistent handling of tabular results across Explorica reports.

    Parameters
    ----------
    data : Sequence, Mapping or TableResult
        Tabular data to normalize. Supported types include:
        - 1D or 2D sequences (e.g., list, tuple of lists)
        - Mapping[str, Sequence] (e.g., dict of column_name -> values)
        MultiIndex rows or columns are not supported.

    Returns
    -------
    TableResult
        A standardized container wrapping a Pandas DataFrame.

    Raises
    ------
    ValueError
        If the input DataFrame has a MultiIndex in rows or columns.

    Examples
    --------
    >>> data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    >>> table_result = normalize_table(data)
    >>> isinstance(table_result, TableResult)
    True
    >>> table_result.table.shape
    (3, 2)
    """
    if isinstance(data, TableResult):
        return data
    df = convert_dataframe(data)
    if isinstance(df.index, pd.MultiIndex):
        raise ValueError("MultiIndex in rows is not supported in normalize_table.")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("MultiIndex in columns is not supported in normalize_table.")

    return TableResult(table=df)


def normalize_assignment(
    data: pd.DataFrame,
    numerical_names: list[Hashable] = None,
    categorical_names: list[Hashable] = None,
    target_name: str = None,
    **kwargs,
) -> FeatureAssignment:
    """
    Normalize feature and target assignment into a `FeatureAssignment` object.

    This function converts casual, user-friendly input (feature name lists and
    an optional target column name) into a fully-populated `FeatureAssignment`
    instance suitable for Explorica report and block presets.

    If feature names are not explicitly provided, they are inferred from the
    input DataFrame using heuristic rules.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing feature and optional target columns.
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
        Takes priority over heuristic inference.
    target_categorical_name : Hashable, optional
        Explicit name of the categorical target column.
        Takes priority over heuristic inference.
    categorical_threshold : int, default=30
        Maximum number of unique values for a column to be considered
        categorical during heuristic inference.

    Returns
    -------
    FeatureAssignment
        A populated `FeatureAssignment` instance containing:
        - numerical_features
        - categorical_features
        - optional numerical_target
        - optional categorical_target

    Notes
    -----
    - Explicitly provided feature and target names always take precedence
      over heuristic inference.
    - If feature names are not provided, numerical features are inferred from
      numeric dtypes, and categorical features are inferred using the same
      categorical detection logic.
    - A single target column may be assigned as both numerical and categorical
      if it satisfies the criteria for both types.
    - Absence of a target assignment indicates insufficient information, not
      an error.


    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.utils import normalize_assignment
    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3, 4],
    ...     "x2": [10, 20, 30, 40],
    ...     "c1": ["a", "b", "a", "b"],
    ...     "y": [0, 1, 0, 1]
    ... })

    # Feature-only assignment
    >>> fa = normalize_assignment(
    ...     data=df,
    ...     numerical_names=["x1", "x2"],
    ...     categorical_names=["c1"]
    ... )
    >>> fa.numerical_features
    ['x1', 'x2']
    >>> fa.categorical_features
    ['c1']
    >>> fa.numerical_target is None
    True
    >>> fa.categorical_target is None
    True

    # Target inference using heuristics
    >>> fa = normalize_assignment(
    ...     data=df,
    ...     numerical_names=["x1", "x2"],
    ...     categorical_names=["c1"],
    ...     target_name="y",
    ...     categorical_threshold=3
    ... )
    >>> fa.numerical_target
    'y'
    >>> fa.categorical_target
    'y'
    """
    other_params = {
        "target_numerical_name": kwargs.get("target_numerical_name", None),
        "target_categorical_name": kwargs.get("target_categorical_name", None),
        "categorical_threshold": kwargs.get("categorical_threshold", 30),
    }
    feature_assignment = FeatureAssignment()
    categories = get_categorical_features(
        data, include_all=True, threshold=other_params["categorical_threshold"]
    )["is_category"].astype("bool")
    # Provide numerical_features by user input or heuristics
    if numerical_names is not None:
        feature_assignment.numerical_features = numerical_names
    else:
        feature_assignment.numerical_features = list(
            data.select_dtypes("number").columns
        )
    # Provide categorical_features by user input or heuristics
    if categorical_names is not None:
        feature_assignment.categorical_features = categorical_names
    else:
        feature_assignment.categorical_features = list(data.columns[categories])

    # Provide by `target_numerical_name` or heuristics
    if other_params["target_numerical_name"] is not None:
        feature_assignment.numerical_target = other_params["target_numerical_name"]
    elif target_name is not None and pd.api.types.is_numeric_dtype(data[target_name]):
        feature_assignment.numerical_target = target_name

    # Provide by `target_categorical_name` or heuristics
    if other_params["target_categorical_name"] is not None:
        feature_assignment.categorical_target = other_params["target_categorical_name"]
    elif target_name in categories.index and categories.loc[target_name]:
        feature_assignment.categorical_target = target_name

    return feature_assignment


def _split_features_by_assignment(
    df: pd.DataFrame,
    feature_assignment: FeatureAssignment,
    categorical_threshold: int = 30,
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
    categorical_threshold : int, default=30
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
            df, include_all=True, threshold=categorical_threshold
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
