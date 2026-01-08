"""
Non-linear relations block preset.

Provides a Block summarizing non-linear dependencies between numerical and
categorical features using eta-squared (η²) and Cramer's V metrics. The block
includes heatmaps for both metrics and a table of top dependency pairs.

Functions
---------
get_nonlinear_relations_block(
    numerical_data, categorical_data, numerical_target=None,
    categorical_target=None, **kwargs
)
    Build a Block instance summarizing non-linear dependencies between features.

Notes
-----
- Only one target type (numerical or categorical) can be provided at a time.
- Non-linear dependencies are computed using η² (numerical-categorical) and
  Cramer's V (categorical-categorical) only.
- This function is intended for internal use in Explorica reports, but is
  exposed as a preset for user convenience.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks.relations_nonlinear import (
...     get_nonlinear_relations_block)
>>> df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
>>> df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
>>> block = get_nonlinear_relations_block(df_num, df_cat)
>>> block.block_config.title
'Non-linear relations'
"""

from typing import Sequence, Mapping, Any, Literal

import numpy as np
import pandas as pd

from ...._utils import handle_nan, convert_series, convert_dataframe
from ....types import TableResult, VisualizationResult
from ...core.block import Block, BlockConfig
from ....interactions import corr_matrix_eta, corr_matrix_cramer_v, high_corr_pairs
from ....visualizations import heatmap


def get_nonlinear_relations_block(
    numerical_data: Sequence[Any] | Mapping[str, Sequence[Any]] = None,
    categorical_data: Sequence[Any] | Mapping[str, Sequence[Any]] = None,
    numerical_target: Sequence[Any] | Mapping[str, Sequence[Any]] = None,
    categorical_target: Sequence[Any] | Mapping[str, Sequence[Any]] = None,
    **kwargs,
) -> Block:
    """
    Generate a `Block` summarizing non-linear dependencies between features.

    Computes non-linear dependency metrics between numerical and categorical
    features, including η² (eta squared) for numerical-categorical pairs and
    Cramer's V for categorical-categorical pairs. Renders corresponding heatmaps
    and a table of top dependency pairs. If the dataset or target is insufficient,
    the block may be empty.

    Parameters
    ----------
    numerical_data : Sequence or Mapping, optional
        Numerical features for dependency analysis;
        must be convertible to a pandas DataFrame.
    categorical_data : Sequence or Mapping, optional
        Categorical features for dependency analysis;
        must be convertible to a pandas DataFrame.
    numerical_target : Sequence or Mapping, optional
        Numerical target variable to include in the analysis.
    categorical_target : Sequence or Mapping, optional
        Categorical target variable to include in the analysis.

    Other Parameters
    ----------------
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy for handling missing values:
        - 'drop' : remove rows with missing values.
        - 'raise': raise an error if missing values are present.
    round_digits : int, default=4
        Number of decimal places to round dependency coefficients in the table.

    Returns
    -------
    Block
        An Explorica `Block` containing a subset of the following components,
        depending on the provided data and targets:

        Visualizations:
        - η² (eta squared) dependency heatmap
            Added if both `numerical_data` and `categorical_data` are provided
            and contain at least one column each. Numerical and categorical
            targets, if provided, are included in the computation.

        - Cramer's V dependency heatmap
            Added if `categorical_data` is provided and contains at least one
            column. A categorical target, if provided, is included in the
            computation.

        Tables:
        - Table of highest non-linear dependency pairs
            Added only if `categorical_target` is provided. The table summarizes
            the strongest non-linear dependencies between features and the
            categorical target using η² and Cramer's V where applicable.

        If none of the above conditions are satisfied, the returned block will
        be empty (`block.empty == True`).

    Notes
    -----
    - Only one target variable type (numerical or categorical) can be provided.
    - If no categorical target is provided, the table of top dependency pairs
      will be omitted.
    - Each component of the block (η² heatmap, Cramer's V heatmap, top dependency table)
      is added only if the corresponding data and/or target are available.
    - If none of the conditions are satisfied, the returned block will be empty.
    - This block is designed for inclusion in non-linear, interaction-focused
      Explorica reports.
    - This function is designed to be tolerant to missing inputs and may return
      an empty block when insufficient data is provided.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks.relations_nonlinear import (
    ...     get_nonlinear_relations_block)
    >>> df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    >>> df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    >>> block = get_nonlinear_relations_block(df_num, df_cat)
    >>> block.block_config.title
    'Non-linear relations'
    """
    other_params = {
        "nan_policy": kwargs.get("nan_policy", "drop"),
        "round_digits": kwargs.get("round_digits", 4),
    }
    df_numerical = convert_dataframe(numerical_data)
    df_categorical = convert_dataframe(categorical_data)

    if numerical_target is not None and categorical_target is not None:
        raise ValueError(
            "Target is ambiguous. Please, provide only `numerical_target` or only"
            "`categorical_target`, not both."
        )
    block = Block(BlockConfig(title="Non-linear relations"))
    series_numerical_target = convert_series(numerical_target)
    series_categorical_target = convert_series(categorical_target)

    # Add heatmap of eta squared dependency matrix
    # Doesn't work unless both types of feature are provided
    if not df_numerical.empty and not df_categorical.empty:
        heatmap_eta: VisualizationResult = _get_eta_squared_heatmap(
            df_numerical,
            df_categorical,
            target_numerical=series_numerical_target,
            target_categorical=series_categorical_target,
            nan_policy=other_params["nan_policy"],
        )
        block.add_visualization(heatmap_eta)

    # Add heatmap of Cramer's V dependency matrix
    # Doesn't work without categorical data
    if not df_categorical.empty:
        heatmap_cramer: VisualizationResult = _get_cramer_v_heatmap(
            df_categorical,
            series_categorical_target,
            nan_policy=other_params["nan_policy"],
        )
        block.add_visualization(heatmap_cramer)

    # Add table of highest dependency pairs
    # Doesn't work if target is not categorical
    if not series_categorical_target.empty:
        pairs: TableResult = _get_highest_dependency_pairs_table(
            df_numerical,
            df_categorical,
            target_categorical=series_categorical_target,
            round_digits=other_params["round_digits"],
            nan_policy=other_params["nan_policy"],
        )
        block.add_table(pairs)

    return block


def _get_eta_squared_heatmap(
    df_numerical: pd.DataFrame,
    df_categorical: pd.DataFrame,
    target_numerical: pd.Series = None,
    target_categorical: pd.Series = None,
    **kwargs,
) -> VisualizationResult:
    """
    Build an eta-squared dependency heatmap between numerical and categorical features.

    Computes the eta-squared (η²) dependency matrix for numerical–categorical
    feature pairs and renders it as a heatmap. Optionally includes a numerical
    or categorical target variable in the analysis.

    Parameters
    ----------
    df_numerical : pandas.DataFrame
        DataFrame containing numerical features.
    df_categorical : pandas.DataFrame
        DataFrame containing categorical features.
    target_numerical : pandas.Series, optional
        Numerical target variable to include in the dependency matrix.
    target_categorical : pandas.Series, optional
        Categorical target variable to include in the dependency matrix.

    Other Parameters
    ----------------
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values.
    cmap : str, default='magma'
        Colormap for the heatmap.
    figsize : tuple, default=(5, 3)
        Figure size for the heatmap.
    annot_threshold : int, default=11
        Maximum allowed size of the dependency matrix dimension for which
        numeric annotations are displayed on heatmap.

    Returns
    -------
    VisualizationResult
        Heatmap visualization of the eta-squared dependency matrix.

    Notes
    -----
    - Eta-squared measures the proportion of variance in a numerical variable
      explained by a categorical variable.
    - This function is intended for internal use in non-linear dependency
      analysis presets.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks.relations_nonlinear import (
    ...     _get_eta_squared_heatmap
    ... )
    >>> df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    >>> df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    >>> heatmap_viz = _get_eta_squared_heatmap(df_num, df_cat)
    >>> type(heatmap_viz)
    <class 'explorica.types.VisualizationResult'>
    """
    other_params = {
        "nan_policy": kwargs.get("nan_policy", "drop"),
        "cmap": kwargs.get("cmap", "magma"),
        "figsize": kwargs.get("figsize", (5, 4)),
        "annot_threshold": kwargs.get("annot_threshold", 11),
    }
    if not target_numerical.empty:
        df_num = pd.concat([df_numerical.copy(), target_numerical], axis=1)
    else:
        df_num = df_numerical.copy()
    if not target_categorical.empty:
        df_cat = pd.concat([df_categorical.copy(), target_categorical], axis=1)
    else:
        df_cat = df_categorical.copy()
    df_num = handle_nan(df_num, other_params["nan_policy"])
    df_cat = handle_nan(df_cat, other_params["nan_policy"])
    dependency_matrix = corr_matrix_eta(df_num, df_cat)
    annot = other_params["annot_threshold"] >= max(dependency_matrix.shape)
    vr = heatmap(
        dependency_matrix,
        annot=annot,
        title="Dependency matrix (eta squared)",
        cmap=other_params["cmap"],
        figsize=other_params["figsize"],
    )
    return vr


def _get_cramer_v_heatmap(
    df_categorical: pd.DataFrame,
    target_categorical: pd.Series = None,
    nan_policy: Literal["drop", "raise"] = "drop",
    **kwargs,
) -> VisualizationResult:
    """
    Build a Cramer's V dependency heatmap between categorical features.

    Computes the Cramer's V dependency matrix for categorical–categorical
    feature pairs and renders it as a heatmap. Optionally includes a categorical
    target variable in the analysis.

    Parameters
    ----------
    df_categorical : pandas.DataFrame
        DataFrame containing categorical features.
    target_categorical : pandas.Series, optional
        Categorical target variable to include in the dependency matrix.
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values.

    Other Parameters
    ----------------
    cmap : str, default='magma'
        Colormap for the heatmap.
    figsize : tuple, default=(5, 4)
        Figure size for the heatmap.
    bias_correction : bool, default=True
        Whether to apply bias correction in Cramer's V computation.
    annot_threshold : int, default=11
        Maximum number of features for which numeric annotations are displayed
        on the heatmap. If the number of features in
        ``df_categorical`` + ``target_categorical`` exceeds
        this threshold, ``annot`` is automatically set to False to prevent
        clutter and unreadable text in the figure.

    Returns
    -------
    VisualizationResult
        Heatmap visualization of the Cramer's V dependency matrix.

    Notes
    -----
    - Cramer's V measures the strength of association between categorical variables.
    - This function is intended for internal use in non-linear dependency
      analysis presets.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks.relations_nonlinear import (
    ... _get_cramer_v_heatmap
    ... )
    >>> df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    >>> heatmap_viz = _get_cramer_v_heatmap(df_cat)
    >>> type(heatmap_viz)
    <class 'explorica.types.VisualizationResult'>
    """
    other_params = {
        "cmap": kwargs.get("cmap", "magma"),
        "figsize": kwargs.get("figsize", (5, 4)),
        "bias_correction": kwargs.get("bias_correction", True),
        "annot_threshold": kwargs.get("annot_threshold", 11),
    }
    if not target_categorical.empty:
        df = handle_nan(
            pd.concat([df_categorical, target_categorical], axis=1), nan_policy
        )
    else:
        df = df_categorical.copy()
    dependency_matrix = corr_matrix_cramer_v(
        df, bias_correction=other_params["bias_correction"]
    )
    annot = other_params["annot_threshold"] >= df.shape[1]
    vr = heatmap(
        dependency_matrix,
        annot=annot,
        title="Dependency matrix (Cramer's V)",
        figsize=other_params["figsize"],
        cmap=other_params["cmap"],
    )
    return vr


def _get_highest_dependency_pairs_table(
    df_numerical: pd.DataFrame,
    df_categorical: pd.DataFrame,
    target_categorical: pd.Series,
    **kwargs,
) -> TableResult:
    """
    Build a table of highest non-linear dependency pairs using a categorical target.

    Computes pairwise dependency coefficients between numerical and categorical
    features, ranks them by absolute value, and filters for non-linear dependency
    methods: η² (eta squared, numerical-categorical) and Cramer's V
    (categorical-categorical), using the provided categorical target variable.

    Parameters
    ----------
    df_numerical : pandas.DataFrame
        DataFrame containing numerical features.
    df_categorical : pandas.DataFrame
        DataFrame containing categorical features.
    target_categorical : pandas.Series
        Categorical target variable to include in the dependency computation.

    Other Parameters
    ----------------
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values.
    rows : int, default=5
        Number of top dependency pairs to include in the table.
    round_digits : int, default=4
        Number of decimal places to round coefficients.


    Returns
    -------
    TableResult
        Table containing the highest non-linear dependency pairs, filtered
        to include only η² (numerical-categorical) and Cramer's V
        (categorical-categorical) methods.

    Notes
    -----
    - Eta-squared (η²) measures the proportion of variance in a numerical
      variable explained by a categorical variable, using the categorical target.
    - Cramer's V measures the strength of association between categorical variables.
    - Intended for internal use in non-linear dependency analysis presets.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks.relations_nonlinear import (
    ...     _get_highest_dependency_pairs_table)
    >>> df_num = pd.DataFrame({'x1': [1,2,3], 'x2': [4,5,6]})
    >>> df_cat = pd.DataFrame({'c1': ['a','b','a'], 'c2': ['x','y','x']})
    >>> target = pd.Series(['a','b','a'], name='target')
    >>> table = _get_highest_dependency_pairs_table(df_num, df_cat, target)
    >>> type(table)
    <class 'explorica.types.TableResult'>
    """
    other_params = {
        "nan_policy": kwargs.get("nan_policy", "drop"),
        "rows": kwargs.get("rows", 5),
        "round_digits": kwargs.get("round_digits", 4),
    }

    df_num = handle_nan(df_numerical.copy(), other_params["nan_policy"])
    df_cat = pd.concat([df_categorical, target_categorical], axis=1)
    df_cat = handle_nan(df_cat, other_params["nan_policy"])
    pairs = high_corr_pairs(
        numeric_features=df_num,
        category_features=df_cat,
        y=target_categorical.name,
        threshold=0.0,
    )

    pairs = pairs.loc[(pairs["method"] == "cramer_v") | (pairs["method"] == "eta")]
    pairs["coef"] = np.round(pairs["coef"], other_params["round_digits"])
    return TableResult(
        pairs.head(other_params["rows"]),
        title="Highest non-linear dependencies",
        description="Includes methods: Cramer's V and η²",
        render_extra={"show_index": False},
    )
