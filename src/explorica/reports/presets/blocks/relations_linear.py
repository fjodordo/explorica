"""
Linear relations block preset.

Provides an overview of linear associations between numeric features and a
specified target variable as an Explorica `Block`.

The block focuses on correlation-based analysis and is intended to be used
as part of interaction-focused reports (e.g., Exploratory Data Analysis
with a defined target). It summarizes pairwise linear relationships using
both Pearson and Spearman correlation coefficients and highlights the
strongest correlations involving the target.

Functions
---------
get_linear_relations_block(data, target, round_digits=4, nan_policy="drop")
    Build a Block instance summarizing linear relationships in a dataset.

Notes
-----
- Only numeric features are included in the analysis.
- Pearson correlation captures linear relationships under the assumption
  of approximately linear dependence.
- Spearman correlation captures monotonic relationships and is less
  sensitive to outliers.
- The target variable is included in correlation matrices and is required
  for ranking the highest correlation pairs.
- Correlation significance (p-values) is not included and may be added
  in future releases.

Examples
--------
>>> import pandas as pd
>>> from explorica.reports.presets.blocks import get_linear_relations_block
>>> df = pd.DataFrame({
...     "x1": [1, 2, 3, 4],
...     "x2": [2, 4, 6, 8],
...     "x3": [4, 3, 2, 1]
... })
>>> y = pd.Series([1, 0, 1, 0], name="target")
>>> block = get_linear_relations_block(df, y)
>>> block.block_config.title
'Linear relations'
>>> block.block_config.tables[0].table
    X       Y    coef    method
0  x1  target -0.4472   pearson
1  x2  target -0.4472   pearson
2  x3  target  0.4472   pearson
3  x1  target -0.4472  spearman
4  x2  target -0.4472  spearman
"""

from typing import Sequence, Mapping, Any, Literal

import numpy as np
import pandas as pd

from ...._utils import handle_nan, convert_series, convert_dataframe
from ....types import TableResult, VisualizationResult, NaturalNumber
from ...core.block import Block, BlockConfig
from ....interactions import (
    corr_matrix_linear,
    high_corr_pairs,
    detect_multicollinearity,
)
from ....visualizations import heatmap, scatterplot, hexbin


def get_linear_relations_block(
    data: Sequence[Any] | Mapping[str, Sequence[Any]],
    target: Sequence[Any] | Mapping[str, Sequence[Any]] = None,
    sample_size_threshold: NaturalNumber = 5000,
    round_digits: int = 4,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> Block:
    """
    Generate a `Block` summarizing linear relationships in a dataset.

    This block provides an overview of linear associations between numeric
    features and a specified target variable. It includes correlation
    matrices (Pearson and Spearman) and a table of the highest correlation
    pairs ranked by absolute coefficient values.

    Parameters
    ----------
    data : Sequence[Any] or Mapping[str, Sequence[Any]]
        Input dataset. Must be convertible to a pandas DataFrame.
        Only numeric columns are considered.
    target : Sequence[Any] or Mapping[str, Sequence[Any]], optional
        Target variable for correlation analysis. Must be convertible to a
        pandas Series. If not provided, target-specific tables and visualizations
        are skipped.
    sample_size_threshold : int, default=5000
        Threshold on the number of observations used to switch between
        scatter plots and hexbin plots for feature-target visualizations.
    round_digits : int, default=4
        Number of decimal places for rounding correlation and diagnostic
        coefficients.
    nan_policy : {'drop', 'raise'}, default='drop'
        Policy for handling missing values:
        - 'drop' : remove rows with missing values.
        - 'raise': raise an error if missing values are present.

    Returns
    -------
    Block
        An Explorica `Block` containing:
        - Pearson correlation matrix between numeric features (and target if provided)
        - Spearman correlation matrix between numeric features (and target if provided)
        - Multicollinearity diagnostic table based on Variance Inflation Factor (VIF),
          included if number of numeric features >= 2
        - Multicollinearity diagnostic table based on highest pairwise correlation,
          included if number of numeric features >= 2
        - If a target is provided:
            - Table of highest correlation pairs (features vs target)
            - Feature-target visualizations (scatterplots if number of
              rows <= sample_size_threshold, hexbin plots otherwise)

    Returns
    -------
    Block
        An Explorica `Block` containing:
        - Pearson correlation matrix (numeric features + target)
        - Spearman correlation matrix (numeric features + target)
        - Table of highest correlation pairs, ranked by absolute coefficient values
        - Multicollinearity diagnostic table based on Variance Inflation Factor (VIF)
        - Multicollinearity diagnostic table based on highest pairwise correlation
        - Feature-target visualizations (scatter or hexbin, depending on sample size)

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.reports.presets.blocks import get_linear_relations_block
    >>> df = pd.DataFrame({
    ...     "x1": [1, 2, 3, 4],
    ...     "x2": [2, 4, 6, 8],
    ...     "x3": [4, 3, 2, 1]
    ... })
    >>> y = pd.Series([1, 0, 1, 0], name="target")
    >>> block = get_linear_relations_block(df, y)
    >>> block.block_config.title
    'Linear relations'
    >>> block.block_config.tables[0].table
        X       Y    coef    method
    0  x1  target -0.4472   pearson
    1  x2  target -0.4472   pearson
    2  x3  target  0.4472   pearson
    3  x1  target -0.4472  spearman
    4  x2  target -0.4472  spearman
    """
    df = convert_dataframe(data).select_dtypes("number")
    df = handle_nan(df, nan_policy)
    y = None
    if target is not None:
        y = convert_series(target)
        y = handle_nan(y, nan_policy)

    block = Block(BlockConfig(title="Linear relations"))

    # Add corr matrices
    block.add_visualization(_get_correlation_matrix_heatmap(df, y, method="pearson"))
    block.add_visualization(_get_correlation_matrix_heatmap(df, y, method="spearman"))

    # Add highest correlation pairs (absolute)
    # Doesn't work without target
    if y is not None:
        pairs = _get_highest_correlation_table(df, y, round_digits)
        block.add_table(
            pairs,
            title="Highest correlation pairs",
            description="Ranked by absolute coefficient values.",
        )

    multicoll_thresholds = {"VIF": 10, "corr": 0.95}

    # Features count must exceed 2 to multicollinearity diagnostic
    if df.shape[1] >= 2:
        # Add multicollinearity VIF diagnostic table
        multicoll_vif = _get_multicollinearity_table(
            df, method="VIF", thresholds=multicoll_thresholds
        )
        block.add_table(multicoll_vif)

        # Add multicollinearity highest correlation diagnostic table
        multicoll_corr = _get_multicollinearity_table(
            df, method="corr", thresholds=multicoll_thresholds
        )
        block.add_table(multicoll_corr)

    # Add visualizations
    # Doesn't work without target
    if y is not None:
        if df.shape[0] <= sample_size_threshold:
            visualizations = _get_scatterplots(df, y, nan_policy=nan_policy)
            for vr in visualizations:
                block.add_visualization(vr)
        else:
            visualizations = _get_hexbins(df, y, nan_policy=nan_policy)
            for vr in visualizations:
                block.add_visualization(vr)

    return block


def _get_highest_correlation_table(
    df: pd.DataFrame,
    y: pd.Series,
    round_digits: int = 4,
    nan_policy: Literal["drop", "raise"] = "drop",
) -> TableResult:
    """
    Build a table of highest linear correlation pairs with the target variable.

    Computes pairwise linear correlation coefficients between numerical features
    and the target variable, ranks them by absolute coefficient value, and
    returns the top correlated pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical features.
    y : pandas.Series
        Target variable for correlation analysis.
    round_digits : int, default=4
        Number of decimal places to round correlation coefficients.
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values before computing correlations.

    Returns
    -------
    TableResult
        Table containing the top correlated feature–target pairs,
        sorted by absolute correlation coefficient.

    Notes
    -----
    - Uses linear correlation metrics provided by `high_corr_pairs`
      (e.g. Pearson, Spearman depending on implementation).
    - Correlations are computed on the combined dataset
      ``[df, y]`` after NaN handling.
    - Intended for internal use in linear relations analysis blocks.
    """
    df_full = handle_nan(pd.concat([df, y], axis=1), nan_policy)
    pairs = TableResult(
        high_corr_pairs(df_full, y=y.name, threshold=0.0).head(5),
        render_extra={"show_index": False},
    )
    pairs.table["coef"] = np.round(pairs.table["coef"], round_digits)
    return pairs


def _get_multicollinearity_table(
    df: pd.DataFrame,
    method: Literal["VIF", "corr"],
    thresholds: dict,
    round_digits: int = 4,
) -> TableResult:
    """
    Build a multicollinearity diagnostic table.

    Detects multicollinearity among numerical features using the specified
    detection method and returns a diagnostic table with boolean flags
    indicating whether multicollinearity is present according to the given
    thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical features.
    method : {'VIF', 'corr'}
        Multicollinearity detection method:
        - 'VIF' : variance inflation factor based detection.
        - 'corr': highest pairwise correlation based detection.
    thresholds : dict
        Threshold values for multicollinearity detection.
        Expected keys:
        - 'VIF'  : threshold for variance inflation factor.
        - 'corr' : threshold for absolute correlation.
    round_digits : int, default=4
        Number of decimal places to round diagnostic statistics.

    Returns
    -------
    TableResult
        Table containing multicollinearity diagnostics with:
        - method-specific statistic (VIF or highest correlation),
        - boolean flag ``is_multicollinearity`` indicating detected issues.

    Notes
    -----
    - For ``method='VIF'``, multicollinearity is assessed using variance
      inflation factors computed for each feature.
    - For ``method='corr'``, multicollinearity is assessed using the highest
      absolute pairwise correlation per feature.
    - Intended for internal use in linear relations analysis blocks.
    """
    multycoll_table = detect_multicollinearity(
        df,
        method=method,
        variance_inflation_threshold=thresholds["VIF"],
        correlation_threshold=thresholds["corr"],
    ).rename({"multicollinearity": "is_multicollinearity"}, axis=1)
    if method == "VIF":
        multycoll_table["VIF"] = np.round(multycoll_table["VIF"], round_digits)
    else:
        multycoll_table["highest_correlation"] = np.round(
            multycoll_table["highest_correlation"], round_digits
        )
    multycoll_table["is_multicollinearity"] = multycoll_table[
        "is_multicollinearity"
    ].astype("bool")
    title = "Multicollinearity diagnostic"
    title += " (VIF)" if method == "VIF" else " (highest correlation)"
    description = (
        f"Calculated with VIF threshold: {thresholds['VIF']}"
        if method == "VIF"
        else f"Calculated with correlation threshold: {thresholds['corr']}"
    )

    multycoll_table = TableResult(
        table=multycoll_table, title=title, description=description
    )
    return multycoll_table


def _get_correlation_matrix_heatmap(
    df: pd.DataFrame,
    y: pd.Series | None,
    method: str,
    nan_policy: Literal["drop", "raise"] = "drop",
    **kwargs,
) -> VisualizationResult:
    """
    Build a correlation matrix heatmap for numerical features and target.

    Computes a linear correlation matrix between numerical features and the
    target variable using the specified correlation method and visualizes
    it as a heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical features.
    y : pandas.Series
        Target variable to include in the correlation matrix.
    method : str
        Correlation method to use. Typically ``'pearson'`` or ``'spearman'``.
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values before computing correlations.

    Other Parameters
    ----------------
    cmap : str, default='magma'
        Colormap for the heatmap.
    figsize : tuple, default=(5, 4)
        Figure size of the heatmap.
    annot_threshold : int, default=11
        Maximum number of features for which numeric annotations are displayed
        on the heatmap. If the number of features in ``df`` + ``y`` exceeds
        this threshold, ``annot`` is automatically set to False to prevent
        clutter and unreadable text in the figure.

    Returns
    -------
    VisualizationResult
        Heatmap visualization of the correlation matrix.

    Notes
    -----
    - Correlations are computed on the combined dataset
      ``[df, y]`` after NaN handling.
    - Intended for internal use in linear relations analysis blocks.
    """
    other_params = {
        "cmap": kwargs.get("cmap", "magma"),
        "figsize": kwargs.get("figsize", (5, 4)),
        "annot_threshold": kwargs.get("annot_threshold", 11),
    }
    df_full = handle_nan(pd.concat([df, y], axis=1), nan_policy)
    title = (
        "Correlation matrix (Pearson)"
        if method == "pearson"
        else "Correlation matrix (Spearman)"
    )
    annot = other_params["annot_threshold"] >= df_full.shape[1]
    vr = heatmap(
        corr_matrix_linear(df_full, method),
        annot=annot,
        title=title,
        figsize=other_params["figsize"],
        cmap=other_params["cmap"],
    )
    return vr


def _get_hexbins(
    df: pd.DataFrame,
    y: pd.Series,
    nan_policy: Literal["drop", "raise"] = "drop",
    **kwargs,
) -> list[VisualizationResult]:
    """
    Build hexbin visualizations for numerical features against a target.

    Generates hexbin density plots for each numerical feature in ``df``
    plotted against the target variable. Intended for large datasets
    where scatter plots become overplotted.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical features.
    y : pandas.Series
        Target variable to plot against each feature.
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values before visualization.

    Other Parameters
    ----------------
    gridsize : int, default=30
        Number of hexagons in the x-direction of the hexbin plot.
    cmap : str, default='magma'
        Colormap used for hexbin density.
    figsize : tuple, default=(5, 3)
        Figure size of each hexbin plot.
    style : str, default='whitegrid'
        Visualization style applied to the plot.

    Returns
    -------
    list of VisualizationResult
        List of hexbin visualizations, one per numerical feature.

    Notes
    -----
    - Hexbin plots are preferred over scatter plots for large sample sizes
      to reduce overplotting.
    - Each returned visualization is tagged with
      ``extra_info = {'kind': 'hexbin'}`` for internal identification.
    - Intended for internal use in linear relations analysis blocks.
    """
    other_params = {
        "gridsize": kwargs.get("gridsize", 30),
        "cmap": kwargs.get("cmap", "magma"),
        "figsize": kwargs.get("figsize", (5, 3)),
        "style": kwargs.get("style", "whitegrid"),
    }
    df_full = handle_nan(pd.concat([df, y], axis=1), nan_policy)
    vis_results = []
    for col in df.columns:
        vr = hexbin(
            df_full[col],
            df_full[y.name],
            cmap=other_params["cmap"],
            figsize=other_params["figsize"],
            gridsize=other_params["gridsize"],
            title=f"Hexbin '{col}' on target",
            xtitle=f"'{col}' value",
            ytitle="target value",
            style=other_params["style"],
        )
        vr.extra_info = {"kind": "hexbin"}
        vis_results.append(vr)
    return vis_results


def _get_scatterplots(
    df: pd.DataFrame,
    y: pd.Series,
    nan_policy: Literal["drop", "raise"] = "drop",
    **kwargs,
) -> list[VisualizationResult]:
    """
    Build scatter plot visualizations for numerical features against a target.

    Generates scatter plots for each numerical feature in ``df`` plotted
    against the target variable. Intended for smaller datasets where
    individual observations can be clearly visualized.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numerical features.
    y : pandas.Series
        Target variable to plot against each feature.
    nan_policy : {'drop', 'raise'}, default='drop'
        How to handle missing values before visualization.

    Other Parameters
    ----------------
    figsize : tuple, default=(5, 3)
        Figure size of each scatter plot.
    style : str, default='whitegrid'
        Visualization style applied to the plot.
    palette : str, default='magma'
        Color palette used for scatter points.

    Returns
    -------
    list of VisualizationResult
        List of scatter plot visualizations, one per numerical feature.

    Notes
    -----
    - Scatter plots are preferred for smaller sample sizes where
      point-wise inspection is meaningful.
    - Each returned visualization is tagged with
      ``extra_info = {'kind': 'scatterplot'}`` for internal identification.
    - Intended for internal use in linear relations analysis blocks.
    """
    other_params = {
        "figsize": kwargs.get("figsize", (5, 3)),
        "style": kwargs.get("style", "whitegrid"),
        "palette": kwargs.get("palette", "magma"),
    }
    vis_results = []
    df_full = handle_nan(pd.concat([df, y], axis=1), nan_policy=nan_policy)
    for col in df.columns:
        vr = scatterplot(
            df_full[col],
            target=df_full[y.name],
            palette=other_params["palette"],
            title=f"Scatterplot '{col}' on target",
            xtitle=f"'{col}' value",
            ytitle="target value",
            style=other_params["style"],
            figsize=other_params["figsize"],
        )
        vr.extra_info = {"kind": "scatterplot"}
        vis_results.append(vr)
    return vis_results
