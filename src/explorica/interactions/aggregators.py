"""
Module provides utilities for aggregating interactions between features
in a dataset. It contains functions to identify and return significant
feature pairs based on various correlation and association measures.

The main function, `high_corr_pairs`, evaluates feature-to-feature
relationships using linear (Pearson, Spearman), non-linear (e.g.
exponential, binomial, power-law), categorical (Cramér's V), and hybrid
(η²) measures. Users can optionally enable non-linear and multiple-
correlation modes.

Private helper functions
------------------------
These functions support the internal computations of `high_corr_pairs`
and are not intended for direct use:

- _high_corr_pairs_get_execution_queue
- _high_corr_pairs_run_execution_queue
- _high_corr_pairs_extract_corr_pairs
- _high_corr_pairs_extract_by_numeric
- _high_corr_pairs_extract_by_multiple
"""

import warnings
from numbers import Number
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    validate_at_least_one_exist,
    validate_lengths_match,
    validate_string_flag,
    validate_unique_column_names,
)
from explorica.interactions.correlation_matrices import CorrelationMatrices

_errors = read_config("messages")["errors"]


def detect_multicollinearity(
    numeric_features: Sequence[Sequence[Number]] = None,
    category_features: Sequence[Sequence] = None,
    method: str = "VIF",
    return_as: str = "dataframe",
    **kwargs,
) -> dict | pd.DataFrame:
    """
    Detect multicollinearity among features using either Variance Inflation Factor (VIF)
    or correlation-based methods.

    Parameters
    ----------
    numeric_features : Sequence of sequences of numbers, optional
        Numerical feature matrix or compatible structure (array-like or DataFrame).
        Required for ``method='VIF'``. Used together with `category_features` when
        correlation-based method is selected.
    category_features : Sequence of sequences, optional
        Categorical feature matrix or compatible structure (array-like or DataFrame).
        Only used with ``method='corr'``. Not evaluated under VIF.
    method : {"VIF", "corr"}, default="VIF"
        Method to detect multicollinearity:
        - "VIF" : Compute Variance Inflation Factor for numerical features.
        - "corr" : Detect multicollinearity based on the highest pairwise absolute
          correlation between features (numeric–numeric, numeric–categorical,
          categorical–categorical).
          Supported correlation metrics include: ``sqrt_eta_squared``, ``cramer_v``,
          ``pearson``, ``spearman``.
    return_as : {"dataframe", "dict"}, default="dataframe"
        Output format of the result:
        - "dataframe" : Pandas DataFrame with features as index and metrics as columns.
        - "dict" : Nested dictionary of the form
          ``{metric: {feature: value, ...}, ...}``.
    variance_inflation_threshold : float, default=10
        Threshold above which a feature is considered collinear in VIF method.
    correlation_threshold : float, default=0.95
        Threshold for the highest absolute correlation of a feature with any other
        feature. If this value is exceeded, the feature is considered collinear.

    Returns
    -------
    dict or pd.DataFrame
        Multicollinearity assessment, depending on ``return_as``:
        - If "dataframe": DataFrame with columns for metrics (e.g., "VIF",
          "multicollinearity") and rows corresponding to features.
        - If "dict": Mapping of metrics to per-feature values.

    Raises
    ------
    ValueError
        If all inputs are empty.
        If lengths of `numeric_features` and `category_features` do not match.
        If any input array contains NaN values.
        If `method` or `return_as` is not one of the supported values.

    Notes
    -----
    - VIF can be infinite if the dataset contains functionally dependent features.
    - Categorical features are not evaluated under VIF.

    Examples
    --------
    >>> import pandas as pd
    >>> from explorica.interactions import detect_multicollinearity
    >>> X_num = pd.DataFrame({"x1": [1, 2, 3], "x2": [2, 4, 6], "x3": [1, 0, 1]})
    >>> detect_multicollinearity(X_num, method="VIF", return_as="dataframe")
            VIF  multicollinearity
    x1     inf                  1
    x2     inf                  1
    x3     1.5                  0
    """
    params = {
        "variance_inflation_threshold": 10,
        "correlation_threshold": 0.95,
        **kwargs,
    }

    validate_string_flag(
        method.lower(),
        {
            "variance_inflation",
            "variance_inflation_factor",
            "vif",
            "corr",
            "correlation",
        },
        err_msg=_errors["unsupported_method_f"].format(method, {"VIF", "corr"}),
    )
    validate_string_flag(
        return_as.lower(),
        {"dataframe", "df", "dict", "dictionary", "mapping"},
        err_msg=_errors["unsupported_method_f"].format(
            return_as, {"dataframe", "dict"}
        ),
    )
    validate_at_least_one_exist(
        (numeric_features, category_features),
        err_msg=_errors["InteractionAnalyzer"]["high_corr_pairs"][
            "features_do_not_exists"
        ],
    )
    df_numeric = convert_dataframe(numeric_features)
    df_category = convert_dataframe(category_features)
    if numeric_features is not None and category_features is not None:
        validate_lengths_match(
            df_numeric,
            df_category,
            err_msg=_errors["arrays_lens_mismatch_f"].format(
                "numeric_features", "category_features"
            ),
        )
    if numeric_features is not None:
        validate_array_not_contains_nan(
            df_numeric,
            err_msg=_errors["array_contains_nans_f"].format("numeric_features"),
        )
    if category_features is not None:
        validate_array_not_contains_nan(
            df_category,
            err_msg=_errors["array_contains_nans_f"].format("category_features"),
        )

    if method.lower() in {"variance_inflation", "variance_inflation_factor", "vif"}:
        result = {"VIF": {}, "multicollinearity": {}}
        df_wc = add_constant(df_numeric)
        cols = df_wc.columns
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="divide by zero"
            )
            for i, col in enumerate(cols):
                if i == 0:
                    # do not calculate VIF for a const
                    continue
                result["VIF"][col] = variance_inflation_factor(df_wc, i)
                result["multicollinearity"][col] = (
                    1.0
                    if result["VIF"][col] >= params["variance_inflation_threshold"]
                    else 0.0
                )

    elif method.lower() in {"corr", "correlation"}:
        result = {"highest_correlation": {}, "multicollinearity": {}}
        pairs = high_corr_pairs(numeric_features, category_features, threshold=0)
        cols = set(df_numeric.columns) | set(df_category.columns)
        for i, col in enumerate(cols):
            subset = pairs.loc[pairs["Y"] == col, "coef"]
            result["highest_correlation"][col] = subset.loc[subset.abs().idxmax()]
            if np.abs(result["highest_correlation"][col]) >= (
                params["correlation_threshold"]
            ):
                result["multicollinearity"][col] = 1
            else:
                result["multicollinearity"][col] = 0
    if return_as in {"dataframe", "df"}:
        result = pd.DataFrame(result)
    return result


def high_corr_pairs(
    numeric_features: Sequence[Sequence[Number]] = None,
    category_features: Sequence[Sequence] = None,
    threshold: float = 0.7,
    **kwargs,
) -> pd.DataFrame | None:
    """
    Finds and returns all significant pairs of
    correlated features from the input datasets.

    This method evaluates feature-to-feature relationships using a set of
    correlation measures, including linear (Pearson, Spearman), non-linear
    (e.g. exponential, binomial, power-law), categorical (Cramér’s V), and
    hybrid (η²). Users can optionally enable non-linear and multiple-correlation
    modes.

    Parameters
    ----------
    numeric_features : pd.DataFrame, optional
        A DataFrame of numerical features.
        Required for linear, η², non-linear, and multiple correlation.
    category_features : pd.DataFrame, optional
        A DataFrame of categorical features.
        Required for Cramér’s V and η² computations.
    y : str, optional
        Target feature name to compute correlations with.
        If None, all pairwise comparisons are evaluated.
    nonlinear_included : bool, default=False
        Whether to include non-linear correlation measures for numeric features.
    multiple_included : bool, default=False
        Whether to include multiple
        correlation analysis (for numeric features only).
    threshold : float, default=0.7
        Minimum absolute value of correlation
        to consider a pair as significantly dependent.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with columns ['X', 'Y', 'coef', 'method'], listing feature
        pairs whose correlation (in absolute value) exceeds the threshold.
        Returns None if no such pairs found.

    Raises
    ------
    ValueError
        If neither input DataFrame is provided.
        If numeric_features or category_features are of unequal lengths.
        If numeric_features or category_features contain NaN values.
        If numeric_features or category_features contain duplicate column name.

    Notes
    -----
    - Linear correlation methods: Pearson, Spearman
    - Non-linear methods (enabled via nonlinear_included):
      exp, binomial, ln, hyperbolic, power
    - Categorical methods: Cramér’s V, η² (eta)
    - The method skips self-comparisons.
    - Targeted correlation (`y`) will produce only
      pairs involving the specified target.
    """

    def validation():
        # Nan's absence check
        if numeric_features is not None:
            validate_array_not_contains_nan(
                numeric_df,
                err_msg=_errors["array_contains_nans_f"].format("numeric_features"),
            )
            validate_unique_column_names(
                numeric_df,
                _errors["duplicate_column_names_f"].format("numeric_features"),
            )
        if category_features is not None:
            validate_array_not_contains_nan(
                category_df,
                err_msg=_errors["array_contains_nans_f"].format("category_features"),
            )
            validate_unique_column_names(
                category_df,
                _errors["duplicate_column_names_f"].format("category_features"),
            )

        # checking at least 1 not None DataFrame
        validate_at_least_one_exist(
            (numeric_df, category_df),
            _errors["InteractionAnalyzer"]["high_corr_pairs"]["features_do_not_exists"],
        )

        # checking for lengths match
        if numeric_df is not None and category_df is not None:
            validate_lengths_match(
                numeric_df,
                category_df,
                _errors["InteractionAnalyzer"]["high_corr_pairs"][
                    "features_lens_mismatch_f"
                ].format(numeric_df.shape[0], category_df.shape[0]),
            )
        if kwargs.get("multiple_included") and len(numeric_cols) > 15:
            warnings.warn(
                "Multifactor correlation with more than 15 features "
                "may be extremely slow (O(n^3)). Consider reducing feature set.",
                UserWarning,
            )

        # checking if 'y' feature is present in at least one input DataFrame
        supported_names = set(numeric_cols) | set(category_cols)
        if y is not None:
            validate_string_flag(
                y,
                supported_names,
                err_msg=_errors["InteractionAnalyzer"]["high_corr_pairs"][
                    "y_do_not_exists_f"
                ].format(y),
            )

    def get_columns(df):
        return list(df.columns) if df is not None else []

    y = kwargs.get("y")

    numeric_df, category_df = None, None
    if numeric_features is not None:
        numeric_df = convert_dataframe(numeric_features)

    if category_features is not None:
        category_df = convert_dataframe(category_features)

    # checking column names of input DataFrame for duplicates

    numeric_cols = get_columns(numeric_df)
    category_cols = get_columns(category_df)

    validation()

    # form a queue of execution
    execution = _high_corr_pairs_get_execution_queue(
        numeric_cols,
        category_cols,
        y=y,
        multiple_included=kwargs.get("multiple_included"),
        nonlinear_included=kwargs.get("nonlinear_included"),
    )

    corr_pairs = _high_corr_pairs_run_execution_queue(
        numeric_df,
        category_df,
        execution,
        threshold,
        y=y,
        category_cols=category_cols,
        numeric_cols=numeric_cols,
    )
    if corr_pairs.shape[0] == 0:
        return None

    return corr_pairs.sort_values(by="coef", key=abs, ascending=False).reset_index(
        drop=True
    )


def _high_corr_pairs_get_execution_queue(
    numeric_cols: list,
    category_cols: list,
    y: str,
    multiple_included: bool,
    nonlinear_included: bool,
) -> list[str]:
    execution = []
    if y is None or y in numeric_cols:
        if len(numeric_cols) >= 2:
            execution.extend(["pearson", "spearman"])
            if nonlinear_included:
                execution.extend(["exp", "binomial", "ln", "hyperbolic", "power"])
            # multiple correlation works with at least 3 features
            if multiple_included:
                if len(numeric_cols) >= 3:
                    execution.append("multiple")

    # eta correlation works with at least 1 numeric and 1 category features
    # or with at least 1 'both' feature
    if y is None or y in category_cols:
        if len(category_cols) >= 2:
            execution.append("cramer_v")
        if len(numeric_cols) > 0 and len(category_cols) > 0:
            # special case validation: feature is compared only with itself
            if not (
                len(category_cols) == 1
                and len(numeric_cols) == 1
                and category_cols == numeric_cols
            ):
                execution.append("eta")
    return execution


def _high_corr_pairs_run_execution_queue(
    numeric_df: pd.DataFrame,
    category_df: pd.DataFrame,
    execution: list[str],
    threshold: float,
    **kwargs,
):

    numeric_cols = kwargs.get("numeric_cols")
    category_cols = kwargs.get("category_cols")
    y = kwargs.get("y")
    corr_pairs = pd.DataFrame()
    # execution cycle
    for method in execution:
        if method in {
            "pearson",
            "spearman",
            "exp",
            "binomial",
            "ln",
            "hyperbolic",
            "power",
        }:
            pairs_by_method = _high_corr_pairs_extract_by_numeric(
                numeric_df, method, threshold, y
            )
            corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
        elif method == "multiple":
            pairs_by_method = _high_corr_pairs_extract_by_multiple(
                numeric_df, threshold, y
            )
            corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
        elif method == "cramer_v":
            corr_matrix = CorrelationMatrices.corr_matrix(category_df, method=method)
            pairs_by_method = _high_corr_pairs_extract_corr_pairs(
                corr_matrix, y, method
            )
            pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
            corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
        elif method == "eta":
            corr_matrix = CorrelationMatrices.corr_matrix(
                numeric_df, method, category_df
            )
            both_cols = set(category_cols) & set(numeric_cols)
            pairs_by_method = _high_corr_pairs_extract_corr_pairs(
                corr_matrix, y, method, drop_diagonal=False
            )
            pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
            # conditions under which feature's comparisons with self occurs
            pairs_by_method = pairs_by_method[
                ~(
                    (pairs_by_method["Y"] == pairs_by_method["X"])
                    & (pairs_by_method["X"].isin(both_cols))
                )
            ]
            corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
    return corr_pairs


def _high_corr_pairs_extract_corr_pairs(
    df: pd.DataFrame, y: str = None, method: str = None, drop_diagonal: bool = True
) -> pd.DataFrame:
    if y is not None:
        columns = pd.Series([y])
    else:
        columns = pd.Series(df.columns)
    pairs = pd.DataFrame()
    for i, col in enumerate(columns):
        col_coefs = df[col]
        if drop_diagonal:
            if y is not None:
                self_comparison = df.columns.get_loc(y)
            else:
                self_comparison = i
            col_coefs = col_coefs.drop(col_coefs.index[self_comparison])
        # ignore feature's comparisons with self
        col_indexes = col_coefs.index.to_numpy()
        col_coefs = col_coefs.reset_index(drop=True)
        col_column = np.repeat(col, col_coefs.size)
        col_method = np.repeat(method, col_coefs.size)
        col_pairs = pd.DataFrame(
            {
                "X": col_indexes,
                "Y": col_column,
                "coef": col_coefs,
                "method": col_method,
            }
        )
        pairs = pd.concat([pairs, col_pairs], ignore_index=True)
    return pairs


def _high_corr_pairs_extract_by_numeric(df, method, threshold, y):
    corr_matrix = CorrelationMatrices.corr_matrix(df, method=method)
    pairs_by_method = _high_corr_pairs_extract_corr_pairs(corr_matrix, y, method)
    pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
    return pairs_by_method


def _high_corr_pairs_extract_by_multiple(df, threshold, y):
    # using the "correlation vector"
    # will be less expensive than using the corr matrix
    if y is not None:
        factors = df.copy()
        target = factors[y]
        del factors[y]
        corr_matrix = CorrelationMatrices.corr_vector_multiple(factors, target)
        columns = pd.Series([y])
    else:
        corr_matrix = CorrelationMatrices.corr_matrix(df, method="multiple")
        columns = pd.Series(corr_matrix["target"]).drop_duplicates()
        corr_matrix = corr_matrix[corr_matrix["target"].isin(columns)]
    corr_matrix = corr_matrix[corr_matrix["corr_coef"] >= threshold]
    col_coefs = corr_matrix["corr_coef"]
    col_method = np.repeat("multiple", col_coefs.size)
    col_pairs = pd.DataFrame(
        {
            "X": corr_matrix["feature_combination"],
            "Y": corr_matrix["target"],
            "coef": corr_matrix["corr_coef"],
            "method": col_method,
        }
    )
    return col_pairs
