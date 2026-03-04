"""
Module provides tools for constructing various types of
correlation and dependence matrices for both numerical and categorical features
in a dataset. This module is intended to be used via the public facade
`InteractionAnalyzer` but can also be used directly for advanced analyses.

The module supports linear correlations (Pearson, Spearman), categorical
associations (Cramér's V, η²), multiple-factor correlations, and correlation
indices for non-linear dependencies.

Functions
---------
corr_matrix(dataset, method="pearson", groups=None)
    Compute a correlation or association matrix using the specified method.
corr_matrix_linear(dataset, method="pearson")
    Compute a correlation matrix for numeric
    features using Pearson or Spearman correlation.
corr_matrix_cramer_v(dataset, bias_correction=True)
    Compute the Cramér's V correlation matrix for categorical variables.
corr_matrix_eta(dataset, categories)
    Compute a correlation matrix between numerical features and categorical features
    using the square root of eta-squared (η²).
corr_vector_multiple(x, y)
    Calculates multiple correlation coefficients between the target variable `y`
    and all possible combinations of 2 or more features from `x`.
corr_matrix_multiple(dataset)
    Compute a multi-factor correlation matrix for all target features in a dataset.
corr_matrix_corr_index(dataset, method="linear")
    Compute a correlation index matrix for all features in a dataset.
"""

import warnings
from itertools import combinations
from numbers import Number
from typing import Sequence

import numpy as np
import pandas as pd

from explorica._utils import (
    convert_dataframe,
    read_config,
    validate_array_not_contains_nan,
    validate_lengths_match,
    validate_string_flag,
    validate_unique_column_names,
)
from explorica.interactions.correlation_metrics import (
    corr_index,
    corr_multiple,
    cramer_v,
    eta_squared,
)

__all__ = [
    "corr_matrix",
    "corr_matrix_linear",
    "corr_matrix_cramer_v",
    "corr_matrix_eta",
    "corr_vector_multiple",
    "corr_matrix_multiple",
    "corr_matrix_corr_index",
]

_errors = read_config("messages")["errors"]
_warns = read_config("messages")["warns"]


def corr_matrix(
    dataset: Sequence[Sequence],
    method: str = "pearson",
    groups: Sequence[Sequence] = None,
) -> pd.DataFrame:
    """
    Compute a correlation or association matrix using the specified method.

    This function supports both classical correlation coefficients and a set of
    nonlinear correlation indices based on specific functional dependencies.
    For nonlinear methods, the correlation is evaluated by fitting transformations
    (e.g., exponential, logarithmic) and computing the strength of the relationship
    between features accordingly.

    Parameters
    ----------
    dataset : Sequence[Sequence]
        Sequence with numeric or categorical features.
        For 'pearson', 'spearman', 'multiple', and all nonlinear methods,
        all columns must be numeric.
        For 'cramer_v', all columns should be categorical.
        For 'eta', numeric features in `dataset` are
        compared to categorical features in `groups`.
    method : str, optional
        Method used to compute correlation or association:

        - 'pearson' : Pearson correlation (linear, continuous features).
        - 'spearman' : Spearman rank correlation (monotonic, non-parametric).
        - 'cramer_v' : Cramér's V (categorical-categorical association).
        - 'eta' : Eta coefficient (numeric-categorical association, asymmetric).
        - 'multiple' : Multiple correlation coefficients for each numeric feature
                       as a target and all remaining
                       numeric features as predictors.
        - 'exp' : Nonlinear correlation index assuming exponential dependence.
        - 'binomial' : Nonlinear correlation index assuming binomial dependence.
        - 'ln' : Nonlinear correlation index assuming logarithmic dependence.
        - 'hyperbolic' : Nonlinear correlation index assuming hyperbolic dependence.
        - 'power' : Nonlinear correlation index assuming power-law dependence.
    groups : Sequence[Sequence], optional
        Sequence of categorical grouping variables required for the 'eta' method.
        Must have the same number of rows as `dataset`.

    Returns
    -------
    pd.DataFrame

        - For 'pearson' and 'spearman':
          symmetric correlation matrix of shape
          (n_numeric_features, n_numeric_features).
        - For 'cramer_v':
          symmetric matrix of shape (n_features, n_features),
          representing categorical associations.
        - For 'eta':
          asymmetric matrix of shape (n_numeric_features, n_grouping_features),
          showing strength of association between numeric and categorical
          variables.
        - For nonlinear methods ('exp', 'binomial', 'ln', 'hyperbolic', 'power'):
          asymmetric matrix of shape (n_numeric_features, n_numeric_features),
          showing the correlation index based on the specified nonlinear model.

    Raises
    ------
    ValueError
        If the specified method is not supported.
        If `groups` is required (for 'eta')
        but not provided or mismatched in length.

    Warns
    -----
    UserWarning
        If Multicollinearity detected in the dataset (for 'multiple'),
        some features are linearly dependent.

    Notes
    -----
    - Pearson, Spearman, and other numerical correlation methods internally select
      only features of numeric type (`Number`) from the provided DataFrame.
      Non-numeric columns (e.g., categorical strings or object types) are ignored.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix
    >>> # Simple usage
    >>> data = pd.DataFrame({
    ...     "X1": [1, 2, 3, 4, 5, 6],
    ...     "X2": [12, 10, 8, 6, 4, 2],
    ...     "X3": [9, 3, 5, 2, 6, 1],
    ...     "X4": [3, 2, 1, 3, 2, 1],
    ... })
    >>> result_df = corr_matrix(data, method="spearman")
    >>> # Round coefficients for doctests reproducibility
    >>> np.round(result_df, 4)
            X1      X2      X3      X4
    X1  1.0000 -1.0000 -0.6000 -0.4781
    X2 -1.0000  1.0000  0.6000  0.4781
    X3 -0.6000  0.6000  1.0000  0.3586
    X4 -0.4781  0.4781  0.3586  1.0000
    """
    supported_methods = {
        "pearson",
        "spearman",
        "cramer_v",
        "eta",
        "multiple",
        "binomial",
        "ln",
        "exp",
        "hyperbolic",
        "power",
    }
    dataset_df = convert_dataframe(dataset)
    groups_df = convert_dataframe(groups)
    validate_unique_column_names(
        dataset_df, "Duplicate column names found in 'dataset' DataFrame"
    )
    validate_unique_column_names(
        groups_df, "Duplicate column names found in 'groups' DataFrame"
    )
    validate_array_not_contains_nan(
        dataset_df,
        err_msg=_errors["array_contains_nans_f"].format("dataset"),
    )
    validate_string_flag(
        method,
        supported_methods,
        err_msg=_errors["unsupported_method_f"].format(method, supported_methods),
    )
    if groups is None and method == "eta":
        raise ValueError("'groups' must be provided when using 'eta' method.")
    if groups is not None:
        validate_array_not_contains_nan(
            groups_df,
            err_msg=_errors["array_contains_nans_f"].format("groups"),
        )
        validate_lengths_match(
            dataset_df,
            groups_df,
            f"Length of 'groups' ({groups_df.shape[0]}) "
            f"must match length of 'dataset' ({dataset_df.shape[0]}).",
        )
    numeric_df = dataset_df.select_dtypes("number")
    matrix = None
    if method in {"pearson", "spearman"}:
        matrix = corr_matrix_linear(numeric_df, method)
    if method == "cramer_v":
        matrix = corr_matrix_cramer_v(dataset_df)
    if method == "eta":
        matrix = corr_matrix_eta(numeric_df, groups_df)
    if method == "multiple":
        matrix = corr_matrix_multiple(numeric_df)
    if method in {"exp", "binomial", "ln", "hyperbolic", "power"}:
        matrix = corr_matrix_corr_index(numeric_df, method)
    return matrix


def corr_matrix_linear(
    dataset: Sequence[Sequence[Number]], method: str = "pearson"
) -> pd.DataFrame:
    """
    Compute a correlation matrix for numeric features.

    Computes using Pearson or Spearman correlation.

    Parameters
    ----------
    dataset : Sequence[Sequence[Number]]
        Input dataset. Must be convertible
        to a pandas DataFrame with numeric columns.
    method : str, default="pearson"
        Correlation method to use. Supported values: "pearson", "spearman".

    Returns
    -------
    pd.DataFrame
        Correlation matrix of numeric features.
        Rows and columns correspond to feature names.

    Raises
    ------
    ValueError
        If `method` is not in {"pearson", "spearman"}.

    Notes
    -----
    - Only numeric columns are considered; non-numeric columns are ignored.
    - The dataset is automatically converted to a pandas DataFrame if it isn't one.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix_linear
    >>> # Pearson method usage
    >>>
    >>> X = pd.DataFrame({
    ...     "X1": [1, 3, 5, 6, 1],
    ...     "X2": [2, 3, 4, 1, 9],
    ...     "X3": [7, 4, 2, 5, 1],
    ...     "X4": [1, 2, 3, 4, 5],
    ... })
    >>> result_df = corr_matrix_linear(X, method="pearson")
    >>> # Round coefficients for doctests reproducibility
    >>> np.round(result_df, 4)
            X1      X2      X3      X4
    X1  1.0000 -0.5210 -0.0367  0.2080
    X2 -0.5210  1.0000 -0.8136  0.6092
    X3 -0.0367 -0.8136  1.0000 -0.7285
    X4  0.2080  0.6092 -0.7285  1.0000

    >>> # Spearman method usage
    >>> result_df = corr_matrix_linear(X, method="spearman")
    >>> np.round(result_df, 4)
            X1      X2      X3      X4
    X1  1.0000 -0.4617  0.1026  0.2052
    X2 -0.4617  1.0000 -0.9000  0.4000
    X3  0.1026 -0.9000  1.0000 -0.7000
    X4  0.2052  0.4000 -0.7000  1.0000
    """
    validate_string_flag(
        method,
        {"pearson", "spearman"},
        err_msg=_errors["unsupported_method_f"].format(method, {"pearson", "spearman"}),
    )
    df = convert_dataframe(dataset)
    validate_array_not_contains_nan(df, _errors["array_contains_nans"])
    numeric_features = df.select_dtypes("number")
    matrix = numeric_features.corr(method=method).fillna(0.0)
    return matrix


def corr_matrix_cramer_v(
    dataset: Sequence[Sequence], bias_correction: bool = True
) -> pd.DataFrame:
    """
    Compute the Cramér's V dependency matrix for categorical variables.

    Useful for exploratory analysis of datasets with multiple categorical
    variables, providing a pairwise overview of their associations. Bias correction
    option is available.

    Parameters
    ----------
    dataset : Sequence of sequences or pandas.DataFrame
        Input dataset containing categorical variables. If a non-DataFrame
        sequence is passed, it will be converted into a DataFrame internally.
    bias_correction : bool, default=True
        Whether to apply bias correction in the calculation of Cramér's V.

    Returns
    -------
    pandas.DataFrame
        A square symmetric correlation matrix where each entry (i, j)
        represents the Cramér's V correlation between columns i and j.
        The diagonal entries are equal to 1.

    Raises
    ------
    ValueError
        If 'dataset' contains NaN values.

    See Also
    --------
    explorica.interactions.correlation_metrics.cramer_v
        The underlying computation function.

    Notes
    -----
    - Cramér's V is a measure of association between two nominal
      (categorical) variables, ranging from 0 (no association) to 1
      (perfect association).
    - This implementation ensures the matrix is symmetric and always has
      ones on the diagonal.
    - The underlying `cramer_v` function may currently produce biased
      results in some cases due to known issues with bias correction.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix_cramer_v
    >>> # Simple usage
    >>> groups_table = pd.DataFrame({
    ...     "Group_A": ["A", "A", "A", "B", "B", "B"],
    ...     "Group_B": [1, 2, 3, 1, 2, 3],
    ...     "Group_C": ["C", "C", "C", "D", "D", "D"],
    ... })
    >>> result_df = corr_matrix_cramer_v(groups_table, bias_correction=False)
    >>> # Round coefficients for doctests reproducibility
    >>> np.round(result_df, 4)
             Group_A  Group_B  Group_C
    Group_A      1.0      0.0      1.0
    Group_B      0.0      1.0      0.0
    Group_C      1.0      0.0      1.0
    """
    df = convert_dataframe(dataset)
    validate_array_not_contains_nan(df, _errors["array_contains_nans"])
    validate_unique_column_names(
        df,
        err_msg=_errors["duplicate_column_names_f"].format("dataset"),
    )
    cols = df.columns
    n = len(cols)
    matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    for i, col1 in enumerate(cols):
        for j in range(i):
            col2 = cols[j]
            v = cramer_v(df[col1], df[col2], bias_correction=bias_correction)
            matrix.iloc[i, j] = v
            matrix.iloc[j, i] = v  # ensure symmetry
    return matrix


def corr_matrix_eta(
    dataset: Sequence[Sequence[Number]], categories: Sequence[Sequence]
) -> pd.DataFrame:
    """
    Compute a dependency matrix based on square root of eta-squared (η²).

    This function measures the strength of association between continuous
    (numeric) variables and categorical variables. The result is a matrix
    of eta coefficients, where rows correspond to numeric features and
    columns correspond to categorical features.

    Parameters
    ----------
    dataset : Sequence[Sequence[Number]] or pandas.DataFrame
        The input dataset containing numeric features.
        Will be converted to a DataFrame if not already.
    categories : Sequence[Sequence] or pandas.DataFrame
        The categorical grouping variables.
        Will be converted to a DataFrame if not already.

    Returns
    -------
    pandas.DataFrame
        A matrix of shape (n_numeric_features, n_categorical_features),
        where each entry is the eta coefficient (sqrt(η²)) between
        the corresponding numeric and categorical variable.

    Raises
    ------
    ValueError
        If 'dataset' or 'category' contains NaN values.
        If input sequences lengths mismatch

    See Also
    --------
    explorica.interactions.correlation_metrics.eta_squared
        The underlying computation function.

    Notes
    -----
    - Eta coefficient values are in [0, 1], with higher values indicating
      stronger association.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix_eta
    >>> # Simple usage
    >>> data = pd.DataFrame({
    ...     "X1": [1, 3, 5, 6, 1, 8],
    ...     "X2": [0, 0, 0, 1, 1, 1],
    ...     "X3": [7, 4, 2, 5, 1, 1],
    ...     "X4": [3, 2, 1, 3, 2, 1],
    ... })
    >>> groups_table = pd.DataFrame({
    ...     "Group_A": ["A", "A", "A", "B", "B", "B"],
    ...     "Group_B": [1, 2, 3, 1, 2, 3],
    ...     "Group_C": ["C", "C", "C", "D", "D", "D"],
    ... })
    >>> result_df = corr_matrix_eta(data, groups_table)
    >>> # Round coefficients for doctests reproducibility
    >>> np.round(result_df, 4)
        Group_A  Group_B  Group_C
    X1   0.3873   0.7246   0.3873
    X2   1.0000   0.0000   1.0000
    X3   0.4523   0.8726   0.4523
    X4   0.0000   1.0000   0.0000
    """
    err_msg = _errors["array_contains_nans_f"]
    validate_array_not_contains_nan(dataset, err_msg.format("dataset"))
    validate_array_not_contains_nan(categories, err_msg.format("categories"))
    err_msg = _errors["arrays_lens_mismatch_f"]
    validate_lengths_match(dataset, categories, err_msg.format("dataset", "categories"))
    numeric_features = convert_dataframe(dataset)
    groups = convert_dataframe(categories)
    category_cols = groups.columns
    numeric_cols = numeric_features.columns
    matrix = pd.DataFrame(
        data=np.zeros((numeric_features.shape[1], groups.shape[1])),
        index=numeric_cols,
        columns=category_cols,
    )
    for i, num_col in enumerate(numeric_cols):
        for j, cat_col in enumerate(category_cols):
            eta = np.sqrt(eta_squared(numeric_features[num_col], groups[cat_col]))
            matrix.iloc[i, j] = eta
    return matrix


def corr_vector_multiple(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute multiple correlation between `y` and predictor combinations from `x`.

    Calculates multiple correlation coefficients between the target variable `y`
    and all possible combinations of 2 or more features from `x`.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix (only numeric features should be used).
    y : pd.Series
        Target vector. The function computes correlation
        between this target and feature combinations.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:

        - 'corr_coef': multiple correlation coefficient for a given combination
        - 'feature_combination': tuple of feature names used in the combination
        - 'target': name of the target variable

    Raises
    ------
    ValueError:
        - If `x` or `y` contains NaN values.
        - If the number of samples in `x` and `y` do not match.

    Warns
    -----
    UserWarning
        If the predictors in x are found to be multicollinear
        (i.e., the determinant of their correlation matrix is zero).

    See Also
    --------
    explorica.interactions.correlation_metrics.corr_multiple
        The underlying computation function.

    Notes
    -----
    - The method renames columns as 'X1', 'X2', etc., to handling duplicate names
    - The method computes all possible
      combinationsof features of size 2 and larger.
    - This method can be computationally expensive for large datasets,
      as it evaluates all possible combinations of features.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_vector_multiple
    >>>
    >>>
    >>> X = pd.DataFrame({
    ...     "X1": [1, 3, 5, 6, 1],
    ...     "X2": [2, 3, 4, 1, 9],
    ...     "X3": [7, 4, 2, 5, 1],
    ... })
    >>> y = pd.Series([1, 2, 3, 4, 5], name="y")
    >>> result_df = corr_vector_multiple(X, y)
    >>> # Round coefficients for doctests reproducibility
    >>> result_df["corr_coef"] = np.round(result_df["corr_coef"], 4)
    >>> result_df
       corr_coef feature_combination target
    0     0.8660            (X1, X2)      y
    1     0.7507            (X1, X3)      y
    2     0.7290            (X2, X3)      y
    3     0.9803        (X1, X2, X3)      y
    """
    if x.isna().sum().sum() + y.isna().sum() != 0:
        raise ValueError(
            "The input 'x' or 'y' contains null values. "
            "Please clean or impute missing data."
        )
    if x.shape[0] != y.size:
        raise ValueError(
            f"Length of 'x' DataFrame ({x.shape[0]}) "
            f"must match length of 'y' series ({y.size})."
        )
    warned_once = False
    factors = x.copy()
    res = y.copy()
    corr_coef = []
    feature_combinations = []
    for i in range(2, factors.shape[1] + 1):
        for combination in combinations(factors.columns, i):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                corr_coef.append(corr_multiple(x=factors[list(combination)], y=res))
                feature_combinations.append(combination)
                if not warned_once:
                    for w in caught:
                        if _warns["multicollinearity"] == str(w.message):
                            warned_once = True
                            break
    corr_vector = pd.DataFrame(
        {
            "corr_coef": corr_coef,
            "feature_combination": feature_combinations,
            "target": res.name,
        }
    )
    if warned_once:
        warnings.warn(
            _warns["multicollinearity"],
            UserWarning,
        )
    return corr_vector


def corr_matrix_multiple(dataset: Sequence[Sequence[Number]]) -> pd.DataFrame:
    """
    Compute a multi-factor correlation matrix for all target features in a dataset.

    For each column in the dataset, this method treats it as a target variable and
    computes correlations between that target and all remaining features
    (predictors) using `corr_vector_multiple`. The results are concatenated into a
    single DataFrame with columns `['corr_coef', 'feature_combination', 'target']`.

    Parameters
    ----------
    dataset : Sequence[Sequence[Number]]
        Input data, where each inner sequence represents a feature/column. Can be
        a pandas DataFrame, numpy array, or nested lists.
        **Note:** The dataset must contain at least 3 columns, since one column is
        treated as the target and at least two others are required as predictors.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:

        - `corr_coef`: correlation coefficient
           for the target-predictor combination.
        - `feature_combination`: tuple or list of predictor feature names.
        - `target`: the name of the current target feature.

    Raises
    ------
    ValueError
        If the input DataFrame contains duplicate column names.
        If the input DataFrame contains NaN values.

    Warns
    -----
    UserWarning
        If a set of predictors is linearly dependent (multicollinearity detected),
        a UserWarning is issued. This warning appears only once, regardless of how
        many targets trigger it.

    See Also
    --------
    explorica.interactions.correlation_matrices.corr_vector_multiple
        The underlying computation function.
    explorica.interactions.correlation_metrics.corr_multiple
        The underlying computation function.

    Notes
    -----
    - The function handles datasets with any number of features >= 3.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix_multiple
    >>> # Simple usage
    >>> X = pd.DataFrame({
    ...     "X1": [1, 3, 5, 6, 1],
    ...     "X2": [2, 3, 4, 1, 9],
    ...     "X3": [7, 4, 2, 5, 1],
    ...     "X4": [1, 2, 3, 4, 5],
    ... })
    >>> result_df = corr_matrix_multiple(X)
    >>> # Round coefficients for doctests reproducibility
    >>> result_df["corr_coef"] = np.round(result_df["corr_coef"], 4)
    >>> result_df = result_df.sort_values(by="corr_coef", ascending=False)
    >>> result_df
        corr_coef feature_combination target
    7      0.9985        (X1, X3, X4)     X2
    11     0.9963        (X1, X2, X4)     X3
    3      0.9958        (X2, X3, X4)     X1
    4      0.9828            (X1, X3)     X2
    15     0.9803        (X1, X2, X3)     X4
    8      0.9763            (X1, X2)     X3
    0      0.9482            (X2, X3)     X1
    5      0.8998            (X1, X4)     X2
    12     0.8660            (X1, X2)     X4
    10     0.8650            (X2, X4)     X3
    1      0.8428            (X2, X4)     X1
    6      0.8140            (X3, X4)     X2
    13     0.7507            (X1, X3)     X4
    9      0.7379            (X1, X4)     X3
    14     0.7290            (X2, X3)     X4
    2      0.2671            (X3, X4)     X1
    """
    warned_once = False
    matrix = pd.DataFrame()
    features = convert_dataframe(dataset)
    features = features.select_dtypes("number")
    validate_array_not_contains_nan(
        dataset,
        "The input dataset contains null values."
        "Please clean or impute missing data.",
    )
    validate_unique_column_names(
        features, "Duplicate column names found in input dataset"
    )
    for target in features.columns:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            factors = features.copy()
            del factors[target]
            y = features[target]
            matrix = pd.concat(
                [matrix, corr_vector_multiple(factors, y)],
                ignore_index=True,
            )
            if not warned_once:
                for w in caught:
                    if _warns["multicollinearity"] == str(w.message):
                        warned_once = True
                        break
    if warned_once:
        print(dataset)
        warnings.warn(
            _warns["multicollinearity"],
            UserWarning,
        )
        print(dataset)
    return matrix


def corr_matrix_corr_index(
    dataset: Sequence[Sequence[Number]], method: str = "linear"
) -> pd.DataFrame:
    r"""
    Compute a correlation index matrix for all features in a dataset.

    This method computes a pairwise correlation index (√R²) between features using
    non-linear regression-based methods. The supported methods are: 'linear',
    'exp', 'binomial', 'ln', 'hyperbolic', and 'power'.

    Parameters
    ----------
    dataset : Sequence[Sequence[Number]]
        Input data, where each inner sequence represents a feature/column. Can be a
        pandas DataFrame, numpy array, dict or nested lists. Must not contain NaN
        values, and column names must be unique.
    method : str, default='linear'
        Method used to compute the correlation index. Supported options are:
        {'exp', 'binomial', 'ln', 'hyperbolic', 'power', 'linear'}.

    Returns
    -------
    pd.DataFrame
        DataFrame of size (n_features x n_features) containing the correlation
        index (:math:`\sqrt{R^2}`) values for each pair of features.

    Raises
    ------
    ValueError
        If the dataset contains NaN values.
        If column names are duplicated.
        If the selected method is not supported.

    See Also
    --------
    explorica.interactions.correlation_metrics.corr_index
        The underlying computation function.

    Notes
    -----
    - Any invalid pair according to these method constraints
      will have NaN as a result.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from explorica.interactions.correlation_matrices import corr_matrix_corr_index
    >>>
    >>> data = pd.DataFrame({
    ...     "X1": [1, 2, 3, 4, 5, 6],
    ...     "X2": [2, 4, 6, 8, 10, 12],
    ...     "X3": [1, 4, 9, 16, 25, 36],
    ...     "X4": [3, 2, 1, 3, 2, 1],
    ... })
    >>> result_df = corr_matrix_corr_index(data, method="binomial")
    >>> # Round coefficients for doctests reproducibility
    >>> np.round(result_df, 4)
            X1      X2      X3      X4
    X1  1.0000  1.0000  1.0000  0.4781
    X2  1.0000  1.0000  1.0000  0.4781
    X3  0.9969  0.9969  1.0000  0.4964
    X4  0.4781  0.4781  0.4696  1.0000
    """
    supported_methods = {"linear", "exp", "binomial", "ln", "hyperbolic", "power"}
    df = convert_dataframe(dataset)
    validate_string_flag(
        method,
        supported_methods,
        err_msg=_errors["unsupported_method_f"].format(method, supported_methods),
    )
    validate_array_not_contains_nan(
        dataset,
        "The input dataset contains null values."
        "Please clean or impute missing data.",
    )
    validate_unique_column_names(df, "Duplicate column names found in input dataset")
    features = df.columns
    n = len(features)
    matrix = pd.DataFrame(np.ones((n, n)), index=features, columns=features)
    for i, col_x in enumerate(features):
        for j, col_y in enumerate(features):
            try:
                corr_coef = corr_index(df[col_x], df[col_y], method=method)
            except ValueError as e:
                if str(e) in (
                    "Method 'ln' requires strictly positive x-values.",
                    "Method 'hyperbolic' can't handle zero x-values.",
                    "Method 'power' can't handle zero x-values.",
                ):
                    corr_coef = np.nan
                else:
                    raise e
            matrix.iloc[i, j] = corr_coef
    return matrix
