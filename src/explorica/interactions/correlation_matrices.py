"""
Module provides tools for constructing various types of
correlation and dependence matrices for both numerical and categorical features
in a dataset. This module is intended to be used via the public facade
`InteractionAnalyzer` but can also be used directly for advanced analyses.

The module supports linear correlations (Pearson, Spearman), categorical
associations (Cramér's V, η²), multiple-factor correlations, and correlation
indices for non-linear dependencies.

Main class
----------
CorrelationMatrices
    A collection of static methods for building correlation and dependence
    matrices from numerical and categorical datasets.
"""

import warnings
from itertools import combinations
from numbers import Number
from typing import Sequence

import numpy as np
import pandas as pd

from explorica._utils import ConvertUtils as cutils
from explorica._utils import ValidationUtils as vutils
from explorica._utils import read_messages
from explorica.interactions.correlation_metrics import CorrelationMetrics as cm


class CorrelationMatrices:
    """
    A collection of static methods for building correlation and dependence
    matrices from numerical and categorical datasets.

    All methods operate on input datasets provided as sequences of sequences
    (nested lists, NumPy arrays, or pandas DataFrames) and return pandas
    DataFrames with the results.

    Methods
    -------
    corr_matrix(dataset, method="pearson", groups=None)
        Aggregator method. Constructs a correlation/dependence matrix
        using one of the supported methods:
        {"pearson", "spearman", "cramer_v", "eta", "multiple",
         "exp", "binomial", "ln", "hyperbolic", "power"}.

    corr_matrix_linear(dataset, method="pearson")
        Builds a correlation matrix using either Pearson or Spearman
        correlation for numerical features.

    corr_matrix_cramer_v(dataset, bias_correction=True)
        Builds a dependence matrix using Cramér’s V for categorical
        features.

    corr_matrix_eta(dataset, categories)
        Builds a dependence matrix using η² between numerical features
        and categorical groupings.

    corr_vector_multiple(x, y)
        Iterates over all possible combinations of features in x with a
        target y, computing multiple correlation coefficients.

    corr_matrix_multiple(dataset)
        Constructs a matrix of multiple correlations by iterating over
        potential target variables. Internally reuses
        ``corr_vector_multiple``.

    corr_matrix_corr_index(dataset, method="linear")
        Builds a matrix of correlation indices as a measure of
        non-linear dependence.

    Notes
    -----
    - All methods are static and can be called directly via the class without
      instantiation.
    - NaN values or duplicate column names will raise a ValueError.
    """

    _errors = read_messages()["errors"]
    _warns = read_messages()["warns"]

    @staticmethod
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
        ------
        UserWarning
            If Multicollinearity detected in the dataset (for 'multiple'),
            some features are linearly dependent.

        Notes
        ------
        - Pearson, Spearman, and other numerical correlation methods internally select
          only features of numeric type (`Number`) from the provided DataFrame.
          Non-numeric columns (e.g., categorical strings or object types) are ignored.
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
        dataset_df = cutils.convert_dataframe(dataset)
        groups_df = cutils.convert_dataframe(groups)
        vutils.validate_unique_column_names(
            dataset_df, "Duplicate column names found in 'dataset' DataFrame"
        )
        vutils.validate_unique_column_names(
            groups_df, "Duplicate column names found in 'groups' DataFrame"
        )

        vutils.validate_array_not_contains_nan(
            dataset_df,
            err_msg=CorrelationMatrices._errors["array_contains_nans_f"].format(
                "dataset"
            ),
        )

        vutils.validate_string_flag(
            method,
            supported_methods,
            err_msg=CorrelationMatrices._errors["usupported_method_f"].format(
                method, supported_methods
            ),
        )

        if groups is None and method == "eta":
            raise ValueError("'groups' must be provided when using 'eta' method.")

        if groups is not None:
            vutils.validate_array_not_contains_nan(
                groups_df,
                err_msg=CorrelationMatrices._errors["array_contains_nans_f"].format(
                    "groups"
                ),
            )
            vutils.validate_lenghts_match(
                dataset_df,
                groups_df,
                f"Length of 'groups' ({groups_df.shape[0]}) "
                f"must match length of 'dataset' ({dataset_df.shape[0]}).",
            )

        numeric_df = dataset_df.select_dtypes("number")
        matrix = None

        if method in {"pearson", "spearman"}:
            matrix = CorrelationMatrices.corr_matrix_linear(numeric_df, method)

        if method == "cramer_v":
            matrix = CorrelationMatrices.corr_matrix_cramer_v(dataset_df)

        if method == "eta":
            matrix = CorrelationMatrices.corr_matrix_eta(numeric_df, groups_df)

        if method == "multiple":
            matrix = CorrelationMatrices.corr_matrix_multiple(numeric_df)

        if method in {"exp", "binomial", "ln", "hyperbolic", "power"}:
            matrix = CorrelationMatrices.corr_matrix_corr_index(numeric_df, method)

        return matrix

    @staticmethod
    def corr_matrix_linear(
        dataset: Sequence[Sequence[Number]], method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute a correlation matrix for numeric
        features using Pearson or Spearman correlation.

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
        """
        vutils.validate_string_flag(
            method,
            {"pearson", "spearman"},
            err_msg=CorrelationMatrices._errors["usupported_method_f"].format(
                method, {"pearson", "spearman"}
            ),
        )
        df = cutils.convert_dataframe(dataset)
        vutils.validate_array_not_contains_nan(
            df, CorrelationMatrices._errors["array_contains_nans"]
        )
        numeric_features = df.select_dtypes("number")
        matrix = numeric_features.corr(method=method).fillna(0.0)
        return matrix

    @staticmethod
    def corr_matrix_cramer_v(
        dataset: Sequence[Sequence], bias_correction: bool = True
    ) -> pd.DataFrame:
        """
        Compute the Cramér's V correlation matrix for categorical variables.

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

        Notes
        -----
        - Cramér's V is a measure of association between two nominal
        (categorical) variables, ranging from 0 (no association) to 1
        (perfect association).
        - This implementation ensures the matrix is symmetric and always has
        ones on the diagonal.
        - The underlying `cramer_v` function may currently produce biased
        results in some cases due to known issues with bias correction.
        """
        df = cutils.convert_dataframe(dataset)
        vutils.validate_array_not_contains_nan(
            df, CorrelationMatrices._errors["array_contains_nans"]
        )
        vutils.validate_unique_column_names(
            df,
            err_msg=CorrelationMatrices._errors["duplicate_column_names_f"].format(
                "dataset"
            ),
        )
        cols = df.columns
        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

        for i, col1 in enumerate(cols):
            for j in range(i):
                col2 = cols[j]
                v = cm.cramer_v(df[col1], df[col2], bias_correction=bias_correction)
                matrix.iloc[i, j] = v
                matrix.iloc[j, i] = v  # ensure symmetry
        return matrix

    @staticmethod
    def corr_matrix_eta(
        dataset: Sequence[Sequence[Number]], categories: Sequence[Sequence]
    ) -> pd.DataFrame:
        """
        Compute a correlation matrix between numerical features and categorical features
        using the square root of eta-squared (η²).

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

        Notes
        -----
        - Eta coefficient values are in [0, 1], with higher values indicating
        stronger association.
        """
        err_msg = CorrelationMatrices._errors["array_contains_nans_f"]
        vutils.validate_array_not_contains_nan(dataset, err_msg.format("dataset"))
        vutils.validate_array_not_contains_nan(categories, err_msg.format("categories"))

        err_msg = CorrelationMatrices._errors["arrays_lens_mismatch_f"]
        vutils.validate_lenghts_match(
            dataset, categories, err_msg.format("dataset", "categories")
        )

        numeric_features = cutils.convert_dataframe(dataset)
        groups = cutils.convert_dataframe(categories)

        category_cols = groups.columns
        numeric_cols = numeric_features.columns

        matrix = pd.DataFrame(
            data=np.zeros((numeric_features.shape[1], groups.shape[1])),
            index=numeric_cols,
            columns=category_cols,
        )
        for i, num_col in enumerate(numeric_cols):
            for j, cat_col in enumerate(category_cols):
                eta = np.sqrt(
                    cm.eta_squared(numeric_features[num_col], groups[cat_col])
                )
                matrix.iloc[i, j] = eta
        return matrix

    @staticmethod
    def corr_vector_multiple(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculates multiple correlation coefficients between the target variable `y`
        and all possible combinations of 2 or more features from `x`.

        Parameters:
        -----------
        x : pd.DataFrame
            Feature matrix (only numeric features should be used).
        y : pd.Series
            Target vector. The function computes correlation
            between this target and feature combinations.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with columns:
            - 'corr_coef': multiple correlation coefficient for a given combination
            - 'feature_combination': tuple of feature names used in the combination
            - 'target': name of the target variable

        Raises:
        -------
        ValueError:
            - If `x` or `y` contains NaN values.
            - If the number of samples in `x` and `y` do not match.

        Warns
        ------
        UserWarning
            If the predictors in x are found to be multicollinear
            (i.e., the determinant of their correlation matrix is zero).

        Notes:
        ------
        - The method renames columns as 'X1', 'X2', etc., to handling duplicate names
        - The method computes all possible
          combinationsof features of size 2 and larger.
        - This method can be computationally expensive for large datasets,
          as it evaluates all possible combinations of features.
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
                    corr_coef.append(
                        cm.corr_multiple(x=factors[list(combination)], y=res)
                    )
                    feature_combinations.append(combination)
                    if not warned_once:
                        for w in caught:
                            if CorrelationMatrices._warns["multicollinearity"] == str(
                                w.message
                            ):
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
                CorrelationMatrices._warns["multicollinearity"],
                UserWarning,
            )
        return corr_vector

    @staticmethod
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
        ------
        UserWarning
            If a set of predictors is linearly dependent (multicollinearity detected),
            a UserWarning is issued. This warning appears only once, regardless of how
            many targets trigger it.

        Notes
        -----
        - The function handles datasets with any number of features >= 3.
        """
        warned_once = False
        matrix = pd.DataFrame()
        features = cutils.convert_dataframe(dataset)
        features = features.select_dtypes("number")
        vutils.validate_array_not_contains_nan(
            dataset,
            "The input dataset contains null values."
            "Please clean or impute missing data.",
        )
        vutils.validate_unique_column_names(
            features, "Duplicate column names found in input dataset"
        )
        for target in features.columns:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                factors = features.copy()
                del factors[target]
                y = features[target]
                matrix = pd.concat(
                    [matrix, CorrelationMatrices.corr_vector_multiple(factors, y)],
                    ignore_index=True,
                )
                if not warned_once:
                    for w in caught:
                        if CorrelationMatrices._warns["multicollinearity"] == str(
                            w.message
                        ):
                            warned_once = True
                            break
        if warned_once:
            print(dataset)
            warnings.warn(
                CorrelationMatrices._warns["multicollinearity"],
                UserWarning,
            )
            print(dataset)
        return matrix

    @staticmethod
    def corr_matrix_corr_index(
        dataset: Sequence[Sequence[Number]], method: str = "linear"
    ) -> pd.DataFrame:
        """
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
            index (√R²) values for each pair of features.

        Raises
        ------
        ValueError
            If the dataset contains NaN values.
            If column names are duplicated.
            If the selected method is not supported.

        Notes
        -----
        - Any invalid pair according to these method constraints
          will have NaN as a result.
        """
        supported_methods = {"linear", "exp", "binomial", "ln", "hyperbolic", "power"}
        df = cutils.convert_dataframe(dataset)
        vutils.validate_string_flag(
            method,
            supported_methods,
            err_msg=CorrelationMatrices._errors["usupported_method_f"].format(
                method, supported_methods
            ),
        )
        vutils.validate_array_not_contains_nan(
            dataset,
            "The input dataset contains null values."
            "Please clean or impute missing data.",
        )
        vutils.validate_unique_column_names(
            df, "Duplicate column names found in input dataset"
        )
        features = df.columns
        n = len(features)
        matrix = pd.DataFrame(np.ones((n, n)), index=features, columns=features)

        for i, col_x in enumerate(features):
            for j, col_y in enumerate(features):
                try:
                    corr_coef = cm.corr_index(df[col_x], df[col_y], method=method)
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
