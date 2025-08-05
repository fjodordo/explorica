"""
interaction_analyzer.py

Module for analyzing interactions between variables in a dataset,
including correlation matrices for numeric features and association
measures for categorical features (such as Cramér's V).
"""
import warnings
from itertools import combinations
from typing import Callable, Optional
from numbers import Number

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.optimize import curve_fit

class InteractionAnalyzer:
    """
    A utility class for analyzing statistical interactions between variables.

    Provides tools for computing correlation matrices using various methods,
    including Pearson, Spearman, and Cramér's V. Designed to support exploratory
    data analysis (EDA) by revealing linear or categorical associations between features.
    """

    def __init__(self):
        pass

    @staticmethod
    def corr_matrix(dataset: pd.DataFrame,
                    method: str = "pearson",
                    groups: pd.DataFrame = None
                    ) -> pd.DataFrame:
        """
        Compute a correlation or association matrix using the specified method.

        This function supports both classical correlation coefficients and a set of nonlinear
        correlation indices based on specific functional dependencies. For nonlinear methods,
        the correlation is evaluated by fitting transformations (e.g., exponential, logarithmic)
        and computing the strength of the relationship between features accordingly.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame with numeric or categorical features.
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
                           as a target and all remaining numeric features as predictors.
            - 'exp' : Nonlinear correlation index assuming exponential dependence.
            - 'binomial' : Nonlinear correlation index assuming binomial dependence.
            - 'ln' : Nonlinear correlation index assuming logarithmic dependence.
            - 'hyperbolic' : Nonlinear correlation index assuming hyperbolic dependence.
            - 'power' : Nonlinear correlation index assuming power-law dependence.

        groups : pd.DataFrame, optional
            DataFrame of categorical grouping variables required for the 'eta' method.
            Must have the same number of rows as `dataset`.

        Returns
        -------
        pd.DataFrame
            - For 'pearson' and 'spearman': 
              symmetric correlation matrix of shape (n_numeric_features, n_numeric_features).
            - For 'cramer_v':
              symmetric matrix of shape (n_features, n_features),
              representing categorical associations.
            - For 'eta':
              asymmetric matrix of shape (n_numeric_features, n_grouping_features),
              showing strength of association between numeric and categorical variables.
            - For nonlinear methods ('exp', 'binomial', 'ln', 'hyperbolic', 'power'):
              asymmetric matrix of shape (n_numeric_features, n_numeric_features),
              showing the correlation index based on the specified nonlinear model.

        Raises
        ------
        ValueError
            If the specified method is not supported.
            If `groups` is required (for 'eta') but not provided or mismatched in length.
        
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
        # Сюда тоже следует добавить валидацию на одинаковые колонки
        supported_methods = {"pearson", "spearman", "cramer_v",
                             "eta", "multiple", "exp", "binomial", "ln",
                             "hyperbolic", "power"}
        if method not in supported_methods:
            raise ValueError(f"Unsupported method '{method}'. Choose from: {supported_methods}")

        if groups is None and method == "eta":
            raise ValueError("'groups' must be provided when using 'eta' method.")

        if groups is not None and groups.shape[0] != dataset.shape[0]:
            raise ValueError(f"Length of 'groups' ({groups.shape[0]}) "
                             f"must match length of 'dataset' ({dataset.shape[0]}).")

        numeric_features = dataset.select_dtypes("number")
        if method in {"pearson", "spearman"}:
            matrix = numeric_features.corr(method=method)

        if method == "cramer_v":
            cols = dataset.columns
            n = len(cols)
            matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

            for i, col1 in enumerate(cols):
                for j in range(i):
                    col2 = cols[j]
                    v = InteractionAnalyzer.cramer_v(dataset[col1], dataset[col2])
                    matrix.iloc[i, j] = v
                    matrix.iloc[j, i] = v  # ensure symmetry

        if method == "eta":
            category_cols = groups.columns
            numeric_cols = numeric_features.columns

            matrix = pd.DataFrame(
                data=np.zeros((numeric_features.shape[1], groups.shape[1])),
                index=numeric_cols,
                columns=category_cols
            )
            for i, num_col in enumerate(numeric_cols):
                for j, cat_col in enumerate(category_cols):
                    eta = np.sqrt(InteractionAnalyzer.eta_squared(
                        numeric_features[num_col], groups[cat_col]))
                    matrix.iloc[i, j] = eta

        if method == "multiple":
            warned_once = False
            matrix = pd.DataFrame()
            features = numeric_features.copy()
            for target in features.columns:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    factors = features.copy()
                    del factors[target]
                    y = features[target]
                    matrix = pd.concat([matrix,
                                        InteractionAnalyzer.corr_vector_multiple(factors, y)],
                                        ignore_index=True)
                    if not warned_once:
                        for w in caught:
                            if ("Matrix of predictors is linearly "
                                "dependent (multicollinearity detected). " 
                                "Interpret results with caution.") == str(w.message):
                                warned_once = True
                                break
            if warned_once:
                warnings.warn("Matrix of predictors is linearly "
                              "dependent (multicollinearity detected). " 
                              "Interpret results with caution.", UserWarning)


        if method in {"exp", "binomial", "ln", "hyperbolic", "power"}:
            features = numeric_features.columns
            n = len(features)
            matrix = pd.DataFrame(np.ones((n, n)), index=features, columns=features)

            for i, col_x in enumerate(features):
                for j, col_y in enumerate(features):
                    try:
                        corr_coef = InteractionAnalyzer.corr_index(numeric_features[col_x],
                                                    numeric_features[col_y], method=method)
                    except ValueError as e:
                        if str(e) in ("Method 'ln' requires strictly positive x-values.",
                                 "Method 'hyperbolic' can't handle zero x-values.",
                                 "Method 'power' can't handle zero x-values."):
                            corr_coef = np.nan
                        else:
                            raise e
                    matrix.iloc[i, j] = corr_coef

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
            Columns are automatically renamed for consistency.
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
        - The method computes all possible combinations of features of size 2 and larger.
        - This method can be computationally expensive for large datasets, 
          as it evaluates all possible combinations of features.
        """
        if x.isna().sum().sum() + y.isna().sum() != 0:
            raise ValueError("The input 'x' or 'y' contains null values. "
                             "Please clean or impute missing data.")
        if x.shape[0] != y.size:
            raise ValueError(f"Length of 'x' DataFrame ({x.shape[0]}) "
                             f"must match length of 'y' series ({y.size}).")

        warned_once = False
        factors = x.copy()
        res = y.copy()
        corr_coef = []
        feature_combinations = []

        for i in range(2, factors.shape[1] + 1):
            for combination in combinations(factors.columns, i):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    corr_coef.append(InteractionAnalyzer.corr_multiple(
                                     x=factors[list(combination)], y=res))
                    feature_combinations.append(combination)
                    if not warned_once:
                        for w in caught:
                            if ("Matrix of predictors is linearly "
                                "dependent (multicollinearity detected). " 
                                "Interpret results with caution.") == str(w.message):
                                warned_once = True
                                break

        corr_vector = pd.DataFrame({"corr_coef": corr_coef, "feature_combination":
                                    feature_combinations, "target": res.name})
        if warned_once:
            warnings.warn("Matrix of predictors is linearly "
                          "dependent (multicollinearity detected). " 
                          "Interpret results with caution.", UserWarning)
        return corr_vector

    @staticmethod
    def high_corr_pairs(numeric_features: pd.DataFrame=None,
                        category_features: pd.DataFrame=None,
                        y: str=None,
                        nonlinear_included: bool=False,
                        multiple_included: bool=False,
                        threshold: float = 0.7
                        ) -> pd.DataFrame | None:
        """
        Finds and returns all significant pairs of correlated features from the input datasets.

        This method evaluates feature-to-feature relationships using a set of correlation measures,
        including linear (Pearson, Spearman), non-linear (e.g. exponential, binomial, power-law),
        categorical (Cramér’s V), and hybrid (η²). Users can optionally enable non-linear
        and multiple-correlation modes.

        Parameters
        ----------
        numeric_features : pd.DataFrame, optional
            A DataFrame of numerical features.
            Required for linear, η², non-linear, and multiple correlation.
        category_features : pd.DataFrame, optional
            A DataFrame of categorical features. Required for Cramér’s V and η² computations.
        y : str, optional
            Target feature name to compute correlations with.
            If None, all pairwise comparisons are evaluated.
        nonlinear_included : bool, default=False
            Whether to include non-linear correlation measures for numeric features.
        multiple_included : bool, default=False
            Whether to include multiple correlation analysis (for numeric features only).
        threshold : float, default=0.7
            Minimum absolute value of correlation to consider a pair as significantly dependent.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame with columns ['X', 'Y', 'coef', 'method'], listing feature pairs
            whose correlation (in absolute value) exceeds the threshold.
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
        - Non-linear methods (enabled via nonlinear_included): exp, binomial, ln, hyperbolic, power
        - Categorical methods: Cramér’s V, η² (eta)
        - The method skips self-comparisons.
        - Targeted correlation (`y`) will produce only pairs involving the specified target.
        """

        # checking at least 1 not None DataFrame
        if numeric_features is None and category_features is None:
            raise ValueError("At least one of the feature DataFrames ('numeric_features' or" \
            " 'category_features') must be provided.")

        # checking for lengths match
        if numeric_features is not None and category_features is not None:
            if numeric_features.shape[0] != category_features.shape[0]:
                raise ValueError(
                    f"Length of 'numeric_features' ({numeric_features.shape[0]}) "
                    f"must match length of 'category_features'({category_features.shape[0]}).")

        # Nan's absence check
        if (numeric_features is not None and numeric_features.isna().values.any()) or \
           (category_features is not None and category_features.isna().values.any()):
            raise ValueError(
                "The input 'numeric_features' or 'category_features' contains null values. "
                "Please clean or impute missing data.")

        # checking column names of input DataFrame for duplicates
        def get_columns(df):
            return list(df.columns) if df is not None else []

        numeric_cols = get_columns(numeric_features)
        category_cols = get_columns(category_features)
        for name, cols in [('numeric_features', numeric_cols),
                       ('category_features', category_cols)]:
            if len(cols) != len(set(cols)):
                raise ValueError(f"Duplicate column names found in '{name}' DataFrame.")

        # checking if 'y' feature is present in at least one input DataFrame
        if y is not None:
            if not any(
                y in df.columns
                for df in (numeric_features, category_features)
            if df is not None
            ):
                raise ValueError(f"Feature y = '{y}' not found in any input DataFrame")

        numeric_methods = {"pearson", "spearman", "exp", "binomial", "ln",
                             "hyperbolic", "power"}

        # form a queue of execution
        execution = []
        # any numeric correlation methods works with at least 2 numeric (or 'both') features
        if y is None or y in numeric_cols:
            if len(numeric_cols) >= 2:
                execution.extend(["pearson", "spearman"])
                if nonlinear_included:
                    execution.extend(["exp", "binomial", "ln",
                                    "hyperbolic", "power"])
                # multiple correlation works with at least 3 features
                if multiple_included:
                    if len(numeric_cols) >= 3:
                        execution.append("multiple")

        # eta correlation works with at least 1 numeric and 1 category features
        # or with at least 1 'both' feature
        if y is None or y in category_cols:
            if len(category_cols) >= 2:
                execution.append("cramer_v")
            if category_features is not None and numeric_features is not None:
                # special case validation: feature is compared only with itself
                if not (len(category_cols) == 1 and
                        len(numeric_cols) == 1 and
                        category_cols == numeric_cols):
                    execution.append("eta")

        def extract_corr_pairs(df: pd.DataFrame,
                               y: str=None,
                               method: str=None,
                               drop_diagonal: bool=True
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
                col_pairs = pd.DataFrame({"X": col_indexes,
                                          "Y": col_column,
                                          "coef": col_coefs,
                                          "method": col_method})
                pairs = pd.concat([pairs, col_pairs], ignore_index=True)
            return pairs

        corr_pairs = pd.DataFrame()
        # execution cycle
        for method in execution:
            if method in numeric_methods:
                corr_matrix = InteractionAnalyzer.corr_matrix(
                        numeric_features, method=method)
                pairs_by_method = extract_corr_pairs(corr_matrix, y, method)
                pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
                corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
            elif method == "multiple":
                # using the "correlation vector" will be less expensive than using the corr matrix
                if y is not None:
                    factors = numeric_features.copy()
                    target = factors[y]
                    del factors[y]
                    corr_matrix = InteractionAnalyzer.corr_vector_multiple(
                        factors, target)
                    columns = pd.Series([y])
                else:
                    corr_matrix = InteractionAnalyzer.corr_matrix(
                        numeric_features, method="multiple"
                    )
                    columns = pd.Series(corr_matrix["target"]).drop_duplicates()
                corr_matrix = corr_matrix[corr_matrix["target"].isin(columns)]
                corr_matrix = corr_matrix[corr_matrix["corr_coef"] >= threshold]
                col_coefs = corr_matrix["corr_coef"]
                col_method = np.repeat(method, col_coefs.size)
                col_pairs = pd.DataFrame({"X": corr_matrix["feature_combination"],
                                          "Y": corr_matrix["target"],
                                          "coef": corr_matrix["corr_coef"],
                                          "method": col_method})
                corr_pairs = pd.concat([corr_pairs, col_pairs], ignore_index=True)
            elif method == "cramer_v":
                corr_matrix = InteractionAnalyzer.corr_matrix(category_features, method=method)
                pairs_by_method = extract_corr_pairs(corr_matrix, y, method)
                pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
                corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)
            elif method == "eta":
                corr_matrix = InteractionAnalyzer.corr_matrix(numeric_features,
                                                              groups=category_features,
                                                              method=method)
                both_cols = set(category_cols) & set(numeric_cols)
                pairs_by_method = extract_corr_pairs(corr_matrix, y, method, drop_diagonal=False)
                pairs_by_method = pairs_by_method[abs(pairs_by_method["coef"]) >= threshold]
                # conditions under which feature's comparisons with self occurs
                pairs_by_method = pairs_by_method[
                    ~((pairs_by_method["Y"] == pairs_by_method["X"]) & (
                        pairs_by_method["X"].isin(both_cols)))]
                corr_pairs = pd.concat([corr_pairs, pairs_by_method], ignore_index=True)

        if corr_pairs.shape[0] == 0:
            return None

        return corr_pairs.sort_values(by="coef", key=abs, ascending=False).reset_index(drop=True)

    @staticmethod
    def cramer_v(
        x: pd.Series,
        y: pd.Series,
        bias_correction: bool = True,
        yates_correction: bool = False
    ) -> float:
        """
        Calculates Cramér's V statistic for measuring the association between two categorical
        variables.

        Parameters
        ----------
        x : pd.Series
            First categorical variable.
        y : pd.Series
            Second categorical variable.
        bias_correction : bool, optional, default=True
            Whether to apply bias correction (recommended for small samples).
        yates_correction : bool, optional, default=False
            Whether to apply Yates' correction for continuity (only applies to 2x2 tables; usually
            set to False when using Cramér's V).

        Returns
        -------
        float
            Cramér's V value, ranging from 0 (no association) to 1 (perfect association).
            Returns 0 if the statistic is undefined (e.g., due to zero denominator).
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix, correction=yates_correction)[0]
        n = confusion_matrix.to_numpy().sum()
        r, k = confusion_matrix.shape
        min_dim = min(r - 1, k - 1)

        if min_dim == 0:
            return 0.0

        if bias_correction:
            correction = ((r - 1) * (k - 1)) / n
            return np.sqrt((chi2 / n - correction) / min_dim)
        else:
            return np.sqrt(chi2 / (n * min(k - 1, r - 1)))

    @staticmethod
    def eta_squared(values: pd.Series,
                    category: pd.Series
    ) -> float:
        """
        Calculate the eta-squared (η²) statistic for categorical and numeric variables.

        η² (eta squared) is a measure of effect size used to quantify the proportion of variance 
        in a numerical variable that can be attributed to differences between categories 
        of a categorical variable.

        Parameters
        ----------
        values : pd.Series
            A numerical pandas Series representing the dependent (response) variable.
        category : pd.Series
            A categorical pandas Series representing the independent (grouping) variable.

        Returns
        -------
        float
            Eta-squared statistic in the range [0, 1], where:
            - 0 means no association between variables,
            - 1 means perfect association (all variance explained by groups).

        Notes
        -----
        If the total variance of `values` is zero, the function returns 0. 
        NaN values should be handled before calling this function.
        """
        df = pd.DataFrame({"category": category, "values": values})
        mean_by_group = df.groupby("category")["values"].mean()
        mean = df["values"].mean()
        n_by_group = df.groupby("category")["values"].count()
        n = df["values"].size

        bg_disperison = np.sum(((mean_by_group - mean)**2) * n_by_group) / n
        dispersion = ((df["values"] - mean)**2).sum()/n

        # zero dispersion in this case indicates a zero coefficient of determination
        eta_sq = bg_disperison/dispersion if dispersion != 0 else 0
        return eta_sq

    @staticmethod
    def corr_index(x: pd.Series,
                   y: pd.Series,
                   method: str="linear",
                   normalization_lower_bound: float=1e-13,
                   normalization_upper_bound: float=1,
                   custom_function: Optional[Callable[[Number], Number]] = None) -> Number:
        """
        Calculates a nonlinear correlation index between two series `x` and `y`,
        based on the proportion of variance explained by the fitted function.

        The index is computed as:

            R_I = sqrt(1 - SSE / SST),

        where SSE is the sum of squared errors between the true `y` and predicted `y(x)`,
        and SST is the total sum of squares `y` with respect to its mean.

        This method is designed to handle non-linear dependencies 
        and is robust to monotonic transformations.
        In degenerate cases (e.g., constant `x` or `y`), a correlation index of 0 is returned.
        
        **Important**: If the model fit yields SSE > SST
        (i.e., performs worse than a mean-based predictor),
        the index is treated as 0. This reflects the model’s
        failure to explain any meaningful variance.


        Supported methods
        ----------
        - 'linear':       y = a * x + b
        - 'binomial':     y = a * x² + b * x + c
        - 'exp':          y = b * exp(a * x)
        - 'ln':           y = a * ln(x) + b
        - 'hyperbolic':   y = a / x + b
        - 'power':        y = a * xᵇ
        - 'custom':       y = your function

        Parameters
        ----------
        x : pd.Series
            The explanatory variable.
        y : pd.Series
            The response variable.
        method : str, default='linear'
            Regression model to use. See supported methods above.
        normalization_lower_bound : float, optional
            Lower bound for normalization of `x`, by default 1e-13.
        normalization_upper_bound : float, optional
            Upper bound for normalization of `x`, by default 1.0.
        custom_function : Callable, optional
            A user-defined function that takes a Number values and returns Number predictions. 
            Required when `method='custom'`

        Returns
        -------
        float
            A nonlinear correlation index in the range (0 ≤ r ≤ 1).

        Raises
        ------
        ValueError
            If:
            - an unsupported method is passed
            - x and y are of unequal lengths
            - any NaNs are present in x or y
            - input values violate the domain of the selected function (e.g., log(0))
            - `method='custom'` is specified but no `custom_function` is provided.

        Notes
        -----
        - If `x` or `y` is constant (i.e., has zero variance), the correlation index is defined as 0 
          to avoid division by zero or meaningless regression.
        - If the fitted model performs worse than the mean (SSE > SST), R_I is also defined as 0.
        - Input `x` is internally normalized to the specified range to:
            1. Prevent numerical instability due to extremely large/small values.
            2. Ensure compatibility with the domain restrictions of certain functions
               (e.g., logarithmic, power, or hyperbolic forms).
        - Parameter estimation is performed using `scipy.optimize.curve_fit` 
          with a least-squares objective.
        """
        models = {"linear": lambda x, a, b: a * x + b,
                  "binomial": lambda x, a, b, c: a * x**2 + b * x + c,
                   "exp": lambda x, a, b: b * np.exp(a * x),
                   "ln": lambda x, a, b: a * np.log(x) + b,
                   "hyperbolic": lambda x, a, b: a / x + b,
                   "power": lambda x, a, b: a * x ** b,
                   "custom": None}
        q = ((y - y.mean())**2).sum()
        if x.max() != x.min() and q != 0:
            x_normalized = normalization_lower_bound + (
            normalization_upper_bound - normalization_lower_bound
            ) * (x - x.min()) / (x.max() - x.min())
        else:
            r_i = 0
            return r_i # return r_i = 0 for constant x or constant y
        if method not in models:
            available = "{'" + "', '".join(models.keys()) + "'}"
            raise ValueError(f"Unsupported method '{method}'. Choose from: {available}")
        if x.size != y.size:
            raise ValueError(f"Length of 'x' series ({x.size}) "
                             f"must match length of 'y' series ({y.size}).")
        if x.isna().sum() + y.isna().sum() != 0:
            raise ValueError("The input 'x' or 'y' contains null values. "
                             "Please clean or impute missing data.")
        if method == "ln" and (x_normalized <= 0).any():
            raise ValueError(f"Method '{method}' requires strictly positive x-values.")
        if method in {"power", "hyperbolic"} and (x_normalized == 0).any():
            raise ValueError(f"Method '{method}' can't handle zero x-values.")

        if method == "custom":
            if custom_function is None:
                raise ValueError("Custom function must be provided when method='custom'")
            models["custom"] = custom_function
            model_values = models[method](x_normalized)
        else:
            coeffs = curve_fit(models[method], x_normalized, y, maxfev=10000)[0]
            model_values = models[method](x_normalized, *coeffs)
        q_e = ((y - model_values)**2).sum()
        r_i = 1 - q_e/q
        r_i =  np.sqrt(max(r_i, 0))
        return r_i

    @staticmethod
    def corr_multiple(x: pd.DataFrame,
                      y: pd.Series) -> float:
        """
        Computes the multiple correlation coefficient (R) between a set of predictor variables (x)
        and a response variable (y), based on the determinant of their correlation matrices.

        This implementation is robust to singular correlation matrices. If the determinant of 
        the predictor correlation matrix is zero, R is defined as zero (no linear dependency).

        Parameters
        ----------
        x : pd.DataFrame
            A DataFrame of predictor (explanatory) variables.
            Each column is assumed to represent one independent variable.
        y : pd.Series
            A Series representing the response (dependent) variable.

        Returns
        -------
        float
            The multiple correlation coefficient (R), ranging from 0 to 1.

        Raises
        ------
        ValueError
            If the number of samples in x and y do not match.
            If any NaN values are present in x or y.
        
        Warns
        ------
        UserWarning
            If the predictors in x are found to be multicollinear 
            (i.e., the determinant of their correlation matrix is zero).

        Notes
        -----
        The method uses the formula:

            R² = 1 - |det(corr(X, Y))| / |det(corr(X)|)

        where corr(X, Y) is the correlation matrix of predictors and the response,
        and corr(X) is the correlation matrix of the predictors alone.

        If corr(X) is singular (det = 0), the function returns R = 0.
        """
        if x.shape[0] != y.size:
            raise ValueError(f"Length of 'x' DataFrame ({x.shape[0]}) "
                             f"must match length of 'y' series ({y.size}).")
        if x.isna().sum().sum() + y.isna().sum() != 0:
            raise ValueError("The input 'x' or 'y' contains null values. "
                             "Please clean or impute missing data.")
        factors = x.copy()
        factors.columns = [f"{col} (X{i})" for i, col in enumerate(factors.columns)]
        full = factors.copy()
        full["Y"] = y
        r_det = abs(np.linalg.det(full.corr()))
        r_f_det = abs(np.linalg.det(factors.corr()))
        if np.isclose(r_f_det, 0, atol=1e-10):
            r_multiple = 0.0
            warnings.warn(
                "Matrix of predictors is linearly "
                "dependent (multicollinearity detected). " 
                "Interpret results with caution.",
                UserWarning
            )
        else:
            r_squared = max(0.0, 1 - r_det/r_f_det)
            r_multiple = np.sqrt(r_squared)
        return r_multiple
