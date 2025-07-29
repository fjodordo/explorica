"""
interaction_analyzer.py

Module for analyzing interactions between variables in a dataset,
including correlation matrices for numeric features and association
measures for categorical features (such as Cramér's V).
"""
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

    def corr_matrix(self,
                    dataset: pd.DataFrame,
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
            For 'pearson' and 'spearman', and all nonlinear methods, all columns must be numeric.
            For 'cramer_v', all columns should be categorical.
            For 'eta', numeric features in `dataset` are 
            compared to categorical features in `groups`.


        method : str, optional
            Method used to compute correlation or association:
            - 'pearson' : Pearson correlation (linear, continuous features).
            - 'spearman' : Spearman rank correlation (monotonic, non-parametric).
            - 'cramer_v' : Cramér's V (categorical-categorical association).
            - 'eta' : Eta coefficient (numeric-categorical association, asymmetric).
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
        
        Notes
        ------
        - Pearson, Spearman, and other numerical correlation methods internally select
          only features of numeric type (`Number`) from the provided DataFrame.
          Non-numeric columns (e.g., categorical strings or object types) are ignored.
        """
        supported_methods = {"pearson", "spearman", "cramer_v",
                             "eta", "exp", "binomial", "ln",
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
                    v = self.cramer_v(dataset[col1], dataset[col2])
                    matrix.iloc[i, j] = v
                    matrix.iloc[j, i] = v  # ensure symmetry

        if method == "eta":
            category_names = groups.columns
            numeric_names = numeric_features.columns

            matrix = pd.DataFrame(
                data=np.zeros((numeric_features.shape[1], groups.shape[1])),
                index=numeric_names,
                columns=category_names
            )
            for i in range(numeric_features.shape[1]):
                for j in range(numeric_features.shape[1]):
                    eta = np.sqrt(self.eta_squared(numeric_features[numeric_names[i]],
                                                   groups[category_names[j]]))
                    matrix.iloc[i, j] = eta

        if method in {"exp", "binomial", "ln", "hyperbolic", "power"}:
            features = numeric_features.columns
            index = [f"{col} (X)" for col in features]
            cols = [f"{col} (Y)" for col in features]
            n = len(cols)
            matrix = pd.DataFrame(np.ones((n, n)), index=index, columns=cols)

            for i, col_x in enumerate(features):
                for j, col_y in enumerate(features):
                    try:
                        corr_coef = self.corr_index(numeric_features[col_x],
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

    def high_corr_pairs(self, dataset: pd.DataFrame, method: str = 'pearson',
                        threshold: float = 0.7) -> pd.DataFrame | None:
        """
        Find pairs of features with high correlation/association.

        Parameters
        ----------
        dataset : pd.DataFrame
            DataFrame with numeric or categorical features.
        method : str, optional
            Correlation method: 'pearson', 'spearman', or 'cramer_v'.
        threshold : float, optional
            Minimum absolute value of correlation to consider.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ['1st', '2nd', 'coef'] for each pair
            exceeding the given threshold. Returns None if no such pairs found.
        """
        correlation_matrix = self.corr_matrix(dataset, method=method)
        corr_pairs = []

        cols = correlation_matrix.columns
        for i, col1 in enumerate(cols):
            for j in range(i + 1, len(cols)):  # only the upper triangle
                coef = correlation_matrix.iloc[i, j]
                if abs(coef) >= threshold:
                    corr_pairs.append([col1, cols[j], coef])

        if not corr_pairs:
            return None

        return pd.DataFrame(corr_pairs, columns=["1st", "2nd", "coef"])\
             .sort_values(by="coef", key=abs, ascending=False)

    def cramer_v(
        self,
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

    def eta_squared(self, values: pd.Series, category: pd.Series) -> float:
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

    def corr_index(self,
                   x: pd.Series,
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
