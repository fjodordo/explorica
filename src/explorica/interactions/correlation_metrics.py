"""
Statistical correlation and association measures.

This module provides a collection of functions for computing various
statistical dependency measures between variables. The functions are
implemented as static methods of the ``CorrelationMetrics`` class and are
used as building blocks for higher-level analysis in the
:class:`explorica.InteractionAnalyzer` facade.

Available measures include:
- Cramér’s V for categorical associations.
- η² (eta squared) for associations between a categorical and a numeric variable.
- Correlation index for non-linear dependencies between numeric variables.
- Multiple correlation coefficient for evaluating the joint effect of multiple
  predictors on a target variable.

Main class
----------
CorrelationMetrics
    A utility class containing static methods for computing statistical
    correlation and association measures.
"""

import warnings
from numbers import Number
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import chi2_contingency

from explorica._utils import ConvertUtils as cutils
from explorica._utils import ValidationUtils as vutils
from explorica._utils import read_messages


class CorrelationMetrics:
    """
    Collection of statistical correlation and association measures.

    This class provides static methods for computing dependency measures
    between numeric and/or categorical variables. It is intended as a
    low-level utility, typically accessed via the
    :class:`exporica.InteractionAnalyzer` facade, but can also be
    used directly.

    Methods
    -------
    cramer_v(x, y, bias_correction=True, yates_correction=False)
        Compute Cramér’s V statistic for association between two categorical
        variables.
    eta_squared(values, categories)
        Compute η² (eta squared), measuring association between a numeric
        variable and a categorical grouping.
    corr_index(x, y, method="linear", custom_function=None, **kwargs)
        Compute the correlation index, a measure of non-linear dependency
        between two numeric variables. Supports linear, exponential, binomial,
        logarithmic, hyperbolic, and custom functional forms.
    corr_multiple(x, y)
        Compute the multiple correlation coefficient between a target variable
        and a set of predictor variables.

    Examples
    --------
    >>> from correlation_metrics import CorrelationMetrics as cm
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 6, 8, 10]
    >>> cm.corr_index(x, y, method="linear")
    np.float64(1.0)
    """

    _warns = read_messages()["warns"]
    _errors = read_messages()["errors"]

    @staticmethod
    def cramer_v(
        x: Sequence,
        y: Sequence,
        bias_correction: bool = True,
        yates_correction: bool = False,
    ) -> float:
        """
        Calculates Cramér's V statistic for measuring
        the association between two categorical variables.

        Parameters
        ----------
        x : Sequence
            First categorical variable.
        y : Sequence
            Second categorical variable.
        bias_correction : bool, optional, default=True
            Whether to apply bias correction (recommended for small samples).
        yates_correction : bool, optional, default=False
            Whether to apply Yates' correction for continuity
            (only applies to 2x2 tables; usually set to False when using Cramér's V).

        Returns
        -------
        float
            Cramér's V value, ranging from 0 (no association) to 1 (perfect
            association). Returns 0 if the statistic is undefined
            (e.g., due to zero denominator).

        Raises
        ------
        ValueError
            If 'x' or 'y' contains NaN values.
            If input sequences lengths mismatch

        Notes
        -----
        The statistic is based on the chi-square test of independence:

        .. math::

            \\phi^2 = \\frac{\\chi^2}{n}

        where :math:`n` is the sample size.

        Without bias correction:

        .. math::

            V = \\sqrt{ \\frac{\\phi^2}{\\min(r - 1, k - 1)} }

        where :math:`r` is the number of rows and :math:`k` is the number of columns
        in the contingency table.

        With bias correction (Bergsma, 2013):

        .. math::

            \\phi^2_{\\text{corr}} =
            \\max\\left( 0, \\phi^2 - \\frac{(r-1)(k-1)}{n-1} \\right)

            V = \\sqrt{ \\frac{\\phi^2_{\\text{corr}}}{\\min(r - 1, k - 1)} }

        If ``yates_correction=True``, Yates' continuity correction is applied when
        computing :math:`\\chi^2` (only affects :math:`2 \\times 2` contingency tables).
        """

        vutils.validate_lenghts_match(
            x, y, err_msg="Length of 'x' must match length of 'y'"
        )
        vutils.validate_array_not_contains_nan(
            x,
            err_msg="""
            The input 'x' contains null values.
            Please clean or impute missing data.""",
        )
        vutils.validate_array_not_contains_nan(
            y,
            err_msg="""The input 'y' contains null values.
            Please clean or impute missing data.""",
        )
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix, correction=yates_correction)[0]
        n = confusion_matrix.to_numpy().sum()
        r, k = confusion_matrix.shape
        min_dim = np.min((r - 1, k - 1))

        result = 0.0
        if min_dim == 0.0:
            return result

        if bias_correction:
            bias = ((r - 1) * (k - 1)) / (n - 1)
            phi2_corr = np.max((0, chi2 / n - bias))
            result = np.sqrt(phi2_corr / min_dim)
        else:
            result = np.sqrt(chi2 / (n * min_dim))
        return result

    @staticmethod
    def eta_squared(values: Sequence[Number], categories: Sequence) -> float:
        """
        Calculate the eta-squared (η²) statistic for categorical and numeric variables.

        η² (eta squared) is a measure of effect size used to quantify the proportion of
        variance in a numerical variable that can be attributed to differences between
        categories of a categorical variable.

        Parameters
        ----------
        values : Sequence[Number]
            A numerical sequence representing the dependent (response) variable.
        categories : Sequence
            A categorical sequence representing the independent
            (grouping) variable.

        Returns
        -------
        float
            Eta-squared statistic in the range [0, 1], where:
            - 0 means no association between variables,
            - 1 means perfect association (all variance explained by groups).

        Raises
        ------
        ValueError
            If the lengths of ``values`` and ``categories`` do not match,
            or if either of them contains NaN values.

        Notes
        -----
        If the total variance of `values` is zero, the function returns 0.
        NaN values should be handled before calling this function.
        """
        vutils.validate_lenghts_match(
            values,
            categories,
            err_msg=CorrelationMetrics._errors["arrays_lens_mismatch_f"].format(
                "values", "categories"
            ),
        )
        vutils.validate_array_not_contains_nan(
            values,
            err_msg=CorrelationMetrics._errors["array_contains_nans_f"].format(
                "values"
            ),
        )
        vutils.validate_array_not_contains_nan(
            categories,
            err_msg=CorrelationMetrics._errors["array_contains_nans_f"].format(
                "categories"
            ),
        )
        df = pd.DataFrame({"category": categories, "values": values})
        mean_by_group = df.groupby("category", observed=True)["values"].mean()
        mean = df["values"].mean()
        n_by_group = df.groupby("category", observed=True)["values"].count()
        n = df["values"].size

        bg_disperison = np.sum(((mean_by_group - mean) ** 2) * n_by_group) / n
        dispersion = ((df["values"] - mean) ** 2).sum() / n

        # zero dispersion in this case indicates a zero coefficient of determination
        eta_sq = bg_disperison / dispersion if dispersion != 0 else 0
        return eta_sq

    @staticmethod
    def corr_index(
        x: Sequence[Number],
        y: Sequence[Number],
        method: str = "linear",
        custom_function: Optional[Callable[[Number], Number]] = None,
        normalization_bounds: Sequence[Number] = None,
    ) -> float:
        """
        Calculates a nonlinear correlation index between two series `x` and `y`,
        based on the proportion of variance explained by the fitted function.

        The index is computed as:

            R_I = sqrt(1 - SSE / SST),

        where SSE is the sum of squared errors between the true `y` and predicted
        `y(x)`, and SST is the total sum of squares `y` with respect to its mean.

        This method is designed to handle non-linear dependencies
        and is robust to monotonic transformations. In degenerate cases
        (e.g., constant `x` or `y`), a correlation index of 0 is returned.

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
        x : Sequence[Number]
            The explanatory variable.
        y : Sequence[Number]
            The response variable.
        method : str, default='linear'
            Regression model to use. See supported methods above.
        normalization_bounds : Sequence[Number], optional
            A pair of numeric values (``lower_bound``, ``upper_bound``) specifying
            the range to which the input ``x`` values are rescaled before fitting.
            If ``None`` (default), automatic normalization is applied **only** for
            methods with domain restrictions:

            - 'exp' → scaled to [0, 5]
            - 'ln' → scaled to [1, 10]
            - 'power' → scaled to [1, 10]
            - 'hyperbolic' → scaled to [2, 20]

            For all other methods, no normalization is applied unless explicitly
            provided by the user.
        custom_function : Callable, optional
            A user-defined function that takes a Number values and returns Number
            predictions. Required when `method='custom'`

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
        - If `x` or `y` is constant (i.e., has zero variance), the correlation index is
          defined as 0 to avoid division by zero or meaningless regression.
        - If the fitted model performs worse than the mean
          (SSE > SST), R_I is also defined as 0.
        - User-provided ``normalization_bounds`` take precedence and will be applied
            regardless of the chosen method.
        - Automatic normalization is enabled selectively for models with restricted
            domains to:
            1. Prevent numerical instability due to extremely large/small values.
            2. Ensure compatibility with the domain restrictions of certain functions
               (e.g., logarithmic, power, or hyperbolic forms).
        - Parameter estimation is performed using `scipy.optimize.curve_fit`
          with a least-squares objective.
        """

        def scale_minmax(seq, lower_bound, upper_bound):
            seq = lower_bound + (upper_bound - lower_bound) * (seq - seq.min()) / (
                seq.max() - seq.min()
            )
            return seq

        x_series = cutils.convert_dataframe(x).iloc[:, 0]
        y_series = cutils.convert_dataframe(y).iloc[:, 0]
        regressions = {
            "models": {
                "linear": lambda x, a, b: a * x + b,
                "binomial": lambda x, a, b, c: a * x**2 + b * x + c,
                "exp": lambda x, a, b: b * np.exp(a * x),
                "ln": lambda x, a, b: a * np.log(x) + b,
                "hyperbolic": lambda x, a, b: a / x + b,
                "power": lambda x, a, b: a * x**b,
                "custom": None,
            },
            "norm_bounds_auto": {
                "exp": (0, 5),
                "ln": (1, 10),
                "power": (1, 10),
                "hyperbolic": (2, 20),
            },
        }
        q = ((y_series - y_series.mean()) ** 2).sum()
        # return r_i = 0 for constant y
        if q == 0 or x_series.min() == x_series.max():
            return 0.0
        if normalization_bounds is None:
            if method in regressions["norm_bounds_auto"]:
                lower_bound, upper_bound = regressions["norm_bounds_auto"][method]
                x_series = scale_minmax(x_series, lower_bound, upper_bound)
        else:
            lower_bound, upper_bound = normalization_bounds
            x_series = scale_minmax(x_series, lower_bound, upper_bound)
        vutils.validate_string_flag(
            method,
            regressions["models"],
            f"Unsupported method '{method}'." f"Choose from: ({regressions['models']})",
        )
        vutils.validate_lenghts_match(
            x_series,
            y_series,
            f"Length of 'x' series ({x_series.size}) "
            f"must match length of 'y' series ({y_series.size}).",
        )
        vutils.validate_array_not_contains_nan(
            x_series,
            "The 'x' contains NaN values. Please clean or impute missing data.",
        )
        vutils.validate_array_not_contains_nan(
            y_series,
            "The 'y' contains NaN values. Please clean or impute missing data.",
        )
        if method == "ln" and (x_series <= 0).any():
            raise ValueError(f"Method '{method}' requires strictly positive x-values.")
        if method in {"power", "hyperbolic"} and (x_series == 0).any():
            raise ValueError(f"Method '{method}' can't handle zero x-values.")

        if method == "custom":
            if custom_function is None:
                raise ValueError(
                    "Custom function must be provided when method='custom'"
                )
            regressions["models"]["custom"] = custom_function
            model_values = regressions["models"][method](x_series)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                coeffs = curve_fit(
                    regressions["models"][method], x_series, y_series, maxfev=10000
                )[0]
            model_values = regressions["models"][method](x_series, *coeffs)
        q_e = ((y_series - model_values) ** 2).sum()
        return np.sqrt(max(1 - q_e / q, 0))

    @staticmethod
    def corr_multiple(x: Sequence[Sequence[Number]], y: Sequence[Number]) -> float:
        """
        Computes the multiple correlation coefficient (R) between a set of predictor
        variables (x) and a response variable (y), based on the determinant of their
        correlation matrices.

        This implementation is robust to singular correlation matrices. If the
        determinant of the predictor correlation matrix is zero, R is defined as zero
        (no linear dependency).

        Parameters
        ----------
        x : Sequence[Sequence[Number]]
            A sequence of predictor (explanatory) variables.
            Each column is assumed to represent one independent variable.
        y : Sequence[Number]
            A series representing the response (dependent) variable.

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
        factors = cutils.convert_dataframe(x)
        y_series = cutils.convert_dataframe(y).iloc[:, 0]
        if x.shape[0] != y.size:
            raise ValueError(
                f"Length of 'x' DataFrame ({factors.shape[0]}) "
                f"must match length of 'y' series ({y_series.size})."
            )
        vutils.validate_array_not_contains_nan(
            factors, "The 'x' contains NaN values. Please clean or impute missing data."
        )
        vutils.validate_array_not_contains_nan(
            y_series,
            "The 'y' contains NaN values. Please clean or impute missing data.",
        )
        factors.columns = [f"{col} (X{i})" for i, col in enumerate(factors.columns)]
        full = factors.copy()
        full["Y"] = y_series
        r_det = abs(np.linalg.det(full.corr()))
        r_f_det = abs(np.linalg.det(factors.corr()))
        if np.isclose(r_f_det, 0, atol=1e-10):
            r_multiple = 0.0
            warnings.warn(
                CorrelationMetrics._warns["multicollinearity"],
                UserWarning,
            )
        else:
            r_squared = max(0.0, 1 - r_det / r_f_det)
            r_multiple = np.sqrt(r_squared)
        return r_multiple
