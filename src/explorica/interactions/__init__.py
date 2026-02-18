"""
Facade for statistical interaction analysis.

This package provides top-level access to functions for measuring
dependencies and correlations between numerical, categorical, and hybrid
feature sets. All major correlation and association functions are
exported here for easy import.

Functions
---------
detect_multicollinearity(numeric_features=None, category_features=None, method="VIF",
    return_as="dataframe", **kwargs
)
    Detect multicollinearity among features using either Variance Inflation Factor (VIF)
    or correlation-based methods.
high_corr_pairs(numeric_features=None, category_features=None, threshold=0.7, **kwargs)
    Finds and returns all significant pairs of
    correlated features from the input datasets.
cramer_v(x, y, bias_correction=True, yates_correction=False)
    Calculates Cramér's V statistic for measuring
    the association between two categorical variables.
eta_squared(values, categories)
    Calculate the eta-squared (η²) statistic for categorical and numeric variables.
    η² (eta squared) is a measure of effect size used to quantify the proportion of
    variance in a numerical variable that can be attributed to differences between
    categories of a categorical variable.
corr_index(x, y, method="linear", custom_function=None, normalization_bounds=None)
    Calculates a nonlinear correlation index between two series `x` and `y`,
    based on the proportion of variance explained by the fitted function.
corr_multiple(x, y)
    Computes the multiple correlation coefficient (R) between a set of predictor
    variables (x) and a response variable (y), based on the determinant of their
    correlation matrices.
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

from explorica.interactions.aggregators import detect_multicollinearity
from explorica.interactions.aggregators import high_corr_pairs
from explorica.interactions.correlation_matrices import (
    corr_matrix,
    corr_matrix_linear,
    corr_matrix_cramer_v,
    corr_matrix_eta,
    corr_matrix_corr_index,
    corr_vector_multiple,
    corr_matrix_multiple,
)
from explorica.interactions.correlation_metrics import (
    cramer_v,
    eta_squared,
    corr_index,
    corr_multiple,
)

__all__ = [
    "cramer_v",
    "corr_index",
    "eta_squared",
    "corr_multiple",
    "corr_matrix",
    "corr_matrix_linear",
    "corr_matrix_cramer_v",
    "corr_matrix_eta",
    "corr_matrix_corr_index",
    "corr_vector_multiple",
    "corr_matrix_multiple",
    "high_corr_pairs",
    "detect_multicollinearity",
]
