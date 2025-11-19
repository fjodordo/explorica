"""
Module for analyzing statistical interactions between variables.

This module provides tools for measuring and comparing dependencies between
different types of features, including numeric, categorical, and hybrid
feature sets. Functions cover a wide spectrum of dependency measures:
linear (Pearson, Spearman), categorical (Cramér’s V), hybrid (η²),
non-linear regression–based indices (exponential, binomial, power-law, etc.),
and multiple-correlation metrics. In addition, matrix and vectorized forms
of these measures are available for batch analysis.

Available functions
-------------------
cramer_v(x, y)
    Computes Cramér’s V, a measure of association between two
    categorical variables.

eta_squared(values, categories)
    Computes η² (eta squared), a measure of association between a
    numerical feature and a categorical grouping variable.

corr_index(x, y, method)
    Computes the correlation index between two numerical features.

corr_multiple(x, y)
    Computes the multiple correlation coefficient between a set of
    (two or more) predictor features and a target variable.

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

high_corr_pairs(numeric_features=None,
                category_features=None,
                threshold=0.7,
                **kwargs) -> pd.DataFrame | None
    Finds and returns all significant pairs of correlated features
    from the input datasets. Supports linear (Pearson, Spearman),
    non-linear (e.g. exponential, binomial, power-law), categorical
    (Cramér’s V), and hybrid (η²) correlation measures. Users may
    optionally enable non-linear and multiple-correlation modes.

- All functions are available at the top level of this module and do not
  require any class instantiation.
- Helper functions remain internal and may be prefixed with `_`.

Examples
--------
>>> import pandas as pd
>>> from explorica.interactions import corr_matrix
>>> df = pd.DataFrame({
...     "x1": [1, 2, 3, 4],
...     "x2": [2, 4, 6, 8],
...     "cat": ["a", "a", "b", "b"]
... })
>>> corr_matrix(df, method="pearson")
          x1        x2
x1  1.000000  1.000000
x2  1.000000  1.000000
"""

from explorica.interactions.aggregators import detect_multicollinearity as dm
from explorica.interactions.aggregators import high_corr_pairs as hcp
from explorica.interactions.correlation_matrices import CorrelationMatrices
from explorica.interactions.correlation_metrics import CorrelationMetrics as cm

cramer_v = cm.cramer_v
eta_squared = cm.eta_squared
corr_index = cm.corr_index
corr_multiple = cm.corr_multiple
corr_matrix = CorrelationMatrices.corr_matrix
corr_matrix_linear = CorrelationMatrices.corr_matrix_linear
corr_matrix_cramer_v = CorrelationMatrices.corr_matrix_cramer_v
corr_matrix_eta = CorrelationMatrices.corr_matrix_eta
corr_matrix_corr_index = CorrelationMatrices.corr_matrix_corr_index
corr_vector_multiple = CorrelationMatrices.corr_vector_multiple
corr_matrix_multiple = CorrelationMatrices.corr_matrix_multiple
high_corr_pairs = hcp
detect_multicollinearity = dm
