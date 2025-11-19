"""
Facade for statistical interaction analysis.

This subpackage provides top-level access to functions for measuring
dependencies and correlations between numerical, categorical, and hybrid
feature sets. All major correlation and association functions are
exported here for easy import.

Functions available at the top level include:
- cramer_v, eta_squared: categorical/hybrid association measures
- corr_index, corr_multiple: numerical correlation indices
- corr_matrix*, corr_vector_multiple, high_corr_pairs: matrix/vectorized utilities
"""

from .interaction_analyzer import (
    corr_index,
    corr_matrix,
    corr_matrix_corr_index,
    corr_matrix_cramer_v,
    corr_matrix_eta,
    corr_matrix_linear,
    corr_matrix_multiple,
    corr_multiple,
    corr_vector_multiple,
    cramer_v,
    detect_multicollinearity,
    eta_squared,
    high_corr_pairs,
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
