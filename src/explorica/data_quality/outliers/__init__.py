"""
Outlier detection and handling utilities.

This subpackage provides classes for detecting, handling, and
analyzing outliers in numerical datasets. Key utilities exported
at the top level:

Functions
---------
detect_iqr(data, iqr_factor=1.5,
    get_boxplot=False, nan_policy="drop", boxplot_kws=None
)
    Detects outliers in a numerical series using the Interquartile
    Range (IQR) method.
detect_zscore(
    data: Sequence[float] | Sequence[Sequence[float]] | Mapping[str, Sequence[Any]],
    threshold: Optional[float] = 2.0,
)
    Detects outliers in a numerical series using the Z-score method.
replace_outliers(data, detection_method="iqr",
    strategy="median", recursive=False, **kwargs
)
    Replaces outliers in sequences or mappings according
    to the specified detection method and replacement strategy.
remove_outliers(data, subset=None, detection_method="iqr",
    recursive=False, **kwargs
)
    Remove outliers from a given sequence of numerical data.
get_skewness(data, method="general")
    Compute the skewness (third standardized moment) of a numeric sequence.
get_kurtosis(data, method="general")
    Compute the **excess kurtosis** (fourth standardized moment minus 3)
    of a numeric sequence.
describe_distributions(data, threshold_skewness=0.25, threshold_kurtosis=0.25,
    return_as="dataframe", **kwargs
)
    Describe shape (skewness / kurtosis) of one or multiple numeric distributions.
"""

from .detection import detect_iqr, detect_zscore
from .handling import replace_outliers, remove_outliers
from .stats import describe_distributions, get_skewness, get_kurtosis

__all__ = [
    "replace_outliers",
    "remove_outliers",
    "get_skewness",
    "get_kurtosis",
    "describe_distributions",
    "detect_iqr",
    "detect_zscore",
]
