"""
Facade for data quality analysis.

This subpackage provides top-level access to functions for data
preprocessing, feature engineering, and outlier handling.
All major utilities are exported here for easy import.

Modules available at the top level include:
- data_preprocessing: missing values, categories, constant features
- feature_engineering: encoding and binning transformations
- outliers: detection and processing of outliers, distribution stats
"""

from .data_quality_handler import (
    DataPreprocessing,
    DataQualityHandler,
    FeatureEngineering,
    bin_numeric,
    check_columns_uniqueness,
    describe_distributions,
    detect_iqr,
    detect_zscore,
    drop_missing,
    freq_encode,
    get_categories,
    get_constant_features,
    get_kurtosis,
    get_missing,
    get_skewness,
    get_summary,
    ordinal_encode,
    remove_outliers,
    replace_outliers,
    set_categories,
)

__all__ = [
    "DataPreprocessing",
    "check_columns_uniqueness",
    "get_missing",
    "drop_missing",
    "get_constant_features",
    "get_categories",
    "set_categories",
    "FeatureEngineering",
    "freq_encode",
    "ordinal_encode",
    "bin_numeric",
    "replace_outliers",
    "remove_outliers",
    "detect_iqr",
    "detect_zscore",
    "get_skewness",
    "get_kurtosis",
    "describe_distributions",
    "DataQualityHandler",
    "get_summary",
]
