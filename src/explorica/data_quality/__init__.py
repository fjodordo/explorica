"""
Facade for data quality analysis.

This package provides a unified interface for data quality analysis, combining
functionality from preprocessing, feature engineering, and outlier detection
submodules into a single convenient namespace.

Classes
-------
DataPreprocessing
    Handling missing values, categorical features, and constant features.
FeatureEngineering
    Encoding methods, binning utilities, and feature transformations.

Methods
-------
get_summary : Comprehensive data quality report

detect_iqr(cls, data, iqr_factor, show_boxplot)
    Detects outliers in a numerical series using the Interquartile Range
    (IQR) method.
detect_zscore(data, threshold)
    Detects outliers in a numerical series using the Z-score method.
remove_outliers(data, subset, detection_method, recursive, **kwargs)
    Remove outliers from a given sequence of numerical data.
replace_outliers(data, detection_method, strategy, recursive, **kwargs)
    Replaces outliers in sequences or mappings according
    to the specified detection method and replacement strategy.
describe_distributions(data, threshold_skewness, threshold_kurtosis,
                       return_as, **kwargs)
    Describe shape (skewness / kurtosis) of one or multiple numeric distributions.
get_skewness(data, method)
    Compute the skewness (third standardized moment) of a numeric sequence.
get_kurtosis(data, method)
    Compute the **excess kurtosis** (fourth standardized moment minus 3)
    of a numeric sequence.
get_constant_features(data, method, threshold, nan_policy)
    Identifies constant and quasi-constant features in the dataset.
get_missing(data, ascending, round_digits)
    Calculate the number and percentage of missing (NaN) values for each column.
drop_missing(data, axis, threshold_pct, threshold_abs, verbose)
    Drops rows or columns containing NaNs according to a specified threshold.
get_categorical_features(data, threshold, **kwargs)
    Identify categorical features in a DataFrame based on unique value counts
    and optionally on data type filters.
set_categorical(data, threshold, nan_policy, verbose, **kwargs)
    Convert eligible columns to Pandas `category` dtype for memory optimization
    and improved performance in certain operations.
freq_encode(data, axis, normalize, round_digits)
    Performs frequency encoding on a categorical feature(s).
ordinal_encode(data, axis, order_method, order_ascending, **kwargs)
    Encode categorical values with ordinal integers
    based on a specified ordering rule.
discretize_continuous(data, bins, binning_method, intervals)
    Discretize continuous numeric data into categorical intervals.
get_entropy(data, method, nan_policy)
    Computes the Shannon entropy of the input data.

Examples
--------
Obtaining distribution metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> import pandas as pd
>>> import explorica.data_quality as data_quality
>>> import seaborn as sns
...
>>> df = sns.load_dataset("titanic")
>>> print(data_quality.describe_distributions(df.select_dtypes("number").dropna(),
                                    threshold_kurtosis=1.0,
                                    threshold_skewness=1.0))

          skewness   kurtosis  is_normal                        desc
survived  0.382140  -1.853969          0                 low-pitched
pclass   -0.467558  -1.418028          0                 low-pitched
age       0.388290   0.168637          1                      normal
sibsp     2.514280   6.987321          0  right-skewed, high-pitched
parch     2.613409   8.782859          0  right-skewed, high-pitched
fare      4.643848  30.699725          0  right-skewed, high-pitched

Finding constants, quasi-constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(data_quality.get_constant_features(df, method="top_value", threshold=0.85))

             top_value_ratio  is_const
survived            0.675824       0.0
pclass              0.862637       1.0
sex                 0.516484       0.0
age                 0.060440       0.0
sibsp               0.598901       0.0
parch               0.664835       0.0
fare                0.038462       0.0
embarked            0.631868       0.0
class               0.862637       1.0
who                 0.478022       0.0
adult_male          0.521978       0.0
deck                0.280220       0.0
embark_town         0.631868       0.0
alive               0.675824       0.0
alone               0.571429       0.0

Obtaining a comprehensive data quality report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> print(data_quality.get_summary(df.dropna()))
                     nans                  duplicates
            count_of_nans pct_of_nans count_of_unique pct_of_unique
survived              0.0         0.0             2.0        0.0110
pclass                0.0         0.0             3.0        0.0165
sex                   0.0         0.0             2.0        0.0110
age                   0.0         0.0            63.0        0.3462
sibsp                 0.0         0.0             4.0        0.0220
parch                 0.0         0.0             4.0        0.0220
fare                  0.0         0.0            93.0        0.5110
embarked              0.0         0.0             3.0        0.0165
class                 0.0         0.0             3.0        0.0165
who                   0.0         0.0             3.0        0.0165
adult_male            0.0         0.0             2.0        0.0110
deck                  0.0         0.0             7.0        0.0385
embark_town           0.0         0.0             3.0        0.0165
alive                 0.0         0.0             2.0        0.0110
alone                 0.0         0.0             2.0        0.0110
...
"""

from .data_preprocessing import DataPreprocessing
from .summary import get_summary
from .feature_engineering import EncodeMethods
from .information_metrics import get_entropy
from .outliers import DetectionMethods, DistributionMetrics, HandlingMethods

get_missing = DataPreprocessing.get_missing
drop_missing = DataPreprocessing.drop_missing
get_constant_features = DataPreprocessing.get_constant_features
get_categorical_features = DataPreprocessing.get_categorical_features
set_categorical = DataPreprocessing.set_categorical

discretize_continuous = EncodeMethods.discretize_continuous
freq_encode = EncodeMethods.freq_encode
ordinal_encode = EncodeMethods.ordinal_encode

detect_iqr = DetectionMethods.detect_iqr
detect_zscore = DetectionMethods.detect_zscore
describe_distributions = DistributionMetrics.describe_distributions
get_skewness = DistributionMetrics.get_skewness
get_kurtosis = DistributionMetrics.get_kurtosis
remove_outliers = HandlingMethods.remove_outliers
replace_outliers = HandlingMethods.replace_outliers

__all__ = [
    "DataPreprocessing",
    "get_missing",
    "drop_missing",
    "get_constant_features",
    "get_categorical_features",
    "set_categorical",
    "EncodeMethods",
    "discretize_continuous",
    "freq_encode",
    "ordinal_encode",
    "replace_outliers",
    "remove_outliers",
    "detect_iqr",
    "detect_zscore",
    "get_skewness",
    "get_kurtosis",
    "describe_distributions",
    "get_summary",
    "get_entropy",
]
