"""
Facade for data quality analysis.

This package provides a unified interface for data quality analysis, combining
functionality from preprocessing, feature engineering, and outlier detection
submodules into a single convenient namespace.

Functions
---------
get_summary(data, return_as = "dataframe",
    auto_round=True, round_digits=4, **kwargs
)
    Compute a data-quality summary for a dataset.
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
# Obtaining distribution metrics
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sns
>>> from explorica.data_quality import (
...     describe_distributions,
...     get_constant_features,
...     get_summary
... )
>>>
>>> df = sns.load_dataset("titanic")
>>> result = describe_distributions(df.select_dtypes("number").dropna(),
...     threshold_kurtosis=1.0, threshold_skewness=1.0)
>>> np.round(result, 4)
          skewness  kurtosis  is_normal                        desc
survived    0.3821   -1.8540          0                 low-pitched
pclass     -0.4676   -1.4180          0                 low-pitched
age         0.3883    0.1686          1                      normal
sibsp       2.5143    6.9873          0  right-skewed, high-pitched
parch       2.6134    8.7829          0  right-skewed, high-pitched
fare        4.6438   30.6997          0  right-skewed, high-pitched

>>> # Finding constants, quasi-constants
>>> result = get_constant_features(df, method="top_value", threshold=0.85)
>>> np.round(result, 4)
             top_value_ratio  is_const
survived              0.6758       0.0
pclass                0.8626       1.0
sex                   0.5165       0.0
age                   0.0604       0.0
sibsp                 0.5989       0.0
parch                 0.6648       0.0
fare                  0.0385       0.0
embarked              0.6319       0.0
class                 0.8626       1.0
who                   0.4780       0.0
adult_male            0.5220       0.0
deck                  0.2802       0.0
embark_town           0.6319       0.0
alive                 0.6758       0.0
alone                 0.5714       0.0

>>> # Obtaining a comprehensive data quality report
>>> result = get_summary(df.dropna())
>>> np.round(result.loc[:, ["nans", "duplicates"]], 4)
                     nans              ...    duplicates
            count_of_nans pct_of_nans  ... pct_of_unique quasi_constant_pct
survived                0         0.0  ...        0.0110             0.6758
pclass                  0         0.0  ...        0.0165             0.8626
sex                     0         0.0  ...        0.0110             0.5165
age                     0         0.0  ...        0.3462             0.0604
sibsp                   0         0.0  ...        0.0220             0.5989
parch                   0         0.0  ...        0.0220             0.6648
fare                    0         0.0  ...        0.5110             0.0385
embarked                0         0.0  ...        0.0165             0.6319
class                   0         0.0  ...        0.0165             0.8626
who                     0         0.0  ...        0.0165             0.4780
adult_male              0         0.0  ...        0.0110             0.5220
deck                    0         0.0  ...        0.0385             0.2802
embark_town             0         0.0  ...        0.0165             0.6319
alive                   0         0.0  ...        0.0110             0.6758
alone                   0         0.0  ...        0.0110             0.5714
<BLANKLINE>
[15 rows x 5 columns]
"""

from .data_preprocessing import (
    drop_missing,
    get_categorical_features,
    get_constant_features,
    get_missing,
    set_categorical,
)
from .feature_engineering import discretize_continuous, freq_encode, ordinal_encode
from .information_metrics import get_entropy
from .outliers import (
    describe_distributions,
    detect_iqr,
    detect_zscore,
    get_kurtosis,
    get_skewness,
    remove_outliers,
    replace_outliers,
)
from .summary import get_summary

__all__ = [
    "get_constant_features",
    "get_categorical_features",
    "get_missing",
    "drop_missing",
    "set_categorical",
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
