"""
The ``data_quality_handler`` module provides a unified facade for handling
data preprocessing, feature engineering, and outlier detection tasks.
It serves as a single entry point to access methods from different
submodules of the ``explorica.data_quality`` package.

This design allows for both modular usage (importing specific classes)
and simplified usage through the facade.

Available classes
-----------------
DataPreprocessor
    Provides functionality for handling missing values, managing categories,
    and detecting constant features.

FeatureEngineer
    Includes encoding methods, binning utilities, and other feature
    transformation techniques.

OutlierHandler
    Provides detection and handling of outliers, as well as statistical
    distribution descriptions.

Available functions

- DataPreprocessor:
    - ``check_columns_uniqueness``
    - ``get_missing``
    - ``drop_missing``
    - ``get_constant_features``
    - ``get_categories``
    - ``set_categories``

- FeatureEngineer:
    - ``freq_encode``
    - ``ordinal_encode``
    - ``bin_numeric``

- OutlierHandler:
    - ``replace_outliers``
    - ``remove_outliers``
    - ``detect_iqr``
    - ``detect_zscore``
    - ``get_skewness``
    - ``get_kurtosis``
    - ``describe_distributions``

Examples
--------
>>> from seaborn import load_dataset
>>> from explorica.data_quality import describe_distributions

>>> df = load_dataset("titanic")

>>> df = df.select_dtypes("number").dropna()
>>> print(describe_distributions(df))

| feature  | is_normal | desc                          | skewness | kurtosis  |
|----------|-----------|-------------------------------|----------|-----------|
| survived | False     | right-skewed, low-pitched     | 0.382140 | -1.853969 |
| pclass   | False     | left-skewed, low-pitched      | -0.467558| -1.418028 |
| age      | False     | right-skewed                  | 0.388290 | 0.168637  |
| sibsp    | False     | right-skewed, high-pitched    | 2.514280 | 6.987321  |
| parch    | False     | right-skewed, high-pitched    | 2.613409 | 8.782859  |
| fare     | False     | right-skewed, high-pitched    | 4.643848 | 30.699725 |

Notes
-----
This is a high-level interface. For more complex or specialized
workflows, use the underlying classes directly:

- ``DataPreprocessor``
- ``FeatureEngineer``
- ``OutlierHandler``
"""

from .data_preprocessor import DataPreprocessor as dp
from .feature_engineer import FeatureEngineer as fe
from .outlier_handler import OutlierHandler as oh

data_preprocessing = dp
check_columns_uniqueness = dp.check_columns_uniqueness
get_missing = dp.get_missing
drop_missing = dp.drop_missing
get_constant_features = dp.get_constant_features
get_categories = dp.get_categories
set_categories = dp.set_categories

feature_engineering = fe
freq_encode = fe.freq_encode
ordinal_encode = fe.ordinal_encode
bin_numeric = fe.bin_numeric

outliers = oh
replace_outliers = oh.replace_outliers
remove_outliers = oh.remove_outliers
detect_iqr = oh.detect_iqr
detect_zscore = oh.detect_zscore
get_skewness = oh.get_skewness
get_kurtosis = oh.get_kurtosis
describe_distributions = oh.describe_distributions
