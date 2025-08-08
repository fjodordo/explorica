"""
Data preprocessing utilities for exploratory data analysis (EDA).

This module provides the `DataPreprocessor` class, which contains static methods
for inspecting and cleaning datasets before analysis or modeling.  
The available tools include:
- Detection of duplicate column combinations
- Missing value analysis and selective removal
- Detection of constant and quasi-constant features
- Identification of categorical features
- Conversion of suitable columns to `pandas.Categorical` type for memory optimization

Notes
-----
All methods are implemented as `@staticmethod`, so the class does not maintain any state.
"""
from itertools import combinations
from typing import Optional, Union

import pandas as pd

class DataPreprocessor:
    """
    A collection of static methods for common data preprocessing tasks.

    This class provides utility functions for dataset inspection, cleaning, and 
    optimization, especially useful in the exploratory data analysis (EDA) stage.

    Methods
    -------
    check_columns_uniqueness(dataset: pd.DataFrame, max_combination_size: int, ...) -> pd.DataFrame
        Check for duplicate rows across all combinations of features up to a specified size.
        Useful for identifying unique feature sets or repeated patterns.

    get_missing(dataset: pd.DataFrame) -> pd.DataFrame
        Return the number and proportion of missing (NaN) values per column.

    drop_missing(dataset: pd.DataFrame, threshold_pct: float = 0.01, 
        threshold_abs: int = None, return_report: bool = False) -> pd.DataFrame
        Drop rows with missing values only in columns where 
        the NaN ratio does not exceed the given threshold.

    get_constant_features(dataset: pd.DataFrame, quasi_constant_threshold: float,
        include_nan: bool = False) -> pd.DataFrame
        Identify constant and quasi-constant features 
        based on the frequency of the most common value.
        Returns a DataFrame with columns: `is_constant` and `top_value_ratio`.

    get_categories(dataset: pd.DataFrame, threshold, ...) -> pd.DataFrame
        Identify columns that can be considered categorical based on the number of unique values
        and optional inclusion of numerical, boolean, and datetime columns.

    set_categories(dataset: pd.DataFrame, threshold, ...) -> pd.DataFrame
        Convert identified categorical columns to `pandas.Categorical` dtype.
        This can significantly reduce memory usage and may improve performance
        when working with repeated values.
    """

    def __init__(self):
        pass

    @staticmethod
    def check_columns_uniqueness(dataset: pd.DataFrame,
                                 max_combination_size: Optional[int] = 2,
                                 lower_threshold: Optional[float] = 0.0,
                                 upper_threshold: Optional[float] = 1.0,
                                 ascending: Optional[bool] = False,
                                 ) -> pd.DataFrame:
        """
        Evaluate all column combinations up to a given size and compute
        the number and percentage of duplicated rows per combination.

        Parameters
        ----------
        dataset : pd.DataFrame
            Input dataset to analyze.
        max_combination_size : int, optional
            Maximum number of columns in each combination (default is 2).
        lower_threshold : float, optional
            Lower bound for the percentage of duplicated rows (inclusive, default is 0.0).
        upper_threshold : float, optional
            Upper bound for the percentage of duplicated rows (inclusive, default is 1.0).
        ascending : bool, optional
            Sort results by count of duplicates in ascending order (default is False).

        Returns
        -------
        pd.DataFrame
            Table of combinations and their duplicate statistics. Includes:
                - combination: tuple of column names
                - count_of_duplicated: number of duplicated rows for that combination
                - pct_of_duplicated: percentage of duplicated rows for that combination

        Raises
        ------
        ValueError
            If thresholds are not in the range [0, 1] or lower_threshold > upper_threshold.

        Notes
        -----
        If max_combination_size > number of columns, it will be capped to the number of columns.
        """
        if not (0 <= lower_threshold <= 1 and 0 <= upper_threshold <= 1):
            raise ValueError("Thresholds must be in the range [0, 1].")

        if lower_threshold > upper_threshold:
            raise ValueError("Lower threshold cannot be greater than upper threshold.")

        results = {"combination": [], "count_of_duplicated": [], "pct_of_duplicated": []}
        dataset_length = dataset.shape[0]

        # Loop through all combinations of columns up to the given size
        for comb_size in range(1, max_combination_size + 1):
            for comb in combinations(dataset.columns, comb_size):
                duplicated_count = dataset.duplicated(subset=comb).sum()
                duplicated_pct = duplicated_count / dataset_length

                results["combination"].append(comb)
                results["count_of_duplicated"].append(duplicated_count)
                results["pct_of_duplicated"].append(duplicated_pct)

        results = pd.DataFrame(results)

        # Apply threshold filtering
        mask = (results["pct_of_duplicated"] >= lower_threshold) & \
            (results["pct_of_duplicated"] <= upper_threshold)
        results = results[mask]
        results = results.sort_values(
            by="count_of_duplicated", ascending=ascending).reset_index(drop=True)
        return results

    @staticmethod
    def get_missing(dataset: pd.DataFrame):
        """
        Calculate the number and percentage of missing (NaN) values for each column.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:
            - `count_of_nans` : int  
            Number of NaN values in each column.
            - `pct_of_nans` : float  
            Proportion of NaN values in each column (0.0 to 1.0).

        Notes
        -----
        - The `pct_of_nans` values are calculated as the fraction of missing values
        relative to the total number of rows in the dataset.
        - Useful for quickly identifying columns with high proportions of missing data
        before applying data cleaning or imputation.
        """
        nans = dataset.isna()
        nan_count = nans.sum()
        nan_ratio = nans.mean()
        missing_values = pd.DataFrame({"count_of_nans" :nan_count,
                                       "pct_of_nans": nan_ratio},
                                       index=nan_count.index)
        return missing_values

    @staticmethod
    def drop_missing(dataset: pd.DataFrame,
                     threshold_pct: Optional[float] = 0.01,
                     threshold_abs: Optional[int] = None,
                     return_report: Optional[bool] = False
                     ) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
        """
        Drops rows with NaNs from columns where 
        the proportion of missing values is below a threshold.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input DataFrame to process.

        threshold_pct : float, optional, default=0.01
            The maximum proportion of NaNs allowed in 
            a column to consider it for row-wise NaN deletion.

        threshold_abs : int, optional
            Alternative to threshold_pct. Defines the maximum absolute number of NaNs
            allowed in a column. Overrides threshold_pct if provided.

        return_report : bool, optional, default=False
            If True, also returns a dictionary summarizing
            the columns affected and number of rows removed.

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, dict]
            Cleaned DataFrame. If return_report=True, returns a tuple (DataFrame, report_dict).
        """
        # Compute threshold
        if threshold_abs is not None:
            threshold = threshold_abs / dataset.shape[0]
        else:
            threshold = threshold_pct

        result_df = dataset.copy()

        # Identify columns where NaN proportion is under the threshold
        nans = DataPreprocessor.get_missing(result_df)
        nan_ratio = nans["pct_of_nans"]
        nan_ratio = nan_ratio[(nan_ratio <= threshold) & (nan_ratio != 0)]
        nans_per_col = nans["count_of_nans"]

        # Drop rows with NaNs in those columns
        before_dropping = result_df.shape[0]
        result_df = result_df.dropna(subset=nan_ratio.index.to_series())
        after_dropping = result_df.shape[0]
        rows_removed = before_dropping - after_dropping


        if return_report:
            report = {
                "initial_lenght": dataset.shape[0],
                "final_lenght": result_df.shape[0],
                "columns_dropped_on": nan_ratio.index.to_series().reset_index(drop=True),
                "nans_per_column": nans_per_col,
                "rows_removed": rows_removed
            }
            return result_df, report

        return result_df

    @staticmethod
    def get_constant_features(dataset: pd.DataFrame,
                              quasi_constant_threshold: Optional[float] = 1.0,
                              include_nan: Optional[bool]=False
                              ) -> pd.DataFrame:
        """
        Identifies constant and quasi-constant features in the dataset.

        Parameters
        ----------
        dataset : pd.DataFrame
            Input DataFrame to analyze.
        quasi_constant_threshold : float, default=1.0
            Threshold for the proportion of the most frequent value.
            Columns with a ratio >= this threshold are flagged as (quasi-)constant.
        include_nan : bool, default=False
            Whether to include NaN values when computing the most frequent value.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by column names with:
            - 'is_const': bool flag if column is (quasi-)constant
            - 'top_value_ratio': proportion of the most frequent value
        """
        def top_ratio(series: pd.Series) -> float:
            s = series if include_nan else series.dropna()
            return s.value_counts().max() / s.size

        result = {"is_const": [], "top_value_ratio": []}
        indexes = []
        for col in dataset.columns:
            const = top_ratio(dataset[col])
            result["is_const"].append(const >= quasi_constant_threshold)
            result["top_value_ratio"].append(const)
            indexes.append(col)
        result = pd.DataFrame(result, index=indexes)
        return result

    @staticmethod
    def get_categories(dataset: pd.DataFrame,
                       threshold: Optional[int] = 30,
                       threshold_only: Optional[bool] = False,
                       include_number: Optional[bool] = False,
                       include_bool: Optional[bool] = False,
                       include_datetime: Optional[bool] = False,
                       sign_bin: Optional[bool] = False,
                       sign_const: Optional[bool] = False,
                       dropna: Optional[bool] = True
                       ) -> pd.DataFrame:
        """
        Identify categorical features in a DataFrame based on the number of unique values
        and the data type.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input dataset.
        threshold : int, optional, default=30
            Maximum number of unique values for a column to be considered categorical.
        threshold_only : bool, optional, default=False
            If True, categoricity is determined only by `threshold`, ignoring data types.
        include_number : bool, optional, default=False
            If True, numeric columns under the unique value threshold 
            will be treated as categorical.
        include_bool : bool, optional, default=False
            If True, boolean columns will be treated as categorical.
        include_datetime : bool, optional, default=False
            If True, datetime columns will be treated as categorical.
        sign_bin : bool, optional, default=False
            If True, add a flag `is_bin` for binary features (exactly two unique values).
        sign_const : bool, optional, default=False
            If True, add a flag `is_const` for constant features (only one unique value).
        dropna : bool, optional, default=True
            Whether to ignore NaN when counting unique values.

        Returns
        -------
        pd.DataFrame
            A DataFrame with:
            - `categories_count` : number of unique values in each column
            - `is_category` : boolean flag for categorical columns
            - optionally `is_bin` and `is_const` if `sign_bin` or `sign_const` are True.

        Notes
        -----
        - This method does not modify the original DataFrame.
        - Flexible logic: you can use only 
          the unique value threshold or combine it with dtype checks.
        """
        def _check_category(col, col_dtype):
            if col not in columns_to_set:
                return False
            if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
                return True
            if pd.api.types.is_bool_dtype(col_dtype):
                return include_bool
            if pd.api.types.is_numeric_dtype(col_dtype):
                return include_number
            if pd.api.types.is_datetime64_any_dtype(col_dtype):
                return include_datetime
            return False

        categories_count = dataset.nunique(dropna=dropna)
        result = pd.DataFrame({"categories_count": categories_count})
        columns_to_set = result[result["categories_count"] <= threshold].index
        if threshold_only:
            is_category = list(dataset.columns.isin(columns_to_set))
        else:
            is_category = [_check_category(col, dataset[col].dtype) for col in dataset.columns]
        result["is_category"] = is_category
        # mark nested bins & constants if required
        if sign_bin:
            result["is_bin"] = (result["is_category"]) & (result["categories_count"] == 2)
        if sign_const:
            result["is_const"] = (result["is_category"]) & (result["categories_count"] == 1)
        return result

    @staticmethod
    def set_categories(dataset: pd.DataFrame,
                       threshold: Optional[int] = 30,
                       include_bin: Optional[bool] = True,
                       include_number: Optional[bool] = False,
                       include_bool: Optional[bool] = False,
                       include_datetime: Optional[bool] = False,
                       ingnore_nans: Optional[bool] = False,
                       ) -> pd.DataFrame:
        """
        Convert eligible columns to Pandas `category` dtype for memory optimization
        and improved performance in certain operations.

        Parameters
        ----------
        dataset : pd.DataFrame
            The input dataset.
        threshold : int, optional, default=30
            Maximum number of unique values for a column to be considered categorical.
        include_bin : bool, optional, default=True
            Whether to include binary features (exactly two unique values) as categorical.
        include_number : bool, optional, default=False
            Whether to include numeric columns under the unique value threshold.
        include_bool : bool, optional, default=False
            Whether to include boolean columns as categorical.
        include_datetime : bool, optional, default=False
            Whether to include datetime columns as categorical.
        ingnore_nans : bool, optional, default=False
            Whether to ignore NaN when determining categoricity.

        Returns
        -------
        pd.DataFrame
            A copy of the original DataFrame with selected columns converted to `category` dtype.

        Notes
        -----
        - Converting to `category` can significantly reduce memory usage,
        especially for string/object columns with many repeated values.
        - `category` stores integer codes (`int8`/`int16`) and a category mapping,
        making comparisons and filtering faster than for `object` dtype.
        - For numeric columns, memory savings may be smaller, but grouping and filtering
        can still be faster.
        - The original DataFrame is not modified — a copy is returned.
        """
        categories = DataPreprocessor.get_categories(dataset=dataset,
                                                     threshold=threshold,
                                                     include_number=include_number,
                                                     include_bool=include_bool,
                                                     include_datetime=include_datetime,
                                                     sign_bin=True,
                                                     dropna=ingnore_nans)
        result_df = dataset.copy()
        if include_bin:
            columns_to_set = categories[categories["is_category"]].index
        else:
            columns_to_set = categories[categories["is_category"] & ~categories["is_bin"]].index
        result_df[columns_to_set] = result_df[columns_to_set].astype("category")
        return result_df
