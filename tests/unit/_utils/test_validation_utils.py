import pytest
import pandas as pd
import numpy as np

from explorica._utils import (
    validate_string_flag, validate_at_least_one_exist,
    validate_lengths_match, validate_array_not_contains_nan,
    validate_unique_column_names)


# tests for ValidationUtils.validate_string_flag

def test_validate_string_flag_positive_case():
    validate_string_flag(arg="A", supported_values=["A", "B", "C"],
                                err_msg="my_error_message")
    
def test_validate_string_flag_negative_case():
    with pytest.raises(ValueError, match="my_error_message"):
        validate_string_flag(arg="D", supported_values=["A", "B", "C"],
                                    err_msg="my_error_message")

# test for ValidationUtils.validate_at_least_one_exist

def test_validate_at_least_one_exist_positive_case():
    validate_at_least_one_exist(values=[None, None, "exists"],
                                       err_msg="my_error_message")

def test_validate_at_least_one_exist_negative_case():
    with pytest.raises(ValueError, match="my_error_message"):
        validate_at_least_one_exist(values=[None, None, None],
                                           err_msg="my_error_message")

# test for ValidationUtils.validate_lenghts_match

def test_validate_lenghts_match_positive_case():
    validate_lengths_match(array1=[1, 2, 3], array2=[3, 4, 5],
                                  err_msg="my_error_message")

def test_validate_lenghts_match_negative_case():
    with pytest.raises(ValueError, match="my_error_message"):
        validate_lengths_match(array1=[1, 2, 3], array2=[4, 5],
                                    err_msg="my_error_message")
        
# tests for ValidationUtils.validate_array_not_contains_nan

def test_validate_array_not_contains_nan_positive_case():
    validate_array_not_contains_nan([1, 2, 3, 4, 5],
                                           err_msg="my_error_message")

def test_validate_array_not_contains_nan_negative_case():
    with pytest.raises(ValueError, match="my_error_message"):
        validate_array_not_contains_nan([1, 2, np.nan, 4, 5],
                                    err_msg="my_error_message")

# tests for ValidationUtils.validate_unique_column_names

def validate_unique_column_names_positive_case():
    df = pd.DataFrame([[1, 2, 3, 4, 5],
                       [1, 2, 3, 4, 5]],
                       columns=["col1", "col2", "col3", "col4", "col5"])
    validate_unique_column_names(df, err_msg="my_error_message")

def validate_unique_column_names_negative_case():
    df = pd.DataFrame([[1, 2, 3, 4, 5],
                       [1, 2, 3, 4, 5]],
                       columns=["col1", "col2", "col2", "col4", "col5"])
    with pytest.raises(ValueError, match="my_error_message"):
        validate_unique_column_names(df, err_msg="my_error_message")
