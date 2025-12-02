from collections import deque

import pytest
import pandas as pd
import numpy as np

from explorica._utils import (convert_params_for_keys, convert_dataframe,
                              convert_numpy, convert_dict, convert_series)

ERR_MSG_MULTIDIMENSIONAL_DATA = "Input data must be 1-dimensional, but contains {} features. Please, provide a single column/sequence."

# tests for ConvertUtils.convert_params_for_keys()

def test_convert_params_for_keys_empty_input():
    """If `arg` is empty and keys are not, raises ValueError."""
    keys = ["A", "B", "C"]
    with pytest.raises(ValueError):
        convert_params_for_keys([], keys)

def test_convert_params_for_keys_lengths_mismatch():
    """Sequence length not matching `keys` should raise ValueError."""
    arg = [1, 2, 3]
    keys = ["A", "B", "C", "D"]
    with pytest.raises(ValueError):
        convert_params_for_keys(arg, keys)

def test_convert_params_for_keys_same_column_names():
    """
    DataFrame with duplicate column names should raise ValueError,
    since keys cannot be uniquely mapped.
    """
    df = pd.DataFrame([[1, 2, 3]], columns=["A", "A", "B"])
    keys = ["A", "A", "C"]
    with pytest.raises(ValueError, match="must be unique"):
        convert_params_for_keys(df, keys)

def test_convert_params_for_keys_unsupported_arg():
    """Unsupported argument type should raise TypeError."""
    class CustomObj:
        pass

    keys = ["A", "B"]
    with pytest.raises(TypeError):
        convert_params_for_keys(CustomObj(), keys)


def test_convert_params_for_keys_example_based():
    """General example test for various input types."""
    keys = ["A", "B", "C"]

    # Single value (broadcast)
    result = convert_params_for_keys(10, keys)
    assert all(v == 10 for v in result.values())

    # Dict or mapping
    arg = {"A": 1, "B": 2, "C": 3}
    result = convert_params_for_keys(arg, keys)
    assert result == arg

    # Series
    s = pd.Series([1, 2, 3], name="A")
    result = convert_params_for_keys(s, keys)
    assert result == {"A": 1, "B": 2, "C": 3}

    # Sequence (matching length)
    seq = [1, 2, 3]
    result = convert_params_for_keys(seq, keys)
    assert result == {"A": 1, "B": 2, "C": 3}

    # DataFrame
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    result = convert_params_for_keys(df, keys)
    assert result["A"] == [1, 2]
    assert result["C"] == [5, 6]

# tests for ConvertUtils.convert_dateframe

one_dim_list = [1, 2, 3]
two_dim_list = [[1, 2], [3, 4]]
three_dim_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
dict_input = {'a': [1, 2], 'b': [3, 4]}
pd_series = pd.Series([1, 2, 3])
pd_df = pd.DataFrame({'a': [1], 'b': [3]})
np_array_1d = np.array([1, 2, 3])
np_array_2d = np.array([[1, 2], [3, 4]])
np_array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
two_dim_list_mixed = [np.array([1, 2, 3, 4, 5]),
                      [1, 2.0, 3, 4, 5],
                      pd.Series([1, 2, 3, 4, 5]),
                      pd.Categorical(["a", "a", "a", "a", "b"], categories=["a", "b"]),
                      (1, 2, 3, 4, 5)]
dict_mixed = {1: np.array([1, 2, 3, 4, 5]),
              2: [1, 2.0, 3, 4, 5],
              3: pd.Series([1, 2, 3, 4, 5]),
              4: pd.Categorical(["a", "a", "a", "a", "b"], categories=["a", "b"]),
              5: (1, 2, 3, 4, 5)}

@pytest.mark.parametrize(
    "input_data",
    [
        one_dim_list,
        two_dim_list,
        three_dim_list,
        dict_input,
        pd_series,
        pd_df,
        np_array_1d,
        np_array_2d,
        two_dim_list_mixed,
        dict_mixed
    ]
)
def test_convert_dataframe_various_inputs(input_data):
    df = convert_dataframe(input_data)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_pd_df_behavior():
    df = convert_dataframe(pd_df)
    pd.testing.assert_frame_equal(df, pd_df)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_pd_series_behavior():
    df = convert_dataframe(pd_series)
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_np_1d_behavior():
    df = convert_dataframe(np_array_1d)
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_np_2d_behavior():
    df = convert_dataframe(np_array_2d)
    assert df.shape[1] != 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_dict_behavior():
    df = convert_dataframe(dict_input)
    assert df.shape[1] != 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_empty_seq_behavior():
    empty_input = []
    df = convert_dataframe(empty_input)
    pd.testing.assert_frame_equal(df, pd.DataFrame())

def test_convert_dataframe_1x1_list_behavior():
    df = convert_dataframe([[1]])
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_df_multiindex_behavior():
    arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
    index = pd.MultiIndex.from_arrays(arrays)
    multi_df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=index)
    df = convert_dataframe(multi_df)
    pd.testing.assert_frame_equal(df, multi_df)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_bigger_ndim():
    with pytest.raises(ValueError, match="dimensional"):
        convert_dataframe(np_array_3d)

def test_convert_dataframe_nested_deque():
    df = convert_dataframe(deque([deque([1, 2, 3]),
                                         deque([2, 3, 4]),
                                         deque([3, 4, 5])]))
    assert df.shape[1] != 1
    assert isinstance(df, pd.DataFrame)


# tests for ConvertUtils.convert_numpy

@pytest.mark.parametrize(
    "input_data",
    [
        one_dim_list,
        two_dim_list,
        three_dim_list,
        dict_input,
        pd_series,
        pd_df,
        np_array_1d,
        np_array_2d,
    ]
)
def test_convert_numpy_various_inputs(input_data):
    arr = convert_numpy(input_data)
    assert isinstance(arr, np.ndarray)

def test_convert_numpy_empty_seq_behavior():
    empty_input = []
    arr = convert_numpy(empty_input)
    assert len(arr) == 0

def test_convert_numpy_lens_mismatch():
    dct = {"a": [42], "b": [1, 2]}
    with pytest.raises(ValueError):
        convert_numpy(dct)


# tests for ConvertUtils.convert_dict

@pytest.mark.parametrize(
    "input_data",
    [
        one_dim_list,
        two_dim_list,
        three_dim_list,
        dict_input,
        pd_series,
        pd_df,
        np_array_1d,
        np_array_2d,
    ]
)
def test_convert_dict_various_inputs(input_data):
    arr = convert_dict(input_data)
    assert isinstance(arr, dict)

# tests for _utils.convert_series()

def mock_convert_dict(data) -> dict:
    """
    Simulated behavior of the convert_dict utility.
    
    The real utility is expected to standardize various inputs (list, pd.Series, etc.)
    into a single-key dictionary for single-feature inputs, or a multi-key
    dictionary for multi-feature inputs (which should fail validation).
    """
    if data == [10, 20, 30]:
        # Typical case: list converted to a single-key dictionary
        return {0: [10, 20, 30]}
    elif data == {'category': ['A', 'B', 'C']}:
        # Typical case: single-key dict passes through
        return data
    elif data == []:
        # Empty list case
        return {}
    elif data == {'feature_a': [1, 2], 'feature_b': [3, 4]}:
        # Failure case: Multi-dimensional input
        return data
    elif data == {}:
        # Empty dict case
        return {}
    elif data == {'data_key': [100, 200], 'other_key': []}:
        return {'data_key': [100, 200], 'other_key': []}
    # Fallback for unexpected inputs during testing
    return {"0": data}

def test_convert_series_successful_conversion_from_list(mocker):
    """Test standard list input is converted and assigned a default name."""
    input_data = [10, 20, 30]
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_data = pd.Series([10, 20, 30], name=0)
    
    result = convert_series(input_data)
    
    pd.testing.assert_series_equal(result, expected_data, check_dtype=False)
    assert result.name == 0

def test_convert_series_successful_conversion_from_single_key_dict(mocker):
    """Test dictionary input is converted and retains the key as name."""
    input_data = {'category': ['A', 'B', 'C']}
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_data = pd.Series(['A', 'B', 'C'], name="category")
    
    result = convert_series(input_data)
    
    pd.testing.assert_series_equal(result, expected_data, check_dtype=False)
    assert result.name == "category"

def test_convert_series_empty_list_input(mocker):
    """Test that an empty list returns an empty Pandas Series."""
    input_data = []
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_data = pd.Series([], dtype='object') # pandas often defaults to object for empty Series
    
    result = convert_series(input_data)
    
    pd.testing.assert_series_equal(result, expected_data, check_names=True)
    assert result.empty is True

def test_convert_series_empty_dict_input(mocker):
    """Test that an empty dictionary returns an empty Pandas Series."""
    input_data = {}
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_data = pd.Series([], dtype='object')
    
    result = convert_series(input_data)
    
    pd.testing.assert_series_equal(result, expected_data, check_names=True)
    assert result.empty is True

def test_convert_series_raises_value_error_for_multidimensional_dict(mocker):
    """Test that input with multiple keys/dimensions raises a ValueError."""
    input_data = {'feature_a': [1, 2], 'feature_b': [3, 4]}
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_error_message = ERR_MSG_MULTIDIMENSIONAL_DATA.format(2)
    
    with pytest.raises(ValueError) as excinfo:
        convert_series(input_data)
        
    assert expected_error_message in str(excinfo.value)
    
def test_convert_series_raises_value_error_for_another_multidimensional_dict(mocker):
    """Test a different multi-dimensional case to ensure robustness."""
    input_data = {'data_key': [100, 200], 'other_key': []}
    mocker.patch("explorica._utils.conversion.convert_dict",
        return_value=mock_convert_dict(input_data))
    expected_error_message = ERR_MSG_MULTIDIMENSIONAL_DATA.format(2)
    
    with pytest.raises(ValueError) as excinfo:
        convert_series(input_data)
    
    assert expected_error_message in str(excinfo.value)