from collections import deque

import pytest
import pandas as pd
import numpy as np

from explorica._utils import (convert_params_for_keys, convert_dataframe,
                              convert_numpy, convert_dict)

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