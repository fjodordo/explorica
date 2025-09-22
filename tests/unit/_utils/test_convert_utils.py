from collections import deque

import pytest
import pandas as pd
import numpy as np

from explorica._utils import ConvertUtils as cutils

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
    df = cutils.convert_dataframe(input_data)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_pd_df_behavior():
    df = cutils.convert_dataframe(pd_df)
    pd.testing.assert_frame_equal(df, pd_df)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_pd_series_behavior():
    df = cutils.convert_dataframe(pd_series)
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_np_1d_behavior():
    df = cutils.convert_dataframe(np_array_1d)
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_np_2d_behavior():
    df = cutils.convert_dataframe(np_array_2d)
    assert df.shape[1] != 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_dict_behavior():
    df = cutils.convert_dataframe(dict_input)
    assert df.shape[1] != 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_empty_seq_behavior():
    empty_input = []
    df = cutils.convert_dataframe(empty_input)
    pd.testing.assert_frame_equal(df, pd.DataFrame())

def test_convert_dataframe_1x1_list_behavior():
    df = cutils.convert_dataframe([[1]])
    assert df.shape[1] == 1
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_df_multiindex_behavior():
    arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
    index = pd.MultiIndex.from_arrays(arrays)
    multi_df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=index)
    df = cutils.convert_dataframe(multi_df)
    pd.testing.assert_frame_equal(df, multi_df)
    assert isinstance(df, pd.DataFrame)

def test_convert_dataframe_bigger_ndim():
    with pytest.raises(ValueError, match="dimensional"):
        cutils.convert_dataframe(np_array_3d)

def test_convert_dataframe_nested_deque():
    df = cutils.convert_dataframe(deque([deque([1, 2, 3]),
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
    arr = cutils.convert_numpy(input_data)
    assert isinstance(arr, np.ndarray)

def test_convert_numpy_empty_seq_behavior():
    empty_input = []
    arr = cutils.convert_numpy(empty_input)
    assert len(arr) == 0

def test_convert_numpy_lens_mismatch():
    dct = {"a": [42], "b": [1, 2]}
    with pytest.raises(ValueError):
        cutils.convert_numpy(dct)


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
    arr = cutils.convert_dict(input_data)
    assert isinstance(arr, dict)