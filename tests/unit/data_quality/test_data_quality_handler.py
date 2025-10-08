import json
import pytest
import pandas as pd
import numpy as np
import explorica.data_quality as data_quality

# tests for DataQualityHandler.get_summary()

summary_structure = {
        'nans': {'count_of_nans': {}, 'pct_of_nans': {}},
        'duplicates': {'count_of_unique': {}, 'pct_of_unique': {}, 'quasi_constant_pct': {}},
        'distribution': {'is_normal': {}, 'desc': {}, 'skewness': {}, 'kurtosis': {}},
        'stats': {'mean': {}, 'std': {}, 'median': {}, 'mode': {}, 'count_of_modes': {}},
        'multicollinearity': {'is_multicollinearity': {}, 'VIF': {}}
    }

def test_get_summary_input_contains_nans():
    df = pd.DataFrame({"1": [1, 2, 3, 4],
                       "2": [1, 2, 3, np.nan]})
    with pytest.raises(ValueError, match="The input 'dataset' contains null values"):
        data_quality.get_summary(df)

def test_get_summary_unsupported_return_method():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [9, 6, 5, 3, 1]})
    with pytest.raises(ValueError, match="Unsupported method 'pytorch.tensor'"):
        data_quality.get_summary(df, return_as="pytorch.tensor")

@pytest.mark.parametrize("dataset", [None, pd.DataFrame([]), np.array([])])
def test_get_summary_empty_dataset(dataset):
    with pytest.raises(ValueError, match="The input 'dataset' is empty"):
        data_quality.get_summary(dataset)

@pytest.mark.parametrize("dataset", [
    # --- pandas structures ---
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
    pd.Series([1, 2, 3]),
    pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
        "value": [10, 15, 20]
    }),            

    # --- numpy structures ---
    np.array([[1, 2, 3], [4, 5, 6]]),
    np.array([1, 2, 3]),
    np.rec.array([(1, 4), (2, 5), (3, 6)],
                 dtype=[('a', 'i4'), ('b', 'i4')]),

    # --- dict structures ---
    {"a": [1, 2, 3], "b": [4, 5, 6]},
    [{"a": 1, "b": 4}, {"a": 2, "b": 5}], 
])
def test_get_summary_different_inputs(dataset):
    assert data_quality.get_summary(dataset) is not None

def _assert_dict_summary_structure(summary: dict,
                                   summary_features: list,
                                   sample: dict=summary_structure):
    assert list(sample) == list(summary)

    # check that each group contains the expected features
    for i, val in sample.items():
        assert list(val) == list(sample[i])
    # check that no unexpected features are present
    for i, val in summary.items():
        assert list(val) == list(sample[i])
        # verify that each metric contains exactly the expected features
        for sub_val in val.values():
            assert list(sub_val) == summary_features

def test_get_summary_json_output_structure():
    # expected data
    expected_features = ["x1", "x2"]

    # returned data
    df = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [5, 3, 1, 6, 1]})
    result = data_quality.get_summary(df, return_as="dict")

    _assert_dict_summary_structure(result, expected_features, summary_structure)
        
def test_get_summary_df_output_structure():
    # expected data
    arrays = []
    for group, metrics in summary_structure.items():
        for metric in metrics:
            arrays.append((group, metric))
    columns = pd.MultiIndex.from_tuples(arrays)
    index = ["A", "B"]
    df_exp = pd.DataFrame(np.random.rand(len(index), len(columns)), index=index, columns=columns)
    df_exp[('distribution', 'desc')] = df_exp[('distribution', 'desc')].astype(object)

    # returned data
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 3, 1, 6, 1]})
    summary = data_quality.get_summary(df, return_as="df")

    # because summary['stats']['mode'] might be number or not
    assert (summary.columns == df_exp.columns).all() & (summary.index == df_exp.index).all()
    try:
        assert (summary.dtypes == df_exp.dtypes).all()
    except AssertionError:
        df_exp[('stats', 'mode')] = df_exp[('stats', 'mode')].astype(object)
        assert (summary.dtypes == df_exp.dtypes).all()

def _assert_df_structures_equality(df1: pd.DataFrame, df2: pd.DataFrame, check_dtypes=True):
    if check_dtypes:
        assert (df1.columns == df2.columns).all() and (
                df1.index == df2.index).all() and (df1.dtypes == df2.dtypes).all()
    else:
        assert (df1.columns == df2.columns).all() and (
                df1.index == df2.index).all()

@pytest.mark.parametrize("ext, loader", [(".csv", lambda path: pd.read_csv(path, index_col=[0])),
                                         (".xlsx", lambda path: pd.read_excel(path, header=[0, 1], index_col=[0])),
                                         (".json", lambda path: json.load(open(path)))])
def test_get_summary_file_output(ext, loader, tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
    path = str(tmp_path / f"summary{ext}")
    data_quality.get_summary(dataset=df, directory=path)
    summary = loader(path)
    if ext == ".csv":
        expected_df = pd.DataFrame(columns=['nans:count_of_nans', 'nans:pct_of_nans', 'duplicates:count_of_unique',
       'duplicates:pct_of_unique', 'duplicates:quasi_constant_pct',
       'distribution:is_normal', 'distribution:desc', 'distribution:skewness',
       'distribution:kurtosis', 'stats:mean', 'stats:std', 'stats:median',
       'stats:mode', 'stats:count_of_modes',
       'multicollinearity:is_multicollinearity', 'multicollinearity:VIF'], index=["A", "B"])
        for col, dtype in zip(expected_df.columns, ["float64"] * 6 + ["object"] + ["float64"] * 9):
            expected_df[col] = expected_df[col].astype(dtype)
        try:
            _assert_df_structures_equality(expected_df, summary)
        except AssertionError:
            expected_df["stats:mode"] = expected_df["stats:mode"].astype("object")
            _assert_df_structures_equality(expected_df, summary)
        _assert_df_structures_equality(expected_df, summary)
    elif ext == ".xlsx":
        expected_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
            (             'nans',        'count_of_nans'),
            (             'nans',          'pct_of_nans'),
            (       'duplicates',      'count_of_unique'),
            (       'duplicates',        'pct_of_unique'),
            (       'duplicates',   'quasi_constant_pct'),
            (     'distribution',            'is_normal'),
            (     'distribution',                 'desc'),
            (     'distribution',             'skewness'),
            (     'distribution',             'kurtosis'),
            (            'stats',                 'mean'),
            (            'stats',                  'std'),
            (            'stats',               'median'),
            (            'stats',                 'mode'),
            (            'stats',       'count_of_modes'),
            ('multicollinearity', 'is_multicollinearity'),
            ('multicollinearity',                  'VIF')]),
            index=["A", "B"])
        for col, dtype in zip(expected_df.columns, ["float64"] * 6 + ["object"] + ["float64"] * 9):
            expected_df[col] = expected_df[col].astype(dtype)
        _assert_df_structures_equality(expected_df, summary, check_dtypes=False)
    elif ext == ".json":
        _assert_dict_summary_structure(summary, summary_features=["A", "B"], sample=summary_structure)
