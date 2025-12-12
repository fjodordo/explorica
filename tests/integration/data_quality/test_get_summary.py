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

def test_get_summary_nan_policy():
    df = pd.DataFrame({"1": [1, 2, 3, 4],
                       "2": [1, 2, 3, np.nan]})
    report = data_quality.get_summary(df, nan_policy = "drop")
    assert isinstance(report, pd.DataFrame)
    report = data_quality.get_summary(df, nan_policy = "include")
    assert isinstance(report, pd.DataFrame)
    with pytest.raises(ValueError):
        data_quality.get_summary(df, nan_policy = "raise")

def test_get_summary_unsupported_return_method():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [9, 6, 5, 3, 1]})
    with pytest.raises(ValueError, match="Unsupported method 'pytorch.tensor'"):
        data_quality.get_summary(df, return_as="pytorch.tensor")

@pytest.mark.parametrize("dataset", [None, pd.DataFrame([]), np.array([])])
def test_get_summary_empty_dataset(dataset):
    summary = data_quality.get_summary(dataset)
    expected_structure = {
        "nans": [
            "count_of_nans",
            "pct_of_nans",
        ],
        "duplicates": [
            "count_of_unique",
            "pct_of_unique",
            "quasi_constant_pct",
        ],
        "distribution": [
            "is_normal",
            "desc",
            "skewness",
            "kurtosis",
        ],
        "stats": [
            "mean",
            "std",
            "median",
            "mode",
            "count_of_modes",
        ],
        "multicollinearity": [
            "is_multicollinearity",
            "VIF",
        ],
    }

    expected_columns = pd.MultiIndex.from_tuples(
        [(section, metric) 
         for section, metrics in expected_structure.items()
         for metric in metrics]
    )
    assert list(summary.columns) == list(expected_columns)
    assert summary.empty

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
        
def test_get_summary_df_structure():
    df = pd.DataFrame({"A": [1,2,3], "B":[3,2,1]})
    summary = data_quality.get_summary(df, return_as="df")

    # 1. Check column MultiIndex
    assert isinstance(summary.columns, pd.MultiIndex)

    # 2. Check, that all groups exist
    expected_groups = set(summary_structure.keys())
    assert set(summary.columns.get_level_values(0)) == expected_groups

    # 3. Check, that all metrics in each group exist
    for group, metrics in summary_structure.items():
        cols = summary.columns[summary.columns.get_level_values(0) == group]
        assert set(cols.get_level_values(1)) == set(metrics)

    # 4. Check, that index is correct (column names correspond source df)
    assert list(summary.index) == ["A","B"]


@pytest.mark.parametrize("ext, loader", [
    (".csv", lambda path: pd.read_csv(path, header=[0, 1], index_col=0)),
    (".xlsx", lambda path: pd.read_excel(path, header=[0, 1], index_col=0)),
    (".json", lambda path: json.load(open(path)))
])
def test_get_summary_file_output(ext, loader, tmp_path):
    df = pd.DataFrame({"A": [1,2,3], "B": [3,2,1]})
    path = tmp_path / f"summary{ext}"

    data_quality.get_summary(df, directory=str(path))
    summary = loader(path)

    # 1. Check index always
    if ext != ".json":
        assert list(summary.index) == ["A", "B"]

    # 2. Check CSV
    if ext == ".csv":
        columns = summary.columns.tolist()
        expected = [(g, m) for g, metrics in summary_structure.items() for m in metrics]

        pass
        assert columns == expected

    # 3. Check XLSX
    elif ext == ".xlsx":
        assert isinstance(summary.columns, pd.MultiIndex)
        assert set(summary.columns.get_level_values(0)) == set(summary_structure.keys())
        for group, metrics in summary_structure.items():
            col_metrics = summary.columns[summary.columns.get_level_values(0) == group]
            assert set(col_metrics.get_level_values(1)) == set(metrics)

    # 4. Check JSON
    elif ext == ".json":
        summary_dict = summary
        expected_sections = set(summary_structure.keys())
        expected_metrics = {sec: set(metrics) for sec, metrics in summary_structure.items()}
        assert set(summary_dict.keys()) == expected_sections
        for sec, metrics in expected_metrics.items():
            assert set(summary_dict[sec].keys()) == metrics
            for metric, values in summary_dict[sec].items():
                assert set(values.keys()) == {"A", "B"}
