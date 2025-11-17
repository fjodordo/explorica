import logging
import json

import pytest
import pandas as pd
import numpy as np

import explorica.data_quality as data_quality

# tests for data_quality.set_categorical()

@pytest.mark.parametrize("data, kwargs", [
    # 1D sequence
    ([1, 2, 3, 4, 5], {}),
    # 2D sequence  
    ([[1, 2], [3, 4], [5, 6]], {}),
    # Dict input
    ({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, {}),
    # With include parameters
    ([[1, 2], [3, 4]], {"include_int": True}),
    # With threshold
    ([1, 2, 3, 4, 5], {"threshold": 3}),
])
def test_set_categorical_different_sequences_and_dtypes(data, kwargs):
    result = data_quality.set_categorical(data, **kwargs)
    assert result is not None


@pytest.mark.parametrize("data, expected_columns, kwargs", 
    [
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'str_col': ['a', 'b', 'a', 'b'],
                'bin_col': [0, 1, 0, 1],
                'const_col': [1, 1, 1, 1]
            }),
            ['str_col'],
            {"threshold": 4}
        ),
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'str_col': ['a', 'b', 'a', 'b'],
                'other_int': [5, 6, 7, 8]
            }),
            ['int_col', 'other_int'],
            {"threshold": 4, "include_int": True}
        ),
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'bin_col': [0, 1, 0, 1],
                'str_bin': ['X', 'Y', 'X', 'Y']
            }),
            ['bin_col', 'str_bin'],
            {"threshold": 2, "include_bin": True, "include_int": False}
        ),
        (
            pd.DataFrame({
                'const_int': [1, 1, 1, 1],
                'const_str': ['A', 'A', 'A', 'A'],
                'var_col': [1, 2, 3, 4]
            }),
            ['const_int', 'const_str'],
            {"threshold": 1, "include_const": True}
        ),
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'str_col': ['a', 'b', 'a', 'b'],
                'bin_col': [0, 1, 0, 1],
                'const_col': [True, True, True, True]
            }),
            ['int_col', 'str_col', 'bin_col'],
            {"threshold": 4, "include_int": True, "include_str": True, "include_bin": True}
        ),
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'str_col': ['a', 'b', 'a', 'b'],
                'bin_col': [0, 1, 0, 1]
            }),
            ['int_col', 'str_col', 'bin_col'],
            {"threshold": 4, "include_all": True}
        ),
        (
            pd.DataFrame({
                'int_col': [1, 2, 3, 4],
                'str_col': ['a', 'b', 'a', 'b'],
                'bin_col': [0, 1, 0, 1]
            }),
            ['bin_col', 'str_col'],
            {"threshold": 4, "include": {"bin", "str"}, "include_int": True}
        ),
        (
            pd.DataFrame({
                'low_card': [1, 1, 2, 2],
                'high_card': [1, 2, 3, 4]
            }),
            ['low_card'],
            {"threshold": 3, "include_all": True}
        ),
        (
            pd.DataFrame({
                'bool_col': [True, False, True, False],
                'int_col': [1, 2, 3, 4]
            }),
            ['bool_col'],
            {"threshold": 2, "include_bool": True}
        ),
        (
            pd.DataFrame(
                [[1, 1, 1, 2, 2, 3], 
                 [4, 1, 2, 2, 3, 3],
                 [7, 1, 1, 2, 2, 3]],
                columns=pd.MultiIndex.from_tuples([
                    ('group_A', 'high_card'),
                    ('group_A', 'low_card'),
                    ('group_A', 'bin_col'),
                    ('group_B', 'high_card'),
                    ('group_B', 'str_col'),
                    ('group_B', 'const_col')
                ])
            ),
            [
                ('group_A', 'low_card'),
                ('group_B', 'high_card'),
                ('group_B', 'const_col')
            ],
            {"threshold": 1, "include_int": True}
        )

    ]
)
def test_set_categorical_include_behavior(data, expected_columns, kwargs):
    result = data_quality.set_categorical(data, **kwargs)
    for col in expected_columns:
        assert result[col].dtype.name == 'category', f"Column {col} should be categorical"

    other_columns = set(data.columns) - set(expected_columns)
    for col in other_columns:
        assert result[col].dtype.name != 'category', f"Column {col} should not be categorical"


@pytest.mark.parametrize("data, kwargs, expected_error", 
    [
        # TypeError для неверного threshold
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            {"threshold": "invalid_string"},
            ValueError
        ),
        (
            pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"]),
            {},
            ValueError
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            {"nan_policy": "invalid_policy"},
            ValueError
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            {"threshold": {"A": "not_a_number"}},
            ValueError
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}),
            {"threshold": [1, 2]},
            ValueError
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]}),
            {"include": {123, "str"}},
            TypeError
        ),
        (
            pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]}),
            {"include": {"invalid_type", "object"}},
            ValueError
        )
    ]
)
def test_set_categorical_error_conditions(data, kwargs, expected_error):
    """Test that set_categorical raises appropriate errors for invalid inputs."""
    with pytest.raises(expected_error):
        data_quality.set_categorical(data, **kwargs)


@pytest.mark.parametrize(
    "data, expected, kwargs",
    [
        (
            pd.DataFrame({
                'ints': [1, 2, 3, 4, 5],
                'strings': ['A', 'B', 'A', 'B', 'A'],
                'floats': [1.1, 2.2, 3.3, 4.4, 5.5]
            }),
            pd.DataFrame({
                'ints': [1, 2, 3, 4, 5],
                'strings': pd.Categorical(['A', 'B', 'A', 'B', 'A']),
                'floats': [1.1, 2.2, 3.3, 4.4, 5.5]
            }),
            {"threshold": 2, "include_str": True}
        ),
        (
            pd.DataFrame({
                'high_card': [1, 2, 3, 4, 5, 6, 7, 8],
                'low_card': [1, 1, 2, 2, 1, 2, 1, 2],
                'strings': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y']
            }),
            pd.DataFrame({
                'high_card': [1, 2, 3, 4, 5, 6, 7, 8],
                'low_card': pd.Categorical([1, 1, 2, 2, 1, 2, 1, 2]),
                'strings': pd.Categorical(['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y'])
            }),
            {"threshold": 3, "include_int": True, "include_str": True}
        ),
        (
            pd.DataFrame({
                'binary_int': [0, 1, 0, 1, 0, 1],
                'binary_str': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
                'normal_int': [1, 2, 3, 4, 5, 6]
            }),
            pd.DataFrame({
                'binary_int': pd.Categorical([0, 1, 0, 1, 0, 1]),
                'binary_str': pd.Categorical(['Yes', 'No', 'Yes', 'No', 'Yes', 'No']),
                'normal_int': [1, 2, 3, 4, 5, 6]
            }),
            {"threshold": 10, "include_bin": True}
        ),
        (
            pd.DataFrame({
                'constant': [5, 5, 5, 5, 5],
                'varying': [1, 2, 3, 4, 5],
                'strings': ['A', 'B', 'C', 'D', 'E']
            }),
            pd.DataFrame({
                'constant': pd.Categorical([5, 5, 5, 5, 5]),
                'varying': [1, 2, 3, 4, 5],
                'strings': pd.Categorical(['A', 'B', 'C', 'D', 'E'])
            }),
            {"threshold": 5, "include_const": True, "include_str": True}
        ),
        (
            pd.DataFrame({
                'ints': [1, 1, 2, 2],
                'floats': [1.0, 1.0, 2.0, 2.0],
                'strings': ['A', 'A', 'B', 'B']
            }),
            pd.DataFrame({
                'ints': pd.Categorical([1, 1, 2, 2]),
                'floats': pd.Categorical([1.0, 1.0, 2.0, 2.0]),
                'strings': pd.Categorical(['A', 'A', 'B', 'B'])
            }),
            {"threshold": 2, "include_all": True}
        ),
        (
            pd.DataFrame({
                'ints': [1, 2, 1, 2],
                'strings': ['A', 'B', 'A', 'B'],
                'floats': [1.1, 2.2, 3.3, 4.4]
            }),
            pd.DataFrame({
                'ints': pd.Categorical([1, 2, 1, 2]),
                'strings': ['A', 'B', 'A', 'B'],
                'floats': [1.1, 2.2, 3.3, 4.4]
            }),
            {"threshold": 2, "include": {"int"}, "include_str": True}
        ),
        (
            pd.DataFrame({
                'low': [1, 1, 2, 2, 1, 2],
                'medium': [1, 2, 3, 1, 2, 3],
                'high': [1, 2, 3, 4, 5, 6]
            }),
            pd.DataFrame({
                'low': pd.Categorical([1, 1, 2, 2, 1, 2]),
                'medium': pd.Categorical([1, 2, 3, 1, 2, 3]),
                'high': [1, 2, 3, 4, 5, 6]  # 6 unique > threshold=4
            }),
            {"threshold": 4, "include_all": True}
        ),
        (
            pd.DataFrame({
                'bools': [True, False, True, False],
                'ints': [1, 2, 3, 4],
                'strings': ['A', 'B', 'C', 'D']
            }),
            pd.DataFrame({
                'bools': pd.Categorical([True, False, True, False]),
                'ints': [1, 2, 3, 4],
                'strings': pd.Categorical(['A', 'B', 'C', 'D'])
            }),
            {"threshold": 4, "include_bool": True, "include_str": True}
        )
    ]
)
def test_set_categorical_example_based(data, expected, kwargs):
    """Test core functionality with specific input-output examples."""
    result = data_quality.set_categorical(data, **kwargs)

    pd.testing.assert_frame_equal(result, expected)
