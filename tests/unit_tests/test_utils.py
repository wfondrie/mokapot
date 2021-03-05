"""Test the utility functions"""
import pytest
import numpy as np
import pandas as pd
from mokapot import utils


@pytest.fixture
def df():
    """Create a simple dataframe."""
    data = {
        "val1": [1, 2, 2, 1, 1, 1, 3, 2, 1],
        "val2": [1] * 9,
        "group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
    }

    max_res = {"val1": [2, 1, 3], "val2": [1, 1, 1], "group": ["A", "B", "C"]}

    return pd.DataFrame(data), pd.DataFrame(max_res)


def test_groupby_max(df):
    """Test that the groupby_max() function works"""
    np.random.seed(42)
    in_df, val1_max = df

    # Verify that the basic idea works:
    idx = utils.groupby_max(in_df, "group", "val1")
    out_df = in_df.loc[idx, :].sort_values("group").reset_index(drop=True)
    pd.testing.assert_frame_equal(val1_max, out_df)

    idx = utils.groupby_max(in_df, ["group"], "val1")
    out_df = in_df.loc[idx, :].sort_values("group").reset_index(drop=True)
    pd.testing.assert_frame_equal(val1_max, out_df)

    # Verify that the shuffling bit works:
    idx1 = set(utils.groupby_max(in_df, "group", "val2"))
    idx2 = set(utils.groupby_max(in_df, "group", "val2"))
    assert idx1 != idx2


def test_flatten():
    """Test that the flatten() function works"""
    nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert utils.flatten(nested_list) == flat_list

    nested_array = np.array(nested_list)
    flat_array = np.array(flat_list)
    np.testing.assert_array_equal(
        np.array(utils.flatten(nested_array)), flat_array
    )


def test_safe_divide():
    """Test that our safe division function works"""
    np_num = np.array([1, 2, 3, 4])
    np_den = np.array([1, 2, 3, 0.0])
    np_out = np.array([1, 1, 1, 0])
    np.testing.assert_array_equal(utils.safe_divide(np_num, np_den), np_out)

    np_out_ones = np.array([1, 1, 1, 1])
    np.testing.assert_array_equal(
        utils.safe_divide(np_num, np_den, ones=True), np_out_ones
    )

    # Test pandas:
    df = pd.DataFrame({"num": np_num, "den": np_den})
    np.testing.assert_array_equal(utils.safe_divide(df.num, df.den), np_out)

    np.testing.assert_array_equal(
        utils.safe_divide(df.num, df.den, ones=True), np_out_ones
    )


def test_tuplize():
    """Test that we can turn things into tuples"""
    list_in = [1, 2, 3]
    list_out = (1, 2, 3)
    assert utils.tuplize(list_in) == list_out

    str_in = "blah"
    str_out = ("blah",)
    assert utils.tuplize(str_in) == str_out

    tuple_in = ("blah", 1, "x")
    tuple_out = ("blah", 1, "x")
    assert utils.tuplize(tuple_in) == tuple_out
