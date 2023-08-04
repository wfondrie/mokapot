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


@pytest.fixture
def psms():
    """Create a standard psms iterable"""
    yield [
        ["1", "1", "HLAQLLR", "-5.75", "0.108", "1.0", "_.dummy._\n"],
        ["2", "0", "HLAQLLR", "-5.81", "0.109", "1.0", "_.dummy._\n"],
        ["3", "0", "NVPTSLLK", "-5.83", "0.11", "1.0", "_.dummy._\n"],
        ["4", "1", "QILVQLR", "-5.92", "0.12", "1.0", "_.dummy._\n"],
        ["5", "1", "HLAQLLR", "-6.05", "0.13", "1.0", "_.dummy._\n"],
        ["6", "0", "QILVQLR", "-6.06", "0.14", "1.0", "_.dummy._\n"],
        ["7", "1", "SRTSVIPGPK", "-6.12", "0.15", "1.0", "_.dummy._\n"],
    ]


def test_groupby_max(df):
    """Test that the groupby_max() function works"""
    in_df, val1_max = df

    # Verify that the basic idea works:
    idx = utils.groupby_max(in_df, "group", "val1", 42)
    out_df = in_df.loc[idx, :].sort_values("group").reset_index(drop=True)
    pd.testing.assert_frame_equal(val1_max, out_df)

    idx = utils.groupby_max(in_df, ["group"], "val1", 42)
    out_df = in_df.loc[idx, :].sort_values("group").reset_index(drop=True)
    pd.testing.assert_frame_equal(val1_max, out_df)

    # Verify that the shuffling bit works:
    idx1 = set(utils.groupby_max(in_df, "group", "val2", 2))
    idx2 = set(utils.groupby_max(in_df, "group", "val2", 1))
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


def test_get_unique_psms_and_peptides(tmp_path, psms):
    psms_iterable = psms
    out_peptides = tmp_path / "peptides.csv"
    with open(out_peptides, "w") as f:
        f.write("PSMId\tLabel\tPeptide\tscore\tproteinIds\n")
    utils.get_unique_peptides_from_psms(
        iterable=psms_iterable,
        peptide_col_index=2,
        out_peptides=out_peptides,
        sep="\t",
    )

    expected_output = pd.DataFrame(
        [
            [1, 1, "HLAQLLR", -5.75, "_.dummy._"],
            [3, 0, "NVPTSLLK", -5.83, "_.dummy._"],
            [4, 1, "QILVQLR", -5.92, "_.dummy._"],
            [7, 1, "SRTSVIPGPK", -6.12, "_.dummy._"],
        ],
        columns=["PSMId", "Label", "Peptide", "score", "proteinIds"],
    )

    output = pd.read_csv(out_peptides, sep="\t")
    pd.testing.assert_frame_equal(expected_output, output)
