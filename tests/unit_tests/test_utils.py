"""Test the utility functions"""
import os

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
def psms_iterator():
    """Create a standard psms iterable"""
    return [
        ["1", "1", "HLAQLLR", "-5.75", "0.108", "1.0", "_.dummy._\n"],
        ["2", "0", "HLAQLLR", "-5.81", "0.109", "1.0", "_.dummy._\n"],
        ["3", "0", "NVPTSLLK", "-5.83", "0.11", "1.0", "_.dummy._\n"],
        ["4", "1", "QILVQLR", "-5.92", "0.12", "1.0", "_.dummy._\n"],
        ["5", "1", "HLAQLLR", "-6.05", "0.13", "1.0", "_.dummy._\n"],
        ["6", "0", "QILVQLR", "-6.06", "0.14", "1.0", "_.dummy._\n"],
        ["7", "1", "SRTSVIPGPK", "-6.12", "0.15", "1.0", "_.dummy._\n"],
    ]


@pytest.fixture
def peptide_csv_file(tmp_path):
    file = tmp_path / "peptides.csv"
    with open(file, "w") as f:
        f.write("PSMId\tLabel\tPeptide\tscore\tproteinIds\n")
    yield file
    os.unlink(file)


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


def test_create_chunks():
    # Case 1: Chunk size is less than data length
    assert utils.create_chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    # Case 2: Chunk size is equal to data length
    assert utils.create_chunks([1, 2, 3, 4, 5], 5) == [[1, 2, 3, 4, 5]]

    # Case 3: Chunk size is greater than data length
    assert utils.create_chunks([1, 2, 3, 4, 5], 10) == [[1, 2, 3, 4, 5]]

    # Case 4: Chunk size is 1
    assert utils.create_chunks([1, 2, 3, 4, 5], 1) == [[1], [2], [3], [4], [5]]

    # Case 6: Empty data array, chunk size doesn't matter
    assert utils.create_chunks([], 3) == []


def test_get_unique_psms_and_peptides(peptide_csv_file, psms_iterator):
    psms_iterator = psms_iterator
    utils.get_unique_peptides_from_psms(
        iterable=psms_iterator,
        peptide_col_index=3,
        out_peptides=peptide_csv_file,
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

    output = pd.read_csv(peptide_csv_file, sep="\t")
    pd.testing.assert_frame_equal(expected_output, output)


def test_convert_targets_column(psms_iterator):
    df = pd.DataFrame(psms_iterator,
                      columns=["PSMId", "Label", "Peptide", "score", "q-value",
                               "posterior_error_prob", "proteinIds"])
    labels = df["Label"].astype(int)
    expect = [True, False, False, True, True, False, True]

    # Test with values in [0, 1] as strings
    df_out = utils.convert_targets_column(df, "Label")
    assert df_out is df  # check that returned and original df are the same
    assert df["Label"].dtype == "bool"
    assert (df["Label"] == expect).all()

    # Test with values in [0, 1]
    df["Label"] = labels
    utils.convert_targets_column(df, "Label")
    assert df["Label"].dtype == "bool"
    assert (df["Label"] == expect).all()

    # Test with values in [-1, 1]
    labels[labels == 0] = -1
    df["Label"] = labels
    utils.convert_targets_column(df, "Label")
    assert df["Label"].dtype == "bool"
    assert (df["Label"] == expect).all()

    # Test with values in [-1, 1] as strings
    df["Label"] = labels.astype(str)
    utils.convert_targets_column(df, "Label")
    assert df["Label"].dtype == "bool"
    assert (df["Label"] == expect).all()

    # Test with bool values (should be idempotent)
    df["Label"] = (labels == 1)
    utils.convert_targets_column(df, "Label")
    assert df["Label"].dtype == "bool"
    assert (df["Label"] == expect).all()

    # Junk in the target column should raise a ValueError
    df["Label"] = labels + 3
    with pytest.raises(ValueError):
        utils.convert_targets_column(df, "Label")
