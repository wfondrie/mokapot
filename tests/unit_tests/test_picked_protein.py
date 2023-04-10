"""Test the picked protein approach functions"""
import pandas as pd

from mokapot.picked_protein import strip_peptides


def test_strip_peptides():
    """Test removing modifications from a peptide sequence"""
    in_df = pd.Series(["A.B.C", "nABCc", "BL[+mod]AH"])
    expected = pd.Series(["B", "ABC", "BLAH"])
    out_df = strip_peptides(in_df)
    pd.testing.assert_series_equal(out_df, expected)

    in_df = pd.Series(["abc"])
    expected = pd.Series(["ABC"])
    out_df = strip_peptides(in_df)
    pd.testing.assert_series_equal(out_df, expected)
