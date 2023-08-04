"""Test that FlashLFQ export is working"""

import pytest
import mokapot
import numpy as np
import pandas as pd


def test_basic(mock_conf, tmp_path):
    """Test that the basic output works"""
    conf = mock_conf
    df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
    expected = pd.DataFrame(
        {
            "File Name": ["c.mzML"] * 2,
            "Base Sequence": ["ABCDXYZ", "ABCDEFG"],
            "Full Sequence": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
            "Peptide Monoisotopic Mass": [1, 2],
            "Scan Retention Time": [1.0, 2.0],
            "Precursor Charge": [2, 3],
            "Protein Accession": ["A|B|C; B|C|A", "A|B|C"],
        }
    )

    pd.testing.assert_frame_equal(df, expected)


def test_with_missing(mock_conf, tmp_path):
    """Test that missing columns causes errors"""
    conf = mock_conf
    cols = conf._optional_columns.copy()
    for col in ["filename", "calcmass", "rt", "charge"]:
        new_cols = cols.copy()
        new_cols[col] = None
        conf._optional_columns = new_cols
        with pytest.raises(ValueError):
            mokapot.to_flashlfq(conf, tmp_path / "test.txt")


def test_no_proteins(mock_conf, tmp_path):
    """Test when no proteins are available"""
    conf = mock_conf
    conf._protein_column = None
    df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
    expected = pd.Series([np.nan, np.nan], name="Protein Accession")
    pd.testing.assert_series_equal(df["Protein Accession"], expected)


def test_fasta_proteins(mock_conf, mock_proteins, tmp_path):
    """Test that using mokapot protein groups works"""
    conf = mock_conf
    conf._proteins = mock_proteins
    conf._has_proteins = True
    df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
    expected = pd.Series(["X|Y|Z", "A|B|C; X|Y|Z"], name="Protein Accession")
    pd.testing.assert_series_equal(df["Protein Accession"], expected)

    conf._proteins.shared_peptides = {}
    df = pd.read_table(mokapot.to_flashlfq(conf, tmp_path / "test.txt"))
    expected = pd.Series(["X|Y|Z"], name="Protein Accession")
    pd.testing.assert_series_equal(df["Protein Accession"], expected)
