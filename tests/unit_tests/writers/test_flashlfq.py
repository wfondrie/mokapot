"""Test that FlashLFQ export is working."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import mokapot


def test_smoke(psms_with_proteins, tmp_path):
    """Run simple sanity checks."""
    conf = psms_with_proteins.assign_confidence()
    test1 = conf.to_flashlfq(tmp_path / "test1.txt")
    mokapot.to_flashlfq(conf, tmp_path / "test2.txt")
    test3 = mokapot.to_flashlfq([conf, conf], tmp_path / "test3.txt")
    with pytest.raises(ValueError):
        mokapot.to_flashlfq("blah", tmp_path / "test4.txt")

    df1 = pl.read_csv(test1, separator="\t")
    df3 = pl.read_csv(test3, separator="\t")
    assert 2 * len(df1) == len(df3)
    assert len(df1.columns) == 7


def test_basic(mock_confidence, tmp_path):
    """Test that the basic output works."""
    df = pl.read_csv(
        mokapot.to_flashlfq(mock_confidence, tmp_path / "test.txt"),
        separator="\t",
    )

    expected = pl.DataFrame(
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

    assert_frame_equal(df, expected)


@pytest.mark.parametrize("column", ["file", "calcmass", "ret_time", "charge"])
def test_with_missing(mock_confidence, tmp_path, column):
    """Test that missing columns causes errors."""
    setattr(mock_confidence.schema, column, None)
    with pytest.raises(ValueError):
        mokapot.to_flashlfq(mock_confidence, tmp_path / "test.txt")


def test_no_proteins(mock_confidence, tmp_path):
    """Test when no proteins are available."""
    mock_confidence.proteins = None
    with pytest.raises(ValueError):
        mokapot.to_flashlfq(mock_confidence, tmp_path / "test.txt")
