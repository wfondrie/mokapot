"""Test that we can parse a FASTA file correctly"""
import pytest
from mokapot import FastaProteins


@pytest.fixture
def missing_fasta(tmp_path):
    """Create a fasta file with a missing entry"""
    out_file = tmp_path / "missing.fasta"
    with open(out_file, "w+") as fasta_ref:
        fasta_ref.write(
            ">sp|test_1|test_1\n"
            ">sp|test_2|test_2\n"
            "TKDIPIIFLSAVNIDKRFITKGYNSGGADY"
        )

    return out_file


def test_fasta_with_missing(missing_fasta):
    """Test that a fasta file can be parsed with missing entries

    See https://github.com/wfondrie/mokapot/issues/13
    """
    FastaProteins(missing_fasta)
