"""Test that parsing Percolator input files works correctly"""

from pathlib import Path

import pytest
import pandas as pd

import mokapot
from mokapot.parsers import pin


@pytest.fixture
def std_pin(tmp_path):
    """Create a standard pin file"""
    out_file = tmp_path / "std.pin"
    with open(str(out_file), "w+") as pin:
        dat = (
            "sPeCid\tLaBel\tpepTide\tsCore\tscanNR\tpRoteins\n"
            "DefaultDirection\t0\t-\t-\t1\t-\t-\n"
            "a\t1\tABC\t5\t2\tprotein1\tprotein2\n"
            "b\t-1\tCBA\t10\t3\tdecoy_protein1\tdecoy_protein2"
        )
        pin.write(dat)

    return out_file


def test_pin_parsing(std_pin):
    """Test pin parsing"""
    dat = mokapot.read_pin(
        std_pin,
        max_workers=4,
    )
    df = pd.read_csv(std_pin, sep="\t")
    assert len(dat) == 1
    assert dat[0].filename == std_pin
    pd.testing.assert_frame_equal(
        df.loc[:, ("sCore",)], df.loc[:, dat[0].feature_columns]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "sPeCid"], df.loc[:, dat[0].specId_column]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "pRoteins"], df.loc[:, dat[0].protein_column]
    )
    pd.testing.assert_frame_equal(
        df.loc[:, ("scanNR",)], df.loc[:, dat[0].spectrum_columns]
    )


def test_pin_wo_dir():
    """Test a PIN file without a DefaultDirection line"""
    mokapot.read_pin(Path("data", "scope2_FP97AA.pin"), max_workers=4)


def test_read_file_in_chunks():
    """Test reading files in chungs"""
    columns = ["SpecId", "Label", "ScanNr", "ExpMass"]
    iterator = pin.read_file_in_chunks(
        Path("data", "scope2_FP97AA.pin"), 100, use_cols=columns
    )
    df = next(iterator)
    assert len(df) == 100
    assert df.iloc[0, 0] == "target_0_9674_2_-1"
    assert df.iloc[0, 2] == 9674

    # Read in different column order than given in file
    columns = ["ExpMass", "SpecId", "Label", "ScanNr"]
    iterator = pin.read_file_in_chunks(
        Path("data", "scope2_FP97AA.pin"), 100, use_cols=columns
    )
    df = next(iterator)
    assert len(df) == 100
    assert df.iloc[0, 1] == "target_0_9674_2_-1"
    assert df.iloc[0, 3] == 9674
