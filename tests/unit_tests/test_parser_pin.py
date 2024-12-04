"""Test that parsing Percolator input files works correctly"""

from pathlib import Path

import pandas as pd
import pytest

import mokapot


@pytest.fixture
def std_pin(tmp_path):
    """Create a standard pin file"""
    out_file = tmp_path / "std.pin"
    with open(str(out_file), "w+") as pin:
        dat = (
            "sPeCid\tLaBel\tpepTide\tsCore\tscanNR\tpRoteins\n"
            "a\t1\tABC\t5\t2\tprotein1:protein2\n"
            "b\t-1\tCBA\t10\t3\tdecoy_protein1:decoy_protein2"
        )
        pin.write(dat)

    return out_file


def test_pin_parsing(std_pin):
    """Test pin parsing"""
    datasets = mokapot.read_pin(
        std_pin,
        max_workers=4,
    )
    df = pd.read_csv(std_pin, sep="\t")
    assert len(datasets) == 1

    pd.testing.assert_frame_equal(
        df.loc[:, ("sCore",)], df.loc[:, datasets[0].feature_columns]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "sPeCid"], df.loc[:, datasets[0].specId_column]
    )
    pd.testing.assert_series_equal(
        df.loc[:, "pRoteins"], df.loc[:, datasets[0].protein_column]
    )
    pd.testing.assert_frame_equal(
        df.loc[:, ("scanNR",)], df.loc[:, datasets[0].spectrum_columns]
    )


def test_pin_wo_dir():
    """Test a PIN file without a DefaultDirection line"""
    mokapot.read_pin(Path("data", "scope2_FP97AA.pin"), max_workers=4)
