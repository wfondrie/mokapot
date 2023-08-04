"""Test that parsing Percolator input files works correctly"""
import pytest
import mokapot
import pandas as pd


@pytest.fixture
def std_pin(tmp_path):
    """Create a standard pin file"""
    out_file = tmp_path / "std_pin"
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
    dat = mokapot.read_pin(std_pin, max_workers=4,)
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
    mokapot.read_pin("data/scope2_FP97AA.pin", max_workers=4)
