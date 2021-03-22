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
            "DefaultDirection\t-\t-\t-\t1\t-\t-\n"
            "a\t1\tABC\t5\t2\tprotein1\tprotein2\n"
            "b\t-1\tCBA\t10\t3\tdecoy_protein1\tdecoy_protein2"
        )
        pin.write(dat)

    return out_file


def test_pin_parsing(std_pin):
    """Test pin parsing"""
    df = mokapot.read_pin(std_pin, to_df=True)
    assert df["LaBel"].dtype == "bool"
    assert len(df) == 2
    assert len(df[df["LaBel"]]) == 1
    assert len(df[df["LaBel"]]) == 1

    dat = mokapot.read_pin(std_pin)
    pd.testing.assert_frame_equal(df.loc[:, ("sCore",)], dat.features)


def test_pin_wo_dir():
    """Test a PIN file without a DefaultDirection line"""
    dat = mokapot.read_pin("data/scope2_FP97AA.pin")
