"""Test that parsing Percolator input files works correctly."""
import pytest

import mokapot


@pytest.fixture
def std_pin(tmp_path):
    """Create a standard pin file."""
    out_file = tmp_path / "std_pin"
    with open(str(out_file), "w+") as pin:
        dat = (
            "sPeCid\tLaBel\tgroup\tpepTide\tsCore\tscanNR\tpRoteins\n"
            "a\t1\t1\tABC\t5\t2\tprotein1\tprotein2\n"
            "b\t-1\t1\tCBA\t10\t3\tdecoy_protein1\tdecoy_protein2"
        )
        pin.write(dat)

    return out_file


@pytest.fixture
def dir_pin(tmp_path):
    """Create a standard pin file."""
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


def test_perc_parsing(dir_pin):
    """Test pin parsing."""
    df = mokapot.percolator_to_df(dir_pin)
    assert len(df) == 2
    assert df["pRoteins"][0] == "protein1\tprotein2"


def test_std_pin(std_pin):
    """Test a PIN file without a DefaultDirection line."""
    dset = mokapot.read_pin(std_pin, group="group")
    assert len(dset) == 2
    assert dset.schema.features == ["sCore"]

    df = mokapot.percolator_to_df(std_pin)
    assert len(df) == 2
    assert df["pRoteins"][0] == "protein1\tprotein2"

    dset = mokapot.read_pin(dset.data.collect(), group="group")
    assert len(dset) == 2
    assert dset.schema.features == ["sCore"]
