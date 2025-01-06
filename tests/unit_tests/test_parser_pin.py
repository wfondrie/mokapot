"""Test that parsing Percolator input files works correctly"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mokapot
from mokapot.parsers.pin import create_chunks_with_identifier
from mokapot.tabular_data import ColumnSelectReader, TabularDataReader


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


def test_read_percolator():
    reader = TabularDataReader.from_path(Path("data", "scope2_FP97AA.pin"))
    mokapot.read_percolator(reader)

    all_cols = reader.get_column_names()

    all_cols.remove("Charge3")
    all_cols.remove("Charge4")
    all_cols.remove("Charge5")
    subset_reader = ColumnSelectReader(reader, selected_columns=all_cols)
    mokapot.read_percolator(subset_reader)


def test_create_chunks_with_identifier():
    identifier = ["ScanNr", "ExpMass", "Label"]
    features = ["lnrSp", "deltLCn", "deltCn", "Sp", "IonFrac"]
    features += ["RefactoredXCorr", "NegLog10PValue", "NegLog10ResEvPValue"]
    N_identifier = len(identifier)
    for cs in range(3, 11):
        chunks = create_chunks_with_identifier(features, identifier, cs)
        lens = np.array([len(set(chunk).intersection(identifier)) for chunk in chunks])
        assert sum(lens) == N_identifier and all(lens % N_identifier == 0)
