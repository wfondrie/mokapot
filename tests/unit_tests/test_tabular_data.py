import numpy as np
import pytest
from pathlib import Path
import pandas as pd
from numpy import dtype
import pyarrow as pa

from mokapot.tabular_data import (
    TabularDataReader,
    CSVFileReader,
    ParquetFileReader,
    DataFrameReader,
    ColumnMappedReader,
    CSVFileWriter,
    auto_finalize,
)


def test_from_path(tmp_path):
    reader = TabularDataReader.from_path(Path(tmp_path, "test.csv"))
    assert isinstance(reader, CSVFileReader)

    reader = TabularDataReader.from_path(Path(tmp_path, "test.parquet"))
    assert isinstance(reader, ParquetFileReader)

    with pytest.warns(UserWarning):
        reader = TabularDataReader.from_path(Path(tmp_path, "test.blah"))
    assert isinstance(reader, CSVFileReader)


@pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning")
def test_csv_file_reader():
    # Note: this pin file is kinda "non-standard" which is why I put the
    # filterwarnings decorator before the test function

    path = Path("data", "phospho_rep1.pin")
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "SpecId": dtype("O"),
        "Label": dtype("int64"),
        "ScanNr": dtype("int64"),
        "ExpMass": dtype("float64"),
        "CalcMass": dtype("float64"),
        "lnrSp": dtype("float64"),
        "deltLCn": dtype("float64"),
        "deltCn": dtype("float64"),
        "Sp": dtype("float64"),
        "IonFrac": dtype("float64"),
        "RefactoredXCorr": dtype("float64"),
        "NegLog10PValue": dtype("float64"),
        "NegLog10ResEvPValue": dtype("float64"),
        "NegLog10CombinePValue": dtype("float64"),
        "PepLen": dtype("int64"),
        "Charge1": dtype("int64"),
        "Charge2": dtype("int64"),
        "Charge3": dtype("int64"),
        "Charge4": dtype("int64"),
        "Charge5": dtype("int64"),
        "enzN": dtype("int64"),
        "enzC": dtype("int64"),
        "enzInt": dtype("int64"),
        "lnNumDSP": dtype("float64"),
        "dM": dtype("float64"),
        "absdM": dtype("float64"),
        "Peptide": dtype("O"),
        "Proteins": dtype("O"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    df = reader.read(["ScanNr", "SpecId"])
    assert df.columns.tolist() == ["ScanNr", "SpecId"]
    assert len(df) == 55398

    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=20000, columns=["ScanNr", "SpecId"]
    )
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [20000, 20000, 15398]


def test_parquet_file_reader():
    path = Path("data", "10k_psms_test.parquet")
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "SpecId": pa.int64(),
        "Label": pa.int64(),
        "ScanNr": pa.int64(),
        "ExpMass": pa.float64(),
        "Mass": pa.float64(),
        "MS8_feature_5": pa.int64(),
        "missedCleavages": pa.int64(),
        "MS8_feature_7": pa.float64(),
        "MS8_feature_13": pa.float64(),
        "MS8_feature_20": pa.float64(),
        "MS8_feature_21": pa.float64(),
        "MS8_feature_22": pa.float64(),
        "MS8_feature_24": pa.int64(),
        "MS8_feature_29": pa.float64(),
        "MS8_feature_30": pa.float64(),
        "MS8_feature_32": pa.float64(),
        "Peptide": pa.string(),
        "Proteins": pa.string(),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    df = reader.read(["ScanNr", "SpecId"])
    assert df.columns.tolist() == ["ScanNr", "SpecId"]
    assert len(df) == 10000

    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=3300, columns=["ScanNr", "SpecId"]
    )
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [3300, 3300, 3300, 100]


def test_dataframe_reader(psm_df_6):
    reader = DataFrameReader(psm_df_6)
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "target": dtype("bool"),
        "spectrum": dtype("int64"),
        "peptide": dtype("O"),
        "protein": dtype("O"),
        "feature_1": dtype("int64"),
        "feature_2": dtype("int64"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    assert len(reader.read()) == 6
    chunk_iterator = reader.get_chunked_data_iterator(
        chunk_size=4, columns=["peptide"]
    )
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [4, 2]
    pd.testing.assert_frame_equal(
        chunks[0], pd.DataFrame({"peptide": ["a", "b", "a", "c"]})
    )
    pd.testing.assert_frame_equal(
        chunks[1], pd.DataFrame({"peptide": ["d", "e"]}, index=[4, 5])
    )

    assert reader.read(["feature_1", "spectrum"]).columns.tolist() == [
        "feature_1",
        "spectrum",
    ]

    # Test whether we can create a reader from a Series
    reader = DataFrameReader.from_series(
        pd.Series(data=[1, 2, 3], name="test")
    )
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]})
    )

    reader = DataFrameReader.from_series(
        pd.Series(data=[1, 2, 3]), name="test"
    )
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]})
    )

    # Test whether we can create a reader from an array
    reader = DataFrameReader.from_array([1, 2, 3], name="test")
    pd.testing.assert_frame_equal(
        reader.read(), pd.DataFrame({"test": [1, 2, 3]})
    )

    reader = DataFrameReader.from_array(
        np.array([1, 2, 3],
        dtype=np.int32), name="test"
    )
    pd.testing.assert_frame_equal(
        reader.read(), 
        pd.DataFrame({"test": [1, 2, 3]}, dtype=np.int32)
    )


def test_column_renaming(psm_df_6):
    orig_reader = DataFrameReader(psm_df_6)
    reader = ColumnMappedReader(
        orig_reader, {"target": "T", "peptide": "Pep", "Targ": "T"}
    )
    names = reader.get_column_names()
    types = reader.get_column_types()
    column_to_types = dict(zip(names, types))

    expected_column_to_types = {
        "T": dtype("bool"),
        "spectrum": dtype("int64"),
        "Pep": dtype("O"),
        "protein": dtype("O"),
        "feature_1": dtype("int64"),
        "feature_2": dtype("int64"),
    }

    for name, type in expected_column_to_types.items():
        assert column_to_types[name] == type

    assert (reader.read().values == orig_reader.read().values).all()
    assert (
            reader.read(["Pep", "protein", "T", "feature_1"]).values
            == orig_reader.read([
        "peptide",
        "protein",
        "target",
        "feature_1",
    ]).values
    ).all()

    renamed_chunk = next(
        reader.get_chunked_data_iterator(
            chunk_size=4, columns=["Pep", "protein", "T", "feature_1"]
        )
    )
    orig_chunk = next(
        orig_reader.get_chunked_data_iterator(
            chunk_size=4, columns=["peptide", "protein", "target", "feature_1"]
        )
    )
    assert (renamed_chunk.values == orig_chunk.values).all()


# todo: tests for writers are still missing
def test_tabular_writer_context_manager(tmp_path):
    # Create a mock class that checks whether it will be correctly initialized
    # and finalized
    class MockWriter(CSVFileWriter):
        initialized = False
        finalized = False

        def initialize(self):
            super().initialize()
            self.initialized = True

        def finalize(self):
            super().finalize()
            self.finalized = True

    # Check that context manager works for one file
    with MockWriter(tmp_path / "test.csv", columns=["a", "b"]) as writer:
        assert writer.initialized
        assert not writer.finalized
    assert writer.finalized

    # Check that it works when an exception is thrown
    try:
        with MockWriter(tmp_path / "test.csv", columns=["a", "b"]) as writer:
            assert writer.initialized
            assert not writer.finalized
            raise RuntimeError("Just testing")
    except RuntimeError:
        pass  # ignore the exception
    finally:
        assert writer.finalized

    # Check that context manager convenience method (auto_finalize) works for
    # multiple files
    writers = [
        MockWriter(tmp_path / "test1.csv", columns=["a", "b"]),
        MockWriter(tmp_path / "test2.csv", columns=["a", "b"]),
    ]

    assert not writers[0].initialized
    assert not writers[1].initialized
    with auto_finalize(writers):
        assert writers[0].initialized
        assert writers[1].initialized
        assert not writers[0].finalized
        assert not writers[1].finalized
    assert writers[0].finalized
    assert writers[1].finalized

    # Now with an exception
    writers = [
        MockWriter(tmp_path / "test1.csv", columns=["a", "b"]),
        MockWriter(tmp_path / "test2.csv", columns=["a", "b"]),
    ]

    try:
        with auto_finalize(writers):
            raise RuntimeError("Just testing")
    except RuntimeError:
        pass
    finally:
        assert writers[0].finalized
        assert writers[1].finalized
