import pytest
from pathlib import Path
import pandas as pd
from numpy import dtype

from mokapot.tabular_data import TabularDataReader, CSVFileReader, \
    ParquetFileReader, DataFrameReader


def test_from_path(tmp_path):
    reader = TabularDataReader.from_path(Path(tmp_path, "test.csv"))
    assert isinstance(reader, CSVFileReader)

    reader = TabularDataReader.from_path(Path(tmp_path, "test.parquet"))
    assert isinstance(reader, ParquetFileReader)

    with pytest.warns(UserWarning):
        reader = TabularDataReader.from_path(Path(tmp_path, "test.blah"))
    assert isinstance(reader, CSVFileReader)


def test_csv_file_reader():
    path = Path("data", "phospho_rep1.pin")
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    expected = ["SpecId", "Label", "ScanNr", "ExpMass", "CalcMass", "lnrSp",
                "deltLCn", "deltCn", "Sp", "IonFrac", "RefactoredXCorr",
                "NegLog10PValue", "NegLog10ResEvPValue",
                "NegLog10CombinePValue", "PepLen", "Charge1", "Charge2",
                "Charge3", "Charge4", "Charge5", "enzN", "enzC", "enzInt",
                "lnNumDSP", "dM", "absdM", "Peptide", "Proteins", ]
    assert names == expected

    expected = [dtype('O'), dtype('int64'), dtype('int64'), dtype('float64'),
                dtype('float64'), dtype('float64'), dtype('float64'),
                dtype('float64'), dtype('float64'), dtype('float64'),
                dtype('float64'), dtype('float64'), dtype('float64'),
                dtype('float64'), dtype('int64'), dtype('int64'),
                dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'),
                dtype('int64'), dtype('int64'), dtype('int64'),
                dtype('float64'), dtype('float64'), dtype('float64'),
                dtype('O'), dtype('O')]

    types = reader.get_column_types()
    assert types == expected

    df = reader.read(["ScanNr", "SpecId"])
    assert df.columns.tolist() == ["ScanNr", "SpecId"]
    assert len(df) == 55398

    chunk_iterator = reader.get_chunked_data_iterator(chunk_size=20000, columns=["ScanNr", "SpecId"])
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [20000, 20000, 15398]



def test_parquet_file_reader():
    path = Path("data", "10k_psms_test.parquet")
    reader = TabularDataReader.from_path(path)
    names = reader.get_column_names()
    expected = ['SpecId',
     'Label',
     'ScanNr',
     'ExpMass',
     'Mass',
     'MS8_feature_5',
     'missedCleavages',
     'MS8_feature_7',
     'MS8_feature_13',
     'MS8_feature_20',
     'MS8_feature_21',
     'MS8_feature_22',
     'MS8_feature_24',
     'MS8_feature_29',
     'MS8_feature_30',
     'MS8_feature_32',
     'Peptide',
     'Proteins']

    assert names == expected

    expected = [dtype('int64'), dtype('int64'), dtype('int64'),
                dtype('float64'), dtype('float64'), dtype('int64'),
                dtype('int64'), dtype('float64'), dtype('float64'),
                dtype('float64'), dtype('float64'), dtype('float64'),
                dtype('int64'), dtype('float64'), dtype('float64'),
                dtype('float64'), dtype('O'), dtype('O')]

    types = reader.get_column_types()
    assert types == expected

    df = reader.read(["ScanNr", "SpecId"])
    assert df.columns.tolist() == ["ScanNr", "SpecId"]
    assert len(df) == 10000

    chunk_iterator = reader.get_chunked_data_iterator(chunk_size=3300, columns=["ScanNr", "SpecId"])
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [3300, 3300, 3300, 100]


def test_dataframe_reader(psm_df_6):
    reader = DataFrameReader(psm_df_6)
    assert reader.get_column_names() == ["target", "spectrum", "peptide", "protein", "feature_1", "feature_2"]
    assert reader.get_column_types() == [dtype('bool'), dtype('int64'), dtype('O'), dtype('O'), dtype('int64'), dtype('int64')]

    assert len(reader.read()) == 6
    chunk_iterator = reader.get_chunked_data_iterator(chunk_size=4, columns=["peptide"])
    chunks = [chunk for chunk in chunk_iterator]
    sizes = [len(chunk) for chunk in chunks]
    assert sizes == [4, 2]
    pd.testing.assert_frame_equal(chunks[0], pd.DataFrame({"peptide": ["a", "b", "a", "c"]}))
    pd.testing.assert_frame_equal(chunks[1], pd.DataFrame({"peptide": ["d", "e"]}, index=[4, 5]))
