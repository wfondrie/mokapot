import pandas as pd
from mokapot.tabular_data import DataFrameReader
from mokapot.streaming import merge_readers, MergedTabularDataReader
import pytest


@pytest.fixture
def readers():
    # Prepare two dataframe readers
    df1 = pd.DataFrame(
        {"foo": [1, 3, 4, 5, 8, 10], "bar": [20, 6, 3, 2, 1, 0], })
    df2 = pd.DataFrame(
        {"foo": [2, 2, 6, 9, 11, 13, 15], "bar": [18, 16, 13, 6, 5, 5, 0], })
    reader1 = DataFrameReader(df1)
    reader2 = DataFrameReader(df2)
    return [reader1, reader2]

def test_merged_tabular_data_reader(readers):
    # Check that complete merged read works
    reader = MergedTabularDataReader(readers, "foo", descending=False, reader_chunk_size=4)
    pd.testing.assert_frame_equal(reader.read(),
        pd.DataFrame({"foo": [1, 2, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15], "bar": [20, 18, 16, 6, 3, 2, 13, 1, 6, 0, 5, 5, 0]}))

    # Check that chunked read works
    iterator = reader.get_chunked_data_iterator(chunk_size=3)
    chunks =  [chunk for chunk in iterator]
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 3, 1]

    # Should raise an error since "bar" column is not ascending
    with pytest.raises(ValueError):
        iterator = merge_readers(readers, priority_column="bar", descending=False)
        values = [row.iloc[0].values.tolist() for row in iterator]

    # Should raise an error since "foo" column is not descending
    with pytest.raises(ValueError):
        iterator = merge_readers(readers, priority_column="foo", descending=True, reader_chunk_size=3)
        values = [row.iloc[0].values.tolist() for row in iterator]


def test_merge_readers(readers):

    # Check that ascending merge on "foo" column works
    iterator = merge_readers(readers, priority_column="foo", descending=False)

    values = [row.iloc[0].values.tolist() for row in iterator]
    assert values == [[1, 20], [2, 18], [2, 16], [3, 6], [4, 3], [5, 2],
                      [6, 13], [8, 1], [9, 6], [10, 0], [11, 5], [13, 5],
                      [15, 0]]

    # Check that descending merge on "bar" column works
    iterator = merge_readers(readers, priority_column="bar", descending=True, reader_chunk_size=3)
    values = [row.iloc[0].values.tolist() for row in iterator]
    assert (values == [[1, 20], [2, 18], [2, 16], [6, 13], [3, 6], [9, 6],
                       [11, 5], [13, 5], [4, 3], [5, 2], [8, 1], [10, 0],
                       [15, 0]])



def test_joined_tabular_data_reader(readers):
    # Check that complete merged read works
    reader = MergedTabularDataReader(readers, "foo", descending=False, reader_chunk_size=4)
    pd.testing.assert_frame_equal(reader.read(),
        pd.DataFrame({"foo": [1, 2, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15], "bar": [20, 18, 16, 6, 3, 2, 13, 1, 6, 0, 5, 5, 0]}))
