import pytest
from mokapot.parsers.helpers import find_column, find_columns


def test_find_columns():
    # Test case where all column names match
    columns = ["test", "Test", "TEST"]
    assert find_columns("Test", columns) == ["test", "Test", "TEST"]

    # Test case where some column names match
    columns = ["test", "Test", "TEST", "no_match"]
    assert find_columns("Test", columns) == ["test", "Test", "TEST"]

    # Test case where no column names match
    columns = ["no_match", "NO_MATCH"]
    assert find_columns("Test", columns) == []

    # Test case where input is an empty list
    assert find_columns("Test", []) == []

    # Test case where input column is case-sensitive
    columns = ["TEST", "test", "test"]
    assert find_columns("TEST", columns) == ["TEST"]


def test_find_column_with_col_none():
    assert find_column(None, ['ID', 'Name', 'Value'], 'id') == 'ID'
    assert find_column(None, ['Id', 'Name', 'Value'], 'Value') == 'Value'
    assert find_column(None, ['RefID', 'Name', 'Value'], 'id') is None


def test_find_column_with_col_matching_columns():
    assert find_column('Name', ['ID', 'Name', 'Value'], 'id') == 'Name'
    assert find_column('Value', ['Id', 'Name', 'Value'], 'Value') == 'Value'
    assert find_column('ID', ['ID', 'Name', 'Value'], 'id') == 'ID'


def test_find_column_with_col_not_matching_columns():
    with pytest.raises(ValueError, match="The 'Invalid' column was not found."):
        find_column('Invalid', ['ID', 'Name', 'Value'], 'id')

    with pytest.raises(ValueError,
                       match="The 'NotPresent' column was not found."):
        find_column('NotPresent', ['Id', 'Name', 'Value'], 'Value')
