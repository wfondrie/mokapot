import pytest
from mokapot.parsers.helpers import (
    find_optional_column,
    find_columns,
    find_required_column,
)


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


def test_find_optional_column():
    # With col=None:
    assert find_optional_column(None, ["ID", "Name", "Value"], "id") == "ID"
    assert (
        find_optional_column(None, ["Id", "Name", "Value"], "Value") == "Value"
    )
    assert find_optional_column(None, ["RefID", "Name", "Value"], "id") is None

    # With matching columns
    assert (
        find_optional_column("Name", ["ID", "Name", "Value"], "id") == "Name"
    )
    assert (
        find_optional_column("Value", ["Id", "Name", "Value"], "Value")
        == "Value"
    )
    assert find_optional_column("ID", ["ID", "Name", "Value"], "id") == "ID"

    # Without matching columns
    with pytest.raises(ValueError, match=".*'Invalid'.*was not found.*"):
        find_optional_column("Invalid", ["ID", "Name", "Value"], "id")

    with pytest.raises(ValueError, match=".*'NotPresent'.*was not found.*"):
        find_optional_column("NotPresent", ["Id", "Name", "Value"], "Value")


def test_find_required_column():
    # Test column found
    assert (
        find_required_column("Score", ["Identifier", "Score", "Norm_Score"])
        == "Score"
    )

    # Test case-insensitive search
    assert (
        find_required_column("score", ["Identifier", "Score", "Norm_Score"])
        == "Score"
    )

    # Test column not found
    with pytest.raises(ValueError):
        find_required_column(
            "Score_missing", ["Identifier", "Score", "Norm_Score"]
        )

    # Test column not unique
    with pytest.raises(ValueError):
        find_required_column(
            "Score", ["Identifier", "Score", "Norm_Score", "Score"]
        )

    # Test empty columns list
    with pytest.raises(ValueError):
        find_required_column("Score", [])
