"""Test column validation"""
import polars as pl
import pytest

from mokapot import PsmSchema


def test_init():
    """Test that we can init it."""
    # Only required:
    cols = PsmSchema(
        target="x",
        spectrum=["y", "z"],
        peptide="p",
    )
    assert cols.target == "x"
    assert cols.group is None

    cols = PsmSchema(
        target="x",
        spectrum=["y", "z"],
        peptide="p",
        group="a",
        file="b",
        scan="c",
        calcmass="d",
        expmass="e",
        ret_time="f",
        charge="g",
        metadata="h",
        features="i",
        score="j",
        desc=False,
    )

    assert cols.spectrum == ["y", "z"]
    assert cols.group == ["a"]
    assert cols.score == "j"
    assert not cols.desc


def test_init_errors():
    """Test initialization errors"""
    with pytest.raises(ValueError):
        PsmSchema(
            target="x",
            spectrum=["y", "z"],
            peptide=None,
        )

    with pytest.raises(ValueError):
        PsmSchema(
            target=["a", "x"],
            spectrum=["y", "z"],
            peptide="p",
        )


def test_good_data():
    """Test that validation works."""
    df = pl.DataFrame(
        {"t": [True, False], "s": [1, 2], "p": ["A", "B"]}
    ).lazy()

    cols = PsmSchema("t", "s", "p")
    cols.validate(df)


def test_missing_column():
    """Test for an error with a missing column."""
    df = pl.DataFrame(
        {"t": [True, False], "s": [1, 2], "p": ["A", "B"]}
    ).lazy()

    cols = PsmSchema("t", "s", "p", "x")
    with pytest.raises(ValueError):
        cols.validate(df)


def test_labels():
    """Test that labels are working correctly."""
    df = pl.DataFrame(
        {"t": [True, False], "s": [1, 2], "p": ["A", "B"], "f": [1, 2]}
    ).lazy()

    cols = PsmSchema("t", "s", "p")
    assert not cols.validate(df)
    assert cols.features == ["f"]

    df = pl.DataFrame({"t": [1, -1], "s": [1, 2], "p": ["A", "B"]}).lazy()

    cols = PsmSchema("t", "s", "p")
    assert cols.validate(df)

    df = pl.DataFrame({"t": [1, 0], "s": [1, 2], "p": ["A", "B"]}).lazy()

    cols = PsmSchema("t", "s", "p")
    assert not cols.validate(df)

    df = pl.DataFrame(
        {"t": [1, 0, -1], "s": [1, 2, 3], "p": ["A", "B", "C"]}
    ).lazy()

    cols = PsmSchema("t", "s", "p")
    with pytest.raises(ValueError):
        cols.validate(df)
