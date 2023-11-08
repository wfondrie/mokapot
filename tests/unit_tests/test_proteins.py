"""Test that our picked protein appraoch is work as we expect."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mokapot import PsmSchema
from mokapot.proteins import Proteins


@pytest.fixture
def data():
    """The data to work with."""
    cols = ["target", "peptide", "score"]
    df = [
        [True, "BLAH.A[+1]BC", 10],
        [True, "BCD", 9],
        [False, "CBA", 9],
        [False, "DCB", 10],
        [True, "EFG", 1],
        [False, "GFE", 1],
        [True, "HIJ", 2],
        [False, "MLK", 2],
    ]

    return pl.DataFrame(df, schema=cols)


@pytest.fixture
def schema():
    """The schema for the data."""
    return PsmSchema(
        target="target",
        spectrum="peptide",
        peptide="peptide",
    )


def test_proteins(data, schema):
    """The proteins with decoy present."""
    peptide_map = {
        "ABC": ["T_A"],
        "BCD": ["T_A", "T_B"],
        "EFG": ["T_B"],
        "HIJ": ["T_B"],
        "KLM": ["T_C"],
    }

    with pl.StringCache():
        picked = (
            Proteins(peptides=peptide_map, rng=42)
            .pick(data=data, schema=schema)
            .with_columns(
                pl.col("stripped sequence").cast(pl.Utf8),
                pl.col("mokapot protein group").cast(pl.Utf8),
            )
            .collect()
        )

    # Expected:
    cols = [
        "target",
        "peptide",
        "stripped sequence",
        "mokapot protein group",
        "# mokapot protein groups",
    ]

    data = [
        [True, "BLAH.A[+1]BC", "ABC", "T_A", 1],
        [True, "BCD", "BCD", "T_A;T_B", 2],
        [False, "CBA", "CBA", "T_A", 1],
        [False, "DCB", "DCB", "T_A;T_B", 2],
        [True, "EFG", "EFG", "T_B", 1],
        [False, "GFE", "GFE", "T_B", 1],
        [True, "HIJ", "HIJ", "T_B", 1],
        [False, "MLK", "MLK", "T_C", 1],
    ]

    expected = pl.DataFrame(data, schema=cols)
    assert_frame_equal(picked, expected)
