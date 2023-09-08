"""Test that our picked protein appraoch is work as we expect."""
import polars as pl
import pytest

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


@pytest.fixture
def proteins_with_decoys():
    """The proteins with decoy present."""
    peptide_map = {
        "ABC": "T_A",
        "BCD": "T_A",
        "HIJ": "T_B",
        "KLM": "T_C",
        "CBA": "D_A",
        "DCB": "D_A",
        "JIH": "D_B",
        "MLK": "D_C",
    }

    protein_map = {f"D_{x}": f"T_{x}" for x in "ABC"}
    shared = {"EFG": "T_B", "GFE": "D_B"}

    return Proteins(
        decoy_prefix="D_",
        peptide_map=peptide_map,
        protein_map=protein_map,
        shared_peptides=shared,
        has_decoys=True,
        rng=42,
    )


def test_with_decoys(data, schema, proteins_with_decoys):
    """Test our mapping."""
    picked = proteins_with_decoys.pick(data=data, schema=schema)
    print(picked.collect())
    raise
