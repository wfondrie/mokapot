"""
This file contains fixtures that are used at multiple points in the tests.
"""
import pytest
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset


@pytest.fixture(scope="session")
def psm_df_6():
    """A DataFrame containing 6 PSMs"""
    data = {
        "target": [True, True, True, False, False, False],
        "spectrum": [1, 2, 3, 4, 5, 1],
        "group": [1, 1, 2, 2, 2, 1],
        "peptide": ["a", "b", "a", "c", "d", "e"],
        "protein": ["A", "B"] * 3,
        "feature_1": [4, 3, 2, 2, 1, 0],
        "feature_2": [2, 3, 4, 1, 2, 3],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def psm_df_1000(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "target": [True] * 500,
        "spectrum": np.arange(500),
        "group": rng.choice(2, size=500),
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "score": np.concatenate(
            [rng.normal(3, size=200), rng.normal(size=300)]
        ),
    }

    decoys = {
        "target": [False] * 500,
        "spectrum": np.arange(500),
        "group": rng.choice(2, size=500),
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "score": rng.normal(size=500),
    }

    fasta_data = "\n".join(
        _make_fasta(100, targets["peptide"], 10, rng)
        + _make_fasta(100, decoys["peptide"], 10, rng, "decoy")
    )

    fasta = tmp_path / "test_1000.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)

    return (pd.concat([pd.DataFrame(targets), pd.DataFrame(decoys)]), fasta)


@pytest.fixture
def psms(psm_df_1000):
    """A small LinearPsmDataset"""
    df, _ = psm_df_1000
    psms = LinearPsmDataset(
        psms=df,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        feature_columns="score",
        copy_data=True,
    )
    return psms


def _make_fasta(
    num_proteins, peptides, peptides_per_protein, random_state, prefix=""
):
    """Create a FASTA string from a set of peptides

    Parameters
    ----------
    num_proteins : int
        The number of proteins to generate.
    peptides : list of str
        A list of peptide sequences.
    peptides_per_protein: int
        The number of peptides per protein.
    random_state : numpy.random.Generator object
        The random state.
    prefix : str
        The prefix, if generating decoys

    Returns
    -------
    list of str
        A list of lines in a FASTA file.
    """
    lines = []
    for protein in range(num_proteins):
        lines.append(f">{prefix}sp|test|test_{protein}")
        lines.append(
            "".join(list(random_state.choice(peptides, peptides_per_protein)))
        )

    return lines


def _random_peptide(length, random_state):
    """Generate a random peptide"""
    return "".join(
        list(random_state.choice(list("ACDEFGHILMNPQSTVWY"), length - 1))
        + ["K"]
    )
