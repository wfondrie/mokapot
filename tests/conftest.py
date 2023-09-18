"""Fixtures that are used at multiple points in the tests."""
import random

import numpy as np
import polars as pl
import pytest

import mokapot
from mokapot import PsmDataset, PsmSchema
from mokapot.confidence import Confidence, ConfidenceEstimates, TdcLevel


@pytest.fixture(scope="session")
def psm_df_easy():
    """A DataFrame containing 6 PSMs."""
    rng = np.random.Generator(np.random.PCG64(42))
    data = {
        "target": [True] * 100 + [False],
        "spectrum": list(range(50)) * 2 + [51],
        "peptide": [_random_peptide(5, rng) for i in range(100)] + ["DECOY"],
        "feature": list(range(50)) * 2 + [-1],
    }

    schema = {
        "target": "target",
        "spectrum": "spectrum",
        "peptide": "peptide",
    }
    return pl.DataFrame(data), schema


@pytest.fixture(scope="session")
def psm_df_6():
    """A DataFrame containing 6 PSMs."""
    data = {
        "target": [True, True, True, False, False, False],
        "spectrum": [1, 2, 3, 4, 5, 1],
        "group": [1, 1, 2, 2, 2, 1],
        "peptide": ["a", "b", "a", "c", "d", "e"],
        "protein": ["A", "B"] * 3,
        "feature_1": [4, 3, 2, 1, 1, 0],
        "feature_2": [2, 3, 3, 1, 2, 2],
    }

    schema = {
        "target": "target",
        "spectrum": "spectrum",
        "group": "group",
        "peptide": "peptide",
        "metadata": "protein",
    }
    return pl.DataFrame(data), schema


@pytest.fixture()
def psm_df_1000(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    # The simulated data:
    rng = np.random.Generator(np.random.PCG64(42))
    peptides = [_random_peptide(6, rng) for _ in range(1000)]
    targets = {
        "target": [True] * 500,
        "spectrum": np.arange(500),
        "group": rng.choice(2, size=500),
        "peptide": peptides[:500],
        "score": np.concatenate(
            [rng.normal(2, size=200), rng.normal(size=300)]
        ),
        "score2": rng.normal(size=500),
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    decoys = {
        "target": [False] * 500,
        "spectrum": np.arange(500),
        "group": rng.choice(2, size=500),
        "peptide": [p[::-1] for p in rng.choice(peptides, 500)],
        "score": rng.normal(size=500),
        "score2": rng.normal(size=500),
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    df = pl.concat([pl.DataFrame(targets), pl.DataFrame(decoys)])

    # The schema for it:
    schema_kwargs = {
        "target": "target",
        "spectrum": "spectrum",
        "peptide": "peptide",
        "features": ["score", "score2"],
        "file": "filename",
        "scan": "spectrum",
        "calcmass": "calcmass",
        "expmass": "expmass",
        "ret_time": "ret_time",
        "charge": "charge",
    }

    # The fasta for it:
    fasta_data = "\n".join(_make_fasta(peptides, 50, rng))

    fasta = tmp_path / "test.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)

    return df, fasta, schema_kwargs


@pytest.fixture
def psms(psm_df_1000):
    """A small PsmDataset."""
    df, _, schema_kwargs = psm_df_1000
    return PsmDataset(
        data=df,
        schema=PsmSchema(**schema_kwargs),
        eval_fdr=0.05,
    )


@pytest.fixture
def psms_with_proteins(psm_df_1000):
    """A small PsmDataset."""
    df, fasta, schema_kwargs = psm_df_1000
    return PsmDataset(
        data=df,
        schema=PsmSchema(**schema_kwargs),
        proteins=mokapot.read_fasta(fasta, missed_cleavages=0),
        eval_fdr=0.05,
    )


def _make_fasta(peptides, peptides_per_protein, random_state, prefix=""):
    """Create a FASTA string from a set of peptides.

    Parameters
    ----------
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
    random.seed(int(random_state.integers(0, 9999)))
    random.shuffle(peptides)

    protein = []
    protein_number = 0
    for peptide in peptides:
        protein.append(peptide)
        if len(protein) == peptides_per_protein:
            lines.append(f">{prefix}sp|test|test_{protein_number}")
            lines.append("".join(protein))
            protein = []
            protein_number += 1

    if protein:
        lines.append(f">{prefix}sp|test|test_{protein_number}")
        lines.append("".join(protein))

    return lines


def _random_peptide(length, random_state):
    """Generate a random peptide."""
    return "".join(
        list(random_state.choice(list("ACDEFGHILMNPQSTVWY"), length - 1))
        + ["K"]
    )


@pytest.fixture
def mock_proteins():
    """Create a mock proteins object."""

    class Proteins:
        def __init__(self):
            self.peptide_map = {"ABCDXYZ": "X|Y|Z"}
            self.shared_peptides = {"ABCDEFG": "A|B|C; X|Y|Z"}
            self.protein_map = {"X|Y|Z": "A|B|C"}

    return Proteins()


@pytest.fixture
def mock_confidence():
    """Create a mock-up of a Confidence object."""
    xdf = pl.DataFrame({"x": [1, 2, 3], "y": list("abc")}).lazy()
    ydf = pl.DataFrame({"x": [4, 5, 6, 7], "y": list("defg")}).lazy()
    zdf = pl.DataFrame({"z": [1, 2, 3], "y": list("abc")}).lazy()

    peptides = pl.DataFrame(
        {
            "fname": ["x/y/c.mzML"] * 2,
            "seq": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
            "calcmass": [1, 2],
            "rt": [60.0, 120.0],
            "charge": [2, 3],
            "mokapot protein group": ["A|B|C; B|C|A", "A|B|C"],
            "mokapot q-value": [0.01, 0.001],
        }
    ).lazy()

    schema = PsmSchema(
        target="foo",
        spectrum="foo",
        peptide="seq",
        file="fname",
        calcmass="calcmass",
        ret_time="rt",
        charge="charge",
    )

    results = {
        TdcLevel(name="x", columns="x", unit="xs", schema=None, rng=None): xdf,
        TdcLevel(name="y", columns="y", unit="ys", schema=None, rng=None): ydf,
        TdcLevel(
            name="peptides",
            columns="seq",
            unit="peptides",
            schema=None,
            rng=None,
        ): peptides,
    }

    decoy_results = {
        TdcLevel(name="z", columns="z", unit="zs", schema=None, rng=None): zdf,
    }

    class MockConfidence(Confidence):
        def __init__(self):
            self.eval_fdr = 0.01
            self._proteins = True
            self.schema = schema
            self._results = ConfidenceEstimates(0.01, results)
            self._decoy_results = ConfidenceEstimates(0.01, decoy_results)

    return MockConfidence()
