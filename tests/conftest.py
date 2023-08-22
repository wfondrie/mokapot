"""
This file contains fixtures that are used at multiple points in the tests.
"""
import numpy as np
import polars as pl
import pytest

from mokapot import PsmDataset, PsmSchema


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

    schema = dict(
        target="target",
        spectrum="spectrum",
        group="group",
        peptide="peptide",
        metadata="protein",
    )
    return pl.DataFrame(data), schema


@pytest.fixture()
def psm_df_1000(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    # The simulated data:
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "target": [True] * 500,
        "spectrum": np.arange(500),
        "group": rng.choice(2, size=500),
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "score": np.concatenate(
            [rng.normal(3, size=200), rng.normal(size=300)]
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
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
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
    schema_kwargs = dict(
        target="target",
        spectrum="spectrum",
        peptide="peptide",
        features=["score", "score2"],
        file="filename",
        scan="spectrum",
        calcmass="calcmass",
        expmass="expmass",
        ret_time="ret_time",
        charge="charge",
    )

    # The fasta for it:
    fasta_data = "\n".join(
        _make_fasta(100, targets["peptide"], 10, rng)
        + _make_fasta(100, decoys["peptide"], 10, rng, "decoy")
    )

    fasta = tmp_path / "test_1000.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)

    return df, fasta, schema_kwargs


@pytest.fixture
def psms(psm_df_1000):
    """A small PsmDataset"""
    df, _, schema_kwargs = psm_df_1000
    return PsmDataset(data=df, schema=PsmSchema(**schema_kwargs))


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


@pytest.fixture
def mock_proteins():
    class Proteins:
        def __init__(self):
            self.peptide_map = {"ABCDXYZ": "X|Y|Z"}
            self.shared_peptides = {"ABCDEFG": "A|B|C; X|Y|Z"}
            self.protein_map = {"X|Y|Z": "A|B|C"}

    return Proteins()


@pytest.fixture
def mock_conf():
    "Create a mock-up of a LinearConfidence object"

    class conf:
        def __init__(self):
            self._optional_columns = {
                "filename": "filename",
                "calcmass": "calcmass",
                "rt": "ret_time",
                "charge": "charge",
            }

            self._protein_column = "protein"
            self._peptide_column = "peptide"
            self._eval_fdr = 0.5
            self._proteins = None
            self._has_proteins = False

            self.peptides = pl.DataFrame(
                {
                    "filename": "a/b/c.mzML",
                    "calcmass": [1, 2],
                    "ret_time": [60, 120],
                    "charge": [2, 3],
                    "peptide": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
                    "mokapot q-value": [0.001, 0.1],
                    "protein": ["A|B|C\tB|C|A", "A|B|C"],
                }
            )

            self.confidence_estimates = {"peptides": self.peptides}
            self.decoy_confidence_estimates = {"peptides": self.peptides}

    return conf()
