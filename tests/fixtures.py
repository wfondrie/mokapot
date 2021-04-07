"""
This file contains fixtures that are used at multiple points in the tests.
"""
import pytest
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset


@pytest.fixture
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


@pytest.fixture
def csm_df_6():
    """A DataFrame containing 6 CSMs"""
    data = {
        "alpha_target": [True, True, True, False, False, False],
        "beta_target": [True, True, False, True, True, False],
        "spectrum": [1, 2, 3, 4, 5, 1],
        "group": [1, 1, 2, 2, 2, 1],
        "alpha_peptide": ["a", "b", "a", "c", "d", "e"],
        "beta_peptide": ["f", "g", "h", "i", "j", "i"],
        "alpha_protein": ["A", "B"] * 3,
        "beta_protein": ["C", "D"] * 3,
        "feature_1": [4, 3, 2, 2, 1, 0],
        "feature_2": [2, 3, 4, 1, 2, 3],
    }
    return pd.DataFrame(data)


@pytest.fixture
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
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
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
        filename_column="filename",
        scan_column="spectrum",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
        copy_data=True,
    )
    return psms


@pytest.fixture
def csm_df_1000(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    rng = np.random.Generator(np.random.PCG64(42))
    tt_hits = {
        "alpha_target": [True] * 300,
        "beta_target": [True] * 300,
        "spectrum": np.arange(300),
        "group": rng.choice(2, size=300),
        "alpha_peptide": [_random_peptide(5, rng) for _ in range(300)],
        "beta_peptide": [_random_peptide(5, rng) for _ in range(300)],
        "score": np.concatenate(
            [rng.normal(4, size=150), rng.normal(2, size=150)]
        ),
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=300),
        "expmass": rng.uniform(500, 2000, size=300),
        "ret_time": rng.uniform(0, 60 * 120, size=300),
        "charge": rng.choice([2, 3, 4], size=300),
    }

    td_hits = {
        "alpha_target": [True] * 250 + [False] * 250,
        "beta_target": [False] * 250 + [True] * 250,
        "spectrum": np.arange(500) + 300,
        "group": rng.choice(2, size=500),
        "alpha_peptide": [_random_peptide(5, rng) for _ in range(500)],
        "beta_peptide": [_random_peptide(5, rng) for _ in range(500)],
        "score": rng.normal(2, size=500),
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    dd_hits = {
        "alpha_target": [False] * 200,
        "beta_target": [False] * 200,
        "spectrum": np.arange(200) + 800,
        "group": rng.choice(2, size=200),
        "alpha_peptide": [_random_peptide(5, rng) for _ in range(200)],
        "beta_peptide": [_random_peptide(5, rng) for _ in range(200)],
        "score": rng.normal(1.5, size=200),
        "filename": "test.mzML",
        "calcmass": rng.uniform(500, 2000, size=200),
        "expmass": rng.uniform(500, 2000, size=200),
        "ret_time": rng.uniform(0, 60 * 120, size=200),
        "charge": rng.choice([2, 3, 4], size=200),
    }

    df = pd.concat(
        [pd.DataFrame(tt_hits), pd.DataFrame(td_hits), pd.DataFrame(dd_hits)]
    )

    target_peptides = np.concatenate(
        [
            df.loc[df["alpha_target"], "alpha_peptide"].values,
            df.loc[df["beta_target"], "beta_peptide"].values,
        ]
    )

    decoy_peptides = np.concatenate(
        [
            df.loc[~df["alpha_target"], "alpha_peptide"].values,
            df.loc[~df["beta_target"], "beta_peptide"].values,
        ]
    )

    fasta_data = "\n".join(
        _make_fasta(100, pd.Series(target_peptides), 20, rng)
        + _make_fasta(100, pd.Series(decoy_peptides), 20, rng, "decoy")
    )

    fasta = tmp_path / "test_1000.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)

    return df, fasta


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
        filename_column="filename",
        scan_column="spectrum",
        calcmass_column="calcmass",
        expmass_column="expmass",
        rt_column="ret_time",
        charge_column="charge",
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


@pytest.fixture
def mock_proteins():
    class proteins:
        def __init__(self):
            self.peptide_map = {"ABCDXYZ": "X|Y|Z"}
            self.shared_peptides = {"ABCDEFG": "A|B|C; X|Y|Z"}

    return proteins()


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

            self.peptides = pd.DataFrame(
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
