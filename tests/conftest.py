"""
This file contains fixtures that are used at multiple points in the tests.
"""
import pytest
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset, OnDiskPsmDataset
from triqler.qvality import getQvaluesFromScores
from mokapot.qvalues import tdc


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
def psm_df_100(tmp_path):
    """A DataFrame with 100 PSMs."""
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "specid": np.arange(50),
        "target": [True] * 50,
        "scannr": np.random.randint(0, 100, 50),
        "calcmass": rng.uniform(500, 2000, size=50),
        "expmass": rng.uniform(500, 2000, size=50),
        "group": rng.choice(2, size=50),
        "peptide": [_random_peptide(5, rng) for _ in range(50)],
        "proteins": ["_dummy" for _ in range(50)],
        "score": np.concatenate([rng.normal(3, size=20), rng.normal(size=30)]),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=50),
        "charge": rng.choice([2, 3, 4], size=50),
    }

    decoys = {
        "specid": np.arange(50, 100),
        "target": [False] * 50,
        "scannr": np.random.randint(0, 100, 50),
        "calcmass": rng.uniform(500, 2000, size=50),
        "expmass": rng.uniform(500, 2000, size=50),
        "group": rng.choice(2, size=50),
        "peptide": [_random_peptide(5, rng) for _ in range(50)],
        "proteins": ["_dummy" for _ in range(50)],
        "score": rng.normal(size=50),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=50),
        "charge": rng.choice([2, 3, 4], size=50),
    }

    pin = tmp_path / "test.pin"
    df = pd.concat([pd.DataFrame(targets), pd.DataFrame(decoys)])
    df.to_csv(pin, sep="\t", index=False)
    return pin


@pytest.fixture()
def psm_df_1000(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "specid": np.arange(500),
        "target": [True] * 500,
        "scannr": np.random.randint(0, 1000, 500),
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "group": [0 for _ in range(500)],
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "proteins": ["_dummy" for _ in range(500)],
        "score": np.concatenate(
            [rng.normal(3, size=200), rng.normal(size=300)]
        ),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    decoys = {
        "specid": np.arange(500, 1000),
        "target": [False] * 500,
        "scannr": np.random.randint(0, 1000, 500),
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "group": [0 for _ in range(500)],
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "proteins": ["_dummy" for _ in range(500)],
        "score": rng.normal(size=500),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    fasta_data = "\n".join(
        _make_fasta(100, targets["peptide"], 10, rng)
        + _make_fasta(100, decoys["peptide"], 10, rng, "decoy")
    )

    fasta = tmp_path / "test_1000.fasta"
    pin = tmp_path / "test.pin"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)
    df = pd.concat([pd.DataFrame(targets), pd.DataFrame(decoys)])
    df.to_csv(pin, sep="\t", index=False)
    return pin, df, fasta


@pytest.fixture
def psms(psm_df_1000):
    """A small LinearPsmDataset"""
    _, df, _ = psm_df_1000
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
def psms_ondisk():
    """A small OnDiskPsmDataset"""
    filename = "data/scope2_FP97AA.pin"
    df_spectra = pd.read_csv(
        filename, sep="\t", usecols=["ScanNr", "ExpMass", "Label"]
    )
    with open(filename) as perc:
        columns = perc.readline().rstrip().split("\t")
    psms = OnDiskPsmDataset(
        filename=filename,
        target_column="Label",
        spectrum_columns=["ScanNr", "ExpMass"],
        peptide_column="Peptide",
        scan_column="ScanNr",
        calcmass_column="CalcMass",
        expmass_column="ExpMass",
        rt_column="ret_time",
        charge_column=None,
        protein_column=None,
        group_column=None,
        feature_columns=[
            "CalcMass",
            "lnrSp",
            "deltLCn",
            "deltCn",
            "Sp",
            "IonFrac",
            "RefactoredXCorr",
            "NegLog10PValue",
            "NegLog10ResEvPValue",
            "NegLog10CombinePValue",
            "enzN",
            "enzC",
            "enzInt",
            "lnNumDSP",
            "dM",
            "absdM",
        ],
        metadata_columns=[
            "SpecId",
            "ScanNr",
            "Peptide",
            "Proteins",
            "Label",
        ],
        filename_column=None,
        specId_column="SpecId",
        spectra_dataframe=df_spectra,
        columns=columns,
    )
    return psms


@pytest.fixture()
def psm_files_4000(tmp_path):
    """Create test files with 1000 PSMs."""
    np.random.seed(1)
    n = 1000
    target_scores1 = np.random.normal(size=n, loc=-5, scale=2)
    target_scores2 = np.random.normal(size=n, loc=0, scale=3)
    target_scores3 = np.random.normal(size=n, loc=7, scale=4)
    decoy_scores1 = np.random.normal(size=n, loc=-9, scale=2)
    decoy_scores2 = np.random.normal(size=n, loc=4, scale=3)
    decoy_scores3 = np.random.normal(size=n, loc=12, scale=4)
    targets = pd.DataFrame(
        np.array(
            [np.ones(n), target_scores1, target_scores2, target_scores3]
        ).transpose(),
        columns=["Label", "feature1", "feature2", "feature3"],
    )
    decoys = pd.DataFrame(
        np.array(
            [-np.ones(n), decoy_scores1, decoy_scores2, decoy_scores3]
        ).transpose(),
        columns=["Label", "feature1", "feature2", "feature3"],
    )
    psms_df = pd.concat([targets, decoys]).reset_index(drop=True)
    NC = len(psms_df)
    psms_df["ScanNr"] = np.random.randint(1, NC // 2 + 1, NC)
    expmass = np.hstack(
        [
            np.random.uniform(50, 500, NC // 2),
            np.random.uniform(50, 500, NC // 2),
        ]
    )
    expmass.sort()
    psms_df["ExpMass"] = expmass
    peptides = np.hstack(
        [np.arange(1, NC // 2 + 1), np.arange(1, NC // 2 + 1)]
    )
    peptides.sort()
    psms_df["Peptide"] = peptides
    psms_df["Proteins"] = "dummy"
    psms_df = pd.concat([psms_df, psms_df]).reset_index(drop=True)
    psms_df["Specid"] = np.arange(1, len(psms_df) + 1)
    psms_df = psms_df[
        [
            "Specid",
            "Label",
            "ScanNr",
            "ExpMass",
            "feature1",
            "feature2",
            "feature3",
            "Peptide",
            "Proteins",
        ]
    ]

    psms_df = psms_df.sample(len(psms_df))
    pin1 = tmp_path / "test1.tab"
    psms_df.to_csv(pin1, sep="\t", index=False)

    psms_df["Specid"] = psms_df["Specid"].sample(len(psms_df)).values
    pin2 = tmp_path / "test2.tab"
    psms_df.to_csv(pin2, sep="\t", index=False)
    return [pin1, pin2]


@pytest.fixture()
def targets_decoys_psms_scored(tmp_path):
    psms_t = tmp_path / "targets.psms"
    psms_d = tmp_path / "decoys.psms"

    np.random.seed(1)
    n = 1000
    psm_id = np.arange(1, n * 2 + 1)
    target_scores = np.random.normal(size=n, loc=-5, scale=2)
    decoy_scores = np.random.normal(size=n, loc=-9, scale=2)
    scores = np.hstack([target_scores, decoy_scores])
    label = np.hstack([np.ones(n), -np.ones(n)])

    idx = np.argsort(-scores)
    scores = scores[idx]
    label = label[idx]
    qval = tdc(scores, label)
    pep = getQvaluesFromScores(target_scores, decoy_scores, includeDecoys=True)[1]
    peptides = np.hstack([np.arange(1, n + 1), np.arange(1, n + 1)])
    peptides.sort()
    df = pd.DataFrame(
        np.array([psm_id, peptides, label, scores, qval, pep]).transpose(),
        columns=[
            "PSMId",
            "peptide",
            "Label",
            "score",
            "q-value",
            "posterior_error_prob",
        ],
    )
    df["proteinIds"] = "dummy"
    df[df["Label"] == 1].drop("Label", axis=1).to_csv(psms_t, sep="\t", index=False)
    df[df["Label"] == -1].drop("Label", axis=1).to_csv(psms_d, sep="\t", index=False)

    return [psms_t, psms_d]


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
