"""
This file contains fixtures that are used at multiple points in the tests.
"""

import logging
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from mokapot import LinearPsmDataset, OnDiskPsmDataset
from triqler.qvality import getQvaluesFromScores
from mokapot.qvalues import tdc
import pyarrow.parquet as pq
import pyarrow as pa
from mokapot.utils import convert_targets_column


@pytest.fixture(autouse=True)
def set_logging(caplog):
    """Add logging to everything."""
    caplog.set_level(level=logging.INFO, logger="mokapot")


@pytest.fixture(scope="session")
def psm_df_6():
    """A DataFrame containing 6 PSMs"""
    data = {
        "target": [True, True, True, False, False, False],
        "spectrum": [1, 2, 3, 4, 5, 1],
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
def psm_df_100_parquet(tmp_path):
    """A DataFrame with 100 PSMs."""
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "specid": np.arange(50),
        "target": [True] * 50,
        "scannr": np.random.randint(0, 100, 50),
        "calcmass": rng.uniform(500, 2000, size=50),
        "expmass": rng.uniform(500, 2000, size=50),
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
        "peptide": [_random_peptide(5, rng) for _ in range(50)],
        "proteins": ["_dummy" for _ in range(50)],
        "score": rng.normal(size=50),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=50),
        "charge": rng.choice([2, 3, 4], size=50),
    }

    pf = tmp_path / "test.parquet"
    df = pd.concat([pd.DataFrame(targets), pd.DataFrame(decoys)])
    df.to_parquet(pf, index=False)
    return pf


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
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "proteins": ["_dummy" for _ in range(500)],
        "score": np.concatenate([
            rng.normal(3, size=200),
            rng.normal(size=300),
        ]),
        "score2": np.concatenate([
            rng.normal(3, size=200),
            rng.normal(size=300),
        ]),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    decoys = {
        "specid": np.arange(500, 1000),
        "target": [False] * 500,
        "spectrum": np.arange(500),
        "score2": rng.normal(size=500),
        "scannr": np.random.randint(0, 1000, 500),
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
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


@pytest.fixture()
def psm_df_1000_parquet(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    rng = np.random.Generator(np.random.PCG64(42))
    targets = {
        "specid": np.arange(500),
        "target": [True] * 500,
        "scannr": np.random.randint(0, 1000, 500),
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
        "peptide": [_random_peptide(5, rng) for _ in range(500)],
        "proteins": ["_dummy" for _ in range(500)],
        "score": np.concatenate([
            rng.normal(3, size=200),
            rng.normal(size=300),
        ]),
        "score2": np.concatenate([
            rng.normal(3, size=200),
            rng.normal(size=300),
        ]),
        "filename": "test.mzML",
        "ret_time": rng.uniform(0, 60 * 120, size=500),
        "charge": rng.choice([2, 3, 4], size=500),
    }

    decoys = {
        "specid": np.arange(500, 1000),
        "target": [False] * 500,
        "spectrum": np.arange(500),
        "score2": rng.normal(size=500),
        "scannr": np.random.randint(0, 1000, 500),
        "calcmass": rng.uniform(500, 2000, size=500),
        "expmass": rng.uniform(500, 2000, size=500),
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
    pf = tmp_path / "test.parquet"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)
    df = pd.concat([pd.DataFrame(targets), pd.DataFrame(decoys)])
    df.to_parquet(pf, index=False)
    return pf, df, fasta


@pytest.fixture
def psms(psm_df_1000):
    """A small LinearPsmDataset"""
    _, df, _ = psm_df_1000
    psms = LinearPsmDataset(
        psms=df,
        target_column="target",
        spectrum_columns="spectrum",
        peptide_column="peptide",
        feature_columns=["score", "score2"],
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
    filename = Path("data", "scope2_FP97AA.pin")
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
        rt_column=None,
        charge_column=None,
        protein_column=None,
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
        metadata_column_types=["int", "int", "int", "string", "int"],
        level_columns=["Peptide"],
        filename_column=None,
        specId_column="SpecId",
        spectra_dataframe=df_spectra,
        columns=columns,
    )
    return psms


@pytest.fixture
def psms_ondisk_from_parquet():
    """A small OnDiskPsmDataset"""
    filename = Path("data/10k_psms_test.parquet")
    df_spectra = pq.read_table(
        filename, columns=["ScanNr", "ExpMass", "Label"]
    ).to_pandas()
    df_spectra = convert_targets_column(df_spectra, "Label")
    columns = pq.ParquetFile(filename).schema.names
    psms = OnDiskPsmDataset(
        filename=filename,
        target_column="Label",
        spectrum_columns=["ScanNr", "ExpMass"],
        peptide_column="Peptide",
        scan_column="ScanNr",
        calcmass_column=None,
        expmass_column="ExpMass",
        rt_column=None,
        charge_column=None,
        protein_column="Proteins",
        feature_columns=[
            "Mass",
            "MS8_feature_5",
            "missedCleavages",
            "MS8_feature_7",
            "MS8_feature_13",
            "MS8_feature_20",
            "MS8_feature_21",
            "MS8_feature_22",
            "MS8_feature_24",
            "MS8_feature_29",
            "MS8_feature_30",
            "MS8_feature_32",
        ],
        metadata_columns=[
            "SpecId",
            "Label",
            "ScanNr",
            "Peptide",
            "Proteins",
            "ExpMass",
        ],
        metadata_column_types=[
            pa.int64(),
            pa.int64(),
            pa.int64(),
            pa.string(),
            pa.int64(),
        ],
        level_columns=["Peptide"],
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
        np.array([
            np.ones(n),
            target_scores1,
            target_scores2,
            target_scores3,
        ]).transpose(),
        columns=["Label", "feature1", "feature2", "feature3"],
    )
    decoys = pd.DataFrame(
        np.array([
            -np.ones(n),
            decoy_scores1,
            decoy_scores2,
            decoy_scores3,
        ]).transpose(),
        columns=["Label", "feature1", "feature2", "feature3"],
    )
    psms_df = pd.concat([targets, decoys]).reset_index(drop=True)
    NC = len(psms_df)
    psms_df["ScanNr"] = np.random.randint(1, NC // 2 + 1, NC)
    expmass = np.hstack([
        np.random.uniform(50, 500, NC // 2),
        np.random.uniform(50, 500, NC // 2),
    ])
    expmass.sort()
    psms_df["ExpMass"] = expmass
    peptides = np.hstack([
        np.arange(1, NC // 2 + 1),
        np.arange(1, NC // 2 + 1),
    ])
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
    pep = getQvaluesFromScores(
        target_scores, decoy_scores, includeDecoys=True
    )[1]
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
    df[df["Label"] == 1].drop("Label", axis=1).to_csv(
        psms_t, sep="\t", index=False
    )
    df[df["Label"] == -1].drop("Label", axis=1).to_csv(
        psms_d, sep="\t", index=False
    )

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
    """Create a mock-up of a LinearConfidence object"""

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

            self.peptides = pd.DataFrame({
                "filename": "a/b/c.mzML",
                "calcmass": [1, 2],
                "ret_time": [60, 120],
                "charge": [2, 3],
                "peptide": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
                "mokapot q-value": [0.001, 0.1],
                "protein": ["A|B|C\tB|C|A", "A|B|C"],
            })

            self.confidence_estimates = {"peptides": self.peptides}
            self.decoy_confidence_estimates = {"peptides": self.peptides}

    return conf()


@pytest.fixture
def merge_sort_data(tmp_path):
    filenames_csv = [tmp_path / f"merge_sort_{i}.csv" for i in range(3)]
    filenames_parquet = [
        tmp_path / f"merge_sort_{i}.parquet" for i in range(3)
    ]
    df = pd.read_csv(
        "data/10k_psms_test.pin",
        sep="\t",
        usecols=[
            "SpecId",
            "Label",
            "ScanNr",
            "ExpMass",
            "Peptide",
            "Proteins",
        ],
    )
    df = df[:15]
    df["score"] = np.arange(0.16, 0.01, -0.01)
    for i, (file_csv, file_parquet) in enumerate(
        zip(filenames_csv, filenames_parquet)
    ):
        df[i::3].to_csv(file_csv, sep="\t", index=False)
        df[i::3].to_parquet(file_parquet, index=False)
    return filenames_csv, filenames_parquet


@pytest.fixture
def confidence_write_data():
    filename = Path("data/confidence_results_test.tsv")
    psm_df = pd.read_csv(filename, sep="\t")
    precursor_df = psm_df.drop_duplicates(subset=["Precursor"])
    mod_pep_df = psm_df.drop_duplicates(subset=["ModifiedPeptide"])
    peptide_df = psm_df.drop_duplicates(subset=["peptide"])
    peptide_grp_df = psm_df.drop_duplicates(subset=["PeptideGroup"])
    df_dict = {
        "psms": psm_df,
        "precursors": precursor_df,
        "modifiedpeptides": mod_pep_df,
        "peptides": peptide_df,
        "peptidegroups": peptide_grp_df,
    }
    return df_dict


@pytest.fixture
def peptide_csv_file(tmp_path):
    file = tmp_path / "peptides.csv"
    with open(file, "w") as f:
        f.write("PSMId\tLabel\tPeptide\tscore\tproteinIds\n")
    yield file
    file.unlink()


@pytest.fixture
def psms_iterator():
    """Create a standard psms iterable"""
    return [
        {
            "PSMId": "1",
            "Label": "1",
            "Peptide": "HLAQLLR",
            "score": "-5.75",
            "q-value": "0.108",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "2",
            "Label": "0",
            "Peptide": "HLAQLLR",
            "score": "-5.81",
            "q-value": "0.109",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "3",
            "Label": "0",
            "Peptide": "NVPTSLLK",
            "score": "-5.83",
            "q-value": "0.11",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "4",
            "Label": "1",
            "Peptide": "QILVQLR",
            "score": "-5.92",
            "q-value": "0.12",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "5",
            "Label": "1",
            "Peptide": "HLAQLLR",
            "score": "-6.05",
            "q-value": "0.13",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "6",
            "Label": "0",
            "Peptide": "QILVQLR",
            "score": "-6.06",
            "q-value": "0.14",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
        {
            "PSMId": "7",
            "Label": "1",
            "Peptide": "SRTSVIPGPK",
            "score": "-6.12",
            "q-value": "0.15",
            "posterior_error_prob": "1.0",
            "proteinIds": "_.dummy._",
        },
    ]


def pytest_sessionstart(session):
    # Set pandas max_columns such, that when debugging you can see all columns
    # of a dataframe instead of just a few
    pd.set_option("display.max_columns", None)

    # Set max width per column
    # pd.set_option("display.max_colwidth", None) #default 50

    # Set max width for output of the whole data frame
    pd.set_option("display.width", None)  # default 80, None means auto-detect

    # Also set full precision
    # (see https://pandas.pydata.org/docs/user_guide/options.html)
    pd.set_option("display.precision", 17)


def pytest_plugin_registered(plugin, manager):
    debugger_active = hasattr(sys, "gettrace") and sys.gettrace() is not None
    if str(plugin).find("xdist.dsession.DSession") != -1:
        if debugger_active:
            manager.unregister(plugin)
