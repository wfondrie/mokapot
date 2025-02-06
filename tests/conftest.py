"""
This file contains fixtures that are used at multiple points in the tests.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from triqler.qvality import getQvaluesFromScores

from mokapot import LinearPsmDataset, OnDiskPsmDataset
from mokapot.qvalues import tdc
from mokapot.utils import make_bool_trarget

from .helpers.random_df import _psm_df_rand

## This section just adds the sorting of the tests, makes the tests marked
## with the slow marker run last.

# I am assigning slow to tests that take more than 10 seconds to run for now.


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test to run last")


def by_slow_marker(item):
    return 0 if item.get_closest_marker("slow") is None else 1


def pytest_collection_modifyitems(session, config, items):
    if config.getoption("--slow-last"):
        items.sort(key=by_slow_marker, reverse=False)


def pytest_addoption(parser, pluginmanager):
    parser.addoption("--slow-last", action="store_true", default=False)


## End of section (slow makrker)


@pytest.fixture(autouse=True)
def set_logging(caplog):
    """Add logging to everything."""
    caplog.set_level(level=logging.INFO, logger="mokapot")


@pytest.fixture(scope="session")
def psm_df_6() -> pd.DataFrame:
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
def psm_df_builder() -> callable:
    return _psm_df_rand


@pytest.fixture()
def psm_df_100(tmp_path) -> Path:
    """A DataFrame with 100 PSMs."""
    data = _psm_df_rand(50, 50, score_diffs=[3.0])
    pin = tmp_path / "test.pin"
    data.df.to_csv(pin, sep="\t", index=False)
    return pin


@pytest.fixture()
def psm_df_100_parquet(tmp_path) -> Path:
    """A DataFrame with 100 PSMs."""
    data = _psm_df_rand(50, 50)
    pf = tmp_path / "test.parquet"
    data.df.to_parquet(pf, index=False)
    return pf


@pytest.fixture()
def psm_df_1000(tmp_path) -> tuple[Path, pd.DataFrame, Path, list[str]]:
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    data = _psm_df_rand(500, 500, score_diffs=[3.0, 3.0], share_ids=True)
    df, _rng, score_cols, fasta_data = (
        data.df,
        data.rng,
        data.score_cols,
        data.fasta_string,
    )

    fasta = tmp_path / "test_1000.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(fasta_data)

    pin = tmp_path / "test.pin"
    df.to_csv(pin, sep="\t", index=False)
    return pin, df, fasta, score_cols


@pytest.fixture()
def psm_df_1000_parquet(tmp_path):
    """A DataFrame with 1000 PSMs from 500 spectra and a FASTA file."""
    # Q: is this docstring accurate? It seems to me that it is 1k psms
    #    from 1k spectra
    data = _psm_df_rand(500, 500, score_diffs=[3.0, 3.0], share_ids=True)
    fasta = tmp_path / "test_1000.fasta"
    with open(fasta, "w+") as fasta_ref:
        fasta_ref.write(data.fasta_string)

    pf = tmp_path / "test.parquet"
    data.df.drop(columns=data.score_cols).to_parquet(pf, index=False)
    return pf, data.df, fasta


@pytest.fixture
def psms_dataset(psm_df_1000) -> LinearPsmDataset:
    """A small LinearPsmDataset"""
    data = _psm_df_rand(500, 500, share_ids=True)

    psms = LinearPsmDataset(
        psms=data.df,
        target_column="target",
        spectrum_columns=data.columns.spectrum_columns,
        peptide_column="peptide",
        feature_columns=data.score_cols,
        copy_data=True,
    )
    return psms


@pytest.fixture
def psms_ondisk() -> OnDiskPsmDataset:
    """A small OnDiskPsmDataset"""
    filename = Path("data", "scope2_FP97AA.pin")
    df_spectra = pd.read_csv(
        filename,
        sep="\t",
        usecols=["ScanNr", "ExpMass", "Label"],
    )
    # Q: why is the exp mass in the spectra dataframe?
    psms = OnDiskPsmDataset(
        filename,
        target_column="Label",
        spectrum_columns=["ScanNr", "ExpMass"],
        peptide_column="Peptide",
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
        extra_confidence_level_columns=[],
        spectra_dataframe=df_spectra,
    )
    return psms


@pytest.fixture
def psms_ondisk_from_parquet() -> OnDiskPsmDataset:
    """A small OnDiskPsmDataset"""
    filename = Path("data") / "10k_psms_test.parquet"
    df_spectra = pq.read_table(
        filename, columns=["ScanNr", "ExpMass", "Label"]
    ).to_pandas()
    df_spectra["Label"] = make_bool_trarget(df_spectra["Label"])
    psms = OnDiskPsmDataset(
        filename,
        target_column="Label",
        spectrum_columns=["ScanNr", "ExpMass"],
        peptide_column="Peptide",
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
        extra_confidence_level_columns=[],
        spectra_dataframe=df_spectra,
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


@pytest.fixture
def mock_proteins():
    class proteins:
        def __init__(self):
            self.peptide_map = {"ABCDXYZ": "X|Y|Z"}
            self.shared_peptides = {"ABCDEFG": "A|B|C; X|Y|Z"}

    return proteins()


# TODO: Remove, this is only used in flashlfq tests
# .     and is not up to date with the current imeplementation.
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
                "filename": Path("a") / "b" / "c.mzML",
                "calcmass": [1, 2],
                "ret_time": [60, 120],
                "charge": [2, 3],
                "peptide": ["B.ABCD[+2.817]XYZ.A", "ABCDE(shcah8)FG"],
                "mokapot q-value": [0.001, 0.1],
                "protein": ["A|B|C\tB|C|A", "A|B|C"],
            })

    return conf()


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
    pd.set_option("display.width", 1000)  # default 80, None means auto-detect

    # Also set full precision
    # (see https://pandas.pydata.org/docs/user_guide/options.html)
    pd.set_option("display.precision", 17)


def pytest_plugin_registered(plugin, manager):
    debugger_active = hasattr(sys, "gettrace") and sys.gettrace() is not None
    if str(plugin).find("xdist.dsession.DSession") != -1:
        if debugger_active:
            manager.unregister(plugin)
