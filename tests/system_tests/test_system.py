"""
These tests run mokapot overall and verify that it generates an output
that is expected.

At least for now, this means testing the correlation between mokapot
results and Percolator results.
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

import mokapot
from mokapot.column_defs import STANDARD_COLUMN_NAME_MAP
from mokapot.tabular_data import CSVFileReader

logging.basicConfig(level=logging.INFO)


def test_compare_to_percolator(tmp_path):
    """Test that mokapot get almost the same answer as Percolator"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
        datasets = mokapot.read_pin(
            Path("data", "phospho_rep1.pin"), max_workers=3
        )
    proteins = mokapot.read_fasta(Path("data", "human_sp_td.fasta"))
    models, scores = mokapot.brew(datasets)
    mokapot.assign_confidence(
        datasets=datasets,
        scores_list=scores,
        dest_dir=tmp_path,
        proteins=proteins,
        prefixes=[None],
        max_workers=4,
    )

    perc_path = Path("data", "percolator.{p}.txt")
    moka_path = tmp_path / "targets.{p}.tsv"

    def format_name(path, **kwargs):
        return path.with_name(path.name.format(**kwargs))

    perc_res = {
        p: CSVFileReader(format_name(perc_path, p=p)).read()
        for p in ["proteins"]
    }
    moka_res = {
        p: CSVFileReader(format_name(moka_path, p=p)).read()
        for p in ["proteins"]
    }
    PEP_COL = STANDARD_COLUMN_NAME_MAP["posterior_error_prob"]
    PG_COL = STANDARD_COLUMN_NAME_MAP["protein_group"]

    for level in ["proteins"]:
        logging.info("Testing level: %s", level)
        perc = perc_res[level]
        moka = moka_res[level]
        if level != "proteins":
            merged = pd.merge(
                moka, perc, on="PSMId", suffixes=("_mokapot", "_percolator")
            )
        else:
            moka["ProteinId"] = moka[PG_COL].str.split(", ", expand=True)[0]
            merged = pd.merge(
                moka,
                perc,
                on="ProteinId",
                suffixes=("_mokapot", "_percolator"),
            )
            pd.set_option("display.max_columns", None)

        assert merged["mokapot_qvalue"].corr(merged["q-value"]) > 0.99
        assert merged[PEP_COL].corr(merged["posterior_error_prob"]) > 0.99
