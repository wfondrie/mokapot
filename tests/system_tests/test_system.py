"""
These tests run mokapot overall and verify that it generates an output
that is expected.

At least for now, this means testing the correlation between mokapot
results and Percolator results.
"""
import os
import logging

import pytest
import pandas as pd
import mokapot

logging.basicConfig(level=logging.INFO)

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


def test_compare_to_percolator(tmp_path):
    """Test that mokapot get almost the same answer as Percolator"""
    dat = mokapot.read_pin(os.path.join("data", "phospho_rep1.pin"))
    proteins = mokapot.read_fasta(os.path.join("data", "human_sp_td.fasta"))
    psms, models, scores, desc = mokapot.brew(dat)
    mokapot.assign_confidence(
        psms=psms,
        scores=scores,
        descs=desc,
        dest_dir=tmp_path,
        proteins=proteins,
        prefixes=[None],
    )

    perc_path = os.path.join("data", "percolator.{p}.txt")
    moka_path = os.path.join(tmp_path, "targets.{p}")
    perc_res = {
        p: mokapot.read_file(perc_path.format(p=p)) for p in ["proteins"]
    }
    moka_res = {
        p: mokapot.read_file(moka_path.format(p=p)) for p in ["proteins"]
    }

    for level in ["proteins"]:
        logging.info("Testing level: %s", level)
        perc = perc_res[level]
        moka = moka_res[level]
        if level != "proteins":
            merged = pd.merge(
                moka, perc, on="PSMId", suffixes=("_mokapot", "_percolator")
            )
        else:
            moka["ProteinId"] = moka["mokapot protein group"].str.split(
                ", ", expand=True
            )[0]
            merged = pd.merge(
                moka,
                perc,
                on="ProteinId",
                suffixes=("_mokapot", "_percolator"),
            )
            pd.set_option("display.max_columns", None)

        assert (
            merged["q-value_mokapot"].corr(merged["q-value_percolator"]) > 0.99
        )
        assert (
            merged["posterior_error_prob_mokapot"].corr(
                merged["posterior_error_prob_percolator"]
            )
            > 0.99
        )
