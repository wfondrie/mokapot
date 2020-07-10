"""
These tests run mokapot overall and verify that it generates an output
that is expected.

At least for now, this means testing the correlation between mokapot
results and Percolator results.
"""
import os
import logging

import pandas as pd
import mokapot

logging.basicConfig(level=logging.INFO)


def test_compare_to_percolator():
    """Test that mokapot get almost the same answer as Percolator"""
    dat = mokapot.read_pin(os.path.join("data", "phospho_rep1.pin"))
    res = mokapot.brew(dat)

    perc_path = os.path.join("data", "percolator.{l}.txt")
    perc_res = {l: mokapot.parsers.read_percolator(perc_path.format(l=l))
                for l in ["psms", "peptides"]}

    for level in ["psms", "peptides"]:
        logging.info("Testing level: %s", level)
        perc = perc_res[level].rename(columns={"PSMId": "SpecId",
                                               "peptide": "Peptide"})
        merged = pd.merge(res._confidence_estimates[level], perc)

        assert merged["mokapot q-value"].corr(merged["q-value"]) > 0.99
        assert merged["mokapot PEP"].corr(merged["posterior_error_prob"]) > 0.99
