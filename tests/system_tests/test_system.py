"""Compare against Percolator.

At least for now, this means testing the correlation between mokapot
results and Percolator results.
"""
import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mokapot

logging.basicConfig(level=logging.INFO)

# Warnings are errors for these tests
pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.skip("memory blowup")
def test_compare_to_percolator():
    """Test that mokapot get almost the same answer as Percolator."""
    rng = np.random.default_rng(42)
    dat = mokapot.read_pin(Path("data", "phospho_rep1.pin"), rng=rng)
    dat.proteins = mokapot.read_fasta(
        Path("data", "human_sp_td.fasta"), rng=rng
    )
    res, _ = mokapot.brew(dat, rng=rng)

    perc_path = Path("data", "percolator.{p}.txt")
    perc_res = {
        p: pl.read_csv(
            perc_path.format(p=p),
            separator="\t",
            truncate_ragged_lines=True,
        )
        for p in ["psms", "peptides", "proteins"]
    }

    for level in ["psms", "peptides", "proteins"]:
        logging.info("Testing level: %s", level)

        if level != "proteins":
            on = ["SpecId", "Peptide"]
            perc = perc_res[level].rename(
                {"PSMId": "SpecId", "peptide": "Peptide"}
            )
        else:
            on = "mokapot protein group"
            perc = (
                perc_res[level]
                .with_columns(pl.col("ProteinId").str.replace(", ", ","))
                .rename({"ProteinId": "mokapot protein group"})
            )

        merged = res.results[level].join(perc, on=on, how="inner")
        assert 0.99 < (
            merged.select(pl.corr("mokapot q-value", "q-value")).item()
        )
        assert 0.99 < (
            merged.select(
                pl.corr("mokapot PEP", "posterior_error_prob")
            ).item()
        )
