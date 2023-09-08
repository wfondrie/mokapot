"""Test that Confidence classes are working correctly."""
import pickle

import numpy as np
import polars as pl
from polars.testing import (
    assert_frame_equal,
    assert_frame_not_equal,
    assert_series_equal,
    assert_series_not_equal,
)

import mokapot
from mokapot import PsmConfidence, PsmSchema


def test_with_proteins(psm_df_1000):
    """Test adding proteins."""
    data, fasta, schema_kwargs = psm_df_1000
    proteins = mokapot.read_fasta(fasta, missed_cleavages=0)

    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=PsmSchema(**schema_kwargs),
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
        proteins=proteins,
    )

    print(conf)  # Tests repr w/ proteins
    prot1 = conf.proteins

    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=PsmSchema(**schema_kwargs),
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
        proteins=proteins,
    )

    prot2 = conf.proteins
    assert_frame_equal(prot1, prot2)


def test_repr(psm_df_easy):
    """Test that repr works."""
    data, schema_kwargs = psm_df_easy
    schema = PsmSchema(**schema_kwargs)
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    print(conf)


def test_random_tie_break(psm_df_easy):
    """Test that ties are broken randomly."""
    data, schema_kwargs = psm_df_easy
    schema = PsmSchema(**schema_kwargs)
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    df1 = conf.psms

    rng = np.random.default_rng(1)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    df2 = conf.psms
    assert_frame_not_equal(df1, df2)

    rng = np.random.default_rng(1)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    df3 = conf.psms
    assert_frame_equal(df2, df3)


def test_groups(psm_df_1000):
    """Test that one group is equivalent to no group."""
    data, _, schema_kwargs = psm_df_1000
    schema = PsmSchema(group="group", **schema_kwargs)
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
    )

    scores0 = conf.psms["mokapot q-value"]

    # Set to 1 group.
    data = data.with_columns(pl.lit(0).alias("group"))
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
    )

    scores1 = conf.psms["mokapot q-value"]
    assert_series_not_equal(scores0, scores1)

    # Remove gruops:
    np.random.seed(42)
    schema.group = None
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
    )
    scores2 = conf.psms["mokapot q-value"]
    assert_series_equal(scores1, scores2)


def test_pickle(psm_df_1000, tmp_path):
    """Test that pickling works."""
    data, _, schema_kwargs = psm_df_1000
    data = data.with_columns(pl.lit(0).alias("group"))

    schema = PsmSchema(**schema_kwargs)

    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["score"],
        eval_fdr=0.01,
        rng=rng,
    )

    pkl_file = tmp_path / "results.pkl"
    with pkl_file.open("wb+") as pkl_dat:
        pickle.dump(conf, pkl_dat)

    with pkl_file.open("rb") as pkl_dat:
        pickle.load(pkl_dat)
