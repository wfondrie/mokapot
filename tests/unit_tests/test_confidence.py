"""Test that Confidence classes are working correctly."""

import pickle

import numpy as np
import polars as pl
import pytest
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
        eval_fdr=0.2,
        rng=rng,
        proteins=proteins,
    )

    prot1 = conf.results.proteins
    rng = np.random.default_rng(42)
    conf = PsmConfidence(
        data=data,
        schema=PsmSchema(**schema_kwargs),
        scores=data["score"],
        eval_fdr=0.2,
        rng=rng,
        proteins=proteins,
    )

    prot2 = conf.results.proteins
    with pl.StringCache():
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

    str(conf)


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
    df1 = conf.results.psms

    rng = np.random.default_rng(1)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    df2 = conf.results.psms
    assert_frame_not_equal(df1, df2)

    rng = np.random.default_rng(1)
    conf = PsmConfidence(
        data=data,
        schema=schema,
        scores=data["feature"],
        eval_fdr=0.2,
        rng=rng,
    )

    df3 = conf.results.psms
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

    scores0 = conf.results.psms.collect()["mokapot q-value"]

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

    scores1 = conf.results.psms.collect()["mokapot q-value"]
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
    scores2 = conf.results.psms.collect()["mokapot q-value"]
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


@pytest.mark.parametrize(
    ("decoys", "stem", "ext", "kwargs"),
    [(False, None, "parquet", {}), (True, "foo", "bar", {"statistics": True})],
)
def test_to_parquet(mock_confidence, tmp_path, decoys, stem, ext, kwargs):
    """Test writing parquet."""
    out = mock_confidence.to_parquet(
        dest_dir=tmp_path,
        decoys=decoys,
        stem=stem,
        ext=ext,
        **kwargs,
    )

    if stem and stem is not None:
        stem = stem + "."
    else:
        stem = ""

    assert (tmp_path / f"{stem}mokapot.x.{ext}").exists()
    assert (tmp_path / f"{stem}mokapot.y.{ext}").exists()
    assert decoys == (tmp_path / f"{stem}mokapot.decoy.z.{ext}").exists()

    assert_frame_equal(
        pl.read_parquet(out[0]),
        mock_confidence.results.x.collect(),
    )
    assert_frame_equal(
        pl.read_parquet(out[1]),
        mock_confidence.results.y.collect(),
    )

    if decoys:
        assert_frame_equal(
            pl.read_parquet(out[3]),
            mock_confidence.decoy_results.z.collect(),
        )


@pytest.mark.parametrize(
    ("decoys", "stem", "ext", "separator"),
    [(False, None, "txt", "\t"), (True, "foo", "bar", ",")],
)
def test_to_txt(mock_confidence, tmp_path, decoys, stem, ext, separator):
    """Test writing parquet."""
    out = mock_confidence.to_txt(
        dest_dir=tmp_path,
        separator=separator,
        decoys=decoys,
        stem=stem,
        ext=ext,
    )

    if stem and stem is not None:
        stem = stem + "."
    else:
        stem = ""

    assert (tmp_path / f"{stem}mokapot.x.{ext}").exists()
    assert (tmp_path / f"{stem}mokapot.y.{ext}").exists()
    assert decoys == (tmp_path / f"{stem}mokapot.decoy.z.{ext}").exists()

    assert_frame_equal(
        pl.read_csv(out[0], separator=separator),
        mock_confidence.results.x.collect(),
    )
    assert_frame_equal(
        pl.read_csv(out[1], separator=separator),
        mock_confidence.results.y.collect(),
    )

    if decoys:
        assert_frame_equal(
            pl.read_csv(out[3], separator=separator),
            mock_confidence.decoy_results.z.collect(),
        )


@pytest.mark.parametrize(
    ("decoys", "stem", "ext", "kwargs"),
    [(False, None, "csv", {}), (True, "foo", "bar", {"separator": "\t"})],
)
def test_to_csv(mock_confidence, tmp_path, decoys, stem, ext, kwargs):
    """Test writing parquet."""
    out = mock_confidence.to_csv(
        dest_dir=tmp_path,
        decoys=decoys,
        stem=stem,
        ext=ext,
        **kwargs,
    )

    if stem and stem is not None:
        stem = stem + "."
    else:
        stem = ""

    assert (tmp_path / f"{stem}mokapot.x.{ext}").exists()
    assert (tmp_path / f"{stem}mokapot.y.{ext}").exists()
    assert decoys == (tmp_path / f"{stem}mokapot.decoy.z.{ext}").exists()

    assert_frame_equal(
        pl.read_csv(out[0], **kwargs),
        mock_confidence.results.x.collect(),
    )
    assert_frame_equal(
        pl.read_csv(out[1], **kwargs),
        mock_confidence.results.y.collect(),
    )

    if decoys:
        assert_frame_equal(
            pl.read_csv(out[3], **kwargs),
            mock_confidence.decoy_results.z.collect(),
        )
