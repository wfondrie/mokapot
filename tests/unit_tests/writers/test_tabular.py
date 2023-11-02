"""Check that writing tab-delimited files works well."""
from functools import partial

import polars as pl
import pytest

import mokapot


@pytest.fixture
def conf(psms):
    """A confidence object."""
    return psms.assign_confidence()


@pytest.mark.parametrize(
    ("writer", "reader"),
    [
        ("to_txt", pl.read_csv),
        ("to_csv", pl.read_csv),
        ("to_parquet", pl.read_parquet),
    ],
)
@pytest.mark.parametrize("obj", (False, True))
@pytest.mark.parametrize(
    ("stem", "decoys", "ext", "out"),
    [
        (None, False, "blah", "mokapot.psms.blah"),
        ("blah", True, "foo", "blah.mokapot.decoy.psms.foo"),
    ],
)
def test_sanity(conf, obj, writer, reader, tmp_path, stem, decoys, ext, out):
    """Run simple sanity checks."""
    out = tmp_path / out
    if obj:
        write_fn = getattr(conf, writer)
    else:
        write_fn = partial(getattr(mokapot, writer), conf=conf)

    if writer == "to_txt":
        reader = partial(reader, separator="\t")

    test = write_fn(
        dest_dir=tmp_path,
        stem=stem,
        decoys=decoys,
        ext=ext,
    )

    assert out in test
    assert len(test) == 2 + (2 * decoys)
    df = reader(out)  # Make sure it's parsable.
    assert len(df.columns) == 11
    assert "mokapot q-value" in df.columns
    assert "mokapot PEP" in df.columns
