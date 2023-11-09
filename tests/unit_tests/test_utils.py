"""Test the utility functions."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mokapot import utils

DF = pl.DataFrame({"a": [1.0, 2], "b": [3.0, 4]})


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ("blah", ["blah"]),
        (DF, [DF]),
    ],
)
def test_listify(obj, expected):
    """Test that we can turn things into lists."""
    assert utils.listify(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (DF, DF.lazy()),
        (DF.lazy(), DF.lazy()),
        ({"a": [1.0, 2], "b": [3.0, 4]}, DF.lazy()),
    ],
)
def test_make_lazy(obj, expected):
    """Test that we can turn things into lists."""
    out = utils.make_lazy(obj)
    assert isinstance(out, pl.LazyFrame)
    assert_frame_equal(out, expected)
