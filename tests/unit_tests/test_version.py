"""Test that the version works."""


def test_importlib():
    """The fast way for Python 3.8+."""
    import mokapot

    assert mokapot.__version__ is not None
    assert mokapot.__version__ != "0.0.0"
