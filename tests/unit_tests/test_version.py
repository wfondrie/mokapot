"""Test that getting the version works"""


def test_importlib():
    """This is the fast way for Python 3.8+"""
    import mokapot

    assert mokapot.__version__ != "0.0.0"
