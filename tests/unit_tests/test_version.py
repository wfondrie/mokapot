"""Test that getting the version works"""


def test_importlib():
    """This is the fast way for Python 3.8+"""
    import mokapot

    assert mokapot.__version__ != "0.0.0"


def test_setuptools():
    """We use this for Python < 3.8"""
    import sys

    sys.modules["importlib.metadata"] = None
    import mokapot

    assert mokapot.__version__ != "0.0.0"
