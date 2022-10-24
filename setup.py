"""
Setup the mokapot package.
"""
import setuptools
import os

setuptools.setup(
    install_requires=[
        f"mokapot_ctree @ file://localhost/{os.getcwd()}/tests/system_tests/sample_plugin/.",
    ]
)
