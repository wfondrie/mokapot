"""Check that writing tab-delimited files works well"""
from pathlib import Path

import pytest
import mokapot
import numpy as np
import pandas as pd


def test_sanity(psms, tmp_path):
    """Run simple sanity checks"""
    conf = psms.assign_confidence()
    test1 = conf.to_txt(dest_dir=tmp_path, file_root="test1")
    test2 = mokapot.to_txt(conf, dest_dir=tmp_path, file_root="test2")
    test3 = mokapot.to_txt([conf, conf], dest_dir=tmp_path, file_root="test3")
    with pytest.raises(ValueError):
        mokapot.to_txt("blah", dest_dir=tmp_path)

    test4 = mokapot.to_txt(conf, dest_dir=tmp_path, decoys=True)
    assert len(test1) == 2
    assert len(test4) == 4

    fnames = [Path(f).name for f in test1]
    assert fnames == ["test1.mokapot.psms.txt", "test1.mokapot.peptides.txt"]

    df1 = pd.read_table(test1[0])
    df3 = pd.read_table(test3[1])
    assert 2 * len(df1) == len(df3)


def test_columns(mock_conf, tmp_path):
    """Test other specific things"""
    conf = mock_conf
    df1 = pd.read_table(mokapot.to_txt(conf, dest_dir=tmp_path)[0])
    assert df1.columns[-1] == "protein"

    test2 = mokapot.to_txt(conf, dest_dir=tmp_path, decoys=True)
    assert len(test2) == 2
