"""Check that writing tab-delimited files works well"""

import mokapot
import pandas as pd


def test_columns(mock_conf, tmp_path):
    """Test other specific things"""
    conf = mock_conf
    df1 = pd.read_table(mokapot.to_txt(conf, dest_dir=tmp_path)[0])
    assert df1.columns[-1] == "protein"

    test2 = mokapot.to_txt(conf, dest_dir=tmp_path, decoys=True)
    assert len(test2) == 2
