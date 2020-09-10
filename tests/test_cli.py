"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""
import os
import subprocess

FILES = [
    os.path.join("data", f)
    for f in os.listdir("data")
    if f.startswith("scope")
]


def test_basic_cli(tmp_path):
    """Test that basic cli works."""
    cmd = ["mokapot", FILES[0], "--dest_dir", tmp_path]
    subprocess.run(cmd, check=True)
    assert os.path.isfile(os.path.join(tmp_path, "mokapot.psms.txt"))
    assert os.path.isfile(os.path.join(tmp_path, "mokapot.peptides.txt"))


def test_cli_options(tmp_path):
    """Test non-defaults"""
    cmd = [
        "mokapot",
        FILES[0],
        FILES[1],
        "--dest_dir",
        tmp_path,
        "--file_root",
        "blah",
        "--train_fdr",
        "0.2",
        "--test_fdr",
        "0.1",
        "--max_iter",
        "2",
        "--seed",
        "100",
        "--direction",
        "RefactoredXCorr",
        "--folds",
        "2",
        "-v",
        "1",
    ]

    subprocess.run(cmd, check=True)
    file_bases = [os.path.basename(os.path.splitext(f)[0]) for f in FILES[0:2]]

    assert os.path.isfile(
        os.path.join(tmp_path, f"blah.{file_bases[0]}.mokapot.psms.txt")
    )
    assert os.path.isfile(
        os.path.join(tmp_path, f"blah.{file_bases[0]}.mokapot.peptides.txt")
    )
    assert os.path.isfile(
        os.path.join(tmp_path, f"blah.{file_bases[1]}.mokapot.psms.txt")
    )
    assert os.path.isfile(
        os.path.join(tmp_path, f"blah.{file_bases[1]}.mokapot.peptides.txt")
    )


def test_cli_aggregate(tmp_path):
    """Test that aggregate results in one result file."""
    cmd = [
        "mokapot",
        FILES[0],
        FILES[1],
        "--dest_dir",
        tmp_path,
        "--file_root",
        "blah",
        "--aggregate",
    ]

    subprocess.run(cmd, check=True)
    file_bases = [os.path.basename(os.path.splitext(f)[0]) for f in FILES[0:2]]

    assert os.path.isfile(os.path.join(tmp_path, "blah.mokapot.psms.txt"))
    assert os.path.isfile(os.path.join(tmp_path, "blah.mokapot.peptides.txt"))
