"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

import sqlite3
import pytest
from pathlib import Path

from mokapot.tabular_data import CSVFileReader
from ..helpers.cli import run_mokapot_cli


@pytest.fixture
def pin_file():
    """Get the scope-ms files"""
    # return Path("data", "10k_psms_test.pin")
    return Path("data", "percolator-noSplit-extended-1000.tab")


@pytest.fixture
def sqlite_db_file(pin_file, tmp_path):
    db_file = Path(tmp_path, "sqlite.db")
    db_file.unlink(missing_ok=True)
    connection = sqlite3.connect(db_file)
    create_table_queries = [
        "CREATE TABLE CANDIDATE (CANDIDATE_ID INTEGER NOT NULL, PSM_FDR REAL, SVM_SCORE REAL, POSTERIOR_ERROR_PROBABILITY REAL, PRIMARY KEY (CANDIDATE_ID));",  # noqa: E501
        "CREATE TABLE PRECURSOR_VALIDATION (PCM_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PCM_ID));",  # noqa: E501
        "CREATE TABLE MODIFIED_PEPTIDE_VALIDATION (MODIFIED_PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL, PRIMARY KEY (MODIFIED_PEPTIDE_ID))",  # noqa: E501
        "CREATE TABLE PEPTIDE_VALIDATION (PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_ID))",  # noqa: E501
        "CREATE TABLE PEPTIDE_GROUP_VALIDATION (PEPTIDE_GROUP_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_GROUP_ID))",  # noqa: E501
    ]
    for query in create_table_queries:
        connection.execute(query)

    reader = CSVFileReader(pin_file)
    df = reader.read()
    candidate_ids = df["SpecId"].values
    for i, c_id in enumerate(candidate_ids):
        connection.execute(
            "INSERT INTO CANDIDATE (CANDIDATE_ID) VALUES(:id);",
            {"id": int(c_id)},
        )
    connection.commit()
    return db_file


def test_sqlite_output(tmp_path, pin_file, sqlite_db_file):
    """Test that basic cli works."""
    params = [
        pin_file,
        ("--dest_dir", tmp_path),
        ("--sqlite_db_path", sqlite_db_file),
        ("--train_fdr", "0.1"),
        ("--test_fdr", "0.2"),
        ("--peps_algorithm", "hist_nnls"),
    ]
    run_mokapot_cli(params)

    # Basic check that there are sufficiently many rows in the result tables
    tables_and_cols = [
        ("CANDIDATE", "PSM_FDR", 600),
        ("PRECURSOR_VALIDATION", "FDR", 300),
        ("MODIFIED_PEPTIDE_VALIDATION", "FDR", 300),
        ("PEPTIDE_VALIDATION", "FDR", 300),
        ("PEPTIDE_GROUP_VALIDATION", "FDR", 300),
    ]

    connection = sqlite3.connect(sqlite_db_file)
    for table_name, column_name, min_rows in tables_and_cols:
        cursor = connection.execute(
            f"SELECT * FROM {table_name} WHERE {column_name} IS NOT NULL;"
        )
        rows = cursor.fetchall()
        assert len(rows) > min_rows
