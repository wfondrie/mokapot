"""
These tests verify that the CLI works as expected.

At least for now, they do not check the correctness of the
output, just that the expect outputs are created.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from mokapot.tabular_data import CSVFileReader

from ..helpers.cli import run_mokapot_cli
from ..helpers.utils import ColumnValidator, TableValidator


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
        ("--verbosity", "3"),
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

    pep_column_validator = ColumnValidator(
        name="PEP",
        col_type="float64",
        value_range=(0.0, 0.9999),
        allow_missing=False,
    )

    # NOTE: The SVM score will actually be the best feature if
    # the best feature outperforms the model.
    svm_score_validator = ColumnValidator(
        name="SVM_SCORE",
        col_type="float64",
        value_range=(-20, 20),
        allow_missing=False,
    )
    fdr_validator = ColumnValidator(
        name="FDR",
        col_type="float64",
        value_range=(1e-32, 1.0),
        allow_missing=False,
    )

    validators = {
        "CANDIDATE": TableValidator(
            columns=[
                ColumnValidator(
                    name="CANDIDATE_ID",
                    col_type="int64",
                    value_range=(1, 200_000),
                    allow_missing=False,
                ),
                ColumnValidator(
                    name="PSM_FDR",
                    col_type="float64",
                    value_range=(1e-32, 1.0),
                    allow_missing=True,
                ),
                ColumnValidator(
                    name="SVM_SCORE",
                    col_type="float64",
                    value_range=(-20, 20),
                    allow_missing=True,
                ),
                ColumnValidator(
                    name="POSTERIOR_ERROR_PROBABILITY",
                    col_type="float64",
                    value_range=(0.0, 0.999),
                    allow_missing=True,
                ),
            ],
            allow_extra=False,
            row_range=(1000, 1000),
        ),
        "PRECURSOR_VALIDATION": TableValidator(
            columns=[
                ColumnValidator(
                    name="PCM_ID",
                    col_type="int64",
                    value_range=(1_000_000, 2_500_000),
                    allow_missing=False,
                ),
                pep_column_validator,
                svm_score_validator,
                fdr_validator,
            ],
            allow_extra=False,
            row_range=(356 - 20, 356 + 20),
        ),
        "MODIFIED_PEPTIDE_VALIDATION": TableValidator(
            columns=[
                ColumnValidator(
                    name="MODIFIED_PEPTIDE_ID",
                    col_type="int64",
                    value_range=(12781, 22445),
                    allow_missing=False,
                ),
                fdr_validator,
                pep_column_validator,
                svm_score_validator,
            ],
            allow_extra=False,
            row_range=(356, 356),
        ),
        "PEPTIDE_VALIDATION": TableValidator(
            columns=[
                ColumnValidator(
                    name="PEPTIDE_ID",
                    col_type="int64",
                    value_range=(4689, 2077305),
                    allow_missing=False,
                ),
                fdr_validator,
                pep_column_validator,
                svm_score_validator,
            ],
            allow_extra=False,
            row_range=(356 - 20, 356 + 20),
        ),
        "PEPTIDE_GROUP_VALIDATION": TableValidator(
            columns=[
                ColumnValidator(
                    name="PEPTIDE_GROUP_ID",
                    col_type="int64",
                    value_range=(9875, 21599),
                    allow_missing=False,
                ),
                fdr_validator,
                pep_column_validator,
                svm_score_validator,
            ],
            allow_extra=False,
            row_range=(356 - 20, 356 + 20),
        ),
    }

    connection = sqlite3.connect(sqlite_db_file)
    for table_name, column_name, min_rows in tables_and_cols:
        cursor = connection.execute(
            f"SELECT * FROM {table_name} WHERE {column_name} IS NOT NULL;"
        )
        rows = cursor.fetchall()
        assert len(rows) > min_rows

        # Test that the data is correct
        df = pd.read_sql(f"SELECT * FROM {table_name};", connection)
        validator = validators[table_name]
        validator.validate(df)
