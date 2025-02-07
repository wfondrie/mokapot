import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from mokapot.tabular_data import ConfidenceSqliteWriter


@pytest.fixture
def confidence_write_data():
    filename = Path("data") / "confidence_results_test.tsv"
    psm_df = pd.read_csv(filename, sep="\t")
    precursor_df = psm_df.drop_duplicates(subset=["Precursor"])
    mod_pep_df = psm_df.drop_duplicates(subset=["ModifiedPeptide"])
    peptide_df = psm_df.drop_duplicates(subset=["peptide"])
    peptide_grp_df = psm_df.drop_duplicates(subset=["PeptideGroup"])
    df_dict = {
        "psms": psm_df,
        "precursors": precursor_df,
        "modifiedpeptides": mod_pep_df,
        "peptides": peptide_df,
        "peptidegroups": peptide_grp_df,
    }
    return df_dict


def test_sqlite_writer(confidence_write_data):
    df_psm = confidence_write_data["psms"]
    candidate_ids = df_psm.PSMId.to_list()
    connection = sqlite3.connect(database=":memory:")
    prepare_tables_sqlite_db(connection, candidate_ids)

    for level, df in confidence_write_data.items():
        confidence_writer = ConfidenceSqliteWriter(
            connection,
            columns=df_psm.columns.to_list(),
            column_types=[],
            level=level,
            qvalue_column="q_value",
            score_column="score",
            pep_column="posterior_error_prob",
        )
        confidence_writer.append_data(df)

    level_tables = {
        "psms": "CANDIDATE",
        "precursors": "PRECURSOR_VALIDATION",
        "modifiedpeptides": "MODIFIED_PEPTIDE_VALIDATION",
        "peptides": "PEPTIDE_VALIDATION",
        "peptidegroups": "PEPTIDE_GROUP_VALIDATION",
    }
    for level, level_table in level_tables.items():
        df = pd.read_sql(f"SELECT * FROM {level_table};", connection)
        df_test = confidence_write_data[level]
        assert len(df_test) == len(df), (
            f"Rows in test data:{len(df_test)} does not match rows in sqlite "
            "database:{len(df)} for level:{level}"
        )  # noqa: E501
        assert df.isnull().sum().sum() == 0, (
            f"Null values found in database after write for level:{level}"
        )


def prepare_tables_sqlite_db(connection, candidate_ids):
    create_table_queries = [
        "CREATE TABLE CANDIDATE (CANDIDATE_ID INTEGER NOT NULL, PSM_FDR REAL, SVM_SCORE REAL, POSTERIOR_ERROR_PROBABILITY REAL, PRIMARY KEY (CANDIDATE_ID));",  # noqa: E501
        "CREATE TABLE PRECURSOR_VALIDATION (PCM_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PCM_ID));",  # noqa: E501
        "CREATE TABLE MODIFIED_PEPTIDE_VALIDATION (MODIFIED_PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL, PRIMARY KEY (MODIFIED_PEPTIDE_ID))",  # noqa: E501
        "CREATE TABLE PEPTIDE_VALIDATION (PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_ID))",  # noqa: E501
        "CREATE TABLE PEPTIDE_GROUP_VALIDATION (PEPTIDE_GROUP_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_GROUP_ID))",  # noqa: E501
    ]
    for query in create_table_queries:
        connection.execute(query)
    for c_id in candidate_ids:
        connection.execute(
            f"INSERT INTO CANDIDATE (CANDIDATE_ID) VALUES({c_id});"
        )
    connection.commit()
