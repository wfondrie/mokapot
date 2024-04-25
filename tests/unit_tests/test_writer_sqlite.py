import pandas as pd
from mokapot.file_io import SqliteWriter


def test_sqlite_writer(confidence_write_data):
    df_psm = confidence_write_data["psms"]
    confidence_writer = SqliteWriter(
        ":memory:", columns=df_psm.columns.to_list()
    )
    candidate_ids = df_psm.PSMId.to_list()
    conn = prepare_tables_sqlite_db(
        confidence_writer.connection, candidate_ids
    )

    for level, df in confidence_write_data.items():
        confidence_writer.append_data(
            df, level, "q_value", "posterior_error_prob"
        )

    level_tables = {
        "psms": "CANDIDATE",
        "precursors": "PRECURSOR_VALIDATION",
        "modifiedpeptides": "MODIFIED_PEPTIDE_VALIDATION",
        "peptides": "PEPTIDE_VALIDATION",
        "peptidegroups": "PEPTIDE_GROUP_VALIDATION",
    }
    for level, level_table in level_tables.items():
        df = pd.read_sql(f"select * from {level_table};", conn)
        df_test = confidence_write_data[level]
        assert len(df_test) == len(
            df
        ), f"Rows in test data:{len(df_test)} does not match rows in sqlite database:{len(df)} for level:{level}"
        assert (
            df.isnull().sum().sum() == 0
        ), f"Null values found in database after write for level:{level}"


def prepare_tables_sqlite_db(connection, candidate_ids):
    create_table_queries = [
        "CREATE TABLE CANDIDATE (CANDIDATE_ID INTEGER NOT NULL, PSM_FDR REAL, SVM_SCORE REAL, POSTERIOR_ERROR_PROBABILITY REAL, PRIMARY KEY (CANDIDATE_ID));",
        "CREATE TABLE PRECURSOR_VALIDATION (PCM_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PCM_ID));",
        "CREATE TABLE MODIFIED_PEPTIDE_VALIDATION (MODIFIED_PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL, PRIMARY KEY (MODIFIED_PEPTIDE_ID))",
        "CREATE TABLE PEPTIDE_VALIDATION (PEPTIDE_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_ID))",
        "CREATE TABLE PEPTIDE_GROUP_VALIDATION (PEPTIDE_GROUP_ID INTEGER NOT NULL, FDR REAL, PEP REAL, SVM_SCORE REAL , PRIMARY KEY (PEPTIDE_GROUP_ID))",
    ]
    for query in create_table_queries:
        connection.execute(query)
    for c_id in candidate_ids:
        connection.execute(
            f"INSERT INTO CANDIDATE (CANDIDATE_ID) VALUES({c_id});"
        )
    return connection
