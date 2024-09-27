from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from typeguard import typechecked

from mokapot.tabular_data import TabularDataWriter, TabularDataReader


@typechecked
class ParquetFileWriter(TabularDataWriter):
    """
    This class is responsible for writing tabular data into Parquet files.


    Attributes:
    -----------
    file_name : Path
        The path to the Parquet file being written.
    """

    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list[np.dtype],
    ):
        super().__init__(columns, column_types)
        self.file_name = file_name
        self.writer = None

    def __str__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    @staticmethod
    def _from_numpy_dtype(type):
        if type == "object":
            return pa.string()
        else:
            return pa.from_numpy_dtype(type)

    def _get_schema(self):
        schema = [
            (name, ParquetFileWriter._from_numpy_dtype(type))
            for name, type in zip(self.columns, self.column_types)
        ]
        return pa.schema(schema)

    def initialize(self):
        if len(self.column_types) > 0:
            self.writer = pq.ParquetWriter(
                self.file_name, schema=self._get_schema()
            )

    def finalize(self):
        self.writer.close()

    def append_data(self, data: pd.DataFrame):
        if self.writer is None:
            # Infer the schema from the first dataframe
            if self.column_types is None or len(self.column_types) == 0:
                self.column_types = data.dtypes.to_list()
            self.initialize()

        schema = self._get_schema()
        table = pa.Table.from_pandas(data, preserve_index=False, schema=schema)
        self.writer.write_table(table)

    def write(self, data: pd.DataFrame):
        data.to_parquet(self.file_name, index=False)

    def get_associated_reader(self):
        return ParquetFileReader(self.file_name)


@typechecked
class ParquetFileReader(TabularDataReader):
    """
    A class for reading Parquet files and retrieving data in tabular format.

    Attributes:
    -----------
    file_name : Path
        The path to the Parquet file.
    """

    def __init__(self, file_name: Path):
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileReader({self.file_name=})"

    def __repr__(self):
        return f"ParquetFileReader({self.file_name=})"

    def get_column_names(self) -> list[str]:
        return pq.ParquetFile(self.file_name).schema.names

    def get_column_types(self) -> list[np.dtype]:
        schema = pq.ParquetFile(self.file_name).schema
        pq_types = schema.to_arrow_schema().types
        return [np.dtype(type.to_pandas_dtype()) for type in pq_types]

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = pq.read_table(self.file_name, columns=columns).to_pandas()
        return result

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        pf = pq.ParquetFile(self.file_name)

        for i, record_batch in enumerate(
            pf.iter_batches(chunk_size, columns=columns)
        ):
            df = record_batch.to_pandas()
            df.index += i * chunk_size
            yield df

    def get_default_extension(self) -> str:
        return ".parquet"
