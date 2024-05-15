import sqlite3
import warnings
from contextlib import contextmanager
from typing import Generator, List

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import pyarrow.parquet as pq
from typeguard import typechecked
from numpy import dtype
import pyarrow as pa

CSV_SUFFIXES = [
    ".csv",
    ".pin",
    ".tab",
    ".peptides",
    ".psms",
    ".proteins",
    ".modifiedpeptides",
    ".peptidegroups",
    ".modified_peptides",
    ".peptide_groups",
    ".precursors",
]
PARQUET_SUFFIXES = [".parquet"]
SQLITE_SUFFIXES = [".db"]


def get_score_column_type(suffix):
    if suffix in PARQUET_SUFFIXES:
        return pa.float64()
    elif suffix in CSV_SUFFIXES:
        return "float"
    else:
        raise ValueError(f"Suffix '{suffix}' does not match expected formats")


@typechecked
class TabularDataReader(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list[dtype]:
        raise NotImplementedError

    @abstractmethod
    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    def _returned_dataframe_is_mutable(self):
        return True

    @staticmethod
    def from_path(
        file_name: Path, column_map: dict[str, str] | None = None, **kwargs
    ) -> "TabularDataReader":
        # Currently, we look only at the suffix, however, in the future we could
        # also look into the file itself (is it ascii? does it have some "magic
        # bytes"? ...)
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            reader = CSVFileReader(file_name, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            reader = ParquetFileReader(file_name, **kwargs)
        else:
            # Fallback
            warnings.warn(
                f"Suffix '{suffix}' not recognized in file name '{file_name}'."
                " Falling back to CSV..."
            )
            reader = CSVFileReader(file_name, **kwargs)

        if column_map is not None:
            reader = ColumnMappedReader(reader, column_map)

        return reader


@typechecked
class ColumnMappedReader(TabularDataReader):
    def __init__(self, reader: TabularDataReader, column_map: dict[str, str]):
        self.reader = reader
        self.column_map = column_map

    def get_column_names(self) -> list[str]:
        return [
            self.column_map.get(column, column)
            for column in self.reader.get_column_names()
        ]

    def get_column_types(self) -> list[dtype]:
        return self.reader.get_column_types()

    def _get_orig_columns(self, columns: list[str] | None) -> list[str] | None:
        if columns is None:
            return None

        all_orig_columns = self.reader.get_column_names()
        all_columns = self.get_column_names()
        reverse_column_map = dict(zip(all_columns, all_orig_columns))
        orig_columns = [reverse_column_map[column] for column in columns]
        return orig_columns

    def _get_mapped_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # todo: enable this again...
        # if self._returned_dataframe_is_mutable():
        #     df.rename(columns=self.column_map, inplace=True, copy=False)
        # else:
        #     df = df.rename(columns=self.column_map, inplace=False, copy=False)
        df = df.rename(columns=self.column_map, inplace=False)
        return df

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.reader.read(columns=self._get_orig_columns(columns))
        return self._get_mapped_dataframe(df)

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        orig_columns = self._get_orig_columns(columns)
        for chunk in self.reader.get_chunked_data_iterator(
            chunk_size, columns=orig_columns
        ):
            yield self._get_mapped_dataframe(chunk)


def _types_from_dataframe(df: pd.DataFrame) -> list[dtype]:
    type_map = df.dtypes
    column_names = df.columns.tolist()
    return [type_map[column_name] for column_name in column_names]


@typechecked
class CSVFileReader(TabularDataReader):
    def __init__(self, file_name: Path, sep: str = "\t"):
        self.file_name = file_name
        self.stdargs = {"sep": sep, "index_col": False}

    def __str__(self):
        return f"CSVFileReader({self.file_name=})"

    def __repr__(self):
        return f"CSVFileReader({self.file_name=},{self.stdargs=})"

    def get_column_names(self) -> list[str]:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=0)
        return df.columns.tolist()

    def get_column_types(self) -> list[dtype]:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=2)
        return _types_from_dataframe(df)

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = pd.read_csv(self.file_name, usecols=columns, **self.stdargs)
        return result if columns is None else result[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        for chunk in pd.read_csv(
            self.file_name,
            usecols=columns,
            chunksize=chunk_size,
            **self.stdargs,
        ):
            yield chunk if columns is None else chunk[columns]


@typechecked
class DataFrameReader(TabularDataReader):
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __str__(self):
        return f"DataFrameReader({self.df.columns=})"

    def __repr__(self):
        return f"DataFrameReader({self.df=})"

    def get_column_names(self) -> list[str]:
        return self.df.columns.tolist()

    def get_column_types(self) -> list[dtype]:
        return _types_from_dataframe(self.df)

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        return self.df if columns is None else self.df[columns]

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        for pos in range(0, len(self.df), chunk_size):
            chunk = self.df.iloc[pos : pos + chunk_size]
            yield chunk if columns is None else chunk[columns]

    def _returned_dataframe_is_mutable(self):
        return False

    @staticmethod
    def from_series(series: pd.Series, name=None) -> "DataFrameReader":
        if name is not None:
            return DataFrameReader(series.to_frame(name=name))
        else:
            return DataFrameReader(series.to_frame())

    @staticmethod
    def from_array(array: list | np.ndarray, name: str) -> "DataFrameReader":
        return DataFrameReader(pd.DataFrame({name: array}))


@typechecked
class ParquetFileReader(TabularDataReader):
    def __init__(self, file_name: Path):
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileReader({self.file_name=})"

    def __repr__(self):
        return f"ParquetFileReader({self.file_name=})"

    def get_column_names(self) -> list[str]:
        return pq.ParquetFile(self.file_name).schema.names

    def get_column_types(self) -> list:
        return pq.ParquetFile(self.file_name).schema.to_arrow_schema().types

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
            df.index = df.index + i * chunk_size
            yield df


@typechecked
class TabularDataWriter(ABC):
    def __init__(
        self,
        columns: list[str],
        column_types: list | None = None,
    ):
        self.columns = columns
        self.column_types = column_types
        # todo: I think the TDW should have a field/option that says whether
        #  data should be appended or whether the file should be cleared first
        #  if it already contains data

    def get_column_names(self) -> list[str]:
        return self.columns

    def get_column_types(self) -> list:
        return self.column_types

    @abstractmethod
    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def check_valid_data(self, data: pd.DataFrame):
        # todo: maybe an exception would be better suited than an assert
        columns = data.columns.tolist()
        if not columns == self.get_column_names():
            raise ValueError(
                f"Column names {columns} do not match {self.get_column_names()}"
            )

        if self.column_types is not None:
            column_types = _types_from_dataframe(data)
            if not column_types == self.get_column_types():
                raise ValueError(
                    f"Column types {column_types} do not match {self.get_column_types()}"
                )

    def write(self, data: pd.DataFrame):
        self.check_valid_data(data)
        self.initialize()
        self.append_data(data)
        self.finalize()

    def initialize(self):
        pass

    def finalize(self):
        pass

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finalize()

    @abstractmethod
    def get_associated_reader(self):
        raise NotImplementedError

    @staticmethod
    def from_suffix(
        file_name: Path, columns: list[str], buffer_size: int = 0, **kwargs
    ) -> "TabularDataWriter":
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            writer = CSVFileWriter(file_name, columns, **kwargs)
        elif suffix in PARQUET_SUFFIXES:
            writer = ParquetFileWriter(file_name, columns, **kwargs)
        elif suffix in SQLITE_SUFFIXES:
            writer = SqliteWriter(file_name, columns, **kwargs)
        else:  # Fallback
            warnings.warn(
                f"Suffix '{suffix}' not recognized in file name '{file_name}'."
                " Falling back to CSV..."
            )
            writer = CSVFileWriter(file_name, columns, **kwargs)

        if buffer_size > 1:
            writer = BufferedWriter(writer, buffer_size)
        return writer


@contextmanager
# @typechecked
def auto_finalize(writers: list[TabularDataWriter]):
    # todo: this method should actually (to be really secure), check which
    #  writers were correctly initialized and if some initialization throws an
    #  error, finalize all that already have been initialized. Similar with
    #  errors during finalization.
    for writer in writers:
        writer.__enter__()
    try:
        yield None
    finally:
        for writer in writers:
            writer.__exit__(None, None, None)


@typechecked
class BufferedWriter(TabularDataWriter):

    writer: TabularDataWriter
    buffer_size: int
    buffer: pd.DataFrame | None

    def __init__(
        self,
        writer: TabularDataWriter,
        buffer_size: int = 1000,
    ):
        super().__init__(writer.columns, writer.column_types)
        self.writer = writer
        self.buffer_size = buffer_size
        self.buffer = None

    def _write_buffer(self, force=False):
        if self.buffer is None:
            return
        while len(self.buffer) >= self.buffer_size:
            self.writer.append_data(self.buffer.iloc[: self.buffer_size])
            self.buffer = self.buffer[self.buffer_size :]
        if force and len(self.buffer) > 0:
            self.writer.append_data(self.buffer)
            self.buffer = None

    def append_data(self, data: pd.DataFrame):
        if self.buffer is None:
            self.buffer = data.copy(deep=True)
        else:
            # This is supposed to be faster than pre-allocating, but we should check
            self.buffer = pd.concat(
                [self.buffer, data], axis=0, ignore_index=True
            )
        self._write_buffer()

    def check_valid_data(self, data: pd.DataFrame):
        return self.writer.check_valid_data(data)

    def write(self, data: pd.DataFrame):
        self.writer.write(data)

    def initialize(self):
        self.writer.initialize()

    def finalize(self):
        self._write_buffer(force=True)
        self.writer.finalize()

    def get_associated_reader(self):
        return self.writer.get_associated_reader()


@typechecked
class CSVFileWriter(TabularDataWriter):

    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
        sep: str = "\t",
    ):
        super().__init__(columns, column_types)
        self.file_name = file_name
        self.stdargs = {"sep": sep, "index": False}

    def __str__(self):
        return f"CSVFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return (
            f"CSVFileWriter({self.file_name=},{self.columns=},{self.stdargs=})"
        )

    def get_schema(self, as_dict: bool = True):
        schema = []
        for name, type in zip(self.columns, self.column_types):
            schema.append((name, type))
        return {name: str(type) for name, type in schema}

    def initialize(self):
        # Just write header information
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.file_name, **self.stdargs)

    def finalize(self):
        # no need to do anything here
        pass

    def append_data(self, data: pd.DataFrame):
        self.check_valid_data(data)
        data.to_csv(self.file_name, mode="a", header=False, **self.stdargs)

    def get_associated_reader(self):
        return CSVFileReader(self.file_name, sep=self.stdargs["sep"])


@typechecked
class ParquetFileWriter(TabularDataWriter):

    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list | None = None,
    ):
        super().__init__(columns, column_types)
        self.file_name = file_name

    def __str__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"ParquetFileWriter({self.file_name=},{self.columns=})"

    def get_schema(self, as_dict: bool = False):
        schema = []
        for name, type in zip(self.columns, self.column_types):
            schema.append((name, type))
        if as_dict:
            return {name: str(type) for name, type in schema}
        return pa.schema(schema)

    def initialize(self):
        self.writer = pq.ParquetWriter(
            self.file_name, schema=self.get_schema()
        )

    def finalize(self):
        self.writer.close()

    def append_data(self, data: pd.DataFrame):
        table = pa.Table.from_pandas(
            data, preserve_index=False, schema=self.get_schema()
        )
        self.writer.write_table(table)

    def write(self, data: pd.DataFrame):
        data.to_parquet(self.file_name, index=False)

    def get_associated_reader(self):
        return CSVFileReader(self.file_name)


@typechecked
class SqliteWriter(TabularDataWriter, ABC):

    connection: sqlite3.Connection

    def __init__(
        self,
        database: str | Path | sqlite3.Connection,
        columns: list[str],
        column_types: list | None = None,
    ) -> None:
        super().__init__(columns, column_types)
        if isinstance(database, sqlite3.Connection):
            self.file_name = None
            self.connection = database
        else:
            self.file_name = database
            self.connection = sqlite3.connect(self.file_name)

    def __str__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def __repr__(self):
        return f"SqliteFileWriter({self.file_name=},{self.columns=})"

    def initialize(self):
        # Nothing to do here, we expect the table(s) to already exist
        pass

    def finalize(self):
        self.connection.commit()
        self.connection.close()

    def append_data(self, data: pd.DataFrame):
        # Must be implemented in derived class
        # todo: maybe we can supply also a default implementation in this class
        #       given the table name and sql column names
        raise NotImplementedError

    def get_associated_reader(self):
        # todo: need an sqlite reader first...
        raise NotImplementedError
