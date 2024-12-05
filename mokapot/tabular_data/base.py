"""
Classes for reading and writing data in tabular form.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from typeguard import typechecked


class BufferType(Enum):
    DataFrame = "DataFrame"
    Records = "Records"
    Dicts = "Dicts"


@typechecked
class TabularDataReader(ABC):
    """
    An abstract class that represents a source for tabular data that can be
    read in either completely or chunk-wise.

    Implementations can be classes that either read from files, from memory
    (e.g. data frames), combine or modify other readers or represent computed
    tabular results.
    """

    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list[np.dtype]:
        raise NotImplementedError

    def get_schema(
        self, as_dict: bool = False
    ) -> dict[str, np.dtype] | list[tuple[str, np.dtype]]:
        schema = list(zip(self.get_column_names(), self.get_column_types()))
        if as_dict:
            return {name: type for name, type in schema}
        return schema

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

    def get_default_extension(self) -> str:
        raise NotImplementedError

    @staticmethod
    def from_path(
        file_name: Path,
        column_map: dict[str, str] | None = None,
        only_columns: list[str] | None = None,
        **kwargs,
    ) -> TabularDataReader:
        # This import has to be here to avoid a circular import ...
        from .format_chooser import reader_from_path

        return reader_from_path(file_name, column_map, only_columns, **kwargs)

    @staticmethod
    def from_series(series: pd.Series, name=None) -> TabularDataReader:
        if name is not None:
            return DataFrameReader(series.to_frame(name=name))
        else:
            return DataFrameReader(series.to_frame())

    @staticmethod
    def from_array(array: np.ndarray, name: str) -> TabularDataReader:
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        elif array.ndim > 1:
            raise ValueError("Array must be 1-dimensional")

        return DataFrameReader(pd.DataFrame({name: array}))


@typechecked
class ColumnSelectReader(TabularDataReader):
    """
    A tabular data reader that returns only certain selected columns of another
    reader.

    Attributes:
    -----------
        reader : TabularDataReader
            The underlying reader for the original data.
        selected_columns : list[str]
            A list that contains names of the selected columns.
    """

    def __init__(self, reader: TabularDataReader, selected_columns: list[str]):
        self.reader = reader
        self.selected_columns = selected_columns

        type_map = reader.get_schema(as_dict=True)
        self.types = [type_map[column] for column in selected_columns]

    def get_column_names(self) -> list[str]:
        return self.selected_columns

    def get_column_types(self) -> list[np.dtype]:
        return self.types

    def _check_columns(self, columns: list[str]):
        for column in columns or []:
            if column not in self.selected_columns:
                raise ValueError(
                    f"Columns ({columns}) are not a subset of "
                    f"the selected columns ({self.selected_columns})"
                )

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        self._check_columns(columns)
        return self.reader.read(columns=self.selected_columns or columns)

    def get_chunked_data_iterator(
        self, chunk_size: int, columns: list[str] | None = None
    ) -> Generator[pd.DataFrame, None, None]:
        self._check_columns(columns)
        return self.reader.get_chunked_data_iterator(
            columns=self.selected_columns or columns
        )


@typechecked
class ColumnMappedReader(TabularDataReader):
    """
    A tabular data reader that renames the columns of another tabular data
    reader to new names.

    Attributes:
    -----------
        reader : TabularDataReader
            The underlying reader for the original data.
        column_map : dict[str, str]
            A dictionary that maps the original column names to the new
            column names.
    """

    def __init__(self, reader: TabularDataReader, column_map: dict[str, str]):
        self.reader = reader
        self.column_map = column_map

    def get_column_names(self) -> list[str]:
        return [
            self.column_map.get(column, column)
            for column in self.reader.get_column_names()
        ]

    def get_column_types(self) -> list[np.dtype]:
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
        # Note: the returned dataframe from here is always mutable, either
        # because the original reader allows this, or because we made a copy
        if self.reader._returned_dataframe_is_mutable():
            df.rename(columns=self.column_map, inplace=True)
        else:
            df = df.rename(columns=self.column_map, inplace=False, copy=False)
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


@typechecked
class DataFrameReader(TabularDataReader):
    """
    This class allows reading pandas DataFrames in the context of tabular data
    readers.

    Attributes:
    -----------
        df : pd.DataFrame
            The DataFrame being read from.
    """

    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __str__(self):
        return f"DataFrameReader({self.df.columns=})"

    def __repr__(self):
        return f"DataFrameReader({self.df=})"

    def get_column_names(self) -> list[str]:
        return self.df.columns.tolist()

    def get_column_types(self) -> list[np.dtype]:
        return self.df.dtypes.tolist()

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


@typechecked
class TabularDataWriter(ABC):
    """
    Abstract base class for writing tabular data to different file formats.

    Attributes:
    -----------
        columns : list[str]
            List of column names
        column_types : list | None
            List of column types (optional)
    """

    def __init__(
        self,
        columns: list[str],
        column_types: list[np.dtype],
    ):
        if len(column_types) not in [0, len(columns)]:
            raise ValueError(
                "`column_types` must have length 0 or same length as `columns`"
            )
        self.columns = columns
        self.column_types = column_types

    def get_column_names(self) -> list[str]:
        return self.columns

    def get_column_types(self) -> list[np.dtype]:
        return self.column_types

    @abstractmethod
    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def check_valid_data(self, data: pd.DataFrame):
        columns = data.columns.tolist()
        if not columns == self.get_column_names():
            raise ValueError(
                f"Column names {columns} do not "
                f"match {self.get_column_names()}"
            )

        if self.column_types is not None:
            column_types = data.dtypes.tolist()
            for own_type, other_type in zip(self.column_types, column_types):
                if not np.can_cast(own_type, other_type, "same_kind"):
                    raise ValueError(
                        f"Column types {column_types} do not match "
                        f"{self.get_column_types()} ({own_type}!={other_type})"
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
        file_name: Path,
        columns: list[str],
        column_types: list[np.dtype],
        buffer_size: int = 0,
        buffer_type: BufferType = BufferType.DataFrame,
        **kwargs,
    ) -> TabularDataWriter:
        # local import needed to avoid circular imports
        from .format_chooser import writer_from_suffix

        return writer_from_suffix(
            file_name, columns, column_types, buffer_size, buffer_type
        )


@contextmanager
# @typechecked
def auto_finalize(writers: list[TabularDataWriter]):
    # todo: nice to have: this method should actually (to be really secure),
    #   check which writers were correctly initialized and if some
    #   initialization throws an error, finalize all that already have been
    #   initialized. Similar with errors during finalization.
    for writer in writers:
        writer.__enter__()
    try:
        yield None
    finally:
        for writer in writers:
            writer.__exit__(None, None, None)


@typechecked
def remove_columns(
    column_names: list[str],
    column_types: list,
    columns_to_remove: list[str],
) -> tuple[list[str], list]:
    temp_columns = [
        (column, type)
        for column, type in zip(column_names, column_types)
        if column not in columns_to_remove
    ]
    temp_column_names, temp_column_types = zip(*temp_columns)
    return (list(temp_column_names), list(temp_column_types))
