import warnings
from typing import Generator

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from typeguard import typechecked


CSV_SUFFIXES = [".csv", ".peptides", ".psms", ".proteins"]

@typechecked
class TabbedFileReader(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_chunked_data_iterator(self, chunk_size: int, columns: list[str] | None = None) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    @staticmethod
    def from_path(file_name: Path, **kwargs) -> "TabbedFileReader":
        # Currently, we look only at the suffix, however, in the future we could
        # also look into the file itself (is it ascii? does it have some "magic
        # bytes"? ...)
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            return CSVFileReader(file_name, **kwargs)

        # Fallback
        warnings.warn(f"Suffix '{suffix}' not recognized in file name '{file_name}'. Falling back to CSV...")
        return CSVFileReader(file_name, **kwargs)


@typechecked
class CSVFileReader(TabbedFileReader):
    def __init__(self, file_name: Path, sep: str = '\t'):
        self.file_name = file_name
        self.stdargs = {'sep': sep, 'index_col': False}

    def get_column_names(self) -> list[str]:
        return pd.read_csv(self.file_name, **self.stdargs,
                           nrows=0).columns.tolist()

    def get_column_types(self) -> list:
        raise NotImplementedError

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        result = pd.read_csv(self.file_name, **self.stdargs)
        return result if columns is None else result[columns]

    def get_chunked_data_iterator(self, chunk_size: int,
                                  columns: list[str] | None = None) \
            -> Generator[pd.DataFrame, None, None]:
        for chunk in pd.read_csv(self.file_name, **self.stdargs,
                                 chunksize=chunk_size):
            yield chunk if columns is None else chunk[columns]



@typechecked
class TabbedFileWriter(ABC):
    @abstractmethod
    def get_column_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_column_types(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def write_header(self):
        raise NotImplementedError

    @abstractmethod
    def append_data(self, data: pd.DataFrame):
        raise NotImplementedError

    def check_valid_data(self, data: pd.DataFrame):
        assert all(data.columns.to_numpy() == self.get_column_names())

    def write(self, data: pd.DataFrame):
        self.write_header()
        self.append_data(data)

    @staticmethod
    def from_suffix(file_name: Path, columns: list[str], **kwargs) -> "TabbedFileWriter":
        suffix = file_name.suffix
        if suffix in CSV_SUFFIXES:
            return CSVFileWriter(file_name, columns, **kwargs)

        # Fallback
        warnings.warn(f"Suffix '{suffix}' not recognized in file name '{file_name}'. Falling back to CSV...")
        return CSVFileWriter(file_name, columns, **kwargs)




@typechecked
class CSVFileWriter(TabbedFileWriter):
    def __init__(self, file_name: Path,
                 columns: list[str],
                 column_types: list | None = None,
                 sep: str = '\t'):
        super().__init__()
        self.file_name = file_name
        self.columns = columns
        self.column_types = column_types
        self.stdargs = {'sep': sep, 'index': False}

    def get_column_names(self):
        return self.columns

    def get_column_types(self):
        return self.column_types

    def write_header(self):
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.file_name, **self.stdargs)

    def append_data(self, data: pd.DataFrame):
        self.check_valid_data(data)
        data.to_csv(self.file_name, mode="a", header=False, **self.stdargs)
