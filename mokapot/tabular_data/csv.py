import warnings
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data import TabularDataReader, TabularDataWriter


@typechecked
class CSVFileReader(TabularDataReader):
    """
    A tabular data reader for reading CSV files.

    Technically speaking this is more of a tsv writter, since it
    more often than not the default separator is tab. But keeping
    the name for consisntency with pandas, where "CSV" is used
    as a generic term to mean a delimited text file.

    Attributes:
    -----------
        file_name :  Path
            The path to the CSV file.
        stdargs : dict
            Arguments for reading CSV file passed on to the pandas
            `read_csv` function.
    """

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

    def get_column_types(self) -> list[np.dtype]:
        df = pd.read_csv(self.file_name, **self.stdargs, nrows=2)
        return df.dtypes.tolist()

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

    def get_default_extension(self) -> str:
        return ".tsv"


@typechecked
class CSVFileWriter(TabularDataWriter):
    """
    CSVFileWriter class for writing tabular data to a CSV file.

    Attributes:
    -----------
    file_name : Path
        The file path where the CSV file will be written.

    sep : str, optional
        The separator string used to separate fields in the CSV file.
        Default is tab character ("\t").
    """

    file_name: Path

    def __init__(
        self,
        file_name: Path,
        columns: list[str],
        column_types: list[np.dtype],
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

    def initialize(self):
        # Just write header information
        if Path(self.file_name).exists():
            warnings.warn(
                f"CSV file {self.file_name} exists, but will be overwritten."
            )
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(self.file_name, **self.stdargs)

    def finalize(self):
        # no need to do anything here
        pass

    def append_data(self, data: pd.DataFrame):
        self.check_valid_data(data)
        # Reorder columns if needed
        data = data.loc[:, self.columns]
        data.to_csv(self.file_name, mode="a", header=False, **self.stdargs)

    def get_associated_reader(self):
        return CSVFileReader(self.file_name, sep=self.stdargs["sep"])

    def read(self):
        return pd.read_csv(self.file_name, sep=self.stdargs["sep"])
