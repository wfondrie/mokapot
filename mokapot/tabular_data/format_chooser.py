import warnings
from pathlib import Path

import numpy as np
from typeguard import typechecked

from mokapot.tabular_data.base import (
    BufferType,
    ColumnMappedReader,
    ColumnSelectReader,
    DataFrameReader,
    TabularDataReader,
    TabularDataWriter,
)
from mokapot.tabular_data.csv import (
    CSVFileReader,
    CSVFileWriter,
)
from mokapot.tabular_data.parquet import (
    ParquetFileReader,
    ParquetFileWriter,
)
from mokapot.tabular_data.sqlite import SqliteWriter
from mokapot.tabular_data.streaming import (
    BufferedWriter,
)
from mokapot.tabular_data.traditional_pin import (
    is_traditional_pin,
    read_traditional_pin,
)

CSV_SUFFIXES = [
    ".csv",
    ".tab",
    ".tsv",
    ".peptides",
    ".psms",
    ".proteins",
    ".modifiedpeptides",
    ".peptidegroups",
    ".modified_peptides",
    ".peptide_groups",
    ".precursors",
]
PIN_SUFFIXES = [".pin"]
PARQUET_SUFFIXES = [".parquet"]
SQLITE_SUFFIXES = [".db"]


@typechecked
def reader_from_path(
    file_name: Path,
    column_map: dict[str, str] | None = None,
    only_columns: list[str] | None = None,
    **kwargs,
) -> TabularDataReader:
    # Currently, we look only at the suffix, however, in the future we
    # could also look into the file itself (is it ascii? does it have
    # some "magic bytes"? ...)

    suffix = file_name.suffix
    if suffix in PIN_SUFFIXES:
        reader = None
        try:
            if is_traditional_pin(file_name):
                reader = DataFrameReader(read_traditional_pin(file_name))
        except ValueError as e:
            msg = "Deprecation warning: Passing files with a .pin extesion"
            msg += " that are not compliant with the format specification"
            msg += " is deprecated."
            msg += " In a future release, this will raise an error."
            msg += " Please use the .tsv extension instead"
            msg += " (and combine the protein column as needed)."
            msg += f" This one failed with the following error: {e}"
            warnings.warn(msg)

        if reader is None:
            reader = CSVFileReader(file_name, **kwargs)
    elif suffix in CSV_SUFFIXES:
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

    if only_columns is not None:
        reader = ColumnSelectReader(reader, only_columns)

    if column_map is not None:
        reader = ColumnMappedReader(reader, column_map)

    return reader


def writer_from_suffix(
    file_name: Path,
    columns: list[str],
    column_types: list[np.dtype],
    buffer_size: int = 0,
    buffer_type: BufferType = BufferType.DataFrame,
    **kwargs,
) -> TabularDataWriter:
    suffix = file_name.suffix
    if suffix in PIN_SUFFIXES:
        writer = CSVFileWriter(file_name, columns, column_types, **kwargs)
    elif suffix in CSV_SUFFIXES:
        writer = CSVFileWriter(file_name, columns, column_types, **kwargs)
    elif suffix in PARQUET_SUFFIXES:
        writer = ParquetFileWriter(file_name, columns, column_types, **kwargs)
    elif suffix in SQLITE_SUFFIXES:
        writer = SqliteWriter(file_name, columns, column_types, **kwargs)
    else:  # Fallback
        warnings.warn(
            f"Suffix '{suffix}' not recognized in file name '{file_name}'."
            " Falling back to CSV..."
        )
        writer = CSVFileWriter(file_name, columns, column_types, **kwargs)

    if buffer_size > 1:
        writer = BufferedWriter(writer, buffer_size, buffer_type)
    return writer
