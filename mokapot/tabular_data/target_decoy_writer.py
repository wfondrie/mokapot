from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from typeguard import typechecked

from mokapot.tabular_data import TabularDataWriter


@typechecked
class TargetDecoyWriter(TabularDataWriter):
    def __init__(
        self,
        writers: Sequence[TabularDataWriter],
        write_decoys: bool = True,
        target_column: str | None = None,
        decoy_column: str | None = None,
    ):
        super().__init__(
            writers[0].get_column_names(), writers[0].get_column_types()
        )
        self.writers = writers
        self.write_decoys = write_decoys
        self.target_column = target_column
        self.decoy_column = decoy_column
        self.output_columns = writers[0].get_column_names()

        assert (target_column is None) != (
            decoy_column is None
        ), "Exactly one of `target_column` and `decoy_column` must be given"

    def initialize(self):
        for writer in self.writers:
            writer.initialize()

    def finalize(self):
        for writer in self.writers:
            writer.finalize()

    def check_valid_data(self, data):
        # We let the `self.writers` to the validation
        pass

    def append_data(self, data: pd.DataFrame):
        out_columns = self.output_columns.copy()
        for x in out_columns:
            if x in data.columns:
                continue
            else:
                warnings.warn(
                    f"Column {x} not found in data,"
                    f" found columns: {data.columns}"
                    "Will try to assign missing values to it."
                )
                data[x] = np.nan

        writers = self.writers
        write_combined = (len(writers) == 1) and self.write_decoys

        if write_combined:
            targets = None
        elif self.target_column is not None:
            targets = data[self.target_column]
        else:
            targets = ~data[self.decoy_column]

        assert write_combined or targets.dtype == bool

        if write_combined:
            # Write decoys and targets combined
            writers[0].append_data(data.loc[:, out_columns])
        elif self.write_decoys:
            # Write decoys and targets separately
            writers[0].append_data(data.loc[targets, out_columns])
            writers[1].append_data(data.loc[~targets, out_columns])
        else:
            # Write targets only
            writers[0].append_data(data.loc[targets, out_columns])
