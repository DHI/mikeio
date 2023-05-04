from pathlib import Path
from typing import Union

import pandas as pd


def read_xyz(filename: Union[str, Path]) -> pd.DataFrame:

    df = pd.read_csv(filename, sep="\t", header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(filename, sep=" ", header=None)

    ncol = df.shape[1]
    NAMES = ["x", "y", "z", "name"]

    df.columns = NAMES[:ncol]  # type: ignore

    return df


def dataframe_to_xyz(self, filename: Union[str, Path]) -> None:
    # TODO validation
    self.to_csv(filename, sep="\t", header=False, index=False)


# Monkey patch method on DataFrame for convenience
pd.DataFrame.to_xyz = dataframe_to_xyz  # type: ignore
