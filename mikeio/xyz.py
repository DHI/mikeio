import pandas as pd


def read_xyz(filename):

    # try:
    df = pd.read_csv(filename, sep="\t", header=None)
    if df.shape[1] == 1:
        df = pd.read_csv(filename, sep=" ", header=None)

    ncol = df.shape[1]
    names = ["x", "y", "z", "name"]

    df.columns = names[0:ncol]

    return df


def dataframe_to_xyz(self, filename):

    self.to_csv(filename, sep="\t", header=False, index=False)


pd.DataFrame.to_xyz = dataframe_to_xyz
