import mikeio
import pandas as pd


def global_warming_csv_to_dfs0(csvfilename: str, dfs0filename: str, year=None) -> None:
    """Convert csv file in global warming format to dfs0

    Parameters
    ----------
    csvfilename: str
        input

    dfs0filename: str
        output

    year: int
        Filter data to specific year, default is to use all years
    """

    # read the file, including datetime handling and conversion of missing values
    df = pd.read_csv(csvfilename, parse_dates=True, index_col="Date", na_values=-99.99)

    if year is not None:
        df = df.loc[str(year)]

    # select and reorder relevant columns
    df = df[["Trend", "Average"]]

    # Write to dfs0
    df.to_dfs0(dfs0filename)


if __name__ == "__main__":

    # it is clever to wrap the main script in a 'if __name__ == "__main__":'
    # this will allow you to import the above function from a different script !
    # example: convert_many_csv_to_dfs0.py

    global_warming_csv_to_dfs0(
        csvfilename="tests/testdata/co2-mm-mlo.csv",
        dfs0filename="mauna_loa_co2.dfs0",
    )
