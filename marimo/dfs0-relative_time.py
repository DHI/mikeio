import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs0 - Relative time axis

        MIKE IO uses a pandas DatetimeIndex to represent the time dimension in dfs files. If the Dfs file has a relative time axis it will be converted to DatetimeIndex by using 1970-1-1 00:00:00 as start time. 
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/eq_relative.dfs0")
    ds
    return (ds,)


@app.cell
def _(ds):
    df = ds.to_dataframe()
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Correcing the dataframe index by subtracting start time to get relative time axis.
        """
    )
    return


@app.cell
def _(df):
    df.index = (df.index - df.index[0]).total_seconds()
    df.index.name = "Relative time (s)"
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df['Item 5'].plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## mikecore

        An alternative is to use the underlying library mikecore to read the file.
        """
    )
    return


@app.cell
def _():
    from mikecore.DfsFileFactory import DfsFileFactory

    dfs = DfsFileFactory.DfsGenericOpen("../tests/testdata/eq_relative.dfs0")
    return DfsFileFactory, dfs


@app.cell
def _(mo):
    mo.md(
        r"""
        Using the `ReadDfs0DataDouble` method you get the data as a numpy array, with the time axis or other type of as the first column.
        """
    )
    return


@app.cell
def _(dfs):
    data = dfs.ReadDfs0DataDouble()

    type(data)
    return (data,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Which can be converted to a pandas dataframe. First we extract the name of items (which in this example hapeens to be not very creative).
        """
    )
    return


@app.cell
def _(dfs):
    index_name = "time"
    items = [i.Name for i in dfs.ItemInfo]
    items = [index_name] + items
    items
    return index_name, items


@app.cell
def _(data, df_1, index_name, items):
    import pandas as df
    df_1 = df_1.DataFrame(data, columns=items).set_index(index_name)
    df_1.head()
    return df, df_1


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

