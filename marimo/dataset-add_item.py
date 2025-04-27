import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dataset - Add new item

        A common workflow is to create a new item based on existing items in a dataset.

        This can be in done in several ways. Let's try one of the options.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import mikeio
    return mikeio, np


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/NorthSea_HD_and_windspeed.dfsu")
    ds
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        1. Create a copy of the DataArray
        """
    )
    return


@app.cell
def _(ds):
    ws2 = ds.Wind_speed.copy()
    ws2.plot.hist();
    return (ws2,)


@app.cell
def _(mo):
    mo.md(
        r"""
        2. Make some modifications
        """
    )
    return


@app.cell
def _(np, ws2):
    ws2.values = np.clip(ws2.values, 1,18)
    ws2.plot.hist();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        3. Assign it to a new name in the dataset
        """
    )
    return


@app.cell
def _(ds, ws2):
    ds["Wind speed 2"] = ws2
    return


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        4. Reorder items if necessary
        """
    )
    return


@app.cell
def _(ds):
    ds2 = ds[["Wind speed 2","Surface elevation", "Wind speed"]]
    ds2
    return (ds2,)


@app.cell
def _(ds2):
    ds2.to_dfs("modified.dfsu")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

