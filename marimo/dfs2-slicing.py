import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs2 - Slicing
        """
    )
    return


@app.cell
def _():
    import mikeio
    import matplotlib.pyplot as plt
    return mikeio, plt


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/waves.dfs2")
    ds
    return (ds,)


@app.cell
def _(ds, plt):
    ds[0].plot()
    plt.axvline(x=1400,color='k',linestyle='dashed', label="Transect")
    plt.legend();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        1. Use `Dataset/DataArray.sel` with physical coordinates.
        """
    )
    return


@app.cell
def _(ds):
    ds.sel(x=1400)[0].plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        2. Use grid coordinates with `Dataset/DataArray.isel`
        """
    )
    return


@app.cell
def _(ds):
    ds.geometry.find_index(x=1400)
    return


@app.cell
def _(ds):
    ds.isel(x=27)[0].plot()
    return


@app.cell
def _(ds):
    ds.sel(x=1400).to_dfs("waves_x1400.dfs1")
    return


@app.cell
def _(mikeio):
    dsnew = mikeio.read("waves_x1400.dfs1")
    dsnew
    return (dsnew,)


@app.cell
def _():
    import os
    os.remove("waves_x1400.dfs1")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

