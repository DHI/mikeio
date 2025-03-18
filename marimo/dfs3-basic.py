import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs3 - Basic
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/dissolved_oxygen.dfs3")
    ds
    return (ds,)


@app.cell
def _(ds):
    ds.geometry
    return


@app.cell
def _(ds):
    do  = ds[0]
    do
    return (do,)


@app.cell
def _(do):
    do.isel(z=-1).plot();
    return


@app.cell
def _(mikeio):
    dst = mikeio.read("../tests/testdata/dissolved_oxygen.dfs3", layers="top")
    return (dst,)


@app.cell
def _(dst):
    dst
    return


@app.cell
def _(dst):
    dst[0].plot();
    return


@app.cell
def _(mikeio):
    dsb = mikeio.read("../tests/testdata/dissolved_oxygen.dfs3",layers="bottom")
    dsb
    return (dsb,)


@app.cell
def _(dsb, plt):
    dsb[0].plot(figsize=(10,10))
    plt.title("Bottom oxygen");
    return


@app.cell
def _(dsb):
    dsb[0].to_numpy()[0,110,56]
    return


@app.cell
def _(dst):
    dst[0].to_numpy()[0,110,56]
    return


@app.cell
def _(dsb):
    dsb[0].to_numpy()[0,58,52]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Local coordinates

        Local coordinates (*"NON-UTM"*) have a different convention. The origin is at the bottom-left corner instead of the element center. This applies to x and y coordinates.
        """
    )
    return


@app.cell
def _(mikeio):
    import numpy as np
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    da = mikeio.DataArray(data,
        geometry=mikeio.Grid3D(
                nx=3, ny=2, nz=2, dy=0.5, dz=1, dx=0.5, projection="NON-UTM"
        ),
    )
    da
    return da, data, np


@app.cell
def _(da):
    da.geometry
    return


@app.cell
def _(da):
    da.isel(z=0).plot();
    return


@app.cell
def _(da):
    da.to_xarray()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

