import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DataArray - Masking

        Similar to numpy arrays, DataArrays can be filtered based on values (e.g. all values above a threshold), which will return a DataArray with boolean values. 
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Surface elevation in Oresund

        Read 3 timesteps of Surface elevation from dfsu2d file.
        """
    )
    return


@app.cell
def _(mikeio):
    fn = "../tests/testdata/oresundHD_run1.dfsu"
    da = mikeio.read(fn, items="Surface elevation", time=[0,2,4])[0]
    da
    return da, fn


@app.cell
def _(da, plt):
    _, _ax = plt.subplots(1, da.n_timesteps, figsize=(11, 5), sharey=True)
    for _step in range(da.n_timesteps):
        da[_step].plot(ax=_ax[_step], vmin=0.08, vmax=0.35)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mask values below 0.2m

        Assume that we are not interested in values below 0.2m. Let us find those and call the DataArray mask. 
        """
    )
    return


@app.cell
def _(da):
    threshold = 0.2
    mask = da<threshold
    mask
    return mask, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's set define a new DataArray wl_capped for which we set all values below the threshold to NaN.
        """
    )
    return


@app.cell
def _(da, mask, np):
    wl_capped = da.copy()
    wl_capped[mask] = np.nan
    wl_capped
    return (wl_capped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's plot both the boolean mask and the new capped DataArray for each of the three timesteps
        """
    )
    return


@app.cell
def _(da, mask, plt, wl_capped):
    _, _ax = plt.subplots(2, da.n_timesteps, figsize=(11, 10), sharey=True)
    for _step in range(da.n_timesteps):
        mask[_step].plot(ax=_ax[0, _step], cmap='Reds', add_colorbar=False)
        wl_capped[_step].plot(ax=_ax[1, _step], vmin=0.08, vmax=0.35)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Boolean indexing with numpy array

        The boolean indexing can also be done with a plain numpy array (it does not need to be a mikeio.DataArray). 

        In this example, we set all elements with depth lower than -10m to NaN. 
        """
    )
    return


@app.cell
def _(da):
    ze = da.geometry.element_coordinates[:,2]
    ze.shape
    return (ze,)


@app.cell
def _(da, np, ze):
    wl_shallow = da.copy()
    wl_shallow[ze<-10] = np.nan  # select all elements with depth lower than -10m
    wl_shallow.shape
    return (wl_shallow,)


@app.cell
def _(wl_shallow):
    wl_shallow.plot();
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

