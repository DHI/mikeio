import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        #  Time interpolation
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
    ds = mikeio.read("../tests/testdata/waves.dfs2")
    ds
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpolate to specific timestep

        A common use case is to interpolate to a shorter timestep, in this case 1h.
        """
    )
    return


@app.cell
def _(ds):
    ds_h = ds.interp_time(3600)
    ds_h
    return (ds_h,)


@app.cell
def _(mo):
    mo.md(
        r"""
        And to store the interpolated data in a new file.
        """
    )
    return


@app.cell
def _(ds_h):
    ds_h.to_dfs("waves_3h.dfs2")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpolate to time axis of another dataset
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Read some non-equidistant data typically found in observed data.
        """
    )
    return


@app.cell
def _(mikeio):
    ts = mikeio.read("../tests/testdata/waves.dfs0")
    ts
    return (ts,)


@app.cell
def _(mo):
    mo.md(
        r"""
        The observed timeseries is longer than the modelled data. Default is to fill values with NaN.
        """
    )
    return


@app.cell
def _(ds, ts):
    dsi = ds.interp_time(ts)
    return (dsi,)


@app.cell
def _(dsi):
    dsi.time
    return


@app.cell
def _(dsi):
    dsi["Sign. Wave Height"].shape
    return


@app.cell
def _(dsi, ts):
    ax = dsi["Sign. Wave Height"].sel(x=250, y=1200).plot(marker='+')
    ts["Sign. Wave Height"].plot(ax=ax,marker='+')
    return (ax,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model validation

        A common metric for model validation is mean absolute error (MAE).

        In the example below we calculate this metric using the model data interpolated to the observed times.

        For a more elaborate model validation library which takes care of these things for you as well as calculating a number of relevant metrics, take a look at [fmskill](https://github.com/DHI/fmskill).

        Use `np.nanmean` to skip NaN.
        """
    )
    return


@app.cell
def _(ts):
    ts["Sign. Wave Height"]
    return


@app.cell
def _(dsi):
    dsi["Sign. Wave Height"].sel(x=250, y=1200)
    return


@app.cell
def _(dsi, ts):
    diff = (ts["Sign. Wave Height"]  - dsi["Sign. Wave Height"].sel(x=250, y=1200))
    diff.plot()
    return (diff,)


@app.cell
def _(diff, np):
    mae = np.abs(diff).nanmean().to_numpy()
    mae
    return (mae,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Clean up
        """
    )
    return


@app.cell
def _():
    import os
    os.remove("waves_3h.dfs2")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

