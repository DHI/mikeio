import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DataArray - Dfsu plotting

        A DataArray with flexible mesh data, can be plotted in many different ways.
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dfsu 2d
        """
    )
    return


@app.cell
def _(mikeio):
    _fn = '../tests/testdata/oresundHD_run1.dfsu'
    ds = mikeio.read(_fn)
    ds
    return (ds,)


@app.cell
def _(ds):
    da = ds["Surface elevation"]
    da
    return (da,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Plot as map
        """
    )
    return


@app.cell
def _(da):
    # default plot is a map, for multiple timestep data, the first timestep will be shown 
    da.plot();
    return


@app.cell
def _(da):
    # plot last time step as contour map
    da[-1].plot.contour(figsize=(5,8));
    return


@app.cell
def _(da, plt):
    _, ax = plt.subplots(1,2)
    da.plot.mesh(ax=ax[0])
    da.plot.outline(ax=ax[1]);
    return (ax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Plot aggregated data
        """
    )
    return


@app.cell
def _(da):
    da.max().plot(title="Max");
    return


@app.cell
def _(da):
    # difference between last and first timestep
    (da[0] - da[-1]).plot.contourf(title="Difference");
    return


@app.cell
def _(da):
    da.mean(axis="space").plot(title="Spatial mean as function of time");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Other plots

        * time series
        * histogram
        """
    )
    return


@app.cell
def _(da):
    # plot all data as histogram
    da.plot.hist(bins=100);
    return


@app.cell
def _(da):
    # plot all points as timeseries
    da.plot.line(alpha=0.01);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Dfsu 3d
        """
    )
    return


@app.cell
def _(mikeio):
    _fn = '../tests/testdata/oresund_sigma_z.dfsu'
    dfs = mikeio.open(_fn)
    dfs
    return (dfs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Read a specific layer
        If only a specific layer is read, then all the standard 2d plotting can be used 
        """
    )
    return


@app.cell
def _(dfs):
    ds_1 = dfs.read(layers='top')
    ds_1
    return (ds_1,)


@app.cell
def _(ds_1):
    ds_1.geometry.is_2d
    return


@app.cell
def _(ds_1):
    ds_1[1].plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Default plotting behaviour for 3d files is to plot surface layer 
        """
    )
    return


@app.cell
def _(dfs):
    ds_2 = dfs.read()
    ds_2
    return (ds_2,)


@app.cell
def _(ds_2):
    ds_2.geometry.is_2d
    return


@app.cell
def _(ds_2):
    ds_2[1].plot()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

