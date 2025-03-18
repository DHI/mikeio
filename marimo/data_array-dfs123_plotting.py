import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # DataArray - Dfs123 plotting

        A DataArray with gridded data, can be plotted in many different ways.
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell
def _(mo):
    mo.md(r"""## Dfs1""")
    return


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/vu_tide_hourly.dfs1")
    ds = ds.rename({"Tidal current component (geographic East)":"Tidal current u-comp"})
    da = ds["Tidal current u-comp"]
    da
    return da, ds


@app.cell
def _(da):
    da.geometry
    return


@app.cell
def _(da):
    da[0:10:2].plot();
    return


@app.cell
def _(da, plt):
    da[0:10:2].plot.line()
    plt.legend(da.time[0:10:2]);
    return


@app.cell
def _(da):
    # plot all points on line as time series
    da.plot.timeseries();
    return


@app.cell
def _(da):
    # first 48 hours...  
    da[:49].plot.pcolormesh();
    return


@app.cell
def _(da):
    # single point on line as timeseries
    da.sel(x=0.5).sel(time=slice("2021-08-01","2021-08-03")).plot();
    return


@app.cell
def _(da):
    # all data as histogram
    da.plot.hist(bins=40);
    return


@app.cell
def _(mo):
    mo.md(r"""## Dfs2""")
    return


@app.cell
def _(mikeio):
    da_1 = mikeio.read('../tests/testdata/gebco_sound.dfs2')[0]
    da_1
    return (da_1,)


@app.cell
def _(da_1):
    da_1.geometry
    return


@app.cell
def _(da_1):
    da_1.plot(figsize=(10, 6))
    return


@app.cell
def _(mo):
    mo.md(r"""It is also possible to customize the labels of the axes as well as the color bar, e.g. for localized adaption.""")
    return


@app.cell
def _(da_1, plt):
    da_1.plot.contourf(figsize=(10, 6), levels=4, label='Højde (m)')
    plt.xlabel('Længdekreds (°)')
    plt.ylabel('Breddekreds (°)')
    return


@app.cell
def _(da_1):
    _ax = da_1.plot.contour(figsize=(8, 8), cmap='plasma')
    _ax.set_xlim([12.5, 12.9])
    _ax.set_ylim([55.8, 56])
    return


@app.cell
def _(da_1):
    da_1.plot.hist(bins=20)
    return


@app.cell
def _(mo):
    mo.md(r"""## Dfs3""")
    return


@app.cell
def _(mikeio):
    fn = "../tests/testdata/test_dfs3.dfs3"
    dfs = mikeio.open(fn)
    dfs
    return dfs, fn


@app.cell
def _(dfs):
    dfs.geometry
    return


@app.cell
def _(dfs):
    ds_1 = dfs.read()
    ds_1
    return (ds_1,)


@app.cell
def _(ds_1):
    ds_1.Temperature.plot()
    return


@app.cell
def _(ds_1):
    _ax = ds_1.Temperature[:, 0, :, :].plot.contourf()
    _ax.grid()
    return


@app.cell
def _(ds_1):
    ds_1.Temperature[:, 0, :, 0].plot()
    return


@app.cell
def _(dfs):
    ds_2 = dfs.read(layers=0)
    ds_2
    return (ds_2,)


@app.cell
def _(ds_2):
    ds_2.Temperature.plot()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
