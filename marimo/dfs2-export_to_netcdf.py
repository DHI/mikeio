import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # DFS2 - Export to NetCDF
        """
    )
    return


@app.cell
def _():
    # pip install mikeio xarray netcdf4
    return


@app.cell
def _():
    import mikeio
    import xarray as xr
    import numpy as np
    return mikeio, np, xr


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/gebco_sound.dfs2")
    return (ds,)


@app.cell
def _(ds):
    ds.Elevation.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The simplest approach is to use the `Dataset.to_xarray()` or `DataArray.to_xarray()` method, if no custom information is neeeded.
        """
    )
    return


@app.cell
def _(ds):
    xr_ds = ds.to_xarray()
    xr_ds
    return (xr_ds,)


@app.cell
def _(ds):
    xr_da = ds.Elevation.to_xarray()
    xr_da
    return (xr_da,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Save it as a NetCDF file:
        """
    )
    return


@app.cell
def _(xr_da):
    xr_da.to_netcdf("gebco_std.nc")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Customized NetCDF
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        1. Time-invariant file -> remove time dimension
        2. Rename y, x to lat, lon
        3. Lowercase names
        4. Add other metadata
        """
    )
    return


@app.cell
def _(ds):
    x = ds.geometry.x
    y = ds.geometry.y
    return x, y


@app.cell
def _(ds, np, x, xr, y):
    res = {}
    spdims = ['lat', 'lon']
    if len(ds.time) > 1:
        dims = ['t'] + spdims
        coords = {'t': ds.time}
    else:
        dims = spdims
        coords = {}
    coords['lon'] = xr.DataArray(x, dims='lon', attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    coords['lat'] = xr.DataArray(y, dims='lat', attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    for da in ds:
        name = da.name.lower()
        res[name] = xr.DataArray(np.squeeze(da.to_numpy()), dims=dims, attrs={'name': name, 'units': da.unit.name, 'eumType': da.type, 'eumUnit': da.unit})
    xr_ds_1 = xr.Dataset(res, coords=coords, attrs={'title': 'Converted from Dfs2 by MIKE IO'})
    return coords, da, dims, name, res, spdims, xr_ds_1


@app.cell
def _(xr_ds_1):
    xr_ds_1
    return


@app.cell
def _(xr_ds_1):
    xr_ds_1.elevation.plot()
    return


@app.cell
def _(xr_ds_1):
    xr_ds_1.to_netcdf('gebco.nc')
    return


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
    os.remove("gebco_std.nc")
    os.remove("gebco.nc")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

