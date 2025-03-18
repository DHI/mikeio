import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Global Forecasting System - Meteorological forecast
        """
    )
    return


@app.cell
def _():
    import xarray
    import pandas as pd
    import mikeio
    return mikeio, pd, xarray


@app.cell
def _(mo):
    mo.md(
        r"""
        The file `gfs_wind.nc` contains a small sample of the [GFS](https://nomads.ncep.noaa.gov/) forecast data downloaded via their OpenDAP service
        """
    )
    return


@app.cell
def _(xarray):
    ds = xarray.open_dataset('../tests/testdata/gfs_wind.nc')
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Running a Mike 21 HD model, needs at least three variables of meteorological forcing
        * Mean Sea Level Pressure
        * U 10m
        * V 10m
        """
    )
    return


@app.cell
def _(ds):
    ds.msletmsl
    return


@app.cell
def _(ds):
    ds.ugrd10m
    return


@app.cell
def _(ds):
    ds.vgrd10m
    return


@app.cell
def _(ds):
    ds.ugrd10m[0].plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Convert to dfs2
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Time
        """
    )
    return


@app.cell
def _(ds, pd):
    time = pd.DatetimeIndex(ds.time)
    time
    return (time,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Variable types
        """
    )
    return


@app.cell
def _(mikeio):
    mikeio.EUMType.Air_Pressure
    return


@app.cell
def _(mikeio):
    mikeio.EUMType.Air_Pressure.units
    return


@app.cell
def _(mikeio):
    mikeio.EUMType.Wind_Velocity
    return


@app.cell
def _(mikeio):
    mikeio.EUMType.Wind_Velocity.units
    return


@app.cell
def _(ds):
    mslp = ds.msletmsl.values / 100 # conversion from Pa to hPa
    u = ds.ugrd10m.values
    v = ds.vgrd10m.values
    return mslp, u, v


@app.cell
def _(ds, mikeio):
    geometry = mikeio.Grid2D(x=ds.lon.values, y=ds.lat.values, projection="LONG/LAT")
    geometry
    return (geometry,)


@app.cell
def _(geometry, mikeio, mslp, time, u, v):
    from mikeio import ItemInfo, EUMType, EUMUnit

    mslp_da = mikeio.DataArray(data=mslp,time=time, geometry=geometry, item=ItemInfo("Mean Sea Level Pressure", EUMType.Air_Pressure, EUMUnit.hectopascal))
    u_da = mikeio.DataArray(data=u,time=time, geometry=geometry, item=ItemInfo("Wind U", EUMType.Wind_Velocity, EUMUnit.meter_per_sec))
    v_da = mikeio.DataArray(data=v,time=time, geometry=geometry, item=ItemInfo("Wind V", EUMType.Wind_Velocity, EUMUnit.meter_per_sec))
    return EUMType, EUMUnit, ItemInfo, mslp_da, u_da, v_da


@app.cell
def _(mikeio, mslp_da, u_da, v_da):
    mds = mikeio.Dataset([mslp_da, u_da, v_da])
    mds
    return (mds,)


@app.cell
def _(mds):
    mds.to_dfs("gfs.dfs2")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up (don't run this)
        """
    )
    return


@app.cell
def _():
    import os
    os.remove("gfs.dfs2")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

