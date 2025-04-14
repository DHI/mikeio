import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Bathymetric data

        [*GEBCO Compilation Group (2020) GEBCO 2020 Grid (doi:10.5285/a29c5465-b138-234d-e053-6c86abc040b9*)](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)

        """
    )
    return


@app.cell
def _():
    import xarray
    import mikeio
    return mikeio, xarray


@app.cell
def _(xarray):
    ds = xarray.open_dataset("../tests/testdata/gebco_2020_n56.3_s55.2_w12.2_e13.1.nc")
    ds
    return (ds,)


@app.cell
def _(ds):
    ds.elevation.plot();
    return


@app.cell
def _(ds):
    ds.elevation.sel(lon=12.74792, lat=55.865, method="nearest")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Check ordering of dimensions, should be (y,x)
        """
    )
    return


@app.cell
def _(ds):
    ds.elevation.dims
    return


@app.cell
def _(ds):
    el = ds.elevation.values
    el.shape
    return (el,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Check that axes are increasing, S->N W->E
        """
    )
    return


@app.cell
def _(ds):
    ds.lat.values[0],ds.lat.values[-1]
    return


@app.cell
def _(ds):
    ds.lat.values[0] < ds.lat.values[-1]
    return


@app.cell
def _(ds):
    ds.lon.values[0],ds.lon.values[-1]
    return


@app.cell
def _(el):
    el[0,0] # Bottom left
    return


@app.cell
def _(el):
    el[-1,0] # Top Left
    return


@app.cell
def _(ds, mikeio):
    geometry = mikeio.Grid2D(x=ds.lon.values, y=ds.lat.values, projection="LONG/LAT")
    geometry
    return (geometry,)


@app.cell
def _(el, geometry, mikeio):
    da = mikeio.DataArray(data=el,
                   item=mikeio.ItemInfo("Elevation", mikeio.EUMType.Total_Water_Depth),
                   geometry=geometry,
                   dims=("y","x") # No time dimension
                   )
    da
    return (da,)


@app.cell
def _(da):
    da.plot();
    return


@app.cell
def _(da):
    da.plot(cmap='coolwarm', vmin=-100, vmax=100);
    return


@app.cell
def _(da):
    da.to_dfs("gebco.dfs2")
    return


@app.cell
def _(mikeio):
    ds_1 = mikeio.read('gebco.dfs2')
    ds_1.Elevation.plot()
    return (ds_1,)


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
    os.remove("gebco.dfs2")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

