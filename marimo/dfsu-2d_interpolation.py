import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - 2D interpolation
        This notebook demonstrates how to interpolate dfsu data to a grid, how to save the gridded data as dfs2 and geotiff. It also shows how to interpolate dfsu data to another mesh. 
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/wind_north_sea.dfsu", items="Wind speed")
    ds
    return (ds,)


@app.cell
def _(ds):
    da = ds.Wind_speed
    da.plot();
    return (da,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpolate to grid
        1. Get an overset grid covering the domain
        2. Then interpolate all data to the new grid and plot. 
        4. The interpolated data is then saved to a dfs2 file.
        """
    )
    return


@app.cell
def _(da):
    g = da.geometry.get_overset_grid(dx=0.1)
    g
    return (g,)


@app.cell
def _(da, g):
    da_grid = da.interp_like(g)
    da_grid
    return (da_grid,)


@app.cell
def _(da_grid):
    da_grid.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Save to dfs2 file
        """
    )
    return


@app.cell
def _(da_grid):
    da_grid.to_dfs("wind_north_sea_interpolated.dfs2")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ![](../images/dfsu_grid_interp.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Save to NetCDF
        """
    )
    return


@app.cell
def _(da_grid):
    xr_da = da_grid.to_xarray()
    xr_da.to_netcdf("wind_north_sea_interpolated.nc")
    return (xr_da,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ![](../images/dfsu_grid_netcdf.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Save to GeoTiff

        Install [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) by running this in a command prompt before running this notebook

        ```
        $ conda install -c conda-forge rasterio
        ```

        Or if you prefer to avoid conda, here is how:
        https://rasterio.readthedocs.io/en/latest/installation.html#windows
        """
    )
    return


@app.cell
def _(da, da_grid, g):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    # Dcoumentation https://rasterio.readthedocs.io/en/latest/index.html

    with rasterio.open(
         fp='wind.tif',
         mode='w',
         driver='GTiff',
         height=g.ny,
         width=g.nx,
         count=1,
         dtype=da.dtype,
         crs='+proj=latlong', # adjust accordingly for projected coordinate systems
         transform=from_origin(g.bbox.left, g.bbox.top, g.dx, g.dy)
         ) as dst:
            dst.write(np.flipud(da_grid[0].to_numpy()), 1) # first time_step
    return dst, from_origin, np, rasterio


@app.cell
def _(mo):
    mo.md(
        r"""
        ![](../images/dfsu_grid_interp_tiff.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Interpolate to other mesh
        Interpolate the data from this coarse mesh onto a finer resolution mesh

        """
    )
    return


@app.cell
def _(mikeio):
    msh = mikeio.Mesh('../tests/testdata/north_sea_2.mesh')
    msh
    return (msh,)


@app.cell
def _(da, msh):
    dsi = da.interp_like(msh)
    dsi
    return (dsi,)


@app.cell
def _(da):
    da[0].plot(figsize=(9,7), show_mesh=True);
    return


@app.cell
def _(dsi):
    dsi[0].plot(figsize=(9,7), show_mesh=True);
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Note: 3 of the new elements are outside the original mesh and data are therefore NaN by default
        """
    )
    return


@app.cell
def _(dsi, np):
    nan_elements = np.where(np.isnan(dsi[0].to_numpy()))[0]
    nan_elements
    return (nan_elements,)


@app.cell
def _(da, msh, nan_elements):
    da.geometry.contains(msh.element_coordinates[nan_elements,:2])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### We can force extrapolation to avoid the NaN values
        """
    )
    return


@app.cell
def _(da, msh):
    dat_interp = da.interp_like(msh, extrapolate=True)
    return (dat_interp,)


@app.cell
def _(dat_interp, np):
    n_nan_elements = np.sum(np.isnan(dat_interp.values))
    n_nan_elements
    return (n_nan_elements,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interpolate scatter data to mesh

        We want to interpolate scatter data onto an existing mesh and create a new dfsu with the interpolated data. 

        **This uses lower level private utility methods not part of the public API**.

        Interpolating from scatter data will soon be possible in a simpler way.
        """
    )
    return


@app.cell
def _():
    from mikeio.spatial._utils import dist_in_meters
    from mikeio._interpolation import get_idw_interpolant
    return dist_in_meters, get_idw_interpolant


@app.cell
def _(mikeio):
    dfs = mikeio.open('../tests/testdata/wind_north_sea.dfsu')
    return (dfs,)


@app.cell
def _(dfs):
    dfs.geometry.plot.mesh();
    return


@app.cell
def _(np):
    # scatter data: x,y,value for 4 points
    scatter= np.array([[1,50,1], [4, 52, 3], [8, 55, 2], [-1, 55, 1.5]])
    scatter
    return (scatter,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's first try the approx for a single element: 

        * calc distance to all interpolation points
        * calc IDW interpolatant weights
        * Interpolate
        """
    )
    return


@app.cell
def _(dfs, dist_in_meters, scatter):
    dist = dist_in_meters(scatter[:,:2], dfs.geometry.element_coordinates[0,:2])
    dist
    return (dist,)


@app.cell
def _(dist, get_idw_interpolant):
    w = get_idw_interpolant(dist, p=2)
    w
    return (w,)


@app.cell
def _(np, scatter, w):
    np.dot(scatter[:,2], w) # interpolated value in element 0
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's do the same for all points in the mesh and plot in the end
        """
    )
    return


@app.cell
def _(dfs, dist_in_meters, get_idw_interpolant, np, scatter):
    dati = np.zeros((1, dfs.geometry.n_elements))
    for j in range(dfs.geometry.n_elements):
        dist_1 = dist_in_meters(scatter[:, :2], dfs.geometry.element_coordinates[j, :2])
        w_1 = get_idw_interpolant(dist_1, p=2)
        dati[0, j] = np.dot(scatter[:, 2], w_1)
    return dati, dist_1, j, w_1


@app.cell
def _(dati, dfs, mikeio):
    da_1 = mikeio.DataArray(data=dati, geometry=dfs.geometry, time=dfs.start_time)
    da_1
    return (da_1,)


@app.cell
def _(da_1):
    da_1.plot(title='Interpolated scatter data')
    return


@app.cell
def _(da_1):
    da_1.to_dfs('interpolated_scatter.dfsu')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up
        """
    )
    return


@app.cell
def _():
    import os

    os.remove("wind_north_sea_interpolated.dfs2")
    os.remove("wind_north_sea_interpolated.nc")
    os.remove("wind.tif")
    os.remove("interpolated_scatter.dfsu")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

