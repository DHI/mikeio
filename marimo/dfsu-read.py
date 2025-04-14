import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Read
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    filename = "../tests/testdata/HD2D.dfsu"
    ds = mikeio.read(filename)
    ds
    return ds, filename


@app.cell
def _(filename, mikeio):
    ds_1 = mikeio.read(filename, items=['Surface elevation', 'Current speed'])
    ds_1
    return (ds_1,)


@app.cell
def _(ds_1):
    ds_1.describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Filter in space to the element at our POI, (discrete values, no interpolation)
        """
    )
    return


@app.cell
def _(ds_1):
    ds_1.sel(x=606200, y=6905480)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Interpolate in space to the location of our POI
        """
    )
    return


@app.cell
def _(ds_1):
    ds_1.interp(x=606200, y=6905480)
    return


@app.cell
def _(ds_1):
    ds_1.interp(x=606200, y=6905480).Surface_elevation.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Convert to a dataframe.
        """
    )
    return


@app.cell
def _(ds_1):
    df = ds_1.sel(x=606200, y=6905480).to_dataframe()
    df.head()
    return (df,)


@app.cell
def _(df):
    df.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Other ways to subset data 
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Assume that we interested in these 3 points only
        """
    )
    return


@app.cell
def _(ds_1):
    pt1 = (606200, 6905480)
    pt2 = (606300, 6905410)
    pt3 = (606400, 6905520)
    pts_x = [pt1[0], pt2[0], pt3[0]]
    pts_y = [pt1[1], pt2[1], pt3[1]]
    elem_ids = ds_1.geometry.find_index(pts_x, pts_y)
    return elem_ids, pt1, pt2, pt3, pts_x, pts_y


@app.cell
def _(mo):
    mo.md(
        r"""
        We can use these element ids either when we select the data from the complete dataset using the method isel() as shown above or already when we read the data from file (particular useful for files larger than memory)
        """
    )
    return


@app.cell
def _(elem_ids, filename, mikeio):
    ds_pts = mikeio.read(filename, elements=elem_ids)
    ds_pts
    return (ds_pts,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Create a new dfsu file

        * Subset of items
        * Renamed variables

        First inspect the source file:
        """
    )
    return


@app.cell
def _(mikeio):
    ds_2 = mikeio.read('../tests/testdata/HD2D.dfsu')
    ds_2
    return (ds_2,)


@app.cell
def _(ds_2):
    outfilename2 = 'HD2D_selected.dfsu'
    newds = ds_2[['U velocity', 'V velocity']].rename({'U velocity': 'eastward_sea_water_velocity', 'V velocity': 'northward_sea_water_velocity'})
    newds
    return newds, outfilename2


@app.cell
def _(newds, outfilename2):
    newds.to_dfs(outfilename2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Read the newly created file to verify the contents.
        """
    )
    return


@app.cell
def _(mikeio, outfilename2):
    newds2 = mikeio.read(outfilename2)
    newds2
    return (newds2,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Write mesh from dfsu file
        Don't you have the original mesh? No problem - you can re-create it from the dfsu file... 
        """
    )
    return


@app.cell
def _(ds_2):
    outmesh = 'mesh_from_HD2D.mesh'
    ds_2.geometry.to_mesh(outmesh)
    return (outmesh,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Clean up
        """
    )
    return


@app.cell
def _(outfilename2, outmesh):
    import os
    os.remove(outfilename2)
    os.remove(outmesh)
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

