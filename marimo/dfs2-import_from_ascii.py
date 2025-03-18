import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Convert Ascii raster into Dfs2

        1. Unzip the zippped file Ascii file (Slottskogen.asc.zip)
        2. Read the Ascii file into a numpy array
        3. Create a grid with the specs from the Ascii file
        4. Create a DataArray with the data from the Ascii file
        5. Create a Dfs2 file
        """
    )
    return


@app.cell
def _():
    import mikeio
    import numpy as np
    return mikeio, np


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. Unzip the zippped file Ascii file (Slottskogen.asc.zip)
        """
    )
    return


@app.cell
def _():
    import zipfile
    with zipfile.ZipFile('../tests/testdata/Slottskogen.asc.zip', 'r') as _f:
        _f.extractall('.')
    return (zipfile,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Read the Ascii file into a numpy array
        """
    )
    return


@app.cell
def _(f):
    filename = 'Slottskogen.asc'
    _f = open(filename, 'r')
    file_contents = f.read()
    print(file_contents[:300])
    _f.close()
    return file_contents, filename


@app.cell
def _(filename, np):
    # We observe that the fist 6 lines are the raster attributes and that fom line 6 starts the array

    # Import the array with numpy
    data = np.loadtxt(filename, skiprows=6)

    # Import the attributes
    with open(filename, 'r') as file:
        ncols = int(file.readline().split()[1])
        nrows = int(file.readline().split()[1])
        xllcorner = float(file.readline().split()[1])
        yllcorner = float(file.readline().split()[1])
        cellsize = float(file.readline().split()[1])
        NODATA_value = float(file.readline().split()[1])
    return (
        NODATA_value,
        cellsize,
        data,
        file,
        ncols,
        nrows,
        xllcorner,
        yllcorner,
    )


@app.cell
def _(data, ncols, np, nrows):
    # Reshape the data into a 2D array
    array = np.reshape(data, (nrows, -ncols))

    # Inverse the y axis
    array = array[::-1, :]
    return (array,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Create a grid with the specs from the Ascii file
        """
    )
    return


@app.cell
def _(cellsize, mikeio, ncols, nrows, xllcorner, yllcorner):
    # Define the geometry. As dfs2 uses the center of the fist cell for its origin wile the ascii file uses the lower left corner, add a shift equal to half the cellsize.
    geometry = mikeio.Grid2D(nx = ncols, ny= nrows, dx=cellsize, dy= cellsize,  origin = (xllcorner+(cellsize/2), yllcorner+(cellsize/2)) ,
                              projection='PROJCS["SWEREF99_12_00",GEOGCS["GCS_SWEREF99",DATUM["D_SWEREF99",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",150000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",12.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
    geometry
    return (geometry,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. Create a DataArray with the data from the Ascii file
        """
    )
    return


@app.cell
def _(array, geometry, mikeio):
    # Now the geometry is set and we can define the whole array
    da = mikeio.DataArray(data=array,
                   item=mikeio.ItemInfo("Elevation", mikeio.EUMType.Elevation),
                   geometry=geometry,
                   dims=("y","x") # No time dimension
                   )
    da
    return (da,)


@app.cell
def _(da):
    # plot the data array
    da.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. Create a Dfs2 file
        """
    )
    return


@app.cell
def _(da):
    # the Mikeio array can be saved as a dfs2
    da.to_dfs("Slottskogen.dfs2")
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
    os.remove("Slottskogen.asc")
    os.remove("Slottskogen.dfs2")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

